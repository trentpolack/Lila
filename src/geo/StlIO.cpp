#include "geo/StlIO.h"

#include "core/Decimal.h"

#include "geo/ColorSets.h"
#include "geo/IoCommon.h"
#include "geo/Normals.h"
#include "geo/PointWeld.h"
#include "geo/Triangulate.h"

#include "core/FileBytes.h"
#include "core/Paths.h"
#include "core/Profile.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <numeric>

namespace lila::geo
{
namespace
{
using detail::appendNumber;
using detail::linearToSrgb;
using detail::srgbToLinear;

constexpr size_t kHeader = 80;
constexpr size_t kTriangle = 50;

template <class T> T load(const char* p)
{
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

/** Corner positions and polygon sizes as read, before welding. */
struct Soup
{
    UninitVector<glm::vec3> corners;
    UninitVector<Index> sizes;
    std::vector<glm::vec4> faceColor; // per polygon when the file has colors, else empty
};

/** The mesh from a soup: welded (or a point per corner), degenerate polygons dropped, unused points removed. */
Mesh buildMesh(Soup& soup, const StlReadOptions& opts, bool& anyColor)
{
    LILA_ZONE();

    const size_t nc = soup.corners.size(), nf = soup.sizes.size();
    UninitVector<Index> pointOf;
    UninitVector<glm::dvec3> points;
    if (opts.weld)
        weldIdentical(soup.corners, pointOf, points);
    else
    {
        pointOf.resize(nc);
        points.resize(nc);
        tbb::parallel_for(size_t(0), nc, [&](size_t i) {
            pointOf[i] = Index(i);
            points[i] = glm::dvec3(soup.corners[i]);
        });
    }
    freeInBackground(std::move(soup.corners));

    UninitVector<Index> start(nf);
    std::exclusive_scan(soup.sizes.begin(), soup.sizes.end(), start.begin(), Index(0));
    UninitVector<uint8_t> keep(nf);
    std::atomic<bool> dropped = false;
    tbb::parallel_for(size_t(0), nf, [&](size_t f) {
        const Index* p = pointOf.data() + start[f];
        bool ok = soup.sizes[f] >= 3;
        for (Index i = 0; i < soup.sizes[f] && ok; ++i)
            for (Index j = i + 1; j < soup.sizes[f] && ok; ++j)
                ok = p[i] != p[j];
        keep[f] = ok;
        if (!ok)
            dropped = true;
    });
    UninitVector<Index> sizes, indices;
    std::vector<glm::vec4> faceColor;
    if (!dropped)
    {
        sizes = std::move(soup.sizes);
        indices = std::move(pointOf);
        faceColor = std::move(soup.faceColor);
    }
    else
    {
        for (size_t f = 0; f < nf; ++f)
        {
            if (!keep[f])
                continue;
            sizes.push_back(soup.sizes[f]);
            indices.insert(indices.end(), pointOf.begin() + start[f], pointOf.begin() + start[f] + soup.sizes[f]);
            if (!soup.faceColor.empty())
                faceColor.push_back(soup.faceColor[f]);
        }
        // Points only the dropped polygons used go, the rest keep their order.
        std::vector<Index> remap(points.size(), kInvalid);
        for (Index p : indices)
            remap[p] = 0;
        Index next = 0;
        for (size_t p = 0; p < points.size(); ++p)
            if (remap[p] == 0)
            {
                points[next] = points[p];
                remap[p] = next++;
            }
        points.resize(next);
        tbb::parallel_for(size_t(0), indices.size(), [&](size_t i) { indices[i] = remap[indices[i]]; });
    }

    Mesh mesh = Mesh::fromPolygons(points, sizes, indices);
    freeInBackground(std::move(points));
    freeInBackground(std::move(indices));
    anyColor = !faceColor.empty();
    if (anyColor)
    {
        std::span<glm::vec4> col = colors::add(mesh, "colorSet");
        tbb::parallel_for(Index(0), mesh.faceCapacity(), [&](Index f) { mesh.forEachFaceCorner(f, [&](Index c) { col[c] = faceColor[f]; }); });
    }
    if (opts.hardAngle >= 0.0)
        setHardEdgesByAngle(mesh, opts.hardAngle);
    return mesh;
}

/** VisCAM sets bit 15 on colored triangles with red high; Materialise ("COLOR=" header) clears it with red low. */
bool triangleColor(uint16_t attr, bool materialise, glm::vec4& out)
{
    const bool has = materialise ? (attr & 0x8000) == 0 : (attr & 0x8000) != 0;
    if (!has)
        return false;
    const float lo = float(attr & 31) / 31.0f, mid = float((attr >> 5) & 31) / 31.0f, hi = float((attr >> 10) & 31) / 31.0f;
    const float r = materialise ? lo : hi, b = materialise ? hi : lo;
    out = glm::vec4(srgbToLinear(r), srgbToLinear(mid), srgbToLinear(b), 1.0f);
    return true;
}

void readBinary(std::string_view data, const StlReadOptions& opts, StlReadResult& r)
{
    const size_t n = load<uint32_t>(data.data() + kHeader);
    const std::string_view header = data.substr(0, kHeader);
    const bool materialise = header.find("COLOR=") != std::string_view::npos;
    const char* tris = data.data() + kHeader + 4;
    // Most writers leave the attribute bytes zero. Trust them as colors when they vary or all carry the VisCAM flag.
    const uint16_t firstAttr = n > 0 ? load<uint16_t>(tris + 48) : 0;
    struct Seen
    {
        bool differ = false, allFlagged = true;
    };
    const Seen seen = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), Seen{},
        [&](const tbb::blocked_range<size_t>& rg, Seen s) {
            for (size_t t = rg.begin(); t < rg.end(); ++t)
            {
                const uint16_t attr = load<uint16_t>(tris + t * kTriangle + 48);
                s.differ |= attr != firstAttr;
                s.allFlagged &= (attr & 0x8000) != 0;
            }
            return s;
        },
        [](Seen a, Seen b) { return Seen{a.differ || b.differ, a.allFlagged && b.allFlagged}; });
    const bool useColors = n > 0 && (seen.differ || seen.allFlagged || materialise);

    Soup soup;
    soup.corners.resize(n * 3);
    soup.sizes.resize(n);
    std::vector<uint8_t> hasColor;
    if (useColors)
    {
        soup.faceColor.resize(n);
        hasColor.resize(n);
    }
    tbb::parallel_for(size_t(0), n, [&](size_t t) {
        const char* p = tris + t * kTriangle;
        for (int k = 0; k < 3; ++k)
            soup.corners[t * 3 + k] = glm::vec3(load<float>(p + 12 + k * 12), load<float>(p + 16 + k * 12), load<float>(p + 20 + k * 12));
        soup.sizes[t] = 3;
        if (useColors)
        {
            glm::vec4 c(1.0f);
            hasColor[t] = triangleColor(load<uint16_t>(p + 48), materialise, c);
            soup.faceColor[t] = c;
        }
    });
    if (useColors && std::ranges::find(hasColor, uint8_t(1)) == hasColor.end())
        soup.faceColor.clear();
    r.binary = true;
    StlSolid& s = r.solids.emplace_back();
    std::string_view name = header.substr(0, header.find('\0'));
    if (name.starts_with("solid "))
        name.remove_prefix(6);
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())))
        name.remove_suffix(1);
    s.name = std::string(name);
    s.mesh = buildMesh(soup, opts, r.colors);
}

struct Tokens
{
    std::string_view text;
    size_t at = 0;

    std::string_view next()
    {
        while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at])))
            ++at;
        const size_t start = at;
        while (at < text.size() && !std::isspace(static_cast<unsigned char>(text[at])))
            ++at;
        return text.substr(start, at - start);
    }
    /** The rest of the line, trimmed. */
    std::string_view line()
    {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t'))
            ++at;
        const size_t start = at;
        while (at < text.size() && text[at] != '\n' && text[at] != '\r')
            ++at;
        std::string_view s = text.substr(start, at - start);
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
            s.remove_suffix(1);
        return s;
    }
    bool number(float& out)
    {
        std::string_view t = next();
        if (!t.empty() && t[0] == '+')
            t.remove_prefix(1);
        auto [ptr, ec] = lila::fromChars(t.data(), t.data() + t.size(), out);
        return ec == std::errc() && ptr == t.data() + t.size();
    }
};

/** One solid's polygons as read: vertex ranges of the loops, from the file's whole vertex list. */
struct AsciiSolid
{
    std::string name;
    std::vector<std::pair<size_t, Index>> loops;
};

/** Token by token, for files that don't keep each keyword on its own line. */
bool readAsciiTokens(std::string_view data, std::vector<glm::vec3>& verts, std::vector<AsciiSolid>& solids, std::string& error)
{
    Tokens tok{data};
    while (true)
    {
        std::string_view t = tok.next();
        if (t.empty())
            break;
        if (t != "solid")
        {
            error = "expected \"solid\", found \"" + std::string(t.substr(0, 32)) + "\"";
            return false;
        }
        AsciiSolid& s = solids.emplace_back();
        s.name = std::string(tok.line());
        size_t loopStart = verts.size();
        while (true)
        {
            t = tok.next();
            if (t.empty())
                break;
            if (t == "endsolid")
            {
                tok.line();
                break;
            }
            if (t == "vertex")
            {
                glm::vec3 v;
                if (!tok.number(v.x) || !tok.number(v.y) || !tok.number(v.z))
                {
                    error = "bad vertex in solid \"" + s.name + "\"";
                    return false;
                }
                verts.push_back(v);
            }
            else if (t == "endloop")
            {
                if (verts.size() - loopStart >= 3)
                    s.loops.emplace_back(loopStart, Index(verts.size() - loopStart));
                loopStart = verts.size();
            }
            // facet normal, outer loop and endfacet carry nothing the mesh needs: normals are recomputed.
        }
    }
    return true;
}

/** Line by line in parallel chunks, each keyword on its own line as writers put them. False when a line doesn't fit that. */
bool readAsciiLines(std::string_view data, std::vector<glm::vec3>& verts, std::vector<AsciiSolid>& solids)
{
    enum class Kind : uint8_t
    {
        Solid,
        EndLoop,
        EndSolid,
    };
    struct Event
    {
        Kind kind;
        size_t vertsBefore; // within the chunk
        std::string_view name;
    };
    struct Chunk
    {
        std::vector<glm::vec3> verts;
        std::vector<Event> events;
    };
    constexpr size_t kChunk = size_t(4) << 20;
    const size_t chunks = (data.size() + kChunk - 1) / kChunk;
    std::vector<Chunk> parts(chunks);
    std::atomic<bool> bad = false;
    tbb::parallel_for(size_t(0), chunks, [&](size_t k) {
        Chunk& out = parts[k];
        const size_t end = std::min(data.size(), (k + 1) * kChunk);
        size_t at = k * kChunk;
        // A chunk owns the lines that start inside it.
        if (at > 0 && data[at - 1] != '\n')
        {
            at = data.find('\n', at);
            at = at == std::string_view::npos ? end : at + 1;
        }
        while (at < end && !bad)
        {
            size_t eol = data.find('\n', at);
            eol = eol == std::string_view::npos ? data.size() : eol;
            Tokens tok{data.substr(0, eol), at};
            at = eol + 1;
            const std::string_view t = tok.next();
            if (t.empty())
                continue;
            bool ok = true;
            if (t == "vertex")
            {
                glm::vec3 v;
                ok = tok.number(v.x) && tok.number(v.y) && tok.number(v.z) && tok.next().empty();
                out.verts.push_back(v);
            }
            else if (t == "endloop" || t == "endfacet")
            {
                if (t == "endloop")
                    out.events.push_back({Kind::EndLoop, out.verts.size(), {}});
                ok = tok.next().empty();
            }
            else if (t == "solid")
                out.events.push_back({Kind::Solid, out.verts.size(), tok.line()});
            else if (t == "endsolid")
                out.events.push_back({Kind::EndSolid, out.verts.size(), {}});
            else
                ok = t == "facet" || t == "outer";
            if (!ok)
                bad = true;
        }
    });
    if (bad)
        return false;

    std::vector<size_t> base(chunks + 1, 0);
    for (size_t k = 0; k < chunks; ++k)
        base[k + 1] = base[k] + parts[k].verts.size();
    verts.resize(base[chunks]);
    tbb::parallel_for(size_t(0), chunks, [&](size_t k) { std::ranges::copy(parts[k].verts, verts.begin() + std::ptrdiff_t(base[k])); });
    AsciiSolid* current = nullptr;
    size_t loopStart = 0;
    for (size_t k = 0; k < chunks; ++k)
        for (const Event& e : parts[k].events)
        {
            const size_t at = base[k] + e.vertsBefore;
            if (e.kind == Kind::Solid)
            {
                if (current)
                    return false;
                current = &solids.emplace_back();
                current->name = std::string(e.name);
                loopStart = at;
            }
            else if (!current)
                return false;
            else if (e.kind == Kind::EndSolid)
                current = nullptr;
            else
            {
                if (at - loopStart >= 3)
                    current->loops.emplace_back(loopStart, Index(at - loopStart));
                loopStart = at;
            }
        }
    return true;
}

void readAscii(std::string_view data, const StlReadOptions& opts, StlReadResult& r)
{
    std::vector<glm::vec3> verts;
    std::vector<AsciiSolid> solids;
    if (!readAsciiLines(data, verts, solids))
    {
        verts.clear();
        solids.clear();
        if (!readAsciiTokens(data, verts, solids, r.error))
            return;
    }
    if (solids.empty())
    {
        r.error = "no solids";
        return;
    }
    for (const AsciiSolid& a : solids)
    {
        Soup soup;
        soup.sizes.resize(a.loops.size());
        UninitVector<size_t> start(a.loops.size());
        size_t total = 0;
        for (size_t i = 0; i < a.loops.size(); ++i)
        {
            start[i] = total;
            soup.sizes[i] = a.loops[i].second;
            total += a.loops[i].second;
        }
        soup.corners.resize(total);
        tbb::parallel_for(size_t(0), a.loops.size(), [&](size_t i) {
            std::copy_n(verts.begin() + std::ptrdiff_t(a.loops[i].first), a.loops[i].second, soup.corners.begin() + std::ptrdiff_t(start[i]));
        });
        StlSolid& s = r.solids.emplace_back();
        s.name = a.name;
        bool colors = false;
        s.mesh = buildMesh(soup, opts, colors);
    }
}

std::string_view trimLeft(std::string_view s)
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    return s;
}

template <class T> void store(std::string& s, T v)
{
    char b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T));
    s.append(b, sizeof(T));
}

struct Triangle
{
    glm::vec3 p[3];
    glm::vec3 normal;
    uint16_t attr = 0;
};

std::vector<Triangle> triangles(const Mesh& m, bool wantColors)
{
    const Triangulation tris = triangulate(m);
    const std::string set = wantColors ? colors::current(m) : std::string();
    std::span<const glm::vec4> col = set.empty() ? std::span<const glm::vec4>() : colors::values(m, set);
    std::vector<Triangle> out(tris.triangleCount());
    tbb::parallel_for(Index(0), tris.triangleCount(), [&](Index t) {
        Triangle& tri = out[t];
        glm::vec4 sum(0.0f);
        for (int k = 0; k < 3; ++k)
        {
            const Index c = tris.corners[t * 3 + k];
            tri.p[k] = glm::vec3(m.position(m.cornerPoint(c)));
            if (!col.empty())
                sum += col[c];
        }
        const glm::dvec3 n = glm::cross(m.position(m.cornerPoint(tris.corners[t * 3 + 1])) - m.position(m.cornerPoint(tris.corners[t * 3])),
                                        m.position(m.cornerPoint(tris.corners[t * 3 + 2])) - m.position(m.cornerPoint(tris.corners[t * 3])));
        const double len = glm::length(n);
        tri.normal = len > 0.0 ? glm::vec3(n / len) : glm::vec3(0.0f);
        if (!col.empty())
        {
            auto q = [](float v) { return uint16_t(std::lround(linearToSrgb(v) * 31.0f)); };
            sum /= 3.0f;
            tri.attr = uint16_t(0x8000 | (q(sum.r) << 10) | (q(sum.g) << 5) | q(sum.b));
        }
    });
    return out;
}
} // namespace

StlReadResult readStl(std::string_view data, const StlReadOptions& opts)
{
    LILA_ZONE();

    StlReadResult r;
    const bool sizedBinary = data.size() >= kHeader + 4 && kHeader + 4 + size_t(load<uint32_t>(data.data() + kHeader)) * kTriangle == data.size();
    if (sizedBinary)
        readBinary(data, opts, r);
    else if (trimLeft(data).starts_with("solid"))
        readAscii(data, opts, r);
    else if (data.size() >= kHeader + 4 && kHeader + 4 + size_t(load<uint32_t>(data.data() + kHeader)) * kTriangle <= data.size())
        readBinary(data, opts, r); // binary with trailing bytes
    else
        r.error = data.size() < kHeader + 4 ? "file too short" : "not an STL file, or truncated";
    return r;
}

StlReadResult readStlFile(const std::string& path, const StlReadOptions& opts)
{
    UninitVector<char> data;
    try
    {
        data = readFileBytes(paths::fromUtf8(path));
    }
    catch (const std::runtime_error&)
    {
        StlReadResult r;
        r.error = "cannot open " + path;
        return r;
    }
    StlReadResult r = readStl(std::string_view(data.data(), data.size()), opts);
    freeInBackground(std::move(data));
    return r;
}

std::string writeStl(std::span<const StlSolid> solids, const StlWriteOptions& opts)
{
    LILA_ZONE();

    std::string s;
    if (opts.binary)
    {
        std::vector<Triangle> all;
        for (const StlSolid& solid : solids)
        {
            std::vector<Triangle> t = triangles(solid.mesh, opts.colors);
            if (all.empty())
                all = std::move(t);
            else
                all.insert(all.end(), t.begin(), t.end());
        }
        // Never "solid" first, or readers that sniff the header take it for ASCII.
        std::string header = solids.size() == 1 && !solids[0].name.empty() ? "Lila STL: " + solids[0].name : std::string("Lila STL");
        header.resize(kHeader, '\0');
        s = header;
        store(s, uint32_t(all.size()));
        s.resize(kHeader + 4 + all.size() * kTriangle);
        tbb::parallel_for(size_t(0), all.size(), [&](size_t i) {
            const Triangle& t = all[i];
            char* at = s.data() + kHeader + 4 + i * kTriangle;
            auto put = [&](auto v) {
                std::memcpy(at, &v, sizeof(v));
                at += sizeof(v);
            };
            for (int k = 0; k < 3; ++k)
                put(t.normal[k]);
            for (const glm::vec3& p : t.p)
                for (int k = 0; k < 3; ++k)
                    put(p[k]);
            put(t.attr);
        });
        return s;
    }
    for (const StlSolid& solid : solids)
    {
        const std::string name = solid.name.empty() ? "mesh" : solid.name;
        s += "solid " + name + "\n";
        const std::vector<Triangle> tris = triangles(solid.mesh, false);
        constexpr size_t kBlock = 16384;
        std::vector<std::string> parts((tris.size() + kBlock - 1) / kBlock);
        tbb::parallel_for(size_t(0), parts.size(), [&](size_t b) {
            std::string& out = parts[b];
            for (size_t i = b * kBlock; i < std::min(tris.size(), (b + 1) * kBlock); ++i)
            {
                const Triangle& t = tris[i];
                out += "  facet normal ";
                for (int k = 0; k < 3; ++k)
                {
                    appendNumber(out, t.normal[k], opts.precision);
                    out += k < 2 ? ' ' : '\n';
                }
                out += "    outer loop\n";
                for (const glm::vec3& p : t.p)
                {
                    out += "      vertex ";
                    for (int k = 0; k < 3; ++k)
                    {
                        appendNumber(out, p[k], opts.precision);
                        out += k < 2 ? ' ' : '\n';
                    }
                }
                out += "    endloop\n  endfacet\n";
            }
        });
        size_t total = s.size() + name.size() + 10;
        for (const std::string& part : parts)
            total += part.size();
        s.reserve(total);
        for (const std::string& part : parts)
            s += part;
        s += "endsolid " + name + "\n";
    }
    return s;
}
} // namespace lila::geo
