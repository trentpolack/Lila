#include "geo/ObjIO.h"

#include "geo/ColorSets.h"
#include "geo/IoCommon.h"
#include "geo/Normals.h"
#include "geo/Primitives.h"
#include "geo/Topology.h"

#include "core/Decimal.h"
#include "core/Profile.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace lila::geo
{
namespace
{
using detail::appendNumber;

constexpr int64_t kNoRef = std::numeric_limits<int64_t>::min();

struct Event
{
    enum Kind : uint8_t
    {
        Object,
        Group,
        Material,
        Smooth
    };
    Kind kind;
    size_t faceIndex; // applies to faces from this one on
    std::string name;
    int32_t value = 0;
};

struct Chunk
{
    std::vector<glm::dvec3> v;
    std::vector<glm::vec4> vColor;
    bool anyColor = false;
    std::vector<glm::vec2> vt;
    std::vector<glm::vec3> vn;
    std::vector<uint32_t> faceSize;
    std::vector<int64_t> refs; // 3 per corner: v, vt, vn
    std::vector<Event> events;
    std::vector<std::string> mtllibs;
    size_t lines = 0;
};

struct Cursor
{
    const char* p;
    const char* end;

    void skipSpace()
    {
        while (p < end && (*p == ' ' || *p == '\t'))
            ++p;
    }
    bool atLineEnd() const { return p >= end || *p == '\n' || *p == '\r' || *p == '#'; }
    void nextLine()
    {
        while (p < end && *p != '\n')
            ++p;
        if (p < end)
            ++p;
    }
    std::string_view word()
    {
        skipSpace();
        const char* s = p;
        while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            ++p;
        return {s, size_t(p - s)};
    }
    std::string_view rest()
    {
        skipSpace();
        const char* s = p;
        while (p < end && *p != '\n' && *p != '\r' && *p != '#')
            ++p;
        const char* e = p;
        while (e > s && (e[-1] == ' ' || e[-1] == '\t'))
            --e;
        return {s, size_t(e - s)};
    }
    bool number(double& out)
    {
        skipSpace();
        if (p < end && *p == '+')
            ++p;
        if (const char* q = fastDecimal(p, end, out))
        {
            p = q;
            return true;
        }
        auto [ptr, ec] = lila::fromChars(p, end, out);
        if (ec != std::errc())
            return false;
        p = ptr;
        return true;
    }
    bool integer(int64_t& out)
    {
        auto [ptr, ec] = std::from_chars(p, end, out);
        if (ec != std::errc())
            return false;
        p = ptr;
        return true;
    }
};

/** OBJ index to 0-based global, or a negative value encoding an index local to the chunk. */
int64_t encodeRef(int64_t raw, size_t localCount)
{
    if (raw > 0)
        return raw - 1;
    if (raw < 0)
        return -(int64_t(localCount) + raw) - 1;
    return kNoRef;
}

void parseChunk(const char* begin, const char* end, Chunk& c)
{
    LILA_ZONE();

    Cursor cur{begin, end};
    while (cur.p < cur.end)
    {
        cur.skipSpace();
        if (cur.atLineEnd())
        {
            cur.nextLine();
            continue;
        }
        const std::string_view kw = cur.word();
        if (kw == "v")
        {
            double x = 0, y = 0, z = 0;
            cur.number(x);
            cur.number(y);
            cur.number(z);
            c.v.push_back({x, y, z});
            double r, g, b;
            if (cur.number(r) && cur.number(g) && cur.number(b))
            {
                c.vColor.resize(c.v.size(), glm::vec4(1.0f));
                c.vColor.back() = glm::vec4(float(r), float(g), float(b), 1.0f);
                c.anyColor = true;
            }
        }
        else if (kw == "vt")
        {
            double u = 0, v = 0;
            cur.number(u);
            cur.number(v);
            c.vt.push_back({float(u), float(v)});
        }
        else if (kw == "vn")
        {
            double x = 0, y = 0, z = 0;
            cur.number(x);
            cur.number(y);
            cur.number(z);
            c.vn.push_back({float(x), float(y), float(z)});
        }
        else if (kw == "f")
        {
            uint32_t n = 0;
            for (;;)
            {
                cur.skipSpace();
                if (cur.atLineEnd())
                    break;
                int64_t vi = 0, ti = 0, ni = 0;
                if (!cur.integer(vi))
                {
                    cur.word();
                    continue;
                }
                if (cur.p < cur.end && *cur.p == '/')
                {
                    ++cur.p;
                    if (cur.p < cur.end && *cur.p != '/')
                        cur.integer(ti);
                    if (cur.p < cur.end && *cur.p == '/')
                    {
                        ++cur.p;
                        cur.integer(ni);
                    }
                }
                c.refs.push_back(encodeRef(vi, c.v.size()));
                c.refs.push_back(encodeRef(ti, c.vt.size()));
                c.refs.push_back(encodeRef(ni, c.vn.size()));
                ++n;
            }
            c.faceSize.push_back(n);
        }
        else if (kw == "o")
            c.events.push_back({Event::Object, c.faceSize.size(), std::string(cur.rest())});
        else if (kw == "g")
            c.events.push_back({Event::Group, c.faceSize.size(), std::string(cur.rest())});
        else if (kw == "usemtl")
            c.events.push_back({Event::Material, c.faceSize.size(), std::string(cur.rest())});
        else if (kw == "s")
        {
            const std::string_view s = cur.rest();
            int32_t value = 0;
            if (s != "off")
                std::from_chars(s.data(), s.data() + s.size(), value);
            c.events.push_back({Event::Smooth, c.faceSize.size(), {}, value});
        }
        else if (kw == "mtllib")
            c.mtllibs.emplace_back(cur.rest());
        else if (kw == "l")
            ++c.lines;
        cur.nextLine();
    }
}

/** Splits text at line starts into roughly equal pieces. */
std::vector<std::pair<const char*, const char*>> splitLines(std::string_view text, size_t pieces)
{
    std::vector<std::pair<const char*, const char*>> out;
    const char* p = text.data();
    const char* end = text.data() + text.size();
    const size_t step = std::max<size_t>(1, text.size() / pieces);
    while (p < end)
    {
        const char* q = std::min(end, p + step);
        while (q < end && q[-1] != '\n')
            ++q;
        out.push_back({p, q});
        p = q;
    }
    return out;
}

/** A layer holding exactly values, one per element of the set, copied in parallel. */
template <class T>
void fillLayer(AttributeSet& set, std::string_view name, const std::vector<T>& values)
{
    Attribute& a = values.size() == set.size() ? set.addUnset(name, AttrTypeOf<T>::value) : set.add(name, AttrTypeOf<T>::value);
    const std::span<T> dst = a.template values<T>();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::min(dst.size(), values.size()), 1 << 16), [&](const tbb::blocked_range<size_t>& r) {
        std::copy(values.begin() + r.begin(), values.begin() + r.end(), dst.begin() + r.begin());
    });
}

struct ObjectBuilder
{
    ObjObject obj;
    std::vector<uint32_t> faceSize;
    std::vector<int64_t> refs; // 3 per corner, resolved global 0-based, -1 when absent
    std::vector<int32_t> material;
    std::vector<int32_t> group;
    std::vector<int32_t> smooth;
};
} // namespace

std::vector<int32_t> smoothingGroupsFromHardEdges(const Mesh& mesh)
{
    LILA_ZONE();

    std::span<const bool> hard = mesh.edgeAttrs().get<bool>(attr::kHard);
    std::vector<uint8_t> block(mesh.edgeCapacity(), 0);
    for (Index e = 0; e < mesh.edgeCapacity(); ++e)
        block[e] = !mesh.edgeValid(e) || !mesh.isManifoldEdge(e) || (!hard.empty() && hard[e]);
    std::vector<Index> comp;
    faceComponents(mesh, comp, block);
    std::vector<int32_t> groups(mesh.faceCapacity(), 0);
    std::vector<int32_t> id;
    int32_t next = 1;
    for (Index f = 0; f < mesh.faceCapacity(); ++f)
    {
        if (!mesh.faceValid(f))
            continue;
        bool anySoft = false;
        mesh.forEachFaceCorner(f, [&](Index c) { anySoft |= !block[mesh.cornerEdge(c)]; });
        if (!anySoft)
            continue;
        if (comp[f] >= id.size())
            id.resize(comp[f] + 1, 0);
        if (!id[comp[f]])
            id[comp[f]] = next++;
        groups[f] = id[comp[f]];
    }
    return groups;
}

void hardEdgesFromSmoothingGroups(Mesh& mesh, std::span<const int32_t> groups)
{
    LILA_ZONE();

    std::span<bool> hard = mesh.edgeAttrs().getOrAdd<bool>(attr::kHard);
    tbb::parallel_for(Index(0), mesh.edgeCapacity(), [&](Index e) {
        if (!mesh.edgeValid(e) || !mesh.isManifoldEdge(e))
            return;
        const Index c = mesh.edgeCorner(e);
        const int32_t a = groups[mesh.cornerFace(c)];
        const int32_t b = groups[mesh.cornerFace(mesh.cornerRadial(c))];
        hard[e] = a == 0 || b == 0 || a != b;
    });
}

ObjReadResult readObj(std::string_view text, const ObjReadOptions& opts)
{
    LILA_ZONE();

    ObjReadResult result;
    const size_t pieces = text.size() < (1u << 20)
        ? 1
        : std::min<size_t>(text.size() >> 20, 4 * size_t(tbb::this_task_arena::max_concurrency()));
    const auto ranges = splitLines(text, pieces);
    std::vector<Chunk> chunks(ranges.size());
    tbb::parallel_for(size_t(0), ranges.size(), [&](size_t i) { parseChunk(ranges[i].first, ranges[i].second, chunks[i]); });

    std::vector<size_t> vBase(chunks.size() + 1, 0), tBase(chunks.size() + 1, 0), nBase(chunks.size() + 1, 0);
    bool anyColor = false;
    bool anySmoothStatement = false;
    size_t lines = 0;
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        lines += chunks[i].lines;
        vBase[i + 1] = vBase[i] + chunks[i].v.size();
        tBase[i + 1] = tBase[i] + chunks[i].vt.size();
        nBase[i + 1] = nBase[i] + chunks[i].vn.size();
        anyColor |= chunks[i].anyColor;
        for (const Event& e : chunks[i].events)
            anySmoothStatement |= e.kind == Event::Smooth;
        result.mtllibs.insert(result.mtllibs.end(), chunks[i].mtllibs.begin(), chunks[i].mtllibs.end());
    }
    std::vector<glm::dvec3> v(vBase.back());
    std::vector<glm::vec2> vt(tBase.back());
    std::vector<glm::vec3> vn(nBase.back());
    std::vector<glm::vec4> vColor;
    if (anyColor)
        vColor.assign(vBase.back(), glm::vec4(1.0f));
    tbb::parallel_for(size_t(0), chunks.size(), [&](size_t i) {
        std::copy(chunks[i].v.begin(), chunks[i].v.end(), v.begin() + vBase[i]);
        std::copy(chunks[i].vt.begin(), chunks[i].vt.end(), vt.begin() + tBase[i]);
        std::copy(chunks[i].vn.begin(), chunks[i].vn.end(), vn.begin() + nBase[i]);
        if (chunks[i].anyColor)
        {
            std::vector<glm::vec4> col = chunks[i].vColor;
            col.resize(chunks[i].v.size(), glm::vec4(1.0f));
            std::copy(col.begin(), col.end(), vColor.begin() + vBase[i]);
        }
    });

    // Sequential pass over faces and state statements, sorting faces into objects.
    std::vector<ObjectBuilder> builders;
    std::string materialName, groupName;
    int32_t smooth = 0;
    int32_t material = -1, group = -1;
    auto nameId = [](std::vector<std::string>& names, const std::string& name) {
        auto it = std::find(names.begin(), names.end(), name);
        if (it != names.end())
            return int32_t(it - names.begin());
        names.push_back(name);
        return int32_t(names.size() - 1);
    };
    auto beginObject = [&](const std::string& name) {
        builders.emplace_back().obj.name = name;
        material = group = -1;
    };

    // References made global in parallel: a chunk's relative indices depend only on the counts before it. A face with a
    // missing point is dropped below; missing UVs or normals stay absent.
    std::vector<std::vector<uint8_t>> faceBad(chunks.size());
    tbb::parallel_for(size_t(0), chunks.size(), [&](size_t ci) {
        Chunk& ch = chunks[ci];
        auto resolve = [&](int64_t r, const std::vector<size_t>& base, size_t count) -> int64_t {
            if (r == kNoRef)
                return -1;
            const int64_t g = r >= 0 ? r : int64_t(base[ci]) + (-r - 1);
            return g >= 0 && size_t(g) < count ? g : -2;
        };
        faceBad[ci].assign(ch.faceSize.size(), 0);
        size_t at = 0;
        for (size_t f = 0; f < ch.faceSize.size(); ++f)
        {
            const uint32_t n = ch.faceSize[f];
            int64_t* refs = ch.refs.data() + at * 3;
            at += n;
            bool bad = n < 3;
            for (uint32_t k = 0; k < n; ++k)
            {
                refs[k * 3] = resolve(refs[k * 3], vBase, v.size());
                refs[k * 3 + 1] = resolve(refs[k * 3 + 1], tBase, vt.size());
                refs[k * 3 + 2] = resolve(refs[k * 3 + 2], nBase, vn.size());
                bad = bad || refs[k * 3] < 0;
            }
            faceBad[ci][f] = bad;
        }
    });

    // Sorting faces into objects follows the state statements in file order; each face only notes where its corners go.
    constexpr uint32_t kSkipped = ~0u;
    std::vector<std::vector<uint32_t>> faceBuilder(chunks.size());
    std::vector<std::vector<uint64_t>> faceCorner(chunks.size());
    std::vector<size_t> builderCorners;
    for (size_t ci = 0; ci < chunks.size(); ++ci)
    {
        const Chunk& ch = chunks[ci];
        faceBuilder[ci].assign(ch.faceSize.size(), kSkipped);
        faceCorner[ci].resize(ch.faceSize.size());
        size_t ev = 0;
        for (size_t f = 0; f <= ch.faceSize.size(); ++f)
        {
            for (; ev < ch.events.size() && ch.events[ev].faceIndex == f; ++ev)
            {
                const Event& e = ch.events[ev];
                switch (e.kind)
                {
                case Event::Object:
                    if (!opts.splitObjects)
                        break;
                    if (!builders.empty() && builders.back().faceSize.empty())
                        builders.back().obj.name = e.name;
                    else
                    {
                        beginObject(e.name);
                        builderCorners.push_back(0);
                    }
                    break;
                case Event::Group: groupName = e.name; group = -1; break;
                case Event::Material: materialName = e.name; material = -1; break;
                case Event::Smooth: smooth = e.value; break;
                }
            }
            if (f == ch.faceSize.size())
                break;
            if (builders.empty())
            {
                beginObject("default");
                builderCorners.push_back(0);
            }
            if (faceBad[ci][f])
                continue;
            ObjectBuilder& b = builders.back();
            if (material < 0 && !materialName.empty())
                material = nameId(b.obj.mesh.materialNames(), materialName);
            if (group < 0 && !groupName.empty())
                group = nameId(b.obj.groupNames, groupName);
            faceBuilder[ci][f] = uint32_t(builders.size() - 1);
            faceCorner[ci][f] = builderCorners.back();
            builderCorners.back() += ch.faceSize[f];
            b.faceSize.push_back(ch.faceSize[f]);
            b.material.push_back(std::max(0, material));
            b.group.push_back(std::max(0, group));
            b.smooth.push_back(smooth);
        }
    }
    for (size_t i = 0; i < builders.size(); ++i)
        builders[i].refs.resize(builderCorners[i] * 3);
    tbb::parallel_for(size_t(0), chunks.size(), [&](size_t ci) {
        const Chunk& ch = chunks[ci];
        size_t at = 0;
        for (size_t f = 0; f < ch.faceSize.size(); ++f)
        {
            const uint32_t n = ch.faceSize[f];
            if (faceBuilder[ci][f] != kSkipped)
                std::copy_n(ch.refs.data() + at * 3, size_t(n) * 3, builders[faceBuilder[ci][f]].refs.data() + faceCorner[ci][f] * 3);
            at += n;
        }
    });

    std::vector<Index> pointOf(v.size(), kInvalid);
    std::vector<int64_t> used;
    for (ObjectBuilder& b : builders)
    {
        if (b.faceSize.empty() && builders.size() > 1)
            continue;
        Mesh& m = b.obj.mesh;

        // Points keep the file's vertex order: flag the referenced ones, then walk the flags in order.
        std::vector<uint8_t> referenced(v.size(), 0);
        tbb::parallel_for(size_t(0), b.refs.size() / 3, [&](size_t i) { referenced[size_t(b.refs[i * 3])] = 1; });
        used.clear();
        for (size_t g = 0; g < referenced.size(); ++g)
            if (referenced[g])
                used.push_back(int64_t(g));
        const size_t corners = b.refs.size() / 3;
        std::vector<glm::dvec3> positions(used.size());
        tbb::parallel_for(size_t(0), used.size(), [&](size_t i) {
            pointOf[size_t(used[i])] = Index(i);
            positions[i] = v[size_t(used[i])];
        });
        {
            std::vector<Index> pts(corners);
            tbb::parallel_for(size_t(0), corners, [&](size_t i) { pts[i] = pointOf[size_t(b.refs[i * 3])]; });
            std::vector<std::string> names = std::move(m.materialNames());
            m = Mesh::fromPolygons(positions, b.faceSize, pts);
            m.materialNames() = std::move(names);
        }
        // Every face made it, so faces and corners line up with the file's; otherwise build again face by face.
        const bool allFaces = m.faceCapacity() == b.faceSize.size();
        if (!allFaces)
        {
            std::vector<std::string> names = std::move(m.materialNames());
            m = Mesh();
            m.materialNames() = std::move(names);
            m.reserve(used.size(), corners, b.faceSize.size(), corners);
            for (const glm::dvec3& pos : positions)
                m.addPoint(pos);
        }
        if (anyColor)
        {
            std::span<glm::vec4> col = m.pointAttrs().getOrAdd<glm::vec4>(attr::kColor);
            for (int64_t g : used)
                col[pointOf[size_t(g)]] = vColor[size_t(g)];
        }

        std::vector<glm::vec2> uv;
        std::vector<glm::vec3> normal;
        std::vector<int32_t> faceMat, faceGroup, faceSmooth;
        bool anyUv = false, anyNormal = false;
        uv.reserve(b.refs.size() / 3);
        normal.reserve(b.refs.size() / 3);
        faceMat.reserve(b.faceSize.size());
        faceGroup.reserve(b.faceSize.size());
        faceSmooth.reserve(b.faceSize.size());
        if (allFaces)
        {
            faceMat = b.material;
            faceGroup = b.group;
            faceSmooth = b.smooth;
            uv.resize(corners);
            normal.resize(corners);
            std::atomic<bool> uvSeen = false, normalSeen = false;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, corners), [&](const tbb::blocked_range<size_t>& r) {
                bool u = false, nm = false;
                for (size_t i = r.begin(); i < r.end(); ++i)
                {
                    const int64_t ti = b.refs[i * 3 + 1], ni = b.refs[i * 3 + 2];
                    uv[i] = ti >= 0 ? vt[size_t(ti)] : glm::vec2(0.0f);
                    normal[i] = ni >= 0 ? vn[size_t(ni)] : glm::vec3(0.0f);
                    u |= ti >= 0;
                    nm |= ni >= 0;
                }
                if (u)
                    uvSeen = true;
                if (nm)
                    normalSeen = true;
            });
            anyUv = uvSeen;
            anyNormal = normalSeen;
        }
        std::vector<Index> pts;
        size_t at = 0;
        for (size_t f = 0; f < b.faceSize.size() && !allFaces; ++f)
        {
            const uint32_t n = b.faceSize[f];
            const int64_t* refs = b.refs.data() + at * 3;
            at += n;
            pts.clear();
            for (uint32_t k = 0; k < n; ++k)
                pts.push_back(pointOf[size_t(refs[k * 3])]);
            const Index face = m.addFace(pts);
            if (face == kInvalid)
                continue;
            faceMat.push_back(b.material[f]);
            faceGroup.push_back(b.group[f]);
            faceSmooth.push_back(b.smooth[f]);
            for (uint32_t k = 0; k < n; ++k)
            {
                const int64_t ti = refs[k * 3 + 1], ni = refs[k * 3 + 2];
                uv.push_back(ti >= 0 ? vt[size_t(ti)] : glm::vec2(0.0f));
                normal.push_back(ni >= 0 ? vn[size_t(ni)] : glm::vec3(0.0f));
                anyUv |= ti >= 0;
                anyNormal |= ni >= 0;
            }
        }
        for (int64_t g : used)
            pointOf[size_t(g)] = kInvalid;
        // Points only used by rejected faces.
        m.removeAllLoose();
        if (m.hasDeleted())
            m.compact();

        // Faces and corners were added in order, so ids match the vectors.
        if (!m.materialNames().empty())
            fillLayer(m.faceAttrs(), attr::kMaterial, faceMat);
        if (!b.obj.groupNames.empty())
            fillLayer(m.faceAttrs(), "group", faceGroup);
        if (anyUv)
            fillLayer(m.cornerAttrs(), attr::uvName(0), uv);
        if (anyNormal && opts.keepNormals)
            fillLayer(m.cornerAttrs(), attr::kNormal, normal);

        if (anySmoothStatement)
        {
            fillLayer(m.faceAttrs(), attr::kSmoothGroup, faceSmooth);
            if (opts.smoothingGroupsToHardEdges)
                hardEdgesFromSmoothingGroups(m, faceSmooth);
        }
        else if (anyNormal)
        {
            // Without smoothing groups, edges whose file normals disagree are hard.
            std::span<bool> hard = m.edgeAttrs().getOrAdd<bool>(attr::kHard);
            tbb::parallel_for(Index(0), m.edgeCapacity(), [&](Index e) {
                if (!m.isManifoldEdge(e))
                    return;
                const Index c = m.edgeCorner(e);
                const Index r = m.cornerRadial(c);
                const bool same = m.cornerPoint(r) == m.cornerPoint(c);
                const Index ra = same ? r : m.cornerNext(r);
                const Index rb = same ? m.cornerNext(r) : r;
                auto close = [&](Index x, Index y) { return glm::dot(normal[x], normal[y]) > 0.9999f; };
                hard[e] = !close(c, ra) || !close(m.cornerNext(c), rb);
            });
        }
        if (anyUv)
            markUvSeams(m, attr::uvName(0));
        if (anyColor)
            colors::adopt(m);
        result.objects.push_back(std::move(b.obj));
    }
    if (result.objects.empty() && lines)
        result.error = std::to_string(lines) + " line elements (l) and no faces; OBJ lines are not imported";
    return result;
}

ObjReadResult readObjFile(const std::string& path, const ObjReadOptions& opts)
{
    LILA_ZONE();

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        ObjReadResult r;
        r.error = "cannot open " + path;
        return r;
    }
    in.seekg(0, std::ios::end);
    std::string text(size_t(in.tellg()), '\0');
    in.seekg(0);
    in.read(text.data(), std::streamsize(text.size()));
    return readObj(text, opts);
}

namespace
{

void appendIndex(std::string& s, size_t v)
{
    char buf[24];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    s.append(buf, r.ptr);
}

template <class T> struct BitsHash
{
    size_t operator()(const T& v) const
    {
        uint64_t h = 1469598103934665603ull;
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
        for (size_t i = 0; i < sizeof(T); ++i)
            h = (h ^ b[i]) * 1099511628211ull;
        return size_t(h);
    }
};
template <class T> struct BitsEq
{
    bool operator()(const T& a, const T& b) const { return std::memcmp(&a, &b, sizeof(T)) == 0; }
};

/** Pools unique values, returning 0-based ids per corner. */
template <class T>
std::vector<size_t> poolValues(const Mesh& m, std::span<const T> values, std::vector<T>& pool)
{
    std::vector<size_t> id(m.cornerCapacity(), 0);
    std::unordered_map<T, size_t, BitsHash<T>, BitsEq<T>> map;
    map.reserve(m.numCorners());
    for (Index f = 0; f < m.faceCapacity(); ++f)
    {
        if (!m.faceValid(f))
            continue;
        m.forEachFaceCorner(f, [&](Index c) {
            auto [it, inserted] = map.try_emplace(values[c], pool.size());
            if (inserted)
                pool.push_back(values[c]);
            id[c] = it->second;
        });
    }
    return id;
}

/** Formats [0, n) with fn into one string, in parallel blocks. */
template <class F> std::string formatBlocks(size_t n, F&& fn)
{
    constexpr size_t kBlock = 16384;
    const size_t blocks = (n + kBlock - 1) / kBlock;
    std::vector<std::string> parts(blocks);
    tbb::parallel_for(size_t(0), blocks, [&](size_t b) {
        std::string& s = parts[b];
        for (size_t i = b * kBlock; i < std::min(n, (b + 1) * kBlock); ++i)
            fn(s, i);
    });
    size_t total = 0;
    for (const std::string& s : parts)
        total += s.size();
    std::string out;
    out.reserve(total);
    for (const std::string& s : parts)
        out += s;
    return out;
}
struct NamedMesh
{
    std::string_view name;
    const Mesh* mesh;
};

std::string writeMeshes(std::span<const NamedMesh> objects, const ObjWriteOptions& opts)
{
    LILA_ZONE();

    std::string out = "# Lila\n";
    size_t vOffset = 0, tOffset = 0, nOffset = 0;
    for (const NamedMesh& obj : objects)
    {
        const Mesh& m = *obj.mesh;
        out += "o " + (obj.name.empty() ? std::string("mesh") : std::string(obj.name)) + "\n";

        std::vector<size_t> pointId(m.pointCapacity(), 0);
        std::vector<Index> live;
        live.reserve(m.numPoints());
        for (Index p = 0; p < m.pointCapacity(); ++p)
            if (m.pointValid(p))
            {
                pointId[p] = live.size();
                live.push_back(p);
            }
        // The vertex color extension holds one color per point: the current set, averaged over each point's corners.
        std::vector<glm::vec4> color;
        if (std::string set = colors::current(m); !set.empty())
            color = colors::pointColors(m, set);
        out += formatBlocks(live.size(), [&](std::string& s, size_t i) {
            const glm::dvec3& p = m.position(live[i]);
            s += "v ";
            appendNumber(s, p.x, opts.precision);
            s += ' ';
            appendNumber(s, p.y, opts.precision);
            s += ' ';
            appendNumber(s, p.z, opts.precision);
            if (!color.empty())
                for (int k = 0; k < 3; ++k)
                {
                    s += ' ';
                    appendNumber(s, color[live[i]][k], opts.precision);
                }
            s += '\n';
        });

        std::span<const glm::vec2> uv = m.cornerAttrs().get<glm::vec2>(attr::uvName(0));
        std::vector<glm::vec2> uvPool;
        std::vector<size_t> uvId;
        if (opts.uvs && !uv.empty())
        {
            uvId = poolValues(m, uv, uvPool);
            out += formatBlocks(uvPool.size(), [&](std::string& s, size_t i) {
                s += "vt ";
                appendNumber(s, uvPool[i].x, opts.precision);
                s += ' ';
                appendNumber(s, uvPool[i].y, opts.precision);
                s += '\n';
            });
        }

        std::vector<glm::vec3> computed;
        std::span<const glm::vec3> nrm;
        std::vector<glm::vec3> nPool;
        std::vector<size_t> nId;
        if (opts.normals)
        {
            nrm = m.cornerAttrs().get<glm::vec3>(attr::kNormal);
            if (nrm.size() != m.cornerCapacity())
            {
                computed = cornerNormals(m);
                nrm = computed;
            }
            nId = poolValues(m, nrm, nPool);
            out += formatBlocks(nPool.size(), [&](std::string& s, size_t i) {
                s += "vn ";
                appendNumber(s, nPool[i].x, opts.precision);
                s += ' ';
                appendNumber(s, nPool[i].y, opts.precision);
                s += ' ';
                appendNumber(s, nPool[i].z, opts.precision);
                s += '\n';
            });
        }

        std::vector<int32_t> smooth;
        if (opts.smoothingGroups)
            smooth = smoothingGroupsFromHardEdges(m);
        std::span<const int32_t> mat = m.faceAttrs().get<int32_t>(attr::kMaterial);
        const std::vector<std::string>& matNames = m.materialNames();

        std::vector<Index> faces;
        faces.reserve(m.numFaces());
        for (Index f = 0; f < m.faceCapacity(); ++f)
            if (m.faceValid(f))
                faces.push_back(f);

        // State statements are emitted where they change, so blocks start from the previous face's state.
        out += formatBlocks(faces.size(), [&](std::string& s, size_t i) {
            const Index f = faces[i];
            const Index prev = i ? faces[i - 1] : kInvalid;
            if (!mat.empty() && !matNames.empty() && (prev == kInvalid || mat[prev] != mat[f]))
            {
                const int32_t id = mat[f];
                s += "usemtl ";
                s += id >= 0 && size_t(id) < matNames.size() ? matNames[size_t(id)] : "material" + std::to_string(id);
                s += '\n';
            }
            if (!smooth.empty() && (prev == kInvalid || smooth[prev] != smooth[f]))
            {
                s += "s ";
                s += smooth[f] ? std::to_string(smooth[f]) : std::string("off");
                s += '\n';
            }
            s += 'f';
            m.forEachFaceCorner(f, [&](Index c) {
                s += ' ';
                appendIndex(s, vOffset + pointId[m.cornerPoint(c)] + 1);
                if (!uvId.empty() || !nId.empty())
                {
                    s += '/';
                    if (!uvId.empty())
                        appendIndex(s, tOffset + uvId[c] + 1);
                    if (!nId.empty())
                    {
                        s += '/';
                        appendIndex(s, nOffset + nId[c] + 1);
                    }
                }
            });
            s += '\n';
        });
        vOffset += live.size();
        tOffset += uvPool.size();
        nOffset += nPool.size();
    }
    return out;
}

} // namespace

std::string writeObj(std::span<const ObjObject> objects, const ObjWriteOptions& opts)
{
    std::vector<NamedMesh> named;
    for (const ObjObject& o : objects)
        named.push_back({o.name, &o.mesh});
    return writeMeshes(named, opts);
}

std::string writeObj(const Mesh& mesh, const ObjWriteOptions& opts)
{
    const NamedMesh named{"mesh", &mesh};
    return writeMeshes(std::span<const NamedMesh>(&named, 1), opts);
}

bool writeObjFile(const std::string& path, const Mesh& mesh, const ObjWriteOptions& opts)
{
    LILA_ZONE();

    const std::string text = writeObj(mesh, opts);
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    out.write(text.data(), std::streamsize(text.size()));
    return bool(out);
}
} // namespace lila::geo
