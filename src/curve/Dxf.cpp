#include "curve/Dxf.h"

#include "core/Decimal.h"

#include "core/UninitVector.h"

#include "curve/CurveMath.h"
#include "curve/Fit.h"
#include "curve/Tessellate.h"
#include "geo/PointWeld.h"
#include "geo/Triangulate.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <span>
#include <unordered_map>

namespace lila::curve
{
namespace
{
using M4 = glm::dmat4;

struct Group
{
    int code = -1;
    std::string_view s; // into the file's bytes
    double num = 0.0;

    double d() const { return num; }
    int64_t i() const { return int64_t(num); }
};

/** Left unset on resize: a big file has hundreds of millions, each written once. */
using Groups = UninitVector<Group>;

enum class ValueType
{
    String,
    Double,
    Int16,
    Int32,
    Int64,
    Bool,
    Bytes,
};

ValueType valueType(int c)
{
    if ((c >= 10 && c <= 59) || (c >= 110 && c <= 149) || (c >= 210 && c <= 239) || (c >= 460 && c <= 469) || (c >= 1010 && c <= 1059))
        return ValueType::Double;
    if ((c >= 60 && c <= 79) || (c >= 170 && c <= 179) || (c >= 270 && c <= 289) || (c >= 370 && c <= 389) || (c >= 400 && c <= 409) ||
        (c >= 1060 && c <= 1070))
        return ValueType::Int16;
    if ((c >= 90 && c <= 99) || (c >= 420 && c <= 429) || (c >= 440 && c <= 459) || c == 1071)
        return ValueType::Int32;
    if (c >= 160 && c <= 169)
        return ValueType::Int64;
    if (c >= 290 && c <= 299)
        return ValueType::Bool;
    if ((c >= 310 && c <= 319) || c == 1004)
        return ValueType::Bytes;
    return ValueType::String;
}

std::string_view trim(std::string_view v)
{
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
        v.remove_prefix(1);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r'))
        v.remove_suffix(1);
    return v;
}

/** Code and value line pairs, parsed in parallel chunks of whole lines; everything after the EOF group is ignored. */
bool readAscii(std::string_view data, Groups& out, std::string& error)
{
    constexpr size_t kChunk = size_t(4) << 20;
    // Chunk k is the lines starting in [k * kChunk, (k + 1) * kChunk).
    const size_t chunks = std::max<size_t>(1, (data.size() + kChunk - 1) / kChunk);
    std::vector<size_t> from(chunks + 1, data.size());
    from[0] = 0;
    tbb::parallel_for(size_t(1), chunks, [&](size_t k) {
        const size_t nl = data.find('\n', k * kChunk - 1);
        from[k] = nl == std::string_view::npos ? data.size() : nl + 1;
    });
    std::vector<size_t> lines(chunks + 1, 0);
    tbb::parallel_for(size_t(0), chunks, [&](size_t k) {
        const auto first = data.begin() + std::ptrdiff_t(from[k]), last = data.begin() + std::ptrdiff_t(from[k + 1]);
        lines[k + 1] = size_t(std::count(first, last, '\n')) + (from[k + 1] == data.size() && last != first && *(last - 1) != '\n');
    });
    for (size_t k = 0; k < chunks; ++k)
        lines[k + 1] += lines[k];
    const size_t pairs = lines[chunks] / 2;
    out.resize(pairs);

    auto lineAt = [&](size_t at, size_t& next) {
        size_t e = data.find('\n', at);
        if (e == std::string_view::npos)
            e = data.size();
        next = e + 1;
        std::string_view l = data.substr(at, e - at);
        if (!l.empty() && l.back() == '\r')
            l.remove_suffix(1);
        return l;
    };
    struct Found
    {
        size_t bad, eof;
    };
    const Found f = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, chunks, 1), Found{pairs, pairs},
        [&](const tbb::blocked_range<size_t>& r, Found found) {
            for (size_t k = r.begin(); k < r.end(); ++k)
            {
                size_t at = from[k], line = lines[k];
                // A pair belongs to the chunk its code line starts in.
                if (line % 2 && at < data.size())
                {
                    lineAt(at, at);
                    ++line;
                }
                for (; line < lines[k + 1] && line / 2 < pairs; line += 2)
                {
                    const size_t p = line / 2;
                    Group& g = out[p];
                    const std::string_view code = trim(lineAt(at, at));
                    const std::string_view value = lineAt(at, at);
                    if (std::from_chars(code.data(), code.data() + code.size(), g.code).ec != std::errc())
                    {
                        found.bad = std::min(found.bad, p);
                        continue;
                    }
                    const ValueType t = valueType(g.code);
                    g.s = {};
                    g.num = 0.0;
                    if (t == ValueType::String || t == ValueType::Bytes)
                        g.s = t == ValueType::String ? value : trim(value);
                    else
                    {
                        std::string_view v = trim(value);
                        if (!v.empty() && v.front() == '+')
                            v.remove_prefix(1);
                        lila::fromChars(v.data(), v.data() + v.size(), g.num);
                    }
                    if (g.code == 0 && g.s == "EOF")
                        found.eof = std::min(found.eof, p);
                }
            }
            return found;
        },
        [](Found a, Found b) { return Found{std::min(a.bad, b.bad), std::min(a.eof, b.eof)}; });
    if (f.bad < f.eof)
    {
        error = "line " + std::to_string(f.bad * 2 + 1) + " is not a group code";
        return false;
    }
    out.resize(f.eof < pairs ? f.eof + 1 : pairs);
    return true;
}

constexpr std::string_view kBinarySentinel{"AutoCAD Binary DXF\r\n\x1a\0", 22};

bool readBinary(std::string_view data, Groups& out, std::string& error)
{
    size_t pos = kBinarySentinel.size();
    // R13 and later write two-byte group codes; R12 one byte, with 255 escaping a two-byte code.
    const bool wide = data.size() > pos + 1 && data[pos] == 0 && data[pos + 1] == 0;
    auto need = [&](size_t n) { return pos + n <= data.size(); };
    auto rd = [&](auto& v) {
        std::memcpy(&v, data.data() + pos, sizeof(v));
        pos += sizeof(v);
    };
    while (pos < data.size())
    {
        Group g;
        if (wide)
        {
            if (!need(2))
                break;
            int16_t c;
            rd(c);
            g.code = uint16_t(c);
        }
        else
        {
            if (!need(1))
                break;
            uint8_t c = uint8_t(data[pos++]);
            if (c == 255)
            {
                if (!need(2))
                    break;
                int16_t w;
                rd(w);
                g.code = uint16_t(w);
            }
            else
                g.code = c;
        }
        switch (valueType(g.code))
        {
        case ValueType::String:
        {
            size_t e = data.find('\0', pos);
            if (e == std::string_view::npos)
            {
                error = "unterminated string in binary DXF";
                return false;
            }
            g.s = data.substr(pos, e - pos);
            pos = e + 1;
            break;
        }
        case ValueType::Double:
        {
            if (!need(8))
                return true;
            double v;
            rd(v);
            g.num = v;
            break;
        }
        case ValueType::Int16:
        {
            if (!need(2))
                return true;
            int16_t v;
            rd(v);
            g.num = v;
            break;
        }
        case ValueType::Int32:
        {
            if (!need(4))
                return true;
            int32_t v;
            rd(v);
            g.num = v;
            break;
        }
        case ValueType::Int64:
        {
            if (!need(8))
                return true;
            int64_t v;
            rd(v);
            g.num = double(v);
            break;
        }
        case ValueType::Bool:
            if (!need(1))
                return true;
            g.num = uint8_t(data[pos++]);
            break;
        case ValueType::Bytes:
        {
            if (!need(1))
                return true;
            size_t n = uint8_t(data[pos++]);
            if (!need(n))
                return true;
            pos += n;
            break;
        }
        }
        out.push_back(std::move(g));
        if (out.back().code == 0 && out.back().s == "EOF")
            break;
    }
    return true;
}

bool validUtf8(std::string_view s)
{
    for (size_t i = 0; i < s.size();)
    {
        uint8_t c = uint8_t(s[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size())
            return false;
        for (size_t k = 1; k < n; ++k)
            if ((uint8_t(s[i + k]) >> 6) != 2)
                return false;
        i += n;
    }
    return true;
}

void appendUtf8(std::string& out, uint32_t cp)
{
    if (cp < 0x80)
        out += char(cp);
    else if (cp < 0x800)
    {
        out += char(0xC0 | (cp >> 6));
        out += char(0x80 | (cp & 0x3F));
    }
    else
    {
        out += char(0xE0 | (cp >> 12));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    }
}

/** Pre-2007 files are in the drawing's code page; anything that isn't UTF-8 is read as Latin-1. \U+XXXX escapes decode either way. */
std::string decodeName(std::string_view raw)
{
    std::string s;
    if (validUtf8(raw))
        s = std::string(raw);
    else
        for (char c : raw)
            appendUtf8(s, uint8_t(c));
    std::string out;
    for (size_t i = 0; i < s.size(); ++i)
    {
        uint32_t cp = 0;
        if (i + 7 <= s.size() && s.compare(i, 3, "\\U+") == 0 &&
            std::from_chars(s.data() + i + 3, s.data() + i + 7, cp, 16).ec == std::errc())
        {
            appendUtf8(out, cp);
            i += 6;
        }
        else
            out += s[i];
    }
    return out;
}

struct Entity
{
    std::string_view type;
    std::span<const Group> g;
    std::vector<Entity> sub; // a POLYLINE's VERTEX records

    const Group* find(int code) const
    {
        for (const Group& x : g)
            if (x.code == code)
                return &x;
        return nullptr;
    }
    double d(int code, double def = 0.0) const
    {
        const Group* x = find(code);
        return x ? x->d() : def;
    }
    int64_t i(int code, int64_t def = 0) const
    {
        const Group* x = find(code);
        return x ? x->i() : def;
    }
    std::string s(int code, const std::string& def = {}) const
    {
        const Group* x = find(code);
        return x ? std::string(x->s) : def;
    }
    D3 point(int code, const D3& def = D3(0.0)) const { return D3(d(code, def.x), d(code + 10, def.y), d(code + 20, def.z)); }
};

/** A polyface mesh's VERTEX record: a point (flags 192) or a face of up to four 1-based point numbers (flags 128). */
struct PolyfaceRecord
{
    int64_t flags = 0;
    D3 point{0.0};
    std::array<int64_t, 4> corners{};
};

/** One record, read in one pass over its groups, the first of each code counting. */
PolyfaceRecord polyfaceRecord(const Entity& e)
{
    PolyfaceRecord v;
    uint32_t seen = 0;
    auto once = [&](uint32_t bit) {
        const bool fresh = !(seen & bit);
        seen |= bit;
        return fresh;
    };
    for (const Group& g : e.g)
    {
        if (g.code == 70 && once(1))
            v.flags = g.i();
        else if (g.code == 10 && once(2))
            v.point.x = g.d();
        else if (g.code == 20 && once(4))
            v.point.y = g.d();
        else if (g.code == 30 && once(8))
            v.point.z = g.d();
        else if (g.code >= 71 && g.code <= 74 && once(16u << (g.code - 71)))
            v.corners[size_t(g.code - 71)] = g.i();
    }
    return v;
}

struct Block
{
    D3 base{0.0};
    std::vector<Entity> entities;
};

/** Groups from `at` up to the next code 0 become one entity; POLYLINE gathers its VERTEX records through SEQEND. */
std::vector<Entity> readEntities(const Groups& gs, size_t& at, std::initializer_list<std::string_view> stops)
{
    std::vector<Entity> out;
    auto one = [&]() {
        Entity e;
        e.type = gs[at].s;
        const size_t first = ++at;
        while (at < gs.size() && gs[at].code != 0)
            ++at;
        e.g = std::span<const Group>(gs).subspan(first, at - first);
        return e;
    };
    while (at < gs.size())
    {
        if (gs[at].code != 0)
        {
            ++at;
            continue;
        }
        if (std::ranges::find(stops, std::string_view(gs[at].s)) != stops.end())
            break;
        Entity e = one();
        if (e.type == "POLYLINE")
        {
            while (at < gs.size() && gs[at].code == 0 && gs[at].s == "VERTEX")
                e.sub.push_back(one());
        }
        if (e.type == "SEQEND")
            continue;
        out.push_back(std::move(e));
    }
    return out;
}

/** The arbitrary axis algorithm: object coordinates of a 2D entity to world, from its extrusion direction. */
M4 ocsMatrix(const D3& extrusion)
{
    D3 n = glm::length(extrusion) > 1e-12 ? glm::normalize(extrusion) : D3(0, 0, 1);
    if (std::abs(n.z - 1.0) < 1e-12)
        return M4(1.0);
    D3 ax = std::abs(n.x) < 1.0 / 64.0 && std::abs(n.y) < 1.0 / 64.0 ? glm::cross(D3(0, 1, 0), n) : glm::cross(D3(0, 0, 1), n);
    ax = glm::normalize(ax);
    D3 ay = glm::normalize(glm::cross(n, ax));
    return M4(glm::dvec4(ax, 0), glm::dvec4(ay, 0), glm::dvec4(n, 0), glm::dvec4(0, 0, 0, 1));
}

M4 ocsOf(const Entity& e) { return ocsMatrix(e.point(210, D3(0, 0, 1))); }

struct Vertex
{
    D3 p;
    double bulge = 0.0;
};

/** Straight segments and bulge arcs (in the XY plane of the vertices' space) as one spline. */
bool polySpline(const std::vector<Vertex>& v, bool closed, Spline& out)
{
    std::vector<Piece> ps;
    size_t n = v.size();
    if (n < 2)
        return false;
    for (size_t k = 0; k + (closed ? 0 : 1) < n; ++k)
    {
        const Vertex& a = v[k];
        const D3& b = v[(k + 1) % n].p;
        D3 chord = b - a.p;
        double len = glm::length(chord);
        if (len < 1e-12)
            continue;
        if (std::abs(a.bulge) < 1e-12)
        {
            ps.push_back(linePiece(a.p, b));
            continue;
        }
        double angle = 4.0 * std::atan(a.bulge);
        D3 left = glm::normalize(D3(-chord.y, chord.x, 0.0));
        D3 center = (a.p + b) * 0.5 + left * (len * 0.5 * (1.0 - a.bulge * a.bulge) / (2.0 * a.bulge));
        center.z = a.p.z;
        std::vector<Piece> arc = arcPieces(center, D3(0, 0, 1), a.p, angle);
        ps.insert(ps.end(), arc.begin(), arc.end());
    }
    if (ps.empty())
        return false;
    out = fromPieces(ps, closed, Kind::Poly);
    return true;
}

double sweepOf(double start, double end, double full)
{
    double s = std::fmod(end - start, full);
    if (s <= 1e-12)
        s += full;
    return s;
}

struct LayerBuild
{
    DxfLayer layer;
    std::vector<D3> corners; // welded when the layer is finished
    std::vector<geo::Index> sizes;
};

class Reader
{
public:
    DxfDocument doc;

    void read(const Groups& gs)
    {
        std::vector<Entity> entities;
        for (size_t at = 0; at < gs.size();)
        {
            if (!(gs[at].code == 0 && gs[at].s == "SECTION") || at + 1 >= gs.size())
            {
                ++at;
                continue;
            }
            const std::string_view name = gs[at + 1].s;
            at += 2;
            if (name == "HEADER")
                readHeader(gs, at);
            else if (name == "TABLES")
                readTables(gs, at);
            else if (name == "BLOCKS")
                readBlocks(gs, at);
            else if (name == "ENTITIES")
            {
                std::vector<Entity> e = readEntities(gs, at, {"ENDSEC"});
                entities.insert(entities.end(), std::make_move_iterator(e.begin()), std::make_move_iterator(e.end()));
            }
            while (at < gs.size() && !(gs[at].code == 0 && gs[at].s == "ENDSEC"))
                ++at;
        }
        for (const Entity& e : entities)
            emit(e, M4(1.0), "0", 0);
        finish();
    }

private:
    std::map<std::string, Block> mBlocks;
    std::map<std::string, bool> mLayerHidden;
    std::vector<LayerBuild> mLayers;
    std::map<std::string, size_t> mLayerIndex;
    std::map<std::string, int> mSkipped;
    int mPaperSpace = 0;
    std::unordered_map<std::string_view, std::string> mDecoded;
    size_t mLastLayer = 0;
    std::set<std::string> mActive;

    void readHeader(const Groups& gs, size_t& at)
    {
        for (; at < gs.size() && gs[at].code != 0; ++at)
            if (gs[at].code == 9 && gs[at].s == "$INSUNITS" && at + 1 < gs.size())
            {
                static const double kScale[] = {0.0, 2.54, 30.48, 160934.4, 0.1, 1.0, 100.0, 100000.0, 2.54e-6, 2.54e-3, 91.44, 1e-8, 1e-7, 1e-4, 10.0, 1000.0, 10000.0};
                int64_t u = gs[at + 1].i();
                doc.unitScale = u >= 0 && u < int64_t(std::size(kScale)) ? kScale[u] : 0.0;
            }
    }

    void readTables(const Groups& gs, size_t& at)
    {
        for (; at < gs.size() && !(gs[at].code == 0 && gs[at].s == "ENDSEC"); ++at)
        {
            if (!(gs[at].code == 0 && gs[at].s == "LAYER"))
                continue;
            std::string name;
            int64_t color = 7, flags = 0;
            for (++at; at < gs.size() && gs[at].code != 0; ++at)
            {
                if (gs[at].code == 2)
                    name = decodeName(gs[at].s);
                else if (gs[at].code == 62)
                    color = gs[at].i();
                else if (gs[at].code == 70)
                    flags = gs[at].i();
            }
            if (!name.empty())
                mLayerHidden[name] = color < 0 || (flags & 1);
            --at;
        }
    }

    void readBlocks(const Groups& gs, size_t& at)
    {
        while (at < gs.size() && !(gs[at].code == 0 && gs[at].s == "ENDSEC"))
        {
            if (!(gs[at].code == 0 && gs[at].s == "BLOCK"))
            {
                ++at;
                continue;
            }
            Entity head;
            const size_t first = ++at;
            while (at < gs.size() && gs[at].code != 0)
                ++at;
            head.g = std::span<const Group>(gs).subspan(first, at - first);
            Block b;
            b.base = head.point(10);
            b.entities = readEntities(gs, at, {"ENDBLK", "ENDSEC"});
            if (at < gs.size() && gs[at].s == "ENDBLK")
                ++at;
            mBlocks[head.s(2)] = std::move(b);
        }
    }

    LayerBuild& layer(const std::string& name)
    {
        if (mLastLayer < mLayers.size() && mLayers[mLastLayer].layer.name == name)
            return mLayers[mLastLayer];
        auto [it, added] = mLayerIndex.try_emplace(name, mLayers.size());
        mLastLayer = it->second;
        if (added)
        {
            LayerBuild& l = mLayers.emplace_back();
            l.layer.name = name;
            auto h = mLayerHidden.find(name);
            l.layer.hidden = h != mLayerHidden.end() && h->second;
        }
        return mLayers[it->second];
    }

    void addCurve(const std::string& layerName, Spline s, const M4& m)
    {
        transform(s, m);
        if (!isValid(s))
            return;
        layer(layerName).layer.curves.push_back(std::move(s));
    }

    void addFace(const std::string& layerName, std::span<const D3> pts, const M4& m)
    {
        LayerBuild& l = layer(layerName);
        const size_t start = l.corners.size();
        // Corners weld when they are the same bit for bit.
        auto same = [](const D3& a, const D3& b) { return std::memcmp(&a, &b, sizeof(D3)) == 0; };
        for (const D3& p : pts)
        {
            const D3 w = D3(m * glm::dvec4(p, 1.0));
            if (l.corners.size() == start || !same(l.corners.back(), w))
                l.corners.push_back(w);
        }
        while (l.corners.size() - start > 1 && same(l.corners.back(), l.corners[start]))
            l.corners.pop_back();
        if (l.corners.size() - start < 3)
            l.corners.resize(start);
        else
            l.sizes.push_back(geo::Index(l.corners.size() - start));
    }
    void addFace(const std::string& layerName, std::initializer_list<D3> pts, const M4& m)
    {
        addFace(layerName, std::span<const D3>(pts.begin(), pts.size()), m);
    }

    /** An entity's layer name, decoded once per name. */
    const std::string& layerName(const Entity& e)
    {
        const Group* g = e.find(8);
        const std::string_view raw = g ? g->s : std::string_view("0");
        auto it = mDecoded.find(raw);
        if (it == mDecoded.end())
            it = mDecoded.emplace(raw, decodeName(raw)).first;
        return it->second;
    }

    void skip(std::string_view type) { ++mSkipped[std::string(type)]; }

    void emit(const Entity& e, const M4& m, const std::string& inherited, int depth)
    {
        if (e.i(67) == 1)
        {
            ++mPaperSpace;
            return;
        }
        if (e.i(60) == 1)
            return;
        const std::string& named = layerName(e);
        const std::string& ln = named == "0" && depth > 0 ? inherited : named;
        const std::string_view t = e.type;
        if (t == "LINE")
        {
            Spline s;
            s.points = {e.point(10), e.point(11)};
            addCurve(ln, std::move(s), m);
        }
        else if (t == "CIRCLE")
        {
            double r = e.d(40);
            if (r > 0.0)
                addCurve(ln, circle(e.point(10), D3(0, 0, 1), r), m * ocsOf(e));
        }
        else if (t == "ARC")
        {
            double r = e.d(40);
            double a0 = glm::radians(e.d(50)), sweep = sweepOf(a0, glm::radians(e.d(51)), glm::two_pi<double>());
            D3 c = e.point(10);
            if (r > 0.0)
                addCurve(ln, arc(c, D3(0, 0, 1), c + D3(std::cos(a0), std::sin(a0), 0.0) * r, sweep), m * ocsOf(e));
        }
        else if (t == "ELLIPSE")
            ellipse(e, ln, m);
        else if (t == "LWPOLYLINE")
            lwpolyline(e, ln, m);
        else if (t == "POLYLINE")
            polyline(e, ln, m);
        else if (t == "SPLINE")
            spline(e, ln, m);
        else if (t == "3DFACE" || t == "SOLID" || t == "TRACE")
        {
            D3 p[4] = {e.point(10), e.point(11), e.point(12), e.point(13)};
            // SOLID and TRACE list their corners in Z order, in object coordinates.
            if (t != "3DFACE")
                addFace(ln, {p[0], p[1], p[3], p[2]}, m * ocsOf(e));
            else
                addFace(ln, {p[0], p[1], p[2], p[3]}, m);
        }
        else if (t == "MESH")
            subdMesh(e, ln, m);
        else if (t == "INSERT")
            insert(e, ln, m, depth);
        else if (t == "ATTRIB" || t == "ATTDEF" || t == "VIEWPORT" || t == "SEQEND")
            return;
        else
            skip(t);
    }

    void ellipse(const Entity& e, const std::string& ln, const M4& m)
    {
        D3 n = e.point(210, D3(0, 0, 1));
        D3 major = e.point(11);
        double ratio = e.d(40, 1.0);
        if (glm::length(major) < 1e-12 || glm::length(n) < 1e-12)
            return;
        D3 minor = glm::normalize(glm::cross(glm::normalize(n), major)) * glm::length(major) * ratio;
        M4 frame(glm::dvec4(major, 0), glm::dvec4(minor, 0), glm::dvec4(glm::normalize(n), 0), glm::dvec4(e.point(10), 1));
        double a0 = e.d(41, 0.0), a1 = e.d(42, glm::two_pi<double>());
        double sweep = sweepOf(a0, a1, glm::two_pi<double>());
        Spline s = sweep >= glm::two_pi<double>() - 1e-9 ? circle(D3(0.0), D3(0, 0, 1), 1.0)
                                                          : arc(D3(0.0), D3(0, 0, 1), D3(std::cos(a0), std::sin(a0), 0.0), sweep);
        addCurve(ln, std::move(s), m * frame);
    }

    void lwpolyline(const Entity& e, const std::string& ln, const M4& m)
    {
        std::vector<Vertex> v;
        double z = e.d(38);
        for (const Group& g : e.g)
        {
            if (g.code == 10)
                v.push_back({D3(g.d(), 0.0, z)});
            else if (g.code == 20 && !v.empty())
                v.back().p.y = g.d();
            else if (g.code == 42 && !v.empty())
                v.back().bulge = g.d();
        }
        Spline s;
        if (polySpline(v, e.i(70) & 1, s))
            addCurve(ln, std::move(s), m * ocsOf(e));
    }

    void polyline(const Entity& e, const std::string& ln, const M4& m)
    {
        int64_t flags = e.i(70);
        if (flags & 64)
        {
            std::vector<D3> pts, face;
            for (const Entity& sub : e.sub)
            {
                const PolyfaceRecord v = polyfaceRecord(sub);
                if ((v.flags & 128) && (v.flags & 64))
                    pts.push_back(v.point);
                else if (v.flags & 128)
                {
                    face.clear();
                    for (int64_t k : v.corners)
                    {
                        k = std::abs(k);
                        if (k > 0 && size_t(k) <= pts.size())
                            face.push_back(pts[size_t(k - 1)]);
                    }
                    addFace(ln, face, m);
                }
            }
            return;
        }
        if (flags & 16)
        {
            std::vector<D3> pts;
            for (const Entity& v : e.sub)
                pts.push_back(v.point(10));
            int64_t M = e.i(71), N = e.i(72);
            if (e.i(75) != 0 && size_t(M * N) != pts.size())
                M = e.i(73), N = e.i(74);
            if (M < 2 || N < 2 || size_t(M * N) != pts.size())
                return;
            bool closedM = flags & 1, closedN = flags & 32;
            for (int64_t a = 0; a < (closedM ? M : M - 1); ++a)
                for (int64_t b = 0; b < (closedN ? N : N - 1); ++b)
                {
                    auto at = [&](int64_t i, int64_t j) { return pts[size_t((i % M) * N + (j % N))]; };
                    addFace(ln, {at(a, b), at(a, b + 1), at(a + 1, b + 1), at(a + 1, b)}, m);
                }
            return;
        }
        const bool is3d = flags & 8;
        const bool splineFit = flags & 4;
        std::vector<Vertex> v;
        for (const Entity& x : e.sub)
        {
            // A spline-fit polyline keeps its frame (control) points beside the fitted ones; the fitted ones are the curve.
            if (splineFit && (x.i(70) & 16))
                continue;
            v.push_back({x.point(10), is3d ? 0.0 : x.d(42)});
        }
        if (!is3d)
            for (Vertex& x : v)
                x.p.z = e.d(30);
        Spline s;
        if (polySpline(v, flags & 1, s))
            addCurve(ln, std::move(s), is3d ? m : m * ocsOf(e));
    }

    void spline(const Entity& e, const std::string& ln, const M4& m)
    {
        int degree = int(e.i(71, 3));
        std::vector<double> knots, weights;
        std::vector<D3> ctrl, fit;
        for (size_t k = 0; k < e.g.size(); ++k)
        {
            const Group& g = e.g[k];
            if (g.code == 40)
                knots.push_back(g.d());
            else if (g.code == 41)
                weights.push_back(g.d());
            else if (g.code == 10)
                ctrl.push_back(D3(g.d(), 0, 0));
            else if (g.code == 20 && !ctrl.empty())
                ctrl.back().y = g.d();
            else if (g.code == 30 && !ctrl.empty())
                ctrl.back().z = g.d();
            else if (g.code == 11)
                fit.push_back(D3(g.d(), 0, 0));
            else if (g.code == 21 && !fit.empty())
                fit.back().y = g.d();
            else if (g.code == 31 && !fit.empty())
                fit.back().z = g.d();
        }
        Spline s;
        if (ctrl.size() >= 2)
        {
            s.kind = Kind::Nurbs;
            s.degree = std::clamp(degree, 1, int(ctrl.size()) - 1);
            s.points = std::move(ctrl);
            if (s.degree > 7)
            {
                doc.warnings.push_back("a degree " + std::to_string(s.degree) + " SPLINE came in as degree 7");
                s.degree = 7;
                knots.clear();
            }
            if (knots.size() == s.points.size() + size_t(s.degree) + 1)
                s.knots = std::move(knots);
            else if (!knots.empty())
                doc.warnings.push_back("a SPLINE's knots didn't match its control points; it came in with uniform knots");
            if (weights.size() == s.points.size() && std::ranges::any_of(weights, [](double w) { return std::abs(w - 1.0) > 1e-12; }))
                s.weights = std::move(weights);
            if (!isValid(s))
            {
                s.knots.clear();
                s.weights.clear();
            }
        }
        else if (fit.size() >= 2)
            s = interpolate(fit, std::min(3, int(fit.size()) - 1), (e.i(70) & 1) && fit.size() > 2);
        else
            return;
        addCurve(ln, std::move(s), m);
    }

    void subdMesh(const Entity& e, const std::string& ln, const M4& m)
    {
        std::vector<D3> pts;
        std::vector<int64_t> list;
        size_t k = 0;
        for (; k < e.g.size() && e.g[k].code != 92; ++k)
            ;
        if (k == e.g.size())
            return;
        size_t nv = size_t(std::max<int64_t>(0, e.g[k++].i()));
        for (; k < e.g.size() && e.g[k].code != 93; ++k)
        {
            const Group& g = e.g[k];
            if (g.code == 10 && pts.size() < nv)
                pts.push_back(D3(g.d(), 0, 0));
            else if (g.code == 20 && !pts.empty())
                pts.back().y = g.d();
            else if (g.code == 30 && !pts.empty())
                pts.back().z = g.d();
        }
        for (; k < e.g.size() && e.g[k].code != 93; ++k)
            ;
        if (k == e.g.size())
            return;
        size_t nl = size_t(std::max<int64_t>(0, e.g[k++].i()));
        for (; k < e.g.size() && list.size() < nl; ++k)
            if (e.g[k].code == 90)
                list.push_back(e.g[k].i());
        for (size_t at = 0; at < list.size();)
        {
            size_t n = size_t(std::max<int64_t>(0, list[at++]));
            std::vector<D3> face;
            for (size_t c = 0; c < n && at < list.size(); ++c, ++at)
                if (list[at] >= 0 && size_t(list[at]) < pts.size())
                    face.push_back(pts[size_t(list[at])]);
            addFace(ln, face, m);
        }
    }

    void insert(const Entity& e, const std::string& ln, const M4& m, int depth)
    {
        std::string name = e.s(2);
        auto it = mBlocks.find(name);
        if (it == mBlocks.end())
        {
            doc.warnings.push_back("INSERT of a missing block '" + decodeName(name) + "'");
            return;
        }
        if (depth >= 32 || mActive.count(name))
            return;
        const Block& b = it->second;
        D3 scale(e.d(41, 1.0), e.d(42, 1.0), e.d(43, 1.0));
        double rot = glm::radians(e.d(50));
        int64_t cols = std::max<int64_t>(1, e.i(70, 1)), rows = std::max<int64_t>(1, e.i(71, 1));
        double dx = e.d(44), dy = e.d(45);
        M4 place = m * ocsOf(e) * glm::translate(M4(1.0), e.point(10)) * glm::rotate(M4(1.0), rot, D3(0, 0, 1));
        mActive.insert(name);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < cols; ++c)
            {
                M4 cell = place * glm::translate(M4(1.0), D3(double(c) * dx, double(r) * dy, 0.0)) * glm::scale(M4(1.0), scale) *
                          glm::translate(M4(1.0), -b.base);
                for (const Entity& x : b.entities)
                    emit(x, cell, ln, depth + 1);
            }
        mActive.erase(name);
    }

    void finish()
    {
        for (LayerBuild& l : mLayers)
        {
            if (!l.sizes.empty())
            {
                UninitVector<geo::Index> pointOf;
                UninitVector<D3> points;
                geo::weldIdentical(l.corners, pointOf, points, true);
                std::vector<D3>().swap(l.corners);
                l.layer.mesh = geo::Mesh::fromPolygons(points, l.sizes, pointOf);
            }
            if (!l.layer.curves.empty() || l.layer.mesh.numFaces() > 0)
                doc.layers.push_back(std::move(l.layer));
        }
        for (auto& [type, count] : mSkipped)
            doc.warnings.push_back("skipped " + std::to_string(count) + " " + type + (count == 1 ? "" : " entities"));
        if (mPaperSpace > 0)
            doc.warnings.push_back("skipped " + std::to_string(mPaperSpace) + " paper space entities");
    }
};

// Writing

std::string exactNumber(double v)
{
    if (std::abs(v) < 1e-300)
        v = 0.0;
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    std::string s(buf, r.ptr);
    if (s.find_first_of(".eE") == std::string::npos)
        s += ".0";
    return s;
}

class Writer
{
public:
    explicit Writer(const DxfWriteOptions& o) : mO(o), mModern(o.version == DxfVersion::R2000) {}

    std::string write(const std::vector<DxfObject>& objects)
    {
        std::vector<std::string> layers = layerNames(objects);
        tables(layers);
        blocks();
        pair(0, "SECTION");
        pair(2, "ENTITIES");
        for (size_t k = 0; k < objects.size(); ++k)
        {
            for (const Spline& s : objects[k].curves)
                curve(s, layers[k]);
            if (objects[k].mesh)
                mesh(*objects[k].mesh, layers[k]);
        }
        pair(0, "ENDSEC");
        if (mModern)
            objectsSection();
        pair(0, "EOF");
        std::string body = std::move(mOut);
        mOut.clear();
        header();
        return mOut + body;
    }

private:
    const DxfWriteOptions& mO;
    bool mModern;
    std::string mOut;
    uint64_t mNext = 0x20;
    uint64_t mModelSpace = 0, mPaperSpace = 0;

    void pair(int code, std::string_view v)
    {
        char c[8];
        auto r = std::to_chars(c, c + sizeof(c), code);
        std::string_view cs(c, size_t(r.ptr - c));
        for (size_t k = cs.size(); k < 3; ++k)
            mOut += ' ';
        mOut += cs;
        mOut += '\n';
        mOut += v;
        mOut += '\n';
    }
    void pair(int code, double v) { pair(code, exactNumber(v)); }
    void pairI(int code, int64_t v) { pair(code, std::to_string(v)); }
    void point(int code, const D3& p)
    {
        pair(code, p.x);
        pair(code + 10, p.y);
        pair(code + 20, p.z);
    }

    static std::string hex(uint64_t h)
    {
        char buf[20];
        auto r = std::to_chars(buf, buf + sizeof(buf), h, 16);
        std::string s(buf, r.ptr);
        std::ranges::transform(s, s.begin(), [](char c) { return char(std::toupper(uint8_t(c))); });
        return s;
    }
    uint64_t handle() { return mNext++; }
    /** Handle and owner, R2000 only. */
    uint64_t own(uint64_t owner, int handleCode = 5)
    {
        uint64_t h = handle();
        if (mModern)
        {
            pair(handleCode, hex(h));
            pair(330, hex(owner));
        }
        return h;
    }

    static std::vector<std::string> layerNames(const std::vector<DxfObject>& objects)
    {
        std::vector<std::string> out;
        std::set<std::string> used{"0"};
        for (const DxfObject& o : objects)
        {
            std::string n = o.layer.empty() ? std::string("Layer") : o.layer;
            for (char& c : n)
                if (std::strchr("<>/\\\":;?*|=`,", c))
                    c = '_';
            std::string u = n;
            for (int k = 2; used.count(u); ++k)
                u = n + "_" + std::to_string(k);
            used.insert(u);
            out.push_back(u);
        }
        return out;
    }

    int unitCode() const
    {
        if (mO.unit == "in")
            return 1;
        if (mO.unit == "ft")
            return 2;
        if (mO.unit == "cm")
            return 5;
        if (mO.unit == "m")
            return 6;
        return 4;
    }

    void header()
    {
        pair(0, "SECTION");
        pair(2, "HEADER");
        pair(9, "$ACADVER");
        pair(1, mModern ? "AC1015" : "AC1009");
        if (mModern)
        {
            pair(9, "$HANDSEED");
            pair(5, hex(mNext));
        }
        pair(9, "$INSUNITS");
        pairI(70, unitCode());
        pair(9, "$MEASUREMENT");
        pairI(70, mO.unit == "in" || mO.unit == "ft" ? 0 : 1);
        pair(0, "ENDSEC");
    }

    uint64_t beginTable(const char* name, int count, const char* subclass = nullptr)
    {
        pair(0, "TABLE");
        pair(2, name);
        uint64_t h = own(0);
        if (mModern)
            pair(100, "AcDbSymbolTable");
        pairI(70, count);
        if (mModern && subclass)
            pair(100, subclass);
        return h;
    }
    void endTable() { pair(0, "ENDTAB"); }
    void record(const char* type, uint64_t table, const char* subclass, int handleCode = 5)
    {
        pair(0, type);
        own(table, handleCode);
        if (mModern)
        {
            pair(100, "AcDbSymbolTableRecord");
            pair(100, subclass);
        }
    }

    void tables(const std::vector<std::string>& layers)
    {
        pair(0, "SECTION");
        pair(2, "TABLES");
        if (mModern)
        {
            beginTable("VPORT", 0);
            endTable();
        }
        const char* cont = mModern ? "Continuous" : "CONTINUOUS";
        uint64_t t = beginTable("LTYPE", mModern ? 3 : 1);
        std::vector<const char*> ltypes = mModern ? std::vector<const char*>{"ByBlock", "ByLayer", "Continuous"} : std::vector<const char*>{cont};
        for (const char* lt : ltypes)
        {
            record("LTYPE", t, "AcDbLinetypeTableRecord");
            pair(2, lt);
            pairI(70, 0);
            pair(3, std::strcmp(lt, cont) == 0 ? "Solid line" : "");
            pairI(72, 65);
            pairI(73, 0);
            pair(40, 0.0);
        }
        endTable();
        t = beginTable("LAYER", int(layers.size()) + 1);
        std::vector<std::string> all{"0"};
        all.insert(all.end(), layers.begin(), layers.end());
        for (const std::string& l : all)
        {
            record("LAYER", t, "AcDbLayerTableRecord");
            pair(2, l);
            pairI(70, 0);
            pairI(62, 7);
            pair(6, cont);
        }
        endTable();
        if (!mModern)
        {
            pair(0, "ENDSEC");
            return;
        }
        t = beginTable("STYLE", 1);
        record("STYLE", t, "AcDbTextStyleTableRecord");
        pair(2, "Standard");
        pairI(70, 0);
        pair(40, 0.0);
        pair(41, 1.0);
        pair(50, 0.0);
        pairI(71, 0);
        pair(42, 2.5);
        pair(3, "txt");
        pair(4, "");
        endTable();
        beginTable("VIEW", 0);
        endTable();
        beginTable("UCS", 0);
        endTable();
        t = beginTable("APPID", 1);
        record("APPID", t, "AcDbRegAppTableRecord");
        pair(2, "ACAD");
        pairI(70, 0);
        endTable();
        t = beginTable("DIMSTYLE", 1, "AcDbDimStyleTable");
        record("DIMSTYLE", t, "AcDbDimStyleTableRecord", 105);
        pair(2, "Standard");
        pairI(70, 0);
        endTable();
        t = beginTable("BLOCK_RECORD", 2);
        pair(0, "BLOCK_RECORD");
        mModelSpace = own(t);
        pair(100, "AcDbSymbolTableRecord");
        pair(100, "AcDbBlockTableRecord");
        pair(2, "*Model_Space");
        pair(0, "BLOCK_RECORD");
        mPaperSpace = own(t);
        pair(100, "AcDbSymbolTableRecord");
        pair(100, "AcDbBlockTableRecord");
        pair(2, "*Paper_Space");
        endTable();
        pair(0, "ENDSEC");
    }

    void blocks()
    {
        if (!mModern)
            return;
        pair(0, "SECTION");
        pair(2, "BLOCKS");
        using Space = std::pair<const char*, uint64_t>;
        for (auto [name, rec] : {Space{"*Model_Space", mModelSpace}, Space{"*Paper_Space", mPaperSpace}})
        {
            pair(0, "BLOCK");
            own(rec);
            pair(100, "AcDbEntity");
            if (rec == mPaperSpace)
                pairI(67, 1);
            pair(8, "0");
            pair(100, "AcDbBlockBegin");
            pair(2, name);
            pairI(70, 0);
            point(10, D3(0.0));
            pair(3, name);
            pair(1, "");
            pair(0, "ENDBLK");
            own(rec);
            pair(100, "AcDbEntity");
            if (rec == mPaperSpace)
                pairI(67, 1);
            pair(8, "0");
            pair(100, "AcDbBlockEnd");
        }
        pair(0, "ENDSEC");
    }

    void objectsSection()
    {
        pair(0, "SECTION");
        pair(2, "OBJECTS");
        uint64_t root = handle(), group = handle();
        pair(0, "DICTIONARY");
        pair(5, hex(root));
        pair(330, "0");
        pair(100, "AcDbDictionary");
        pairI(281, 1);
        pair(3, "ACAD_GROUP");
        pair(350, hex(group));
        pair(0, "DICTIONARY");
        pair(5, hex(group));
        pair(330, hex(root));
        pair(100, "AcDbDictionary");
        pairI(281, 1);
        pair(0, "ENDSEC");
    }

    uint64_t entity(const char* type, const std::string& layer, uint64_t owner, const char* subclass)
    {
        pair(0, type);
        uint64_t h = own(owner);
        if (mModern)
            pair(100, "AcDbEntity");
        pair(8, layer);
        if (mModern && subclass)
            pair(100, subclass);
        return h;
    }

    void curve(const Spline& s, const std::string& layer)
    {
        if (s.kind == Kind::Poly)
            return polyline(s.points, s.closed, layer);
        if (!mModern || !mO.splines)
        {
            Spline p = convert(s, Kind::Poly, std::max(1e-9, mO.tolerance));
            return polyline(p.points, p.closed, layer);
        }
        // Closed splines go out as clamped open ones meeting at the start: exact, and read the same by every program.
        Spline n = s.closed ? fromPieces(pieces(s), false, Kind::Nurbs) : toNurbs(s);
        std::vector<double> knots = knotVector(n);
        std::vector<glm::dvec4> cv = controlPoints(n);
        if (cv.size() < 2 || knots.size() != cv.size() + size_t(n.degree) + 1)
            return;
        bool rational = std::ranges::any_of(cv, [](const glm::dvec4& c) { return std::abs(c.w - 1.0) > 1e-12; });
        entity("SPLINE", layer, mModelSpace, "AcDbSpline");
        pairI(70, rational ? 4 : 0);
        pairI(71, n.degree);
        pairI(72, int64_t(knots.size()));
        pairI(73, int64_t(cv.size()));
        pairI(74, 0);
        pair(42, 1e-10);
        pair(43, 1e-10);
        for (double k : knots)
            pair(40, k);
        for (const glm::dvec4& c : cv)
        {
            point(10, D3(c) / c.w);
            if (rational)
                pair(41, c.w);
        }
    }

    void polyline(const std::vector<D3>& pts, bool closed, const std::string& layer)
    {
        if (pts.size() < 2)
            return;
        bool flat = std::ranges::all_of(pts, [&](const D3& p) { return std::abs(p.z - pts[0].z) < 1e-9; });
        if (mModern && flat)
        {
            entity("LWPOLYLINE", layer, mModelSpace, "AcDbPolyline");
            pairI(90, int64_t(pts.size()));
            pairI(70, closed ? 1 : 0);
            pair(38, pts[0].z);
            for (const D3& p : pts)
            {
                pair(10, p.x);
                pair(20, p.y);
            }
            return;
        }
        uint64_t pl = entity("POLYLINE", layer, mModelSpace, "AcDb3dPolyline");
        pairI(66, 1);
        point(10, D3(0.0));
        pairI(70, 8 | (closed ? 1 : 0));
        for (const D3& p : pts)
        {
            entity("VERTEX", layer, pl, "AcDbVertex");
            if (mModern)
                pair(100, "AcDb3dPolylineVertex");
            point(10, p);
            pairI(70, 32);
        }
        entity("SEQEND", layer, pl, nullptr);
    }

    void mesh(const geo::Mesh& m, const std::string& layer)
    {
        constexpr size_t kMaxPoints = 32767;
        // Faces of up to four corners, as point lists; bigger ones triangulated.
        std::vector<geo::Index> facePoints;
        std::vector<size_t> faceStart{0};
        std::vector<geo::Index> tri, pts;
        for (geo::Index f = 0; f < m.faceCapacity(); ++f)
        {
            if (!m.faceValid(f))
                continue;
            if (m.faceSize(f) <= 4)
            {
                m.forEachFaceCorner(f, [&](geo::Index c) { facePoints.push_back(m.cornerPoint(c)); });
                faceStart.push_back(facePoints.size());
                continue;
            }
            tri.clear();
            geo::triangulateFace(m, f, tri);
            for (size_t k = 0; k + 2 < tri.size(); k += 3)
            {
                for (size_t j = 0; j < 3; ++j)
                    facePoints.push_back(m.cornerPoint(tri[k + j]));
                faceStart.push_back(facePoints.size());
            }
        }
        // Polyfaces hold at most kMaxPoints points: faces go in order, a new one starting when the next face's new points
        // would not fit. Each point is numbered from 1 in the polyface that first uses it.
        struct Chunk
        {
            size_t firstFace = 0, endFace = 0;
            std::vector<geo::Index> points;
        };
        std::vector<Chunk> chunks(1);
        std::vector<int> local(m.pointCapacity(), 0);
        std::vector<uint32_t> stamp(m.pointCapacity(), 0);
        uint32_t epoch = 1;
        std::vector<int> faceLocal(facePoints.size());
        const size_t faces = faceStart.size() - 1;
        for (size_t f = 0; f < faces; ++f)
        {
            size_t fresh = 0;
            for (size_t k = faceStart[f]; k < faceStart[f + 1]; ++k)
                fresh += stamp[facePoints[k]] == epoch ? 0 : 1;
            if (chunks.back().points.size() + fresh > kMaxPoints)
            {
                chunks.back().endFace = f;
                chunks.push_back({f, f, {}});
                ++epoch;
            }
            Chunk& c = chunks.back();
            for (size_t k = faceStart[f]; k < faceStart[f + 1]; ++k)
            {
                const geo::Index p = facePoints[k];
                if (stamp[p] != epoch)
                {
                    stamp[p] = epoch;
                    c.points.push_back(p);
                    local[p] = int(c.points.size());
                }
                faceLocal[k] = local[p];
            }
        }
        chunks.back().endFace = faces;
        std::erase_if(chunks, [](const Chunk& c) { return c.endFace == c.firstFace; });

        // Each polyface takes 2 handles plus one per point and face, so they are written in parallel from known handles.
        std::vector<uint64_t> firstHandle(chunks.size());
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            firstHandle[i] = mNext;
            mNext += 2 + chunks[i].points.size() + (chunks[i].endFace - chunks[i].firstFace);
        }
        std::vector<std::string> text(chunks.size());
        tbb::parallel_for(size_t(0), chunks.size(), [&](size_t i) {
            Writer w(mO);
            w.mNext = firstHandle[i];
            w.mModelSpace = mModelSpace;
            const Chunk& c = chunks[i];
            w.polyface(m, c.points, std::span<const size_t>(faceStart).subspan(c.firstFace, c.endFace - c.firstFace + 1), faceLocal, layer);
            text[i] = std::move(w.mOut);
        });
        size_t total = mOut.size();
        for (const std::string& part : text)
            total += part.size();
        mOut.reserve(total);
        for (const std::string& part : text)
            mOut += part;
    }

    /** One polyface POLYLINE: faces [start[0], start.back()) of local, which numbers points from 1. */
    void polyface(const geo::Mesh& m, std::span<const geo::Index> points, std::span<const size_t> start, std::span<const int> local,
                  const std::string& layer)
    {
        uint64_t pl = entity("POLYLINE", layer, mModelSpace, "AcDbPolyFaceMesh");
        pairI(66, 1);
        point(10, D3(0.0));
        pairI(70, 64);
        pairI(71, int64_t(points.size()));
        pairI(72, int64_t(start.size() - 1));
        for (geo::Index p : points)
        {
            entity("VERTEX", layer, pl, "AcDbVertex");
            if (mModern)
                pair(100, "AcDbPolyFaceMeshVertex");
            point(10, m.position(p));
            pairI(70, 192);
        }
        for (size_t f = 0; f + 1 < start.size(); ++f)
        {
            entity("VERTEX", layer, pl, mModern ? "AcDbFaceRecord" : nullptr);
            point(10, D3(0.0));
            pairI(70, 128);
            for (size_t k = start[f]; k < start[f + 1]; ++k)
                pairI(71 + int(k - start[f]), local[k]);
        }
        entity("SEQEND", layer, pl, nullptr);
    }
};
} // namespace

DxfDocument readDxf(std::string_view data)
{
    Groups gs;
    std::string error;
    bool ok = data.substr(0, kBinarySentinel.size()) == kBinarySentinel ? readBinary(data, gs, error) : readAscii(data, gs, error);
    Reader r;
    if (!ok)
    {
        r.doc.error = error;
        return std::move(r.doc);
    }
    if (std::ranges::none_of(gs, [](const Group& g) { return g.code == 0 && g.s == "SECTION"; }))
    {
        r.doc.error = "not a DXF file";
        return std::move(r.doc);
    }
    r.read(gs);
    return std::move(r.doc);
}

double dxfUnitScale(std::string_view unit)
{
    if (unit == "mm")
        return 0.1;
    if (unit == "cm")
        return 1.0;
    if (unit == "m")
        return 100.0;
    if (unit == "in")
        return 2.54;
    if (unit == "ft")
        return 30.48;
    return 0.0;
}

std::string writeDxf(const std::vector<DxfObject>& objects, const DxfWriteOptions& o)
{
    Writer w(o);
    return w.write(objects);
}
} // namespace lila::curve
