#include "geo/PlyIO.h"

#include "core/Decimal.h"

#include "geo/ColorSets.h"
#include "geo/IoCommon.h"
#include "geo/Normals.h"
#include "geo/PointWeld.h"
#include "geo/Primitives.h"

#include "core/FileBytes.h"
#include "core/Paths.h"
#include "core/Profile.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <span>

namespace lila::geo
{
namespace
{
using detail::linearToSrgb;
using detail::srgbToLinear;

enum class Type : uint8_t
{
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
};

std::optional<Type> parseType(std::string_view s)
{
    if (s == "char" || s == "int8")
        return Type::I8;
    if (s == "uchar" || s == "uint8")
        return Type::U8;
    if (s == "short" || s == "int16")
        return Type::I16;
    if (s == "ushort" || s == "uint16")
        return Type::U16;
    if (s == "int" || s == "int32")
        return Type::I32;
    if (s == "uint" || s == "uint32")
        return Type::U32;
    if (s == "float" || s == "float32")
        return Type::F32;
    if (s == "double" || s == "float64")
        return Type::F64;
    return std::nullopt;
}

size_t typeSize(Type t)
{
    switch (t)
    {
    case Type::I8:
    case Type::U8: return 1;
    case Type::I16:
    case Type::U16: return 2;
    case Type::I32:
    case Type::U32:
    case Type::F32: return 4;
    case Type::F64: return 8;
    }
    return 0;
}

/** Integer color channels are normalized by their type's range; floats are already 0 to 1. */
float channelScale(Type t)
{
    switch (t)
    {
    case Type::U16: return 1.0f / 65535.0f;
    case Type::F32:
    case Type::F64: return 1.0f;
    default: return 1.0f / 255.0f;
    }
}

struct Property
{
    std::string name;
    Type type = Type::F32;
    bool list = false;
    Type countType = Type::U8;
};

struct Element
{
    std::string name;
    size_t count = 0;
    std::vector<Property> props;

    int find(std::initializer_list<std::string_view> names) const
    {
        for (std::string_view n : names)
            for (size_t i = 0; i < props.size(); ++i)
                if (props[i].name == n)
                    return int(i);
        return -1;
    }
};

enum class Format
{
    Ascii,
    Little,
    Big,
};

bool wideType(Type t) { return t == Type::F64 || t == Type::I32 || t == Type::U32; }

/** Reads values from one record of the body, in any of the three encodings, never past its end. */
class RecordReader
{
public:
    RecordReader(const char* at, const char* end, Format format) : mAt(at), mEnd(end), mFormat(format) {}

    bool ok() const { return mOk; }
    const char* at() const { return mAt; }

    double value(Type t)
    {
        if (mFormat == Format::Ascii)
            return asciiValue();
        const size_t n = typeSize(t);
        if (size_t(mEnd - mAt) < n)
        {
            mOk = false;
            return 0.0;
        }
        unsigned char b[8];
        std::memcpy(b, mAt, n);
        mAt += n;
        if ((mFormat == Format::Big) == (std::endian::native == std::endian::little))
            std::reverse(b, b + n);
        switch (t)
        {
        case Type::I8: return double(std::bit_cast<int8_t>(b[0]));
        case Type::U8: return double(b[0]);
        case Type::I16: return double(load<int16_t>(b));
        case Type::U16: return double(load<uint16_t>(b));
        case Type::I32: return double(load<int32_t>(b));
        case Type::U32: return double(load<uint32_t>(b));
        case Type::F32: return double(load<float>(b));
        case Type::F64: return load<double>(b);
        }
        return 0.0;
    }

    void skip(Type t, size_t count)
    {
        if (mFormat != Format::Ascii)
        {
            if (size_t(mEnd - mAt) / typeSize(t) < count)
                mOk = false;
            else
                mAt += count * typeSize(t);
            return;
        }
        for (size_t i = 0; i < count && mOk; ++i)
            asciiValue();
    }

private:
    template <class T> static T load(const unsigned char* b)
    {
        T v;
        std::memcpy(&v, b, sizeof(T));
        return v;
    }

    double asciiValue()
    {
        while (mAt < mEnd && std::isspace(static_cast<unsigned char>(*mAt)))
            ++mAt;
        const char* first = mAt;
        if (first < mEnd && *first == '+')
            ++first;
        double v = 0.0;
        auto [ptr, ec] = lila::fromChars(first, mEnd, v);
        if (ec != std::errc() || ptr == first)
        {
            mOk = false;
            return 0.0;
        }
        mAt = ptr;
        return v;
    }

    const char* mAt;
    const char* mEnd;
    Format mFormat;
    bool mOk = true;
};

struct Header
{
    Format format = Format::Ascii;
    std::vector<Element> elements;
    std::vector<std::string> textureFiles;
    size_t bodyStart = 0;
    std::string error;
};

Header parseHeader(std::string_view data)
{
    Header h;
    size_t at = 0;
    auto nextLine = [&](std::string_view& line) {
        if (at >= data.size())
            return false;
        size_t end = data.find('\n', at);
        if (end == std::string_view::npos)
            end = data.size();
        line = data.substr(at, end - at);
        at = std::min(end + 1, data.size());
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        return true;
    };
    auto words = [](std::string_view line) {
        std::vector<std::string_view> w;
        size_t i = 0;
        while (i < line.size())
        {
            while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])))
                ++i;
            const size_t s = i;
            while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i])))
                ++i;
            if (i > s)
                w.push_back(line.substr(s, i - s));
        }
        return w;
    };

    std::string_view line;
    if (!nextLine(line) || line != "ply")
    {
        h.error = "not a PLY file";
        return h;
    }
    bool sawFormat = false;
    while (nextLine(line))
    {
        std::vector<std::string_view> w = words(line);
        if (w.empty())
            continue;
        if (w[0] == "end_header")
        {
            h.bodyStart = at;
            if (!sawFormat)
                h.error = "no format line";
            return h;
        }
        if (w[0] == "comment" || w[0] == "obj_info")
        {
            if (w.size() >= 3 && (w[1] == "TextureFile" || w[1] == "texturefile"))
            {
                const size_t pos = line.find(w[1]) + w[1].size();
                std::string_view rest = line.substr(pos);
                while (!rest.empty() && std::isspace(static_cast<unsigned char>(rest.front())))
                    rest.remove_prefix(1);
                h.textureFiles.emplace_back(rest);
            }
            continue;
        }
        if (w[0] == "format" && w.size() >= 2)
        {
            sawFormat = true;
            if (w[1] == "ascii")
                h.format = Format::Ascii;
            else if (w[1] == "binary_little_endian")
                h.format = Format::Little;
            else if (w[1] == "binary_big_endian")
                h.format = Format::Big;
            else
            {
                h.error = "unknown format \"" + std::string(w[1]) + "\"";
                return h;
            }
            continue;
        }
        if (w[0] == "element" && w.size() >= 3)
        {
            Element& e = h.elements.emplace_back();
            e.name = std::string(w[1]);
            auto [p, ec] = std::from_chars(w[2].data(), w[2].data() + w[2].size(), e.count);
            if (ec != std::errc())
            {
                h.error = "bad element count for \"" + e.name + "\"";
                return h;
            }
            continue;
        }
        if (w[0] == "property" && !h.elements.empty())
        {
            Property p;
            if (w.size() >= 5 && w[1] == "list")
            {
                std::optional<Type> ct = parseType(w[2]), it = parseType(w[3]);
                if (!ct || !it)
                {
                    h.error = "unknown property type in \"" + std::string(line) + "\"";
                    return h;
                }
                p.list = true;
                p.countType = *ct;
                p.type = *it;
                p.name = std::string(w[4]);
            }
            else if (w.size() >= 3)
            {
                std::optional<Type> t = parseType(w[1]);
                if (!t)
                {
                    h.error = "unknown property type in \"" + std::string(line) + "\"";
                    return h;
                }
                p.type = *t;
                p.name = std::string(w[2]);
            }
            else
            {
                h.error = "bad property line \"" + std::string(line) + "\"";
                return h;
            }
            h.elements.back().props.push_back(std::move(p));
            continue;
        }
        h.error = "unexpected header line \"" + std::string(line) + "\"";
        return h;
    }
    h.error = "no end_header";
    return h;
}

/** Values of one property, as float unless the type needs double to stay exact. */
struct Column
{
    bool wide = false;
    UninitVector<double> d;
    UninitVector<float> f;

    void resize(size_t n, Type t)
    {
        wide = wideType(t);
        if (wide)
            d.resize(n);
        else
            f.resize(n);
    }
    void set(size_t k, double v)
    {
        if (wide)
            d[k] = v;
        else
            f[k] = float(v);
    }
    double operator[](size_t k) const { return wide ? d[k] : double(f[k]); }
    size_t size() const { return wide ? d.size() : f.size(); }
};

/** One element's values: scalars as columns, lists flattened with offsets. Only wanted properties are kept. */
struct Columns
{
    std::vector<Column> scalar; // per property, empty when unwanted or a list
    std::vector<Column> listValues;
    std::vector<UninitVector<uint64_t>> listOffsets; // count + 1 per list property kept
};

enum class Pass
{
    Scalars, // wanted scalars, and list lengths into listOffsets
    Lists,   // wanted list values, at the offsets the lengths summed to
    Skip,
};

bool readRecord(const Element& e, const std::vector<bool>& want, RecordReader& c, size_t k, Columns& out, Pass pass)
{
    for (size_t i = 0; i < e.props.size(); ++i)
    {
        const Property& p = e.props[i];
        if (!p.list)
        {
            if (pass == Pass::Scalars && want[i])
                out.scalar[i].set(k, c.value(p.type));
            else
                c.skip(p.type, 1);
            continue;
        }
        const double n = c.value(p.countType);
        if (!c.ok() || !(n >= 0.0 && n <= 1e7) || n != std::floor(n))
            return false;
        const size_t count = size_t(n);
        if (want[i] && pass == Pass::Scalars)
            out.listOffsets[i][k] = count;
        else if (want[i] && pass == Pass::Lists)
        {
            Column& v = out.listValues[i];
            const uint64_t at = out.listOffsets[i][k];
            for (size_t j = 0; j < count; ++j)
                v.set(at + j, c.value(p.type));
            continue;
        }
        c.skip(p.type, count);
    }
    return c.ok();
}

/** Where an element's records start in the body: every stride bytes from base, or listed with the end last. */
struct Records
{
    size_t base = 0;
    size_t stride = 0;
    std::span<const size_t> table;

    size_t start(size_t k) const { return table.empty() ? base + k * stride : table[k]; }
};

/** Start of every line holding more than white space, then the body's end. */
std::vector<size_t> lineStarts(std::string_view body)
{
    constexpr size_t kChunk = size_t(4) << 20;
    const size_t chunks = (body.size() + kChunk - 1) / kChunk;
    std::vector<std::vector<size_t>> found(chunks);
    tbb::parallel_for(size_t(0), chunks, [&](size_t k) {
        const size_t end = std::min(body.size(), (k + 1) * kChunk);
        size_t at = k * kChunk;
        // A chunk owns the lines that start inside it.
        if (at > 0 && body[at - 1] != '\n')
        {
            at = body.find('\n', at);
            at = at == std::string_view::npos ? end : at + 1;
        }
        while (at < end)
        {
            size_t q = at;
            while (q < body.size() && body[q] != '\n' && std::isspace(static_cast<unsigned char>(body[q])))
                ++q;
            if (q < body.size() && body[q] != '\n')
                found[k].push_back(at);
            const size_t eol = body.find('\n', q);
            at = eol == std::string_view::npos ? body.size() : eol + 1;
        }
    });
    size_t total = 1;
    for (const std::vector<size_t>& f : found)
        total += f.size();
    std::vector<size_t> out;
    out.reserve(total);
    for (const std::vector<size_t>& f : found)
        out.insert(out.end(), f.begin(), f.end());
    out.push_back(body.size());
    return out;
}

/** Turns list lengths into offsets in place and returns their sum. */
uint64_t lengthsToOffsets(UninitVector<uint64_t>& v)
{
    return tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, v.size()), uint64_t(0),
        [&](const tbb::blocked_range<size_t>& r, uint64_t sum, bool final) {
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                const uint64_t n = v[i];
                if (final)
                    v[i] = sum;
                sum += n;
            }
            return sum;
        },
        std::plus<uint64_t>());
}

/** Decodes an element's wanted properties from its records, in parallel. */
bool readElement(std::string_view body, Format format, const Element& e, const Records& rec, const std::vector<bool>& want,
                 Columns& out)
{
    const size_t np = e.props.size();
    out.scalar.assign(np, {});
    out.listValues.assign(np, {});
    out.listOffsets.assign(np, {});
    bool lists = false;
    for (size_t i = 0; i < np; ++i)
    {
        if (!want[i])
            continue;
        if (e.props[i].list)
        {
            out.listOffsets[i].resize(e.count + 1);
            out.listOffsets[i][e.count] = 0;
            lists = true;
        }
        else
            out.scalar[i].resize(e.count, e.props[i].type);
    }
    auto run = [&](Pass pass) {
        std::atomic<bool> bad = false;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, e.count, 4096), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t k = r.begin(); k < r.end(); ++k)
            {
                RecordReader c(body.data() + rec.start(k), body.data() + rec.start(k + 1), format);
                if (!readRecord(e, want, c, k, out, pass))
                {
                    bad = true;
                    return;
                }
            }
        });
        return !bad;
    };
    if (!run(Pass::Scalars))
        return false;
    if (!lists)
        return true;
    for (size_t i = 0; i < np; ++i)
        if (!out.listOffsets[i].empty())
            out.listValues[i].resize(lengthsToOffsets(out.listOffsets[i]), e.props[i].type);
    return run(Pass::Lists);
}

/**
 * A binary element's records from byte `at` of the body. Records without lists, or whose lists all have the first
 * record's lengths (a triangle mesh), sit at a fixed stride; otherwise each start is found by walking them.
 */
bool binaryRecords(std::string_view body, Format format, const Element& e, size_t at, Records& rec, std::vector<size_t>& table)
{
    rec = {at, 0, {}};
    if (e.count == 0)
        return true;
    Columns none;
    const std::vector<bool> nothing(e.props.size(), false);
    // Byte ranges of the first record's list lengths.
    std::vector<std::pair<size_t, size_t>> lengths;
    RecordReader first(body.data() + at, body.data() + body.size(), format);
    for (size_t i = 0; i < e.props.size() && first.ok(); ++i)
    {
        const Property& p = e.props[i];
        if (!p.list)
        {
            first.skip(p.type, 1);
            continue;
        }
        const size_t offset = size_t(first.at() - (body.data() + at));
        const double n = first.value(p.countType);
        if (!(n >= 0.0 && n <= 1e7))
            return false;
        lengths.emplace_back(offset, typeSize(p.countType));
        first.skip(p.type, size_t(n));
    }
    if (!first.ok())
        return false;
    const size_t stride = size_t(first.at() - (body.data() + at));
    bool fixed = stride > 0 && (body.size() - at) / stride >= e.count;
    if (fixed && !lengths.empty())
    {
        std::atomic<bool> differs = false;
        tbb::parallel_for(tbb::blocked_range<size_t>(1, e.count, 16384), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t k = r.begin(); k < r.end() && !differs; ++k)
                for (auto [offset, size] : lengths)
                    if (std::memcmp(body.data() + at + offset, body.data() + at + k * stride + offset, size) != 0)
                    {
                        differs = true;
                        break;
                    }
        });
        fixed = !differs;
    }
    if (fixed)
    {
        rec.stride = stride;
        return true;
    }
    table.resize(e.count + 1);
    RecordReader c(body.data() + at, body.data() + body.size(), format);
    for (size_t k = 0; k < e.count; ++k)
    {
        table[k] = size_t(c.at() - body.data());
        if (!readRecord(e, nothing, c, k, none, Pass::Skip))
            return false;
    }
    table[e.count] = size_t(c.at() - body.data());
    rec.table = table;
    return true;
}

/** An ASCII element's records walked one by one, for files that break records across lines. */
bool asciiRecords(std::string_view body, const Element& e, size_t at, std::vector<size_t>& table)
{
    Columns none;
    const std::vector<bool> nothing(e.props.size(), false);
    table.resize(e.count + 1);
    RecordReader c(body.data() + at, body.data() + body.size(), Format::Ascii);
    for (size_t k = 0; k < e.count; ++k)
    {
        table[k] = size_t(c.at() - body.data());
        if (!readRecord(e, nothing, c, k, none, Pass::Skip))
            return false;
    }
    table[e.count] = size_t(c.at() - body.data());
    return true;
}

bool isBuiltinVertexProperty(std::string_view n)
{
    static constexpr std::string_view known[] = {"x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "alpha", "r", "g", "b", "a",
                                                 "diffuse_red", "diffuse_green", "diffuse_blue", "diffuse_alpha", "s", "t", "u", "v",
                                                 "texture_u", "texture_v", "texture_s", "texture_t"};
    return std::ranges::find(known, n) != std::end(known);
}

bool isBuiltinFaceProperty(std::string_view n)
{
    static constexpr std::string_view known[] = {"vertex_indices", "vertex_index", "texcoord", "red", "green", "blue", "alpha", "r", "g", "b", "a"};
    return std::ranges::find(known, n) != std::end(known);
}

/** A property name usable as a Lila attribute, or empty when it would collide with a built-in layer. */
std::string extraAttributeName(std::string_view name)
{
    if (name == attr::kNormal || name == attr::kColor || name == attr::kMaterial || name == attr::kHard || name.starts_with("uv") ||
        name.starts_with(colors::kLayerPrefix))
        return "ply_" + std::string(name);
    return std::string(name);
}

struct ColorProps
{
    int r = -1, g = -1, b = -1, a = -1;
    bool any() const { return r >= 0 && g >= 0 && b >= 0; }
};

ColorProps colorProps(const Element& e)
{
    return {e.find({"red", "r", "diffuse_red"}), e.find({"green", "g", "diffuse_green"}), e.find({"blue", "b", "diffuse_blue"}),
            e.find({"alpha", "a", "diffuse_alpha"})};
}

glm::vec4 colorAt(const Element& e, const Columns& c, const ColorProps& p, size_t k)
{
    auto ch = [&](int i) { return float(c.scalar[size_t(i)][k]) * channelScale(e.props[size_t(i)].type); };
    return glm::vec4(srgbToLinear(ch(p.r)), srgbToLinear(ch(p.g)), srgbToLinear(ch(p.b)), p.a >= 0 ? std::clamp(ch(p.a), 0.0f, 1.0f) : 1.0f);
}
/** Reads every element's wanted properties. Returns the element that failed, or nothing. */
std::optional<std::string> readBody(std::string_view body, const Header& h, const std::vector<std::vector<bool>>& want,
                                    const std::vector<Columns*>& into)
{
    std::vector<size_t> lines;
    bool byLine = h.format == Format::Ascii;
    if (byLine)
        lines = lineStarts(body);
    // ASCII writers put one record per line; when a file doesn't, its records are walked instead.
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        size_t at = 0, line = 0;
        std::optional<std::string> failed;
        for (size_t i = 0; i < h.elements.size() && !failed; ++i)
        {
            const Element& e = h.elements[i];
            Columns discard;
            Columns& out = into[i] ? *into[i] : discard;
            Records rec;
            std::vector<size_t> table;
            bool ok;
            if (h.format != Format::Ascii)
                ok = binaryRecords(body, h.format, e, at, rec, table);
            else if (byLine)
            {
                ok = line + e.count < lines.size();
                if (ok)
                    rec.table = std::span<const size_t>(lines).subspan(line, e.count + 1);
                line += e.count;
            }
            else
            {
                ok = asciiRecords(body, e, at, table);
                rec.table = table;
            }
            const bool wanted = into[i] != nullptr;
            ok = ok && (!wanted || readElement(body, h.format, e, rec, want[i], out));
            if (ok)
                at = rec.start(e.count);
            else
                failed = e.name;
        }
        if (!failed || !byLine)
            return failed;
        byLine = false;
    }
    return std::nullopt;
}

/** File value to a point index, kInvalid when it isn't one. */
Index pointIndex(double v, size_t points) { return v >= 0.0 && v < double(points) ? Index(v) : kInvalid; }

bool hasRepeatedPoint(std::span<const Index> pts)
{
    if (pts.size() > 16)
    {
        std::vector<Index> sorted(pts.begin(), pts.end());
        std::ranges::sort(sorted);
        return std::ranges::adjacent_find(sorted) != sorted.end();
    }
    for (size_t i = 0; i < pts.size(); ++i)
        for (size_t j = i + 1; j < pts.size(); ++j)
            if (pts[i] == pts[j])
                return true;
    return false;
}

} // namespace

PlyReadResult readPly(std::string_view data, const PlyReadOptions& opts)
{
    LILA_ZONE();

    PlyReadResult r;
    Header h = parseHeader(data);
    if (!h.error.empty())
    {
        r.error = h.error;
        return r;
    }
    r.textureFiles = std::move(h.textureFiles);

    const Element* vertexEl = nullptr;
    const Element* faceEl = nullptr;
    const Element* edgeEl = nullptr;
    Columns vertex, face, edge;
    std::vector<std::vector<bool>> want(h.elements.size());
    std::vector<Columns*> into(h.elements.size(), nullptr);
    for (size_t k = 0; k < h.elements.size(); ++k)
    {
        const Element& e = h.elements[k];
        want[k].assign(e.props.size(), false);
        if (e.name == "vertex" && !vertexEl)
        {
            vertexEl = &e;
            into[k] = &vertex;
            for (size_t i = 0; i < e.props.size(); ++i)
                want[k][i] = !e.props[i].list;
        }
        else if (e.name == "face" && !faceEl)
        {
            faceEl = &e;
            into[k] = &face;
            for (size_t i = 0; i < e.props.size(); ++i)
            {
                const Property& p = e.props[i];
                want[k][i] = p.list ? (p.name == "vertex_indices" || p.name == "vertex_index" || p.name == "texcoord") : true;
                if (p.list && !want[k][i])
                    r.skipped.push_back("face property " + p.name);
            }
        }
        else if (e.name == "edge" && !edgeEl)
        {
            edgeEl = &e;
            into[k] = &edge;
            for (size_t i = 0; i < e.props.size(); ++i)
                want[k][i] = e.props[i].name == "vertex1" || e.props[i].name == "vertex2";
        }
        else if (e.count > 0)
            r.skipped.push_back("element " + e.name);
    }
    if (std::optional<std::string> failed = readBody(data.substr(h.bodyStart), h, want, into))
    {
        r.error = "truncated or malformed data in element \"" + *failed + "\"";
        return r;
    }
    if (!vertexEl)
    {
        r.error = "no vertex element";
        return r;
    }
    const int px = vertexEl->find({"x"}), py = vertexEl->find({"y"}), pz = vertexEl->find({"z"});
    if (px < 0 || py < 0 || pz < 0)
    {
        r.error = "vertex element has no x, y, z";
        return r;
    }

    Mesh& m = r.mesh;
    std::vector<size_t> kept;
    const size_t nv = vertexEl->count;
    const int fIdx = faceEl ? faceEl->find({"vertex_indices", "vertex_index"}) : -1;
    const size_t nf = fIdx >= 0 ? faceEl->count : 0;
    {
        UninitVector<glm::dvec3> points(nv);
        const Column &x = vertex.scalar[size_t(px)], &y = vertex.scalar[size_t(py)], &z = vertex.scalar[size_t(pz)];
        tbb::parallel_for(size_t(0), nv, [&](size_t k) { points[k] = glm::dvec3(x[k], y[k], z[k]); });
        UninitVector<Index> sizes(nf), indices;
        if (fIdx >= 0)
        {
            const Column& idx = face.listValues[size_t(fIdx)];
            const UninitVector<uint64_t>& off = face.listOffsets[size_t(fIdx)];
            indices.resize(idx.size());
            std::atomic<bool> anyBad = false;
            tbb::parallel_for(size_t(0), nf, [&](size_t f) {
                sizes[f] = Index(off[f + 1] - off[f]);
                for (uint64_t j = off[f]; j < off[f + 1]; ++j)
                    indices[j] = pointIndex(idx[j], nv);
                const std::span<const Index> pts(indices.data() + off[f], sizes[f]);
                if (pts.size() < 3 || std::ranges::find(pts, kInvalid) != pts.end() || hasRepeatedPoint(pts))
                    anyBad = true;
            });
            // Faces addFace would refuse are dropped here, so the rest keep their order and kept maps them to the file's.
            if (anyBad)
            {
                UninitVector<Index> goodSizes, goodIndices;
                for (size_t f = 0; f < nf; ++f)
                {
                    const std::span<const Index> pts(indices.data() + off[f], sizes[f]);
                    if (pts.size() < 3 || std::ranges::find(pts, kInvalid) != pts.end() || hasRepeatedPoint(pts))
                        continue;
                    kept.push_back(f);
                    goodSizes.push_back(sizes[f]);
                    goodIndices.insert(goodIndices.end(), pts.begin(), pts.end());
                }
                r.skipped.push_back(std::to_string(nf - kept.size()) + " bad faces");
                sizes = std::move(goodSizes);
                indices = std::move(goodIndices);
            }
        }
        m = Mesh::fromPolygons(points, sizes, indices);
        freeInBackground(std::move(points));
        freeInBackground(std::move(indices));
    }
    auto source = [&](Index f) { return kept.empty() ? size_t(f) : kept[f]; };
    if (edgeEl && edgeEl->find({"vertex1"}) >= 0 && edgeEl->find({"vertex2"}) >= 0)
    {
        const Column& a = edge.scalar[size_t(edgeEl->find({"vertex1"}))];
        const Column& b = edge.scalar[size_t(edgeEl->find({"vertex2"}))];
        for (size_t k = 0; k < edgeEl->count; ++k)
        {
            const Index pa = pointIndex(a[k], nv), pb = pointIndex(b[k], nv);
            if (pa != kInvalid && pb != kInvalid && pa != pb)
                m.addEdge(pa, pb);
        }
    }
    r.pointCloud = m.numFaces() == 0 && m.numEdges() == 0;

    const int nx = vertexEl->find({"nx"}), ny = vertexEl->find({"ny"}), nz = vertexEl->find({"nz"});
    const bool hasNormals = opts.keepNormals && nx >= 0 && ny >= 0 && nz >= 0;
    const ColorProps vcol = colorProps(*vertexEl);
    const int ps = vertexEl->find({"s", "u", "texture_u", "texture_s"}), pt = vertexEl->find({"t", "v", "texture_v", "texture_t"});
    const bool hasUv = ps >= 0 && pt >= 0;
    auto normalAt = [&](size_t k) {
        return glm::vec3(float(vertex.scalar[size_t(nx)][k]), float(vertex.scalar[size_t(ny)][k]), float(vertex.scalar[size_t(nz)][k]));
    };
    auto uvAt = [&](size_t k) { return glm::vec2(float(vertex.scalar[size_t(ps)][k]), float(vertex.scalar[size_t(pt)][k])); };
    auto perPoint = [&](auto&& fn) { tbb::parallel_for(size_t(0), nv, fn); };
    auto perCorner = [&](auto&& fn) { tbb::parallel_for(Index(0), m.cornerCapacity(), fn); };
    auto perFace = [&](auto&& fn) { tbb::parallel_for(Index(0), m.faceCapacity(), fn); };

    for (size_t i = 0; i < vertexEl->props.size(); ++i)
    {
        const Property& p = vertexEl->props[i];
        if (p.list)
        {
            r.skipped.push_back("vertex property " + p.name);
            continue;
        }
        if (isBuiltinVertexProperty(p.name))
            continue;
        std::span<float> layer = m.pointAttrs().getOrAdd<float>(extraAttributeName(p.name));
        perPoint([&](size_t k) { layer[k] = float(vertex.scalar[i][k]); });
    }

    if (m.numFaces() == 0)
    {
        if (hasNormals)
        {
            std::span<glm::vec3> n = m.pointAttrs().getOrAdd<glm::vec3>(attr::kNormal);
            perPoint([&](size_t k) { n[k] = normalAt(k); });
        }
        if (vcol.any())
        {
            std::span<glm::vec4> c = m.pointAttrs().getOrAdd<glm::vec4>(attr::kColor);
            perPoint([&](size_t k) { c[k] = colorAt(*vertexEl, vertex, vcol, k); });
        }
        if (hasUv)
        {
            std::span<glm::vec2> uv = m.pointAttrs().getOrAdd<glm::vec2>(attr::uvName(0));
            perPoint([&](size_t k) { uv[k] = uvAt(k); });
        }
        return r;
    }

    // Faces were added in order without deletions, so face and corner ids are contiguous.
    if (hasNormals)
    {
        std::span<glm::vec3> n = m.cornerAttrs().getOrAdd<glm::vec3>(attr::kNormal);
        perCorner([&](Index c) { n[c] = normalAt(m.cornerPoint(c)); });
    }
    const int tc = faceEl->find({"texcoord"});
    if (tc >= 0 && faceEl->props[size_t(tc)].list)
    {
        const Column& v = face.listValues[size_t(tc)];
        const UninitVector<uint64_t>& off = face.listOffsets[size_t(tc)];
        std::span<glm::vec2> uv = m.cornerAttrs().getOrAdd<glm::vec2>(attr::uvName(0));
        perFace([&](Index f) {
            const size_t src = source(f);
            if (off[src + 1] - off[src] != uint64_t(m.faceSize(f)) * 2)
                return;
            uint64_t j = off[src];
            m.forEachFaceCorner(f, [&](Index c) {
                uv[c] = glm::vec2(float(v[j]), float(v[j + 1]));
                j += 2;
            });
        });
        markUvSeams(m, attr::uvName(0));
    }
    else if (hasUv)
    {
        std::span<glm::vec2> uv = m.cornerAttrs().getOrAdd<glm::vec2>(attr::uvName(0));
        perCorner([&](Index c) { uv[c] = uvAt(m.cornerPoint(c)); });
    }
    if (vcol.any())
    {
        UninitVector<glm::vec4> pointColors(nv);
        perPoint([&](size_t k) { pointColors[k] = colorAt(*vertexEl, vertex, vcol, k); });
        colors::add(m, "colorSet");
        colors::setFromPoints(m, "colorSet", pointColors);
    }
    const ColorProps fcol = colorProps(*faceEl);
    if (fcol.any())
    {
        const std::string set = vcol.any() ? "faceColor" : "colorSet";
        std::span<glm::vec4> col = colors::add(m, set);
        perFace([&](Index f) {
            const glm::vec4 c = colorAt(*faceEl, face, fcol, source(f));
            m.forEachFaceCorner(f, [&](Index k) { col[k] = c; });
        });
    }
    for (size_t i = 0; i < faceEl->props.size(); ++i)
    {
        const Property& p = faceEl->props[i];
        if (p.list || isBuiltinFaceProperty(p.name))
            continue;
        std::span<float> layer = m.faceAttrs().getOrAdd<float>(extraAttributeName(p.name));
        perFace([&](Index f) { layer[f] = float(face.scalar[i][source(f)]); });
    }

    // Points no face or edge uses.
    const bool anyLoose = tbb::parallel_reduce(
        tbb::blocked_range<Index>(0, m.pointCapacity()), false,
        [&](const tbb::blocked_range<Index>& rg, bool found) {
            for (Index p = rg.begin(); p < rg.end() && !found; ++p)
                found = m.pointEdge(p) == kInvalid;
            return found;
        },
        std::logical_or<bool>());
    if (anyLoose)
    {
        std::vector<Index> loose;
        for (Index p = 0; p < m.pointCapacity(); ++p)
            if (m.pointEdge(p) == kInvalid)
                loose.push_back(p);
        m.removeLoose({}, loose);
        m.compact();
    }
    return r;
}

PlyReadResult readPlyFile(const std::string& path, const PlyReadOptions& opts)
{
    UninitVector<char> data;
    try
    {
        data = readFileBytes(paths::fromUtf8(path));
    }
    catch (const std::runtime_error&)
    {
        PlyReadResult r;
        r.error = "cannot open " + path;
        return r;
    }
    PlyReadResult r = readPly(std::string_view(data.data(), data.size()), opts);
    freeInBackground(std::move(data));
    return r;
}

namespace
{
/** A written vertex: its point and the attributes it carries, compared byte for byte. */
struct VertexKey
{
    Index point;
    std::array<uint32_t, 9> attrs; // normal, uv and color bits
};

template <class T> void putRaw(char*& at, T v)
{
    std::memcpy(at, &v, sizeof(T));
    if constexpr (std::endian::native == std::endian::big)
        std::reverse(at, at + sizeof(T));
    at += sizeof(T);
}

void putText(std::string& s, float v, int precision)
{
    char buf[64];
    auto r = precision > 0 ? std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, precision) : std::to_chars(buf, buf + sizeof(buf), v);
    s.append(buf, r.ptr);
}

void putText(std::string& s, uint64_t v)
{
    char buf[24];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    s.append(buf, r.ptr);
}

/** Text for items [0, n) in parallel blocks, joined in order. */
template <class Fn> void appendBlocks(std::string& s, size_t n, Fn&& item)
{
    constexpr size_t kBlock = 16384;
    std::vector<std::string> parts((n + kBlock - 1) / kBlock);
    tbb::parallel_for(size_t(0), parts.size(), [&](size_t b) {
        std::string& out = parts[b];
        for (size_t i = b * kBlock; i < std::min(n, (b + 1) * kBlock); ++i)
            item(out, i);
    });
    size_t total = s.size();
    for (const std::string& p : parts)
        total += p.size();
    s.reserve(total);
    for (const std::string& p : parts)
        s += p;
}
} // namespace

std::string writePly(const Mesh& mesh, const PlyWriteOptions& opts)
{
    LILA_ZONE();

    std::vector<glm::vec3> computedNormals;
    std::span<const glm::vec3> normals;
    if (opts.normals && mesh.numFaces() > 0)
    {
        normals = mesh.cornerAttrs().get<glm::vec3>(attr::kNormal);
        if (normals.empty())
        {
            computedNormals = cornerNormals(mesh);
            normals = computedNormals;
        }
    }
    std::span<const glm::vec2> uvs = opts.uvs ? mesh.cornerAttrs().get<glm::vec2>(attr::uvName(0)) : std::span<const glm::vec2>();
    const std::string set = opts.colors ? colors::current(mesh) : std::string();
    std::span<const glm::vec4> cols = set.empty() ? std::span<const glm::vec4>() : colors::values(mesh, set);
    // Point clouds carry their attributes per point.
    std::span<const glm::vec3> pointNormals = opts.normals ? mesh.pointAttrs().get<glm::vec3>(attr::kNormal) : std::span<const glm::vec3>();
    std::span<const glm::vec4> pointCols = opts.colors ? mesh.pointAttrs().get<glm::vec4>(attr::kColor) : std::span<const glm::vec4>();
    std::span<const glm::vec2> pointUvs = opts.uvs ? mesh.pointAttrs().get<glm::vec2>(attr::uvName(0)) : std::span<const glm::vec2>();
    const bool writeNormals = !normals.empty() || !pointNormals.empty();
    const bool writeUvs = !uvs.empty() || !pointUvs.empty();
    const bool writeColors = !cols.empty() || !pointCols.empty();

    // Corners in write order, faces in id order, then the points no face uses.
    std::vector<Index> faces;
    faces.reserve(mesh.numFaces());
    size_t maxFace = 0;
    for (Index f = 0; f < mesh.faceCapacity(); ++f)
        if (mesh.faceValid(f))
        {
            faces.push_back(f);
            maxFace = std::max<size_t>(maxFace, mesh.faceSize(f));
        }
    UninitVector<size_t> faceStart(faces.size() + 1);
    faceStart[0] = 0;
    for (size_t i = 0; i < faces.size(); ++i)
        faceStart[i + 1] = faceStart[i] + mesh.faceSize(faces[i]);
    const size_t cornerCount = faceStart[faces.size()];
    UninitVector<Index> corners(cornerCount);
    std::vector<std::atomic<uint8_t>> used(mesh.pointCapacity());
    tbb::parallel_for(size_t(0), faces.size(), [&](size_t i) {
        size_t at = faceStart[i];
        mesh.forEachFaceCorner(faces[i], [&](Index c) {
            corners[at++] = c;
            used[mesh.cornerPoint(c)].store(1, std::memory_order_relaxed);
        });
    });
    std::vector<Index> unused;
    for (Index p = 0; p < mesh.pointCapacity(); ++p)
        if (mesh.pointValid(p) && !used[p].load(std::memory_order_relaxed))
            unused.push_back(p);

    auto keyFor = [&](Index p, Index c) {
        glm::vec3 n(0.0f);
        glm::vec2 uv(0.0f);
        glm::vec4 col(0.0f);
        if (writeNormals)
            n = c != kInvalid && !normals.empty() ? normals[c] : (!pointNormals.empty() ? pointNormals[p] : glm::vec3(0.0f));
        if (writeUvs)
            uv = c != kInvalid && !uvs.empty() ? uvs[c] : (!pointUvs.empty() ? pointUvs[p] : glm::vec2(0.0f));
        if (writeColors)
            col = c != kInvalid && !cols.empty() ? cols[c] : (!pointCols.empty() ? pointCols[p] : glm::vec4(1.0f));
        VertexKey k{p, {}};
        std::memcpy(&k.attrs[0], &n, 12);
        std::memcpy(&k.attrs[3], &uv, 8);
        std::memcpy(&k.attrs[5], &col, 16);
        return k;
    };
    UninitVector<VertexKey> keys(cornerCount + unused.size());
    tbb::parallel_for(size_t(0), keys.size(), [&](size_t i) {
        keys[i] = i < cornerCount ? keyFor(mesh.cornerPoint(corners[i]), corners[i]) : keyFor(unused[i - cornerCount], kInvalid);
    });
    UninitVector<Index> vertexOf, firstKey;
    numberDistinct<VertexKey>(keys, vertexOf, firstKey);
    const size_t vertexCount = firstKey.size();
    // A point's vertex, for loose edges: the first written for it.
    std::vector<uint32_t> pointVertex(mesh.pointCapacity(), ~0u);
    for (size_t i = 0; i < keys.size(); ++i)
        if (pointVertex[keys[i].point] == ~0u)
            pointVertex[keys[i].point] = vertexOf[i];
    std::vector<std::pair<uint32_t, uint32_t>> looseEdges;
    for (Index e = 0; e < mesh.edgeCapacity(); ++e)
        if (mesh.edgeValid(e) && mesh.edgeCorner(e) == kInvalid)
            looseEdges.emplace_back(pointVertex[mesh.edgePoint(e, 0)], pointVertex[mesh.edgePoint(e, 1)]);

    std::string s = "ply\nformat ";
    s += opts.binary ? "binary_little_endian 1.0\n" : "ascii 1.0\n";
    s += "comment Lila\n";
    s += "element vertex " + std::to_string(vertexCount) + "\n";
    s += "property float x\nproperty float y\nproperty float z\n";
    if (writeNormals)
        s += "property float nx\nproperty float ny\nproperty float nz\n";
    if (writeUvs)
        s += "property float s\nproperty float t\n";
    if (writeColors)
        s += "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n";
    const bool wideCount = maxFace > 255;
    if (!faces.empty())
    {
        s += "element face " + std::to_string(faces.size()) + "\n";
        s += wideCount ? "property list int int vertex_indices\n" : "property list uchar int vertex_indices\n";
    }
    if (!looseEdges.empty())
        s += "element edge " + std::to_string(looseEdges.size()) + "\nproperty int vertex1\nproperty int vertex2\n";
    s += "end_header\n";

    auto byte = [](float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    auto vertexValues = [&](size_t v, auto&& f32, auto&& u8) {
        const VertexKey& k = keys[firstKey[v]];
        const glm::dvec3& p = mesh.position(k.point);
        for (int i = 0; i < 3; ++i)
            f32(float(p[i]));
        float a[9];
        std::memcpy(a, k.attrs.data(), sizeof(a));
        if (writeNormals)
            for (int i = 0; i < 3; ++i)
                f32(a[i]);
        if (writeUvs)
        {
            f32(a[3]);
            f32(a[4]);
        }
        if (writeColors)
        {
            for (int i = 0; i < 3; ++i)
                u8(byte(linearToSrgb(a[5 + i])));
            u8(byte(a[8]));
        }
    };
    if (opts.binary)
    {
        const size_t stride = 12 + (writeNormals ? 12 : 0) + (writeUvs ? 8 : 0) + (writeColors ? 4 : 0);
        const size_t countBytes = wideCount ? 4 : 1;
        const size_t head = s.size(), facesAt = head + vertexCount * stride;
        const size_t edgesAt = facesAt + faces.size() * countBytes + cornerCount * 4;
        s.resize(edgesAt + looseEdges.size() * 8);
        tbb::parallel_for(size_t(0), vertexCount, [&](size_t v) {
            char* at = s.data() + head + v * stride;
            vertexValues(v, [&](float x) { putRaw(at, x); }, [&](uint8_t x) { putRaw(at, x); });
        });
        tbb::parallel_for(size_t(0), faces.size(), [&](size_t i) {
            char* at = s.data() + facesAt + i * countBytes + faceStart[i] * 4;
            const size_t n = faceStart[i + 1] - faceStart[i];
            if (wideCount)
                putRaw(at, int32_t(n));
            else
                putRaw(at, uint8_t(n));
            for (size_t j = faceStart[i]; j < faceStart[i + 1]; ++j)
                putRaw(at, int32_t(vertexOf[j]));
        });
        char* at = s.data() + edgesAt;
        for (auto [a, b] : looseEdges)
        {
            putRaw(at, int32_t(a));
            putRaw(at, int32_t(b));
        }
        return s;
    }
    appendBlocks(s, vertexCount, [&](std::string& out, size_t v) {
        bool first = true;
        auto sep = [&] {
            if (!first)
                out += ' ';
            first = false;
        };
        vertexValues(
            v,
            [&](float x) {
                sep();
                putText(out, x, opts.precision);
            },
            [&](uint8_t x) {
                sep();
                putText(out, uint64_t(x));
            });
        out += '\n';
    });
    appendBlocks(s, faces.size(), [&](std::string& out, size_t i) {
        putText(out, uint64_t(faceStart[i + 1] - faceStart[i]));
        for (size_t j = faceStart[i]; j < faceStart[i + 1]; ++j)
        {
            out += ' ';
            putText(out, uint64_t(vertexOf[j]));
        }
        out += '\n';
    });
    for (auto [a, b] : looseEdges)
    {
        putText(s, uint64_t(a));
        s += ' ';
        putText(s, uint64_t(b));
        s += '\n';
    }
    return s;
}
} // namespace lila::geo
