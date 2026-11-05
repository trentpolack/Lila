#include "io/SceneFormats.h"

#include "core/Decimal.h"
#include "core/PackedArray.h"
#include "core/Profile.h"
#include "core/UninitVector.h"

#include <tbb/parallel_for.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <charconv>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <sstream>
#include <stdexcept>

namespace lila::io
{
using json = nlohmann::json;

namespace
{
constexpr char kMagic[8] = {'L', 'I', 'L', 'A', '\0', '\r', '\n', '\x1a'};
// 3: blobs carry a "dtype" (f32, i32, u32, bool; f64 when absent, as in every version 2 file) and big ones continue
// in "BLB+" chunks.
constexpr uint32_t kBinaryVersion = 3;
constexpr uint32_t kAsciiVersion = 2;
constexpr size_t kBlobThreshold = 64; // numbers; smaller arrays stay inline in the JSON chunk
constexpr uint32_t kFlagZstd = 1;
// Level 1 is several times faster than the default 3 on mesh data and only a few percent bigger.
constexpr int kZstdLevel = 1;
// Big arrays split into parts of this size, so one huge mesh does not leave a single thread compressing it.
constexpr size_t kPartBytes = size_t(1) << 20;

uint32_t crc32(const void* data, size_t size)
{
    static const auto table = [] {
        std::array<std::array<uint32_t, 256>, 8> t{};
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; ++i)
            for (int k = 1; k < 8; ++k)
                t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFF];
        return t;
    }();
    uint32_t c = 0xFFFFFFFFu;
    auto* p = static_cast<const uint8_t*>(data);
    // Slicing-by-8: eight bytes per step through eight tables, the same result as the bytewise tail loop.
    for (; size >= 8; size -= 8, p += 8)
    {
        uint32_t lo, hi;
        std::memcpy(&lo, p, 4);
        std::memcpy(&hi, p + 4, 4);
        lo ^= c;
        c = table[7][lo & 0xFF] ^ table[6][(lo >> 8) & 0xFF] ^ table[5][(lo >> 16) & 0xFF] ^ table[4][lo >> 24] ^
            table[3][hi & 0xFF] ^ table[2][(hi >> 8) & 0xFF] ^ table[1][(hi >> 16) & 0xFF] ^ table[0][hi >> 24];
    }
    for (; size; --size, ++p)
        c = table[0][(c ^ *p) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

template <class T> void put(std::string& out, T v)
{
    char buf[sizeof(T)];
    std::memcpy(buf, &v, sizeof(T));
    out.append(buf, sizeof(T));
}

template <class T> T get(std::string_view in, size_t& at)
{
    if (at + sizeof(T) > in.size())
        throw std::runtime_error("truncated file");
    T v;
    std::memcpy(&v, in.data() + at, sizeof(T));
    at += sizeof(T);
    return v;
}

struct Blob
{
    packed::DType dtype = packed::DType::F64;
    std::vector<uint8_t> bytes;
    std::vector<size_t> shape;
};

/** A numeric array (possibly nested evenly, like a list of vec3) and its shape, or false. */
bool numericShape(const json& j, std::vector<size_t>& shape, size_t& count)
{
    if (!j.is_array() || j.empty())
        return false;
    if (j[0].is_number())
    {
        for (const json& v : j)
            if (!v.is_number())
                return false;
        shape = {j.size()};
        count = j.size();
        return true;
    }
    if (!j[0].is_array())
        return false;
    size_t width = j[0].size();
    if (width == 0 || width > 16)
        return false;
    for (const json& row : j)
    {
        if (!row.is_array() || row.size() != width)
            return false;
        for (const json& v : row)
            if (!v.is_number())
                return false;
    }
    shape = {j.size(), width};
    count = j.size() * width;
    return true;
}

/** Numeric arrays big enough to be blobs, in document order; their contents are left for the caller. */
void findBlobs(json& j, std::vector<std::pair<json*, std::vector<size_t>>>& found)
{
    if (packed::is(j))
    {
        if (packed::count(j) < kBlobThreshold)
            j = packed::unpack(j);
        else
            found.push_back({&j, {}});
        return;
    }
    std::vector<size_t> shape;
    size_t count = 0;
    if (j.is_array() && numericShape(j, shape, count) && count >= kBlobThreshold)
    {
        found.push_back({&j, std::move(shape)});
        return;
    }
    if (j.is_array() || j.is_object())
        for (auto& child : j)
            findBlobs(child, found);
}

/** The narrowest type that holds every value exactly; integers stay integers so they read back as such. */
packed::DType narrowest(const json& a, bool nested)
{
    bool i32 = true, u32 = true, f32 = true;
    auto visit = [&](const json& v) {
        if (v.is_number_unsigned())
        {
            const uint64_t u = v.get<uint64_t>();
            i32 = i32 && u <= uint64_t(INT32_MAX);
            u32 = u32 && u <= uint64_t(UINT32_MAX);
        }
        else if (v.is_number_integer())
        {
            const int64_t s = v.get<int64_t>();
            i32 = i32 && s >= INT32_MIN && s <= INT32_MAX;
            u32 = u32 && s >= 0 && s <= int64_t(UINT32_MAX);
        }
        else
            i32 = u32 = false;
        const double d = v.get<double>();
        f32 = f32 && double(float(d)) == d;
    };
    for (const json& v : a)
        if (nested)
            for (const json& x : v)
                visit(x);
        else
            visit(v);
    if (i32)
        return packed::DType::I32;
    if (u32)
        return packed::DType::U32;
    return f32 ? packed::DType::F32 : packed::DType::F64;
}

template <class T> void fill(const json& a, bool nested, std::vector<uint8_t>& out, size_t count)
{
    out.resize(count * sizeof(T));
    T* p = reinterpret_cast<T*>(out.data());
    for (const json& v : a)
        if (nested)
            for (const json& x : v)
                *p++ = x.get<T>();
        else
            *p++ = v.get<T>();
}

void extractBlobs(json& j, std::vector<Blob>& blobs)
{
    std::vector<std::pair<json*, std::vector<size_t>>> found;
    findBlobs(j, found);
    blobs.resize(found.size());
    tbb::parallel_for(size_t(0), found.size(), [&](size_t i) {
        json& a = *found[i].first;
        Blob& b = blobs[i];
        if (packed::is(a))
        {
            b.dtype = packed::dtype(a);
            const size_t n = packed::count(a);
            const size_t w = size_t(packed::width(a));
            b.shape = w > 1 ? std::vector<size_t>{n / w, w} : std::vector<size_t>{n};
            b.bytes = std::move(a.get_binary());
        }
        else
        {
            b.shape = found[i].second;
            const bool nested = b.shape.size() == 2;
            const size_t n = nested ? b.shape[0] * b.shape[1] : b.shape[0];
            b.dtype = narrowest(a, nested);
            switch (b.dtype)
            {
            case packed::DType::F64: fill<double>(a, nested, b.bytes, n); break;
            case packed::DType::F32: fill<float>(a, nested, b.bytes, n); break;
            case packed::DType::I32: fill<int32_t>(a, nested, b.bytes, n); break;
            case packed::DType::U32: fill<uint32_t>(a, nested, b.bytes, n); break;
            case packed::DType::Bool: break; // narrowest() only picks numeric types
            }
        }
        json placeholder = {{"$blob", i}, {"shape", b.shape}};
        if (b.dtype != packed::DType::F64)
            placeholder["dtype"] = packed::dtypeName(b.dtype);
        a = std::move(placeholder);
    });
}

/** Each placeholder, and whether it sits inside a mesh (which meshFromJson reads packed, so it stays packed). */
/** A blob's bytes as read from a binary scene. */
using BlobBytes = UninitVector<char>;

void findPlaceholders(json& j, std::vector<std::pair<json*, bool>>& found, bool inMesh = false)
{
    if (j.is_object() && j.contains("$blob"))
    {
        found.emplace_back(&j, inMesh);
        return;
    }
    if (j.is_object() && j.contains("faceSizes") && j.contains("points"))
        inMesh = true;
    if (j.is_array() || j.is_object())
        for (auto& child : j)
            findPlaceholders(child, found, inMesh);
}

void restoreBlobs(json& j, const std::vector<BlobBytes>& blobs, std::vector<std::string>* warnings)
{
    struct Use
    {
        json* at;
        const BlobBytes* bytes; // null when missing, damaged or the wrong size
        packed::DType dtype = packed::DType::F64;
        int width = 1;
        bool keepPacked = false;
    };
    std::vector<std::pair<json*, bool>> found;
    findPlaceholders(j, found);
    std::vector<Use> uses;
    uses.reserve(found.size());
    for (auto [p, inMesh] : found)
    {
        Use& u = uses.emplace_back(Use{p, nullptr});
        u.keepPacked = inMesh;
        const size_t index = (*p)["$blob"].get<size_t>();
        const std::vector<size_t> shape = p->value("shape", std::vector<size_t>{});
        bool known = true;
        if (auto it = p->find("dtype"); it != p->end())
            known = it->is_string() && packed::dtypeFromName(it->get_ref<const std::string&>(), u.dtype);
        u.width = shape.size() == 2 && shape[1] > 0 ? int(shape[1]) : 1;
        size_t expected = 0;
        for (size_t k = 0; k < shape.size(); ++k)
            expected = k ? expected * shape[k] : shape[k];
        if (known && index < blobs.size() && !blobs[index].empty() &&
            (shape.empty() || blobs[index].size() == expected * packed::dtypeSize(u.dtype)))
            u.bytes = &blobs[index];
        else if (warnings)
            warnings->push_back("data block " + std::to_string(index) + " is missing or damaged; that array is empty");
    }
    tbb::parallel_for(size_t(0), uses.size(), [&](size_t i) {
        const Use& u = uses[i];
        if (!u.bytes)
            *u.at = json::array();
        else if (u.keepPacked)
            *u.at = json::binary(json::binary_t::container_type(u.bytes->begin(), u.bytes->end()),
                                 uint64_t(u.dtype) | uint64_t(u.width) << 8);
        else
            *u.at = packed::toArray(u.dtype, reinterpret_cast<const uint8_t*>(u.bytes->data()),
                                    u.bytes->size() / packed::dtypeSize(u.dtype), u.width);
    });
}

struct Chunk
{
    char fourcc[4];
    std::string_view raw;
    std::string stored;
    uint32_t crc = 0;
};

// One zstd context per thread: making a fresh one for every chunk costs more than compressing a small chunk.
struct FreeCCtx
{
    void operator()(ZSTD_CCtx* c) const { ZSTD_freeCCtx(c); }
};
struct FreeDCtx
{
    void operator()(ZSTD_DCtx* c) const { ZSTD_freeDCtx(c); }
};

ZSTD_CCtx* compressContext()
{
    thread_local std::unique_ptr<ZSTD_CCtx, FreeCCtx> ctx(ZSTD_createCCtx());
    return ctx.get();
}

ZSTD_DCtx* decompressContext()
{
    thread_local std::unique_ptr<ZSTD_DCtx, FreeDCtx> ctx(ZSTD_createDCtx());
    return ctx.get();
}

void compress(Chunk& c)
{
    LILA_ZONE();

    c.crc = crc32(c.raw.data(), c.raw.size());
    // Compressing into a reused per-thread buffer and keeping only the result touches far less fresh memory than a
    // worst-case sized buffer per chunk would.
    thread_local std::vector<char> scratch;
    scratch.resize(std::max(scratch.size(), ZSTD_compressBound(c.raw.size())));
    size_t n = ZSTD_compressCCtx(compressContext(), scratch.data(), scratch.size(), c.raw.data(), c.raw.size(), kZstdLevel);
    if (ZSTD_isError(n))
        throw std::runtime_error(std::string("compression failed: ") + ZSTD_getErrorName(n));
    c.stored.assign(scratch.data(), n);
}

// ------------------------------------------------------------------- ASCII helpers

/** Text to append to. MSVC's std::ostringstream stops at 2 GB, which a dense mesh's text runs past. */
struct Text
{
    std::string s;

    Text& operator<<(std::string_view v)
    {
        s.append(v);
        return *this;
    }
    Text& operator<<(const std::string& v) { return *this << std::string_view(v); }
    Text& operator<<(const char* v) { return *this << std::string_view(v); }
    Text& operator<<(char c)
    {
        s.push_back(c);
        return *this;
    }
    Text& operator<<(uint32_t v)
    {
        char buf[16];
        s.append(buf, std::to_chars(buf, buf + sizeof(buf), v).ptr);
        return *this;
    }
};

std::string number(double v)
{
    if (v == 0.0)
        return "0"; // also folds -0
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof(buf), v); // shortest form that round-trips exactly
    return std::string(buf, r.ptr);
}


std::string scalar(const json& v)
{
    switch (v.type())
    {
    case json::value_t::number_float: return number(v.get<double>());
    case json::value_t::number_integer:
    case json::value_t::number_unsigned: return v.dump();
    case json::value_t::boolean: return v.get<bool>() ? "true" : "false";
    case json::value_t::null: return "null";
    case json::value_t::string: return v.dump(); // quoted with JSON escapes
    default: return "json " + v.dump();
    }
}

/** scalar() appended in place, without a string per number: the bulk of a mesh's text. */
void appendScalar(Text& out, const json& v)
{
    char buf[32];
    if (v.is_number_float())
    {
        const double d = v.get<double>();
        if (d == 0.0)
            out.s.push_back('0');
        else
            out.s.append(buf, std::to_chars(buf, buf + sizeof(buf), d).ptr);
    }
    else if (v.is_number_unsigned())
        out.s.append(buf, std::to_chars(buf, buf + sizeof(buf), v.get<uint64_t>()).ptr);
    else if (v.is_number_integer())
        out.s.append(buf, std::to_chars(buf, buf + sizeof(buf), v.get<int64_t>()).ptr);
    else
        out << scalar(v);
}

bool isFlatNumbers(const json& v)
{
    if (!v.is_array())
        return false;
    for (const json& x : v)
        if (!x.is_number())
            return false;
    return true;
}

/**
 * One value of a packed array, formatted as its plain json value would be by appendScalar, except that f32 values take
 * the shortest text that reads back as the same float: half the digits of the double they would print as.
 */
void appendPacked(std::string& s, packed::DType t, const uint8_t* data, size_t i)
{
    char buf[32];
    switch (t)
    {
    case packed::DType::Bool: s.append(data[i] ? "true" : "false"); return;
    case packed::DType::I32: { int32_t v; std::memcpy(&v, data + i * 4, 4); s.append(buf, std::to_chars(buf, buf + sizeof(buf), v).ptr); return; }
    case packed::DType::U32: { uint32_t v; std::memcpy(&v, data + i * 4, 4); s.append(buf, std::to_chars(buf, buf + sizeof(buf), v).ptr); return; }
    case packed::DType::F32:
    case packed::DType::F64:
    {
        double d;
        if (t == packed::DType::F32)
        {
            float f;
            uint32_t bits;
            std::memcpy(&f, data + i * 4, 4);
            std::memcpy(&bits, &f, 4);
            // Text is read back as a double and then rounded to float. Checked over every float, that rounds twice to
            // a different float for this one value (and its negative) only, so it is written with double digits.
            if (f != 0.0f && (bits & 0x7FFFFFFFu) != 0x15AE43FDu)
            {
                s.append(buf, std::to_chars(buf, buf + sizeof(buf), f).ptr);
                return;
            }
            d = f;
        }
        else
            std::memcpy(&d, data + i * 8, 8);
        if (d == 0.0)
            s.push_back('0');
        else
            s.append(buf, std::to_chars(buf, buf + sizeof(buf), d).ptr);
        return;
    }
    }
}

/**
 * A packed array written exactly as writeValue writes the plain array it stands for, without building that array:
 * rows of `width` for nested ones, rows of 3 for long flat numbers, one per line for booleans. Big ones format in
 * parallel blocks.
 */
void writePacked(Text& out, const std::string& indent, const std::string& key, const json& v)
{
    const packed::DType t = packed::dtype(v);
    const uint8_t* data = v.get_binary().data();
    const size_t n = packed::count(v);
    const size_t width = size_t(packed::width(v));
    const bool bools = t == packed::DType::Bool;
    if (width == 1 && n <= 16 && !bools)
    {
        out << indent << key;
        for (size_t i = 0; i < n; ++i)
        {
            out << ' ';
            appendPacked(out.s, t, data, i);
        }
        out << '\n';
        return;
    }
    const size_t perRow = bools ? 1 : width > 1 ? width : 3;
    out << indent << key << (width == 1 && !bools ? " [/3\n" : " [\n");
    const size_t rows = (n + perRow - 1) / perRow;
    constexpr size_t kBlock = 65536;
    std::vector<std::string> blocks((rows + kBlock - 1) / kBlock);
    tbb::parallel_for(size_t(0), blocks.size(), [&](size_t b) {
        std::string& s = blocks[b];
        for (size_t r = b * kBlock; r < std::min(rows, (b + 1) * kBlock); ++r)
        {
            s.append(indent).append("  ");
            for (size_t i = r * perRow; i < std::min(n, (r + 1) * perRow); ++i)
            {
                if (i > r * perRow)
                    s.push_back(' ');
                appendPacked(s, t, data, i);
            }
            s.push_back('\n');
        }
    });
    for (std::string& s : blocks)
        out << s;
    out << indent << "]\n";
}

bool hasPacked(const json& v)
{
    if (packed::is(v))
        return true;
    if (v.is_array() || v.is_object())
        for (const json& x : v)
            if (hasPacked(x))
                return true;
    return false;
}

/**
 * v.dump(), except packed arrays come out as the plain arrays they stand for (rows of their width), formatted in
 * parallel blocks: a mesh is one such line, millions of numbers long.
 */
void appendJson(std::string& s, const json& v)
{
    if (packed::is(v))
    {
        const packed::DType t = packed::dtype(v);
        const uint8_t* data = v.get_binary().data();
        const size_t n = packed::count(v), width = size_t(packed::width(v));
        const size_t rows = width > 1 ? (n + width - 1) / width : n;
        constexpr size_t kBlock = 65536;
        std::vector<std::string> blocks((rows + kBlock - 1) / kBlock);
        tbb::parallel_for(size_t(0), blocks.size(), [&](size_t b) {
            std::string& out = blocks[b];
            for (size_t r = b * kBlock; r < std::min(rows, (b + 1) * kBlock); ++r)
            {
                if (r)
                    out.push_back(',');
                if (width == 1)
                {
                    appendPacked(out, t, data, r);
                    continue;
                }
                out.push_back('[');
                for (size_t i = r * width; i < std::min(n, (r + 1) * width); ++i)
                {
                    if (i > r * width)
                        out.push_back(',');
                    appendPacked(out, t, data, i);
                }
                out.push_back(']');
            }
        });
        s.push_back('[');
        for (const std::string& b : blocks)
            s.append(b);
        s.push_back(']');
        return;
    }
    if (!hasPacked(v))
    {
        s.append(v.dump());
        return;
    }
    if (v.is_array())
    {
        s.push_back('[');
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i)
                s.push_back(',');
            appendJson(s, v[i]);
        }
        s.push_back(']');
        return;
    }
    s.push_back('{');
    bool first = true;
    for (auto& [k, x] : v.items())
    {
        if (!first)
            s.push_back(',');
        first = false;
        s.append(json(k).dump()).push_back(':');
        appendJson(s, x);
    }
    s.push_back('}');
}

void writeValue(Text& out, const std::string& indent, const std::string& key, const json& v)
{
    if (packed::is(v))
        return writePacked(out, indent, key, v);
    if (v.is_object() && hasPacked(v))
    {
        out << indent << key << " json ";
        appendJson(out.s, v);
        out << '\n';
        return;
    }
    if (isFlatNumbers(v) && v.size() <= 16 && !v.empty())
    {
        out << indent << key;
        for (const json& x : v)
        {
            out << ' ';
            appendScalar(out, x);
        }
        out << '\n';
        return;
    }
    if (isFlatNumbers(v) && !v.empty())
    {
        // Long flat arrays wrap at 3 per line, which is one vector per line for positions.
        out << indent << key << " [/3\n";
        for (size_t i = 0; i < v.size(); i += 3)
        {
            out << indent << "  ";
            for (size_t k = i; k < std::min(v.size(), i + 3); ++k)
            {
                if (k > i)
                    out << ' ';
                appendScalar(out, v[k]);
            }
            out << '\n';
        }
        out << indent << "]\n";
        return;
    }
    if (v.is_array() && !v.empty())
    {
        bool rows = true;
        for (const json& x : v)
            rows = rows && (x.is_number() || x.is_string() || x.is_boolean() || (isFlatNumbers(x) && !x.empty() && x.size() <= 16));
        if (rows)
        {
            out << indent << key << " [\n";
            for (const json& x : v)
            {
                out << indent << "  ";
                if (x.is_array())
                    for (size_t k = 0; k < x.size(); ++k)
                    {
                        if (k)
                            out << ' ';
                        appendScalar(out, x[k]);
                    }
                else
                    appendScalar(out, x);
                out << '\n';
            }
            out << indent << "]\n";
            return;
        }
    }
    out << indent << key << ' ' << scalar(v) << '\n';
}

std::vector<std::string> tokenize(std::string_view s)
{
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size())
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
            ++i;
        if (i >= s.size())
            break;
        size_t start = i;
        if (s[i] == '"')
        {
            ++i;
            while (i < s.size() && s[i] != '"')
                i += s[i] == '\\' ? 2 : 1;
            ++i;
        }
        else
            while (i < s.size() && s[i] != ' ' && s[i] != '\t')
                ++i;
        out.emplace_back(s.substr(start, std::min(i, s.size()) - start));
    }
    return out;
}

json parseScalar(const std::string& t)
{
    if (t == "true")
        return true;
    if (t == "false")
        return false;
    if (t == "null")
        return nullptr;
    if (!t.empty() && t[0] == '"')
        return json::parse(t);
    // Integers stay integers so enums and ints round-trip with their type.
    bool integral = t.find_first_of(".eEn") == std::string::npos;
    if (integral)
    {
        int64_t iv = 0;
        auto r = std::from_chars(t.data(), t.data() + t.size(), iv);
        if (r.ec == std::errc() && r.ptr == t.data() + t.size())
            return iv;
    }
    double d = 0.0;
    auto r = lila::fromChars(t.data(), t.data() + t.size(), d);
    if (r.ec != std::errc() || r.ptr != t.data() + t.size())
        throw std::runtime_error("can't read value '" + t + "'");
    return d;
}

/** Value from the tokens after a key on one line. */
/**
 * Parses a "json" line the way json::parse does, except long arrays of numbers (flat, or rows of equal width) become
 * packed arrays straight away instead of a json node per number. Meshes are such lines, millions of numbers long;
 * packed arrays anywhere but inside a mesh are unpacked again afterwards, since only meshFromJson reads them.
 */
class FastJson
{
public:
    explicit FastJson(std::string_view text) : s(text) {}

    json parse()
    {
        json v = value();
        skip();
        if (i != s.size())
            throw std::runtime_error("unexpected text after the json value");
        unpackOutsideMeshes(v, false);
        return v;
    }

private:
    static constexpr size_t kPackFrom = 64;
    std::string_view s;
    size_t i = 0;

    void skip()
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n'))
            ++i;
    }

    char peek()
    {
        skip();
        return i < s.size() ? s[i] : '\0';
    }

    void expect(char c)
    {
        if (peek() != c)
            throw std::runtime_error(std::string("expected '") + c + "' in json");
        ++i;
    }

    json value()
    {
        switch (peek())
        {
        case '{': return object();
        case '[': return array();
        case '"': return string();
        case 't': return literal("true", true);
        case 'f': return literal("false", false);
        case 'n': return literal("null", nullptr);
        default:
        {
            double d = 0.0;
            int64_t n = 0;
            bool integral = false;
            if (!number(d, n, integral))
                throw std::runtime_error("bad value in json");
            return integral ? json(n) : json(d);
        }
        }
    }

    json literal(std::string_view word, json result)
    {
        if (s.substr(i, word.size()) != word)
            throw std::runtime_error("bad literal in json");
        i += word.size();
        return result;
    }

    json string()
    {
        const size_t start = i++;
        while (i < s.size() && s[i] != '"')
            i += s[i] == '\\' ? 2 : 1;
        ++i;
        return json::parse(s.substr(start, i - start));
    }

    /** A number at i, as an integer when it is one. */
    bool number(double& d, int64_t& n, bool& integral)
    {
        skip();
        size_t end = i;
        while (end < s.size() && numeric(s[end]))
            ++end;
        if (end == i)
            return false;
        const std::string_view t = s.substr(i, end - i);
        integral = t.find_first_of(".eE") == std::string_view::npos;
        if (integral && std::from_chars(t.data(), t.data() + t.size(), n).ec == std::errc())
        {
            d = double(n);
            i = end;
            return true;
        }
        integral = false;
        if (lila::fromChars(t.data(), t.data() + t.size(), d).ec != std::errc())
            return false;
        i = end;
        return true;
    }

    json object()
    {
        expect('{');
        json obj = json::object();
        if (peek() == '}')
        {
            ++i;
            return obj;
        }
        for (;;)
        {
            if (peek() != '"')
                throw std::runtime_error("expected a key in json");
            std::string key = string().get<std::string>();
            expect(':');
            obj[key] = value();
            if (peek() == ',')
            {
                ++i;
                continue;
            }
            expect('}');
            return obj;
        }
    }

    json array()
    {
        const size_t start = i;
        if (json packedArray; numbers(packedArray))
            return packedArray;
        i = start;
        expect('[');
        json arr = json::array();
        if (peek() == ']')
        {
            ++i;
            return arr;
        }
        for (;;)
        {
            arr.push_back(value());
            if (peek() == ',')
            {
                ++i;
                continue;
            }
            expect(']');
            return arr;
        }
    }

    static bool numeric(char c) { return (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'; }

    /** Numbers of one chunk of an array's text, separated by commas, with rows bracketed when nested. */
    struct Piece
    {
        std::vector<double> values;
        bool allIntegral = true, nonNegative = true, ok = true;
    };

    static void parsePiece(std::string_view text, bool nested, int width, Piece& out)
    {
        size_t k = 0, inRow = 0;
        bool inside = false;
        while (k < text.size())
        {
            const char c = text[k];
            if (c == ',' || c == ' ' || c == '\t' || c == '\r' || c == '\n')
            {
                ++k;
                continue;
            }
            if (c == '[' || c == ']')
            {
                if (!nested || inside != (c == ']') || (c == ']' && int(inRow) != width))
                {
                    out.ok = false;
                    return;
                }
                inside = c == '[';
                inRow = 0;
                ++k;
                continue;
            }
            if (nested && !inside)
            {
                out.ok = false;
                return;
            }
            size_t e = k;
            while (e < text.size() && numeric(text[e]))
                ++e;
            const std::string_view tok = text.substr(k, e - k);
            if (tok.empty())
            {
                out.ok = false;
                return;
            }
            double d;
            if (tok.find_first_of(".eE") == std::string_view::npos)
            {
                int64_t n;
                if (std::from_chars(tok.data(), tok.data() + tok.size(), n).ec != std::errc())
                {
                    out.ok = false;
                    return;
                }
                d = double(n);
                out.allIntegral = out.allIntegral && n <= int64_t(UINT32_MAX);
            }
            else
            {
                if (parseDouble(tok.data(), tok.data() + tok.size(), d).ec != std::errc())
                {
                    out.ok = false;
                    return;
                }
                out.allIntegral = false;
            }
            out.nonNegative = out.nonNegative && d >= 0.0;
            out.values.push_back(d);
            ++inRow;
            k = e;
        }
    }

    /**
     * A long array of numbers, flat or rows of one width, as a packed array; false (i unspecified) otherwise. The
     * array's text is found first, then parsed in chunks split at commas between values (or rows) in parallel.
     */
    bool numbers(json& out)
    {
        expect('[');
        const size_t open = i;
        const bool nested = peek() == '[';
        // The extent: the first ']' (flat) or "]]" (rows) closes it. Only numbers, commas, spaces and one level of
        // brackets may appear inside, which parsePiece checks; anything else falls back to the general parser.
        int width = 0;
        if (nested)
        {
            const size_t rowEnd = s.find(']', open);
            if (rowEnd == std::string_view::npos)
                return false;
            const std::string_view row = s.substr(open + 1, rowEnd - open - 1);
            if (row.find_first_of("[]{}\"") != std::string_view::npos)
                return false;
            width = int(std::count(row.begin(), row.end(), ',')) + 1;
        }
        const size_t end = nested ? s.find("]]", open) + 1 : s.find(']', open);
        if (end == std::string_view::npos || end == 0 || end >= s.size())
            return false;
        if (nested && width < 2)
            return false;
        const std::string_view body = s.substr(open, end - open);
        // Split points: a comma at depth 0, after a value (flat) or after a row (nested).
        constexpr size_t kChunk = size_t(1) << 22;
        std::vector<size_t> cuts{0};
        for (size_t at = kChunk; at < body.size();)
        {
            size_t c = body.find(nested ? "]," : ",", at);
            if (c == std::string_view::npos)
                break;
            c += nested ? 2 : 1;
            cuts.push_back(c);
            at = c + kChunk;
        }
        cuts.push_back(body.size());
        std::vector<Piece> pieces(cuts.size() - 1);
        tbb::parallel_for(size_t(0), pieces.size(), [&](size_t k) { parsePiece(body.substr(cuts[k], cuts[k + 1] - cuts[k]), nested, width, pieces[k]); });
        size_t count = 0;
        bool allIntegral = true, nonNegative = true;
        for (const Piece& pc : pieces)
        {
            if (!pc.ok)
                return false;
            count += pc.values.size();
            allIntegral = allIntegral && pc.allIntegral;
            nonNegative = nonNegative && pc.nonNegative;
        }
        if (count < kPackFrom || (nested && count % size_t(width) != 0))
            return false;
        i = end + 1;
        const int w = nested ? width : 1;
        if (allIntegral && nonNegative)
        {
            uint32_t* dst = nullptr;
            out = packed::alloc(count, dst, w);
            size_t at = 0;
            for (const Piece& pc : pieces)
                for (double d : pc.values)
                    dst[at++] = uint32_t(d);
        }
        else
        {
            double* dst = nullptr;
            out = packed::alloc(count, dst, w);
            size_t at = 0;
            for (const Piece& pc : pieces)
            {
                std::memcpy(dst + at, pc.values.data(), pc.values.size() * sizeof(double));
                at += pc.values.size();
            }
        }
        return true;
    }

    static void unpackOutsideMeshes(json& v, bool inMesh)
    {
        if (packed::is(v))
        {
            if (!inMesh)
                v = packed::unpack(v);
            return;
        }
        if (v.is_object() && v.contains("faceSizes") && v.contains("points"))
            inMesh = true;
        if (v.is_array() || v.is_object())
            for (json& x : v)
                unpackOutsideMeshes(x, inMesh);
    }
};

json parseInline(std::string_view rest)
{
    std::string_view trimmed = rest;
    while (!trimmed.empty() && trimmed.front() == ' ')
        trimmed.remove_prefix(1);
    if (trimmed.rfind("json ", 0) == 0)
        return FastJson(trimmed.substr(5)).parse();
    auto tokens = tokenize(trimmed);
    if (tokens.empty())
        return nullptr;
    if (tokens.size() == 1)
        return parseScalar(tokens[0]);
    json arr = json::array();
    for (const std::string& t : tokens)
        arr.push_back(parseScalar(t));
    return arr;
}

std::string_view trimRight(std::string_view s)
{
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
        s.remove_suffix(1);
    return s;
}
} // namespace

SceneFormat detectFormat(std::string_view bytes)
{
    if (bytes.size() >= 8 && std::memcmp(bytes.data(), kMagic, 8) == 0)
        return SceneFormat::Binary;
    size_t i = 0;
    while (i < bytes.size() && std::isspace(static_cast<unsigned char>(bytes[i])))
        ++i;
    if (i < bytes.size() && bytes[i] == '{')
        return SceneFormat::Json;
    return SceneFormat::Ascii;
}

void encodeBinary(json header, const std::function<void(std::string_view)>& write)
{
    LILA_ZONE();

    std::vector<Blob> blobs;
    extractBlobs(header, blobs);

    std::string text = header.dump();
    header = json();
    std::vector<Chunk> chunks(1);
    std::memcpy(chunks[0].fourcc, "JSON", 4);
    chunks[0].raw = text;
    for (const Blob& b : blobs)
    {
        const std::string_view bytes(reinterpret_cast<const char*>(b.bytes.data()), b.bytes.size());
        for (size_t at = 0; at < bytes.size(); at += kPartBytes)
        {
            Chunk& c = chunks.emplace_back();
            std::memcpy(c.fourcc, at == 0 ? "BLOB" : "BLB+", 4);
            c.raw = bytes.substr(at, kPartBytes);
        }
    }

    // Chunks compress in parallel, roughly in file order, while this thread writes each one as soon as it is ready.
    std::vector<std::atomic<bool>> done(chunks.size());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex errorLock;
    std::thread compressor([&] {
        tbb::parallel_for(size_t(0), chunks.size(), size_t(1), [&](size_t) {
            const size_t i = next.fetch_add(1);
            try
            {
                if (!failed)
                    compress(chunks[i]);
            }
            catch (...)
            {
                std::scoped_lock lock(errorLock);
                if (!error)
                    error = std::current_exception();
                failed = true;
            }
            done[i] = true;
            done[i].notify_one();
        });
    });
    try
    {
        std::string head(kMagic, 8);
        put<uint32_t>(head, kBinaryVersion);
        put<uint32_t>(head, uint32_t(chunks.size()));
        write(head);
        for (size_t i = 0; i < chunks.size() && !failed; ++i)
        {
            done[i].wait(false);
            if (failed)
                break;
            Chunk& c = chunks[i];
            head.clear();
            head.append(c.fourcc, 4);
            put<uint32_t>(head, kFlagZstd);
            put<uint32_t>(head, c.crc);
            put<uint64_t>(head, c.raw.size());
            put<uint64_t>(head, c.stored.size());
            write(head);
            write(c.stored);
        }
    }
    catch (...)
    {
        failed = true;
        compressor.join();
        throw;
    }
    compressor.join();
    if (error)
        std::rethrow_exception(error);
    // Handing the big buffers to another thread keeps their release (hundreds of MB on a heavy scene) off the save.
    std::thread([blobs = std::move(blobs), chunks = std::move(chunks)] {}).detach();
}

std::string encodeBinary(json header)
{
    std::string out;
    encodeBinary(std::move(header), [&](std::string_view piece) { out += piece; });
    return out;
}

namespace
{
json decodeBinaryImpl(std::string_view in, std::vector<std::string>* warnings, bool headerOnly)
{

    if (in.size() < 16 || std::memcmp(in.data(), kMagic, 8) != 0)
        throw std::runtime_error("not a binary Lila scene");
    size_t at = 8;
    uint32_t version = get<uint32_t>(in, at);
    if (version > kBinaryVersion)
        throw std::runtime_error("scene was saved by a newer Lila (binary format " + std::to_string(version) + ")");
    uint32_t count = get<uint32_t>(in, at);

    struct Raw
    {
        std::string fourcc;
        uint32_t flags, crc;
        uint64_t rawSize;
        std::string_view stored;
    };
    std::vector<Raw> raws;
    for (uint32_t i = 0; i < count; ++i)
    {
        Raw r;
        if (at + 4 > in.size())
        {
            if (warnings)
                warnings->push_back("file ends early; later data blocks are lost");
            break;
        }
        r.fourcc.assign(in.data() + at, 4);
        at += 4;
        r.flags = get<uint32_t>(in, at);
        r.crc = get<uint32_t>(in, at);
        r.rawSize = get<uint64_t>(in, at);
        uint64_t storedSize = get<uint64_t>(in, at);
        if (at + storedSize > in.size())
        {
            if (warnings)
                warnings->push_back("file ends early; later data blocks are lost");
            break;
        }
        r.stored = in.substr(at, size_t(storedSize));
        at += size_t(storedSize);
        raws.push_back(r);
    }

    // A blob is its BLOB chunk plus the BLB+ chunks right after it, decompressed straight into one buffer per blob.
    struct Part
    {
        size_t blob = SIZE_MAX, offset = 0;
    };
    std::vector<Part> parts(raws.size());
    std::vector<size_t> blobSizes;
    for (size_t i = 1; i < raws.size() && !headerOnly; ++i)
    {
        if (raws[i].fourcc == "BLOB")
            blobSizes.push_back(0);
        else if (raws[i].fourcc != "BLB+" || blobSizes.empty() || parts[i - 1].blob != blobSizes.size() - 1)
            continue;
        parts[i] = {blobSizes.size() - 1, blobSizes.back()};
        blobSizes.back() += size_t(raws[i].rawSize);
    }
    std::vector<BlobBytes> blobs(blobSizes.size());
    for (size_t b = 0; b < blobs.size(); ++b)
        blobs[b].resize(blobSizes[b]);
    std::string header;
    if (!raws.empty())
        header.resize(size_t(raws[0].rawSize));
    std::vector<char> ok(raws.size(), 0);
    tbb::parallel_for(size_t(0), headerOnly ? std::min<size_t>(raws.size(), 1) : raws.size(), [&](size_t i) {
        const Raw& r = raws[i];
        char* dst = i == 0 ? header.data() : parts[i].blob != SIZE_MAX ? blobs[parts[i].blob].data() + parts[i].offset : nullptr;
        if (!dst)
            return;
        if (r.flags & kFlagZstd)
        {
            size_t n = ZSTD_decompressDCtx(decompressContext(), dst, size_t(r.rawSize), r.stored.data(), r.stored.size());
            if (ZSTD_isError(n) || n != r.rawSize)
                return;
        }
        else if (r.stored.size() == r.rawSize)
            std::memcpy(dst, r.stored.data(), r.stored.size());
        else
            return;
        ok[i] = crc32(dst, size_t(r.rawSize)) == r.crc;
    });

    if (raws.empty() || raws[0].fourcc != "JSON" || !ok[0])
        throw std::runtime_error("the scene header is damaged; try the .lilaa copy or an autosave");
    json doc = json::parse(header);
    if (headerOnly)
        return doc;
    // A damaged part loses its whole array.
    for (size_t i = 1; i < raws.size(); ++i)
        if (parts[i].blob != SIZE_MAX && !ok[i])
            BlobBytes().swap(blobs[parts[i].blob]);
    restoreBlobs(doc, blobs, warnings);
    // Gigabytes take long enough to free that it is done off this thread.
    std::thread([b = std::move(blobs)] {}).detach();
    return doc;
}
} // namespace

json decodeBinary(std::string_view in, std::vector<std::string>* warnings)
{
    LILA_ZONE();

    return decodeBinaryImpl(in, warnings, false);
}

json decodeBinaryHeader(std::string_view in)
{
    LILA_ZONE();

    return decodeBinaryImpl(in, nullptr, true);
}

// ------------------------------------------------------------------- ASCII

std::string encodeAscii(const json& doc)
{
    LILA_ZONE();

    Text out;
    out << "lila " << kAsciiVersion << " ascii\n";
    for (const char* key : {"upAxis", "units"})
        if (doc.contains(key))
            out << key << ' ' << doc[key].get<std::string>() << '\n';
    if (doc.contains("time"))
        out << "time " << scalar(doc["time"]) << '\n';
    // Anything else at the top level (future sections) as one JSON line each, so nothing is ever dropped.
    for (auto& [key, value] : doc.items())
        if (key != "lila" && key != "upAxis" && key != "units" && key != "time" && key != "nodes" && key != "connections" &&
            key != "selection" && key != "scripts")
            out << "doc " << key << ' ' << value.dump() << '\n';

    static const json kNone = json::array();
    auto member = [&](const char* key) -> const json& {
        auto it = doc.find(key);
        return it == doc.end() ? kNone : *it;
    };
    for (const json& n : member("nodes"))
    {
        out << "\nnode " << n.value("type", "") << ' ' << n.value("name", "") << '\n';
        if (n.contains("parent"))
            out << "  parent " << n["parent"].get<std::string>() << '\n';
        if (n.contains("attrs"))
            for (auto& [attr, value] : n["attrs"].items())
                writeValue(out, "  ", attr, value);
        for (auto& [key, value] : n.items())
            if (key != "name" && key != "type" && key != "parent" && key != "attrs")
                out << "  @" << key << ' ' << value.dump() << '\n';
    }

    const json& conns = member("connections");
    if (!conns.empty())
        out << '\n';
    for (const json& c : conns)
        out << "connect " << c[0].get<std::string>() << ' ' << c[1].get<std::string>() << '\n';

    const json& sel = member("selection");
    if (!sel.empty())
    {
        out << "\nselect";
        for (const json& s : sel)
            out << ' ' << s.get<std::string>();
        out << '\n';
    }

    // Script source verbatim, one "|" line per source line, so scripts diff line by line too.
    for (const json& script : member("scripts"))
    {
        out << "\nscript " << json(script.value("name", "")).dump() << (script.value("runOnOpen", false) ? " runOnOpen" : "") << '\n';
        std::string_view src = script.contains("text") && script["text"].is_string() ? std::string_view(script["text"].get_ref<const std::string&>()) : std::string_view();
        for (size_t start = 0;;)
        {
            size_t end = src.find('\n', start);
            out << "  |" << src.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start) << '\n';
            if (end == std::string_view::npos)
                break;
            start = end + 1;
        }
    }
    return std::move(out.s);
}

json decodeAscii(std::string_view text, std::vector<std::string>* warnings)
{
    LILA_ZONE();

    json doc = {{"lila", 1}, {"nodes", json::array()}, {"connections", json::array()}, {"selection", json::array()}};
    json* node = nullptr;
    auto warn = [&](size_t line, const std::string& msg) {
        if (warnings)
            warnings->push_back("line " + std::to_string(line) + ": " + msg);
    };

    std::vector<std::string_view> lines;
    std::vector<std::string_view> rawLines; // script lines keep their trailing whitespace
    for (size_t start = 0; start <= text.size();)
    {
        size_t end = text.find('\n', start);
        if (end == std::string_view::npos)
            end = text.size();
        std::string_view raw = text.substr(start, end - start);
        if (!raw.empty() && raw.back() == '\r')
            raw.remove_suffix(1);
        rawLines.push_back(raw);
        lines.push_back(trimRight(raw));
        start = end + 1;
    }
    json* script = nullptr;
    bool scriptHasLine = false;

    bool sawHeader = false;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        std::string_view line = lines[i];
        if (line.empty() || line[0] == '#')
            continue;
        bool indented = line[0] == ' ' || line[0] == '\t';
        std::string_view body = line;
        while (!body.empty() && (body.front() == ' ' || body.front() == '\t'))
            body.remove_prefix(1);
        size_t sp = body.find(' ');
        std::string key(body.substr(0, sp));
        std::string_view rest = sp == std::string_view::npos ? std::string_view() : body.substr(sp + 1);

        try
        {
            if (indented && script)
            {
                std::string_view raw = rawLines[i];
                size_t bar = raw.find('|');
                if (bar == std::string_view::npos || raw.find_first_not_of(" \t") != bar)
                {
                    warn(i + 1, "script line without '|', ignored");
                    continue;
                }
                std::string& src = (*script)["text"].get_ref<std::string&>();
                if (scriptHasLine)
                    src += '\n';
                src += raw.substr(bar + 1);
                scriptHasLine = true;
                continue;
            }
            if (indented)
            {
                if (!node)
                {
                    warn(i + 1, "attribute outside a node, ignored");
                    continue;
                }
                if (key == "parent")
                {
                    (*node)["parent"] = std::string(rest);
                    continue;
                }
                if (!key.empty() && key[0] == '@')
                {
                    (*node)[key.substr(1)] = json::parse(rest);
                    continue;
                }
                if (rest == "[" || rest == "[/3")
                {
                    bool flat = rest == "[/3";
                    json arr = json::array();
                    size_t j = i + 1;
                    for (; j < lines.size(); ++j)
                    {
                        std::string_view row = lines[j];
                        while (!row.empty() && (row.front() == ' ' || row.front() == '\t'))
                            row.remove_prefix(1);
                        if (row == "]")
                            break;
                        try
                        {
                            json v = parseInline(row);
                            if (flat && v.is_array())
                                for (auto& x : v)
                                    arr.push_back(x);
                            else
                                arr.push_back(v);
                        }
                        catch (const std::exception& e)
                        {
                            warn(j + 1, e.what());
                        }
                    }
                    if (j >= lines.size())
                        warn(i + 1, "array '" + key + "' never closed");
                    (*node)["attrs"][key] = std::move(arr);
                    i = j;
                    continue;
                }
                (*node)["attrs"][key] = parseInline(rest);
                continue;
            }

            node = nullptr;
            script = nullptr;
            if (key == "lila")
            {
                sawHeader = true;
                continue;
            }
            if (key == "upAxis" || key == "units")
                doc[key] = std::string(rest);
            else if (key == "time")
                doc["time"] = parseScalar(std::string(rest));
            else if (key == "doc")
            {
                size_t s2 = rest.find(' ');
                doc[std::string(rest.substr(0, s2))] = json::parse(rest.substr(s2 + 1));
            }
            else if (key == "node")
            {
                auto t = tokenize(rest);
                if (t.size() != 2)
                {
                    warn(i + 1, "expected 'node <type> <name>'");
                    continue;
                }
                doc["nodes"].push_back({{"type", t[0]}, {"name", t[1]}});
                node = &doc["nodes"].back();
            }
            else if (key == "connect")
            {
                auto t = tokenize(rest);
                if (t.size() == 2)
                    doc["connections"].push_back({t[0], t[1]});
                else
                    warn(i + 1, "expected 'connect <src> <dst>'");
            }
            else if (key == "select")
                for (const std::string& t : tokenize(rest))
                    doc["selection"].push_back(t);
            else if (key == "script")
            {
                auto t = tokenize(rest);
                if (t.empty() || t[0].empty() || t[0][0] != '"')
                {
                    warn(i + 1, "expected 'script \"<name>\"'");
                    continue;
                }
                json entry = {{"name", json::parse(t[0])}, {"text", ""}};
                if (t.size() > 1 && t[1] == "runOnOpen")
                    entry["runOnOpen"] = true;
                doc["scripts"].push_back(std::move(entry));
                script = &doc["scripts"].back();
                scriptHasLine = false;
            }
            else
                warn(i + 1, "unknown line '" + key + "', ignored");
        }
        catch (const std::exception& e)
        {
            warn(i + 1, e.what());
        }
    }
    if (!sawHeader)
        throw std::runtime_error("not a Lila ASCII scene (no 'lila' header line)");
    return doc;
}
} // namespace lila::io
