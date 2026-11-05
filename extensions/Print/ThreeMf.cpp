#include "ThreeMf.h"

#include <lila/Decimal.h>

#include <lila/MeshOps.h>

#include <miniz.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <map>
#include <optional>
#include <span>

namespace print::threemf
{
using lila::geo::Index;
using lila::geo::Mesh;

namespace
{
constexpr const char* kModelPath = "3D/3dmodel.model";

const char* kContentTypes = R"(<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>
)";

const char* kRels = R"(<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>
)";

float linearToSrgb(float c)
{
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

float srgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

int byteOf(float c) { return int(std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f)); }

std::string escape(std::string_view s)
{
    std::string out;
    for (char c : s)
        switch (c)
        {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        default: out += c;
        }
    return out;
}

std::string unescape(std::string_view s)
{
    std::string out;
    for (size_t i = 0; i < s.size(); ++i)
    {
        if (s[i] != '&')
        {
            out += s[i];
            continue;
        }
        const size_t end = s.find(';', i);
        if (end == std::string_view::npos)
        {
            out += s.substr(i);
            break;
        }
        const std::string_view ent = s.substr(i + 1, end - i - 1);
        out += ent == "amp" ? "&" : ent == "lt" ? "<" : ent == "gt" ? ">" : ent == "quot" ? "\"" : ent == "apos" ? "'" : "";
        i = end;
    }
    return out;
}

std::string modelXml(const std::vector<Object>& objects, const WriteOptions& opts)
{
    std::string x;
    x.reserve(1 << 16);
    x += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    x += std::format("<model unit=\"{}\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" "
                     "xmlns:m=\"http://schemas.microsoft.com/3dmanufacturing/material/2015/02\">\n",
                     opts.unit);
    x += " <metadata name=\"Application\">Lila</metadata>\n <resources>\n";
    int nextId = 1;
    std::vector<int> objectIds;
    for (const Object& o : objects)
    {
        const Mesh& m = o.mesh;
        std::vector<Index> slot(m.pointCapacity(), lila::geo::kInvalid);
        std::vector<Index> points;
        for (Index p = 0; p < m.pointCapacity(); ++p)
            if (m.pointValid(p))
            {
                slot[p] = Index(points.size());
                points.push_back(p);
            }

        int group = 0;
        const std::string set = opts.colors ? lila::geo::colors::current(m) : std::string();
        if (!set.empty())
        {
            group = nextId++;
            const std::vector<glm::vec4> colors = lila::geo::colors::pointColors(m, set);
            x += std::format("  <m:colorgroup id=\"{}\">\n", group);
            for (Index p : points)
            {
                const glm::vec4& c = colors[p];
                x += std::format("   <m:color color=\"#{:02X}{:02X}{:02X}{:02X}\"/>\n", byteOf(linearToSrgb(c.r)), byteOf(linearToSrgb(c.g)),
                                 byteOf(linearToSrgb(c.b)), byteOf(c.a));
            }
            x += "  </m:colorgroup>\n";
        }

        const int id = nextId++;
        objectIds.push_back(id);
        x += std::format("  <object id=\"{}\" type=\"model\" name=\"{}\"", id, escape(o.name));
        if (group)
            x += std::format(" pid=\"{}\" pindex=\"0\"", group);
        x += ">\n   <mesh>\n    <vertices>\n";
        for (Index p : points)
        {
            const glm::dvec3& v = m.position(p);
            x += std::format("     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>\n", v.x, v.y, v.z);
        }
        x += "    </vertices>\n    <triangles>\n";
        const lila::geo::Triangulation tris = lila::geo::triangulate(m);
        for (Index t = 0; t < tris.triangleCount(); ++t)
        {
            const Index a = slot[m.cornerPoint(tris.corners[t * 3])], b = slot[m.cornerPoint(tris.corners[t * 3 + 1])],
                        c = slot[m.cornerPoint(tris.corners[t * 3 + 2])];
            if (group)
                x += std::format("     <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" pid=\"{}\" p1=\"{}\" p2=\"{}\" p3=\"{}\"/>\n", a, b, c, group, a, b, c);
            else
                x += std::format("     <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n", a, b, c);
        }
        x += "    </triangles>\n   </mesh>\n  </object>\n";
    }
    x += " </resources>\n <build>\n";
    for (int id : objectIds)
        x += std::format("  <item objectid=\"{}\"/>\n", id);
    x += " </build>\n</model>\n";
    return x;
}

struct Tag
{
    std::string_view name; // without a namespace prefix
    std::map<std::string_view, std::string_view> attrs;
    bool closing = false;
    bool selfClosing = false;
};

/** The next tag at or after pos, skipping declarations and comments; nullopt at the end. */
std::optional<Tag> nextTag(std::string_view xml, size_t& pos)
{
    while (true)
    {
        const size_t lt = xml.find('<', pos);
        if (lt == std::string_view::npos)
            return std::nullopt;
        const size_t gt = xml.find('>', lt);
        if (gt == std::string_view::npos)
            return std::nullopt;
        pos = gt + 1;
        std::string_view body = xml.substr(lt + 1, gt - lt - 1);
        if (body.empty() || body[0] == '?' || body[0] == '!')
            continue;
        Tag t;
        if (body[0] == '/')
        {
            t.closing = true;
            body.remove_prefix(1);
        }
        if (!body.empty() && body.back() == '/')
        {
            t.selfClosing = true;
            body.remove_suffix(1);
        }
        size_t i = body.find_first_of(" \t\r\n");
        std::string_view name = body.substr(0, i);
        if (const size_t colon = name.find(':'); colon != std::string_view::npos)
            name = name.substr(colon + 1);
        t.name = name;
        while (i != std::string_view::npos && i < body.size())
        {
            i = body.find_first_not_of(" \t\r\n", i);
            if (i == std::string_view::npos)
                break;
            const size_t eq = body.find('=', i);
            if (eq == std::string_view::npos)
                break;
            std::string_view key = body.substr(i, eq - i);
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t'))
                key.remove_suffix(1);
            const size_t q0 = body.find_first_of("\"'", eq);
            if (q0 == std::string_view::npos)
                break;
            const size_t q1 = body.find(body[q0], q0 + 1);
            if (q1 == std::string_view::npos)
                break;
            t.attrs[key] = body.substr(q0 + 1, q1 - q0 - 1);
            i = q1 + 1;
        }
        return t;
    }
}

template <class T> T number(const Tag& t, std::string_view key, T fallback = T{})
{
    auto it = t.attrs.find(key);
    if (it == t.attrs.end())
        return fallback;
    std::string_view s = it->second;
    while (!s.empty() && s.front() == ' ')
        s.remove_prefix(1);
    T v{};
    const auto r = lila::fromChars(s.data(), s.data() + s.size(), v);
    return r.ec == std::errc() ? v : fallback;
}

glm::vec4 parseColor(std::string_view s)
{
    if (s.empty() || s[0] != '#')
        return glm::vec4(1.0f);
    auto hex = [&](size_t at) {
        int v = 0;
        std::from_chars(s.data() + at, s.data() + at + 2, v, 16);
        return float(v) / 255.0f;
    };
    const bool alpha = s.size() >= 9;
    if (s.size() < 7)
        return glm::vec4(1.0f);
    return {srgbToLinear(hex(1)), srgbToLinear(hex(3)), srgbToLinear(hex(5)), alpha ? hex(7) : 1.0f};
}

/** A 3MF transform: 12 numbers, row vectors (m00 m01 m02 m10 ... m30 m31 m32). */
glm::dmat4 parseTransform(std::string_view s)
{
    double v[12];
    int n = 0;
    const char* p = s.data();
    const char* end = s.data() + s.size();
    while (n < 12 && p < end)
    {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            ++p;
        const auto r = lila::fromChars(p, end, v[n]);
        if (r.ec != std::errc())
            break;
        p = r.ptr;
        ++n;
    }
    if (n != 12)
        return glm::dmat4(1.0);
    glm::dmat4 m(1.0);
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 3; ++col)
            m[row][col] = v[row * 3 + col];
    return m;
}
} // namespace

double unitToCm(std::string_view unit)
{
    if (unit == "micron")
        return 1e-4;
    if (unit == "millimeter")
        return 0.1;
    if (unit == "centimeter")
        return 1.0;
    if (unit == "inch")
        return 2.54;
    if (unit == "foot")
        return 30.48;
    if (unit == "meter")
        return 100.0;
    return 0.0;
}

std::string write(const std::vector<Object>& objects, const WriteOptions& opts)
{
    const std::string model = modelXml(objects, opts);
    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, model.size() / 4 + 1024))
        return {};
    bool ok = mz_zip_writer_add_mem(&zip, "[Content_Types].xml", kContentTypes, std::char_traits<char>::length(kContentTypes), MZ_DEFAULT_LEVEL) &&
              mz_zip_writer_add_mem(&zip, "_rels/.rels", kRels, std::char_traits<char>::length(kRels), MZ_DEFAULT_LEVEL) &&
              mz_zip_writer_add_mem(&zip, kModelPath, model.data(), model.size(), MZ_DEFAULT_LEVEL);
    void* buf = nullptr;
    size_t size = 0;
    ok = ok && mz_zip_writer_finalize_heap_archive(&zip, &buf, &size);
    std::string out = ok ? std::string(static_cast<const char*>(buf), size) : std::string();
    mz_zip_writer_end(&zip);
    return out;
}

Model read(std::string_view zipBytes)
{
    Model out;
    mz_zip_archive zip{};
    if (!mz_zip_reader_init_mem(&zip, zipBytes.data(), zipBytes.size(), 0))
    {
        out.error = "not a 3MF (zip) file";
        return out;
    }
    std::string xml;
    const mz_uint count = mz_zip_reader_get_num_files(&zip);
    for (mz_uint i = 0; i < count; ++i)
    {
        char name[512];
        mz_zip_reader_get_filename(&zip, i, name, sizeof(name));
        out.entries.push_back(name);
        std::string_view n(name);
        const bool model = n.size() > 6 && n.substr(n.size() - 6) == ".model" && (n.starts_with("3D/") || n.starts_with("/3D/"));
        if (model && xml.empty())
        {
            size_t size = 0;
            void* data = mz_zip_reader_extract_to_heap(&zip, i, &size, 0);
            if (data)
            {
                xml.assign(static_cast<const char*>(data), size);
                mz_free(data);
            }
        }
    }
    mz_zip_reader_end(&zip);
    if (xml.empty())
    {
        out.error = "no 3D model part in the package";
        return out;
    }

    struct Parsed
    {
        std::string name;
        Mesh mesh;
        std::vector<Index> points;
        int pid = 0, pindex = 0;
        std::vector<std::array<int, 4>> triColors; // pid, p1, p2, p3 per face
    };
    std::map<int, Parsed> objects;
    std::map<int, std::vector<glm::vec4>> groups;
    std::vector<std::pair<int, glm::dmat4>> items;
    Parsed* cur = nullptr;
    int curGroup = 0;
    size_t pos = 0;
    while (std::optional<Tag> t = nextTag(xml, pos))
    {
        if (t->name == "model" && !t->closing)
        {
            if (auto it = t->attrs.find("unit"); it != t->attrs.end())
                out.unit = std::string(it->second);
        }
        else if (t->name == "object")
        {
            if (t->closing)
                cur = nullptr;
            else
            {
                const int id = number<int>(*t, "id");
                cur = &objects[id];
                if (auto it = t->attrs.find("name"); it != t->attrs.end())
                    cur->name = unescape(it->second);
                cur->pid = number<int>(*t, "pid");
                cur->pindex = number<int>(*t, "pindex");
                if (t->selfClosing)
                    cur = nullptr;
            }
        }
        else if (t->name == "vertex" && cur)
            cur->points.push_back(cur->mesh.addPoint({number<double>(*t, "x"), number<double>(*t, "y"), number<double>(*t, "z")}));
        else if (t->name == "triangle" && cur)
        {
            const int v1 = number<int>(*t, "v1", -1), v2 = number<int>(*t, "v2", -1), v3 = number<int>(*t, "v3", -1);
            const int n = int(cur->points.size());
            if (v1 < 0 || v2 < 0 || v3 < 0 || v1 >= n || v2 >= n || v3 >= n || v1 == v2 || v2 == v3 || v1 == v3)
                continue;
            if (cur->mesh.addFace({cur->points[v1], cur->points[v2], cur->points[v3]}) == lila::geo::kInvalid)
                continue;
            const int pid = number<int>(*t, "pid", cur->pid);
            const int p1 = number<int>(*t, "p1", cur->pindex);
            cur->triColors.push_back({pid, p1, number<int>(*t, "p2", p1), number<int>(*t, "p3", p1)});
        }
        else if (t->name == "colorgroup")
            curGroup = t->closing ? 0 : number<int>(*t, "id");
        else if (t->name == "color" && curGroup)
        {
            auto it = t->attrs.find("color");
            groups[curGroup].push_back(parseColor(it != t->attrs.end() ? it->second : std::string_view()));
        }
        else if (t->name == "item" && !t->closing)
        {
            auto it = t->attrs.find("transform");
            items.emplace_back(number<int>(*t, "objectid"), it != t->attrs.end() ? parseTransform(it->second) : glm::dmat4(1.0));
        }
    }

    for (auto& [id, item] : items)
    {
        auto it = objects.find(id);
        if (it == objects.end())
            continue;
        Parsed& p = it->second;
        Object o{p.name, p.mesh};
        for (Index v = 0; v < o.mesh.pointCapacity(); ++v)
            if (o.mesh.pointValid(v))
                o.mesh.setPosition(v, glm::dvec3(item * glm::dvec4(o.mesh.position(v), 1.0)));
        bool colored = false;
        for (const auto& c : p.triColors)
            colored |= groups.contains(c[0]);
        if (colored)
        {
            std::span<glm::vec4> values = lila::geo::colors::add(o.mesh, "colorSet");
            lila::geo::colors::setCurrent(o.mesh, "colorSet");
            Index f = 0;
            for (const auto& c : p.triColors)
            {
                for (; f < o.mesh.faceCapacity() && !o.mesh.faceValid(f); ++f)
                    ;
                if (f >= o.mesh.faceCapacity())
                    break;
                auto g = groups.find(c[0]);
                const std::vector<Index> corners = o.mesh.faceCorners(f);
                for (int k = 0; k < 3 && k < int(corners.size()); ++k)
                    if (g != groups.end() && c[1 + k] >= 0 && c[1 + k] < int(g->second.size()))
                        values[corners[k]] = g->second[c[1 + k]];
                ++f;
            }
            out.colors = true;
        }
        out.objects.push_back(std::move(o));
    }
    if (out.objects.empty())
        out.error = "the model has no mesh objects to build";
    return out;
}
} // namespace print::threemf
