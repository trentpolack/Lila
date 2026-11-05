#include "gn/NodesCommon.h"

#include "core/Decimal.h"

#include "geo/ObjIO.h"
#include "geo/PlyIO.h"
#include "geo/StlIO.h"

#include <cctype>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace lila::gn::nodes::data
{
/** Geometry read once per path, reader and modification time. Throws ExecError naming the node when the file is missing. */
GeometryPtr cachedRead(const std::string& label, const std::string& path, const std::string& kind, const std::function<GeometryPtr()>& read)
{
    static std::mutex mutex;
    static std::map<std::string, std::pair<std::filesystem::file_time_type, GeometryPtr>> cache;
    if (path.empty())
        return makeGeometry({});
    std::error_code ec;
    const auto stamp = std::filesystem::last_write_time(path, ec);
    if (ec)
        throw ExecError(label + ": can't find " + path);
    const std::string key = kind + "|" + path;
    {
        std::lock_guard lock(mutex);
        auto it = cache.find(key);
        if (it != cache.end() && it->second.first == stamp)
            return it->second.second;
    }
    GeometryPtr g = read();
    std::lock_guard lock(mutex);
    if (cache.size() > 64)
        cache.clear();
    cache[key] = {stamp, g};
    return g;
}

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
    return s;
}

std::string trim(std::string_view s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a])))
        ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;
    return std::string(s.substr(a, b - a));
}

/** Rows as text cells, the first being the header. */
struct Table
{
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

/** RFC 4180: quoted cells may hold commas, newlines and doubled quotes. */
Table parseCsv(std::string_view text)
{
    std::vector<std::vector<std::string>> lines(1);
    std::string cell;
    bool quoted = false, any = false;
    auto endCell = [&] {
        lines.back().push_back(trim(cell));
        cell.clear();
    };
    for (size_t i = 0; i < text.size(); ++i)
    {
        char ch = text[i];
        if (quoted)
        {
            if (ch == '"' && i + 1 < text.size() && text[i + 1] == '"')
            {
                cell += '"';
                ++i;
            }
            else if (ch == '"')
                quoted = false;
            else
                cell += ch;
            continue;
        }
        if (ch == '"')
            quoted = any = true;
        else if (ch == ',')
        {
            endCell();
            any = true;
        }
        else if (ch == '\n' || ch == '\r')
        {
            if (ch == '\r' && i + 1 < text.size() && text[i + 1] == '\n')
                ++i;
            if (any || !cell.empty())
                endCell();
            if (!lines.back().empty())
                lines.emplace_back();
            any = false;
        }
        else
        {
            cell += ch;
            any = true;
        }
    }
    if (any || !cell.empty())
        endCell();
    if (lines.back().empty())
        lines.pop_back();
    Table t;
    if (lines.empty())
        return t;
    t.header = std::move(lines[0]);
    t.rows.assign(std::make_move_iterator(lines.begin() + 1), std::make_move_iterator(lines.end()));
    return t;
}

std::string cellText(const json& v)
{
    if (v.is_string())
        return v.get<std::string>();
    if (v.is_boolean())
        return v.get<bool>() ? "true" : "false";
    if (v.is_number())
        return v.dump();
    return {};
}

/** An array of objects, or an object holding one under "rows". Arrays of three numbers become name_x, name_y, name_z. */
Table parseJsonTable(std::string_view text)
{
    json j = json::parse(text, nullptr, false);
    if (j.is_discarded())
        throw ExecError("Read Data Table: the file isn't valid JSON");
    if (j.is_object() && j.contains("rows"))
        j = j["rows"];
    if (!j.is_array())
        throw ExecError("Read Data Table: JSON must be an array of objects, or an object with a rows array");
    Table t;
    std::map<std::string, size_t> column;
    auto col = [&](const std::string& name) {
        auto [it, added] = column.try_emplace(name, t.header.size());
        if (added)
            t.header.push_back(name);
        return it->second;
    };
    for (const json& row : j)
    {
        if (!row.is_object())
            continue;
        std::vector<std::pair<size_t, std::string>> cells;
        for (auto it = row.begin(); it != row.end(); ++it)
        {
            const json& v = it.value();
            if (v.is_array() && v.size() == 3 && v[0].is_number() && v[1].is_number() && v[2].is_number())
            {
                for (int a = 0; a < 3; ++a)
                    cells.push_back({col(it.key() + "_" + "xyz"[a]), v[a].dump()});
                continue;
            }
            if (v.is_string() || v.is_number() || v.is_boolean())
                cells.push_back({col(it.key()), cellText(v)});
        }
        std::vector<std::string> r;
        for (auto& [k, s] : cells)
        {
            if (r.size() <= k)
                r.resize(k + 1);
            r[k] = std::move(s);
        }
        t.rows.push_back(std::move(r));
    }
    for (auto& r : t.rows)
        r.resize(t.header.size());
    return t;
}

bool parseNumber(const std::string& s, double& out)
{
    if (s.empty())
        return false;
    const char* end = s.data() + s.size();
    auto [p, ec] = lila::fromChars(s.data(), end, out);
    return ec == std::errc() && p == end;
}

/** The vector a column belongs to ("pos" for pos_x or pos.x; "position" for bare x) and which component, or -1. */
int vectorPart(const std::string& name, std::string& base)
{
    std::string l = lower(name);
    if (l == "x" || l == "y" || l == "z")
    {
        base = "position";
        return l[0] - 'x';
    }
    if (l.size() > 2 && (l[l.size() - 2] == '_' || l[l.size() - 2] == '.') && l.back() >= 'x' && l.back() <= 'z')
    {
        base = name.substr(0, name.size() - 2);
        return l.back() - 'x';
    }
    return -1;
}

/** One point per row; columns become point attributes. Text columns become ints numbering their distinct values in order of first use. */
geo::Mesh tableToPoints(const Table& t)
{
    geo::Mesh m;
    const size_t n = t.rows.size();
    for (size_t r = 0; r < n; ++r)
        m.addPoint(D3(0.0));

    std::map<std::string, std::array<int, 3>> vectors;
    std::vector<uint8_t> inVector(t.header.size(), 0);
    for (size_t k = 0; k < t.header.size(); ++k)
    {
        std::string base;
        int part = vectorPart(t.header[k], base);
        if (part < 0)
            continue;
        auto [it, added] = vectors.try_emplace(base, std::array<int, 3>{-1, -1, -1});
        it->second[size_t(part)] = int(k);
    }
    auto number = [&](size_t r, int k) {
        double v = 0.0;
        if (k >= 0 && size_t(k) < t.rows[r].size())
            parseNumber(t.rows[r][size_t(k)], v);
        return v;
    };
    for (auto& [base, cols] : vectors)
    {
        if (cols[0] < 0 || cols[1] < 0 || cols[2] < 0)
            continue;
        for (int c : cols)
            inVector[size_t(c)] = 1;
        std::vector<D3> v(n);
        for (size_t r = 0; r < n; ++r)
            v[r] = {number(r, cols[0]), number(r, cols[1]), number(r, cols[2])};
        if (lower(base) == "position" || lower(base) == "p")
        {
            for (size_t r = 0; r < n; ++r)
                m.setPosition(Index(r), v[r]);
            continue;
        }
        writeAttribute(m, base, Domain::Point, Column::of(std::move(v)), T::Vector);
    }

    for (size_t k = 0; k < t.header.size(); ++k)
    {
        if (inVector[k] || t.header[k].empty())
            continue;
        bool numeric = true, integral = true, boolean = true;
        for (size_t r = 0; r < n; ++r)
        {
            const std::string& s = t.rows[r][k];
            if (s.empty())
                continue;
            double v;
            if (!parseNumber(s, v))
                numeric = false;
            else if (v != std::floor(v) || std::abs(v) > 9e15)
                integral = false;
            std::string l = lower(s);
            if (l != "true" && l != "false")
                boolean = false;
        }
        const std::string& name = t.header[k];
        if (boolean)
        {
            std::vector<uint8_t> v(n);
            for (size_t r = 0; r < n; ++r)
                v[r] = lower(t.rows[r][k]) == "true";
            writeAttribute(m, name, Domain::Point, Column::of(std::move(v)), T::Bool);
        }
        else if (numeric && integral)
        {
            std::vector<int64_t> v(n);
            for (size_t r = 0; r < n; ++r)
                v[r] = int64_t(number(r, int(k)));
            writeAttribute(m, name, Domain::Point, Column::of(std::move(v)), T::Int);
        }
        else if (numeric)
        {
            std::vector<double> v(n);
            for (size_t r = 0; r < n; ++r)
                v[r] = number(r, int(k));
            writeAttribute(m, name, Domain::Point, Column::of(std::move(v)), T::Float);
        }
        else
        {
            std::map<std::string, int64_t> ids;
            std::vector<int64_t> v(n);
            for (size_t r = 0; r < n; ++r)
                v[r] = ids.try_emplace(t.rows[r][k], int64_t(ids.size())).first->second;
            writeAttribute(m, name, Domain::Point, Column::of(std::move(v)), T::Int);
        }
    }
    return m;
}

std::string readText(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw ExecError("can't read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string extensionOf(const std::string& path) { return lower(std::filesystem::path(path).extension().string()); }

void readDataTable()
{
    registerDecl({
        .type = "ReadDataTable",
        .label = "Read Data Table",
        .category = "Input",
        .doc = "A point per row of a CSV file (header row first) or JSON array of objects. Columns x, y, z (or position_x...) place "
               "the points; other columns become point attributes: numbers, true/false, vectors from name_x/_y/_z or three-number "
               "arrays, and text as ints numbering its distinct values. Reread when the file changes.",
        .props = {prop("format", {"auto", "csv", "json"}, "auto")},
        .sockets = {.inputs = {in("path", T::String, "")}, .outputs = {geoOut("points"), fout("rows", T::Int)}},
        .exec =
            [](ExecContext& c) {
                const std::string path = c.s("path");
                std::string format = c.prop("format");
                if (format == "auto")
                    format = extensionOf(path) == ".json" ? "json" : "csv";
                GeometryPtr g = cachedRead("Read Data Table", path, format, [&] {
                    std::string text = readText(path);
                    Table t = format == "json" ? parseJsonTable(text) : parseCsv(text);
                    return pointGeometry(tableToPoints(t));
                });
                c.out("rows", constantInt(g->points ? int64_t(g->points->numPoints()) : 0));
                c.out("points", g);
            },
    });
}

GeometryPtr readMeshFile(const std::string& path)
{
    const std::string ext = extensionOf(path);
    geo::Mesh out;
    if (ext == ".obj")
    {
        geo::ObjReadResult r = geo::readObjFile(path);
        if (!r.ok())
            throw ExecError("Import Mesh: " + r.error);
        for (geo::ObjObject& o : r.objects)
            out.append(o.mesh);
    }
    else if (ext == ".stl")
    {
        geo::StlReadResult r = geo::readStlFile(path);
        if (!r.ok())
            throw ExecError("Import Mesh: " + r.error);
        for (geo::StlSolid& s : r.solids)
            out.append(s.mesh);
    }
    else if (ext == ".ply")
    {
        geo::PlyReadResult r = geo::readPlyFile(path);
        if (!r.ok())
            throw ExecError("Import Mesh: " + r.error);
        r.mesh.compact();
        if (r.pointCloud)
            return pointGeometry(std::move(r.mesh));
        out = std::move(r.mesh);
    }
    else
        throw ExecError("Import Mesh: " + path + " isn't obj, stl or ply");
    out.compact();
    return meshGeometry(std::move(out));
}

void importMesh()
{
    registerDecl({
        .type = "ImportMesh",
        .label = "Import Mesh",
        .category = "Input",
        .doc = "Reads an OBJ, STL or PLY file (a PLY without faces as points). Objects in the file are joined. Reread when the file changes.",
        .sockets = {.inputs = {in("path", T::String, ""), bounded(in("scale", T::Float, 1.0, "Uniform scale, like 100 for a file in metres."), 0.0)},
                    .outputs = {geoOut("geometry")}},
        .exec =
            [](ExecContext& c) {
                const std::string path = c.s("path");
                GeometryPtr g = cachedRead("Import Mesh", path, "mesh", [&] { return readMeshFile(path); });
                const double s = c.f("scale");
                if (s != 1.0 && !g->empty())
                    g = makeGeometry(transformed(*g, glm::dmat4(glm::dmat3(s))));
                c.out("geometry", g);
            },
    });
}
} // namespace lila::gn::nodes::data

namespace lila::gn::nodes
{
void registerDataNodes()
{
    data::readDataTable();
    data::importMesh();
}
} // namespace lila::gn::nodes
