#include <doctest/doctest.h>

#include "core/Decimal.h"

#include "ops/Command.h"
#include "geo/Mesh.h"
#include "io/SceneFile.h"
#include "math/MathCore.h"
#include "scene/Graph.h"
#include "scene/MeshNodes.h"
#include "scene/Node.h"

#include <ufbx.h>

#include <bit>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

using namespace lila;
using namespace lila::scene;
using nlohmann::json;
namespace fs = std::filesystem;

namespace
{
fs::path precisionFile(const std::string& name)
{
    fs::path dir = fs::temp_directory_path() / "lila_fbx_tests";
    fs::create_directories(dir);
    return dir / name;
}

/** Points with full 53-bit mantissas from a millimetre to 50 km off the origin, as CAD exports write them. */
std::vector<glm::dvec3> cadPoints()
{
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::vector<glm::dvec3> pts;
    for (double reach : {0.1, 10.0, 1.0e3, 1.0e5, 5.0e6})
        for (int i = 0; i < 300; ++i)
            pts.emplace_back(unit(rng) * reach, unit(rng) * reach, unit(rng) * reach);
    return pts;
}

std::string number(double v)
{
    char buf[32];
    return std::string(buf, std::to_chars(buf, buf + sizeof(buf), v).ptr);
}

bool sameBits(const glm::dvec3& a, const glm::dvec3& b)
{
    return std::bit_cast<uint64_t>(a.x) == std::bit_cast<uint64_t>(b.x) && std::bit_cast<uint64_t>(a.y) == std::bit_cast<uint64_t>(b.y) &&
           std::bit_cast<uint64_t>(a.z) == std::bit_cast<uint64_t>(b.z);
}

size_t mismatches(std::span<const glm::dvec3> got, std::span<const glm::dvec3> want)
{
    if (got.size() != want.size())
        return std::max(got.size(), want.size());
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        bad += sameBits(got[i], want[i]) ? 0 : 1;
    return bad;
}

std::vector<glm::dvec3> meshPoints(const std::string& name)
{
    Graph& g = Graph::current();
    g.evaluate();
    Node* n = g.find(name);
    REQUIRE(n);
    CHECK(n->as<glm::dmat4>(Transform::kWorldMatrix) == glm::dmat4(1.0));
    MeshPtr m = n->as<MeshPtr>(MeshNode::kOutMesh);
    REQUIRE(m);
    std::vector<glm::dvec3> out;
    for (geo::Index p = 0; p < m->pointCapacity(); ++p)
        if (m->pointValid(p))
            out.push_back(m->position(p));
    return out;
}

std::vector<glm::dvec3> objVertices(const fs::path& path)
{
    std::ifstream f(path);
    std::vector<glm::dvec3> out;
    for (std::string line; std::getline(f, line);)
    {
        if (!line.starts_with("v "))
            continue;
        glm::dvec3 v;
        const char* p = line.data() + 2;
        const char* end = line.data() + line.size();
        for (int k = 0; k < 3; ++k)
        {
            while (p < end && *p == ' ')
                ++p;
            p = lila::fromChars(p, end, v[k]).ptr;
        }
        out.push_back(v);
    }
    return out;
}

void importSource(const std::vector<glm::dvec3>& pts)
{
    fs::path src = precisionFile("cad_source.obj");
    {
        std::ofstream f(src, std::ios::binary);
        f << "o Part\n";
        for (const glm::dvec3& p : pts)
            f << "v " << number(p.x) << ' ' << number(p.y) << ' ' << number(p.z) << '\n';
        for (size_t i = 0; i + 2 < pts.size(); i += 3)
            f << "f " << i + 1 << ' ' << i + 2 << ' ' << i + 3 << '\n';
    }
    Graph::current().clear();
    json names = ops::execute("importObj", {{"args", {src.string()}}});
    REQUIRE(names.size() == 1);
    CHECK(names[0].get<std::string>() == "Part");
}
} // namespace

TEST_CASE("cad-precision points survive import and export bit for bit")
{
    io::registerCommands();
    math::setUpAxis(math::UpAxis::Y);
    const std::vector<glm::dvec3> pts = cadPoints();

    // The data needs doubles: float32, as Maya and Blender store vertices, moves almost every point.
    size_t floatLoss = 0;
    for (const glm::dvec3& p : pts)
        floatLoss += sameBits(glm::dvec3(glm::vec3(p)), p) ? 0 : 1;
    CHECK(floatLoss > pts.size() * 9 / 10);

    SUBCASE("obj import")
    {
        importSource(pts);
        CHECK(mismatches(meshPoints("Part"), pts) == 0);
    }

    SUBCASE("obj export and reimport")
    {
        importSource(pts);
        fs::path out = precisionFile("cad_roundtrip.obj");
        ops::execute("exportObj", {{"args", {out.string(), "Part"}}});
        CHECK(mismatches(objVertices(out), pts) == 0);
        Graph::current().clear();
        ops::execute("importObj", {{"args", {out.string()}}});
        CHECK(mismatches(meshPoints("Part"), pts) == 0);
    }

    SUBCASE("fbx export and reimport, every axis profile")
    {
        for (const char* profile : {"generic", "ue5", "ue6"})
        {
            CAPTURE(profile);
            importSource(pts);
            fs::path out = precisionFile(std::string("cad_roundtrip_") + profile + ".fbx");
            ops::execute("exportFbx", {{"args", {out.string(), "Part"}}, {"profile", profile}, {"atOrigin", false}});
            if (std::string_view(profile) == "generic")
            {
                ufbx_load_opts opts{};
                ufbx_error err{};
                ufbx_scene* scene = ufbx_load_file(out.string().c_str(), &opts, &err);
                REQUIRE(scene);
                REQUIRE(scene->meshes.count == 1);
                const ufbx_vertex_vec3& v = scene->meshes.data[0]->vertex_position;
                std::vector<glm::dvec3> raw;
                for (size_t i = 0; i < v.values.count; ++i)
                    raw.emplace_back(v.values.data[i].x, v.values.data[i].y, v.values.data[i].z);
                ufbx_free_scene(scene);
                CHECK(mismatches(raw, pts) == 0);
            }
            Graph::current().clear();
            ops::execute("importFbx", {{"args", {out.string()}}});
            CHECK(mismatches(meshPoints("Part"), pts) == 0);
        }
    }

    SUBCASE("scene save and open, binary and text")
    {
        for (const char* ext : {".lila", ".lilaa"})
        {
            CAPTURE(ext);
            importSource(pts);
            fs::path out = precisionFile(std::string("cad_roundtrip") + ext);
            ops::execute("saveScene", {{"args", {out.string()}}});
            Graph::current().clear();
            ops::execute("openScene", {{"args", {out.string()}}, {"force", true}});
            CHECK(mismatches(meshPoints("Part"), pts) == 0);
        }
    }
}
