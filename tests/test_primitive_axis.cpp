#include <doctest/doctest.h>

#include "TestUtil.h"

#include "geo/Mesh.h"
#include "math/MathCore.h"
#include "ops/Command.h"
#include "scene/Graph.h"
#include "scene/LastOp.h"
#include "scene/MeshNodes.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using namespace lila;
using json = nlohmann::json;
using namespace testutil;

namespace
{
int64_t upAxisOf(const std::string& generator)
{
    scene::Node* n = scene::Graph::current().find(generator);
    REQUIRE(n);
    return n->as<int64_t>(n->type().attrIndex("upAxis"));
}

glm::dvec3 faceNormal(const geo::Mesh& m)
{
    for (geo::Index f = 0; f < m.faceCapacity(); ++f)
        if (m.faceValid(f))
            return m.faceNormal(f);
    return glm::dvec3(0.0);
}
} // namespace

TEST_CASE("primitive upAxis turns the geometry to each axis")
{
    run("newScene");
    struct Case
    {
        const char* axis;
        glm::dvec3 dir;
    };
    for (const Case& c : {Case{"x", {1, 0, 0}}, Case{"y", {0, 1, 0}}, Case{"z", {0, 0, 1}}, Case{"-x", {-1, 0, 0}}, Case{"-y", {0, -1, 0}},
                          Case{"-z", {0, 0, -1}}})
    {
        CAPTURE(c.axis);
        // Base at the origin, so the side it rises toward shows the sign.
        json r = run("polyCylinder", {{"radius", 10.0}, {"height", 100.0}, {"baseAtOrigin", true}, {"upAxis", c.axis}});
        geo::Bounds b = meshOf(r[0]).bounds();
        const glm::dvec3 size = b.size();
        CHECK(glm::dot(size, glm::abs(c.dir)) == doctest::Approx(100.0));
        CHECK(size.x + size.y + size.z == doctest::Approx(140.0).epsilon(0.01));
        const glm::dvec3 centre = (b.min + b.max) * 0.5;
        CHECK(glm::dot(centre, c.dir) == doctest::Approx(50.0));

        // A plane faces along its up, so the turn keeps the winding.
        json plane = run("polyPlane", {{"width", 100.0}, {"depth", 100.0}, {"upAxis", c.axis}});
        CHECK(glm::dot(faceNormal(meshOf(plane[0])), c.dir) == doctest::Approx(1.0));
    }
}

TEST_CASE("every primitive has upAxis, defaulting to the scene up axis")
{
    run("newScene");
    for (const char* type : {"polyCube", "polySphere", "polyCylinder", "polyCapsule", "polyCone", "polyPlane", "polyTorus", "polyDisc", "polyPipe",
                             "polyQuadSphere", "polyIcoSphere"})
    {
        CAPTURE(type);
        json r = run(type);
        CHECK(upAxisOf(r[1]) == 1);
        json z = run(type, {{"upAxis", "z"}});
        CHECK(upAxisOf(z[1]) == 2);
        const geo::Bounds a = meshOf(r[0]).bounds(), b = meshOf(z[0]).bounds();
        CHECK(a.size().y == doctest::Approx(b.size().z));
    }
    CHECK_THROWS_AS(run("polyCube", {{"width", 100.0}, {"height", 100.0}, {"depth", 100.0}, {"upAxis", "w"}}), CommandError);

    math::setUpAxis(math::UpAxis::Z);
    json r = run("polyCone", {{"radius", 50.0}, {"height", 100.0}});
    math::setUpAxis(math::UpAxis::Y);
    CHECK(upAxisOf(r[1]) == 2);
}

TEST_CASE("primitive upAxis survives saving, and files without it load as y")
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "lila_tests";
    fs::create_directories(dir);
    for (const char* ext : {".lila", ".lilaa"})
    {
        CAPTURE(ext);
        run("newScene");
        json r = run("polyTorus", {{"radius", 50.0}, {"sectionRadius", 20.0}, {"upAxis", "-x"}});
        const glm::dvec3 before = meshOf(r[0]).bounds().size();
        const std::string path = (dir / (std::string("prim_axis") + ext)).string();
        run("saveScene", {{"args", {path}}});
        run("newScene");
        run("openScene", {{"args", {path}}});
        CHECK(upAxisOf(r[1]) == 3);
        const glm::dvec3 after = meshOf(r[0]).bounds().size();
        CHECK(after.x == doctest::Approx(before.x));
        CHECK(after.y == doctest::Approx(before.y));
    }

    // An older file has no upAxis line on the generator.
    const std::string path = (dir / "prim_axis.lilaa").string();
    std::stringstream text;
    text << std::ifstream(path).rdbuf();
    const std::string stripped = std::regex_replace(text.str(), std::regex("\n[ \t]+upAxis [^\n]*"), "");
    REQUIRE(stripped != text.str());
    const std::string old = (dir / "prim_axis_old.lilaa").string();
    std::ofstream(old) << stripped;
    run("newScene");
    run("openScene", {{"args", {old}}});
    CHECK(upAxisOf("PolyTorus") == 1);
    CHECK(meshOf("Torus").bounds().size().y == doctest::Approx(40.0));
}

TEST_CASE("new primitives fit a 10 cm cube, and files leaving sizes out load at 1 m")
{
    namespace fs = std::filesystem;
    run("newScene");
    for (const char* type : {"polyCube", "polySphere", "polyCylinder", "polyCapsule", "polyCone", "polyTorus", "polyDisc", "polyQuadSphere",
                             "polyIcoSphere", "polyPlane"})
    {
        CAPTURE(std::string(type));
        const glm::dvec3 size = meshOf(run(type)[0]).bounds().size();
        // Faceted spheres fall a little short of the cube.
        CHECK(std::max({size.x, size.y, size.z}) <= 10.0 + 1e-9);
        CHECK(std::max({size.x, size.y, size.z}) > 9.0);
    }
    CHECK(meshOf(run("polyPipe")[0]).bounds().size().y == doctest::Approx(20.0));
    run("spaceLocator", {{"name", "Loc"}});
    CHECK(run("getAttr", {{"args", {"Loc.drawSize"}}}).get<double>() == doctest::Approx(5.0));

    // A file leaving the sizes out means the 1 m defaults, and a locator's drawSize 1.
    const fs::path dir = fs::temp_directory_path() / "lila_tests";
    fs::create_directories(dir);
    run("newScene");
    run("polyCube");
    run("spaceLocator", {{"name", "Loc"}});
    const std::string path = (dir / "prim_sizes.lilaa").string();
    run("saveScene", {{"args", {path}}});
    std::stringstream text;
    text << std::ifstream(path).rdbuf();
    const std::string stripped = std::regex_replace(text.str(), std::regex("\n[ \t]+(width|height|depth|drawSize) [^\n]*"), "");
    REQUIRE(stripped != text.str());
    const std::string old = (dir / "prim_sizes_old.lilaa").string();
    std::ofstream(old) << stripped;
    run("newScene");
    run("openScene", {{"args", {old}}});
    CHECK(meshOf("Cube").bounds().size().x == doctest::Approx(100.0));
    CHECK(run("getAttr", {{"args", {"Loc.drawSize"}}}).get<double>() == doctest::Approx(1.0));
}

TEST_CASE("a primitive created from the UI is adjustable, upAxis included")
{
    scene::lastOp::install();
    run("newScene");
    json r = runUi("polyTorus", {{"radius", 50.0}, {"sectionRadius", 20.0}, {"position", {10, 0, 0}}});
    REQUIRE(scene::lastOp::live());
    REQUIRE(scene::lastOp::current()->command == "polyTorus");
    CHECK(scene::lastOp::current()->call["upAxis"] == "y");
    for (const FlagDef* f : scene::lastOp::editableFlags("polyTorus"))
        if (f->name == "radius")
            CHECK(scene::lastOp::flagValue(*scene::lastOp::current(), *f) == 50.0);

    scene::lastOp::rerun({{"upAxis", "z"}, {"radius", 80.0}});
    json gens = run("ls", {{"type", "polyTorus"}});
    REQUIRE(gens.size() == 1);
    CHECK(upAxisOf(gens[0]) == 2);
    const geo::Bounds b = meshOf(r[0]).bounds();
    CHECK(b.size().x == doctest::Approx(200.0));
    CHECK(b.size().z == doctest::Approx(40.0));
    CHECK(run("getAttr", {{"args", {std::string(r[0]) + ".translateX"}}}).get<double>() == doctest::Approx(10.0));

    run("undo");
    gens = run("ls", {{"type", "polyTorus"}});
    REQUIRE(gens.size() == 1);
    CHECK(upAxisOf(gens[0]) == 1);
    run("undo");
    CHECK(run("ls", {{"type", "polyTorus"}}).empty());
}

TEST_CASE("moving a created primitive keeps it adjustable and the move survives adjusting")
{
    scene::lastOp::install();
    run("newScene");
    json r = runUi("polyQuadSphere", {{"radius", 50.0}});
    const std::string name = r[0];
    REQUIRE(scene::lastOp::live());
    runUi("move", {{"args", {name}}, {"by", {0, 30, 0}}});
    CHECK(scene::lastOp::reachable());
    runUi("setAttr", {{"args", {name + ".translateX", 5.0}}});
    CHECK(scene::lastOp::reachable());

    scene::lastOp::rerun({{"subdivisions", 5}});
    auto tx = [&](const char* attr) { return run("getAttr", {{"args", {name + "." + attr}}}).get<double>(); };
    CHECK(tx("translateY") == doctest::Approx(30.0));
    CHECK(tx("translateX") == doctest::Approx(5.0));
    // A second adjustment keeps the move too.
    scene::lastOp::rerun({{"subdivisions", 4}});
    CHECK(tx("translateY") == doctest::Approx(30.0));
    json gens = run("ls", {{"type", "polyQuadSphere"}});
    REQUIRE(gens.size() == 1);
    CHECK(run("getAttr", {{"args", {std::string(gens[0]) + ".subdivisions"}}}).get<int64_t>() == 4);

    // Moving another object ends it.
    json other = runUi("polyCube", {{"width", 100.0}, {"height", 100.0}, {"depth", 100.0}});
    runUi("move", {{"args", {name}}, {"by", {0, 1, 0}}});
    CHECK(scene::lastOp::current()->command == "polyCube");
    CHECK_FALSE(scene::lastOp::reachable());
    (void)other;
}

TEST_CASE("placing a primitive turns the scene up to the surface normal")
{
    run("newScene");
    json r = run("polyCylinder", {{"height", 100.0}, {"radius", 10.0}, {"upAxis", "x"}, {"position", {0, 0, 0}}, {"normal", {0, 0, 1}}});
    scene::Graph& g = scene::Graph::current();
    g.evaluate();
    // The cylinder's axis is x in object space; standing its y on +z leaves the axis along world x.
    const glm::dmat4 w = g.find(r[0].get<std::string>())->as<glm::dmat4>(scene::Transform::kWorldMatrix);
    const glm::dvec3 axis = glm::dvec3(w * glm::dvec4(1, 0, 0, 0));
    CHECK(std::abs(axis.x) == doctest::Approx(1.0));
}
