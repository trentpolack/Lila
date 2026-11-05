#include "ops/Command.h"
#include "ops/Undo.h"
#include "geo/Mesh.h"
#include "scene/Graph.h"
#include "scene/MeshNodes.h"
#include "scene/Transform.h"

#include <doctest/doctest.h>

using namespace lila;

namespace
{
json run(const char* cmd, json call = json::object()) { return ops::execute(cmd, call); }

scene::MeshPtr outMesh(const std::string& path)
{
    scene::Graph& g = scene::Graph::current();
    g.evaluate();
    scene::Node* n = g.find(path);
    REQUIRE(n);
    return n->as<scene::MeshPtr>(scene::MeshNode::kOutMesh);
}

double width(const geo::Mesh& m) { return m.bounds().max.x - m.bounds().min.x; }

void reset()
{
    scene::Graph::current().clear();
    UndoStack::get().clear();
}
} // namespace

TEST_CASE("clay commands: create, add, cut, group, refresh, undo")
{
    reset();
    const json made = run("createClay", {{"name", "Blob"}, {"symmetry", "off"}});
    const std::string object = made[0], mesh = made[1];
    scene::MeshPtr start = outMesh(mesh);
    REQUIRE(start);
    CHECK(width(*start) == doctest::Approx(20.0).epsilon(0.02));

    const std::string arm = run("addClayPrimitive", {{"args", {object}}, {"shape", "capsule"}, {"radius", 3.0}, {"endRadius", 3.0},
                                                     {"length", 20.0}, {"rotation", {0, 0, 90}}, {"position", {15, 0, 0}}});
    CHECK(width(*outMesh(mesh)) == doctest::Approx(38.0).epsilon(0.02));

    run("addClayPrimitive", {{"args", {arm}}, {"operation", "subtract"}, {"radius", 4.0}, {"position", {25, 0, 0}}, {"name", "Cut"}});
    const double cut = width(*outMesh(mesh));
    CHECK(cut < 37.0);

    // Grouping the arm with its cut keeps the cut off the body; it still trims the arm.
    const std::string group = run("groupClay", {{"args", {arm, "Cut"}}});
    CHECK(width(*outMesh(mesh)) == doctest::Approx(cut).epsilon(0.02));
    scene::Graph& g = scene::Graph::current();
    CHECK(g.find(group)->type().name == "clayGroup");

    UndoStack::get().undo();
    CHECK(g.find(group) == nullptr);
    CHECK(width(*outMesh(mesh)) == doctest::Approx(cut).epsilon(0.02));

    CHECK(run("refreshClay", {{"args", {arm}}}).size() == 1);
}

TEST_CASE("clay commands: createClay mirrors by default and starts with the shape asked for")
{
    reset();
    const json made = run("createClay", {{"shape", "capsule"}});
    scene::Graph& g = scene::Graph::current();
    const scene::Node* surface = g.find(made[2].get<std::string>());
    REQUIRE(surface);
    CHECK(surface->as<int64_t>(surface->type().attrIndex("symmetry")) == 1);
    const scene::Node* first = g.find(made[3].get<std::string>());
    REQUIRE(first);
    CHECK(first->as<int64_t>(first->type().attrIndex("shape")) == 3);
}

TEST_CASE("clay commands: symmetry, a bad voxel size warns, convert bakes a plain mesh")
{
    reset();
    const json made = run("createClay", {{"name", "Pair"}, {"symmetry", "x"}, {"empty", true}});
    const std::string object = made[0], mesh = made[1], surface = made[2];
    CHECK(made[3].is_null());
    run("addClayPrimitive", {{"args", {object}}, {"radius", 3.0}, {"position", {10, 0, 0}}});
    scene::MeshPtr m = outMesh(mesh);
    CHECK(m->bounds().min.x == doctest::Approx(-13.0).epsilon(0.02));

    scene::Graph& g = scene::Graph::current();
    scene::Node* s = g.find(surface);
    g.setValue({s->id(), s->type().attrIndex("voxelSize"), -1}, 0.01);
    run("addClayPrimitive", {{"args", {object}}, {"radius", 900.0}});
    g.evaluate();
    CHECK_FALSE(s->as<std::string>(s->type().attrIndex("warning")).empty());
    UndoStack::get().undo();
    g.setValue({s->id(), s->type().attrIndex("voxelSize"), -1}, 0.5);
    g.evaluate();
    CHECK(s->as<std::string>(s->type().attrIndex("warning")).empty());

    // Kept by default, hidden; keepSource off deletes it.
    const std::string kept = run("convertClayToMesh", {{"args", {object}}, {"voxelSize", 0.5}, {"name", "Kept"}});
    CHECK(g.find(surface) != nullptr);
    CHECK_FALSE(g.find(object)->as<bool>(scene::Transform::kVisibility));
    CHECK(g.find(kept) != nullptr);
    const std::string baked = run("convertClayToMesh", {{"args", {object}}, {"voxelSize", 0.5}, {"keepSource", false}});
    CHECK(g.find(surface) == nullptr);
    scene::MeshPtr out = outMesh(baked);
    REQUIRE(out);
    CHECK(out->faceCapacity() > 0);
    CHECK(!g.source({g.find(baked)->id(), scene::MeshNode::kInMesh, -1}));
}

TEST_CASE("clay commands: a stroke is a chain in its own group, and clayShapeAt picks the shape at a point")
{
    reset();
    const json made = run("createClay", {{"name", "Horn"}, {"empty", true}});
    const std::string object = made[0], mesh = made[1];
    const std::string stroke = run("addClayStroke", {{"args", {object}}, {"points", {{0, 8, 0}, {0, 20, 0}, {6, 28, 0}}}, {"radius", 3.0}, {"endRadius", 1.0}});
    scene::Graph& g = scene::Graph::current();
    REQUIRE(g.find(stroke));
    CHECK(g.find(stroke)->type().name == "clayGroup");
    CHECK(g.children(g.find(stroke)->id()).size() == 2);
    scene::MeshPtr m = outMesh(mesh);
    CHECK(m->bounds().max.y == doctest::Approx(29.0).epsilon(0.03));
    CHECK(m->bounds().max.x == doctest::Approx(7.0).epsilon(0.05));

    CHECK(run("clayShapeAt", {{"args", {object}}, {"point", {100, 0, 0}}}).is_string());
    const json span = run("clayShapeAt", {{"args", {object}}, {"point", {3, 24, 0}}});
    REQUIRE(span.is_string());
    CHECK(g.parent(g.find(span.get<std::string>())->id()) == g.find(stroke)->id());

    // Extending from a span adds to its stroke rather than starting a new one.
    const std::string same = run("addClayStroke", {{"args", {span}}, {"extend", true}, {"points", {{6, 28, 0}, {12, 30, 0}}}, {"radius", 1.0}});
    CHECK(g.find(same)->id() == g.find(stroke)->id());
    CHECK(g.children(g.find(stroke)->id()).size() == 3);
    CHECK(outMesh(mesh)->bounds().max.x == doctest::Approx(13.0).epsilon(0.05));
}

TEST_CASE("clay commands: clayJoints chains a stroke and weights the converted mesh from the clay")
{
    reset();
    const json made = run("createClay", {{"name", "Worm"}});
    const std::string object = made[0];
    run("addClayStroke", {{"args", {object}}, {"points", {{0, 8, 0}, {0, 20, 0}, {0, 32, 0}}}, {"radius", 3.0}, {"endRadius", 2.0}, {"name", "Tail"}});
    const std::string mesh = run("convertClayToMesh", {{"args", {object}}, {"voxelSize", 0.5}});
    const json r = run("clayJoints", {{"args", {object}}, {"mesh", mesh}});
    scene::Graph& g = scene::Graph::current();
    REQUIRE(r[1].is_string());
    // The root, then a joint at each span start and one at the end: 3 in the tail.
    const scene::Node* root = g.find(r[0].get<std::string>());
    REQUIRE(root);
    CHECK(g.children(root->id()).size() == 1);

    // A point at the tail tip leans on the tail's last joints; one at the bottom of the body on the root.
    const scene::MeshPtr m = outMesh(mesh);
    uint32_t tip = 0, bottom = 0;
    for (geo::Index p = 0; p < m->pointCapacity(); ++p)
    {
        if (m->position(p).y > m->position(tip).y)
            tip = p;
        if (m->position(p).y < m->position(bottom).y)
            bottom = p;
    }
    const json w = run("skinPercent", {{"args", {mesh}}, {"vertices", {tip, bottom}}});
    REQUIRE(w.size() == 2);
    CHECK(w[0]["weights"].value("Tail_3", 0.0) + w[0]["weights"].value("Tail_2", 0.0) > 0.9);
    CHECK(w[1]["weights"].value("Worm_root", 0.0) > 0.9);
}

TEST_CASE("clay commands: addClayLink joins two shapes and follows them as they move and resize")
{
    reset();
    const json made = run("createClay", {{"name", "Chain"}, {"symmetry", "off"}});
    const std::string object = made[0], mesh = made[1], body = made[3];
    const std::string head = run("addClayPrimitive", {{"args", {object}}, {"radius", 5.0}, {"position", {60, 0, 0}}});
    scene::Graph& g = scene::Graph::current();

    // The widest the surface gets around a point on the axis, among vertices within 1 cm of it along that axis.
    auto girth = [&](int axis, double at) {
        double r = 0.0;
        const scene::MeshPtr m = outMesh(mesh);
        for (geo::Index p = 0; p < m->pointCapacity(); ++p)
            if (m->pointValid(p) && std::abs(m->position(p)[axis] - at) < 1.0)
                r = std::max(r, std::abs(m->position(p)[(axis + 2) % 3]));
        return r;
    };
    CHECK(girth(0, 30.0) == 0.0);

    const std::string link = run("addClayLink", {{"args", {body, head}}, {"thickness", 0.5}, {"blend", 0.0}});
    REQUIRE(g.find(link));
    CHECK(g.find(link)->type().name == "clayLink");
    // Halfway, the cone's radius is halfway between 0.5 * 10 and 0.5 * 5.
    CHECK(girth(0, 30.0) == doctest::Approx(3.75).epsilon(0.1));

    run("setAttr", {{"args", {head + ".translate", 0, 60, 0}}});
    CHECK(girth(0, 30.0) == 0.0);
    CHECK(girth(1, 30.0) == doctest::Approx(3.75).epsilon(0.1));

    run("setAttr", {{"args", {head + ".radius", 15.0}}});
    CHECK(girth(1, 30.0) == doctest::Approx(6.25).epsilon(0.1));

    run("setAttr", {{"args", {link + ".thickness", 1.0}}});
    CHECK(girth(1, 30.0) == doctest::Approx(12.5).epsilon(0.1));

    UndoStack::get().undo();
    UndoStack::get().undo();
    UndoStack::get().undo();
    UndoStack::get().undo();
    CHECK(g.find(link) == nullptr);
    CHECK(girth(0, 30.0) == 0.0);

    CHECK_THROWS(run("addClayLink", {{"args", {body, body}}}));
    CHECK_THROWS(run("addClayLink", {{"args", {body, object}}}));
}

TEST_CASE("clay commands: duplicate copies shapes and links into the clay, links joining the copied ends")
{
    reset();
    const json made = run("createClay", {{"name", "Hand"}, {"symmetry", "off"}});
    const std::string object = made[0], clay = made[2], palm = made[3];
    const std::string knuckle = run("addClayPrimitive", {{"args", {object}}, {"radius", 3.0}, {"position", {0, 20, 0}}});
    const std::string tip = run("addClayPrimitive", {{"args", {object}}, {"radius", 2.0}, {"position", {0, 30, 0}}});
    const std::string toKnuckle = run("addClayLink", {{"args", {palm, knuckle}}, {"skipSelect", true}});
    const std::string toTip = run("addClayLink", {{"args", {knuckle, tip}}, {"skipSelect", true}});
    scene::Graph& g = scene::Graph::current();
    g.evaluate();
    const size_t items = g.elements({g.find(clay)->id(), g.find(clay)->type().attrIndex("items")}).size();

    const json copies = run("duplicate", {{"args", {knuckle, tip, toKnuckle, toTip}}});
    REQUIRE(copies.size() == 4);
    CHECK(g.elements({g.find(clay)->id(), g.find(clay)->type().attrIndex("items")}).size() == items + 4);

    auto endOf = [&](const std::string& link, const char* end) {
        const scene::Node* n = g.find(link);
        return g.node(g.source({n->id(), n->type().attrIndex(end), -1})->node)->path();
    };
    // The knuckle-to-tip copy joins the copied knuckle and tip; the palm link's copy still starts at the palm.
    CHECK(endOf(copies[3], "start") == copies[0].get<std::string>());
    CHECK(endOf(copies[3], "end") == copies[1].get<std::string>());
    CHECK(endOf(copies[2], "start") == palm);
    CHECK(endOf(copies[2], "end") == copies[0].get<std::string>());

    // Moving the copied finger moves its links, leaving the first finger as it was.
    run("move", {{"args", {copies[0], copies[1]}}, {"by", {15, 0, 0}}});
    const scene::MeshPtr m = outMesh(made[1]);
    CHECK(m->bounds().max.x > 15.0);
    CHECK(m->bounds().min.x == doctest::Approx(-10.0).epsilon(0.05));
}

TEST_CASE("clay commands: a duplicated clay object meshes its own shapes")
{
    reset();
    const json made = run("createClay", {{"name", "Blob"}, {"symmetry", "off"}});
    const json copy = run("duplicate", {{"args", {made[0]}}});
    REQUIRE(copy.size() == 1);
    const std::string object = copy[0];
    run("move", {{"args", {object}}, {"by", {100, 0, 0}}});
    scene::Graph& g = scene::Graph::current();
    std::string mesh;
    for (scene::NodeId c : g.children(g.find(object)->id()))
        if (g.node(c)->type().isA("mesh"))
            mesh = g.node(c)->path();
    REQUIRE(!mesh.empty());
    const std::string shape = run("clayShapeAt", {{"args", {object}}, {"point", {100, 0, 0}}});
    run("setAttr", {{"args", {shape + ".radius", 4.0}}});
    CHECK(width(*outMesh(mesh)) == doctest::Approx(8.0).epsilon(0.05));
    CHECK(width(*outMesh("|Blob|BlobSurface")) == doctest::Approx(20.0).epsilon(0.05));
}
