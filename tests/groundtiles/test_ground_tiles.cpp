#include "TileCore.h"

#include "ops/Command.h"
#include "extensions/ExtensionManager.h"
#include "geo/ColorSets.h"
#include "geo/Mesh.h"
#include "scene/Graph.h"
#include "scene/Materials.h"
#include "scene/MeshNodes.h"
#include "scene/Transform.h"

#include <doctest/doctest.h>

#include <QImage>

#include <cmath>
#include <functional>
#include <filesystem>
#include <fstream>
#include <set>

using namespace lila;
using namespace lila::scene;
using json = nlohmann::json;
namespace fs = std::filesystem;
using Approx = doctest::Approx;

namespace
{
json run(const char* cmd, json call = json::object()) { return ops::execute(cmd, call); }

Graph& graph() { return Graph::current(); }

void ensureLoaded()
{
    auto& em = extensions::ExtensionManager::get();
    if (!em.isLoaded("GroundTiles"))
        REQUIRE(em.load(LILA_TEST_GROUND_TILES_PACKAGE) == "GroundTiles");
}

void fresh()
{
    ensureLoaded();
    run("newScene");
}

/** A box w x h x d centred on centre, cut into segments along the axis it is longest on. */
std::string box(const std::string& name, glm::dvec3 size, int segments, glm::dvec3 centre = glm::dvec3(0.0))
{
    int axis = 0;
    for (int i = 1; i < 3; ++i)
        if (size[i] > size[axis])
            axis = i;
    const int a = (axis + 1) % 3, b = (axis + 2) % 3;
    json points = json::array(), faces = json::array();
    const glm::dvec2 ring[4] = {{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};
    for (int s = 0; s <= segments; ++s)
        for (const glm::dvec2& r : ring)
        {
            glm::dvec3 p(0.0);
            p[axis] = (double(s) / segments - 0.5) * size[axis];
            p[a] = r.x * size[a];
            p[b] = r.y * size[b];
            p += centre;
            points.push_back({p.x, p.y, p.z});
        }
    for (int s = 0; s < segments; ++s)
        for (int k = 0; k < 4; ++k)
        {
            const int i0 = s * 4 + k, i1 = s * 4 + (k + 1) % 4;
            faces.push_back({i0, i1, i1 + 4, i0 + 4});
        }
    faces.push_back({3, 2, 1, 0});
    const int top = segments * 4;
    faces.push_back({top, top + 1, top + 2, top + 3});
    return run("createMesh", {{"points", points}, {"faces", faces}, {"name", name}}).get<std::string>();
}

/** The reference grass blade: 0.2 x 1 x 0.05, 16 segments along its length, which runs up the up axis. */
std::string blade(const std::string& up = "Z", const std::string& name = "testBlade")
{
    return up == "Z" ? box(name, {0.2, 0.05, 1.0}, 16) : box(name, {0.2, 1.0, 0.05}, 16);
}

json source(const std::string& name, double weight = 1.0, int64_t id = 0, double offset = 0.0, bool enabled = true)
{
    return {{"name", name}, {"weight", weight}, {"objectId", id}, {"heightOffset", offset}, {"enabled", enabled}};
}

json preset(const std::string& name) { return run("groundTilePresetSettings", {{"args", {name}}}); }

json generateTile(json values, std::vector<json> sources, const std::string& presetName = "Test", json palette = json::object(), json exp = nullptr)
{
    json call = {{"values", values}, {"sources", json(sources)}, {"preset", presetName}, {"palette", palette}};
    if (!exp.is_null())
        call["export"] = exp;
    return run("groundTileGenerate", call);
}

struct Box
{
    glm::dvec3 lo{1e300}, hi{-1e300};
};

Box worldBox(NodeId id)
{
    Graph& g = graph();
    g.evaluate();
    Box b;
    std::vector<NodeId> ids = g.descendants(id);
    ids.push_back(id);
    for (NodeId n : ids)
    {
        Node* node = g.node(n);
        if (!node || !node->type().isA("mesh"))
            continue;
        const MeshPtr& m = node->as<MeshPtr>(MeshNode::kOutMesh);
        const glm::dmat4 w = node->as<glm::dmat4>(Transform::kWorldMatrix);
        for (geo::Index p = 0; p < m->pointCapacity(); ++p)
            if (m->pointValid(p))
            {
                const glm::dvec3 q(w * glm::dvec4(m->position(p), 1.0));
                b.lo = glm::min(b.lo, q);
                b.hi = glm::max(b.hi, q);
            }
    }
    return b;
}

Box worldBox(const std::string& name) { return worldBox(graph().find(name)->id()); }

std::vector<NodeId> children(const std::string& group) { return graph().children(graph().find(group)->id()); }

std::vector<NodeId> meshChildren(const std::string& group)
{
    std::vector<NodeId> out;
    for (NodeId c : children(group))
        if (graph().node(c)->type().isA("mesh"))
            out.push_back(c);
    return out;
}

const geo::Mesh& meshOf(NodeId id)
{
    graph().evaluate();
    return *graph().node(id)->as<MeshPtr>(MeshNode::kOutMesh);
}

std::set<std::string> slotNames(NodeId id)
{
    std::set<std::string> out;
    for (const auto& [slot, count] : materials::slotUsage(meshOf(id)))
        if (count)
            out.insert(slot);
    return out;
}

std::set<std::array<int, 3>> pngColours(const std::string& path)
{
    QImage img(QString::fromStdString(path));
    REQUIRE(!img.isNull());
    img = img.convertToFormat(QImage::Format_RGB32);
    std::set<std::array<int, 3>> out;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x)
        {
            const QRgb c = img.pixel(x, y);
            out.insert({qRed(c), qGreen(c), qBlue(c)});
        }
    return out;
}

bool hasColour(const std::set<std::array<int, 3>>& found, const glm::dvec3& c)
{
    const std::array<int, 3> want{int(std::lround(c.r * 255)), int(std::lround(c.g * 255)), int(std::lround(c.b * 255))};
    for (const auto& f : found)
        if (std::abs(f[0] - want[0]) <= 2 && std::abs(f[1] - want[1]) <= 2 && std::abs(f[2] - want[2]) <= 2)
            return true;
    return false;
}

fs::path scratch(const std::string& name)
{
    const fs::path p = fs::temp_directory_path() / "lila_groundtiles_tests" / name;
    fs::create_directories(p);
    return p;
}

json readJson(const fs::path& p)
{
    std::ifstream f(p);
    return json::parse(f);
}

json loose(json values)
{
    values["combineResult"] = false;
    return values;
}
} // namespace

TEST_CASE("ground tiles: schema and presets")
{
    ensureLoaded();
    CHECK(groundtiles::builtInPresets().size() == 23);
    CHECK(groundtiles::defaultPreset() == "Mushed Grass");
    CHECK(groundtiles::parameters().size() == 74);
    const json d = groundtiles::defaults();
    CHECK(d["combineResult"] == true);
    CHECK(d["snapBaseToGround"] == true);
    for (const std::string& name : groundtiles::builtInPresets())
    {
        const json* o = groundtiles::builtInOverrides(name);
        REQUIRE(o);
        CHECK_MESSAGE(!o->contains("combineResult"), name);
        CHECK_MESSAGE(o->value("snapBaseToGround", true), name);
        const json v = preset(name);
        CHECK(v.size() == groundtiles::parameters().size());
        const std::string stem = groundtiles::presetStem(name);
        CHECK(std::all_of(stem.begin(), stem.end(), [](char c) { return std::isalnum((unsigned char)c); }));
        for (auto it = o->begin(); it != o->end(); ++it)
            CHECK_MESSAGE(v[it.key()] == groundtiles::coerce(*groundtiles::findParam(it.key()), it.value()), (name + " " + it.key()));
    }
    CHECK(preset("Raked Gravel")["distribution"] == "Rows");
    CHECK(preset("Pebbles & Gravel")["maxTilt"] == 180.0);
    CHECK(preset("Forest Floor Leaves")["maxTilt"] == 28.0);
    CHECK(groundtiles::presetStem("Mushed Grass") == "MushedGrass");
    CHECK(groundtiles::presetStem("Reeds & River Grass") == "ReedsRiverGrass");
    CHECK(groundtiles::presetStem("Bark & Needle Litter") == "BarkNeedleLitter");
    CHECK(groundtiles::presetStem("") == "GroundTile");
    CHECK(groundtiles::nodeStem("Reeds & River Grass") == "Reeds_River_Grass");

    // Stale and partial settings land on the schema.
    const json s = groundtiles::sanitize({{"count", "17.6"}, {"distribution", "Nope"}, {"wrapEdges", 0}, {"unknown", 3}});
    CHECK(s["count"] == 18);
    CHECK(s["distribution"] == "Clustered");
    CHECK(s["wrapEdges"] == false);
    CHECK(!s.contains("unknown"));

    const json presets = run("groundTilePresets");
    CHECK(presets["builtIn"].size() == 23);
    run("groundTileSavePreset", {{"args", {"Mine"}}, {"values", {{"count", 33}}}});
    CHECK(run("groundTilePresets")["user"] == json::array({"Mine"}));
    CHECK(preset("Mine")["count"] == 33);
    CHECK_THROWS(run("groundTileDeletePreset", {{"args", {"Mushed Grass"}}}));
    run("groundTileDeletePreset", {{"args", {"Mine"}}});
    CHECK(run("groundTilePresets")["user"].empty());
}

TEST_CASE("ground tiles: lowest height and clustering readout")
{
    json v = groundtiles::defaults();
    v.update({{"layersEnabled", true}, {"layer1Height", -0.30}, {"layer2Height", 0.0}, {"layer3Height", 0.35}});
    CHECK(groundtiles::lowestHeight(v) == Approx(-0.30));
    v["layer1Weight"] = 0.0;
    CHECK(groundtiles::lowestHeight(v) == Approx(0.0));
    v.update({{"layer1Weight", 0.0}, {"layer2Weight", 0.0}, {"layer3Weight", 0.0}});
    CHECK(groundtiles::lowestHeight(v) == Approx(-0.30));
    v.update({{"layersEnabled", false}, {"baseHeight", -2.5}});
    CHECK(groundtiles::lowestHeight(v) == Approx(-2.5));

    json c = groundtiles::defaults();
    c.update({{"tileSize", 200.0}, {"clusterCount", 18}, {"clusterRadius", 12.0}, {"clusterStrength", 0.85}});
    CHECK(groundtiles::clusterCoverage(c) == Approx(18 * 3.14159265358979 * 144.0 / 40000.0));
    CHECK(groundtiles::describeClustering(c) == "Clusters cover 0.20x the tile: clear clumping.");
    c["clusterRadius"] = 60.0;
    CHECK(groundtiles::describeClustering(c).find("no clumping") != std::string::npos);
}

TEST_CASE("ground tiles: distributions are deterministic and stay on the tile")
{
    for (const char* mode : {"Uniform", "Clustered", "Stratified", "Poisson", "Rows"})
    {
        json v = groundtiles::defaults();
        v.update({{"distribution", mode}, {"tileSize", 50.0}, {"count", 200}, {"minSpacing", 2.0}, {"rowSpacing", 5.0}, {"seed", 99}});
        groundtiles::Random a(99), b(99), c(100);
        const auto pa = groundtiles::scatterPoints(v, a), pb = groundtiles::scatterPoints(v, b), pc = groundtiles::scatterPoints(v, c);
        CHECK_MESSAGE(pa == pb, mode);
        CHECK_MESSAGE(pa != pc, mode);
        CHECK_MESSAGE(!pa.empty(), mode);
        for (const glm::dvec2& p : pa)
        {
            CHECK(p.x >= -25.0);
            CHECK(p.x < 25.0);
            CHECK(p.y >= -25.0);
            CHECK(p.y < 25.0);
        }
        if (std::string(mode) == "Poisson")
            // Spacing holds across the seam too.
            for (size_t i = 0; i < pa.size(); ++i)
                for (size_t j = i + 1; j < pa.size(); ++j)
                {
                    double dx = std::abs(pa[i].x - pa[j].x), dy = std::abs(pa[i].y - pa[j].y);
                    dx = std::min(dx, 50.0 - dx);
                    dy = std::min(dy, 50.0 - dy);
                    CHECK(dx * dx + dy * dy >= 4.0 - 1e-9);
                }
    }
    const auto w = groundtiles::wrappedPositions(-9.5, 9.6, 10.0, 1.0);
    CHECK(w.size() == 4);
    CHECK(groundtiles::wrappedPositions(0.0, 0.0, 10.0, 1.0).size() == 1);
}

TEST_CASE("ground tiles: every preset generates")
{
    for (const std::string& name : groundtiles::builtInPresets())
    {
        fresh();
        const std::string b = blade();
        json v = loose(preset(name));
        v.update({{"tileSize", 30.0}, {"count", 12}, {"clusterCount", 3}, {"minSpacing", 2.0}, {"rowSpacing", 8.0}});
        const json r = generateTile(v, {source(b)}, name);
        REQUIRE_MESSAGE(r["group"].is_string(), name);
        const std::string grp = r["group"];
        CHECK_MESSAGE(meshChildren(grp).size() == r["instances"].get<size_t>(), name);
        CHECK(graph().find(b));
        const json stored = run("groundTileSettings", {{"args", {grp}}});
        CHECK(stored["preset"] == name);
        CHECK(stored["sources"][0]["name"] == b);
        CHECK(stored["values"]["tileSize"] == Approx(30.0));
        int strays = 0;
        for (NodeId c : meshChildren(grp))
        {
            const Box bb = worldBox(c);
            if (bb.hi.x < -15.0 - 5.0 || bb.lo.x > 15.0 + 5.0)
                ++strays;
        }
        CHECK_MESSAGE(strays == 0, name);
    }
    fresh();
    json v = loose(preset("Mushed Grass"));
    v.update({{"upAxis", "Y"}, {"tileSize", 30.0}, {"count", 12}, {"clusterCount", 3}});
    CHECK(generateTile(v, {source(blade("Y"))}, "Mushed Grass")["group"].is_string());
}

TEST_CASE("ground tiles: generation is deterministic for a seed")
{
    auto points = [](int64_t seed) {
        fresh();
        const std::string b = blade();
        json v = preset("Dry Grass");
        v.update({{"tileSize", 20.0}, {"count", 120}, {"seed", seed}, {"jitterEnabled", true}, {"latticeEnabled", true}});
        const json r = generateTile(v, {source(b)}, "Dry Grass");
        REQUIRE(r["group"].is_string());
        const std::vector<NodeId> kids = meshChildren(r["group"]);
        REQUIRE(kids.size() == 1);
        const geo::Mesh& m = meshOf(kids[0]);
        return std::vector<glm::dvec3>(m.positions().begin(), m.positions().end());
    };
    const auto a = points(7), b = points(7), c = points(8);
    CHECK(a.size() > 120 * 66);
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("ground tiles: every modifier changes the instance")
{
    auto single = [](json overrides) {
        fresh();
        const std::string b = blade();
        json v = groundtiles::defaults();
        v.update({{"tileSize", 100.0}, {"count", 1}, {"wrapEdges", false}, {"minScale", 1.0}, {"maxScale", 1.0}, {"lengthVariation", 0.0},
                  {"widthVariation", 0.0}, {"thicknessVariation", 0.0}, {"maxTilt", 0.0}, {"rotationMode", "Range"}, {"yawMin", 0.0},
                  {"yawMax", 0.0}, {"layersEnabled", false}, {"baseHeight", 0.0}, {"bendEnabled", false}, {"flareEnabled", false}});
        v.update(overrides);
        const json r = generateTile(v, {source(b)});
        REQUIRE(r["group"].is_string());
        const Box bb = worldBox(meshChildren(r["group"])[0]);
        return bb.hi - bb.lo;
    };
    const glm::dvec3 plain = single(json::object());
    const glm::dvec3 bent = single({{"bendEnabled", true}, {"minBend", 30.0}, {"maxBend", 30.0}});
    const glm::dvec3 bentMore = single({{"bendEnabled", true}, {"minBend", 60.0}, {"maxBend", 60.0}});
    CHECK(bent.y > plain.y * 4.0);
    CHECK(bentMore.y > bent.y);
    CHECK(bentMore.z < bent.z);
    CHECK(single({{"flareEnabled", true}, {"minFlareEnd", 3.0}, {"maxFlareEnd", 3.0}}).x > plain.x * 1.5);
    CHECK(glm::length(single({{"twistEnabled", true}, {"minTwist", 90.0}, {"maxTwist", 90.0}}) - plain) > 0.001);
    CHECK(glm::length(single({{"sineEnabled", true}, {"minSineAmplitude", 0.5}, {"maxSineAmplitude", 0.5}, {"sineWavelength", 1.0}}) - plain) > 0.01);
    CHECK(single({{"jitterEnabled", true}, {"jitterAmplitude", 1.0}, {"jitterFrequency", 20.0}}) != plain);
    CHECK(single({{"latticeEnabled", true}, {"latticeDivisions", 3}, {"latticeAmplitude", 0.2}}) != plain);
    CHECK(single({{"flattenEnabled", true}, {"minFlatten", 0.25}, {"maxFlatten", 0.25}}).z == Approx(plain.z * 0.25).epsilon(1e-6));
    CHECK(single({{"shearEnabled", true}, {"maxShear", 0.5}}) != plain);
}

TEST_CASE("ground tiles: source axes")
{
    fresh();
    // A blade lying along +X with its base at the origin.
    const std::string lying = box("lyingBlade", {47.59, 6.87, 3.06}, 16, {23.795, 0.0, 0.0});
    const json info = run("groundTileSourceInfo", {{"args", {lying}}});
    CHECK(info["valid"] == true);
    CHECK(info["length"] == "X");
    CHECK(info["thickness"] == "Z");
    CHECK(info["describe"] == "grows +X, thin on Z");
    CHECK(run("groundTileSourceInfo", {{"args", {lying}}, {"baseEnd", "Max"}})["describe"] == "grows -X, thin on Z");
    CHECK(run("groundTileSourceInfo", {{"args", {lying}}, {"lengthAxis", "Y"}})["length"] == "Y");
    CHECK(run("groundTileSourceInfo", {{"args", {"nothing"}}})["valid"] == false);

    // The reach is from the pivot.
    const std::string offset = box("offsetPivot", {10.0, 2.0, 2.0}, 1, {5.0, 0.0, 0.0});
    CHECK(run("groundTileSourceInfo", {{"args", {offset}}})["reach"].get<double>() == Approx(std::sqrt(100.0 + 1.0 + 1.0)).epsilon(1e-4));
}

TEST_CASE("ground tiles: wrapping duplicates edge crossers onto the opposite edge")
{
    fresh();
    json v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 20.0}, {"count", 200}, {"jitterEnabled", true}});
    const json r = generateTile(v, {source(blade())}, "Mushed Grass");
    CHECK(r["instances"].get<int>() > 200);

    for (auto [name, atBase] : std::vector<std::pair<std::string, bool>>{
             {"Mushed Grass", false}, {"Dry Grass", true}, {"Pebbles & Gravel", true}, {"Raked Gravel", true}, {"Forest Floor Leaves", true}})
    {
        fresh();
        const std::string mesh = box("seamBlade", {47.59, 6.87, 3.06}, 8, {atBase ? 23.795 : 0.0, 0.0, 0.0});
        const double tile = 200.0, edge = tile * 0.5;
        json values = loose(preset(name));
        values.update({{"tileSize", tile}, {"count", 250}, {"seed", 7}});
        const json g = generateTile(values, {source(mesh)}, name);
        REQUIRE(g["group"].is_string());
        std::vector<Box> boxes;
        for (NodeId c : meshChildren(g["group"]))
            boxes.push_back(worldBox(c));
        auto twin = [&](const Box& b, int axis, double offset) {
            for (const Box& o : boxes)
            {
                bool same = true;
                for (int i = 0; i < 3 && same; ++i)
                {
                    const double d = i == axis ? offset : 0.0;
                    same = std::abs(b.lo[i] + d - o.lo[i]) < 0.01 && std::abs(b.hi[i] + d - o.hi[i]) < 0.01;
                }
                if (same)
                    return true;
            }
            return false;
        };
        int unmatched = 0;
        for (const Box& b : boxes)
            for (int axis : {0, 1})
            {
                if (b.hi[axis] > edge && !twin(b, axis, -tile))
                    ++unmatched;
                if (b.lo[axis] < -edge && !twin(b, axis, tile))
                    ++unmatched;
            }
        CHECK_MESSAGE(unmatched == 0, name);
    }
}

TEST_CASE("ground tiles: placement, pivots and the source's own rotation")
{
    fresh();
    const std::string mesh = box("offsetPivot", {10.0, 2.0, 2.0}, 1, {5.0, 0.0, 0.0});
    json v = groundtiles::defaults();
    v.update({{"tileSize", 100.0}, {"count", 4}, {"wrapEdges", false}, {"seed", 3}, {"minScale", 1.0}, {"maxScale", 1.0}, {"maxTilt", 0.0},
              {"layersEnabled", false}, {"baseHeight", 0.0}, {"bendEnabled", false}, {"flareEnabled", false}, {"widthVariation", 0.0},
              {"snapBaseToGround", false}, {"combineResult", false}});
    json r = generateTile(v, {source(mesh)});
    for (NodeId c : meshChildren(r["group"]))
    {
        Node* n = graph().node(c);
        const glm::dvec3 pivot(n->as<glm::dmat4>(Transform::kWorldMatrix) * glm::dvec4(n->as<glm::dvec3>(Transform::kRotatePivot), 1.0));
        CHECK(pivot.z == Approx(0.0).scale(1.0).epsilon(1e-4));
    }
    v["snapBaseToGround"] = true;
    r = generateTile(v, {source(mesh)});
    for (NodeId c : meshChildren(r["group"]))
        CHECK(std::abs(worldBox(c).lo.z) < 1e-4);

    fresh();
    const std::string tilted = box("tilted", {20.0, 2.0, 2.0}, 8);
    graph().setValue({graph().find(tilted)->id(), Transform::kRotate, -1}, glm::dvec3(0.0, 0.0, 40.0));
    v.update({{"count", 3}, {"seed", 5}, {"rotationMode", "Range"}, {"yawMin", 0.0}, {"yawMax", 0.0}});
    r = generateTile(v, {source(tilted)});
    for (NodeId c : meshChildren(r["group"]))
        CHECK(graph().node(c)->as<glm::dvec3>(Transform::kRotate).z == Approx(40.0).epsilon(1e-6));
}

TEST_CASE("ground tiles: combine, object ids and the ID map bake")
{
    fresh();
    const std::string grass = blade("Z", "grass"), weed = blade("Z", "weed");
    const json palette = {{"1", {0.25, 0.75, 0.125}}, {"-1", {0.5, 0.25, 0.75}}};
    json v = preset("Mushed Grass");
    v.update({{"tileSize", 4.0}, {"count", 40}, {"seed", 7}, {"minScale", 3.0}, {"maxScale", 4.0}, {"combineResult", true}, {"groundPlane", true}, {"groundId", -1}});
    const json r = generateTile(v, {source(grass, 1.0, 0), source(weed, 1.0, 1)}, "Test", palette);
    const std::vector<NodeId> kids = meshChildren(r["group"]);
    REQUIRE(kids.size() == 1);
    CHECK(slotNames(kids[0]) == std::set<std::string>{"terrainID_0", "terrainID_1", "terrainID_n1"});
    CHECK(groundtiles::idMaterialName(-1) == "terrainID_n1");
    Node* idOne = materials::find(graph(), "terrainID_1");
    REQUIRE(idOne);
    CHECK(idOne->as<glm::dvec3>(material::kBaseColor) == glm::dvec3(0.25, 0.75, 0.125));

    const std::string plane = run("groundTileCreateBakePlane", {{"args", {r["group"]}}});
    const std::string path = (scratch("idmap") / "combined_id.png").string();
    fs::remove(path);
    CHECK(run("groundTileBakeIdMap", {{"args", {r["group"]}}, {"path", path}, {"resolution", 128}}) == path);
    const auto found = pngColours(path);
    for (int64_t id : {0, 1, -1})
        CHECK_MESSAGE(hasColour(found, groundtiles::idColour(id, palette)), id);
    // The ground covers the whole tile, so no texel is left empty.
    CHECK(!found.count({0, 0, 0}));
    // Exact palette values and nothing between them: no filtering.
    CHECK(found.size() <= 3);
    CHECK(QImage(QString::fromStdString(path)).width() == 128);

    const json legend = run("groundTileSettings", {{"args", {r["group"]}}})["legend"];
    CHECK(legend["-1"]["sources"] == json::array({"Ground Plane"}));
    CHECK(legend["-1"]["colour"] == json::array({0.5, 0.25, 0.75}));
    CHECK(legend["1"]["sources"] == json::array({weed}));
    CHECK(legend["0"]["colour"] == json::array({1.0, 0.0, 0.0}));
}

TEST_CASE("ground tiles: ids, masks and the colour set on separate instances")
{
    fresh();
    const std::string grass = blade("Z", "grass"), broken = blade("Z", "broken"), weed = blade("Z", "weed");
    const json palette = {{"1", {0.25, 0.75, 0.125}}};
    json v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 4.0}, {"count", 60}, {"seed", 3}, {"minScale", 3.0}, {"maxScale", 4.0}});
    const json r = generateTile(v, {source(grass, 1.0, 0), source(broken, 1.0, 0), source(weed, 1.0, 1)}, "Test", palette);
    int paired = 0, weeds = 0;
    std::map<std::pair<float, float>, int> seen;
    std::set<float> bands;
    for (NodeId c : meshChildren(r["group"]))
    {
        const std::string name = graph().node(c)->name();
        const geo::Mesh& m = meshOf(c);
        if (name.rfind("weed", 0) == 0)
        {
            ++weeds;
            CHECK(slotNames(c) == std::set<std::string>{"terrainID_1"});
        }
        else
        {
            ++paired;
            CHECK(slotNames(c) == std::set<std::string>{"terrainID_0"});
        }
        CHECK(geo::colors::sets(m) == std::vector<std::string>{"terrainMasks"});
        CHECK(geo::colors::current(m) == "terrainMasks");
        const std::vector<glm::vec4> cols = geo::colors::pointColors(m, "terrainMasks");
        float lo = 1.0f, hi = 0.0f;
        std::set<float> gs, bs;
        // Averaging a point's corners can move the last bit, so the per-instance values are compared rounded.
        auto rounded = [](float v) { return std::round(v * 1e5f) / 1e5f; };
        for (const glm::vec4& c4 : cols)
        {
            lo = std::min(lo, c4.r);
            hi = std::max(hi, c4.r);
            gs.insert(rounded(c4.g));
            bs.insert(rounded(c4.b));
        }
        CHECK(lo == Approx(0.0));
        CHECK(hi == Approx(1.0));
        CHECK(gs.size() == 1);
        CHECK(bs.size() == 1);
        bands.insert(*bs.begin());
        ++seen[{*gs.begin(), *bs.begin()}];
    }
    CHECK(paired > 0);
    CHECK(weeds > 0);
    for (float b : bands)
        CHECK((b == 0.0f || b == 0.5f || b == 1.0f));
    CHECK(bands.size() > 1);
    // Wrapped twins carry the same masks.
    CHECK(std::any_of(seen.begin(), seen.end(), [](const auto& e) { return e.second > 1; }));

    const json legend = run("groundTileSettings", {{"args", {r["group"]}}})["legend"];
    CHECK(legend.size() == 2);
    CHECK(legend["0"]["sources"] == json::array({grass, broken}));
    CHECK(legend["1"]["colour"] == json::array({0.25, 0.75, 0.125}));
    CHECK(run("groundTileSettings", {{"args", {r["group"]}}})["palette"] == palette);

    const std::string plane = run("groundTileCreateBakePlane", {{"args", {r["group"]}}});
    CHECK(run("groundTileSettings", {{"args", {r["group"]}}})["bakePlane"] == plane);
    const std::string path = (scratch("idmap") / "loose_id.png").string();
    const auto found = pngColours(run("groundTileBakeIdMap", {{"args", {r["group"]}}, {"path", path}, {"resolution", 128}}));
    CHECK(hasColour(found, {0.25, 0.75, 0.125}));
    CHECK(hasColour(found, {1.0, 0.0, 0.0}));
}

TEST_CASE("ground tiles: ground plane")
{
    fresh();
    const json palette = {{"-1", {0.5, 0.25, 0.75}}};
    json v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 4.0}, {"count", 40}, {"seed", 5}, {"minScale", 3.0}, {"maxScale", 4.0}, {"groundPlane", true}, {"groundHeight", 0.0},
              {"groundSubdivisions", 4}, {"groundId", -1}});
    const json r = generateTile(v, {source(blade())}, "Test", palette);
    std::vector<NodeId> grounds;
    for (NodeId c : meshChildren(r["group"]))
        if (graph().node(c)->name().find("_Ground") != std::string::npos)
            grounds.push_back(c);
    REQUIRE(grounds.size() == 1);
    CHECK(meshChildren(r["group"]).size() == r["instances"].get<size_t>() + 1);
    const Box gb = worldBox(grounds[0]);
    CHECK(gb.lo.z == Approx(groundtiles::lowestHeight(v)));
    CHECK(gb.hi.x - gb.lo.x == Approx(4.0));
    CHECK(gb.hi.y - gb.lo.y == Approx(4.0));
    CHECK(meshOf(grounds[0]).numFaces() == 16);
    CHECK(slotNames(grounds[0]) == std::set<std::string>{"terrainID_n1"});
    const std::string plane = run("groundTileCreateBakePlane", {{"args", {r["group"]}}});
    CHECK(plane.find("_Ground") == std::string::npos);
    const auto found = pngColours(run("groundTileBakeIdMap", {{"args", {r["group"]}}, {"path", (scratch("idmap") / "ground_id.png").string()}, {"resolution", 128}}));
    CHECK(hasColour(found, {0.5, 0.25, 0.75}));
    CHECK(hasColour(found, {1.0, 0.0, 0.0}));
    CHECK(!found.count({0, 0, 0}));

    fresh();
    v.update({{"groundHeight", -5.0}});
    const json r2 = generateTile(v, {source(blade())}, "Test", palette);
    for (NodeId c : meshChildren(r2["group"]))
        if (graph().node(c)->name().find("_Ground") != std::string::npos)
            CHECK(worldBox(c).lo.z == Approx(groundtiles::lowestHeight(v) - 5.0));
    CHECK(run("groundTileSettings", {{"args", {r2["group"]}}})["metrics"]["groundTiles_minHeight"].get<double>() == Approx(groundtiles::lowestHeight(v) - 5.0));
}

TEST_CASE("ground tiles: height metrics and the measured profile")
{
    fresh();
    json v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 60.0}, {"count", 40}, {"seed", 9}});
    json r = generateTile(v, {source(blade())});
    json m = run("groundTileSettings", {{"args", {r["group"]}}})["metrics"];
    const Box b = worldBox(r["group"].get<std::string>());
    CHECK(m["groundTiles_minHeight"].get<double>() == Approx(b.lo.z));
    CHECK(m["groundTiles_maxHeight"].get<double>() == Approx(b.hi.z));
    CHECK(m["groundTiles_heightRange"].get<double>() == Approx(b.hi.z - b.lo.z));
    CHECK(m["groundTiles_bakePlaneHeight"].get<double>() == Approx(groundtiles::lowestHeight(v)));
    CHECK(m["groundTiles_heightAbovePlane"].get<double>() == Approx(b.hi.z - groundtiles::lowestHeight(v)));
    CHECK(!m.contains("groundTiles_heightP99"));

    fresh();
    v.update({{"upAxis", "Y"}, {"count", 30}});
    r = generateTile(v, {source(blade("Y"))});
    m = run("groundTileSettings", {{"args", {r["group"]}}})["metrics"];
    const Box by = worldBox(r["group"].get<std::string>());
    CHECK(m["groundTiles_minHeight"].get<double>() == Approx(by.lo.y));
    CHECK(m["groundTiles_maxHeight"].get<double>() == Approx(by.hi.y));

    fresh();
    v = preset("Mushed Grass");
    v.update({{"tileSize", 20.0}, {"count", 200}, {"seed", 6}, {"combineResult", true}, {"minScale", 1.0}, {"maxScale", 1.0}, {"layersEnabled", false}, {"baseHeight", 0.0}});
    r = generateTile(v, {source(blade())});
    run("groundTileCreateBakePlane", {{"args", {r["group"]}}});
    const json p = run("groundTileMeasureHeight", {{"args", {r["group"]}}, {"grid", 96}});
    const double p50 = p["groundTiles_heightP50"], p99 = p["groundTiles_heightP99"], top = p["groundTiles_heightOverTile"];
    CHECK(0.0 <= p50);
    CHECK(p50 <= p99);
    CHECK(p99 <= top + 1e-9);
    m = run("groundTileSettings", {{"args", {r["group"]}}})["metrics"];
    CHECK(top <= m["groundTiles_heightAbovePlane"].get<double>() + 1e-3);
    CHECK(m["groundTiles_heightP99"].get<double>() == Approx(p99));
    // Undo drops the stamp again.
    run("undo");
    CHECK(!run("groundTileSettings", {{"args", {r["group"]}}})["metrics"].contains("groundTiles_heightP99"));

    fresh();
    v["combineResult"] = false;
    r = generateTile(v, {source(blade())});
    run("groundTileCreateBakePlane", {{"args", {r["group"]}}});
    CHECK_THROWS(run("groundTileMeasureHeight", {{"args", {r["group"]}}, {"grid", 8}}));
}

TEST_CASE("ground tiles: per source height offset")
{
    fresh();
    const std::string low = blade("Z", "lowSource"), high = blade("Z", "highSource");
    json v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 40.0}, {"count", 60}, {"seed", 11}, {"layersEnabled", false}, {"baseHeight", 0.0}, {"snapBaseToGround", true}});
    const json r = generateTile(v, {source(low, 1.0, 0, 0.0), source(high, 1.0, 0, 5.0)});
    int lows = 0, highs = 0;
    double lowTop = -1e9;
    for (NodeId c : meshChildren(r["group"]))
    {
        const Box b = worldBox(c);
        if (graph().node(c)->name().rfind("lowSource", 0) == 0)
        {
            ++lows;
            CHECK(std::abs(b.lo.z) < 1e-3);
            lowTop = std::max(lowTop, b.hi.z);
        }
        else
        {
            ++highs;
            CHECK(b.lo.z == Approx(5.0).epsilon(1e-4));
        }
    }
    CHECK(lows > 0);
    CHECK(highs > 0);
    CHECK(lowTop < 5.0);
}

TEST_CASE("ground tiles: bake plane")
{
    for (auto [up, tile] : std::vector<std::pair<std::string, double>>{{"Z", 200.0}, {"Z", 7.5}, {"Y", 512.0}})
    {
        fresh();
        json v = groundtiles::defaults();
        v.update({{"upAxis", up}, {"tileSize", tile}});
        const std::string plane = run("groundTileCreateBakePlane", {{"values", v}, {"preset", "Test"}});
        const Box b = worldBox(plane);
        std::vector<double> spans{b.hi.x - b.lo.x, b.hi.y - b.lo.y, b.hi.z - b.lo.z};
        std::sort(spans.begin(), spans.end());
        CHECK(spans[0] == Approx(0.0));
        CHECK(spans[1] == Approx(tile));
        CHECK(spans[2] == Approx(tile));
        const geo::Mesh& m = meshOf(graph().find(plane)->id());
        const geo::Attribute* uv = m.cornerAttrs().find(geo::attr::uvName(0));
        REQUIRE(uv);
        float lo = 1.0f, hi = 0.0f;
        for (const glm::vec2& t : uv->values<glm::vec2>())
        {
            lo = std::min({lo, t.x, t.y});
            hi = std::max({hi, t.x, t.y});
        }
        CHECK(lo == Approx(0.0));
        CHECK(hi == Approx(1.0));
        CHECK(!materials::slotUsage(m).empty());
        CHECK(materials::slotUsage(m).front().first.find("_BakePlaneMtl") != std::string::npos);
    }
    fresh();
    const json pebbles = preset("Pebbles & Gravel");
    CHECK(worldBox(run("groundTileCreateBakePlane", {{"values", pebbles}}).get<std::string>()).lo.z == Approx(pebbles["layer1Height"].get<double>()));

    fresh();
    json v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 6.0}, {"count", 6}});
    const json r = generateTile(v, {source(blade())});
    graph().setValue({graph().find(r["group"].get<std::string>())->id(), Transform::kTranslate, -1}, glm::dvec3(500.0, 0.0, 25.0));
    const Box b = worldBox(run("groundTileCreateBakePlane", {{"args", {r["group"]}}}).get<std::string>());
    const glm::dvec3 centre = (b.lo + b.hi) * 0.5;
    CHECK(centre.x == Approx(500.0));
    CHECK(centre.z == Approx(25.0 + groundtiles::lowestHeight(v)));
}

TEST_CASE("ground tiles: export pair, sidecars and the bake handoff")
{
    fresh();
    json v = preset("Mushed Grass");
    v.update({{"tileSize", 6.0}, {"count", 12}, {"seed", 2}, {"combineResult", true}});
    const fs::path folder = scratch("export");
    for (const char* stale : {"Grass_Hi.fbx", "Grass_Lo.fbx", "Grass_colours.json", "Grass_ids.json"})
        fs::remove(folder / stale);
    const json exp = {{"path", folder.string()}, {"name", "Grass"}, {"material", "Grass"}};
    const json r = generateTile(v, {source(blade())}, "Test", json::object(), exp);
    const std::string grp = r["group"];
    graph().setValue({graph().find(grp)->id(), Transform::kTranslate, -1}, glm::dvec3(12.0, 3.0, 1.5));
    const std::string plane = run("groundTileCreateBakePlane", {{"args", {grp}}});
    const std::string planeMaterial = materials::slotUsage(meshOf(graph().find(plane)->id())).front().first;
    CHECK(planeMaterial.find("_BakePlaneMtl") != std::string::npos);

    const json out = run("groundTileExport", {{"args", {grp}}});
    CHECK(materials::find(graph(), planeMaterial));
    CHECK(!graph().find("Grass"));
    CHECK(fs::file_size(out["high"].get<std::string>()) > 1024);
    CHECK(fs::file_size(out["low"].get<std::string>()) > 0);
    CHECK(fs::path(out["high"].get<std::string>()).filename() == "Grass_Hi.fbx");
    CHECK(fs::path(out["low"].get<std::string>()).filename() == "Grass_Lo.fbx");
    CHECK(run("groundTileSettings", {{"args", {grp}}})["bakePlane"] == plane);
    CHECK(worldBox(plane).lo.z == Approx(1.5 + groundtiles::lowestHeight(v)));
    CHECK(run("groundTileSettings", {{"args", {grp}}})["export"] == exp);

    const json ids = readJson(folder / "Grass_ids.json");
    for (const char* key : {"tile", "preset", "tileSize", "upAxis", "minHeight", "maxHeight", "bakePlaneHeight", "heightAbovePlane", "ids"})
        CHECK_MESSAGE(ids.contains(key), key);
    CHECK(ids["tile"] == "Grass");
    CHECK(ids["tileSize"] == Approx(6.0));
    CHECK(ids["heightAbovePlane"].is_number());
    CHECK(ids["ids"].size() == 1);
    CHECK(ids["ids"]["0"]["colour"].size() == 3);
    CHECK(ids["ids"]["0"]["sources"].is_array());

    const fs::path coloursPath = folder / "Grass_colours.json";
    json colours = readJson(coloursPath);
    CHECK(colours.size() == ids["ids"].size());
    for (auto& [key, entry] : colours.items())
    {
        std::set<std::string> fields;
        for (auto& [f, _] : entry.items())
            fields.insert(f);
        CHECK(fields == std::set<std::string>{"base", "root", "occlusion", "perInstance", "tip", "roughness"});
        CHECK(entry["base"].get<std::string>().size() == 7);
        CHECK(entry["base"].get<std::string>()[0] == '#');
        CHECK(entry["roughness"].is_number_float());
    }
    CHECK(colours["0"]["base"] == "#708a3d");
    CHECK(groundtiles::toHex({1.0, 0.0, 0.0}) == "#ff0000");
    CHECK(groundtiles::toHex({0.30, 0.42, 0.16}) == "#4c6b29");
    // Hand edits survive every later export.
    colours["0"]["base"] = "#123456";
    std::ofstream(coloursPath) << colours.dump(4);
    run("groundTileExport", {{"args", {grp}}});
    CHECK(readJson(coloursPath)["0"]["base"] == "#123456");
    const nlohmann::ordered_json seeded = groundtiles::defaultColourRecipes({{"-1", json::object()}, {"2", json::object()}});
    CHECK(seeded["-1"]["roughness"] == 0.70);
    CHECK(seeded["2"]["roughness"] == 0.45);
    CHECK(seeded.begin().key() == "-1");

    const json handoff = run("groundTileBakeData", {{"args", {grp}}});
    CHECK(handoff["high"] == out["high"]);
    CHECK(handoff["low"] == out["low"]);
    CHECK(handoff["bakePlane"] == plane);
    CHECK(handoff["bakeMaterial"] == planeMaterial);
    CHECK(handoff["legend"].contains("0"));
    CHECK(handoff["tileSize"] == Approx(6.0));

    // The high poly keeps its masks, ID materials and world placement through the file.
    const Box before = worldBox(meshChildren(grp)[0]);
    run("newScene");
    run("importFbx", {{"args", {out["high"]}}});
    bool masks = false, idMaterial = false;
    Box after;
    graph().evaluate();
    std::vector<NodeId> meshes;
    for (Node* n : [] {
             std::vector<Node*> all;
             for (const std::string& name : run("ls", {{"type", "mesh"}}).get<std::vector<std::string>>())
                 all.push_back(graph().find(name));
             return all;
         }())
    {
        const geo::Mesh& m = meshOf(n->id());
        masks = masks || geo::colors::has(m, "terrainMasks");
        for (const auto& [slot, count] : materials::slotUsage(m))
            idMaterial = idMaterial || slot.rfind("terrainID_", 0) == 0;
        const Box b = worldBox(n->id());
        after.lo = glm::min(after.lo, b.lo);
        after.hi = glm::max(after.hi, b.hi);
        meshes.push_back(n->id());
    }
    CHECK(!meshes.empty());
    CHECK(masks);
    CHECK(idMaterial);
    CHECK(glm::length(after.lo - before.lo) < 1e-3);

    run("newScene");
    run("importFbx", {{"args", {out["low"]}}});
    const std::vector<std::string> low = run("ls", {{"type", "mesh"}}).get<std::vector<std::string>>();
    REQUIRE(low.size() == 1);
    CHECK(materials::slotUsage(meshOf(graph().find(low[0])->id())).front().first == "Grass");

    // A name clash aborts before anything is renamed.
    fresh();
    const json r2 = generateTile(v, {source(blade())}, "Test", json::object(), exp);
    run("groundTileCreateBakePlane", {{"args", {r2["group"]}}});
    box("Grass", {1, 1, 1}, 1);
    CHECK_THROWS(run("groundTileExport", {{"args", {r2["group"]}}}));
    CHECK(graph().find("Grass"));
    CHECK(run("groundTileSettings", {{"args", {r2["group"]}}}).contains("bakePlane"));
    CHECK_THROWS(run("groundTileExport", {{"args", {r2["group"]}}, {"name", ""}}));
}

TEST_CASE("ground tiles: vertex colour mask split")
{
    ensureLoaded();
    const fs::path folder = scratch("split");
    for (const auto& e : fs::directory_iterator(folder))
        fs::remove(e.path());
    QImage made(8, 8, QImage::Format_RGB32);
    for (int i = 0; i < 64; ++i)
        made.setPixel(i % 8, i / 8, qRgb(i * 4 % 256, 90, 180));
    REQUIRE(made.save(QString::fromStdString((folder / "MushedGrass_vertexcolor.png").string()), "PNG"));
    const json found = run("groundTileFindVertexMap", {{"args", {folder.string()}}, {"name", "MushedGrass"}});
    REQUIRE(found.is_string());
    CHECK(fs::path(found.get<std::string>()).filename() == "MushedGrass_vertexcolor.png");
    const json written = run("groundTileSplitMasks", {{"args", {found}}});
    REQUIRE(written.size() == 3);
    CHECK(fs::path(written[0].get<std::string>()).filename() == "MushedGrass_maskR.png");
    CHECK(fs::path(written[1].get<std::string>()).filename() == "MushedGrass_maskG.png");
    CHECK(fs::path(written[2].get<std::string>()).filename() == "MushedGrass_maskB.png");
    const QImage r(QString::fromStdString(written[0].get<std::string>())), g(QString::fromStdString(written[1].get<std::string>())),
        b(QString::fromStdString(written[2].get<std::string>()));
    CHECK(qRed(r.pixel(1, 0)) == 4);
    CHECK(qGreen(r.pixel(2, 0)) == 8);
    CHECK(qRed(g.pixel(3, 3)) == 90);
    CHECK(qBlue(b.pixel(5, 6)) == 180);
    CHECK_THROWS(run("groundTileSplitMasks", {{"args", {(folder / "no.png").string()}}}));
    CHECK(run("groundTileFindVertexMap", {{"args", {folder.string() + "_nope"}}}).is_null());
}

TEST_CASE("ground tiles: shared meshes, weighting and disabled sources")
{
    fresh();
    json v = loose(preset("Pebbles & Gravel"));
    v.update({{"tileSize", 30.0}, {"count", 10}, {"minSpacing", 2.0}, {"useInstances", true}});
    for (const char* f : {"bendEnabled", "flareEnabled", "twistEnabled", "sineEnabled", "latticeEnabled", "jitterEnabled", "flattenEnabled", "vertexColours"})
        v[f] = false;
    json r = generateTile(v, {source(blade())}, "Pebbles");
    std::vector<NodeId> kids = meshChildren(r["group"]);
    REQUIRE(kids.size() > 1);
    CHECK(graph().node(kids[0])->as<MeshPtr>(MeshNode::kInMesh) == graph().node(kids[1])->as<MeshPtr>(MeshNode::kInMesh));

    fresh();
    const std::string first = blade("Z", "first"), second = blade("Z", "second");
    v = loose(preset("Mushed Grass"));
    v.update({{"tileSize", 30.0}, {"count", 60}, {"bendEnabled", false}, {"flareEnabled", false}});
    r = generateTile(v, {source(first, 1.0), source(second, 0.0)});
    REQUIRE(r["group"].is_string());
    for (NodeId c : meshChildren(r["group"]))
        CHECK(graph().node(c)->name().rfind("second", 0) != 0);

    fresh();
    r = generateTile(preset("Mushed Grass"), {source(blade(), 1.0, 0, 0.0, false)});
    CHECK(r["group"].is_null());
    CHECK(!r["warnings"].empty());
}

TEST_CASE("ground tiles: regenerate, delete and undo")
{
    fresh();
    json v = preset("Mushed Grass");
    v.update({{"tileSize", 10.0}, {"count", 20}});
    const json r = generateTile(v, {source(blade())}, "Mushed Grass");
    const std::string grp = r["group"];
    CHECK(grp.find("Mushed_Grass_GroundTile_GRP") == 0);
    const json again = run("groundTileRegenerate", {{"args", {grp}}, {"values", {{"seed", 4}}}});
    REQUIRE(again["group"].is_string());
    CHECK(run("groundTileSettings", {{"args", {again["group"]}}})["values"]["seed"] == 4);
    CHECK(run("groundTileSettings", {{"args", {again["group"]}}})["values"]["count"] == 20);
    run("groundTileDelete", {{"args", {again["group"]}}});
    CHECK(!graph().find(again["group"].get<std::string>()));
    run("undo");
    CHECK(graph().find(again["group"].get<std::string>()));
    // Selecting anything inside the group finds it.
    graph().select({meshChildren(again["group"])[0]});
    CHECK(run("groundTileSettings")["group"] == again["group"]);
}

TEST_CASE("ground tiles: texture set built natively from the palette and masks")
{
    fresh();
    const std::string a = blade("Z", "bladeA"), b = blade("Z", "bladeB");
    json v = preset("Dry Grass");
    v.update({{"tileSize", 20.0}, {"count", 40}});
    const json r = generateTile(v, {source(a, 1.0, 0), source(b, 1.0, 3)}, "Dry Grass");
    REQUIRE(r["group"].is_string());
    const json t = run("groundTileTextureSet", {{"args", {r["group"]}}, {"bake", false}, {"name", "GrassTex"}});
    CHECK(t["set"] == "GrassTex");
    CHECK(t["ids"] == 2);
    CHECK(t["bakeProject"].is_null());

    const json stack = run("texLayerStack", {{"args", {"GrassTex"}}});
    int folders = 0, idSelects = 0, vertexMasks = 0;
    std::function<void(const json&)> walk = [&](const json& layers) {
        for (const json& l : layers)
        {
            if (l.value("kind", std::string()) == "folder")
                ++folders;
            for (const json& m : l.value("masks", nlohmann::json::array()))
            {
                if (m.value("type", std::string()) == "genIdSelect")
                    ++idSelects;
                if (m.value("bindings", nlohmann::json::object()).value("position", std::string()) == "vertexColor")
                    ++vertexMasks;
            }
            walk(l.value("children", nlohmann::json::array()));
        }
    };
    walk(stack["layers"]);
    CHECK(folders == 2);
    CHECK(idSelects == 2);
    CHECK(vertexMasks == 6);

    const fs::path out = fs::temp_directory_path() / "lila_groundtiles_tests" / "tex";
    fs::remove_all(out);
    const json e = run("texEvaluate", {{"args", {"GrassTex"}}, {"resolution", 32}, {"backend", "cpu"}, {"folder", out.string()},
                                       {"channels", {"baseColor", "roughness", "height"}}});
    CHECK(e["backend"] == "cpu");
    REQUIRE(e["files"].size() == 3);
    for (const json& f : e["files"])
        CHECK(fs::exists(fs::path(f.get<std::string>())));
    CHECK(QImage(QString::fromStdString(e["files"][0].get<std::string>())).width() == 32);
}
