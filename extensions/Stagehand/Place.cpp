// Library assets in the scene: each one loaded once as a hidden master under StagehandLibrary, and placed as instances
// (or copies) of it on surfaces. Plus the library commands: librarySearch, libraryAdd, libraryRemove, libraryReload,
// libraryPaths, place, replaceWith and libraryRefresh.

#include "Kit.h"
#include "Library.h"
#include "Stagehand.h"

#include <lila/Command.h>
#include <lila/Extension.h>
#include <lila/Log.h>
#include <lila/Mesh.h>
#include <lila/Prefs.h>
#include <lila/Profile.h>
#include <lila/Relocate.h>
#include <lila/Transform.h>
#include <lila/Undo.h>
#include <lila/Workspace.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <random>
#include <set>

namespace fs = std::filesystem;
using namespace lila;

namespace stagehand
{
using T = ArgType;

namespace
{
using library::Asset;

constexpr const char* kMasters = "StagehandLibrary";
/** Stagehand's own node types under a mesh or group: data, not geometry. */
const std::set<std::string> kTagTypes = {kBodyType, "stagehandSnap", "stagehandAsset", "stagehandAssembly"};
constexpr const char* kAssetType = "stagehandAsset";

glm::dvec3 vecFlag(const Args& a, const char* flag, const glm::dvec3& fallback, const char* cmd)
{
    if (!a.has(flag))
        return fallback;
    const json& j = a.flags[flag];
    if (!j.is_array() || j.size() != 3)
        throw CommandError(std::string(cmd) + ": " + flag + " takes [x, y, z]");
    return {j[0].get<double>(), j[1].get<double>(), j[2].get<double>()};
}

/** The model file's modification time, which tells a master it's stale. */
std::string stamp(const fs::path& p)
{
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    return ec ? std::string() : std::format("{}", t.time_since_epoch().count());
}

Asset need(const std::string& id, const char* cmd)
{
    std::optional<Asset> a = library::find(id);
    if (!a)
        throw CommandError(std::string(cmd) + ": no library asset '" + id + "' (librarySearch lists them)");
    return *a;
}

/** The node carrying an asset id under a placed or master root, or null. */
Node* assetTag(Graph& g, NodeId root)
{
    for (NodeId c : g.children(root))
        if (Node* n = g.node(c); n && n->type().name == kAssetType)
            return n;
    return nullptr;
}

std::string tagValue(Node& tag, const char* attr) { return tag.as<std::string>(tag.type().attrIndex(attr)); }

/** Every mesh at or under a node. */
std::vector<NodeId> meshesUnder(Graph& g, NodeId root)
{
    std::vector<NodeId> out, stack{root};
    while (!stack.empty())
    {
        const NodeId id = stack.back();
        stack.pop_back();
        if (Node* n = g.node(id); n && n->type().isA("mesh"))
            out.push_back(id);
        for (NodeId c : g.children(id))
            stack.push_back(c);
    }
    return out;
}

/** Bounds of every mesh under root, in root's own space. False when there are none. */
bool localBounds(Graph& g, NodeId root, glm::dvec3& lo, glm::dvec3& hi)
{
    g.evaluate();
    Node* r = g.node(root);
    if (!r)
        return false;
    const glm::dmat4 toRoot = glm::inverse(r->as<glm::dmat4>(scene::Transform::kWorldMatrix));
    lo = glm::dvec3(1e300);
    hi = glm::dvec3(-1e300);
    bool any = false;
    for (NodeId id : meshesUnder(g, root))
    {
        Node* n = g.node(id);
        const auto mesh = n->as<scene::MeshPtr>(n->type().attrIndex("outMesh"));
        if (!mesh)
            continue;
        const glm::dmat4 m = toRoot * n->as<glm::dmat4>(scene::Transform::kWorldMatrix);
        for (geo::Index p = 0; p < mesh->pointCapacity(); ++p)
            if (mesh->pointValid(p))
            {
                const glm::dvec3 q(m * glm::dvec4(mesh->position(p), 1.0));
                lo = glm::min(lo, q);
                hi = glm::max(hi, q);
                any = true;
            }
    }
    return any;
}

NodeId mastersGroup(Graph& g)
{
    if (Node* n = g.find(kMasters))
        return n->id();
    const json made = ops::execute("group", {{"empty", true}, {"name", kMasters}});
    const std::string path = made.is_array() ? made[0].get<std::string>() : made.get<std::string>();
    ops::execute("setAttr", {{"args", {path + ".visibility", false}}});
    return g.find(path)->id();
}

NodeId findMaster(Graph& g, const std::string& id)
{
    Node* group = g.find(kMasters);
    if (!group)
        return scene::kNoNode;
    for (NodeId c : g.children(group->id()))
        if (Node* tag = assetTag(g, c); tag && tagValue(*tag, "asset") == id)
            return c;
    return scene::kNoNode;
}

/** New top-level nodes since `before`. */
std::vector<NodeId> newRoots(Graph& g, const std::set<NodeId>& before)
{
    std::vector<NodeId> out;
    for (NodeId id : g.nodes())
        if (!before.count(id) && g.parent(id) == scene::kNoNode && g.node(id)->type().dag)
            out.push_back(id);
    return out;
}

void applyPhysics(Graph& g, NodeId root, const json& physics)
{
    if (!physics.is_object() || physics.empty())
        return;
    json names = json::array();
    for (NodeId m : meshesUnder(g, root))
        names.push_back(g.node(m)->path());
    if (names.empty())
        return;
    json call = {{"args", names}};
    for (const char* k : {"role", "material", "shape"})
        if (physics.contains(k) && physics[k].is_string())
            call[k] = physics[k];
    if (physics.contains("maxParts") && physics["maxParts"].is_number_integer())
        call["maxParts"] = physics["maxParts"];
    ops::execute("physicsBody", call);
}

/**
 * Loads an asset's model under a new hidden master root. glTF, FBX and OBJ import undoably; a .lila model comes in
 * through a file reference, which clears undo.
 */
NodeId loadMaster(Graph& g, const Asset& a)
{
    LILA_ZONE();

    const NodeId group = mastersGroup(g);
    const json made = ops::execute("group", {{"empty", true}, {"name", identifier(a.name, "Asset")}, {"parent", g.node(group)->path()}});
    const NodeId root = g.find(made.is_array() ? made[0].get<std::string>() : made.get<std::string>())->id();
    // The model's own nodes may share the root's name, so its path is looked up fresh each time.
    auto rootPath = [&] { return g.node(root)->path(); };

    std::set<NodeId> before;
    for (NodeId id : g.nodes())
        before.insert(id);
    const std::string file = a.model.generic_string();
    std::string ext = a.model.extension().string();
    for (char& c : ext)
        c = char(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".lila" || ext == ".lilaa")
    {
        log::write(LogLevel::Warning, "Library: " + a.id + " is a .lila scene; loading it clears undo (store props as .glb to keep it)");
        const json ns = ops::execute("createReference", {{"args", {file}}, {"namespace", identifier(a.name, "Asset") + "_lib"}});
        ops::execute("importReference", {{"args", {ns}}});
    }
    else if (ext == ".glb" || ext == ".gltf")
        ops::execute("importFile", {{"args", {file}}, {"options", {{"select", false}}}});
    else
        ops::execute("importFile", {{"args", {file}}});
    const std::vector<NodeId> roots = newRoots(g, before);
    if (roots.empty())
        throw CommandError("place: " + a.id + "'s model " + file + " brought in nothing");
    json args = json::array();
    for (NodeId r : roots)
        args.push_back(g.node(r)->path());
    args.push_back(rootPath());
    ops::execute("parent", {{"args", args}});

    const json tag = ops::execute("createNode", {{"args", {kAssetType}}, {"name", "asset"}, {"parent", rootPath()}, {"skipSelect", true}});
    const std::string tagPath = tag.get<std::string>();
    ops::execute("setAttr", {{"args", {tagPath + ".asset", a.id}}});
    ops::execute("setAttr", {{"args", {tagPath + ".file", file}}});
    ops::execute("setAttr", {{"args", {tagPath + ".stamp", stamp(a.model)}}});
    applyPhysics(g, root, a.physics);
    // Snap points come back as stagehandSnap locators where they were when the asset was saved.
    for (const json& sp : a.snapPoints)
    {
        if (!sp.is_object() || !sp.contains("matrix") || !sp["matrix"].is_array() || sp["matrix"].size() != 16)
            continue;
        const json snap = ops::execute("createNode", {{"args", {"stagehandSnap"}}, {"name", sp.value("name", std::string("snap"))},
                                                      {"parent", rootPath()}, {"skipSelect", true}});
        const std::string path = snap.get<std::string>();
        glm::dmat4 m;
        for (int k = 0; k < 16; ++k)
            m[k / 4][k % 4] = sp["matrix"][size_t(k)].get<double>();
        scene::Transform::setLocalMatrix(g, g.find(path)->id(), m);
        ops::execute("setAttr", {{"args", {path + ".snapType", sp.value("type", std::string("joint"))}}});
        ops::execute("setAttr", {{"args", {path + ".accepts", sp.value("accepts", json::array())}}});
    }
    return root;
}

NodeId ensureMaster(Graph& g, const Asset& a)
{
    const NodeId m = findMaster(g, a.id);
    return m != scene::kNoNode ? m : loadMaster(g, a);
}

/** Rotation taking +Y to n. */
glm::dquat alignUp(const glm::dvec3& n)
{
    const glm::dvec3 up(0, 1, 0);
    const glm::dvec3 to = glm::normalize(n);
    const double d = glm::dot(up, to);
    if (d > 1.0 - 1e-9)
        return glm::dquat(1, 0, 0, 0);
    if (d < -1.0 + 1e-9)
        return glm::angleAxis(3.14159265358979323846, glm::dvec3(1, 0, 0));
    return glm::normalize(glm::dquat(1.0 + d, glm::cross(up, to)));
}

/** The first visible mesh surface straight down from above point (excluding some), as hit point and normal. */
std::optional<physics::RayHit> surfaceBelow(Graph& g, const glm::dvec3& point, const std::set<NodeId>& skip, NodeId only = scene::kNoNode,
                                           double above = 1e5)
{
    LILA_ZONE();

    g.evaluate();
    physics::Settings settings;
    settings.origin = point;
    physics::RigidScene s(settings);
    std::vector<NodeId> bodies;
    for (NodeId id : only != scene::kNoNode ? std::vector<NodeId>{only} : g.nodes())
    {
        if (skip.count(id) || !physical(g, id))
            continue;
        Node* n = g.node(id);
        const auto mesh = n->as<scene::MeshPtr>(n->type().attrIndex("outMesh"));
        if (!mesh)
            continue;
        // Only meshes whose footprint is under the point.
        const glm::dmat4 w = n->as<glm::dmat4>(scene::Transform::kWorldMatrix);
        glm::dvec3 lo, hi;
        if (!worldBoundsOf(g, id, lo, hi) || point.x < lo.x || point.x > hi.x || point.z < lo.z || point.z > hi.z)
            continue;
        physics::BodyDesc b;
        b.role = physics::Role::Static;
        b.shape = shapeOf(g, id, ShapeChoice::Mesh, 0.0, true);
        const Frame f = frameOf(w);
        b.position = f.position;
        b.rotation = f.rotation;
        s.add(b);
        bodies.push_back(id);
    }
    if (bodies.empty())
        return std::nullopt;
    return s.raycast(point + glm::dvec3(0, above, 0), {0, -1, 0}, above + 1e5);
}

struct Placement
{
    glm::dvec3 at{0.0};
    glm::dvec3 normal{0, 1, 0};
    double rotate = 0.0; // degrees about the normal
    glm::dvec3 scale{1.0};
    std::string mode = "instance";
    std::string name;
};

json matrixJson(const glm::dmat4& w)
{
    json m = json::array();
    for (int k = 0; k < 16; ++k)
        m.push_back(w[k / 4][k % 4]);
    return m;
}

/** Column-major 16 numbers as a matrix; identity for anything else. */
glm::dmat4 matrixFrom(const json& j)
{
    glm::dmat4 m(1.0);
    if (j.is_array() && j.size() == 16)
        for (int k = 0; k < 16; ++k)
            m[k / 4][k % 4] = j[size_t(k)].get<double>();
    return m;
}

/** A copy of a plain asset's master, top level and shown, where the master stands. */
NodeId copyOfMaster(Graph& g, const Asset& a, const std::string& mode)
{
    const NodeId master = ensureMaster(g, a);
    json made;
    const std::string masterPath = g.node(master)->path();
    if (mode == "copy")
        made = ops::execute("duplicate", {{"args", {masterPath}}});
    else if (mode == "instance")
        made = ops::execute("instance", {{"args", {masterPath}}});
    else
        throw CommandError("place: mode is instance or copy, not '" + mode + "'");
    Node* root = g.find(made.is_array() ? made[0].get<std::string>() : made.get<std::string>());
    if (!root)
        throw CommandError("place: lost the placed copy of " + a.id);
    // Unparenting may rename it to stay unique at the top, so it is kept by id.
    const NodeId id = root->id();
    ops::execute("parent", {{"args", {root->fullPath()}}, {"world", true}});
    ops::execute("setAttr", {{"args", {g.node(id)->fullPath() + ".visibility", true}}});
    return id;
}

/**
 * A copy of an asset, top level: a plain asset's master copied, or an assembly's group with each member (assemblies
 * too) copied in where it stood, and the assembly's own model beside them.
 */
NodeId instantiate(Graph& g, const Asset& a, const std::string& mode, int depth = 0)
{
    if (a.assembly.is_null())
        return copyOfMaster(g, a, mode);
    if (depth > 16)
        throw CommandError("place: " + a.id + " nests assemblies too deep (does it hold itself?)");
    const json made = ops::execute("group", {{"empty", true}, {"name", identifier(a.name, "Asset")}});
    const NodeId root = g.find(made.is_array() ? made[0].get<std::string>() : made.get<std::string>())->id();
    tagAssembly(g, root, a.assembly.value("kind", std::string("levelInstance")));
    const json tag = ops::execute("createNode", {{"args", {kAssetType}}, {"name", "asset"}, {"parent", g.node(root)->fullPath()}, {"skipSelect", true}});
    ops::execute("setAttr", {{"args", {tag.get<std::string>() + ".asset", a.id}}});
    auto adopt = [&](NodeId child, const glm::dmat4& local) {
        setWorld(g, child, local);
        ops::execute("parent", {{"args", {g.node(child)->fullPath(), g.node(root)->fullPath()}}});
    };
    if (!a.model.empty())
        adopt(copyOfMaster(g, a, mode), matrixFrom(a.assembly.value("modelMatrix", json())) * g.node(ensureMaster(g, a))->as<glm::dmat4>(scene::Transform::kWorldMatrix));
    for (const json& m : a.assembly.value("members", json::array()))
    {
        const std::string id = m.value("asset", std::string());
        std::optional<Asset> member = library::find(id);
        if (!member)
            throw CommandError("place: " + a.id + " holds " + id + ", which isn't in the library");
        adopt(instantiate(g, *member, mode, depth + 1), matrixFrom(m.value("matrix", json())));
    }
    return root;
}

/** Puts one copy of an asset at a placement, its bounds' bottom centre on the point. Returns its root. */
NodeId placeOne(Graph& g, const Asset& a, const Placement& p)
{
    LILA_ZONE();

    const NodeId id = instantiate(g, a, p.mode);
    glm::dvec3 lo, hi;
    if (!localBounds(g, id, lo, hi))
        throw CommandError("place: " + a.id + " has no meshes");
    const glm::dvec3 base((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    Node* root = g.node(id);
    if (!p.name.empty())
        ops::execute("rename", {{"args", {root->path(), p.name}}});

    const glm::dquat turn = alignUp(p.normal) * glm::angleAxis(glm::radians(p.rotate), glm::dvec3(0, 1, 0));
    glm::dmat4 world = glm::translate(glm::dmat4(1.0), p.at) * glm::mat4_cast(turn) * glm::scale(glm::dmat4(1.0), p.scale) *
                       glm::translate(glm::dmat4(1.0), -base);
    setWorld(g, id, world);
    return id;
}

/** The asset a placed root came from, or empty. */
std::string assetOf(Graph& g, NodeId root)
{
    Node* tag = assetTag(g, root);
    return tag ? tagValue(*tag, "asset") : std::string();
}

json placedJson(Graph& g, NodeId root, const Asset& a, const std::string& mode)
{
    json meshes = json::array();
    for (NodeId m : meshesUnder(g, root))
        meshes.push_back(g.node(m)->path());
    g.evaluate();
    const glm::dmat4 w = g.node(root)->as<glm::dmat4>(scene::Transform::kWorldMatrix);
    return {{"node", g.node(root)->path()}, {"asset", a.id}, {"mode", mode}, {"position", v3(glm::dvec3(w[3]))}, {"meshes", meshes}};
}

/**
 * Renders thumb.png for starter props that have none (or all, with force): each model is brought into the scene,
 * captured alone and taken out again inside an undo chunk that is then cancelled, so neither the scene nor undo keeps
 * a trace. Needs captureView, which the app and batch mode have.
 */
json renderStarterThumbnails(Graph& g, bool force)
{
    LILA_ZONE();

    std::vector<Asset> todo;
    for (const Asset& a : library::all())
        if (a.tier == "starter" && !a.model.empty() && (force || a.thumbnail.empty()))
            todo.push_back(a);
    json out = {{"thumbnails", 0}, {"missing", int64_t(todo.size())}};
    if (todo.empty())
        return out;
    if (!ops::registry().find("captureView"))
    {
        out["thumbnailNote"] = "no renderer here: thumbnails are drawn by the app or batch mode";
        return out;
    }
    UndoStack& undo = UndoStack::get();
    if (undo.chunkOpen())
    {
        out["thumbnailNote"] = "skipped inside another command";
        return out;
    }
    const std::vector<NodeId> previous = g.selection();
    int64_t rendered = 0;
    undo.openChunk("Starter thumbnails");
    try
    {
        for (const Asset& a : todo)
        {
            std::set<NodeId> before;
            for (NodeId id : g.nodes())
                before.insert(id);
            ops::execute("importFile", {{"args", {a.model.generic_string()}}});
            const std::vector<NodeId> roots = newRoots(g, before);
            json names = json::array();
            for (NodeId r : roots)
                names.push_back(g.node(r)->path());
            if (names.empty())
                continue;
            ops::execute("select", {{"args", names}});
            ops::execute("captureView", {{"path", (a.dir / "thumb.png").generic_string()},
                                         {"view", "three-quarter"},
                                         {"width", 256},
                                         {"height", 256},
                                         {"display", "shaded"},
                                         {"isolate", true},
                                         {"frameSelection", true},
                                         {"grid", false},
                                         {"showSelection", false}});
            ops::execute("delete", {{"args", names}});
            ++rendered;
        }
    }
    catch (const std::exception& e)
    {
        out["thumbnailNote"] = std::string("stopped: ") + e.what();
    }
    undo.cancelChunk();
    std::vector<NodeId> keep;
    for (NodeId id : previous)
        if (g.node(id))
            keep.push_back(id);
    g.select(keep);
    library::reload();
    out["thumbnails"] = rendered;
    out["missing"] = int64_t(todo.size()) - rendered;
    return out;
}

library::Query queryOf(const Args& a)
{
    library::Query q;
    if (!a.positional.empty() && a.positional[0].is_string())
        q.text = a.positional[0].get<std::string>();
    if (a.has("tags"))
        for (const json& t : a.flags["tags"])
            q.tags.push_back(t.get<std::string>());
    q.collection = a.get<std::string>("collection", "");
    q.tier = a.get<std::string>("tier", "");
    q.maxSize = a.get<double>("maxSize", 0.0);
    q.minSize = a.get<double>("minSize", 0.0);
    q.limit = size_t(std::max<int64_t>(0, a.get<int64_t>("limit", 50)));
    return q;
}
} // namespace

NodeId libraryMaster(Graph& g, const std::string& id)
{
    const std::optional<Asset> a = library::find(id);
    if (!a)
        throw CommandError("no library asset '" + id + "' (librarySearch lists them)");
    return ensureMaster(g, *a);
}

std::vector<NodeId> meshesOf(Graph& g, NodeId root) { return meshesUnder(g, root); }

std::optional<physics::RayHit> surfaceUnder(Graph& g, const glm::dvec3& point, const std::set<NodeId>& skip, double above)
{
    return surfaceBelow(g, point, skip, scene::kNoNode, above);
}

bool boundsUnder(Graph& g, NodeId root, glm::dvec3& lo, glm::dvec3& hi) { return localBounds(g, root, lo, hi); }

bool registerLibraryNodes(ext::Context& ctx)
{
    const scene::NodeType* base = scene::types::find("transform");
    if (!base)
        return false;
    scene::NodeType t;
    t.name = kAssetType;
    t.base = base;
    t.dag = true;
    t.icon = "books";
    t.doc = "Which library asset the object it sits under was placed from, so it can be refreshed, replaced and counted.";
    t.ownAttrs = {
        {.name = "asset", .type = scene::PlugType::String, .defaultValue = std::string(), .doc = "The asset's id, <Collection>/<Asset>."},
        {.name = "file", .type = scene::PlugType::String, .defaultValue = std::string(), .flags = scene::AttrInput | scene::AttrStorable | scene::AttrFile,
         .doc = "The model it was loaded from."},
        {.name = "stamp", .type = scene::PlugType::String, .defaultValue = std::string(), .flags = scene::AttrInput | scene::AttrStorable | scene::AttrHidden,
         .doc = "The model's modification time when loaded."},
    };
    t.create = base->create;
    lila::scene::relocate::registerNeutral(t.name);
    return scene::types::registerType(ctx, std::move(t)) != nullptr;
}

void registerLibraryCommands(ext::Context& ctx)
{
    const std::vector<FlagDef> queryFlags = {
        {"tags", "t", T::List, nullptr, "Only assets with every one of these tags."},
        {"collection", "c", T::String, nullptr, "Only this collection or the ones under it."},
        {"tier", "", T::String, nullptr, "workspace, studio, user or starter."},
        {"maxSize", "mx", T::Float, nullptr, "Largest side at most this, cm.", {}, {}, 0.0},
        {"minSize", "mn", T::Float, nullptr, "Largest side at least this, cm.", {}, {}, 0.0},
        {"limit", "l", T::Int, int64_t(50), "At most this many; 0 for all.", {}, {}, 0.0},
    };
    ctx.registerCommand({
        .name = "librarySearch",
        .doc = "Find props in the library by words, tags, collection and size. Best match first.\n\n"
               "Every word must match the name, tags, collection, id or description (prefixes and one-letter typos count). "
               "Each result has id, name, collection, tier, model, thumbnail, tags, description, size [x, y, z] cm, physics, "
               "unreal and snapPoints. Pass an id to place().",
        .positionalDoc = "words (optional)",
        .flags = queryFlags,
        .undoable = false,
        .run = [](const Args& a) -> json {
            json out = json::array();
            for (const Asset& x : library::search(queryOf(a)))
                out.push_back(x.toJson());
            return out;
        },
        .examples = {"librarySearch('wooden crate')", "librarySearch(tags=['book'], maxSize=40)"},
    });

    ctx.registerCommand({
        .name = "libraryPaths",
        .doc = "The library folders searched, in the order they win, with collections, tags and load errors.\n\n"
               "Workspace <workspace>/Library, then studio folders (pref stagehand /library/paths and env LILA_LIBRARY_PATH), then the "
               "user's. add= appends a studio folder to the pref.",
        .flags = {{"add", "a", T::String, nullptr, "A studio library folder to add."}},
        .undoable = false,
        .run = [](const Args& a) -> json {
            if (a.has("add"))
            {
                json paths = PrefStore::get().value("stagehand", "/library/paths", json::array());
                if (!paths.is_array())
                    paths = json::array();
                paths.push_back(a.get<std::string>("add"));
                PrefStore::get().set("stagehand", "/library/paths", paths);
                library::reload();
            }
            json roots = json::array();
            for (const library::Root& r : library::roots(false))
                roots.push_back({{"tier", r.tier}, {"dir", r.dir.generic_string()}, {"exists", fs::is_directory(r.dir)}});
            return {{"roots", roots}, {"collections", library::collections()}, {"tags", library::tags()}, {"assets", library::all().size()},
                    {"errors", library::errors()}};
        },
        .examples = {"libraryPaths(add='P:/Studio/PropLibrary')"},
    });

    ctx.registerCommand({
        .name = "libraryReload",
        .doc = "Scan the library folders again (after files changed outside Lila).",
        .undoable = false,
        .run = [](const Args&) -> json {
            library::reload();
            return int64_t(library::all().size());
        },
    });

    ctx.registerCommand({
        .name = "libraryStarter",
        .doc = "The starter props, built and given thumbnails: industrial (crates, barrels, pallets, beams, I-beams, pipes, panels, cable spools, scaffolding), "
               "debris (stone chunks S to XL, a rubble pile, a broken slab, rebar) and facility (lockers, consoles, supply crates, "
               "cable trays), each with physics and tags. Built from code into the user's config folder the first time the "
               "library is used; rebuild=True writes them again. Thumbnails are rendered where the app or batch mode can draw. "
               "Returns dir, built, assets (ids for place), thumbnails (rendered now) and missing.",
        .flags = {{"rebuild", "rb", T::Bool, false, "Write the props again even if they are there."},
                  {"thumbnails", "th", T::Bool, true, "Render the thumbnails still missing (all of them after a rebuild)."}},
        .undoable = false,
        .run = [](const Args& a) -> json {
            if (!library::starterEnabled())
                throw CommandError("libraryStarter: the starter library is off (pref stagehand /library/starter)");
            json out = library::buildStarter(a.get<bool>("rebuild", false));
            library::reload();
            if (a.get<bool>("thumbnails", true))
                out.update(renderStarterThumbnails(Graph::current(), out.value("built", false)));
            return out;
        },
        .examples = {"libraryStarter()['assets']", "place('Industrial/Barrel', on='Floor', count=6, spacing=[70, 0, 0])"},
    });

    ctx.registerCommand({
        .name = "libraryAdd",
        .doc = "Save objects to the library as one asset: the model as .glb, a thumbnail and asset.json with tags, size and physics.\n\n"
               "tier= picks the folder: workspace (default when a workspace is open, so it is checked in), user, or a studio "
               "folder's path. Physics comes from the flags, else from the first mesh's physicsBody. Returns the asset.",
        .positionalDoc = "objects (default the selection)",
        .flags = {{"name", "n", T::String, nullptr, "Asset name (default the first object's)."},
                  {"collection", "c", T::String, "Props", "Collection folder, may nest: 'Props/Kitchen'."},
                  {"tags", "t", T::List, json::array(), "Words to find it by."},
                  {"description", "d", T::String, "", "One line on what it is."},
                  {"tier", "", T::String, nullptr, "workspace, user, or a library folder path."},
                  {"role", "r", T::String, nullptr, "Physics role it places with: dynamic, static, kinematic, off."},
                  {"material", "m", T::String, nullptr, "Physics material it places with."},
                  {"unreal", "", T::String, "", "The Unreal asset it stands for, /Game/..."},
                  {"thumbnail", "", T::Bool, true, "Render thumb.png (skipped where there is no GPU)."},
                  {"overwrite", "o", T::Bool, false, "Replace an asset of the same id."}},
        .undoable = false,
        .run = [](const Args& a) -> json {
            LILA_ZONE();

            Graph& g = Graph::current();
            const std::vector<std::string> objects = a.objects();
            if (objects.empty())
                throw CommandError("libraryAdd: give objects or select them");
            std::vector<NodeId> ids;
            for (const std::string& o : objects)
            {
                Node* n = g.find(o);
                if (!n)
                    throw CommandError("libraryAdd: no node named '" + o + "'");
                ids.push_back(n->id());
            }
            const std::string tier = a.get<std::string>("tier", lila::Workspace::get().isOpen() ? "workspace" : "user");
            const fs::path base = library::tierDir(tier);
            if (base.empty())
                throw CommandError("libraryAdd: tier '" + tier + "' has no folder (open a workspace, or give a path)");
            const std::string name = identifier(a.get<std::string>("name", g.node(ids[0])->name()), "Asset");
            const std::string collection = a.get<std::string>("collection", "Props");
            const fs::path dir = base / library::utf8Path(collection) / name;
            if (fs::exists(dir / "asset.json") && !a.get<bool>("overwrite", false))
                throw CommandError("libraryAdd: " + (dir / "asset.json").generic_string() + " exists (overwrite=True replaces it)");
            fs::create_directories(dir);

            // An assembly saves its library members by id and where they stand; only the rest of it becomes the model.
            g.evaluate();
            json assembly;
            std::vector<NodeId> modelNodes = ids;
            if (ids.size() == 1 && assemblyTag(g, ids[0]))
            {
                const glm::dmat4 toRoot = glm::inverse(g.node(ids[0])->as<glm::dmat4>(scene::Transform::kWorldMatrix));
                json members = json::array();
                modelNodes.clear();
                for (NodeId c : g.children(ids[0]))
                {
                    Node* n = g.node(c);
                    if (!n || !n->type().isA("transform") || n->type().name == kAssetType || n->type().name == "stagehandAssembly")
                        continue;
                    if (const std::string member = assetOf(g, c); !member.empty())
                        members.push_back({{"asset", member}, {"matrix", matrixJson(toRoot * n->as<glm::dmat4>(scene::Transform::kWorldMatrix))}});
                    else
                        modelNodes.push_back(c);
                }
                assembly = {{"kind", assemblyKind(g, ids[0])}, {"members", members}, {"modelMatrix", matrixJson(toRoot)}};
            }

            const std::vector<NodeId> previous = g.selection();
            json names = json::array();
            for (NodeId id : ids)
                names.push_back(g.node(id)->path());
            ops::execute("select", {{"args", names}});
            if (a.get<bool>("thumbnail", true))
            {
                try
                {
                    ops::execute("captureView", {{"path", (dir / "thumb.png").generic_string()},
                                                 {"view", "three-quarter"},
                                                 {"width", 256},
                                                 {"height", 256},
                                                 {"display", "shaded"},
                                                 {"isolate", true},
                                                 {"frameSelection", true},
                                                 {"grid", false},
                                                 {"showSelection", false}});
                }
                catch (const std::exception& e)
                {
                    log::write(LogLevel::Info, std::string("libraryAdd: no thumbnail: ") + e.what());
                }
            }
            fs::path model = dir / (name + ".glb");
            if (modelNodes.empty())
                model.clear();
            else
            {
                // Bodies, snap points and tags travel in asset.json and come back on placing; in the model they would
                // land as empty transforms beside a second body. They go for the export only, in a cancelled chunk.
                UndoStack& undo = UndoStack::get();
                const bool scratch = undo.enabled() && !undo.chunkOpen();
                if (scratch)
                    undo.openChunk("libraryAdd export");
                try
                {
                    json tags = json::array();
                    for (NodeId id : modelNodes)
                        for (NodeId d : g.descendants(id))
                            if (Node* n = g.node(d); n && kTagTypes.count(n->type().name))
                                tags.push_back(n->fullPath());
                    if (scratch && !tags.empty())
                        ops::execute("delete", {{"args", tags}});
                    json parts = json::array();
                    for (NodeId id : modelNodes)
                        parts.push_back(g.node(id)->path());
                    ops::execute("select", {{"args", parts}});
                    ops::execute("exportGltf", {{"args", {model.generic_string()}}});
                }
                catch (...)
                {
                    if (scratch)
                        undo.cancelChunk();
                    throw;
                }
                if (scratch)
                    undo.cancelChunk();
            }
            json restore = json::array();
            for (NodeId id : previous)
                if (Node* n = g.node(id))
                    restore.push_back(n->path());
            if (restore.empty())
                ops::execute("select", {{"clear", true}});
            else
                ops::execute("select", {{"args", restore}});

            // Size from the meshes' world bounds; physics from the flags, else the first body found.
            g.evaluate();
            glm::dvec3 lo(1e300), hi(-1e300);
            json physics = json::object();
            for (NodeId id : ids)
                for (NodeId m : meshesUnder(g, id))
                {
                    Node* n = g.node(m);
                    const auto mesh = n->as<scene::MeshPtr>(n->type().attrIndex("outMesh"));
                    const glm::dmat4 w = n->as<glm::dmat4>(scene::Transform::kWorldMatrix);
                    if (mesh)
                        for (geo::Index p = 0; p < mesh->pointCapacity(); ++p)
                            if (mesh->pointValid(p))
                            {
                                const glm::dvec3 q(w * glm::dvec4(mesh->position(p), 1.0));
                                lo = glm::min(lo, q);
                                hi = glm::max(hi, q);
                            }
                    const BodySettings s = bodySettings(g, m);
                    if (s.hasNode && physics.empty())
                    {
                        const char* roles[] = {"dynamic", "static", "kinematic", "off"};
                        physics = {{"role", roles[int(s.role)]}, {"material", s.material}};
                    }
                }
            if (a.has("role"))
                physics["role"] = a.get<std::string>("role");
            if (a.has("material"))
                physics["material"] = a.get<std::string>("material");
            const glm::dvec3 size = lo.x <= hi.x ? hi - lo : glm::dvec3(0.0);
            // Snap points under the objects, by world matrix: the model is exported where it stands, so they match it.
            json snaps = json::array();
            std::vector<NodeId> walk(ids.begin(), ids.end());
            while (!walk.empty())
            {
                const NodeId id = walk.back();
                walk.pop_back();
                for (NodeId c : g.children(id))
                    walk.push_back(c);
                Node* n = g.node(id);
                if (!n || n->type().name != "stagehandSnap")
                    continue;
                const glm::dmat4 w = n->as<glm::dmat4>(scene::Transform::kWorldMatrix);
                json m = json::array();
                for (int k = 0; k < 16; ++k)
                    m.push_back(w[k / 4][k % 4]);
                snaps.push_back({{"name", n->name()},
                                 {"type", n->as<std::string>(n->type().attrIndex("snapType"))},
                                 {"accepts", kit::acceptsOf(*n)},
                                 {"matrix", m}});
            }
            json meta = {{"name", a.get<std::string>("name", name)},
                         {"tags", a.flags.value("tags", json::array())},
                         {"description", a.get<std::string>("description", "")},
                         {"size", v3(size)},
                         {"physics", physics},
                         {"unreal", a.get<std::string>("unreal", "")},
                         {"snapPoints", snaps}};
            if (!model.empty())
                meta["model"] = model.filename().generic_string();
            if (!assembly.is_null())
                meta["assembly"] = assembly;
            std::ofstream(dir / "asset.json") << meta.dump(4) << "\n";
            library::reload();
            const fs::path rel = fs::relative(dir, base);
            std::optional<Asset> made = library::find(rel.generic_string());
            return made ? made->toJson() : meta;
        },
        .examples = {"libraryAdd('Crate', collection='Props/Storage', tags=['crate', 'wood'], material='wood', role='dynamic')"},
    });

    ctx.registerCommand({
        .name = "libraryRemove",
        .doc = "Delete an asset's folder from the library. Not undoable; placed copies in scenes stay.",
        .positionalDoc = "asset id",
        .undoable = false,
        .run = [](const Args& a) -> json {
            if (a.positional.empty())
                throw CommandError("libraryRemove: give an asset id");
            const Asset x = need(a.positional[0].get<std::string>(), "libraryRemove");
            std::error_code ec;
            fs::remove_all(x.dir, ec);
            if (ec)
                throw CommandError("libraryRemove: " + ec.message());
            library::reload();
            return x.id;
        },
    });

    ctx.registerCommand({
        .name = "place",
        .doc = "Put a library asset in the scene: its bounds' bottom centre on a point, standing along the surface normal.\n\n"
               "Where: at= a world point (with snap, it lands on the first surface below), or on= a mesh (its top, or the "
               "point of it below at=). The first placement loads the asset as a hidden master under StagehandLibrary; each "
               "placement is an instance of it (mode='copy' for an independent copy), so editing one mesh edits them all. "
               "count= places several, spread by spacing= along X and seeded rotation jitter. drop=True lets them fall "
               "with physics after. Returns each placed root with its meshes.",
        .positionalDoc = "asset id (or a name only one asset has)",
        .flags = {{"at", "", T::List, nullptr, "World point [x, y, z], cm."},
                  {"on", "", T::String, nullptr, "Mesh to place on: its top centre, or below at=."},
                  {"snap", "sn", T::Bool, true, "Land at= on the surface below it."},
                  {"normal", "nr", T::List, nullptr, "Up direction to stand along (default the surface's, or +Y)."},
                  {"align", "al", T::Bool, false, "Stand along the surface normal instead of upright."},
                  {"rotate", "ro", T::Float, 0.0, "Turn about the up direction, degrees."},
                  {"scale", "sc", T::Any, 1.0, "Uniform scale, or [x, y, z]."},
                  {"mode", "m", T::String, nullptr, "instance or copy (default pref stagehand /library/insertMode, instance).", {"instance", "copy"}},
                  {"count", "n", T::Int, int64_t(1), "How many.", {}, {}, 1.0},
                  {"spacing", "sp", T::List, nullptr, "Offset between copies, [x, y, z] cm (default the asset's width along X)."},
                  {"jitter", "j", T::Float, 0.0, "Random turn per copy, up to this many degrees either way.", {}, {}, 0.0},
                  {"seed", "s", T::Int, int64_t(1), "Seed for jitter."},
                  {"drop", "dr", T::Bool, false, "Let them fall with physics onto what is below."},
                  {"name", "nm", T::String, nullptr, "Name for a single placement."}},
        .run = [](const Args& a) -> json {
            LILA_ZONE();

            Graph& g = Graph::current();
            if (a.positional.empty())
                throw CommandError("place: give an asset id (librarySearch finds them)");
            const Asset x = need(a.positional[0].get<std::string>(), "place");
            Placement p;
            p.mode = a.get<std::string>("mode", PrefStore::get().value<std::string>("stagehand", "/library/insertMode", "instance"));
            p.rotate = a.get<double>("rotate", 0.0);
            if (a.has("scale"))
            {
                const json& s = a.flags["scale"];
                p.scale = s.is_number() ? glm::dvec3(s.get<double>()) : vecFlag(a, "scale", glm::dvec3(1.0), "place");
            }
            NodeId on = scene::kNoNode;
            if (a.has("on"))
            {
                Node* n = g.find(a.get<std::string>("on"));
                if (!n)
                    throw CommandError("place: no node named '" + a.get<std::string>("on") + "'");
                on = n->id();
            }
            glm::dvec3 at = vecFlag(a, "at", glm::dvec3(0.0), "place");
            if (on != scene::kNoNode && !a.has("at"))
            {
                glm::dvec3 lo, hi;
                if (localBounds(g, on, lo, hi))
                {
                    const glm::dmat4 w = g.node(on)->as<glm::dmat4>(scene::Transform::kWorldMatrix);
                    at = glm::dvec3(w * glm::dvec4((lo.x + hi.x) * 0.5, hi.y, (lo.z + hi.z) * 0.5, 1.0));
                }
            }
            const int64_t count = std::max<int64_t>(1, a.get<int64_t>("count", 1));
            glm::dvec3 spacing(x.size.x > 0.0 ? x.size.x * 1.1 : 50.0, 0.0, 0.0);
            spacing = vecFlag(a, "spacing", spacing, "place");
            std::mt19937 rng(uint32_t(a.get<int64_t>("seed", 1)));
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            const double jitter = a.get<double>("jitter", 0.0);

            json out = json::array();
            std::vector<NodeId> placed;
            for (int64_t k = 0; k < count; ++k)
            {
                Placement one = p;
                one.at = at + spacing * double(k);
                one.rotate += jitter * u(rng);
                if (count == 1 && a.has("name"))
                    one.name = a.get<std::string>("name");
                if (a.get<bool>("snap", true) && (a.has("at") || count > 1 || on != scene::kNoNode))
                    if (auto hit = surfaceBelow(g, one.at + glm::dvec3(0, 1.0, 0), {}, on))
                    {
                        one.at = hit->point;
                        if (a.get<bool>("align", false))
                            one.normal = hit->normal;
                    }
                if (a.has("normal"))
                    one.normal = vecFlag(a, "normal", {0, 1, 0}, "place");
                const NodeId root = placeOne(g, x, one);
                placed.push_back(root);
                out.push_back(placedJson(g, root, x, one.mode));
            }
            if (a.get<bool>("drop", false))
            {
                json meshes = json::array();
                for (NodeId r : placed)
                    for (NodeId m : meshesUnder(g, r))
                        meshes.push_back(g.node(m)->path());
                ops::execute("drop", {{"args", meshes}});
                for (size_t k = 0; k < placed.size(); ++k)
                    out[k] = placedJson(g, placed[k], x, p.mode);
            }
            return out;
        },
        .examples = {"place('Props/Crate', at=[200, 0, 50], rotate=30)", "place('Books/Hardback', on='shelf_top', count=12, spacing=[4, 0, 0], jitter=3)",
                     "place('Kitchen/Mug', on='desk', drop=True)"},
    });

    ctx.registerCommand({
        .name = "replaceWith",
        .doc = "Swap objects for a library asset, each copy standing where the object stood, turned as it was turned about the up "
               "axis: blockout cubes into crates. fit=True scales each copy to the object's bounds. The originals are deleted.",
        .positionalDoc = "objects (default the selection)",
        .flags = {{"asset", "a", T::String, nullptr, "The asset id."},
                  {"fit", "f", T::Bool, false, "Scale each copy to fill the object's bounds."},
                  {"mode", "m", T::String, nullptr, "instance or copy.", {"instance", "copy"}},
                  {"keep", "k", T::Bool, false, "Keep the originals, hidden."}},
        .run = [](const Args& a) -> json {
            LILA_ZONE();

            Graph& g = Graph::current();
            if (!a.has("asset"))
                throw CommandError("replaceWith: give asset=");
            const Asset x = need(a.get<std::string>("asset"), "replaceWith");
            json out = json::array();
            for (const std::string& name : a.objects())
            {
                Node* n = g.find(name);
                if (!n)
                    throw CommandError("replaceWith: no node named '" + name + "'");
                const NodeId id = n->id();
                glm::dvec3 lo, hi;
                if (!localBounds(g, id, lo, hi))
                    continue;
                const glm::dmat4 w = n->as<glm::dmat4>(scene::Transform::kWorldMatrix);
                const Frame f = frameOf(w);
                // Yaw only: the heading of the object's local X on the ground.
                const glm::dvec3 xAxis = f.rotation * glm::dvec3(1, 0, 0);
                Placement p;
                p.mode = a.get<std::string>("mode", PrefStore::get().value<std::string>("stagehand", "/library/insertMode", "instance"));
                p.at = glm::dvec3(w * glm::dvec4((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5, 1.0));
                p.rotate = glm::degrees(std::atan2(-xAxis.z, xAxis.x));
                if (a.get<bool>("fit", false) && x.size.x > 0.0 && x.size.y > 0.0 && x.size.z > 0.0)
                    p.scale = (hi - lo) * glm::abs(f.scale) / x.size;
                const NodeId root = placeOne(g, x, p);
                if (a.get<bool>("keep", false))
                    ops::execute("setAttr", {{"args", {g.node(id)->path() + ".visibility", false}}});
                else
                    ops::execute("delete", {{"args", {g.node(id)->path()}}});
                out.push_back(placedJson(g, root, x, p.mode));
            }
            return out;
        },
        .examples = {"replaceWith(['box1', 'box2', 'box3'], asset='Props/Crate', fit=True)"},
    });

    ctx.registerCommand({
        .name = "libraryRefresh",
        .doc = "Reload the masters of library assets whose files changed on disk, so every instance placed from them updates.\n\n"
               "Instances follow their master's meshes; copies keep what they were. Returns the assets reloaded.",
        .flags = {{"force", "f", T::Bool, false, "Reload every master, changed or not."}},
        .run = [](const Args& a) -> json {
            LILA_ZONE();

            Graph& g = Graph::current();
            library::reload();
            json out = json::array();
            Node* group = g.find(kMasters);
            if (!group)
                return out;
            const std::vector<NodeId> masters = g.children(group->id());
            for (NodeId m : masters)
            {
                Node* tag = assetTag(g, m);
                if (!tag)
                    continue;
                const std::optional<Asset> x = library::find(tagValue(*tag, "asset"));
                if (!x || (!a.get<bool>("force", false) && stamp(x->model) == tagValue(*tag, "stamp")))
                    continue;
                // A fresh master, then each old mesh takes the new one's geometry so instances keep following it.
                const NodeId fresh = loadMaster(g, *x);
                const std::vector<NodeId> before = meshesUnder(g, m), after = meshesUnder(g, fresh);
                for (size_t k = 0; k < before.size() && k < after.size(); ++k)
                {
                    Node* from = g.node(after[k]);
                    const auto mesh = from->as<scene::MeshPtr>(from->type().attrIndex("outMesh"));
                    g.setValue({before[k], scene::MeshNode::kInMesh, -1}, mesh);
                }
                ops::execute("setAttr", {{"args", {tag->path() + ".stamp", stamp(x->model)}}});
                ops::execute("delete", {{"args", {g.node(fresh)->path()}}});
                out.push_back(x->id);
            }
            return out;
        },
    });
}
} // namespace stagehand
