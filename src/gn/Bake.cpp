#include "gn/Bake.h"

#include "gn/Exec.h"
#include "gn/GeometryIO.h"
#include "gn/Gn.h"
#include "gn/NodesCommon.h"
#include "gn/Zones.h"

#include "core/Paths.h"
#include "core/Profile.h"
#include "ops/Command.h"
#include "scene/Anim.h"
#include "scene/Graph.h"
#include "scene/Modifiers.h"
#include "scene/Simulation.h"

#include <format>
#include <list>
#include <mutex>

namespace lila::gn
{
namespace
{
std::mutex gCaptureMutex;
Capture* gCapture = nullptr;

/** A baked file as the node outputs it. */
struct Loaded
{
    GeometryPtr geometry;
    std::map<std::string, Value> items;
};

/** Recently read bake files, so playing back a baked range reads each frame once. */
std::shared_ptr<const Loaded> loadBake(const std::filesystem::path& file, const std::vector<ZoneItem>& items)
{
    static std::mutex mutex;
    static std::list<std::pair<std::string, std::shared_ptr<const Loaded>>> recent;
    std::error_code ec;
    auto stamp = std::filesystem::last_write_time(file, ec);
    if (ec)
        return nullptr;
    std::string key = std::format("{}|{}", paths::toUtf8(file), stamp.time_since_epoch().count());
    {
        std::lock_guard lock(mutex);
        for (auto it = recent.begin(); it != recent.end(); ++it)
            if (it->first == key)
            {
                recent.splice(recent.begin(), recent, it);
                return it->second;
            }
    }
    json doc = readFile(file);
    auto loaded = std::make_shared<Loaded>();
    loaded->geometry = geometryFromJson(doc.at("geometry"));
    const json& stored = doc.value("items", json::object());
    for (const ZoneItem& i : items)
        if (stored.contains(i.name))
            loaded->items[i.name] = valueFromStored(i.type, stored[i.name]);
    std::lock_guard lock(mutex);
    recent.emplace_front(key, loaded);
    if (recent.size() > 48)
        recent.pop_back();
    return loaded;
}

std::string frameFile(int64_t frame) { return std::format("frame_{:06}.lnb", frame); }

/** The file holding a baked node's result at a frame; empty when the node isn't baked. */
std::filesystem::path bakedFile(const GNode& n, double time)
{
    auto baked = n.values.find("baked");
    auto dir = n.values.find("directory");
    if (baked == n.values.end() || !baked->is_object() || dir == n.values.end() || !dir->is_string())
        return {};
    std::filesystem::path folder = paths::resolvePath(dir->get<std::string>());
    if (baked->value("mode", "") != "animation")
        return folder / "still.lnb";
    int64_t frame = std::clamp(scene::sim::frameOf(time), baked->value("start", int64_t(0)), baked->value("end", int64_t(0)));
    return folder / frameFile(frame);
}

void bakeNode()
{
    using namespace nodes;
    registerDecl({
        .type = "Bake",
        .label = "Bake",
        .category = "Geometry",
        .doc = "Saves what reaches it to disk, then gives that back instead of evaluating the nodes before it: freeze an expensive "
               "result, or a simulation, so playing and editing downstream stay fast. Bake and Delete Bake are in the Node Editor's "
               "side panel. Still saves one result; Animation saves one per frame over the playback range. Items carry values "
               "besides the geometry.",
        .props = {prop("mode", {"still", "animation"}, "still", "still: one result for every frame; animation: one per frame.")},
        .dynamicSockets =
            [](const Group&, const GNode& n) {
                Sockets s;
                s.inputs.push_back(geoIn("geometry"));
                s.outputs.push_back(geoOut("geometry"));
                for (const ZoneItem& i : itemsOf(n))
                {
                    bool data = isDataType(i.type);
                    s.inputs.push_back(data ? fin(i.name, i.type, zeroJson(i.type)) : in(i.name, i.type));
                    s.outputs.push_back(data ? fout(i.name, i.type) : out(i.name, i.type));
                }
                return s;
            },
        .exec =
            [](ExecContext& c) {
                LILA_ZONE();

                std::vector<ZoneItem> items = itemsOf(c.node);
                auto passThrough = [&] {
                    c.out("geometry", c.geometry("geometry"));
                    for (const ZoneItem& i : items)
                        if (int o = c.sockets.outputIndex(i.name), k = c.sockets.inputIndex(i.name); o >= 0 && k >= 0 && !c.inputs[size_t(k)].empty())
                            c.outputs[size_t(o)] = c.inputs[size_t(k)][0];
                };
                std::vector<std::string> names;
                for (const ZoneItem& i : items)
                    names.push_back(i.name);
                if (recordCapture(c, names))
                {
                    passThrough();
                    return;
                }
                std::filesystem::path file = bakedFile(c.node, c.env.time);
                if (file.empty())
                {
                    passThrough();
                    return;
                }
                std::shared_ptr<const Loaded> loaded = loadBake(file, items);
                if (!loaded)
                    throw ExecError(c.node.name + ": the baked file " + paths::toUtf8(file) + " is missing; bake again or delete the bake");
                c.out("geometry", loaded->geometry);
                for (const ZoneItem& i : items)
                    if (int o = c.sockets.outputIndex(i.name); o >= 0)
                        c.outputs[size_t(o)] = loaded->items.contains(i.name) ? loaded->items.at(i.name) : defaultValue(i.type);
            },
        .actions = {{"Bake", "Bake", "Save what reaches this node and read it back from now on."}},
    });
}

using scene::Graph;
using scene::Node;

std::string arg(const Args& a, size_t i, const char* cmd, const char* what)
{
    if (a.positional.size() <= i || !a.positional[i].is_string())
        throw CommandError(std::string(cmd) + ": " + what + " required");
    return a.positional[i].get<std::string>();
}

/** The group node and the Bake node in it, checked. */
std::pair<Node*, GNode> bakeTarget(const Args& a, const char* cmd)
{
    std::string group = arg(a, 0, cmd, "group"), node = arg(a, 1, cmd, "node");
    Node* gn = findGroupNode(Graph::current(), group);
    if (!gn)
        throw CommandError(std::string(cmd) + ": no geometry-nodes group named '" + group + "'");
    Group grp = readGroup(*gn);
    const GNode* n = grp.node(node);
    if (!n || n->type != "Bake")
        throw CommandError(std::string(cmd) + ": " + node + " isn't a Bake node in " + group);
    return {gn, *n};
}

/** Writes the node's values back undoably through the group. */
void setNodeValues(Node& groupNode, const std::string& node, const std::function<void(json&)>& fn)
{
    Group grp = readGroup(groupNode);
    if (GNode* n = grp.node(node))
        fn(n->values);
    commitGroup(Graph::current(), groupNode, std::move(grp));
}

std::filesystem::path defaultBakeDir(const std::string& group, const std::string& node)
{
    std::filesystem::path base = paths::currentSceneDir();
    if (base.empty())
        base = paths::userConfigDir() / "untitled";
    return base / "bakes" / (group + "_" + node);
}
} // namespace

CaptureScope::CaptureScope(Capture& c)
{
    std::lock_guard lock(gCaptureMutex);
    gCapture = &c;
}

CaptureScope::~CaptureScope()
{
    std::lock_guard lock(gCaptureMutex);
    gCapture = nullptr;
}

bool recordCapture(const ExecContext& c, const std::vector<std::string>& items)
{
    std::lock_guard lock(gCaptureMutex);
    if (!gCapture || gCapture->group != c.group.name || gCapture->node != c.node.name)
        return false;
    gCapture->geometry = c.geometry("geometry");
    for (const std::string& name : items)
        if (int k = c.sockets.inputIndex(name); k >= 0 && size_t(k) < c.inputs.size() && !c.inputs[size_t(k)].empty())
            gCapture->items[name] = c.inputs[size_t(k)][0];
    gCapture->got = true;
    return true;
}

bool bakeCaptureActive()
{
    std::lock_guard lock(gCaptureMutex);
    return gCapture != nullptr;
}

bool isBaked(const GNode& n) { return n.type == "Bake" && n.values.contains("baked"); }

void registerBakeNode() { bakeNode(); }

void registerBakeCommands()
{
    using Arg = ArgType;
    ops::registerCommand({
        .name = "gnBake",
        .doc = "Bake a Bake node: evaluate a mesh running its group, save what reaches the node (once, or per frame in animation mode) "
               "and switch the node to reading the saved files. Returns the folder written.",
        .positionalDoc = "group node",
        .flags = {{"start", "", Arg::Int, nullptr, "First frame (animation mode); defaults to the playback range's."},
                  {"end", "", Arg::Int, nullptr, "Last frame (animation mode); defaults to the playback range's."},
                  {"modifier", "", Arg::String, nullptr, "The modifier to evaluate when several meshes run the group; defaults to the first."},
                  {"directory", "", Arg::String, nullptr, "Folder for the files; defaults to bakes/<group>_<node> next to the scene."}},
        .run = [](const Args& a) -> json {
            LILA_ZONE();

            auto [groupNode, bake] = bakeTarget(a, "gnBake");
            Graph& g = Graph::current();
            std::string groupName = groupNode->name();
            Group grp = readGroup(*groupNode);
            std::vector<scene::NodeId> mods = instancesOf(g, groupNode->id());
            scene::NodeId mod = mods.empty() ? scene::kNoNode : mods.front();
            if (a.has("modifier"))
            {
                Node* m = g.find(a.get<std::string>("modifier"));
                mod = m && std::find(mods.begin(), mods.end(), m->id()) != mods.end() ? m->id() : scene::kNoNode;
                if (mod == scene::kNoNode)
                    throw CommandError("gnBake: " + a.get<std::string>("modifier") + " isn't a modifier running " + groupName);
            }
            if (mod == scene::kNoNode)
                throw CommandError("gnBake: no mesh runs " + groupName + "; apply it to one first");

            // Baking again starts from the live result.
            if (isBaked(bake))
                setNodeValues(*groupNode, bake.name, [](json& v) { v.erase("baked"); });

            std::filesystem::path dir = a.has("directory") ? paths::fromUtf8(a.get<std::string>("directory")) : defaultBakeDir(groupName, bake.name);
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
            std::filesystem::create_directories(dir, ec);

            const bool animation = bake.values.value("mode", "still") == "animation";
            const auto& pb = scene::anim::playback();
            int64_t start = animation ? a.get<int64_t>("start", int64_t(std::llround(scene::anim::frameOf(pb.minTime)))) : 0;
            int64_t end = animation ? a.get<int64_t>("end", int64_t(std::llround(scene::anim::frameOf(pb.maxTime)))) : 0;
            if (end < start)
                throw CommandError("gnBake: end comes before start");
            const std::vector<ZoneItem> items = itemsOf(bake);
            const double before = g.time();
            Capture capture{groupName, bake.name};
            try
            {
                CaptureScope scope(capture);
                for (int64_t f = start; f <= end; ++f)
                {
                    if (animation)
                        g.setTime(scene::anim::secondsOf(double(f)), false);
                    capture.got = false;
                    g.touch(mod);
                    g.evaluateUpstream({scene::modifiers::meshOf(g, mod)});
                    if (!capture.got)
                        throw CommandError("gnBake: nothing reached " + bake.name + "; is it connected to the output and the modifier enabled?");
                    json stored = json::object();
                    for (const ZoneItem& i : items)
                        if (capture.items.contains(i.name))
                            stored[i.name] = valueToJson(i.type, capture.items[i.name]);
                    writeFile(dir / (animation ? frameFile(f) : std::string("still.lnb")),
                              {{"geometry", geometryToJson(*capture.geometry)}, {"items", stored}});
                }
            }
            catch (...)
            {
                g.setTime(before, false);
                throw;
            }
            g.setTime(before, false);
            std::string stored = paths::storedPath(dir);
            setNodeValues(*groupNode, bake.name, [&](json& v) {
                v["directory"] = stored;
                v["baked"] = {{"mode", animation ? "animation" : "still"}, {"start", start}, {"end", end}};
            });
            return paths::toUtf8(dir);
        },
    });

    ops::registerCommand({
        .name = "gnDeleteBake",
        .doc = "Delete a Bake node's saved files and go back to evaluating the nodes before it.",
        .positionalDoc = "group node",
        .run = [](const Args& a) -> json {
            auto [groupNode, bake] = bakeTarget(a, "gnDeleteBake");
            if (auto dir = bake.values.find("directory"); dir != bake.values.end() && dir->is_string())
            {
                std::error_code ec;
                std::filesystem::remove_all(paths::resolvePath(dir->get<std::string>()), ec);
            }
            setNodeValues(*groupNode, bake.name, [](json& v) {
                v.erase("baked");
                v.erase("directory");
            });
            return nullptr;
        },
    });
}
} // namespace lila::gn
