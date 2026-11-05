#include "scene/LodGroup.h"

#include "core/Decimal.h"

#include "ops/Command.h"
#include "ops/Undo.h"
#include "core/Log.h"
#include "core/Profile.h"
#include "geo/Mesh.h"
#include "scene/Graph.h"
#include "scene/MeshNodes.h"
#include "scene/ProxySet.h"
#include "scene/SceneIndex.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>

namespace lila::scene
{
namespace lod
{
namespace
{
using T = ArgType;

constexpr uint32_t kStored = AttrInput | AttrStorable;
const std::vector<std::string> kDisplayNames = {"viewport", "screenSize", "level", "all"};

/** Local bounds of a mesh, kept while the mesh lives: views ask every frame and the mesh rarely changes. */
geo::Bounds meshBounds(const MeshPtr& m)
{
    struct Entry
    {
        std::weak_ptr<const geo::Mesh> mesh;
        geo::Bounds bounds;
    };
    static std::unordered_map<const geo::Mesh*, Entry> cache;
    if (auto it = cache.find(m.get()); it != cache.end() && it->second.mesh.lock() == m)
        return it->second.bounds;
    if (cache.size() > 512)
        std::erase_if(cache, [](const auto& e) { return e.second.mesh.expired(); });
    geo::Bounds b = m->bounds();
    cache[m.get()] = {m, b};
    return b;
}

void extendWorld(Graph& g, NodeId id, geo::Bounds& out)
{
    const Node* n = g.node(id);
    if (!n || !n->type().isA("mesh"))
        return;
    const MeshPtr& m = n->as<MeshPtr>(MeshNode::kOutMesh);
    if (!m || !m->numPoints())
        return;
    const geo::Bounds local = meshBounds(m);
    if (!local.valid())
        return;
    const glm::dmat4 w = n->as<glm::dmat4>(Transform::kWorldMatrix);
    for (int c = 0; c < 8; ++c)
        out.extend(glm::dvec3(w * glm::dvec4(c & 1 ? local.max.x : local.min.x, c & 2 ? local.max.y : local.min.y, c & 4 ? local.max.z : local.min.z, 1.0)));
}

/** World bounds of the meshes at and under a level. */
geo::Bounds levelBounds(Graph& g, NodeId level)
{
    geo::Bounds b;
    extendWorld(g, level, b);
    for (NodeId d : g.descendants(level))
        extendWorld(g, d, b);
    return b;
}

std::string stripLodSuffix(std::string name)
{
    const size_t at = name.rfind('_');
    if (at == std::string::npos || at + 4 > name.size())
        return name;
    std::string tail = name.substr(at + 1);
    std::ranges::transform(tail, tail.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (tail.starts_with("lod") && tail.size() > 3 && std::all_of(tail.begin() + 3, tail.end(), [](unsigned char c) { return std::isdigit(c); }))
        name.resize(at);
    return name;
}

std::vector<double> sizesArg(const json& v)
{
    if (v.is_string())
        return parseSizes(v.get<std::string>());
    if (v.is_number())
        return {v.get<double>()};
    std::vector<double> out;
    if (v.is_array())
        for (const json& x : v)
        {
            if (!x.is_number())
                throw CommandError("lodGroup: screenSizes must be numbers");
            out.push_back(x.get<double>());
        }
    return out;
}

void renameLevels(Graph& g, NodeId group, const std::string& base)
{
    const std::vector<NodeId> lv = levels(g, group);
    for (size_t i = 0; i < lv.size(); ++i)
        g.rename(lv[i], std::format("{}_LOD{}", base, i));
}

/** Appends count levels made from LOD0 by polyReduce, each keeping ratio^index of its triangles. */
void generateLevels(Graph& g, NodeId group, int count, double ratio)
{
    const std::vector<NodeId> lv = levels(g, group);
    if (lv.empty() || !g.node(lv[0])->type().isA("mesh"))
        throw CommandError("lodGroup: generate needs LOD0 to be a mesh");
    for (int k = 0; k < count; ++k)
    {
        const size_t index = levels(g, group).size();
        if (index >= size_t(kMaxLevels))
        {
            log::warn("lodGroup: Unreal takes at most {} LODs; stopped generating at LOD{}", kMaxLevels, index - 1);
            break;
        }
        json made = ops::execute("duplicate", {{"args", {g.path(lv[0])}}});
        if (!made.is_array() || made.empty())
            throw CommandError("lodGroup: could not duplicate LOD0");
        Node* copy = g.find(made[0].get<std::string>());
        ops::execute("polyReduce", {{"args", {g.path(copy->id())}}, {"ratio", std::pow(ratio, double(index))}});
        g.setParent(copy->id(), group, true);
    }
}
} // namespace

const char* modeName(Mode m)
{
    switch (m)
    {
    case Mode::Level: return "level";
    case Mode::All: return "all";
    default: return "screenSize";
    }
}

bool parseMode(std::string_view name, Mode* out)
{
    for (Mode m : {Mode::ScreenSize, Mode::Level, Mode::All})
        if (name == modeName(m))
        {
            *out = m;
            return true;
        }
    return false;
}

bool isGroup(const Node& n) { return n.type().isA("lodGroup"); }

std::vector<NodeId> levels(const Graph& g, NodeId group)
{
    std::vector<NodeId> out;
    for (NodeId c : g.children(group))
        if (const Node* n = g.node(c); n && n->type().isA("transform"))
            out.push_back(c);
    return out;
}

std::vector<double> parseSizes(std::string_view text)
{
    std::vector<double> out;
    size_t i = 0;
    while (i < text.size())
    {
        while (i < text.size() && (text[i] == ' ' || text[i] == ',' || text[i] == '\t' || text[i] == '[' || text[i] == ']'))
            ++i;
        size_t end = i;
        while (end < text.size() && text[end] != ' ' && text[end] != ',' && text[end] != '\t' && text[end] != ']')
            ++end;
        double v = 0.0;
        if (end > i && lila::fromChars(text.data() + i, text.data() + end, v).ec == std::errc())
            out.push_back(v);
        i = end;
    }
    return out;
}

std::string formatSizes(const std::vector<double>& sizes)
{
    std::string out;
    for (double v : sizes)
        out += (out.empty() ? "" : " ") + std::format("{}", v);
    return out;
}

double autoScreenSize(int level, double previous)
{
    if (level <= 0)
        return 1.0;
    // Unreal's tolerance between neighbouring levels, so each sits strictly below the one before.
    return std::clamp(std::pow(0.75, double(level)), 0.0, previous - 0.01);
}

std::vector<double> screenSizes(const Node& group, size_t count)
{
    const std::vector<double> own = parseSizes(group.as<std::string>(LodGroup::kScreenSizes));
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i)
        out[i] = i < own.size() ? own[i] : autoScreenSize(int(i), i ? out[i - 1] : 2.0);
    return out;
}

double boundsScreenSize(const glm::dvec3& center, double radius, const View& view)
{
    const double multiple = std::max(0.5 * std::abs(view.proj[0][0]), 0.5 * std::abs(view.proj[1][1]));
    const double dist = view.ortho ? 1.0 : std::max(1.0, glm::length(center - view.eye));
    return 2.0 * multiple * radius / dist;
}

int levelForScreenSize(const std::vector<double>& sizes, double screenSize)
{
    for (int i = int(sizes.size()) - 1; i > 0; --i)
        if (sizes[size_t(i)] > screenSize)
            return i;
    return 0;
}

int shownLevel(Graph& g, NodeId group, const View& view)
{
    const Node* n = g.node(group);
    if (!n || !isGroup(*n))
        return -1;
    const std::vector<NodeId> lv = levels(g, group);
    if (lv.size() < 2)
        return -1;
    Mode mode = view.mode;
    int level = view.level;
    switch (Display(n->as<int64_t>(LodGroup::kLodDisplay)))
    {
    case Display::ScreenSize: mode = Mode::ScreenSize; break;
    case Display::Level:
        mode = Mode::Level;
        level = int(n->as<int64_t>(LodGroup::kDisplayLevel));
        break;
    case Display::All: mode = Mode::All; break;
    default: break;
    }
    if (mode == Mode::All)
        return -1;
    const int last = int(lv.size()) - 1;
    if (mode == Mode::Level)
        return std::clamp(level, 0, last);
    // Unreal sizes the group by its LOD0 bounds; fall back to any level when LOD0 has no geometry.
    geo::Bounds b = levelBounds(g, lv[0]);
    for (size_t i = 1; i < lv.size() && !b.valid(); ++i)
        b = levelBounds(g, lv[i]);
    if (!b.valid())
        return 0;
    const double screen = boundsScreenSize(b.center(), glm::length(b.size()) * 0.5, view);
    return levelForScreenSize(screenSizes(*n, lv.size()), screen);
}

std::unordered_set<NodeId> hidden(Graph& g, const View& view)
{
    LILA_ZONE();

    std::unordered_set<NodeId> out;
    for (const Node* n : sceneIndex::ofType(g, "lodGroup"))
    {
        const NodeId id = n->id();
        const int shown = shownLevel(g, id, view);
        if (shown < 0)
            continue;
        const std::vector<NodeId> lv = levels(g, id);
        for (size_t i = 0; i < lv.size(); ++i)
        {
            if (int(i) == shown)
                continue;
            out.insert(lv[i]);
            for (NodeId d : g.descendants(lv[i]))
                out.insert(d);
        }
    }
    out.merge(proxy::hidden(g, view));
    return out;
}

std::vector<double> exportScreenSizes(const Graph& g, const std::vector<NodeId>& roots)
{
    auto sizesOf = [&](NodeId id) -> std::vector<double> {
        const Node* n = g.node(id);
        if (!n || !isGroup(*n) || parseSizes(n->as<std::string>(LodGroup::kScreenSizes)).empty())
            return {};
        return screenSizes(*n, std::min(levels(g, id).size(), size_t(kMaxLevels)));
    };
    for (NodeId r : roots)
    {
        if (auto s = sizesOf(r); !s.empty())
            return s;
        for (NodeId d : g.descendants(r))
            if (auto s = sizesOf(d); !s.empty())
                return s;
    }
    return {};
}

void registerTypes()
{
    if (types::find("lodGroup"))
        return;
    NodeType t;
    t.name = "lodGroup";
    t.icon = "stack-2";
    t.base = types::find("transform");
    t.dag = true;
    t.doc = "Levels of detail: its children, in order, are LOD0, LOD1, ... Exports to FBX as an LOD group, which Unreal imports as one "
            "mesh with those LODs. Viewports show one level by screen size, a fixed level, or all of them.";
    t.ownAttrs = {
        {.name = "screenSizes", .type = PlugType::String, .defaultValue = std::string(), .flags = kStored,
         .doc = "Unreal LOD screen size per level, space separated (1 0.5 0.25). Levels without one use Unreal's automatic sizes.",
         .niceName = "Screen Sizes"},
        {.name = "lodDisplay", .type = PlugType::Enum, .defaultValue = int64_t(0), .flags = kStored, .enumNames = kDisplayNames,
         .doc = "Which level views show: viewport follows each view's LOD setting; the others override it for this group.", .niceName = "LOD Display"},
        {.name = "displayLevel", .type = PlugType::Int, .defaultValue = int64_t(0), .flags = kStored,
         .doc = "The level shown when lodDisplay is level.", .minValue = 0.0, .maxValue = double(kMaxLevels - 1)},
    };
    t.create = [] { return std::make_unique<LodGroup>(); };
    types::registerType(std::move(t));
}

void registerCommands()
{
    if (ops::find("lodGroup"))
        return;
    ops::registerCommand({
        .name = "lodGroup",
        .doc = "Make an LOD group from objects (LOD0 first, in order), or edit the LOD group named. Returns the group's path.\n\n"
               "With generate, LOD0 (one mesh) gets that many reduced copies as the next levels. Unreal imports the group as one "
               "mesh with these LODs.",
        .positionalDoc = "objects (LOD0 first), or one lodGroup to edit",
        .flags = {
            {"name", "n", T::String, nullptr, "Group name, and the base of the levels' names. Defaults to LOD0's name without any _LODn."},
            {"screenSizes", "ss", T::Any, nullptr, "Unreal LOD screen size per level, like [1, 0.5, 0.25]; [] uses Unreal's automatic sizes."},
            {"generate", "g", T::Int, 0, "Reduced copies of LOD0 to add as further levels.", {}, {}, 0.0, double(kMaxLevels - 1)},
            {"ratio", "r", T::Float, 0.5, "Triangles each generated level keeps of the one before, 0 to 1.", {}, {}, 0.0, 1.0},
            {"autoName", "an", T::Bool, nullptr, "Rename the levels <name>_LOD0, <name>_LOD1, ... On by default when creating."},
        },
        .run = [](const Args& a) -> json {
            LILA_ZONE();

            Graph& g = Graph::current();
            std::vector<NodeId> objects;
            for (const std::string& name : a.objects())
            {
                Node* n = g.find(name);
                if (!n || !n->type().isA("transform"))
                    throw CommandError("lodGroup: " + name + " is not an object");
                if (std::ranges::find(objects, n->id()) == objects.end())
                    objects.push_back(n->id());
            }
            if (objects.empty())
                throw CommandError("lodGroup: select the levels, LOD0 first, or one mesh with generate");
            const int generate = std::max(0, a.get<int>("generate", 0));
            const double ratio = std::clamp(a.get<double>("ratio", 0.5), 0.01, 1.0);
            std::optional<std::vector<double>> sizes;
            if (a.has("screenSizes"))
            {
                sizes = sizesArg(a.flags["screenSizes"]);
                for (double s : *sizes)
                    if (!(s > 0.0 && s <= 2.0))
                        throw CommandError("lodGroup: screen sizes run from above 0 to 2");
            }

            UndoChunk chunk("LOD Group");
            const bool editing = objects.size() == 1 && isGroup(*g.node(objects[0]));
            NodeId group = kNoNode;
            std::string base;
            if (editing)
            {
                group = objects[0];
                base = a.has("name") ? a.get<std::string>("name") : stripLodSuffix(g.node(group)->name());
                if (a.has("name"))
                    g.rename(group, base);
            }
            else
            {
                for (NodeId id : objects)
                    for (NodeId p = g.parent(id); p != kNoNode; p = g.parent(p))
                        if (std::ranges::find(objects, p) != objects.end())
                            throw CommandError("lodGroup: " + g.path(id) + " is under another level, " + g.path(p));
                base = a.has("name") ? a.get<std::string>("name") : stripLodSuffix(g.node(objects[0])->name());
                if (base.empty())
                    base = "LOD_Group";
                const NodeId parent = g.parent(objects[0]);
                // At LOD0's position, so exporting the group at the origin keeps LOD0 where an export of it alone would.
                const glm::dvec3 at(g.node(objects[0])->as<glm::dmat4>(Transform::kWorldMatrix)[3]);
                Node* made = g.createNode(*types::find("lodGroup"), {}, parent);
                group = made->id();
                const glm::dmat4 parentWorld = parent != kNoNode ? g.node(parent)->as<glm::dmat4>(Transform::kWorldMatrix) : glm::dmat4(1.0);
                g.setValue({group, Transform::kTranslate, -1}, glm::dvec3(glm::inverse(parentWorld) * glm::dvec4(at, 1.0)));
                for (NodeId id : objects)
                    g.setParent(id, group, true);
            }
            if (generate > 0)
                generateLevels(g, group, generate, ratio);
            if (levels(g, group).size() > size_t(kMaxLevels))
                log::warn("lodGroup: {} has {} levels; Unreal imports at most {}", g.path(group), levels(g, group).size(), kMaxLevels);
            if (a.get<bool>("autoName", !editing))
                renameLevels(g, group, base);
            if (!editing)
                g.rename(group, base);
            if (sizes)
                g.setValue({group, LodGroup::kScreenSizes, -1}, formatSizes(*sizes));
            g.select({group});
            return g.path(group);
        },
        .examples = {"lodGroup(\"Rock_LOD0\", \"Rock_LOD1\", \"Rock_LOD2\", screenSizes=[1, 0.4, 0.15])",
                     "lodGroup(\"Rock\", generate=3, ratio=0.5)"},
    });
}
} // namespace lod
} // namespace lila::scene
