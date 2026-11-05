#include "bake/HighFiles.h"

#include "core/Paths.h"
#include "core/Profile.h"
#include "core/Workspace.h"
#include "fbx/Fbx.h"
#include "geo/Mesh.h"
#include "geo/ObjIO.h"
#include "ops/Undo.h"
#include "scene/Anim.h"
#include "scene/Graph.h"
#include "scene/MeshNodes.h"
#include "scene/Modifiers.h"
#include "seq/Alembic.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <list>
#include <mutex>
#include <stdexcept>

namespace lila::bake
{
namespace fs = std::filesystem;

namespace
{
/** Files kept in memory; a dense high can be gigabytes, so only the most recent few. */
constexpr size_t kCacheEntries = 3;

struct HighCache
{
    std::mutex mutex;
    std::list<std::pair<std::string, std::vector<FileMesh>>> entries;
    HighFileStats stats;
};

HighCache& cache()
{
    static HighCache c;
    return c;
}

std::string lower(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

uint64_t fnv(std::string_view s, uint64_t h = 1469598103934665603ull)
{
    for (unsigned char c : s)
        h = (h ^ c) * 1099511628211ull;
    return h;
}

bool keep(const std::string& name, const std::string& filter) { return filter.empty() || name.find(filter) != std::string::npos; }

std::vector<FileMesh> readObj(const fs::path& path, const HighEntry& e)
{
    geo::ObjReadResult r = geo::readObjFile(paths::toUtf8(path));
    if (!r.ok())
        throw std::runtime_error(paths::toUtf8(path) + ": " + r.error);
    std::vector<FileMesh> out;
    for (geo::ObjObject& o : r.objects)
        if (keep(o.name, e.object) && o.mesh.numFaces())
            out.push_back({paths::toUtf8(path.filename()) + ":" + o.name, std::make_shared<geo::Mesh>(std::move(o.mesh))});
    return out;
}

std::vector<FileMesh> readAbc(const fs::path& path, const HighEntry& e)
{
    std::string err;
    const double fps = std::max(1e-6, scene::anim::playback().fps);
    std::vector<seq::abc::MeshSample> samples = seq::abc::readMeshes(path, e.frame / fps, e.object, &err);
    if (!err.empty())
        throw std::runtime_error(paths::toUtf8(path) + ": " + err);
    std::vector<FileMesh> out;
    for (seq::abc::MeshSample& s : samples)
        if (s.mesh.numFaces())
            out.push_back({paths::toUtf8(path.filename()) + ":" + s.path, std::make_shared<geo::Mesh>(std::move(s.mesh))});
    return out;
}

/** Imports into a private graph, so nothing reaches the scene or undo, and keeps each mesh as exported with its world matrix. */
std::vector<FileMesh> readFbx(const fs::path& path, const HighEntry& e)
{
    UndoStack& undo = UndoStack::get();
    const bool was = undo.enabled();
    undo.setEnabled(false);
    std::vector<FileMesh> out;
    try
    {
        scene::Graph g;
        fbx::ImportOptions opts;
        opts.skins = false;
        opts.animation = false;
        opts.setPlayback = false;
        const fbx::ImportResult r = fbx::importFbx(g, path, opts);
        if (!r.ok)
            throw std::runtime_error(paths::toUtf8(path) + ": " + r.error);
        g.evaluate();
        for (scene::NodeId id : r.nodes)
        {
            scene::Node* n = g.node(id);
            if (!n || !n->type().isA("mesh") || !keep(n->path(), e.object))
                continue;
            scene::MeshPtr m = scene::modifiers::exportMesh(g, id);
            if (!m)
                m = std::get<scene::MeshPtr>(g.value({id, scene::MeshNode::kOutMesh, -1}));
            if (!m || !m->numFaces())
                continue;
            const scene::PlugValue w = g.value({id, scene::Transform::kWorldMatrix, -1});
            out.push_back({paths::toUtf8(path.filename()) + ":" + n->path(), m,
                           std::holds_alternative<glm::dmat4>(w) ? std::get<glm::dmat4>(w) : glm::dmat4(1.0)});
        }
    }
    catch (...)
    {
        undo.setEnabled(was);
        throw;
    }
    undo.setEnabled(was);
    return out;
}
} // namespace

fs::path resolveHighFile(const std::string& file)
{
    fs::path p = paths::fromUtf8(file);
    if (p.is_relative())
        p = (Workspace::get().isOpen() ? Workspace::get().root() : fs::current_path()) / p;
    return p.lexically_normal();
}

std::string highFileKey(const HighEntry& e)
{
    const fs::path p = resolveHighFile(e.file);
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    if (ec)
        return {};
    const auto stamp = fs::last_write_time(p, ec).time_since_epoch().count();
    uint64_t h = fnv(lower(paths::toUtf8(p)));
    h = fnv(std::to_string(size) + "|" + std::format("{}", stamp) + "|" + e.object + "|" + std::to_string(e.frame), h);
    if (lower(p.extension().string()) == ".abc")
        h = fnv(std::to_string(scene::anim::playback().fps), h);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

std::vector<FileMesh> loadHighFile(const HighEntry& e)
{
    LILA_ZONE();

    const fs::path path = resolveHighFile(e.file);
    const std::string key = highFileKey(e);
    if (key.empty())
        throw std::runtime_error("high file " + paths::toUtf8(path) + " doesn't exist");
    HighCache& c = cache();
    {
        std::lock_guard lock(c.mutex);
        auto it = std::ranges::find_if(c.entries, [&](const auto& kv) { return kv.first == key; });
        if (it != c.entries.end())
        {
            c.entries.splice(c.entries.begin(), c.entries, it);
            ++c.stats.hits;
            return it->second;
        }
    }
    const std::string ext = lower(path.extension().string());
    std::vector<FileMesh> meshes;
    if (ext == ".obj")
        meshes = readObj(path, e);
    else if (ext == ".abc")
        meshes = readAbc(path, e);
    else if (ext == ".fbx")
        meshes = readFbx(path, e);
    else
        throw std::runtime_error("high file " + paths::toUtf8(path) + ": only .fbx, .obj and .abc can be streamed");
    if (meshes.empty())
        throw std::runtime_error("high file " + paths::toUtf8(path) + " holds no mesh" + (e.object.empty() ? "" : " matching '" + e.object + "'"));

    std::lock_guard lock(c.mutex);
    ++c.stats.reads;
    c.entries.emplace_front(key, meshes);
    while (c.entries.size() > kCacheEntries)
        c.entries.pop_back();
    return meshes;
}

HighFileStats highFileStats()
{
    std::lock_guard lock(cache().mutex);
    HighFileStats s = cache().stats;
    s.entries = cache().entries.size();
    return s;
}

void clearHighFileCache()
{
    std::lock_guard lock(cache().mutex);
    cache().entries.clear();
    cache().stats = {};
}
} // namespace lila::bake
