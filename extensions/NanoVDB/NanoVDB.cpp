// Geometry-node volumes (gn::Volume, a dense float grid) to and from NanoVDB files.

#include "Volumes.h"

#include <lila/Command.h>
#include <lila/Deformer.h>
#include <lila/Extension.h>
#include <lila/GeometryNodes.h>
#include <lila/Graph.h>
#include <lila/Paths.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <glm/gtc/matrix_transform.hpp>

#include <filesystem>
#include <format>
#include <list>
#include <mutex>

using namespace lila;

namespace nanovdbext
{
using json = nlohmann::json;

/** The geometry-nodes volume a target shows: a gn modifier's, or the top one on a mesh's stack that made one. */
std::shared_ptr<const gn::Volume> volumeOf(scene::Graph& g, scene::Node& target)
{
    g.evaluate();
    std::vector<scene::NodeId> mods = {target.id()};
    if (!gn::isGnModifier(target))
    {
        mods = scene::modifiers::stack(g, target.id());
        std::reverse(mods.begin(), mods.end());
    }
    for (scene::NodeId m : mods)
        if (scene::Node* n = g.node(m); n && gn::isGnModifier(*n))
            if (gn::GeometryPtr geo = gn::modifierGeometry(m); geo && geo->volume && !geo->volume->empty())
                return geo->volume;
    return nullptr;
}

/** A sparse NanoVDB float grid of the voxels that differ from the background, with the volume's transform. */
nanovdb::GridHandle<nanovdb::HostBuffer> toNano(const gn::Volume& v)
{
    using BuildGrid = nanovdb::tools::build::Grid<float>;
    BuildGrid grid(v.background, v.name, v.isSdf() ? nanovdb::GridClass::LevelSet : nanovdb::GridClass::FogVolume);
    glm::dmat3 linear(v.indexToWorld);
    glm::dmat3 inverse = glm::inverse(linear);
    grid.mMap.set(linear, inverse, glm::dvec3(v.indexToWorld[3]));
    auto acc = grid.getAccessor();
    for (int k = 0; k < v.dims.z; ++k)
        for (int j = 0; j < v.dims.y; ++j)
            for (int i = 0; i < v.dims.x; ++i)
                if (float x = v.values[v.index(i, j, k)]; x != v.background)
                    acc.setValue(nanovdb::Coord(i, j, k), x);
    return nanovdb::tools::createNanoGrid(grid);
}

/** A NanoVDB float grid made dense over its active voxels. Throws past about 64 million voxels. */
gn::Volume fromNano(const nanovdb::NanoGrid<float>& grid)
{
    gn::Volume v;
    v.name = grid.gridName();
    v.kind = grid.gridClass() == nanovdb::GridClass::LevelSet ? gn::Volume::Kind::Sdf : gn::Volume::Kind::Density;
    v.background = grid.tree().background();
    const nanovdb::CoordBBox box = grid.indexBBox();
    if (box.empty())
        return v;
    const nanovdb::Coord lo = box.min(), hi = box.max();
    v.dims = {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};
    if (double(v.dims.x) * v.dims.y * v.dims.z > 64.0e6)
        throw std::runtime_error("the grid is too big to read whole (" + std::to_string(v.dims.x) + " x " + std::to_string(v.dims.y) + " x " +
                                 std::to_string(v.dims.z) + " voxels)");
    const nanovdb::Map& map = grid.map();
    glm::dmat4 m(1.0);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            m[c][r] = map.mMatD[3 * r + c];
    m[3] = glm::dvec4(map.mVecD[0], map.mVecD[1], map.mVecD[2], 1.0);
    // Voxel (0, 0, 0) of the dense grid is the box's lowest corner.
    v.indexToWorld = m * glm::translate(glm::dmat4(1.0), glm::dvec3(lo[0], lo[1], lo[2]));
    v.values.assign(size_t(v.dims.x) * size_t(v.dims.y) * size_t(v.dims.z), v.background);
    auto acc = grid.getAccessor();
    for (int k = 0; k < v.dims.z; ++k)
        for (int j = 0; j < v.dims.y; ++j)
            for (int i = 0; i < v.dims.x; ++i)
                v.values[v.index(i, j, k)] = acc.getValue(nanovdb::Coord(lo[0] + i, lo[1] + j, lo[2] + k));
    return v;
}

/** Recently read files, so a node reading the same file each evaluation decodes it once. */
std::shared_ptr<const gn::Volume> readVolume(const std::filesystem::path& file, const std::string& gridName)
{
    static std::mutex mutex;
    static std::list<std::pair<std::string, std::shared_ptr<const gn::Volume>>> recent;
    std::error_code ec;
    auto stamp = std::filesystem::last_write_time(file, ec);
    if (ec)
        throw std::runtime_error("can't find " + paths::toUtf8(file));
    std::string key = std::format("{}|{}|{}", paths::toUtf8(file), gridName, stamp.time_since_epoch().count());
    {
        std::lock_guard lock(mutex);
        for (auto& [k, v] : recent)
            if (k == key)
                return v;
    }
    nanovdb::GridHandle<nanovdb::HostBuffer> handle =
        gridName.empty() ? nanovdb::io::readGrid(paths::toUtf8(file)) : nanovdb::io::readGrid(paths::toUtf8(file), gridName);
    const nanovdb::NanoGrid<float>* grid = handle.grid<float>();
    if (!grid)
        throw std::runtime_error(paths::toUtf8(file) + " has no float grid" + (gridName.empty() ? "" : " named " + gridName));
    auto v = std::make_shared<const gn::Volume>(fromNano(*grid));
    std::lock_guard lock(mutex);
    recent.emplace_front(key, v);
    if (recent.size() > 8)
        recent.pop_back();
    return v;
}

gn::NodeDecl readNode()
{
    using T = gn::SocketType;
    return {
        .type = "ReadNanoVDB",
        .label = "Read NanoVDB",
        .category = "Volume",
        .doc = "A float grid from a NanoVDB file (.nvdb) as a volume: level sets as signed distances, fog volumes as densities. Read "
               "whole over the grid's active voxels.",
        .sockets = {.inputs = {{.id = "path", .type = T::String, .defaultValue = "", .doc = "The .nvdb file; relative paths are against the scene."},
                               {.id = "grid", .type = T::String, .defaultValue = "", .doc = "Grid name; empty reads the first."}},
                    .outputs = {{.id = "volume", .type = T::Geometry}}},
        .exec =
            [](gn::ExecContext& c) {
                std::string path = c.s("path");
                if (path.empty())
                {
                    c.out("volume", gn::makeGeometry({}));
                    return;
                }
                try
                {
                    gn::Geometry g;
                    g.volume = readVolume(paths::resolvePath(path), c.s("grid"));
                    c.out("volume", gn::makeGeometry(std::move(g)));
                }
                catch (const std::exception& e)
                {
                    throw gn::ExecError(c.node.name + ": " + e.what());
                }
            },
    };
}

void registerCommands(ext::Context& ctx)
{
    using AT = ArgType;
    ctx.registerCommand({
        .name = "exportNanoVDB",
        .doc = "Write the volume a mesh's geometry nodes make (or a geometry-nodes modifier's) to a NanoVDB file: a sparse float grid, "
               "a level set for signed distances and a fog volume for densities. Returns the file written.",
        .positionalDoc = "target path",
        .flags = {{"compress", "", AT::Bool, true, "Zip-compress the grid."}},
        .undoable = false,
        .run = [](const Args& a) -> json {
            if (a.positional.size() < 2 || !a.positional[0].is_string() || !a.positional[1].is_string())
                throw CommandError("exportNanoVDB: target and path required");
            scene::Graph& g = scene::Graph::current();
            scene::Node* target = g.find(a.positional[0].get<std::string>());
            if (!target)
                throw CommandError("exportNanoVDB: no node '" + a.positional[0].get<std::string>() + "'");
            std::shared_ptr<const gn::Volume> v = volumeOf(g, *target);
            if (!v)
                throw CommandError("exportNanoVDB: " + target->name() + " has no geometry-nodes volume");
            std::filesystem::path path = paths::fromUtf8(a.positional[1].get<std::string>());
            if (path.extension().empty())
                path += ".nvdb";
            try
            {
                if (path.has_parent_path())
                    std::filesystem::create_directories(path.parent_path());
                nanovdb::io::writeGrid(paths::toUtf8(path), toNano(*v), a.get<bool>("compress", true) ? nanovdb::io::Codec::ZIP : nanovdb::io::Codec::NONE);
            }
            catch (const std::exception& e)
            {
                throw CommandError(std::string("exportNanoVDB: ") + e.what());
            }
            return paths::toUtf8(path);
        },
    });
}
} // namespace nanovdbext

LILA_EXTENSION_EXPORT int lilaExtensionInit(lila::ext::Context& ctx)
{
    if (!gn::registerDecl(ctx, nanovdbext::readNode()) || !gn::registerDecl(ctx, nanovdbext::readVdbNode()))
        return 1;
    nanovdbext::registerCommands(ctx);
    nanovdbext::registerVdbCommands(ctx);
    return 0;
}
