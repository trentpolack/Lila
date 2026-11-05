// Geometry-node volumes to and from OpenVDB files (.vdb), the format Unreal, Blender, Houdini and EmberGen read.

#include "Volumes.h"

#include <lila/Command.h>
#include <lila/GeometryNodes.h>
#include <lila/Graph.h>
#include <lila/Paths.h>

#include <openvdb/openvdb.h>

#include <filesystem>
#include <format>
#include <list>
#include <mutex>

using namespace lila;

namespace nanovdbext
{
namespace
{
using json = nlohmann::json;

/** OpenVDB maps row vectors (index * M), so its matrix holds glm's columns as rows. */
openvdb::math::Mat4d toVdb(const glm::dmat4& m)
{
    openvdb::math::Mat4d out;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out(i, j) = m[i][j];
    return out;
}

glm::dmat4 fromVdb(const openvdb::math::Mat4d& m)
{
    glm::dmat4 out;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out[i][j] = m(i, j);
    return out;
}

openvdb::FloatGrid::Ptr toVdbGrid(const gn::Volume& v)
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(v.background);
    grid->setName(v.name);
    grid->setGridClass(v.isSdf() ? openvdb::GRID_LEVEL_SET : openvdb::GRID_FOG_VOLUME);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(toVdb(v.indexToWorld)));
    auto acc = grid->getAccessor();
    for (int k = 0; k < v.dims.z; ++k)
        for (int j = 0; j < v.dims.y; ++j)
            for (int i = 0; i < v.dims.x; ++i)
                if (float x = v.values[v.index(i, j, k)]; x != v.background)
                    acc.setValue(openvdb::Coord(i, j, k), x);
    return grid;
}

/** A float grid made dense over its active voxels. Throws past about 64 million voxels. */
gn::Volume fromVdbGrid(const openvdb::FloatGrid& grid)
{
    gn::Volume v;
    v.name = grid.getName();
    v.kind = grid.getGridClass() == openvdb::GRID_LEVEL_SET ? gn::Volume::Kind::Sdf : gn::Volume::Kind::Density;
    v.background = grid.background();
    const openvdb::CoordBBox box = grid.evalActiveVoxelBoundingBox();
    if (box.empty())
        return v;
    const openvdb::Coord lo = box.min(), hi = box.max();
    v.dims = {hi.x() - lo.x() + 1, hi.y() - lo.y() + 1, hi.z() - lo.z() + 1};
    if (double(v.dims.x) * v.dims.y * v.dims.z > 64.0e6)
        throw std::runtime_error("the grid is too big to read whole (" + std::to_string(v.dims.x) + " x " + std::to_string(v.dims.y) + " x " +
                                 std::to_string(v.dims.z) + " voxels)");
    auto affine = grid.transform().baseMap()->getAffineMap();
    v.indexToWorld = fromVdb(affine->getMat4()) * glm::dmat4(glm::dvec4(1, 0, 0, 0), glm::dvec4(0, 1, 0, 0), glm::dvec4(0, 0, 1, 0),
                                                              glm::dvec4(lo.x(), lo.y(), lo.z(), 1));
    v.values.assign(size_t(v.dims.x) * size_t(v.dims.y) * size_t(v.dims.z), v.background);
    auto acc = grid.getConstAccessor();
    for (int k = 0; k < v.dims.z; ++k)
        for (int j = 0; j < v.dims.y; ++j)
            for (int i = 0; i < v.dims.x; ++i)
                v.values[v.index(i, j, k)] = acc.getValue(openvdb::Coord(lo.x() + i, lo.y() + j, lo.z() + k));
    return v;
}

std::shared_ptr<const gn::Volume> readVdb(const std::filesystem::path& file, const std::string& gridName)
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
    openvdb::io::File f(paths::toUtf8(file));
    f.open();
    openvdb::FloatGrid::Ptr grid;
    if (!gridName.empty())
        grid = openvdb::gridPtrCast<openvdb::FloatGrid>(f.readGrid(gridName));
    else
        for (auto it = f.beginName(); it != f.endName() && !grid; ++it)
            grid = openvdb::gridPtrCast<openvdb::FloatGrid>(f.readGrid(it.gridName()));
    f.close();
    if (!grid)
        throw std::runtime_error(paths::toUtf8(file) + " has no float grid" + (gridName.empty() ? "" : " named " + gridName));
    auto v = std::make_shared<const gn::Volume>(fromVdbGrid(*grid));
    std::lock_guard lock(mutex);
    recent.emplace_front(key, v);
    if (recent.size() > 8)
        recent.pop_back();
    return v;
}
} // namespace

gn::NodeDecl readVdbNode()
{
    using T = gn::SocketType;
    return {
        .type = "ReadVDB",
        .label = "Read VDB",
        .category = "Volume",
        .doc = "A float grid from an OpenVDB file (.vdb, from Houdini, Blender or EmberGen) as a volume: level sets as signed distances, "
               "fog volumes as densities. Read whole over the grid's active voxels.",
        .sockets = {.inputs = {{.id = "path", .type = T::String, .defaultValue = "", .doc = "The .vdb file; relative paths are against the scene."},
                               {.id = "grid", .type = T::String, .defaultValue = "", .doc = "Grid name, such as density; empty reads the first float grid."}},
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
                    g.volume = readVdb(paths::resolvePath(path), c.s("grid"));
                    c.out("volume", gn::makeGeometry(std::move(g)));
                }
                catch (const std::exception& e)
                {
                    throw gn::ExecError(c.node.name + ": " + e.what());
                }
            },
    };
}

void registerVdbCommands(ext::Context& ctx)
{
    openvdb::initialize();
    ctx.registerCommand({
        .name = "exportVDB",
        .doc = "Write the volume a mesh's geometry nodes make (or a geometry-nodes modifier's) to an OpenVDB file: a sparse float grid, a "
               "level set for signed distances and a fog volume for densities, for Unreal's sparse volume textures, Houdini or Blender. "
               "Returns the file written.",
        .positionalDoc = "target path",
        .undoable = false,
        .run = [](const Args& a) -> json {
            if (a.positional.size() < 2 || !a.positional[0].is_string() || !a.positional[1].is_string())
                throw CommandError("exportVDB: target and path required");
            scene::Graph& g = scene::Graph::current();
            scene::Node* target = g.find(a.positional[0].get<std::string>());
            if (!target)
                throw CommandError("exportVDB: no node '" + a.positional[0].get<std::string>() + "'");
            std::shared_ptr<const gn::Volume> v = volumeOf(g, *target);
            if (!v)
                throw CommandError("exportVDB: " + target->name() + " has no geometry-nodes volume");
            std::filesystem::path path = paths::fromUtf8(a.positional[1].get<std::string>());
            if (path.extension().empty())
                path += ".vdb";
            try
            {
                if (path.has_parent_path())
                    std::filesystem::create_directories(path.parent_path());
                openvdb::io::File f(paths::toUtf8(path));
                f.write({toVdbGrid(*v)});
                f.close();
            }
            catch (const std::exception& e)
            {
                throw CommandError(std::string("exportVDB: ") + e.what());
            }
            return paths::toUtf8(path);
        },
    });
}
} // namespace nanovdbext
