#include <doctest/doctest.h>

#include "TestUtil.h"
#include "bake/Backend.h"
#include "bake/Bake.h"
#include "core/ImageIO.h"
#include "core/MainQueue.h"
#include "geo/Bvh.h"
#include "geo/ColorSets.h"
#include "geo/Mesh.h"
#include "geo/Primitives.h"
#include "ops/Command.h"
#include "ops/Undo.h"
#include "render/Baker.h"
#include "render/Renderer.h"
#include "render/VulkanContext.h"
#include "scene/Graph.h"
#include "scene/Materials.h"
#include "scene/MeshNodes.h"

#include <glm/gtc/constants.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <thread>

using namespace lila;
using namespace baketest;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace
{
geo::Mesh moved(geo::Mesh m, const glm::dvec3& by, bool flip = false)
{
    for (geo::Index p = 0; p < m.pointCapacity(); ++p)
    {
        glm::dvec3 q = m.position(p);
        // A half turn about X flips which way a flat plane faces.
        if (flip)
            q = glm::dvec3(q.x, -q.y, -q.z);
        m.setPosition(p, q + by);
    }
    return m;
}

/** The GPU backends this machine has: "gpu-rt" with ray query, "gpu-compute" always, none without a Vulkan device. */
const std::vector<std::string>& gpuBackends()
{
    static std::vector<std::string> names = [] {
        std::vector<std::string> out;
        try
        {
            render::createGpuBakeBackend(render::GpuTrace::Software);
        }
        catch (const std::exception& e)
        {
            // Port bring-up must not report success by silently skipping every GPU comparison.
            if (std::getenv("LILA_REQUIRE_GPU"))
                FAIL("GPU required for this run: " << e.what());
            MESSAGE("no GPU for bake tests: " << e.what());
            return out;
        }
        bake::registerBackend("gpu-compute", [] { return render::createGpuBakeBackend(render::GpuTrace::Software); });
        out.push_back("gpu-compute");
        if (render::Renderer::get().context()->rayQuery)
        {
            bake::registerBackend("gpu-rt", [] { return render::createGpuBakeBackend(render::GpuTrace::Hardware); });
            out.push_back("gpu-rt");
        }
        MESSAGE("bake GPU: " << render::Renderer::get().deviceName() << std::string(out.size() > 1 ? " (ray query)" : " (compute only)"));
        return out;
    }();
    return names;
}

struct Baked
{
    json result;
    std::map<std::string, image::Image> maps;
};

Baked bakeWith(const std::string& p, const std::vector<std::string>& maps, const std::string& backend)
{
    Baked b;
    b.result = ops::execute("bake", {{"project", p}, {"maps", maps}, {"backend", backend}});
    for (const json& o : b.result["outputs"])
    {
        std::string err;
        b.maps[o["map"].get<std::string>()] = image::read(fs::path(o["file"].get<std::string>()), &err);
        INFO(err);
        REQUIRE(b.maps[o["map"].get<std::string>()].width > 0);
    }
    return b;
}

glm::vec4 texelAt(const image::Image& img, size_t i) { return {img.rgba[i * 4], img.rgba[i * 4 + 1], img.rgba[i * 4 + 2], img.rgba[i * 4 + 3]}; }

float angleDegOrFlat(glm::vec3 a, glm::vec3 b)
{
    if (glm::length(a) < 1e-6f || glm::length(b) < 1e-6f)
        return glm::length(a - b) < 1e-6f ? 0.0f : 180.0f;
    return glm::degrees(std::acos(std::clamp(glm::dot(glm::normalize(a), glm::normalize(b)), -1.0f, 1.0f)));
}

/**
 * GPU output against the CPU's, per texel: at most outlierFraction of the texels may differ by more than tol (rays along
 * triangle edges may pick the neighbour), and none by more than hardTol.
 */
void compare(const Baked& cpu, const Baked& gpu, const std::string& map, float tol, float hardTol, double outlierFraction = 0.002)
{
    INFO("map " << map);
    REQUIRE(cpu.maps.count(map));
    REQUIRE(gpu.maps.count(map));
    const image::Image& a = cpu.maps.at(map);
    const image::Image& b = gpu.maps.at(map);
    REQUIRE(a.width == b.width);
    REQUIRE(a.height == b.height);
    const bool normal = map == "normal" || map == "objectNormal";
    size_t outliers = 0;
    float worst = 0.0f;
    const size_t n = size_t(a.width) * size_t(a.height);
    for (size_t i = 0; i < n; ++i)
    {
        const glm::vec4 x = texelAt(a, i), y = texelAt(b, i);
        const float d = normal ? angleDegOrFlat(glm::vec3(x) * 2.0f - 1.0f, glm::vec3(y) * 2.0f - 1.0f)
                               : std::max({std::abs(x.r - y.r), std::abs(x.g - y.g), std::abs(x.b - y.b), std::abs(x.a - y.a)});
        worst = std::max(worst, d);
        if (d > tol)
            ++outliers;
    }
    INFO("worst " << worst << ", " << outliers << " of " << n << " texels over " << tol);
    CHECK(double(outliers) <= outlierFraction * double(n));
    CHECK(worst <= hardTol);
}

uint64_t stat(const Baked& b, const char* key) { return b.result["stats"][key].get<uint64_t>(); }

json meta(const Baked& b, const std::string& map)
{
    for (const json& o : b.result["outputs"])
        if (o["map"] == map)
            return o["meta"];
    return json();
}
} // namespace

TEST_CASE("MeshBvh exports a flat tree with every triangle once")
{
    geo::IcoSphereOptions io;
    io.subdivisions = 4;
    const geo::Mesh m = geo::makeIcoSphere(io);
    geo::MeshBvh bvh(m);
    const geo::FlatBvh flat = bvh.exportFlat();
    REQUIRE(!flat.nodes.empty());
    CHECK(flat.corners.size() == size_t(bvh.triangleCount()) * 3);
    std::vector<int> seen(bvh.triangleCount(), 0);
    for (size_t s = 0; s < flat.corners.size(); s += 3)
    {
        uint32_t t;
        std::memcpy(&t, &flat.corners[s].w, 4);
        REQUIRE(t < seen.size());
        ++seen[t];
    }
    CHECK(std::ranges::all_of(seen, [](int c) { return c == 1; }));
    size_t leafSlots = 0;
    for (const geo::FlatBvh::Node& n : flat.nodes)
        leafSlots += n.count;
    CHECK(leafSlots == bvh.triangleCount());
}

TEST_CASE("GPU bake matches the CPU: bump plane normals, height, position")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_bump");
    const double A = 3.0, k = glm::two_pi<double>() / 100.0;
    const std::string low = addMesh("bump_low", plane(1));
    const std::string high = addMesh("bump_high", displaced(200, [&](double x, double z) { return A * std::cos(k * x) + A * std::cos(k * z); }));
    const std::string p = project(dir, 96, 4);
    ops::execute("bakeGroup", {{"args", {"bump"}}, {"project", p}, {"lows", {low}}, {"highs", {high}}, {"frontal", 10.0}, {"rear", 10.0}});
    const std::vector<std::string> maps{"normal", "objectNormal", "height", "position", "alpha"};
    const Baked cpu = bakeWith(p, maps, "cpu");
    for (const std::string& be : gpuBackends())
    {
        INFO("backend " << be);
        const Baked gpu = bakeWith(p, maps, be);
        CHECK(stat(gpu, "samples") == stat(cpu, "samples"));
        CHECK(stat(gpu, "misses") == stat(cpu, "misses"));
        compare(cpu, gpu, "normal", 0.05f, 0.5f);
        compare(cpu, gpu, "objectNormal", 0.05f, 0.5f);
        compare(cpu, gpu, "height", 1e-4f, 2e-3f);
        compare(cpu, gpu, "position", 1e-4f, 2e-3f);
        compare(cpu, gpu, "alpha", 0.0f, 0.0f, 0.0);
        CHECK(meta(gpu, "height")["heightRange"].get<double>() == doctest::Approx(meta(cpu, "height")["heightRange"].get<double>()).epsilon(1e-3));
    }
}

TEST_CASE("GPU bake matches the CPU: sphere over an icosphere, normals and AO")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_sphere");
    geo::SphereOptions so;
    so.subdivAxis = 32;
    so.subdivHeight = 32;
    const std::string low = addMesh("ball_low", geo::makeSphere(so));
    geo::IcoSphereOptions io;
    io.subdivisions = 6;
    const std::string high = addMesh("ball_high", geo::makeIcoSphere(io));
    // A floor under the ball for the AO rays to find.
    const std::string floorHigh = addMesh("floor_high", moved(plane(4), {0.0, -52.0, 0.0}));
    const std::string p = project(dir, 64, 4);
    ops::execute("bakeGroup",
                 {{"args", {"ball"}}, {"project", p}, {"lows", {low}}, {"highs", {high, floorHigh}}, {"frontal", 3.0}, {"rear", 3.0}});
    ops::execute("setBakeMap", {{"args", {"ao"}}, {"project", p}, {"options", {{"rays", 32}, {"distance", 40.0}}}});
    const std::vector<std::string> maps{"normal", "objectNormal", "position", "ao"};
    const Baked cpu = bakeWith(p, maps, "cpu");
    for (const std::string& be : gpuBackends())
    {
        INFO("backend " << be);
        const Baked gpu = bakeWith(p, maps, be);
        CHECK(stat(gpu, "samples") == stat(cpu, "samples"));
        CHECK(double(stat(gpu, "misses")) <= double(stat(cpu, "misses")) + 0.001 * double(stat(cpu, "samples")));
        compare(cpu, gpu, "normal", 0.05f, 1.0f, 0.005);
        compare(cpu, gpu, "objectNormal", 0.05f, 1.0f, 0.005);
        compare(cpu, gpu, "position", 1e-4f, 2e-3f, 0.005);
        // 32 rays: one ray flipping at a grazing angle moves a sample by 1/32, a texel by a quarter of that.
        compare(cpu, gpu, "ao", 0.01f, 0.1f, 0.02);
    }
}

TEST_CASE("GPU bake matches the CPU: exact material IDs, islands and coverage")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_id");
    geo::Mesh hi = plane(16);
    std::vector<uint32_t> left, right;
    for (geo::Index f = 0; f < hi.faceCapacity(); ++f)
        (hi.faceCentroid(f).x < 0.0 ? left : right).push_back(f);
    scene::materials::assign(hi, "matA", left);
    scene::materials::assign(hi, "matB", right);
    geo::Mesh a = plane(1, {0.0f, 0.0f}, {0.4f, 1.0f});
    a.append(moved(plane(1, {0.6f, 0.0f}, {1.0f, 1.0f}), {0.0, 0.0, 0.0}));
    const std::string low = addMesh("tile_low", std::move(a));
    const std::string high = addMesh("tile_high", moved(std::move(hi), {0.0, 1.0, 0.0}));
    const std::string p = project(dir, 33, 16, "png");
    ops::execute("bakeGroup", {{"args", {"tile"}}, {"project", p}, {"lows", {low}}, {"highs", {high}}, {"frontal", 3.0}, {"rear", 3.0}});
    const std::vector<std::string> maps{"id", "uvIslands", "alpha"};
    const Baked cpu = bakeWith(p, maps, "cpu");
    for (const std::string& be : gpuBackends())
    {
        INFO("backend " << be);
        const Baked gpu = bakeWith(p, maps, be);
        compare(cpu, gpu, "id", 0.0f, 0.0f, 0.0);
        compare(cpu, gpu, "uvIslands", 0.0f, 0.0f, 0.0);
        compare(cpu, gpu, "alpha", 0.0f, 0.0f, 0.0);
    }

    // Object source and a palette override.
    ops::execute("setBakeMap", {{"args", {"id"}}, {"project", p}, {"options", {{"source", "object"}, {"palette", {{"tile_high", "#ff8000"}}}}}});
    const Baked cpu2 = bakeWith(p, {"id"}, "cpu");
    for (const std::string& be : gpuBackends())
        compare(cpu2, bakeWith(p, {"id"}, be), "id", 0.0f, 0.0f, 0.0);
}

TEST_CASE("GPU bake matches the CPU: isolated groups, AO scopes and backfaces")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_isolate");
    const std::string low1 = addMesh("a_low", plane(1, {0.0f, 0.0f}, {0.5f, 1.0f}));
    const std::string low2 = addMesh("b_low", plane(1, {0.5f, 0.0f}, {1.0f, 1.0f}));
    const std::string high1 = addMesh("a_high", moved(plane(4), {0.0, 1.0, 0.0}));
    const std::string high2 = addMesh("b_high", moved(plane(4), {0.0, 0.5, 0.0}));
    const std::string p = project(dir, 32, 4);
    for (auto [grp, low, high] : {std::tuple{"a", low1, high1}, {"b", low2, high2}})
        ops::execute("bakeGroup", {{"args", {grp}}, {"project", p}, {"lows", {low}}, {"highs", {high}}, {"frontal", 3.0}, {"rear", 3.0}});
    for (const char* scope : {"group", "all", "self"})
    {
        INFO("scope " << scope);
        ops::execute("setBakeMap", {{"args", {"ao"}}, {"project", p}, {"options", {{"rays", 16}, {"distance", 50.0}, {"scope", scope}}}});
        const Baked cpu = bakeWith(p, {"height", "ao"}, "cpu");
        for (const std::string& be : gpuBackends())
        {
            INFO("backend " << be);
            const Baked gpu = bakeWith(p, {"height", "ao"}, be);
            compare(cpu, gpu, "height", 1e-4f, 2e-3f, 0.0);
            compare(cpu, gpu, "ao", 0.02f, 0.1f, 0.01);
        }
    }

    // A high facing away: every ray meets a back face, and the GPU must call it one too.
    newScene();
    const std::string low = addMesh("c_low", plane(1));
    const std::string flipped = addMesh("c_high", moved(plane(4), {0.0, 1.0, 0.0}, true));
    const std::string p2 = project(dir, 16, 1);
    ops::execute("bakeGroup", {{"args", {"c"}}, {"project", p2}, {"lows", {low}}, {"highs", {flipped}}, {"frontal", 3.0}, {"rear", 3.0}});
    const Baked cpu = bakeWith(p2, {"height"}, "cpu");
    CHECK(stat(cpu, "backfaces") == stat(cpu, "samples"));
    for (const std::string& be : gpuBackends())
    {
        INFO("backend " << be);
        const Baked gpu = bakeWith(p2, {"height"}, be);
        CHECK(stat(gpu, "backfaces") == stat(cpu, "backfaces"));
        CHECK(stat(gpu, "misses") == stat(cpu, "misses"));
    }
}

TEST_CASE("GPU bake matches the CPU: self highs per low and unlinked occluders, group and all scopes")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_links");
    // A floor and a lid 0.5 cm over it, each baking onto itself; a roof and a second group's low above, all unlinked from the floor.
    const std::string floor = addMesh("floor", plane(1, {0.0f, 0.0f}, {0.5f, 1.0f}));
    const std::string lid = addMesh("lid", moved(plane(1, {0.5f, 0.0f}, {1.0f, 1.0f}), {0.0, 0.5, 0.0}));
    const std::string roof = addMesh("roof", moved(plane(2), {0.0, 5.0, 0.0}));
    const std::string other = addMesh("other", moved(plane(1), {0.0, 4.0, 0.0}));
    const std::string p = project(dir, 32, 1);
    ops::execute("bakeGroup", {{"args", {"pair"}}, {"project", p}, {"lows", {floor, lid}}, {"self", true}, {"addHigh", {roof}},
                               {"frontal", 2.0}, {"rear", 2.0},
                               {"unlinks", json::array({json{{"lows", {floor}}, {"meshes", {roof, other, lid}}}})}});
    ops::execute("bakeGroup", {{"args", {"far"}}, {"project", p}, {"lows", {other}}, {"self", true}, {"frontal", 0.2}, {"rear", 0.2}});
    for (const char* scope : {"group", "all"})
    {
        INFO("scope " << scope);
        ops::execute("setBakeMap", {{"args", {"ao"}}, {"project", p}, {"options", {{"rays", 16}, {"distance", 50.0}, {"scope", scope}}}});
        // Each group writes its own files: key the images by file name so neither group's replaces the other's.
        auto bakeAll = [&](const std::string& be) {
            Baked b;
            b.result = ops::execute("bake", {{"project", p}, {"maps", {"height", "ao"}}, {"backend", be}});
            for (const json& o : b.result["outputs"])
                b.maps[fs::path(o["file"].get<std::string>()).stem().string()] = image::read(fs::path(o["file"].get<std::string>()));
            return b;
        };
        const Baked cpu = bakeAll("cpu");
        REQUIRE(cpu.maps.size() == 4);
        // The floor (u < 0.5) is open to the sky through the unlinked roof; the lid (u > 0.5) stays under it.
        const image::Image& ao = cpu.maps.at("pair_ao");
        CHECK(texelAt(ao, size_t(16 * 32 + 8)).r > 0.95f);
        CHECK(texelAt(ao, size_t(16 * 32 + 24)).r < 0.8f);
        // The floor bakes itself, not the lid 0.5 cm above it.
        CHECK(texelAt(cpu.maps.at("pair_height"), size_t(16 * 32 + 8)).r == doctest::Approx(0.5f).epsilon(0.02));
        for (const std::string& be : gpuBackends())
        {
            INFO("backend " << be);
            const Baked gpu = bakeAll(be);
            for (const char* grp : {"pair", "far"})
            {
                compare(cpu, gpu, std::string(grp) + "_height", 1e-4f, 2e-3f, 0.0);
                compare(cpu, gpu, std::string(grp) + "_ao", 0.02f, 0.1f, 0.01);
            }
        }
    }
}

TEST_CASE("GPU dilation fills the same texels as the CPU and copies values exactly")
{
    if (gpuBackends().empty())
        return;
    const int W = 256, H = 200;
    bake::MapDef def;
    def.id = "test";
    auto make = [&] {
        bake::TextureBuffer t;
        t.def = &def;
        t.width = W;
        t.height = H;
        t.value.assign(size_t(W) * H, glm::vec4(0.0f));
        t.covered.assign(size_t(W) * H, 0);
        for (int y = 40; y < 160; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = size_t(y) * W + x;
                if (x < 100)
                    t.value[i] = glm::vec4(1, 0, 0, 1), t.covered[i] = 1;
                else if (x >= 140)
                    t.value[i] = glm::vec4(0, 0, float(y) / 255.0f, 1), t.covered[i] = 1;
            }
        return t;
    };
    std::unique_ptr<bake::Backend> gpu = render::createGpuBakeBackend(render::GpuTrace::Software);
    for (int padding : {0, 6, 32, -1})
    {
        INFO("padding " << padding);
        bake::TextureBuffer a = make(), b = make();
        bake::dilateCpu(a.value, a.covered, W, H, padding);
        gpu->dilate(b, padding);
        CHECK(a.covered == b.covered);
        size_t reachedA = 0, reachedB = 0, differ = 0;
        for (size_t i = 0; i < a.value.size(); ++i)
        {
            reachedA += a.value[i].a > 0.0f;
            reachedB += b.value[i].a > 0.0f;
            differ += a.value[i] != b.value[i];
        }
        CHECK(reachedA == reachedB);
        // Seeds may differ only where two islands are equally near.
        CHECK(differ <= size_t(W) * H / 100);
    }
}

TEST_CASE("GPU bake speed against the CPU" * doctest::skip())
{
    // Run with: lila_bake_gpu_tests --no-skip -tc="GPU bake speed*"
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_speed");
    const char* resEnv = std::getenv("LILA_BAKE_BENCH_RES");
    const int res = resEnv ? std::atoi(resEnv) : 2048;
    const double A = 3.0, k = glm::two_pi<double>() / 25.0;
    const std::string low = addMesh("bench_low", plane(8));
    const std::string high = addMesh("bench_high", displaced(1000, [&](double x, double z) { return A * std::cos(k * x) * std::sin(k * z); }));
    const std::string p = project(dir, res, 4, "png");
    ops::execute("bakeGroup", {{"args", {"bench"}}, {"project", p}, {"lows", {low}}, {"highs", {high}}, {"frontal", 10.0}, {"rear", 10.0}});
    ops::execute("setBakeMap", {{"args", {"ao"}}, {"project", p}, {"options", {{"rays", 64}, {"distance", 20.0}}}});
    std::vector<std::string> backends = gpuBackends();
    if (!std::getenv("LILA_BAKE_BENCH_NO_CPU"))
        backends.push_back("cpu");
    for (const std::string& be : backends)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const char* mapsEnv = std::getenv("LILA_BAKE_BENCH_MAPS");
        const json maps = mapsEnv ? json::array({std::string(mapsEnv)}) : json::array({"normal", "ao"});
        const json r = ops::execute("bake", {{"project", p}, {"maps", maps}, {"backend", be}});
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        MESSAGE(be << ": " << res << "x" << res << " ss4 " << maps.dump() << " (ao 64 rays), 2M tri high: " << s << " s (bake " << r["seconds"].get<double>()
                   << " s)");
    }
}

namespace
{
/** As compare, for maps that store a unit vector (n * 0.5 + 0.5): the angle between texels, degrees. */
void compareAngle(const Baked& cpu, const Baked& gpu, const std::string& map, float tolDeg, float hardDeg, double outlierFraction)
{
    INFO("map " << map);
    REQUIRE(cpu.maps.count(map));
    REQUIRE(gpu.maps.count(map));
    const image::Image& a = cpu.maps.at(map);
    const image::Image& b = gpu.maps.at(map);
    REQUIRE(a.width == b.width);
    size_t outliers = 0;
    float worst = 0.0f;
    const size_t n = size_t(a.width) * size_t(a.height);
    for (size_t i = 0; i < n; ++i)
    {
        const float d = angleDegOrFlat(glm::vec3(texelAt(a, i)) * 2.0f - 1.0f, glm::vec3(texelAt(b, i)) * 2.0f - 1.0f);
        worst = std::max(worst, d);
        outliers += d > tolDeg ? 1 : 0;
    }
    INFO("worst " << worst << " deg, " << outliers << " of " << n << " texels over " << tolDeg);
    CHECK(double(outliers) <= outlierFraction * double(n));
    CHECK(worst <= hardDeg);
}

void checkMeta(const Baked& cpu, const Baked& gpu, const std::string& map, const char* key, double rel)
{
    INFO("map " << map << " meta " << key);
    const json a = meta(cpu, map), b = meta(gpu, map);
    REQUIRE(a.contains(key));
    REQUIRE(b.contains(key));
    CHECK(b[key].get<double>() == doctest::Approx(a[key].get<double>()).epsilon(rel));
}
} // namespace

TEST_CASE("GPU bake matches the CPU: thickness, bent normals, rounded edges, curvature family, gradient, normal to height")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_raymaps");
    geo::SphereOptions so;
    so.subdivAxis = 32;
    so.subdivHeight = 32;
    const std::string low = addMesh("ball_low", geo::makeSphere(so));
    geo::IcoSphereOptions io;
    io.subdivisions = 5;
    const std::string high = addMesh("ball_high", geo::makeIcoSphere(io));
    const std::string floorHigh = addMesh("floor_high", moved(plane(4), {0.0, -52.0, 0.0}));
    const std::string p = project(dir, 64, 4);
    ops::execute("bakeGroup",
                 {{"args", {"ball"}}, {"project", p}, {"lows", {low}}, {"highs", {high, floorHigh}}, {"frontal", 3.0}, {"rear", 3.0}});
    auto opts = [&](const char* map, const json& o) { ops::execute("setBakeMap", {{"args", {map}}, {"project", p}, {"options", o}}); };
    opts("thickness", {{"rays", 32}});
    opts("bentNormal", {{"rays", 32}, {"distance", 30.0}, {"greenUp", true}});
    opts("roundedNormal", {{"rays", 16}, {"radius", 4.0}, {"space", "object"}});
    opts("edgeWear", {{"noise", 0.5}, {"noiseScale", 4.0}});
    opts("gradient", {{"direction", {0.0, 1.0, 0.5}}});
    const std::vector<std::string> maps{"thickness", "bentNormal", "roundedNormal", "curvature", "cavity", "convexity", "normalToCurvature",
                                        "edgeWear", "gradient", "normalToHeight", "objectNormal"};
    const Baked cpu = bakeWith(p, maps, "cpu");
    for (const std::string& be : gpuBackends())
    {
        INFO("backend " << be);
        const Baked gpu = bakeWith(p, maps, be);
        CHECK(stat(gpu, "samples") == stat(cpu, "samples"));
        // 32 rays: one ray changing its mind at a grazing angle moves a sample by 1/32 of the range.
        compare(cpu, gpu, "thickness", 0.02f, 0.15f, 0.03);
        checkMeta(cpu, gpu, "thickness", "thicknessRange", 0.02);
        compareAngle(cpu, gpu, "bentNormal", 2.0f, 25.0f, 0.03);
        compareAngle(cpu, gpu, "roundedNormal", 3.0f, 40.0f, 0.05);
        compare(cpu, gpu, "curvature", 0.01f, 0.1f, 0.01);
        checkMeta(cpu, gpu, "curvature", "curvatureRange", 1e-3);
        compare(cpu, gpu, "cavity", 0.01f, 0.1f, 0.01);
        compare(cpu, gpu, "convexity", 0.01f, 0.1f, 0.01);
        compare(cpu, gpu, "normalToCurvature", 0.02f, 0.3f, 0.02);
        checkMeta(cpu, gpu, "normalToCurvature", "curvatureRange", 0.02);
        compare(cpu, gpu, "edgeWear", 0.02f, 0.5f, 0.02);
        compare(cpu, gpu, "gradient", 1e-3f, 0.02f, 0.005);
        compare(cpu, gpu, "normalToHeight", 0.02f, 0.2f, 0.02);
        compareAngle(cpu, gpu, "objectNormal", 0.05f, 1.0f, 0.005);
    }
}

TEST_CASE("GPU bake matches the CPU: material transfer, vertex colours, the high's normal map, UV maps and errors")
{
    if (gpuBackends().empty())
        return;
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_transfer");
    image::Image tex{4, 4, {}};
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            tex.rgba.insert(tex.rgba.end(), {x < 2 ? 1.0f : 0.2f, y < 2 ? 0.1f : 0.9f, 0.3f, x < 2 ? 0.5f : 1.0f});
    image::write(dir / "bc.png", tex, 8);
    const glm::vec3 tilt(std::sin(glm::radians(30.0f)), 0.0f, std::cos(glm::radians(30.0f)));
    image::Image nm{2, 2, {}};
    for (int i = 0; i < 4; ++i)
        nm.rgba.insert(nm.rgba.end(), {tilt.x * 0.5f + 0.5f, 0.5f, tilt.z * 0.5f + 0.5f, 1.0f});
    image::write(dir / "n.png", nm, 16);
    ops::execute("createMaterial", {{"args", {"M_Const"}}, {"baseColor", {0.2, 0.4, 0.6}}, {"roughness", 0.7}, {"metallic", 1.0},
                                    {"emissive", {0.5, 0.0, 0.0}}, {"emissiveIntensity", 2.0}});
    ops::execute("createMaterial", {{"args", {"M_Tex"}}, {"baseColorMap", (dir / "bc.png").generic_string()}, {"blendMode", "masked"},
                                    {"opacity", 0.8}, {"roughness", 0.25}, {"normalMap", (dir / "n.png").generic_string()}});
    // The high covers the middle of the low only (misses around it); a strip far above the left edge is beyond the rays.
    geo::Mesh hi = moved(plane(8), {0.0, 1.0, 0.0});
    for (geo::Index pt = 0; pt < hi.pointCapacity(); ++pt)
    {
        const glm::dvec3 q = hi.position(pt);
        hi.setPosition(pt, glm::dvec3(q.x * 0.6, q.y, q.z * 0.6));
    }
    std::vector<uint32_t> left, right;
    for (geo::Index f = 0; f < hi.faceCapacity(); ++f)
        (hi.faceCentroid(f).x < 0.0 ? left : right).push_back(f);
    scene::materials::assign(hi, "M_Const", left);
    scene::materials::assign(hi, "M_Tex", right);
    for (const char* set : {"first", "paint"})
    {
        std::span<glm::vec4> c = geo::colors::add(hi, set, glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        if (std::string(set) == "paint")
            for (geo::Index cn = 0; cn < hi.cornerCapacity(); ++cn)
            {
                const glm::dvec3 q = hi.position(hi.cornerPoint(cn));
                c[cn] = glm::vec4(float(q.x / 60.0 + 0.5), float(q.z / 60.0 + 0.5), 0.25f, 1.0f);
            }
    }
    geo::Mesh far = moved(plane(1), {-45.0, 20.0, 0.0});
    for (geo::Index pt = 0; pt < far.pointCapacity(); ++pt)
    {
        const glm::dvec3 q = far.position(pt);
        far.setPosition(pt, glm::dvec3(-45.0 + (q.x + 45.0) * 0.1, q.y, q.z));
    }
    geo::Mesh lowMesh = plane(2, {0.0f, 0.0f}, {0.45f, 1.0f});
    lowMesh.append(moved(plane(2, {0.55f, 0.0f}, {1.0f, 1.0f}), {0.0, 0.0, 0.0}));
    const std::string low = addMesh("t_low", std::move(lowMesh));
    const std::string high = addMesh("t_high", std::move(hi));
    const std::string farHigh = addMesh("far_high", std::move(far));
    const std::string p = project(dir, 48, 4, "png");
    ops::execute("bakeGroup", {{"args", {"t"}}, {"project", p}, {"lows", {low}}, {"highs", {high, farHigh}}, {"frontal", 3.0}, {"rear", 3.0}});
    auto opts = [&](const char* map, const json& o) { ops::execute("setBakeMap", {{"args", {map}}, {"project", p}, {"options", o}}); };
    opts("vertexColor", {{"set", "paint"}});
    opts("id", {{"source", "vertexColor"}});
    opts("wireframe", {{"width", 1.5}});
    const std::vector<std::string> maps{"baseColor", "roughness", "metallic", "emissive", "materialOpacity", "normal", "vertexColor",
                                        "id",        "wireframe", "islandBorder", "uvIslands", "error", "alpha"};
    const Baked cpu = bakeWith(p, maps, "cpu");
    for (const std::string& be : gpuBackends())
    {
        INFO("backend " << be);
        const Baked gpu = bakeWith(p, maps, be);
        CHECK(stat(gpu, "samples") == stat(cpu, "samples"));
        CHECK(stat(gpu, "misses") == stat(cpu, "misses"));
        for (const char* m : {"baseColor", "roughness", "metallic", "emissive", "materialOpacity", "vertexColor"})
            compare(cpu, gpu, m, 2.0f / 255.0f, 0.02f, 0.01);
        compare(cpu, gpu, "normal", 0.1f, 1.0f, 0.01);
        compare(cpu, gpu, "id", 0.0f, 0.0f, 0.0);
        compare(cpu, gpu, "wireframe", 0.0f, 1.0f, 0.01);
        compare(cpu, gpu, "islandBorder", 0.0f, 1.0f, 0.01);
        compare(cpu, gpu, "uvIslands", 0.0f, 0.0f, 0.0);
        compare(cpu, gpu, "error", 0.0f, 1.0f, 0.005);
        compare(cpu, gpu, "alpha", 0.0f, 0.0f, 0.0);
        CHECK(meta(gpu, "error")["texels"] == meta(cpu, "error")["texels"]);
    }
}

TEST_CASE("Async bakes run off the main thread, report each texture as it lands and can be cancelled")
{
    const std::string be = gpuBackends().empty() ? std::string("cpu") : gpuBackends().back();
    newScene();
    const fs::path dir = tempDir("lila_bake_gpu_async");
    const std::string low = addMesh("as_low", plane(1));
    const std::string high = addMesh("as_high", moved(plane(8), {0.0, 1.0, 0.0}));
    const std::string p = project(dir, 64, 4, "png");
    ops::execute("bakeGroup", {{"args", {"as"}}, {"project", p}, {"lows", {low}}, {"highs", {high}}, {"frontal", 3.0}, {"rear", 3.0}});
    // Events of earlier bakes still queued for the main thread go first.
    mainq::drain();
    std::vector<bake::BakeEvent> events;
    const int listener = bake::addBakeListener([&](const bake::BakeEvent& e) { events.push_back(e); });
    auto waitIdle = [] {
        const auto t0 = std::chrono::steady_clock::now();
        while (bake::bakeRunning() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        mainq::drain();
    };

    const json started = ops::execute("bake", {{"project", p}, {"maps", {"normal", "height", "ao"}}, {"backend", be}, {"async", true}});
    CHECK(started.value("async", false));
    waitIdle();
    REQUIRE(!events.empty());
    CHECK(events.front().kind == bake::BakeEvent::Kind::Started);
    CHECK(events.back().kind == bake::BakeEvent::Kind::Finished);
    size_t textures = 0;
    for (const bake::BakeEvent& e : events)
        if (e.kind == bake::BakeEvent::Kind::Texture)
        {
            ++textures;
            CHECK(fs::exists(fs::path(e.data["file"].get<std::string>())));
        }
    CHECK(textures == 3);
    const json done = events.back().data;
    REQUIRE(done.contains("outputs"));
    CHECK(done["outputs"].size() == 3);
    CHECK(done["backend"] == (be == "cpu" ? "cpu" : "gpu"));
    const json status = ops::execute("bakeStatus", {{"project", p}});
    CHECK(status["last"]["outputs"].size() == 3);
    CHECK_FALSE(status["running"].get<bool>());

    // A long bake stops on bakeCancel and says so.
    events.clear();
    ops::execute("setBakeOutput", {{"project", p}, {"resolution", 2048}, {"supersampling", 16}});
    ops::execute("setBakeMap", {{"args", {"ao"}}, {"project", p}, {"options", {{"rays", 4096}}}});
    ops::execute("bake", {{"project", p}, {"maps", {"ao"}}, {"backend", be}, {"async", true}});
    CHECK(bake::bakeRunning());
    CHECK_THROWS(ops::execute("bake", {{"project", p}, {"maps", {"ao"}}, {"backend", be}, {"async", true}}));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(ops::execute("bakeCancel", json::object()).get<bool>());
    waitIdle();
    REQUIRE(!events.empty());
    CHECK(events.back().kind == bake::BakeEvent::Kind::Finished);
    CHECK(events.back().data.value("cancelled", false));
    bake::removeBakeListener(listener);
}

TEST_CASE("Bake benchmark: Painter maps at 2K from 1M triangles, 4K normal and AO from 10M" * doctest::skip())
{
    // Run with: lila_bake_gpu_tests --no-skip -tc="Bake benchmark*" (LILA_BAKE_BENCH_CHECK=1 also checks the budgets).
    if (gpuBackends().empty())
        return;
    const bool check = std::getenv("LILA_BAKE_BENCH_CHECK") != nullptr;
    auto scene = [](int axis, int height) {
        newScene();
        geo::SphereOptions lo;
        lo.subdivAxis = 32;
        lo.subdivHeight = 16;
        addMesh("ball_low", geo::makeSphere(lo));
        geo::SphereOptions hi;
        hi.subdivAxis = axis;
        hi.subdivHeight = height;
        addMesh("ball_high", geo::makeSphere(hi));
    };
    auto run = [&](const std::string& what, const json& call, double budget) {
        for (int pass = 0; pass < 2; ++pass)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const json r = ops::execute(call.contains("target") ? "bakeForTarget" : "bake", call);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const std::string line = what + (pass ? " (rebake)" : "") + ": " + std::to_string(s) + " s on " + r.value("backendMode", std::string()) +
                                     ", " + std::to_string(r["stats"]["groups"][0]["highTriangles"].get<uint64_t>()) + " high triangles; timings " +
                                     r["timings"].dump();
            MESSAGE(line);
            if (check)
                CHECK(s < budget);
        }
    };

    scene(1000, 500);
    fs::path dir = tempDir("lila_bake_gpu_bench_painter");
    std::string p = ops::execute("createBakeProject", {{"name", "P"}}).get<std::string>();
    ops::execute("setBakeOutput", {{"project", p}, {"folder", dir.generic_string()}});
    run("Painter, 8 maps, 2048, ss4", {{"project", p}, {"target", "substancePainter"}, {"backend", gpuBackends().back()}}, 3.0);

    scene(3200, 1600);
    dir = tempDir("lila_bake_gpu_bench_10m");
    p = ops::execute("createBakeProject", {{"name", "P"}}).get<std::string>();
    ops::execute("setBakeOutput", {{"project", p}, {"folder", dir.generic_string()}, {"resolution", 4096}});
    ops::execute("autoGroupBake", {{"project", p}});
    run("normal + AO, 4096, ss4", {{"project", p}, {"maps", {"normal", "ao"}}, {"backend", gpuBackends().back()}}, 10.0);
}
