#include "render/Renderer.h"

#include "core/Log.h"
#include "core/Paths.h"
#include "core/Profile.h"
#include "render/Backdrop.h"
#include "render/Compute.h"
#include "render/DrawBatches.h"
#include "render/Environment.h"
#include "render/GpuImage.h"
#include "render/GpuProfile.h"
#include "render/HdrImage.h"
#include "render/DisplayOptions.h"
#include "render/Surface.h"
#include "render/VirtualTexture.h"
#include "render/VulkanContext.h"
#include "render/VulkanRequirements.h"

#include <glm/gtc/packing.hpp>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <set>
#include <span>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace lila::render
{
namespace
{
constexpr int kFramesInFlight = 2;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr int kBrdfLutSize = 128;
constexpr float kXrayOpacity = 0.35f;

struct GridPush
{
    glm::mat4 invViewProj;
    glm::mat4 viewProj;
    glm::vec4 eye;
    glm::vec4 params;
    glm::vec4 color;
    glm::vec4 axisX;
    glm::vec4 axisZ;
    glm::vec4 origin;
};

struct MeshPush
{
    glm::mat4 viewProj;
    /** The device address of the frame's instance buffer (GpuInstance entries) and this draw's first entry in it. */
    uint64_t instances = 0;
    uint32_t firstInstance = 0;
    uint32_t instancePad = 0;
    /** meshWirePS: the device address of the mesh's edge mask (RenderMesh::edgeMask), the section's first triangle and the wire width in pixels. */
    uint64_t edgeMask = 0;
    uint32_t firstTriangle = 0;
    float wireWidth = 1.0f;
    /** The frame's lights (GpuLight entries) and how many. */
    uint64_t lights = 0;
    uint32_t lightCount = 0;
    uint32_t lightPad = 0;
    glm::vec4 extra{}; // live bevel radius, 1 when the bevel pass ran, environment light saturation
    glm::vec4 eye;
    glm::vec4 base;
    glm::vec4 params;
    glm::vec4 env;
    glm::vec4 surface;
    glm::vec3 emissive{0.0f};
    uint32_t flags = 0;
    float gamma = 1.0f;
    /** Word offset of the draw's style block (stylised shading models) in the frame's table, or kNoStyle. */
    uint32_t style = 0xffffffffu;
    /** Back-face tint: display-encoded rgb and mix, 8 bits each; 0 is off. */
    uint32_t backTint = 0;
    /** Material::sampling. */
    uint32_t sampling = 0;
    /** The device address of the frame's table: bindless texture handles, then style blocks. Maps hold 16-bit handles into it. */
    uint64_t textureTable = 0;
    uint32_t maps[2] = {0xffffffffu, 0xffffffffu}; // base color | normal << 16, ORM | emissive << 16
};
static_assert(sizeof(MeshPush) == 256, "matches MeshPush in mesh.slang");

struct GpuMesh;

/** One material section of a shaded batch, its push constants worked out once for as long as the frame draws the same list. */
struct SectionRecord
{
    uint32_t batch = 0; // in DrawBatches::shaded
    MeshSection section;
    GpuMesh* mesh = nullptr;
    MeshPush push;
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool material = false;    // reads the frame's texture table
    bool translucent = false; // its material blends
    bool lean = false;        // draws with the lean pipelines (see leanPush)
};

/** What section records depend on besides the draws: frame settings, the texture table and the GPU mesh cache. */
struct RecordKey
{
    glm::vec4 params{0.0f}, env{0.0f}, backTint{0.0f};
    float gamma = 0.0f, saturation = 0.0f;
    uint32_t lightCount = 0;
    bool bevel = false, grey = false, twoSided = false, xray = false, table = false, instances = false, bindless = false, wire = false;
    uint64_t uploads = 0, evictions = 0;
    std::vector<uint32_t> textureTable;

    bool operator==(const RecordKey&) const = default;
};

// Draw-level bits above the material flags, matching mesh.slang.
constexpr uint32_t kVertexColorShaded = 1u << 11;
constexpr uint32_t kVertexColorUnlit = 1u << 12;
constexpr uint32_t kColorChannelShift = 13;
constexpr uint32_t kOneSidedLighting = 1u << 16;
constexpr uint32_t kNormalXY = 1u << 17; // two-channel normal map: Z rebuilt in the shader
constexpr uint32_t kGreyBase = 1u << 18;  // base colour map supplies opacity only

// Placeholder textures bound where a material has no map; never evicted.
constexpr uint64_t kWhiteSrgbKey = ~uint64_t(0);
constexpr uint64_t kWhiteLinearKey = ~uint64_t(0) - 1;
constexpr uint64_t kFlatNormalKey = ~uint64_t(0) - 2;
constexpr uint32_t kMaxMaterialSets = 1024;
/** Texture table entries: a bindless index, or a UDIM set's header (this bit, then its row count). */
constexpr uint32_t kUdimEntry = 0x80000000u;
constexpr uint32_t kNoSlot = 0xffffffffu;
constexpr uint32_t kNoHandle = 0xffffu;
constexpr uint32_t kMaxBindless = 16384;
constexpr uint32_t kNoStyle = 0xffffffffu;
constexpr uint32_t kMaterialBindings = 5; // base color, normal, ORM, emissive, stroke sheet

/**
 * A section whose shading reads no tangent, vertex colour or second uv: the lean pipelines specialise those reads away,
 * so the driver neither fetches them nor passes them on, which dense meshes are bound by.
 */
bool leanPush(uint32_t flags, uint32_t style)
{
    return style == kNoStyle && (flags & (MatNormalMap | MatUv1 | kVertexColorShaded | kVertexColorUnlit)) == 0;
}

struct ContactPush
{
    glm::mat4 viewProj;
    glm::vec4 center;
    glm::vec4 axisA;
    glm::vec4 axisB;
};

struct AoPush
{
    glm::mat4 invProj;
    glm::mat4 view;
    glm::vec4 params; // radius, intensity, pixels per world unit at depth 1, orthographic
    glm::vec4 size;   // extent, 1 / extent
    glm::vec4 options{0.0f}; // x: 1 to take normals from the depth instead of the prepass
};

constexpr VkFormat kAoNormalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kAoFormat = VK_FORMAT_R32_SFLOAT;
constexpr uint32_t kMaxAoSets = 512;

/** The scene target's format: compressed linear radiance (see scene_color.slang), MSAA, blended and resolved. */
constexpr VkFormat kSceneFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kSceneFallbackFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
/** Temporal antialiasing history, ping-ponged per viewport. */
constexpr VkFormat kHistoryFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

struct TaaPush
{
    glm::mat4 invViewProj;
    glm::mat4 prevViewProj;
    glm::vec4 size;
    glm::vec4 params; // weight of this frame, clip the history, ignore the history
};

struct ComposePush
{
    glm::vec4 params; // display gamma, dither, exposure, tonemapper
    glm::vec4 grade;  // contrast, saturation, post stack on, AgX look
    glm::vec4 white;  // white balance, LUT intensity
    glm::vec4 lift;   // rgb, vignette
    glm::vec4 gamma;  // rgb, grain
    glm::vec4 gain;   // rgb, grain seed
    glm::vec4 lens;   // chromatic aberration, distortion, sharpen, bloom
    glm::vec4 size;
    glm::vec4 misc;   // auto exposure on, bloom scale
};

struct BloomPush
{
    glm::vec4 size;   // destination extent, source texel size
    glm::vec4 params; // first level, tent radius
};

struct ExposurePush
{
    glm::vec4 size;
    glm::vec4 range; // first and last bin's log2 luminance, lowest and highest exposure (stops)
    glm::vec4 adapt; // blend this frame, compensation, low and high percentile
};

/** One light as mesh.slang reads it. */
struct GpuLight
{
    glm::vec4 position;  // xyz, w kind
    glm::vec4 direction; // xyz where the light travels, w range (cm, 0 none)
    glm::vec4 radiance;  // rgb, w radius (cm)
    glm::vec4 cone;      // cos outer, 1 / (cos inner - cos outer), width, height
    glm::vec4 right;     // xyz area width axis, w shadow texel (cm)
    glm::mat4 shadow;    // world to the light's shadow clip space
    glm::vec4 shadowRect; // atlas uv offset xy, scale zw; zw 0 without a shadow
    glm::vec4 shadowDepth; // LightDraw::shadowDepth
};
static_assert(sizeof(GpuLight) == 176, "matches GpuLight in mesh.slang");

/** The light buffer's first entry, ahead of the lights: the frame's fog. GpuLight sized, so the lights stay aligned. */
struct GpuFrame
{
    glm::vec4 fogColor;  // rgb, w density per cm (0 off)
    glm::vec4 fogParams; // height, falloff per cm, start, sun scatter
    glm::vec4 fogExtra;  // anisotropy, up axis (1 Y, 2 Z)
    glm::vec4 shadowing; // one past the light each traced shadow channel is for, 0 for none
    glm::vec4 pad[7];
};
static_assert(sizeof(GpuFrame) == sizeof(GpuLight), "the light buffer's header is one light long");

/** A traced placement's mesh for rtbevel.slang: its buffers' addresses and world matrix rows. */
struct TraceInstance
{
    uint64_t vertices = 0;
    uint64_t indices = 0;
    glm::vec4 rows[3]{};
};

/** Shadow maps share one atlas of 2 by 2 tiles; lights past four cast none. */
constexpr uint32_t kShadowAtlas = 4096;
constexpr uint32_t kShadowTile = 2048;
constexpr uint32_t kMaxLights = 256;

constexpr uint32_t kMaxBloomLevels = 6;
constexpr uint32_t kBloomSets = 3 + 2 * (kMaxBloomLevels - 1); // first level per compose input, then the rest down, then up
constexpr VkFormat kBloomFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

/**
 * Sub-pixel camera offset for sample i of the temporal sequence, in pixels: none for the first, so a single frame matches
 * an unjittered one, then Halton (2, 3) points around the pixel centre.
 */
glm::vec2 jitterOffset(uint32_t i)
{
    if (i == 0)
        return glm::vec2(0.0f);
    auto halton = [](uint32_t n, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        for (; n > 0; n /= base)
        {
            f /= float(base);
            r += f * float(n % base);
        }
        return r;
    };
    return {halton(i, 2) - 0.5f, halton(i, 3) - 0.5f};
}

/** Point i of the aperture sequence on the unit disc, or on a regular polygon of `blades` sides (3 or more). */
glm::vec2 apertureSample(uint32_t i, int blades, float rotation)
{
    auto halton = [](uint32_t n, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        for (; n > 0; n /= base)
        {
            f /= float(base);
            r += f * float(n % base);
        }
        return r;
    };
    const float u = halton(i + 1, 5), w = halton(i + 1, 7);
    constexpr float kTau = 6.28318531f;
    if (blades < 3)
    {
        // Concentric (Shirley) mapping keeps the sequence's spacing on the disc.
        const float a = 2.0f * u - 1.0f, b = 2.0f * w - 1.0f;
        if (a == 0.0f && b == 0.0f)
            return glm::vec2(0.0f);
        const float r = std::abs(a) > std::abs(b) ? a : b;
        const float phi = std::abs(a) > std::abs(b) ? 0.785398f * (b / a) : 1.570796f - 0.785398f * (a / b);
        return r * glm::vec2(std::cos(phi), std::sin(phi));
    }
    // A triangle of the polygon by u, then a uniform point in it.
    const float sector = std::floor(u * float(blades));
    const float t = u * float(blades) - sector;
    const float a0 = (sector / float(blades)) * kTau + glm::radians(rotation), a1 = a0 + kTau / float(blades);
    float s1 = std::sqrt(t), s2 = w;
    return s1 * (1.0f - s2) * glm::vec2(std::cos(a0), std::sin(a0)) + s1 * s2 * glm::vec2(std::cos(a1), std::sin(a1));
}

struct PostPush
{
    glm::vec4 size;
    glm::vec4 params;  // Kuwahara on, radius, outlines on, outline width
    glm::vec4 outline; // colour, depth sensitivity
};

struct ExternalPush
{
    glm::vec4 size;   // image extent, then its inverse
    glm::vec4 projZ;  // proj[2][2], proj[3][2], proj[2][3], proj[3][3]: view-space z to clip z and w
    glm::vec4 params; // x: 1 when there is depth (2 for window depth), y: rows bottom up, z: premultiplied, w: colour multiplier
    glm::vec4 unproject;
};

struct WirePush
{
    glm::mat4 viewProj;
    glm::vec4 depthPull{1.0f, 0.0f, 0.0f, 0.0f};
    uint64_t instances = 0; // as in MeshPush; each entry's color is its wire colour
    uint32_t firstInstance = 0;
    uint32_t instancePad = 0;
};
static_assert(sizeof(WirePush) == 96, "matches WirePush in wire.slang");

struct LinesPush
{
    glm::mat4 viewProj;
    glm::vec4 depthPull{1.0f, 0.0f, 0.0f, 0.0f};
};

struct PointsPush
{
    glm::mat4 viewProj;
    glm::vec4 depthPull{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 colors[3]{};
    glm::vec4 sizes{4.0f, 6.0f, 0.0f, 0.0f}; // device pixels: normal, selected
};
static_assert(sizeof(PointsPush) == 144, "matches PointsPush in points.slang");

struct PointsCullPush
{
    glm::mat4 viewProj;
    glm::uvec4 grid{0}; // cells across and down, cell size in pixels, points
    glm::vec4 viewport{0.0f};
    VkDeviceAddress points = 0, cells = 0, out = 0, args = 0;
};
static_assert(sizeof(PointsCullPush) == 128, "matches CullPush in points_cull.slang");

/** Pixels' worth of world distance that lines over meshes are pulled toward the camera, plus half their width. */
constexpr double kLinePullPixels = 2.0;

/**
 * Clip-space z scale (x) and w factor (y) that move a point toward the camera along its own view ray by `pixels`
 * pixels' worth of world distance at its depth, leaving its screen position alone. Rasterizer depth bias never reaches
 * line primitives, and an edge in a crease loses the depth test to the faces either side of it without this.
 */
glm::vec4 depthPull(const glm::dmat4& proj, uint32_t heightPx, double pixels)
{
    // World size of one pixel: at unit distance for perspective, everywhere for orthographic.
    const double pixel = 2.0 / (std::abs(proj[1][1]) * double(std::max(heightPx, 1u)));
    const double a = proj[2][2];
    if (proj[3][3] != 0.0)
        return {1.0f, float(a * pixels * pixel), 0.0f, 0.0f};
    const double scale = 1.0 / (1.0 - std::min(pixels * pixel, 0.1));
    return {float(scale), float(a * (scale - 1.0)), 0.0f, 0.0f};
}

/**
 * Clips a world-space line list to the view volume, with x and y given twice the view's room. The rasterizer
 * clips only z exactly, so a wide line crossing the near plane otherwise reaches it with an end far outside its guard
 * band and floods the view. Reads v and writes only the moved ends into out, which already holds a copy (mapped memory
 * is slow to read). Lines wholly outside move to `behind`, a point behind the camera that every view clips.
 */
void clipLineList(const LineVertex* v, LineVertex* out, size_t count, const glm::dmat4& viewProj, const glm::vec3& behind)
{
    for (size_t i = 0; i + 1 < count; i += 2)
    {
        const glm::dvec3 a(v[i].pos), b(v[i + 1].pos);
        const glm::dvec4 ca = viewProj * glm::dvec4(a, 1.0), cb = viewProj * glm::dvec4(b, 1.0);
        // Inside where each is >= 0: the x and y guard planes, the far plane (reversed-Z z >= 0) and the near one (z <= w).
        auto planes = [](const glm::dvec4& c) {
            constexpr double kGuard = 2.0;
            return std::array<double, 6>{kGuard * c.w + c.x, kGuard * c.w - c.x, kGuard * c.w + c.y, kGuard * c.w - c.y, c.z, c.w - c.z};
        };
        const std::array<double, 6> da = planes(ca), db = planes(cb);
        double t0 = 0.0, t1 = 1.0;
        for (size_t k = 0; k < da.size() && t0 <= t1; ++k)
        {
            if (da[k] < 0.0 && db[k] < 0.0)
                t0 = 2.0;
            else if (da[k] < 0.0)
                t0 = std::max(t0, da[k] / (da[k] - db[k]));
            else if (db[k] < 0.0)
                t1 = std::min(t1, da[k] / (da[k] - db[k]));
        }
        if (t0 > t1)
        {
            out[i].pos = out[i + 1].pos = behind;
            continue;
        }
        if (t0 > 0.0)
            out[i].pos = glm::vec3(glm::mix(a, b, t0));
        if (t1 < 1.0)
            out[i + 1].pos = glm::vec3(glm::mix(a, b, t1));
    }
}

struct ImagePlanePush
{
    glm::mat4 viewProj;
    glm::vec4 corners[4];
    glm::vec4 uv[2]; // per corner: xy, zw
    glm::vec4 params; // opacity, brightness, contrast, grayscale
    glm::vec4 texel;
    glm::vec4 outline; // rgb: line color, display-encoded; w: 1 for outline mode
    glm::vec4 key;     // rgb: background to clear, display-encoded; w: tolerance, 0 for none
};
static_assert(sizeof(ImagePlanePush) == 224, "matches ImagePlanePush in imageplane.slang");

struct AllocatedBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
};

struct HeightfieldPush
{
    glm::mat4 viewProj;
    glm::vec4 model0, model1, model2;
    glm::vec4 grid;
    glm::vec4 origin;
    glm::vec4 color;
    glm::vec4 light;
    uint64_t heights = 0;
    uint32_t cols = 0;
    uint32_t rows = 0;
};
static_assert(sizeof(HeightfieldPush) == 192, "matches HeightfieldPush in heightfield.slang");

/** A heightfield's heights on the GPU, read by heightfieldVS through their address. */
struct GpuHeights
{
    AllocatedBuffer buffer;
    VkDeviceAddress address = 0;
    std::shared_ptr<const std::vector<float>> keep;
    uint64_t lastUsed = 0;
};

struct GpuMesh
{
    AllocatedBuffer vertices, indices, edges, edgeIndices;
    uint32_t indexCount = 0;
    uint32_t edgePointCount = 0;
    uint32_t edgeIndexCount = 0; // 0 draws edges as a plain line list
    compute::Buffer edgeMask;    // RenderMesh::edgeMask, read by meshWirePS through its address
    bool sharedVertices = false; // the vertex buffer belongs to the shareVertices mesh's entry
    /** The mesh's bottom-level acceleration structure for ray-traced effects; built the first time one needs it. */
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    compute::Buffer blasBuffer;
    VkDeviceAddress blasAddress = 0;
    std::shared_ptr<const RenderMesh> keep;
    uint64_t lastUsed = 0;
};

struct GpuPoints
{
    AllocatedBuffer buffer;
    VkDeviceAddress address = 0; // for thinning a dense cloud (points_cull.slang)
    uint32_t count = 0;
    std::shared_ptr<const PointCloud> keep;
    uint64_t lastUsed = 0;
};

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct Frame
{
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkBuffer lines = VK_NULL_HANDLE;
    VmaAllocation linesAlloc = VK_NULL_HANDLE;
    void* linesMapped = nullptr;
    size_t linesCapacity = 0;
    std::vector<AllocatedBuffer> staging; // texture uploads recorded into this frame, freed once its fence signals
    compute::Buffer table;       // bindless texture table, written before the frame is recorded
    compute::Buffer instances;   // GpuInstance per mesh placement, written before the frame is recorded
    compute::Buffer lights;      // GpuLight per light
    compute::Buffer tlasInstances; // the scene's placements for its top-level acceleration structure
    compute::Buffer traceTable;    // TraceInstance per placement, by its custom index
    std::vector<compute::Buffer> scratch; // acceleration structure builds' scratch, freed with the staging
    vt::Feedback feedback;       // virtual texture page requests, read once the fence signals
    // GPU timestamps: frame start, before and after ambient occlusion, frame end.
    VkQueryPool timestamps = VK_NULL_HANDLE;
    bool timed = false, aoTimed = false;
    /** Dense component dots thinned to one per screen cell (points_cull.slang): cell grids, kept points, draw counts. */
    compute::Buffer pointCells, pointOut, pointArgs;
    /** Secondary command buffers the section draws record into on worker threads; a pool each, so no two threads share one. */
    struct Chunk
    {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
    };
    std::vector<Chunk> chunks;
};

struct Viewport
{
    uintptr_t window = 0;
    SurfaceWindow surfaceWindow;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_SRGB;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkSemaphore> rendered; // per swapchain image
    Image color;                       // MSAA target, resolved into the swapchain image; none at one sample
    Image depth;
    // Ambient occlusion: single-sample prepass, raw GTAO, denoised result sampled by the mesh pass.
    Image aoDepth, aoNormal, aoRaw, aoFinal;
    VkDescriptorSet gtaoSet = VK_NULL_HANDLE, denoiseSet = VK_NULL_HANDLE, aoSampleSet = VK_NULL_HANDLE;
    // Viewport post: the scene resolved here, then filtered into the frame with the prepass depth and normals.
    Image postColor;
    VkDescriptorSet postSet = VK_NULL_HANDLE;
    // The scene target (MSAA, resolved into scene), its resolved depth, and the temporal history read by the compose pass.
    Image sceneMs, scene, sceneDepth;
    std::array<Image, 2> history;
    VkDescriptorSet taaSets[2][2]{};  // writes history[i] from history[1 - i]; [1] reads aoDepth (single-sample fast AO)
    VkDescriptorSet composeSets[3]{}; // reads scene, history[0], history[1]
    int historyIndex = 0;             // the one written last
    bool historyValid = false;
    /** Frames blended into the history since the view last changed, and the next sample of the jitter sequence. */
    uint32_t accumulated = 0, jitterIndex = 0;
    glm::dmat4 prevViewProj{1.0}, prevView{1.0}, prevProj{1.0};
    uint64_t submitted = 0, renderingSerial = 0, drawnSerial = 0; // submit count, the one being drawn, the one drawn last
    uint64_t uploadEpoch = 0;                                        // Impl::uploadEpoch when the last frame was drawn
    // Post: the bloom chains (half resolution, a view per level), the auto exposure histogram and its eased state.
    Image bloomDown, bloomUp, histogram, exposureState;
    std::vector<VkImageView> bloomDownViews, bloomUpViews;
    uint32_t bloomLevels = 0;
    VkDescriptorSet bloomSets[kBloomSets]{};
    VkDescriptorSet exposureSets[3]{}; // reading scene, history[0], history[1]
    bool postImagesReady = false;      // moved out of UNDEFINED into GENERAL
    bool exposureValid = false;
    uint64_t lutKey = 0;               // the LUT the compose sets point at
    std::chrono::steady_clock::time_point lastExposure{};
    bool wantsMore = false;
    /** Still accumulating with no new state: the next of those frames draws at convergeAt, one per display refresh. */
    bool converging = false;
    std::chrono::steady_clock::time_point convergeAt{};
    std::chrono::duration<double> refreshInterval{1.0 / 60.0};
    std::chrono::steady_clock::time_point refreshCheck{}; // the monitor's rate is read again after this, as windows move
    /** A new state waits until this to draw, which holds the view to its frame cap. */
    std::chrono::steady_clock::time_point drawAt{};
    bool vsync = false; // the present mode the swapchain was made with
    VkDescriptorSet rtaoSet = VK_NULL_HANDLE, bevelSet = VK_NULL_HANDLE, shadowSet = VK_NULL_HANDLE;
    uint64_t rtaoGeneration = 0, bevelGeneration = 0, shadowGeneration = 0; // Impl::tlasGeneration the sets point at
    Image bevel;    // rounded normals from the bevel pass
    Image rtShadow; // traced visibility, a channel per light                  // still accumulating: draw again without a new state
    std::array<Frame, kFramesInFlight> frames;
    int frameIndex = 0;

    int width = 0, height = 0;
    std::atomic<bool> needsRecreate = true;
    bool pending = false;
    bool hasState = false; // nothing is drawn before the first submit
    int retries = 0;       // redraws in a row after an out-of-date swapchain
    int msaaGeneration = -1; // Impl::msaaGeneration the targets were made for
    FrameState state;
    /** The last frame's mesh draws and their batches, reused while a new frame draws the same list. Render thread only. */
    DrawList keptMeshes;
    DrawBatches keptBatches;
    bool keptShaderWire = false;
    std::vector<uint32_t> keptDynamic; // indices of the kept draws whose meshes deform
    std::vector<SectionRecord> keptRecords;
    /** The GPU meshes the kept records draw from, with the meshes they share vertices with; pointers into meshCache. */
    std::vector<GpuMesh*> keptGpuMeshes;
    /** The kept draws' material maps, each once, in the order the draws first use them, with their placeholders. */
    std::vector<std::pair<TexturePtr, uint64_t>> keptMaps;
    std::vector<TexturePtr> keptTextures;
    bool keptMapsValid = false;
    std::vector<ContactPush> keptContacts; // without viewProj
    glm::vec2 keptContactKey{-1.0f};
    bool keptContactsValid = false;
    RecordKey keptRecordKey;
    bool keptRecordsValid = false;
    Renderer::FrameStats timing; // render thread only; copied to stats under the mutex
    Renderer::FrameStats stats;

    // Offscreen captures draw into an owned image instead of a swapchain and read it back into a host buffer.
    bool offscreen = false;
    Image offscreenTarget;
    AllocatedBuffer readback;
    void* readbackMapped = nullptr;
    bool readbackWanted = true; // false lets an offscreen frame go without copying the picture back
    bool hdrWanted = false;
    AllocatedBuffer hdrReadback;
    void* hdrMapped = nullptr;
};

struct OffscreenJob
{
    FrameState state;
    int width = 0, height = 0;
    bool done = false, ok = false;
    std::vector<uint8_t> pixels;
    std::string error;
    Renderer::FrameStats timing;
    int repeat = 1;
    int samples = 1; // jittered frames averaged into the result
    /** Frames drawn in order instead of `state`, each `samples` times, all averaged into one image. */
    std::vector<FrameState> frames;
    bool hdr = false;         // also read back the scene's linear light
    std::vector<float> hdrPixels; // RGBA, scene-linear (exposed), alpha coverage

    const std::string& envPath() const { return frames.empty() ? state.envPath : frames.front().envPath; }
};

std::vector<uint32_t> loadSpirv(const char* name)
{
    std::string err;
    std::vector<uint32_t> code = compute::loadSpirv(name, &err);
    if (code.empty())
        log::error("render: {}", err);
    return code;
}

/** fromDisplay in scene_color.slang: a display-linear colour as the scene target stores it, for its clear colour. */
glm::vec3 sceneFromDisplay(glm::vec3 c, float gamma)
{
    for (int i = 0; i < 3; ++i)
        c[i] = std::clamp(std::pow(std::max(c[i], 0.0f), std::max(gamma, 0.01f)), 0.0f, 1.0f);
    constexpr float start = 0.8f - 0.04f, d = 1.0f - start, desaturation = 0.15f, maxPeak = 0.9f;
    float newPeak = std::max(c.r, std::max(c.g, c.b));
    if (newPeak > maxPeak)
    {
        c *= maxPeak / newPeak;
        newPeak = maxPeak;
    }
    if (newPeak >= start)
    {
        const float peak = d * d / (1.0f - newPeak) - d + start;
        const float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
        c = glm::max((c - g * newPeak) / (1.0f - g), glm::vec3(0.0f)) * (peak / newPeak);
    }
    const float y = std::min(c.r, std::min(c.g, c.b));
    float offset = 0.04f;
    if (y < 0.04f)
    {
        const float x = std::sqrt(std::max(y, 0.0f) / 6.25f);
        offset = x - 6.25f * x * x;
    }
    c += offset;
    return c / (1.0f + std::max(c.r, std::max(c.g, c.b)));
}

glm::vec3 srgbToLinear(glm::vec3 c)
{
    for (int i = 0; i < 3; ++i)
        c[i] = c[i] <= 0.04045f ? c[i] / 12.92f : std::pow((c[i] + 0.055f) / 1.055f, 2.4f);
    return c;
}
} // namespace

struct Renderer::Impl
{
    VulkanContext vk;
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    bool quit = false;
    bool started = false;

    std::map<ViewportId, std::unique_ptr<Viewport>> viewports; // guarded by mutex for add/remove/state
    std::vector<ViewportId> toRemove;
    std::condition_variable removed;
    std::deque<std::shared_ptr<OffscreenJob>> offscreenJobs; // guarded by mutex
    std::condition_variable offscreenDone;
    struct QueuedTask
    {
        GpuTask step;
        std::promise<void> done;
    };
    std::deque<std::unique_ptr<QueuedTask>> tasks; // guarded by mutex
    std::vector<std::function<void(VulkanContext&)>> stopHooks;
    ViewportId nextId = 1;

    VkPipelineLayout gridLayout = VK_NULL_HANDLE, linesLayout = VK_NULL_HANDLE, overlayTrisLayout = VK_NULL_HANDLE;
    VkPipeline gridPipeline = VK_NULL_HANDLE, linesPipeline = VK_NULL_HANDLE, overlayPipeline = VK_NULL_HANDLE;
    VkPipeline overlayTrisPipeline = VK_NULL_HANDLE;
    VkPipelineLayout meshLayout = VK_NULL_HANDLE, wireLayout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE, wirePipeline = VK_NULL_HANDLE, wireXrayPipeline = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, GpuMesh> meshCache; // render thread only
    std::unordered_map<uint64_t, GpuPoints> pointCache; // render thread only
    VkPipelineLayout pointsLayout = VK_NULL_HANDLE, pointsCullLayout = VK_NULL_HANDLE;
    VkPipeline pointsNearestPipeline = VK_NULL_HANDLE, pointsKeepPipeline = VK_NULL_HANDLE;
    /** By s.points index: where that cloud's thinned points went this frame (PointsCull), or none when it draws whole. */
    std::vector<int> frameCulledPoints;
    VkPipeline pointsPipeline = VK_NULL_HANDLE;
    uint64_t frameCounter = 0;
    VkDescriptorSetLayout atlasSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet atlasSet = VK_NULL_HANDLE;
    VkSampler linearSampler = VK_NULL_HANDLE;
    Image atlasImage;
    const OverlayAtlas* uploadedAtlas = nullptr;
    std::shared_ptr<const OverlayAtlas> atlasKeepAlive;
    VkFormat pipelineFormat = VK_FORMAT_UNDEFINED;
    int msaaGeneration = 0; // bumped whenever vk.msaa changes; viewports rebuild their targets to match
    VkPipeline backgroundPipeline = VK_NULL_HANDLE, linesOnTopPipeline = VK_NULL_HANDLE, tris3dPipeline = VK_NULL_HANDLE;
    /** Lines at a set width over every surface (debug drawing's on-top lines). */
    VkPipeline linesOverPipeline = VK_NULL_HANDLE;
    VkPipelineLayout contactLayout = VK_NULL_HANDLE;
    VkPipeline contactPipeline = VK_NULL_HANDLE;
    VkPipelineLayout postLayout = VK_NULL_HANDLE;
    VkPipeline postPipeline = VK_NULL_HANDLE;
    VkFormat sceneFormat = kSceneFormat;
    VkPipelineLayout taaLayout = VK_NULL_HANDLE, composeLayout = VK_NULL_HANDLE;
    VkPipeline taaPipeline = VK_NULL_HANDLE, composePipeline = VK_NULL_HANDLE;
    VkPipeline composeSinglePipeline = VK_NULL_HANDLE; // into postColor: one sample, no depth
    VkDescriptorSetLayout composeSetLayout = VK_NULL_HANDLE, postComputeLayout = VK_NULL_HANDLE;
    VkPipelineLayout bloomLayout = VK_NULL_HANDLE, exposureLayout = VK_NULL_HANDLE;
    VkPipeline bloomDownPipeline = VK_NULL_HANDLE, bloomUpPipeline = VK_NULL_HANDLE;
    VkPipeline histogramPipeline = VK_NULL_HANDLE, averagePipeline = VK_NULL_HANDLE;
    /** Colour lookup tables by Lut3D::key, and the identity bound while there is none. */
    std::unordered_map<uint64_t, Image> luts;
    Image identityLut;
    /** Bumped by every mesh, texture or environment upload: a frame drawn after one may differ from the history. */
    uint64_t uploadEpoch = 0;

    // Image-based lighting. The set always holds valid images (a black placeholder until the first environment).
    VkDescriptorSetLayout envSetLayout = VK_NULL_HANDLE;
    /** Depth from each shadow-casting light, a tile each, sampled through the environment set. */
    Image shadowAtlas;
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    /** The scene's top-level acceleration structure, rebuilt in place each frame that traces; bigger when it outgrows it. */
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    compute::Buffer tlasBuffer, tlasScratch;
    uint32_t tlasCapacity = 0;
    uint64_t tlasGeneration = 0;
    VkDescriptorSetLayout rtaoSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout rtaoLayout = VK_NULL_HANDLE;
    VkPipeline rtaoPipeline = VK_NULL_HANDLE;
    VkPipelineLayout bevelLayout = VK_NULL_HANDLE;
    VkPipeline bevelPipeline = VK_NULL_HANDLE;
    VkPipelineLayout rtShadowLayout = VK_NULL_HANDLE;
    VkPipeline rtShadowPipeline = VK_NULL_HANDLE;
    /** The lights whose shadows this frame traces instead of shadow maps (one past each index), a mask channel each. */
    std::vector<uint32_t> frameRtLights;
    bool frameRtShadow = false;
    /** The frame's placements' meshes for the bevel pass, and whether it ran (the mesh shader reads its result then). */
    VkDeviceAddress frameTraceTable = 0;
    bool frameBevel = false;
    Image blackBevel; // bound where no bevel pass ran
    /** The frame's lights on the GPU, and the shadow tiles to draw: tile index and the light's view-projection. */
    VkDeviceAddress frameLightAddress = 0;
    uint32_t frameLightCount = 0;
    std::vector<std::pair<uint32_t, glm::mat4>> frameShadows;
    VkDescriptorSet envSet = VK_NULL_HANDLE;
    VkSampler envSampler = VK_NULL_HANDLE;
    Image envCube, brdfLut;
    AllocatedBuffer envUniforms;
    float envMaxLod = 0.0f;
    bool envLoaded = false;
    // Worker that builds environments; guarded by mutex.
    std::thread envWorker;
    bool envBusy = false;
    std::optional<std::string> envRequested;
    std::unique_ptr<EnvironmentData> envReady;
    // The last built environment's CPU probe, read by the main thread for overlay contrast.
    mutable std::mutex probeMutex;
    std::string probePath;
    std::shared_ptr<const EnvProbe> probe;

    // Material textures, uploaded through the frame's own command buffer with mips blitted on the GPU. Render thread only.
    struct GpuTexture
    {
        Image image; // owned; empty for a GPU image
        VkImageView view = VK_NULL_HANDLE;
        TexturePtr keep; // a GPU image's texture, kept alive while descriptors point at it
        uint32_t slot = kNoSlot; // index in the bindless array
        uint64_t lastUsed = 0;
    };
    struct MaterialSet
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        uint64_t lastUsed = 0;
    };
    std::unordered_map<uint64_t, GpuTexture> textures;
    std::map<std::array<uint64_t, kMaterialBindings>, MaterialSet> materialSets; // base color, normal, ORM, emissive, stroke keys
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool materialPool = VK_NULL_HANDLE;
    VkSampler materialSampler = VK_NULL_HANDLE;
    std::unordered_map<VkFormat, bool> blittable;
    bool warnedPool = false;
    // Bindless materials: every texture holds a slot of one runtime array (set 1 of the mesh layout), and each frame
    // writes a table the shader reads through push constants, mapping a material's maps (and UDIM tiles) to slots.
    bool bindless = false;
    VkDescriptorSetLayout bindlessLayout = VK_NULL_HANDLE;
    VkDescriptorPool bindlessPool = VK_NULL_HANDLE;
    VkDescriptorSet bindlessSet = VK_NULL_HANDLE;
    uint32_t bindlessCapacity = 0, nextSlot = 0;
    std::vector<uint32_t> freeSlots;
    bool warnedSlots = false, warnedTable = false;
    // The frame being recorded: its table, the handle of each (texture, placeholder) pair in it, and its address.
    std::vector<uint32_t> frameTable;
    std::map<std::pair<uint64_t, uint64_t>, uint32_t> frameHandles;
    /** Style block offsets in the frame's table per (draw, material slot), for slots with a stylised shading model. */
    std::map<std::pair<const MeshDraw*, uint32_t>, uint32_t> frameStyles;
    VkDeviceAddress frameTableAddress = 0;
    /** The frame's mesh draws grouped for instancing, and the address of their instance buffer (0 if it failed). */
    DrawBatches frameBatches;
    /** The frame's shaded sections, ready to record; the viewport's kept list. */
    std::vector<SectionRecord> offscreenRecords;
    std::vector<SectionRecord>* frameRecords = &offscreenRecords;
    uint64_t meshEvictions = 0, meshUploads = 0;
    /** The cull mode emitSection last set and on which command buffer; reset whenever a pipeline is bound, which may fix it. */
    VkCommandBuffer cullCmd = VK_NULL_HANDLE;
    VkCullModeFlags cullMode = 0;
    VkDeviceAddress frameInstanceAddress = 0;
    /** Mesh draw calls and placements recorded this frame, for FrameStats. */
    uint64_t frameMeshDraws = 0, frameMeshInstances = 0;
    /** The frame draws a virtual texture map: it writes feedback. */
    bool frameUsesVt = false;
    VkPipeline meshBlendPipeline = VK_NULL_HANDLE;
    /** meshPS with the wire drawn over it (meshWirePS); null without fragment shader barycentrics, and wires draw as lines. */
    VkPipeline meshWirePipeline = VK_NULL_HANDLE, meshWireBlendPipeline = VK_NULL_HANDLE;
    /** meshPipeline and meshWirePipeline for lean sections (SectionRecord::lean). */
    VkPipeline meshLeanPipeline = VK_NULL_HANDLE, meshWireLeanPipeline = VK_NULL_HANDLE;
    // Image planes sample through set 0 of their layout, a material set holding the image as its base color.
    VkPipelineLayout imagePlaneLayout = VK_NULL_HANDLE;
    VkPipeline imageOverPipeline = VK_NULL_HANDLE, imageDepthPipeline = VK_NULL_HANDLE, imageBlendPipeline = VK_NULL_HANDLE;
    VkPipeline imageBehindPipeline = VK_NULL_HANDLE;
    /** An external image's depth alone, in the scene pass, so wires hide behind its surfaces. */
    VkPipeline externalDepthPipeline = VK_NULL_HANDLE;
    /** The heightfield draw path: a grid drawn from heights alone, in the scene pass. */
    VkPipelineLayout heightfieldLayout = VK_NULL_HANDLE;
    VkPipeline heightfieldPipeline = VK_NULL_HANDLE;
    std::unordered_map<const void*, GpuHeights> heightCache; // render thread only
    /** The composed scene over what the frame pass already holds (an external image), by the scene's coverage. */
    VkPipeline composeOverPipeline = VK_NULL_HANDLE;
    /** The grid in display space, over an external renderer's image in the frame pass. */
    VkPipeline gridFramePipeline = VK_NULL_HANDLE;
    VkPipelineLayout externalLayout = VK_NULL_HANDLE;
    VkPipeline externalPipeline = VK_NULL_HANDLE;

    // Ambient occlusion. Set 2 of the mesh layout holds the AO texture; whiteAoSet stands in while AO is off.
    VkDescriptorSetLayout aoSampleLayout = VK_NULL_HANDLE, aoComputeLayout = VK_NULL_HANDLE;
    VkDescriptorPool aoPool = VK_NULL_HANDLE;
    VkSampler pointSampler = VK_NULL_HANDLE;
    Image whiteAo;
    VkDescriptorSet whiteAoSet = VK_NULL_HANDLE;
    VkPipelineLayout aoLayout = VK_NULL_HANDLE;
    VkPipeline gtaoPipeline = VK_NULL_HANDLE, denoisePipeline = VK_NULL_HANDLE, prepassPipeline = VK_NULL_HANDLE;
    VkPipelineLayout aoApplyLayout = VK_NULL_HANDLE;
    VkPipeline aoApplyPipeline = VK_NULL_HANDLE; // fast AO: multiplies the frame by aoFinal

    // ---------------------------------------------------------------- resources

    Image createImage(VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, VkSampleCountFlagBits samples, VkImageAspectFlags aspect)
    {
        Image img;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {extent.width, extent.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = samples;
        ici.usage = usage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        vmaCreateImage(vk.allocator, &ici, &aci, &img.image, &img.alloc, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCreateImageView(vk.device, &vci, nullptr, &img.view);
        return img;
    }

    void destroyImage(Image& img)
    {
        if (img.view)
            vkDestroyImageView(vk.device, img.view, nullptr);
        if (img.image)
            vmaDestroyImage(vk.allocator, img.image, img.alloc);
        img = {};
    }

    void destroySwapchain(Viewport& v)
    {
        if (v.offscreen)
        {
            v.views.clear();
            destroyImage(v.offscreenTarget);
            if (v.readback.buffer)
                vmaDestroyBuffer(vk.allocator, v.readback.buffer, v.readback.alloc);
            if (v.hdrReadback.buffer)
                vmaDestroyBuffer(vk.allocator, v.hdrReadback.buffer, v.hdrReadback.alloc);
            v.readback = {};
            v.hdrReadback = {};
            v.readbackMapped = nullptr;
            v.hdrMapped = nullptr;
        }
        for (VkImageView view : v.views)
            vkDestroyImageView(vk.device, view, nullptr);
        for (VkSemaphore s : v.rendered)
            vkDestroySemaphore(vk.device, s, nullptr);
        v.views.clear();
        v.rendered.clear();
        v.images.clear();
        destroyImage(v.color);
        destroyImage(v.depth);
        for (Image* img : {&v.aoDepth, &v.aoNormal, &v.aoRaw, &v.aoFinal, &v.bevel, &v.rtShadow, &v.postColor, &v.sceneMs, &v.scene, &v.sceneDepth, &v.history[0], &v.history[1],
                           &v.bloomDown, &v.bloomUp, &v.histogram, &v.exposureState})
            destroyImage(*img);
        for (std::vector<VkImageView>* views : {&v.bloomDownViews, &v.bloomUpViews})
        {
            for (VkImageView view : *views)
                vkDestroyImageView(vk.device, view, nullptr);
            views->clear();
        }
        v.historyValid = false;
        v.postImagesReady = false;
        v.exposureValid = false;
    }

    bool createOffscreenTargets(Viewport& v)
    {
        LILA_ZONE();

        vkDeviceWaitIdle(vk.device);
        destroySwapchain(v);
        VkExtent2D extent{uint32_t(v.width), uint32_t(v.height)};
        v.format = pipelineFormat != VK_FORMAT_UNDEFINED ? pipelineFormat : VK_FORMAT_B8G8R8A8_SRGB;
        v.extent = extent;
        v.offscreenTarget = createImage(v.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, extent,
                                        VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        if (!v.offscreenTarget.image || !v.offscreenTarget.view)
            return false;
        v.images = {v.offscreenTarget.image};
        v.views = {v.offscreenTarget.view};

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = VkDeviceSize(extent.width) * extent.height * 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(vk.allocator, &bci, &aci, &v.readback.buffer, &v.readback.alloc, &info) != VK_SUCCESS)
            return false;
        v.readbackMapped = info.pMappedData;
        if (v.hdrWanted)
        {
            bci.size = VkDeviceSize(extent.width) * extent.height * 8;
            if (vmaCreateBuffer(vk.allocator, &bci, &aci, &v.hdrReadback.buffer, &v.hdrReadback.alloc, &info) != VK_SUCCESS)
                return false;
            v.hdrMapped = info.pMappedData;
        }

        if (vk.msaa != VK_SAMPLE_COUNT_1_BIT)
            v.color = createImage(v.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent, vk.msaa,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
        v.msaaGeneration = msaaGeneration;
        v.depth = createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent,
                              vk.msaa, VK_IMAGE_ASPECT_DEPTH_BIT);
        createAoImages(v, extent);
        createSceneTargets(v, extent);
        if (pipelineFormat != v.format)
            createPipelines(v.format);
        return true;
    }

    bool createSwapchain(Viewport& v)
    {
        LILA_ZONE();

        if (v.offscreen)
            return createOffscreenTargets(v);
        vkDeviceWaitIdle(vk.device);
        VkSurfaceCapabilitiesKHR caps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical, v.surface, &caps) != VK_SUCCESS)
            return false;
        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX)
            extent = {std::clamp(uint32_t(v.width), caps.minImageExtent.width, caps.maxImageExtent.width),
                      std::clamp(uint32_t(v.height), caps.minImageExtent.height, caps.maxImageExtent.height)};
        if (extent.width == 0 || extent.height == 0)
            return false;

        uint32_t fn = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, v.surface, &fn, nullptr);
        if (!fn)
        {
            log::error("render: the window surface has no usable formats");
            return false;
        }
        std::vector<VkSurfaceFormatKHR> formats(fn);
        vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, v.surface, &fn, formats.data());
        VkSurfaceFormatKHR chosen = formats[0];
        for (auto& f : formats)
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                chosen = f;

        VkSwapchainKHR old = v.swapchain;
        destroySwapchain(v);

        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = v.surface;
        sci.minImageCount = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount)
            sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
        sci.imageFormat = chosen.format;
        sci.imageColorSpace = chosen.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = surfaceCompositeAlpha(caps.supportedCompositeAlpha);
        sci.presentMode = presentMode(v.surface, v.vsync);
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = old;
        if (VkResult r = vkCreateSwapchainKHR(vk.device, &sci, nullptr, &v.swapchain); r != VK_SUCCESS)
        {
            log::error("render: swapchain creation failed ({})", vkResultName(r));
            v.swapchain = VK_NULL_HANDLE;
            return false;
        }
        if (old)
            vkDestroySwapchainKHR(vk.device, old, nullptr);

        v.format = chosen.format;
        v.extent = extent;
        uint32_t n = 0;
        vkGetSwapchainImagesKHR(vk.device, v.swapchain, &n, nullptr);
        v.images.resize(n);
        vkGetSwapchainImagesKHR(vk.device, v.swapchain, &n, v.images.data());
        for (VkImage img : v.images)
        {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = img;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = v.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView view;
            vkCreateImageView(vk.device, &vci, nullptr, &view);
            v.views.push_back(view);

            VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkSemaphore s;
            vkCreateSemaphore(vk.device, &sem, nullptr, &s);
            v.rendered.push_back(s);
        }

        if (vk.msaa != VK_SAMPLE_COUNT_1_BIT)
            v.color = createImage(v.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, extent, vk.msaa,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
        v.msaaGeneration = msaaGeneration;
        v.depth = createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent,
                              vk.msaa, VK_IMAGE_ASPECT_DEPTH_BIT);
        createAoImages(v, extent);
        createSceneTargets(v, extent);

        if (pipelineFormat != v.format)
            createPipelines(v.format);
        return true;
    }

    void initFrames(Viewport& v)
    {
        for (Frame& f : v.frames)
        {
            f.feedback.destroy(vk);
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pci.queueFamilyIndex = vk.queueFamily;
            vkCreateCommandPool(vk.device, &pci, nullptr, &f.pool);
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = f.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(vk.device, &ai, &f.cmd);
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(vk.device, &fci, nullptr, &f.fence);
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            vkCreateSemaphore(vk.device, &sci, nullptr, &f.acquired);
            if (vk.properties.limits.timestampComputeAndGraphics)
            {
                VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qci.queryCount = 4;
                vkCreateQueryPool(vk.device, &qci, nullptr, &f.timestamps);
            }
        }
        v.gtaoSet = allocateSet(aoPool, aoComputeLayout);
        v.denoiseSet = allocateSet(aoPool, aoComputeLayout);
        v.aoSampleSet = allocateSet(aoPool, aoSampleLayout);
        v.postSet = allocateSet(aoPool, aoSampleLayout);
        for (auto& sets : v.taaSets)
            for (VkDescriptorSet& set : sets)
                set = allocateSet(aoPool, aoComputeLayout);
        for (VkDescriptorSet& set : v.composeSets)
            set = allocateSet(aoPool, composeSetLayout);
        for (VkDescriptorSet& set : v.bloomSets)
            set = allocateSet(aoPool, postComputeLayout);
        for (VkDescriptorSet& set : v.exposureSets)
            set = allocateSet(aoPool, postComputeLayout);
        if (rtaoSetLayout)
        {
            v.rtaoSet = allocateSet(aoPool, rtaoSetLayout);
            v.bevelSet = allocateSet(aoPool, rtaoSetLayout);
            v.shadowSet = allocateSet(aoPool, rtaoSetLayout);
        }
        v.rtaoGeneration = v.bevelGeneration = v.shadowGeneration = 0;
    }

    void destroyViewport(Viewport& v)
    {
        vkDeviceWaitIdle(vk.device);
        for (Frame& f : v.frames)
        {
            freeStaging(f);
            compute::destroy(vk, f.table);
            compute::destroy(vk, f.instances);
            compute::destroy(vk, f.lights);
            compute::destroy(vk, f.pointCells);
            compute::destroy(vk, f.pointOut);
            compute::destroy(vk, f.pointArgs);
            compute::destroy(vk, f.tlasInstances);
            compute::destroy(vk, f.traceTable);
            if (f.lines)
            {
                vmaUnmapMemory(vk.allocator, f.linesAlloc);
                vmaDestroyBuffer(vk.allocator, f.lines, f.linesAlloc);
            }
            vkDestroySemaphore(vk.device, f.acquired, nullptr);
            if (f.timestamps)
                vkDestroyQueryPool(vk.device, f.timestamps, nullptr);
            vkDestroyFence(vk.device, f.fence, nullptr);
            vkDestroyCommandPool(vk.device, f.pool, nullptr);
            for (const Frame::Chunk& c : f.chunks)
                vkDestroyCommandPool(vk.device, c.pool, nullptr);
            f.chunks.clear();
        }
        destroySwapchain(v);
        for (VkDescriptorSet set : {v.gtaoSet, v.denoiseSet, v.aoSampleSet, v.postSet, v.taaSets[0][0], v.taaSets[0][1], v.taaSets[1][0],
                                    v.taaSets[1][1], v.composeSets[0], v.composeSets[1], v.composeSets[2]})
            if (set)
                vkFreeDescriptorSets(vk.device, aoPool, 1, &set);
        for (VkDescriptorSet set : v.bloomSets)
            if (set)
                vkFreeDescriptorSets(vk.device, aoPool, 1, &set);
        for (VkDescriptorSet set : v.exposureSets)
            if (set)
                vkFreeDescriptorSets(vk.device, aoPool, 1, &set);
        for (VkDescriptorSet set : {v.rtaoSet, v.bevelSet, v.shadowSet})
            if (set)
                vkFreeDescriptorSets(vk.device, aoPool, 1, &set);
        if (v.swapchain)
            vkDestroySwapchainKHR(vk.device, v.swapchain, nullptr);
        if (v.surface)
            vkDestroySurfaceKHR(vk.instance, v.surface, nullptr);
    }

    void ensureLines(Frame& f, size_t bytes)
    {
        if (bytes <= f.linesCapacity)
            return;
        if (f.lines)
        {
            vmaUnmapMemory(vk.allocator, f.linesAlloc);
            vmaDestroyBuffer(vk.allocator, f.lines, f.linesAlloc);
        }
        size_t cap = std::max<size_t>(bytes * 2, 64 * 1024);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = cap;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        vmaCreateBuffer(vk.allocator, &bci, &aci, &f.lines, &f.linesAlloc, nullptr);
        vmaMapMemory(vk.allocator, f.linesAlloc, &f.linesMapped);
        f.linesCapacity = cap;
    }

    void initDescriptors()
    {
        VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 1;
        lci.pBindings = &b;
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &atlasSetLayout);

        VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4}};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 16;
        pci.poolSizeCount = 2;
        pci.pPoolSizes = sizes;
        vkCreateDescriptorPool(vk.device, &pci, nullptr, &descriptorPool);

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &atlasSetLayout;
        vkAllocateDescriptorSets(vk.device, &ai, &atlasSet);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(vk.device, &sci, nullptr, &linearSampler);

        // A 1x1 white atlas so the set is always valid before real text arrives.
        auto blank = std::make_shared<OverlayAtlas>();
        blank->width = blank->height = 1;
        blank->pixels = {255};
        uploadAtlas(blank);
        initEnvironment();
        initMaterials();
        initAo();
    }

    void destroyDescriptors()
    {
        destroyAo();
        destroyMaterials();
        destroyEnvironment();
        destroyImage(atlasImage);
        if (linearSampler)
            vkDestroySampler(vk.device, linearSampler, nullptr);
        if (descriptorPool)
            vkDestroyDescriptorPool(vk.device, descriptorPool, nullptr);
        if (atlasSetLayout)
            vkDestroyDescriptorSetLayout(vk.device, atlasSetLayout, nullptr);
        linearSampler = VK_NULL_HANDLE;
        descriptorPool = VK_NULL_HANDLE;
        atlasSetLayout = VK_NULL_HANDLE;
    }

    /** Sampled image with a full mip chain and optional cube layers, ready to be filled by uploadImage. */
    Image createTexture(VkFormat format, VkExtent2D extent, uint32_t mips, bool cube, VkImageUsageFlags extraUsage = 0,
                        VkComponentMapping swizzle = {})
    {
        Image img;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {extent.width, extent.height, 1};
        ici.mipLevels = mips;
        ici.arrayLayers = cube ? 6 : 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extraUsage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vmaCreateImage(vk.allocator, &ici, &aci, &img.image, &img.alloc, nullptr);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.components = swizzle;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, ici.arrayLayers};
        vkCreateImageView(vk.device, &vci, nullptr, &img.view);
        return img;
    }

    /**
     * Copies tightly packed texels into every mip and layer of an image and leaves it shader-readable. Waits for
     * the queue, so only for rare uploads, and only when no in-flight frame can still be reading the image.
     */
    void uploadImage(VkImage image, VkExtent2D extent, uint32_t mips, uint32_t layers, size_t texelBytes,
                     const std::vector<const void*>& mipData)
    {
        LILA_ZONE();

        std::vector<VkBufferImageCopy> regions;
        size_t total = 0;
        for (uint32_t mip = 0; mip < mips; ++mip)
        {
            uint32_t w = std::max(extent.width >> mip, 1u), h = std::max(extent.height >> mip, 1u);
            VkBufferImageCopy region{};
            region.bufferOffset = total;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, layers};
            region.imageExtent = {w, h, 1};
            regions.push_back(region);
            total += size_t(w) * h * layers * texelBytes;
        }

        VkBuffer staging;
        VmaAllocation stagingAlloc;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = total;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        vmaCreateBuffer(vk.allocator, &bci, &aci, &staging, &stagingAlloc, nullptr);
        void* mapped;
        vmaMapMemory(vk.allocator, stagingAlloc, &mapped);
        for (uint32_t mip = 0; mip < mips; ++mip)
        {
            size_t end = mip + 1 < mips ? size_t(regions[mip + 1].bufferOffset) : total;
            std::memcpy(static_cast<char*>(mapped) + regions[mip].bufferOffset, mipData[mip], end - size_t(regions[mip].bufferOffset));
        }
        vmaUnmapMemory(vk.allocator, stagingAlloc);
        vmaFlushAllocation(vk.allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

        VkCommandPool pool;
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = vk.queueFamily;
        vkCreateCommandPool(vk.device, &pci, nullptr, &pool);
        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(vk.device, &cai, &cmd);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        cullCmd = VK_NULL_HANDLE;
        barrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, mips, layers);
        vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());
        barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, mips, layers);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk.queue);
        vkDestroyCommandPool(vk.device, pool, nullptr);
        vmaDestroyBuffer(vk.allocator, staging, stagingAlloc);
    }

    /** Rare (font changes), so it just waits for the queue. */
    void uploadAtlas(std::shared_ptr<const OverlayAtlas> atlas)
    {
        LILA_ZONE();

        vkDeviceWaitIdle(vk.device);
        destroyImage(atlasImage);
        VkExtent2D extent{uint32_t(atlas->width), uint32_t(atlas->height)};
        atlasImage = createTexture(VK_FORMAT_R8_UNORM, extent, 1, false);
        uploadImage(atlasImage.image, extent, 1, 1, 1, {atlas->pixels.data()});

        VkDescriptorImageInfo ii{linearSampler, atlasImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = atlasSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
        uploadedAtlas = atlas.get();
        atlasKeepAlive = std::move(atlas);
    }

    // ---------------------------------------------------------------- environment

    void initEnvironment()
    {
        LILA_ZONE();

        VkDescriptorSetLayoutBinding b[4] = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 4;
        lci.pBindings = b;
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &envSetLayout);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &envSetLayout;
        vkAllocateDescriptorSets(vk.device, &ai, &envSet);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        vkCreateSampler(vk.device, &sci, nullptr, &envSampler);

        VkSamplerCreateInfo ssci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ssci.magFilter = ssci.minFilter = VK_FILTER_NEAREST;
        ssci.addressModeU = ssci.addressModeV = ssci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(vk.device, &ssci, nullptr, &shadowSampler);
        shadowAtlas = createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, {kShadowAtlas, kShadowAtlas},
                                  VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkDescriptorImageInfo ii{shadowSampler, shadowAtlas.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = envSet;
            w.dstBinding = 3;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &ii;
            vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
        }

        std::vector<uint16_t> lut = buildBrdfLut(kBrdfLutSize);
        VkExtent2D lutExtent{uint32_t(kBrdfLutSize), uint32_t(kBrdfLutSize)};
        brdfLut = createTexture(VK_FORMAT_R16G16_SFLOAT, lutExtent, 1, false);
        uploadImage(brdfLut.image, lutExtent, 1, 1, 4, {lut.data()});

        EnvironmentData placeholder;
        placeholder.size = 1;
        placeholder.mips = {std::vector<uint16_t>(6 * 4, 0)};
        uploadEnvironment(placeholder);
        envLoaded = false;
    }

    /** Swaps in a new environment. Waits for the GPU so no in-flight frame still samples the old one. */
    void uploadEnvironment(const EnvironmentData& env)
    {
        LILA_ZONE();

        ++uploadEpoch;
        vkDeviceWaitIdle(vk.device);
        destroyImage(envCube);
        destroyBuffer(envUniforms);
        uint32_t mips = uint32_t(env.mips.size());
        VkExtent2D extent{uint32_t(env.size), uint32_t(env.size)};
        envCube = createTexture(VK_FORMAT_R16G16B16A16_SFLOAT, extent, mips, true);
        std::vector<const void*> data;
        for (const auto& mip : env.mips)
            data.push_back(mip.data());
        uploadImage(envCube.image, extent, mips, 6, 8, data);
        envUniforms = createBuffer(env.sh.data(), sizeof(env.sh), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        envMaxLod = float(mips - 1);
        envLoaded = true;

        VkDescriptorImageInfo cube{envSampler, envCube.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo lut{linearSampler, brdfLut.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo ubo{envUniforms.buffer, 0, sizeof(env.sh)};
        VkWriteDescriptorSet w[3]{};
        for (uint32_t i = 0; i < 3; ++i)
        {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = envSet;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        }
        w[0].pImageInfo = &cube;
        w[1].pImageInfo = &lut;
        w[2].pBufferInfo = &ubo;
        vkUpdateDescriptorSets(vk.device, 3, w, 0, nullptr);
    }

    void destroyEnvironment()
    {
        destroyImage(envCube);
        destroyImage(brdfLut);
        destroyImage(shadowAtlas);
        if (shadowSampler)
            vkDestroySampler(vk.device, shadowSampler, nullptr);
        shadowSampler = VK_NULL_HANDLE;
        destroyBuffer(envUniforms);
        if (envSampler)
            vkDestroySampler(vk.device, envSampler, nullptr);
        if (envSetLayout)
            vkDestroyDescriptorSetLayout(vk.device, envSetLayout, nullptr);
        envSampler = VK_NULL_HANDLE;
        envSetLayout = VK_NULL_HANDLE;
    }

    /** Starts building the environment for path on a worker. Call with mutex held and no build running. */
    void startEnvironmentBuild(const std::string& path)
    {
        if (envWorker.joinable())
            envWorker.join();
        envBusy = true;
        envRequested = path;
        envWorker = std::thread([this, path] {
            LILA_THREAD_NAME("Environment");
            auto t0 = std::chrono::steady_clock::now();
            auto env = std::make_unique<EnvironmentData>();
            HdrImage hdr;
            std::string error;
            if (path.empty())
                *env = buildStudioEnvironment();
            else if (loadHdr(std::filesystem::path(reinterpret_cast<const char8_t*>(path.c_str())), hdr, error))
                *env = buildEnvironment(hdr);
            else
            {
                log::error("render: cannot load environment {}: {}", path, error);
                *env = buildStudioEnvironment();
            }
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            log::info("render: environment '{}' built in {} ms", path.empty() ? "studio" : path, ms);
            {
                std::shared_ptr<const EnvProbe> built = makeEnvProbe(*env);
                std::scoped_lock probeLock(probeMutex);
                probe = std::move(built);
                probePath = path;
            }

            std::scoped_lock lock(mutex);
            envReady = std::move(env);
            envBusy = false;
            for (auto& [id, v] : viewports)
                v->pending = true;
            cv.notify_all();
        });
    }

    /** With mutex held: starts a build when the wanted environment changed and hands over a finished one. */
    std::unique_ptr<EnvironmentData> pollEnvironment(const std::string& wanted)
    {
        if (!envBusy && !envReady && wanted != envRequested)
            startEnvironmentBuild(wanted);
        return std::move(envReady);
    }

    /** Host-visible buffer; on ReBAR GPUs VMA puts it in device memory, so no staging copy is needed. */
    AllocatedBuffer createBuffer(const void* data, size_t size, VkBufferUsageFlags usage)
    {
        AllocatedBuffer b;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = std::max<size_t>(size, 16);
        bci.usage = usage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        vmaCreateBuffer(vk.allocator, &bci, &aci, &b.buffer, &b.alloc, &info);
        constexpr size_t kBlock = size_t(4) << 20;
        if (size > kBlock)
            tbb::parallel_for(size_t(0), (size + kBlock - 1) / kBlock, [&](size_t k) {
                std::memcpy(static_cast<char*>(info.pMappedData) + k * kBlock, static_cast<const char*>(data) + k * kBlock,
                            std::min(kBlock, size - k * kBlock));
            });
        else if (size)
            std::memcpy(info.pMappedData, data, size);
        vmaFlushAllocation(vk.allocator, b.alloc, 0, VK_WHOLE_SIZE);
        return b;
    }

    /**
     * A device-local copy of data, filled by a copy recorded into cmd. The staging buffer is freed once the frame's fence
     * signals. Falls back to host memory when device memory is full.
     */
    AllocatedBuffer deviceBuffer(VkCommandBuffer cmd, Frame& f, const void* data, size_t size, VkBufferUsageFlags usage)
    {
        AllocatedBuffer b;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = std::max<size_t>(size, 16);
        bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateBuffer(vk.allocator, &bci, &aci, &b.buffer, &b.alloc, nullptr) != VK_SUCCESS)
            return createBuffer(data, size, usage);
        if (size)
        {
            AllocatedBuffer staging = createBuffer(data, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            f.staging.push_back(staging);
            const VkBufferCopy region{0, 0, size};
            vkCmdCopyBuffer(cmd, staging.buffer, b.buffer, 1, &region);
        }
        return b;
    }

    void destroyBuffer(AllocatedBuffer& b)
    {
        if (b.buffer)
            vmaDestroyBuffer(vk.allocator, b.buffer, b.alloc);
        b = {};
    }

    /**
     * The mesh's GPU copy, made on first use. With cmd and f the copy goes to device memory through a copy recorded into
     * cmd; without them it reads from host memory, which is slow for dense meshes once the BAR heap is full.
     */
    GpuMesh* gpuMesh(const std::shared_ptr<const RenderMesh>& mesh, VkCommandBuffer cmd = VK_NULL_HANDLE, Frame* f = nullptr)
    {
        auto upload = [&](const void* data, size_t size, VkBufferUsageFlags usage) {
            return f ? deviceBuffer(cmd, *f, data, size, usage) : createBuffer(data, size, usage);
        };
        // Static meshes can be traced: their buffers feed acceleration structure builds.
        const VkBufferUsageFlags traceable = vk.rayQuery ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                         : 0;
        // Looked up first: inserting this entry may rehash the cache under the base's.
        AllocatedBuffer shared;
        if (mesh->shareVertices)
            shared = gpuMesh(mesh->shareVertices, cmd, f)->vertices;
        auto it = meshCache.find(mesh->key);
        if (it == meshCache.end())
        {
            LILA_ZONE_N("gpuMesh upload");
            GpuMesh g;
            if (mesh->shareVertices)
            {
                g.vertices = shared;
                g.sharedVertices = true;
            }
            else if (mesh->dynamic)
            {
                DynamicVertices& dv = *mesh->dynamic;
                std::scoped_lock lock(dv.mutex);
                g.vertices = upload(dv.vertices.data(), dv.vertices.size() * sizeof(MeshVertex),
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                dv.dirty.clear();
                dv.allDirty = false;
            }
            else
                g.vertices = upload(mesh->vertices.data(), mesh->vertices.size() * sizeof(MeshVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | traceable);
            g.indices = upload(mesh->indices.data(), mesh->indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | traceable);
            g.edges = upload(mesh->edgePoints.data(), mesh->edgePoints.size() * sizeof(glm::vec3),
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | (mesh->dynamic ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0));
            g.indexCount = uint32_t(mesh->indices.size());
            g.edgePointCount = uint32_t(mesh->edgePoints.size());
            if (!mesh->edgeMask.empty() && meshWirePipeline)
                g.edgeMask = maskBuffer(cmd, f, mesh->edgeMask);
            if (!mesh->edgeIndices.empty())
            {
                g.edgeIndices = upload(mesh->edgeIndices.data(), mesh->edgeIndices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
                g.edgeIndexCount = uint32_t(mesh->edgeIndices.size());
            }
            g.keep = mesh;
            it = meshCache.emplace(mesh->key, std::move(g)).first;
            ++uploadEpoch;
            ++meshUploads;
        }
        else if (mesh->shareVertices)
            it->second.vertices = shared;
        it->second.lastUsed = frameCounter;
        return &it->second;
    }

    /** A storage buffer with a device address holding words, copied through a staging buffer when cmd and f are given. */
    compute::Buffer maskBuffer(VkCommandBuffer cmd, Frame* f, const std::vector<uint32_t>& words)
    {
        const size_t bytes = words.size() * sizeof(uint32_t);
        try
        {
            if (!f)
                return compute::hostBuffer(vk, words.data(), bytes);
            compute::Buffer b = compute::makeBuffer(vk, bytes, 0, compute::Host::None);
            AllocatedBuffer staging = createBuffer(words.data(), bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            f->staging.push_back(staging);
            const VkBufferCopy region{0, 0, bytes};
            vkCmdCopyBuffer(cmd, staging.buffer, b.buffer, 1, &region);
            return b;
        }
        catch (const std::exception& e)
        {
            log::warn("render: {}; the wire draws as lines", e.what());
            return {};
        }
    }

    /** Logs the first failed submit or present; after a lost device nothing draws again until Lila restarts. */
    void checkQueue(VkResult r, const char* what)
    {
        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR || std::exchange(queueFailed, true))
            return;
        log::error("render: {} failed ({}); the viewport stops drawing until Lila restarts", what, vkResultName(r));
    }
    bool queueFailed = false;

    /**
     * Drops GPU copies no frame has drawn for a while, and at once those nothing else holds any more: views draw on
     * demand, so an idle viewport would otherwise keep a replaced dense mesh in memory indefinitely.
     */
    void evictMeshes()
    {
        const bool sweep = frameCounter % 120 == 0;
        std::vector<uint64_t> stale;
        for (auto& [key, g] : meshCache)
            if ((sweep && g.lastUsed + 240 < frameCounter) || (g.keep.use_count() == 1 && g.lastUsed != frameCounter))
                stale.push_back(key);
        if (stale.empty())
            return;
        vkDeviceWaitIdle(vk.device);
        for (uint64_t key : stale)
        {
            GpuMesh& g = meshCache[key];
            destroyBlas(g);
            if (!g.sharedVertices)
                destroyBuffer(g.vertices);
            destroyBuffer(g.indices);
            destroyBuffer(g.edges);
            destroyBuffer(g.edgeIndices);
            compute::destroy(vk, g.edgeMask);
            meshCache.erase(key);
            ++meshEvictions;
        }
    }

    /** Drops component dots no frame has drawn for a while: each selection change makes a new cloud. */
    void evictPoints()
    {
        if (frameCounter % 30 != 0)
            return;
        std::vector<uint64_t> stale;
        for (auto& [key, g] : pointCache)
            if (g.lastUsed + 8 < frameCounter)
                stale.push_back(key);
        if (stale.empty())
            return;
        vkDeviceWaitIdle(vk.device);
        for (uint64_t key : stale)
        {
            destroyBuffer(pointCache[key].buffer);
            pointCache.erase(key);
        }
    }

    void destroyMeshes()
    {
        for (auto& [key, g] : pointCache)
            destroyBuffer(g.buffer);
        pointCache.clear();
        for (auto& [key, g] : heightCache)
            destroyBuffer(g.buffer);
        heightCache.clear();
        for (auto& [key, g] : meshCache)
        {
            destroyBlas(g);
            if (!g.sharedVertices)
                destroyBuffer(g.vertices);
            destroyBuffer(g.indices);
            destroyBuffer(g.edges);
            destroyBuffer(g.edgeIndices);
            compute::destroy(vk, g.edgeMask);
        }
        meshCache.clear();
        if (tlas)
            vkDestroyAccelerationStructureKHR(vk.device, tlas, nullptr);
        tlas = VK_NULL_HANDLE;
        compute::destroy(vk, tlasBuffer);
        compute::destroy(vk, tlasScratch);
        tlasCapacity = 0;
    }

    // ---------------------------------------------------------------- materials

    void initMaterials()
    {
        LILA_ZONE();

        VkDescriptorSetLayoutBinding b[kMaterialBindings];
        for (uint32_t i = 0; i < kMaterialBindings; ++i)
            b[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = kMaterialBindings;
        lci.pBindings = b;
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &materialSetLayout);

        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxMaterialSets * kMaterialBindings};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pci.maxSets = kMaxMaterialSets;
        pci.poolSizeCount = 1;
        pci.pPoolSizes = &size;
        vkCreateDescriptorPool(vk.device, &pci, nullptr, &materialPool);

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        sci.anisotropyEnable = vk.anisotropy;
        sci.maxAnisotropy = vk.anisotropy ? std::min(8.0f, vk.properties.limits.maxSamplerAnisotropy) : 1.0f;
        vkCreateSampler(vk.device, &sci, nullptr, &materialSampler);
        initBindless();

        auto placeholder = [&](uint64_t key, VkFormat format, std::array<uint8_t, 4> texel) {
            GpuTexture& t = textures[key];
            t.image = createTexture(format, {1, 1}, 1, false);
            t.view = t.image.view;
            uploadImage(t.image.image, {1, 1}, 1, 1, 4, {texel.data()});
            assignSlot(t);
        };
        placeholder(kWhiteSrgbKey, VK_FORMAT_R8G8B8A8_SRGB, {255, 255, 255, 255});
        placeholder(kWhiteLinearKey, VK_FORMAT_R8G8B8A8_UNORM, {255, 255, 255, 255});
        placeholder(kFlatNormalKey, VK_FORMAT_R8G8B8A8_UNORM, {128, 128, 255, 255});
    }

    /** Whether images of a format can have their mips blitted (and be filtered). */
    bool canBlit(VkFormat f)
    {
        auto it = blittable.find(f);
        if (it != blittable.end())
            return it->second;
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(vk.physical, f, &fp);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return blittable[f] = (fp.optimalTilingFeatures & need) == need;
    }

    /** The bindless array, when the device has descriptor indexing and the bindless mesh shader is there. */
    void initBindless()
    {
        if (!vk.descriptorIndexing || loadSpirv("mesh_bindless.spv").empty())
            return;
        LILA_ZONE();

        VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p12};
        vkGetPhysicalDeviceProperties2(vk.physical, &p2);
        // The rest of the mesh layout (environment and AO sets) counts against the same per-stage limits.
        uint32_t limit = std::min({p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxDescriptorSetUpdateAfterBindSamplers,
                                   p12.maxPerStageDescriptorUpdateAfterBindSampledImages, p12.maxPerStageDescriptorUpdateAfterBindSamplers,
                                   p12.maxPerStageUpdateAfterBindResources});
        bindlessCapacity = std::min(kMaxBindless, limit > 16 ? limit - 16 : 0);
        if (bindlessCapacity < 64)
            return;

        VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                         VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        bfi.bindingCount = 1;
        bfi.pBindingFlags = &flags;
        VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, bindlessCapacity, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.pNext = &bfi;
        lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        lci.bindingCount = 1;
        lci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &bindlessLayout) != VK_SUCCESS)
            return;

        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, bindlessCapacity};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pci.maxSets = 1;
        pci.poolSizeCount = 1;
        pci.pPoolSizes = &size;
        if (vkCreateDescriptorPool(vk.device, &pci, nullptr, &bindlessPool) != VK_SUCCESS)
        {
            destroyBindless();
            return;
        }
        VkDescriptorSetVariableDescriptorCountAllocateInfo vci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
        vci.descriptorSetCount = 1;
        vci.pDescriptorCounts = &bindlessCapacity;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.pNext = &vci;
        ai.descriptorPool = bindlessPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &bindlessLayout;
        if (vkAllocateDescriptorSets(vk.device, &ai, &bindlessSet) != VK_SUCCESS)
        {
            destroyBindless();
            return;
        }
        bindless = true;
        log::info("render: bindless materials ({} textures)", bindlessCapacity);
    }

    void destroyBindless()
    {
        if (bindlessPool)
            vkDestroyDescriptorPool(vk.device, bindlessPool, nullptr);
        if (bindlessLayout)
            vkDestroyDescriptorSetLayout(vk.device, bindlessLayout, nullptr);
        bindlessPool = VK_NULL_HANDLE;
        bindlessLayout = VK_NULL_HANDLE;
        bindlessSet = VK_NULL_HANDLE;
        bindless = false;
        freeSlots.clear();
        nextSlot = 0;
    }

    /** Gives a texture a bindless slot pointing at its view. Without one it samples as its placeholder. */
    void assignSlot(GpuTexture& t)
    {
        if (!bindless || t.slot != kNoSlot || !t.view)
            return;
        if (!freeSlots.empty())
        {
            t.slot = freeSlots.back();
            freeSlots.pop_back();
        }
        else if (nextSlot < bindlessCapacity)
            t.slot = nextSlot++;
        else
        {
            if (!warnedSlots)
                log::warn("render: more than {} material textures; the rest draw blank", bindlessCapacity);
            warnedSlots = true;
            return;
        }
        VkDescriptorImageInfo info{materialSampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = bindlessSet;
        w.dstBinding = 0;
        w.dstArrayElement = t.slot;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
    }

    void destroyMaterials()
    {
        for (auto& [key, t] : textures)
            destroyImage(t.image);
        textures.clear();
        materialSets.clear();
        destroyBindless();
        if (materialSampler)
            vkDestroySampler(vk.device, materialSampler, nullptr);
        if (materialPool)
            vkDestroyDescriptorPool(vk.device, materialPool, nullptr);
        if (materialSetLayout)
            vkDestroyDescriptorSetLayout(vk.device, materialSetLayout, nullptr);
        materialSampler = VK_NULL_HANDLE;
        materialPool = VK_NULL_HANDLE;
        materialSetLayout = VK_NULL_HANDLE;
    }

    static void mipBarrier(VkCommandBuffer cmd, VkImage image, uint32_t baseMip, uint32_t mips, VkImageLayout from, VkImageLayout to,
                           VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
    {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStage;
        b.srcAccessMask = srcAccess;
        b.dstStageMask = dstStage;
        b.dstAccessMask = dstAccess;
        b.oldLayout = from;
        b.newLayout = to;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mips, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    /**
     * Records the upload of a texture and its mip chain into cmd, ahead of the frame's rendering. The staging buffer
     * lives until the frame's fence signals. False for malformed images, which then draw as if absent.
     */
    bool uploadTexture(VkCommandBuffer cmd, Frame& f, const TextureImage& t)
    {
        LILA_ZONE();

        if (t.blocks)
            return uploadBlocks(cmd, f, t);
        const size_t bytes = size_t(t.width) * size_t(t.height) * size_t(texelBytes(t.format));
        if (t.width <= 0 || t.height <= 0 || t.rgba.size() < bytes)
            return false;
        ++uploadEpoch;
        VkFormat format = vkFormat(t.format, t.srgb);
        VkExtent2D extent{uint32_t(t.width), uint32_t(t.height)};
        uint32_t mips = 1;
        if (canBlit(format))
            while ((std::max(extent.width, extent.height) >> mips) > 0)
                ++mips;

        GpuTexture g;
        g.image = createTexture(format, extent, mips, false, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, sampleSwizzle(t.format));
        g.view = g.image.view;
        g.lastUsed = frameCounter;
        AllocatedBuffer staging = createBuffer(t.rgba.data(), bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        f.staging.push_back(staging);

        constexpr auto transfer = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mipBarrier(cmd, g.image.image, 0, mips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                   transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyBufferToImage(cmd, staging.buffer, g.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        for (uint32_t mip = 1; mip < mips; ++mip)
        {
            mipBarrier(cmd, g.image.image, mip - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, transfer,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT, transfer, VK_ACCESS_2_TRANSFER_READ_BIT);
            int32_t sw = int32_t(std::max(extent.width >> (mip - 1), 1u)), sh = int32_t(std::max(extent.height >> (mip - 1), 1u));
            int32_t dw = int32_t(std::max(extent.width >> mip, 1u)), dh = int32_t(std::max(extent.height >> mip, 1u));
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 1};
            blit.srcOffsets[1] = {sw, sh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            blit.dstOffsets[1] = {dw, dh, 1};
            vkCmdBlitImage(cmd, g.image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                           VK_FILTER_LINEAR);
        }
        constexpr auto sampled = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        if (mips > 1)
            mipBarrier(cmd, g.image.image, 0, mips - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, transfer,
                       VK_ACCESS_2_TRANSFER_READ_BIT, sampled, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        mipBarrier(cmd, g.image.image, mips - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, transfer,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT, sampled, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        assignSlot(g);
        textures[t.key] = g;
        return true;
    }

    /** A BC5 normal map (or UDIM set of them), whose blue the shader rebuilds. */
    static bool isTwoChannelNormal(const TexturePtr& t)
    {
        if (t && t->isUdimSet())
            return t->tiles.front().image && t->tiles.front().image->blocks == dds::Format::BC5;
        return t && t->blocks == dds::Format::BC5;
    }

    static VkFormat blockFormat(dds::Format f, bool srgb)
    {
        switch (f)
        {
        case dds::Format::BC1: return srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case dds::Format::BC2: return srgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK;
        case dds::Format::BC3: return srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
        case dds::Format::BC4: return VK_FORMAT_BC4_UNORM_BLOCK;
        case dds::Format::BC5: return VK_FORMAT_BC5_UNORM_BLOCK;
        case dds::Format::BC7: return srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
        default: return VK_FORMAT_UNDEFINED;
        }
    }

    bool canSampleBlocks(VkFormat f)
    {
        if (!vk.bcTextures || f == VK_FORMAT_UNDEFINED)
            return false;
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(vk.physical, f, &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                          VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        return (fp.optimalTilingFeatures & need) == need;
    }

    /**
     * A block-compressed texture: its levels copied as they are, no mips generated. Devices without BC sampling get
     * the largest level decoded to RGBA8 instead, through the plain path.
     */
    bool uploadBlocks(VkCommandBuffer cmd, Frame& f, const TextureImage& t)
    {
        const dds::Format bf = *t.blocks;
        const VkFormat format = blockFormat(bf, t.srgb);
        size_t total = 0;
        for (int l = 0, w = t.width, h = t.height; l < t.mipLevels; ++l, w = std::max(1, w / 2), h = std::max(1, h / 2))
            total += dds::levelBytes(bf, w, h);
        if (t.width <= 0 || t.height <= 0 || t.mipLevels < 1 || t.rgba.size() < total)
            return false;
        if (!canSampleBlocks(format))
        {
            TextureImage plain;
            plain.width = t.width;
            plain.height = t.height;
            plain.srgb = t.srgb && bf != dds::Format::BC4 && bf != dds::Format::BC5;
            plain.rgba = dds::decodeRgba8(bf, t.width, t.height, t.rgba.data());
            plain.key = t.key;
            return uploadTexture(cmd, f, plain);
        }
        ++uploadEpoch;
        constexpr auto R = VK_COMPONENT_SWIZZLE_R, G = VK_COMPONENT_SWIZZLE_G, zero = VK_COMPONENT_SWIZZLE_ZERO,
                       one = VK_COMPONENT_SWIZZLE_ONE;
        // A BC5 normal map leaves blue empty; the shader rebuilds it (MatNormalXY).
        const VkComponentMapping swizzle = bf == dds::Format::BC4 ? VkComponentMapping{R, R, R, one}
                                           : bf == dds::Format::BC5 ? VkComponentMapping{R, G, zero, one}
                                                                    : VkComponentMapping{};
        const uint32_t mips = uint32_t(t.mipLevels);
        GpuTexture g;
        g.image = createTexture(format, VkExtent2D{uint32_t(t.width), uint32_t(t.height)}, mips, false, 0, swizzle);
        g.view = g.image.view;
        g.lastUsed = frameCounter;
        AllocatedBuffer staging = createBuffer(t.rgba.data(), total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        f.staging.push_back(staging);

        constexpr auto transfer = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mipBarrier(cmd, g.image.image, 0, mips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                   transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        std::vector<VkBufferImageCopy> regions;
        size_t offset = 0;
        for (uint32_t l = 0; l < mips; ++l)
        {
            const uint32_t w = std::max(uint32_t(t.width) >> l, 1u), h = std::max(uint32_t(t.height) >> l, 1u);
            VkBufferImageCopy r{};
            r.bufferOffset = offset;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            r.imageExtent = {w, h, 1};
            regions.push_back(r);
            offset += dds::levelBytes(bf, int(w), int(h));
        }
        vkCmdCopyBufferToImage(cmd, staging.buffer, g.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());
        mipBarrier(cmd, g.image.image, 0, mips, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, transfer,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        assignSlot(g);
        textures[t.key] = g;
        return true;
    }

    /** Records a GPU image's queued updates and registers it as a texture on first use. False when it has no device image. */
    bool prepareGpuImage(VkCommandBuffer cmd, Frame& f, const TexturePtr& t)
    {
        std::vector<compute::Buffer> staging;
        const bool ok = t->gpu->prepare(vk, cmd, staging);
        for (const compute::Buffer& b : staging)
            f.staging.push_back({b.buffer, b.alloc});
        if (!ok)
            return false;
        auto [it, added] = textures.try_emplace(t->key);
        if (added)
        {
            it->second.view = t->gpu->view();
            it->second.keep = t;
            it->second.lastUsed = frameCounter;
            assignSlot(it->second);
        }
        return true;
    }

    /**
     * Key of an uploaded texture, or the placeholder's when the map is missing or failed to upload. A UDIM set gives its
     * first uploaded tile (what the fixed sampler path shows) and keeps all its tiles alive.
     */
    uint64_t textureKey(const TexturePtr& t, uint64_t placeholder)
    {
        if (!t)
            return placeholder;
        if (t->vt)
            return bindless ? t->key : placeholder;
        if (t->isUdimSet())
        {
            uint64_t first = placeholder;
            for (const UdimTile& tile : t->tiles)
                if (tile.image && !tile.image->isUdimSet())
                    if (uint64_t k = textureKey(tile.image, placeholder); k != placeholder && first == placeholder)
                        first = k;
            return first;
        }
        auto it = textures.find(t->key);
        if (it == textures.end())
            return placeholder;
        it->second.lastUsed = frameCounter;
        return t->key;
    }

    /** Uploads a map (every tile of a UDIM set) the first time a frame uses it, and lands a GPU image's updates. */
    void prepareTexture(VkCommandBuffer cmd, Frame& f, const TexturePtr& t, const char* what)
    {
        if (!t)
            return;
        if (t->isUdimSet())
        {
            for (const UdimTile& tile : t->tiles)
                if (tile.image && !tile.image->isUdimSet())
                    prepareTexture(cmd, f, tile.image, what);
            return;
        }
        if (t->vt)
        {
            for (const TexturePtr& chunk : vt::chunkTextures(*t))
                prepareGpuImage(cmd, f, chunk);
            frameUsesVt = true;
        }
        else if (t->gpu)
            prepareGpuImage(cmd, f, t);
        else if (!textures.contains(t->key) && !uploadTexture(cmd, f, *t))
            log::warn("render: skipped a malformed {}x{} {}", t->width, t->height, what);
    }

    /** Uploads every map this frame's materials use for the first time, then writes the frame's texture table. Runs before rendering begins. */
    /** The draws' material maps, each once, in first-use order with their placeholders, and the distinct textures among them. */
    static void collectMaps(const FrameState& s, std::vector<std::pair<TexturePtr, uint64_t>>& maps, std::vector<TexturePtr>& textures)
    {
        maps.clear();
        textures.clear();
        std::set<std::pair<const void*, uint64_t>> seenMaps;
        std::unordered_set<const void*> seenTextures;
        std::unordered_set<const void*> seenLists;
        auto addTexture = [&](const TexturePtr& t) {
            if (seenTextures.insert(t.get()).second)
                textures.push_back(t);
        };
        auto addMap = [&](const TexturePtr& t, uint64_t placeholder) {
            if (t && seenMaps.insert({t.get(), placeholder}).second)
                maps.emplace_back(t, placeholder);
        };
        for (const MeshDraw& d : s.meshes)
        {
            if (!(d.flags & DrawShaded) || d.materials.empty() || !seenLists.insert(&d.materials.list()).second)
                continue;
            for (const Material& mat : d.materials)
            {
                for (const TexturePtr* t : {&mat.baseColorMap, &mat.normalMap, &mat.ormMap, &mat.emissiveMap, &mat.style.strokeMap})
                    addTexture(*t);
                addMap(mat.baseColorMap, kWhiteSrgbKey);
                addMap(mat.normalMap, kFlatNormalKey);
                addMap(mat.ormMap, kWhiteLinearKey);
                addMap(mat.emissiveMap, kWhiteSrgbKey);
                if (mat.style.strokeMap && !mat.style.strokeMap->isUdimSet())
                    addMap(mat.style.strokeMap, kWhiteLinearKey);
            }
        }
    }

    void prepareTextures(VkCommandBuffer cmd, Frame& f, const FrameState& s, Viewport* v = nullptr)
    {
        LILA_ZONE();

        std::vector<std::pair<TexturePtr, uint64_t>> scratchMaps;
        std::vector<TexturePtr> scratchTextures;
        if (v && !v->keptMapsValid)
        {
            collectMaps(s, v->keptMaps, v->keptTextures);
            v->keptMapsValid = true;
        }
        else if (!v)
            collectMaps(s, scratchMaps, scratchTextures);
        const auto& maps = v ? v->keptMaps : scratchMaps;
        const auto& unique = v ? v->keptTextures : scratchTextures;

        frameUsesVt = false;
        if (bindless)
        {
            std::vector<compute::Buffer> staging;
            vt::recordFrame(vk, cmd, staging);
            for (const compute::Buffer& b : staging)
                f.staging.push_back({b.buffer, b.alloc});
        }

        for (const TexturePtr& t : unique)
            prepareTexture(cmd, f, t, "texture");
        for (const ImagePlaneDraw& p : s.imagePlanes)
            prepareTexture(cmd, f, p.image, "image plane");
        prepareTexture(cmd, f, s.externalColor, "viewport renderer image");
        prepareTexture(cmd, f, s.externalDepth, "viewport renderer depth");
        writeTextureTable(f, s, maps);
    }

    /** The bindless slot a texture key samples, its placeholder's when it has none. */
    uint32_t slotOf(uint64_t key, uint64_t placeholder)
    {
        auto it = textures.find(key);
        if (it != textures.end() && it->second.slot != kNoSlot)
            return it->second.slot;
        it = textures.find(placeholder);
        return it != textures.end() && it->second.slot != kNoSlot ? it->second.slot : 0;
    }

    /**
     * The frame's table handle for a map, added on first use: one entry holding the slot for a plain texture, or for a
     * UDIM set a header (kUdimEntry | rows), the placeholder's slot, then ten slots per row of tiles (U 0..9).
     */
    uint32_t tableHandle(const TexturePtr& t, uint64_t placeholder)
    {
        if (!t)
            return kNoHandle;
        auto [it, added] = frameHandles.try_emplace({t->key, placeholder}, kNoHandle);
        if (!added)
            return it->second;
        const uint32_t handle = uint32_t(frameTable.size());
        const uint32_t fallback = slotOf(placeholder, placeholder);
        if (t->vt)
            vt::tableEntry(*t, frameTable, [&](const TexturePtr& chunk) { return slotOf(chunk->key, placeholder); });
        else if (!t->isUdimSet())
            frameTable.push_back(slotOf(textureKey(t, placeholder), placeholder));
        else
        {
            int rows = 1;
            for (const UdimTile& tile : t->tiles)
                if (tile.udim >= 1001)
                    rows = std::max(rows, (tile.udim - 1001) / 10 + 1);
            rows = std::min(rows, 100);
            frameTable.push_back(kUdimEntry | uint32_t(rows));
            frameTable.push_back(fallback);
            const size_t first = frameTable.size();
            frameTable.resize(first + size_t(rows) * 10, fallback);
            for (const UdimTile& tile : t->tiles)
            {
                const int index = tile.udim - 1001;
                if (index >= 0 && index < rows * 10 && tile.image && !tile.image->isUdimSet())
                    frameTable[first + size_t(index)] = slotOf(textureKey(tile.image, placeholder), placeholder);
            }
        }
        if (handle >= kNoHandle)
        {
            if (!warnedTable)
                log::warn("render: too many material maps in one frame; some draw without textures");
            warnedTable = true;
            frameTable.resize(handle);
            return it->second = kNoHandle;
        }
        return it->second = handle;
    }

    /** Builds the table every bindless draw this frame reads its maps through and copies it into the frame's buffer. */
    void writeTextureTable(Frame& f, const FrameState& s, const std::vector<std::pair<TexturePtr, uint64_t>>& maps)
    {
        frameTable.clear();
        frameHandles.clear();
        frameStyles.clear();
        // The virtual texture feedback header, filled in once the frame knows its feedback buffer.
        frameTable.assign(vt::kTableHeaderWords, 0u);
        if (bindless)
            for (const auto& [map, placeholder] : maps)
                tableHandle(map, placeholder);
        writeStyles(s);
        if (frameTable.size() <= vt::kTableHeaderWords)
            frameTable.push_back(slotOf(kWhiteSrgbKey, kWhiteSrgbKey));
        const size_t bytes = frameTable.size() * sizeof(uint32_t);
        if (f.table.size < bytes)
        {
            // The frame's fence has signalled, so nothing reads the old table any more.
            compute::destroy(vk, f.table);
            try
            {
                f.table = compute::makeBuffer(vk, std::max<size_t>(bytes * 2, 4096), 0, compute::Host::Write);
            }
            catch (const std::exception& e)
            {
                log::error("render: {}", e.what());
                frameTableAddress = 0;
                return;
            }
        }
        std::memcpy(f.table.mapped, frameTable.data(), bytes);
        vmaFlushAllocation(vk.allocator, f.table.alloc, 0, VK_WHOLE_SIZE);
        frameTableAddress = f.table.address;
    }

    /**
     * Groups the frame's mesh draws for instancing and writes their placements into the frame's instance buffer. kept, a
     * view's batches from its last frame, is reused when reuse says this frame draws the same list, and refreshed otherwise.
     */
    void writeInstances(Frame& f, const FrameState& s, DrawBatches* kept = nullptr, bool reuse = false)
    {
        LILA_ZONE();

        if (kept && reuse)
        {
            rebind(*kept, s.meshes.list());
            frameBatches = *kept;
        }
        else
        {
            frameBatches = batchDraws(s, meshWirePipeline != VK_NULL_HANDLE);
            if (kept)
                *kept = frameBatches;
        }
        frameInstanceAddress = 0;
        const size_t bytes = std::max<size_t>(frameBatches.instances.size(), 1) * sizeof(GpuInstance);
        if (f.instances.size < bytes)
        {
            // The frame's fence has signalled, so nothing reads the old buffer any more.
            compute::destroy(vk, f.instances);
            try
            {
                f.instances = compute::makeBuffer(vk, std::max<size_t>(bytes * 2, 64 * sizeof(GpuInstance)), 0, compute::Host::Write);
            }
            catch (const std::exception& e)
            {
                log::error("render: {}", e.what());
                frameBatches = {};
                return;
            }
        }
        if (!frameBatches.instances.empty())
        {
            std::memcpy(f.instances.mapped, frameBatches.instances.data(), frameBatches.instances.size() * sizeof(GpuInstance));
            vmaFlushAllocation(vk.allocator, f.instances.alloc, 0, VK_WHOLE_SIZE);
        }
        frameInstanceAddress = f.instances.address;
    }

    /** The frame's lights into its light buffer, handing shadow tiles to the first four that cast shadows. */
    void writeLights(Frame& f, const FrameState& s)
    {
        frameLightAddress = 0;
        frameLightCount = 0;
        frameShadows.clear();
        frameRtLights.clear();
        const bool flat = s.lighting == FrameState::Lighting::Flat;
        if ((s.lights.empty() || flat) && !s.fog.on)
            return;
        const size_t count = flat ? 0 : std::min<size_t>(s.lights.size(), kMaxLights);
        const size_t bytes = (count + 1) * sizeof(GpuLight);
        if (f.lights.size < bytes)
        {
            compute::destroy(vk, f.lights);
            try
            {
                f.lights = compute::makeBuffer(vk, std::max<size_t>(bytes * 2, 16 * sizeof(GpuLight)), 0, compute::Host::Write);
            }
            catch (const std::exception& e)
            {
                log::error("render: {}", e.what());
                return;
            }
        }
        GpuFrame header{};
        if (s.fog.on)
        {
            header.fogColor = glm::vec4(s.fog.color * s.envIntensity, s.fog.density);
            header.fogParams = glm::vec4(s.fog.height, s.fog.falloff, s.fog.start, s.fog.sunScatter);
            header.fogExtra = glm::vec4(s.fog.anisotropy, s.zUp ? 2.0f : 1.0f, 0.0f, 0.0f);
        }
        std::memcpy(f.lights.mapped, &header, sizeof(header));
        if (s.rtShadows && vk.rayQuery && rtShadowPipeline)
            for (size_t i = 0; i < count && frameRtLights.size() < 4; ++i)
                if (s.lights[i].castsShadows)
                    frameRtLights.push_back(uint32_t(i) + 1);
        auto* out = static_cast<GpuLight*>(f.lights.mapped) + 1;
        constexpr uint32_t tilesPerRow = kShadowAtlas / kShadowTile;
        for (size_t i = 0; i < count; ++i)
        {
            const LightDraw& l = s.lights[i];
            GpuLight g{};
            // The kind in the low four bits, the lighting channels above: exact in a float up to 20 channels.
            g.position = glm::vec4(l.position, float(uint32_t(l.kind) | ((l.lightingChannels & 0xFFFFFu) << 4)));
            g.direction = glm::vec4(glm::normalize(l.direction), l.range);
            g.radiance = glm::vec4(l.radiance, l.radius);
            g.cone = glm::vec4(l.cosOuter, 1.0f / std::max(l.cosInner - l.cosOuter, 1e-4f), l.width, l.height);
            g.right = glm::vec4(l.right, l.shadowTexel);
            const uint32_t tile = uint32_t(frameShadows.size());
            if (l.shadows && tile < tilesPerRow * tilesPerRow && shadowPipeline &&
                std::ranges::find(frameRtLights, uint32_t(i) + 1) == frameRtLights.end())
            {
                g.shadow = l.shadowViewProj;
                g.shadowDepth = l.shadowDepth;
                const float scale = float(kShadowTile) / float(kShadowAtlas);
                g.shadowRect = glm::vec4(float(tile % tilesPerRow) * scale, float(tile / tilesPerRow) * scale, scale, scale);
                frameShadows.emplace_back(tile, l.shadowViewProj);
            }
            out[i] = g;
        }
        vmaFlushAllocation(vk.allocator, f.lights.alloc, 0, VK_WHOLE_SIZE);
        frameLightAddress = f.lights.address;
        frameLightCount = uint32_t(count);
    }

    /** Depth from each shadow-casting light into its atlas tile, leaving the atlas ready for the mesh shader to sample. */
    void recordShadows(VkCommandBuffer cmd, const FrameState& s)
    {
        constexpr auto fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto depthStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        constexpr auto depthWrite = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        constexpr auto read = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        if (frameShadows.empty())
        {
            barrier(cmd, shadowAtlas.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, fragment, 0, fragment, read,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
            return;
        }
        LILA_ZONE();
        barrier(cmd, shadowAtlas.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, fragment, 0, depthStage, depthWrite,
                VK_IMAGE_ASPECT_DEPTH_BIT);
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = shadowAtlas.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {0.0f, 0}; // reversed-Z
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {kShadowAtlas, kShadowAtlas}};
        ri.layerCount = 1;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 0, 1, &envSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 2, 1, &whiteAoSet, 0, nullptr);
        constexpr uint32_t tilesPerRow = kShadowAtlas / kShadowTile;
        cullCmd = VK_NULL_HANDLE;
        for (const auto& [tile, viewProj] : frameShadows)
        {
            const VkViewport viewport{float(tile % tilesPerRow * kShadowTile), float(tile / tilesPerRow * kShadowTile), float(kShadowTile),
                                      float(kShadowTile), 0.0f, 1.0f};
            const VkRect2D scissor{{int32_t(tile % tilesPerRow * kShadowTile), int32_t(tile / tilesPerRow * kShadowTile)}, {kShadowTile, kShadowTile}};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            VkDescriptorSet boundSet = VK_NULL_HANDLE;
            for (SectionRecord& r : *frameRecords)
                if (!r.translucent)
                    emitSection(cmd, s, r.push, r.set, *r.mesh, r.section, frameBatches.shaded[r.batch].count, viewProj, boundSet);
        }
        vkCmdEndRendering(cmd);
        barrier(cmd, shadowAtlas.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, depthStage, depthWrite,
                fragment, read, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    /** Appends a style block per shaded draw section whose material uses a stylised shading model. */
    void writeStyles(const FrameState& s)
    {
        for (const MeshDraw& d : s.meshes)
        {
            if (!(d.flags & DrawShaded) || !d.mesh)
                continue;
            for (uint32_t slot = 0; slot < d.materials.size(); ++slot)
            {
                const MaterialStyle& style = d.materials[slot].style;
                if (style.model == ShadingModel::DefaultLit)
                    continue;
                uint32_t stroke = 0xffffffffu;
                if (style.strokeMap && !style.strokeMap->isUdimSet() && textureKey(style.strokeMap, 0) != 0)
                {
                    // The fixed-sampler path binds the sheet itself; bindless reads its slot through the table.
                    auto it = frameHandles.find({style.strokeMap->key, kWhiteLinearKey});
                    if (!bindless)
                        stroke = 0;
                    else if (it != frameHandles.end() && it->second != kNoHandle)
                        stroke = it->second;
                }
                const auto block = packStyle(style, stroke, d.mesh->boundsMin, d.mesh->boundsMax, s.zUp);
                frameStyles[{&d, slot}] = uint32_t(frameTable.size());
                frameTable.insert(frameTable.end(), block.begin(), block.end());
            }
        }
    }

    /** Uploads the meshes this frame draws for the first time into device memory. Runs before rendering begins. */
    /** known: the draws are the last frame's, whose meshes that frame already uploaded. */
    void prepareMeshes(VkCommandBuffer cmd, Frame& f, const FrameState& s, bool known = false)
    {
        LILA_ZONE();

        const size_t staged = f.staging.size();
        if (!known)
            for (const MeshDraw& d : s.meshes)
                if (d.mesh && !meshCache.contains(d.mesh->key))
                    gpuMesh(d.mesh, cmd, &f);
        for (const PointDraw& p : s.points)
            if (p.cloud && !p.cloud->points.empty() && !pointCache.contains(p.cloud->key))
            {
                GpuPoints g;
                g.buffer = deviceBuffer(cmd, f, p.cloud->points.data(), p.cloud->points.size() * sizeof(PointVertex),
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
                VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                bai.buffer = g.buffer.buffer;
                g.address = vkGetBufferDeviceAddress(vk.device, &bai);
                g.count = uint32_t(p.cloud->points.size());
                g.keep = p.cloud;
                pointCache.emplace(p.cloud->key, std::move(g));
            }
        if (f.staging.size() != staged)
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        prepareHeightfields(cmd, f, s);
    }

    // ---------------------------------------------------------------- heightfield draw path

    /** Uploads the heights of heightfields drawn for the first time, keyed by their buffer. Runs before rendering begins. */
    void prepareHeightfields(VkCommandBuffer cmd, Frame& f, const FrameState& s)
    {
        if (s.heightfields.empty())
            return;
        LILA_ZONE();

        bool uploaded = false;
        for (const HeightfieldDraw& d : s.heightfields)
        {
            if (!d.heights || d.heights->size() < size_t(d.nx) * size_t(d.nz) || d.nx < 2 || d.nz < 2)
                continue;
            auto it = heightCache.find(d.heights.get());
            if (it != heightCache.end())
            {
                it->second.lastUsed = frameCounter;
                continue;
            }
            GpuHeights g;
            g.buffer = deviceBuffer(cmd, f, d.heights->data(), d.heights->size() * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            g.address = bufferAddress(g.buffer.buffer);
            g.keep = d.heights;
            g.lastUsed = frameCounter;
            heightCache.emplace(d.heights.get(), std::move(g));
            uploaded = true;
        }
        if (uploaded)
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    /** The heightfields, in the scene pass, opaque and depth tested like meshes. */
    void drawHeightfields(VkCommandBuffer cmd, const FrameState& s, const glm::mat4& viewProj)
    {
        if (s.heightfields.empty() || !heightfieldPipeline)
            return;
        LILA_ZONE();

        bool bound = false;
        for (const HeightfieldDraw& d : s.heightfields)
        {
            auto it = d.heights ? heightCache.find(d.heights.get()) : heightCache.end();
            if (it == heightCache.end())
                continue;
            // Every step-th sample, so a draw stays near four million vertices however large the grid.
            const double samples = double(d.nx) * double(d.nz);
            const int step = std::max(1, int(std::ceil(std::sqrt(samples / 4.0e6))));
            HeightfieldPush p{};
            p.viewProj = viewProj;
            const glm::mat4 rows = glm::transpose(d.world);
            p.model0 = rows[0];
            p.model1 = rows[1];
            p.model2 = rows[2];
            p.grid = glm::vec4(float(d.nx), float(d.nz), float(step), d.cell);
            p.origin = glm::vec4(d.origin, s.exposure, d.selected ? 1.0f : 0.0f);
            p.color = glm::vec4(d.color, 1.0f);
            const glm::vec3 key = s.zUp ? glm::vec3(0.35f, -0.45f, 0.82f) : glm::vec3(0.35f, 0.82f, 0.45f);
            p.light = glm::vec4(glm::normalize(key), s.zUp ? 1.0f : 0.0f);
            p.heights = it->second.address;
            p.cols = uint32_t((d.nx - 1 + step - 1) / step);
            p.rows = uint32_t((d.nz - 1 + step - 1) / step);
            if (!bound)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, heightfieldPipeline);
                bound = true;
            }
            vkCmdPushConstants(cmd, heightfieldLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
            vkCmdDraw(cmd, p.cols * p.rows * 6, 1, 0, 0);
        }
    }

    /** Drops the GPU copies of heights no frame has drawn for a while, and at once those nothing else holds. */
    void evictHeightfields()
    {
        std::vector<const void*> stale;
        for (auto& [key, g] : heightCache)
            if (g.lastUsed + 240 < frameCounter || (g.keep.use_count() == 1 && g.lastUsed != frameCounter))
                stale.push_back(key);
        if (stale.empty())
            return;
        vkDeviceWaitIdle(vk.device);
        for (const void* key : stale)
        {
            destroyBuffer(heightCache[key].buffer);
            heightCache.erase(key);
        }
    }

    /** Copies the vertices edited since the last frame into the GPU copies of dynamic meshes. Runs before rendering begins. */
    /** dynamic, when given, lists the indices of the draws with dynamic meshes, so the rest needn't be looked at. */
    void prepareDynamicMeshes(VkCommandBuffer cmd, Frame& f, const FrameState& s, const std::vector<uint32_t>* dynamic = nullptr)
    {
        LILA_ZONE();

        std::vector<VkBufferCopy> regions;
        std::vector<MeshVertex> data;
        bool first = true;
        const size_t count = dynamic ? dynamic->size() : s.meshes.size();
        for (size_t k = 0; k < count; ++k)
        {
            const MeshDraw& d = s.meshes[dynamic ? (*dynamic)[k] : k];
            if (!d.mesh || !d.mesh->dynamic)
                continue;
            auto it = meshCache.find(d.mesh->key);
            if (it == meshCache.end())
            {
                gpuMesh(d.mesh);
                continue;
            }
            DynamicVertices& dv = *d.mesh->dynamic;
            if (!dv.pending.exchange(false, std::memory_order_acquire) && it->second.keep == d.mesh)
                continue;
            regions.clear();
            data.clear();
            {
                std::scoped_lock lock(dv.mutex);
                if (dv.allDirty)
                {
                    data = dv.vertices;
                    regions.push_back({0, 0, data.size() * sizeof(MeshVertex)});
                }
                else if (!dv.dirty.empty())
                {
                    std::sort(dv.dirty.begin(), dv.dirty.end());
                    dv.dirty.erase(std::unique(dv.dirty.begin(), dv.dirty.end()), dv.dirty.end());
                    // Short gaps are cheaper to copy than to split into separate regions.
                    for (size_t i = 0; i < dv.dirty.size();)
                    {
                        size_t j = i + 1;
                        while (j < dv.dirty.size() && dv.dirty[j] - dv.dirty[j - 1] <= 16)
                            ++j;
                        uint32_t begin = dv.dirty[i], end = std::min<uint32_t>(dv.dirty[j - 1] + 1, uint32_t(dv.vertices.size()));
                        if (begin < end)
                        {
                            regions.push_back({data.size() * sizeof(MeshVertex), begin * sizeof(MeshVertex), (end - begin) * sizeof(MeshVertex)});
                            data.insert(data.end(), dv.vertices.begin() + begin, dv.vertices.begin() + end);
                        }
                        i = j;
                    }
                }
                dv.dirty.clear();
                dv.allDirty = false;
            }
            // Every snapshot of a live mesh shares one key, so the wire's points come from whichever copy was uploaded first.
            GpuMesh& g = it->second;
            const bool edgesStale = g.keep != d.mesh && !d.mesh->edgePoints.empty() && d.mesh->edgePoints.size() == g.edgePointCount;
            if (g.keep != d.mesh)
                g.keep = d.mesh;
            if (regions.empty() && !edgesStale)
                continue;
            if (first)
            {
                // Earlier frames may still read these vertices.
                memoryBarrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                              VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                first = false;
            }
            if (!regions.empty())
            {
                AllocatedBuffer staging = createBuffer(data.data(), data.size() * sizeof(MeshVertex), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                f.staging.push_back(staging);
                vkCmdCopyBuffer(cmd, staging.buffer, g.vertices.buffer, uint32_t(regions.size()), regions.data());
            }
            if (edgesStale)
            {
                const size_t bytes = d.mesh->edgePoints.size() * sizeof(glm::vec3);
                AllocatedBuffer staging = createBuffer(d.mesh->edgePoints.data(), bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                f.staging.push_back(staging);
                const VkBufferCopy region{0, 0, bytes};
                vkCmdCopyBuffer(cmd, staging.buffer, g.edges.buffer, 1, &region);
            }
        }
        if (!first)
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                          VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
    }

    static void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                              VkAccessFlags2 dstAccess)
    {
        VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStage;
        b.srcAccessMask = srcAccess;
        b.dstStageMask = dstStage;
        b.dstAccessMask = dstAccess;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    /** Descriptor set binding a material's maps, made on first use and kept while frames use it. */
    VkDescriptorSet materialSet(const Material* m)
    {
        std::array<uint64_t, kMaterialBindings> keys = {kWhiteSrgbKey, kFlatNormalKey, kWhiteLinearKey, kWhiteSrgbKey, kWhiteLinearKey};
        if (m)
            keys = {textureKey(m->baseColorMap, kWhiteSrgbKey), textureKey(m->normalMap, kFlatNormalKey), textureKey(m->ormMap, kWhiteLinearKey),
                    textureKey(m->emissiveMap, kWhiteSrgbKey), textureKey(m->style.strokeMap, kWhiteLinearKey)};
        auto it = materialSets.find(keys);
        if (it == materialSets.end())
        {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = materialPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &materialSetLayout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(vk.device, &ai, &set) != VK_SUCCESS)
            {
                if (!warnedPool)
                    log::warn("render: out of material descriptor sets; some materials draw without textures");
                warnedPool = true;
                std::array<uint64_t, kMaterialBindings> fallback = {kWhiteSrgbKey, kFlatNormalKey, kWhiteLinearKey, kWhiteSrgbKey, kWhiteLinearKey};
                auto fb = materialSets.find(fallback);
                return fb != materialSets.end() ? fb->second.set : VK_NULL_HANDLE;
            }
            VkDescriptorImageInfo info[kMaterialBindings];
            VkWriteDescriptorSet w[kMaterialBindings]{};
            for (uint32_t i = 0; i < kMaterialBindings; ++i)
            {
                info[i] = {materialSampler, textures[keys[i]].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet = set;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[i].pImageInfo = &info[i];
            }
            vkUpdateDescriptorSets(vk.device, kMaterialBindings, w, 0, nullptr);
            it = materialSets.emplace(keys, MaterialSet{set, 0}).first;
        }
        it->second.lastUsed = frameCounter;
        return it->second.set;
    }

    /** Drops textures and sets no frame has used for a while. */
    void evictTextures()
    {
        if (frameCounter % 120 != 60)
            return;
        auto stale = [&](uint64_t last) { return last + 240 < frameCounter; };
        std::vector<uint64_t> deadTextures;
        for (auto& [key, t] : textures)
            if (key < kFlatNormalKey && stale(t.lastUsed))
                deadTextures.push_back(key);
        // Kept records bind their sets without looking them up again, so lastUsed alone can't tell they are in use.
        std::unordered_set<VkDescriptorSet> recorded;
        for (auto& [id, v] : viewports)
            if (v->keptRecordsValid)
                for (const SectionRecord& r : v->keptRecords)
                    recorded.insert(r.set);
        std::vector<std::array<uint64_t, kMaterialBindings>> deadSets;
        for (auto& [keys, set] : materialSets)
            if ((stale(set.lastUsed) && !recorded.contains(set.set)) || std::ranges::any_of(keys, [&](uint64_t k) { return std::ranges::find(deadTextures, k) != deadTextures.end(); }))
                deadSets.push_back(keys);
        if (deadTextures.empty() && deadSets.empty())
            return;
        vkDeviceWaitIdle(vk.device);
        for (const auto& keys : deadSets)
        {
            vkFreeDescriptorSets(vk.device, materialPool, 1, &materialSets[keys].set);
            materialSets.erase(keys);
        }
        for (uint64_t key : deadTextures)
        {
            GpuTexture& t = textures[key];
            destroyImage(t.image);
            if (t.slot != kNoSlot)
                freeSlots.push_back(t.slot);
            textures.erase(key);
        }
    }

    void freeStaging(Frame& f)
    {
        for (AllocatedBuffer& b : f.staging)
            destroyBuffer(b);
        f.staging.clear();
        for (compute::Buffer& b : f.scratch)
            compute::destroy(vk, b);
        f.scratch.clear();
    }

    /** Frees the staging of frames the GPU has finished; true while some frame still holds staging. */
    bool reclaimStaging()
    {
        bool held = false;
        for (auto& [id, v] : viewports)
            for (Frame& f : v->frames)
                if (!f.staging.empty())
                {
                    if (vkGetFenceStatus(vk.device, f.fence) == VK_SUCCESS)
                        freeStaging(f);
                    else
                        held = true;
                }
        return held;
    }

    // ---------------------------------------------------------------- ambient occlusion

    VkDescriptorSet allocateSet(VkDescriptorPool pool, VkDescriptorSetLayout layout)
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(vk.device, &ai, &set) != VK_SUCCESS)
            log::error("render: out of ambient occlusion descriptor sets");
        return set;
    }

    void writeImage(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, VkImageView view, VkImageLayout layout,
                    VkSampler sampler = VK_NULL_HANDLE)
    {
        if (!set)
            return;
        VkDescriptorImageInfo ii{type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? VK_NULL_HANDLE : sampler ? sampler : pointSampler, view, layout};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
    }

    void initAo()
    {
        LILA_ZONE();

        VkDescriptorSetLayoutBinding sb[5];
        for (uint32_t i = 0; i < 5; ++i)
            sb[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 5;
        lci.pBindings = sb;
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &aoSampleLayout);

        VkDescriptorSetLayoutBinding cb[4];
        for (uint32_t i = 0; i < 4; ++i)
            cb[i] = {i, i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        lci.bindingCount = 4;
        lci.pBindings = cb;
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &aoComputeLayout);

        for (uint32_t i = 0; i < 4; ++i)
            cb[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        lci.bindingCount = 4;
        lci.pBindings = cb;
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &composeSetLayout);
        for (uint32_t i = 0; i < 4; ++i)
            cb[i] = {i, i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &postComputeLayout);

        if (vk.rayQuery)
        {
            VkDescriptorSetLayoutBinding rb[4] = {
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            };
            lci.bindingCount = 4;
            lci.pBindings = rb;
            vkCreateDescriptorSetLayout(vk.device, &lci, nullptr, &rtaoSetLayout);
        }

        VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxAoSets * 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxAoSets * 2},
                                         {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 64}};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pci.maxSets = kMaxAoSets;
        pci.poolSizeCount = vk.rayQuery ? 3 : 2;
        pci.pPoolSizes = sizes;
        vkCreateDescriptorPool(vk.device, &pci, nullptr, &aoPool);

        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(vk.physical, kSceneFormat, &fp);
        constexpr VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        sceneFormat = (fp.optimalTilingFeatures & need) == need ? kSceneFormat : kSceneFallbackFormat;

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(vk.device, &sci, nullptr, &pointSampler);

        float one = 1.0f;
        whiteAo = createTexture(kAoFormat, {1, 1}, 1, false);
        uploadImage(whiteAo.image, {1, 1}, 1, 1, sizeof(float), {&one});
        whiteAoSet = allocateSet(aoPool, aoSampleLayout);
        for (uint32_t b = 0; b < 3; ++b)
            writeImage(whiteAoSet, b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, whiteAo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const uint16_t zero[4] = {0, 0, 0, 0};
        blackBevel = createTexture(VK_FORMAT_R16G16B16A16_SFLOAT, {1, 1}, 1, false);
        uploadImage(blackBevel.image, {1, 1}, 1, 1, sizeof(zero), {zero});
        writeImage(whiteAoSet, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, blackBevel.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        writeImage(whiteAoSet, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, whiteAo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    void destroyAo()
    {
        destroyImage(whiteAo);
        if (pointSampler)
            vkDestroySampler(vk.device, pointSampler, nullptr);
        if (aoPool)
            vkDestroyDescriptorPool(vk.device, aoPool, nullptr);
        for (Image* img : {&identityLut, &blackBevel})
            destroyImage(*img);
        for (auto& [key, img] : luts)
            destroyImage(img);
        luts.clear();
        for (VkDescriptorSetLayout* l : {&aoSampleLayout, &aoComputeLayout, &composeSetLayout, &postComputeLayout, &rtaoSetLayout})
            if (*l)
            {
                vkDestroyDescriptorSetLayout(vk.device, *l, nullptr);
                *l = VK_NULL_HANDLE;
            }
        pointSampler = VK_NULL_HANDLE;
        aoPool = VK_NULL_HANDLE;
    }

    /** Sized with the swapchain. The two AO images stay in GENERAL so compute can write and shaders can sample them. */
    void createAoImages(Viewport& v, VkExtent2D extent)
    {
        LILA_ZONE();

        constexpr auto sampled = VK_IMAGE_USAGE_SAMPLED_BIT;
        v.aoDepth = createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT,
                                VK_IMAGE_ASPECT_DEPTH_BIT);
        v.aoNormal = createImage(kAoNormalFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT);
        v.aoRaw = createImage(kAoFormat, VK_IMAGE_USAGE_STORAGE_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        v.aoFinal = createImage(kAoFormat, VK_IMAGE_USAGE_STORAGE_BIT | sampled | VK_IMAGE_USAGE_TRANSFER_DST_BIT, extent, VK_SAMPLE_COUNT_1_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT);

        constexpr auto sampler = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        constexpr auto storage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        constexpr auto depthRead = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        constexpr auto shaderRead = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        for (VkDescriptorSet set : {v.gtaoSet, v.denoiseSet})
        {
            writeImage(set, 0, sampler, v.aoDepth.view, depthRead);
            writeImage(set, 1, sampler, v.aoNormal.view, shaderRead);
        }
        writeImage(v.gtaoSet, 2, sampler, v.aoFinal.view, general);
        writeImage(v.gtaoSet, 3, storage, v.aoRaw.view, general);
        writeImage(v.denoiseSet, 2, sampler, v.aoRaw.view, general);
        writeImage(v.denoiseSet, 3, storage, v.aoFinal.view, general);
        writeImage(v.rtaoSet, 0, sampler, v.aoDepth.view, depthRead);
        writeImage(v.rtaoSet, 1, sampler, v.aoNormal.view, shaderRead);
        writeImage(v.rtaoSet, 3, storage, v.aoRaw.view, general);
        v.bevel = createImage(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        writeImage(v.bevelSet, 0, sampler, v.aoDepth.view, depthRead);
        writeImage(v.bevelSet, 1, sampler, v.aoNormal.view, shaderRead);
        writeImage(v.bevelSet, 3, storage, v.bevel.view, general);
        writeImage(v.aoSampleSet, 3, sampler, v.bevel.view, general);
        v.rtShadow = createImage(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT);
        writeImage(v.shadowSet, 0, sampler, v.aoDepth.view, depthRead);
        writeImage(v.shadowSet, 1, sampler, v.aoNormal.view, shaderRead);
        writeImage(v.shadowSet, 3, storage, v.rtShadow.view, general);
        writeImage(v.aoSampleSet, 4, sampler, v.rtShadow.view, general);
        writeImage(v.aoSampleSet, 0, sampler, v.aoFinal.view, general);
        writeImage(v.aoSampleSet, 1, sampler, v.aoDepth.view, depthRead);
        writeImage(v.aoSampleSet, 2, sampler, v.aoNormal.view, shaderRead);

        v.postColor = createImage(v.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        writeImage(v.postSet, 0, sampler, v.postColor.view, shaderRead);
        writeImage(v.postSet, 1, sampler, v.aoDepth.view, depthRead);
        writeImage(v.postSet, 2, sampler, v.aoNormal.view, shaderRead);
    }

    /** The scene target, its resolved depth and the temporal history, with the sets the TAA and compose passes read them through. */
    void createSceneTargets(Viewport& v, VkExtent2D extent)
    {
        LILA_ZONE();

        constexpr auto sampled = VK_IMAGE_USAGE_SAMPLED_BIT;
        const bool msaa = vk.msaa != VK_SAMPLE_COUNT_1_BIT;
        if (msaa)
        {
            v.sceneMs = createImage(sceneFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, extent, vk.msaa,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
            v.sceneDepth = createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | sampled, extent, VK_SAMPLE_COUNT_1_BIT,
                                       VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        constexpr auto copied = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        v.scene = createImage(sceneFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | sampled | copied, extent, VK_SAMPLE_COUNT_1_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
        for (Image& h : v.history)
            h = createImage(kHistoryFormat, VK_IMAGE_USAGE_STORAGE_BIT | sampled | copied, extent, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        v.historyValid = false;

        constexpr auto sampler = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto shaderRead = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        constexpr auto depthRead = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        const VkImageView depthViews[2] = {msaa ? v.sceneDepth.view : v.depth.view, v.aoDepth.view};
        for (int i = 0; i < 2; ++i)
            for (int d = 0; d < 2; ++d)
            {
                VkDescriptorSet set = v.taaSets[i][d];
                writeImage(set, 0, sampler, v.scene.view, shaderRead);
                writeImage(set, 1, sampler, depthViews[d], depthRead);
                writeImage(set, 2, sampler, v.history[1 - i].view, general, linearSampler);
                writeImage(set, 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, v.history[i].view, general);
            }
        createPostTargets(v, extent);
    }

    /** A 2D image of `mips` levels with a view of each besides the whole one, for compute passes writing level by level. */
    Image createLevels(VkFormat format, VkImageUsageFlags usage, VkExtent2D extent, uint32_t mips, std::vector<VkImageView>& levels)
    {
        Image img;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {extent.width, extent.height, 1};
        ici.mipLevels = mips;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.usage = usage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(vk.allocator, &ici, &aci, &img.image, &img.alloc, nullptr) != VK_SUCCESS)
            return {};
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        vkCreateImageView(vk.device, &vci, nullptr, &img.view);
        for (uint32_t mip = 0; mip < mips; ++mip)
        {
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
            VkImageView view = VK_NULL_HANDLE;
            vkCreateImageView(vk.device, &vci, nullptr, &view);
            levels.push_back(view);
        }
        return img;
    }

    /** Bloom chains, the exposure histogram and state, and the sets the bloom, exposure and compose passes read them through. */
    void createPostTargets(Viewport& v, VkExtent2D extent)
    {
        LILA_ZONE();

        constexpr auto storage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkExtent2D half{std::max(extent.width / 2, 1u), std::max(extent.height / 2, 1u)};
        uint32_t levels = 1;
        while (levels < kMaxBloomLevels && std::min(half.width, half.height) >> levels >= 8)
            ++levels;
        v.bloomLevels = levels;
        v.bloomDown = createLevels(kBloomFormat, storage, half, levels, v.bloomDownViews);
        v.bloomUp = createLevels(kBloomFormat, storage, half, levels, v.bloomUpViews);
        v.histogram = createImage(VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, {64, 1}, VK_SAMPLE_COUNT_1_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
        v.exposureState = createImage(VK_FORMAT_R32_SFLOAT, storage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, {2, 1}, VK_SAMPLE_COUNT_1_BIT,
                                      VK_IMAGE_ASPECT_COLOR_BIT);
        v.postImagesReady = false;
        v.exposureValid = false;
        v.lutKey = 0;
        if (!v.bloomDown.image || !v.bloomUp.image)
            return;

        constexpr auto sampler = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        constexpr auto storageType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto shaderRead = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const std::pair<VkImageView, VkImageLayout> inputs[3] = {{v.scene.view, shaderRead}, {v.history[0].view, general}, {v.history[1].view, general}};
        for (int i = 0; i < 3; ++i)
        {
            writeImage(v.bloomSets[i], 0, sampler, inputs[i].first, inputs[i].second, linearSampler);
            writeImage(v.bloomSets[i], 2, storageType, v.bloomDownViews[0], general);
            writeImage(v.exposureSets[i], 0, sampler, inputs[i].first, inputs[i].second);
            writeImage(v.exposureSets[i], 2, storageType, v.histogram.view, general);
            writeImage(v.exposureSets[i], 3, storageType, v.exposureState.view, general);
            writeImage(v.composeSets[i], 0, sampler, inputs[i].first, inputs[i].second, linearSampler);
            writeImage(v.composeSets[i], 1, sampler, v.bloomUpViews[0], general, linearSampler);
            writeImage(v.composeSets[i], 3, sampler, v.exposureState.view, general);
        }
        for (uint32_t mip = 1; mip < levels; ++mip)
        {
            VkDescriptorSet down = v.bloomSets[3 + mip - 1];
            writeImage(down, 0, sampler, v.bloomDownViews[mip - 1], general, linearSampler);
            writeImage(down, 2, storageType, v.bloomDownViews[mip], general);
        }
        // Up level mip blends the level below it (the down chain's last level at the bottom) into the down chain's own level.
        for (uint32_t mip = 0; mip + 1 < levels; ++mip)
        {
            VkDescriptorSet up = v.bloomSets[3 + kMaxBloomLevels - 1 + mip];
            writeImage(up, 0, sampler, mip + 2 == levels ? v.bloomDownViews[mip + 1] : v.bloomUpViews[mip + 1], general, linearSampler);
            writeImage(up, 1, sampler, v.bloomDownViews[mip], general, linearSampler);
            writeImage(up, 2, storageType, v.bloomUpViews[mip], general);
        }
    }


    static bool anyShaded(const FrameState& s)
    {
        return std::ranges::any_of(s.meshes, [](const MeshDraw& d) { return (d.flags & DrawShaded) && d.mesh && !d.mesh->indices.empty(); });
    }

    /**
     * Records the depth/normal prepass and both AO dispatches ahead of the main pass, leaving v.aoFinal ready for the
     * mesh shader. The images are shared by both frames in flight, so each first use waits on the previous frame's reads.
     */
    void recordAo(VkCommandBuffer cmd, Frame& f, Viewport& v, const FrameState& s, const glm::mat4& viewProj, bool gtao = true)
    {
        LILA_ZONE();

        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto colorStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        constexpr auto depthStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        constexpr auto depthWrite = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        constexpr auto read = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;

        barrier(cmd, v.aoDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, compute | fragment, 0, depthStage,
                depthWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
        barrier(cmd, v.aoNormal.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, compute | fragment, 0, colorStage,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, color);

        VkRenderingAttachmentInfo normal{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        normal.imageView = v.aoNormal.view;
        normal.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        normal.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        normal.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = v.aoDepth.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {0.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, v.extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &normal;
        ri.pDepthAttachment = &depth;
        // The main pass's section records, so the prepass costs a draw call per section and nothing more.
        if (chunked())
        {
            ChunkPass pass;
            pass.color = kAoNormalFormat;
            pass.extent = v.extent;
            pass.pipeline = prepassPipeline;
            pass.aoSet = whiteAoSet;
            pass.viewProj = viewProj;
            const std::vector<VkCommandBuffer> chunks = recordChunks(f, s, pass);
            ri.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
            vkCmdBeginRendering(cmd, &ri);
            vkCmdExecuteCommands(cmd, uint32_t(chunks.size()), chunks.data());
            cullCmd = VK_NULL_HANDLE;
        }
        else
        {
            vkCmdBeginRendering(cmd, &ri);
            VkViewport viewport{0, 0, float(v.extent.width), float(v.extent.height), 0, 1};
            VkRect2D scissor{{0, 0}, v.extent};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, prepassPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 0, 1, &envSet, 0, nullptr);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 2, 1, &whiteAoSet, 0, nullptr);
            cullCmd = VK_NULL_HANDLE;
            VkDescriptorSet boundSet = VK_NULL_HANDLE;
            for (SectionRecord& r : *frameRecords)
                if (!r.translucent)
                    emitSection(cmd, s, r.push, r.set, *r.mesh, r.section, frameBatches.shaded[r.batch].count, viewProj, boundSet);
        }
        vkCmdEndRendering(cmd);

        barrier(cmd, v.aoDepth.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, depthStage,
                depthWrite, compute | fragment, read, VK_IMAGE_ASPECT_DEPTH_BIT);
        barrier(cmd, v.aoNormal.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorStage,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, compute | fragment, read, color);
        if (gtao && !(s.aoRayTraced && recordRtao(cmd, f, v, s, viewProj)))
            recordGtao(cmd, v, s, false);
    }

    /** Fast AO: GTAO from the main pass's depth (resolved into aoDepth) with normals taken from it, ready for applyAo. */
    void recordFastAo(VkCommandBuffer cmd, Viewport& v, const FrameState& s)
    {
        LILA_ZONE();

        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto depthStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        // A depth resolve writes in the colour attachment output stage.
        barrier(cmd, v.aoDepth.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                depthStage | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, compute, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT);
        barrier(cmd, v.aoNormal.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, compute | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                0, compute, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        recordGtao(cmd, v, s, true);
    }

    /** Multiplies the frame's meshes by the fast AO, in a rendering pass that continues the main one. */
    void applyAo(VkCommandBuffer cmd, Viewport& v)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, aoApplyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, aoApplyLayout, 0, 1, &v.aoSampleSet, 0, nullptr);
        // Applied to the display-encoded colour as is, which darkens about as much as the accurate mode does.
        const glm::vec4 push(1.0f, 0.0f, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, aoApplyLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    /** GTAO and its blur from aoDepth (and aoNormal unless depthNormals), leaving aoFinal ready for fragment shaders. */
    void recordGtao(VkCommandBuffer cmd, Viewport& v, const FrameState& s, bool depthNormals)
    {
        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto read = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        constexpr auto storageWrite = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier(cmd, v.aoRaw.image, VK_IMAGE_LAYOUT_UNDEFINED, general, compute, 0, compute, storageWrite, color);

        glm::dmat4 proj = s.proj;
        AoPush p;
        p.invProj = glm::mat4(glm::inverse(proj));
        p.view = glm::mat4(s.view);
        bool ortho = proj[3][3] == 1.0;
        p.params = glm::vec4(std::max(s.aoRadius, 0.01f), std::max(s.aoIntensity, 0.0f), float(proj[0][0] * 0.5 * v.extent.width),
                             ortho ? 1.0f : 0.0f);
        p.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        p.options.x = depthNormals ? 1.0f : 0.0f;
        uint32_t gx = (v.extent.width + 7) / 8, gy = (v.extent.height + 7) / 8;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gtaoPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoLayout, 0, 1, &v.gtaoSet, 0, nullptr);
        vkCmdPushConstants(cmd, aoLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, gx, gy, 1);

        barrier(cmd, v.aoRaw.image, general, general, compute, storageWrite, compute, read, color);
        recordDenoise(cmd, v, s);
    }

    /** The edge-aware blur from aoRaw into aoFinal, leaving it ready for fragment shaders. */
    void recordDenoise(VkCommandBuffer cmd, Viewport& v, const FrameState& s)
    {
        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto read = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        constexpr auto storageWrite = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        glm::dmat4 proj = s.proj;
        AoPush p;
        p.invProj = glm::mat4(glm::inverse(proj));
        p.view = glm::mat4(s.view);
        bool ortho = proj[3][3] == 1.0;
        p.params = glm::vec4(std::max(s.aoRadius, 0.01f), std::max(s.aoIntensity, 0.0f), float(proj[0][0] * 0.5 * v.extent.width),
                             ortho ? 1.0f : 0.0f);
        p.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        uint32_t gx = (v.extent.width + 7) / 8, gy = (v.extent.height + 7) / 8;
        barrier(cmd, v.aoFinal.image, VK_IMAGE_LAYOUT_UNDEFINED, general, fragment, 0, compute, storageWrite, color);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, denoisePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoLayout, 0, 1, &v.denoiseSet, 0, nullptr);
        vkCmdPushConstants(cmd, aoLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, gx, gy, 1);
        barrier(cmd, v.aoFinal.image, general, general, compute, storageWrite, fragment, read, color);
    }

    // ---------------------------------------------------------------- ray tracing

    VkDeviceAddress bufferAddress(VkBuffer b) const
    {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, b};
        return vkGetBufferDeviceAddress(vk.device, &ai);
    }

    void destroyBlas(GpuMesh& g)
    {
        if (g.blas)
            vkDestroyAccelerationStructureKHR(vk.device, g.blas, nullptr);
        compute::destroy(vk, g.blasBuffer);
        g.blas = VK_NULL_HANDLE;
        g.blasAddress = 0;
    }

    /** The mesh's bottom-level structure, recorded into cmd; its scratch goes with the frame. */
    bool buildBlas(VkCommandBuffer cmd, Frame& f, GpuMesh& g, uint32_t vertexCount)
    {
        VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = bufferAddress(g.vertices.buffer);
        geom.geometry.triangles.vertexStride = sizeof(MeshVertex);
        geom.geometry.triangles.maxVertex = std::max(vertexCount, 1u) - 1;
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = bufferAddress(g.indices.buffer);
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = 1;
        info.pGeometries = &geom;
        const uint32_t triangles = g.indexCount / 3;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &triangles, &sizes);
        try
        {
            g.blasBuffer = compute::makeBuffer(vk, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                               compute::Host::None);
            f.scratch.push_back(compute::makeBuffer(vk, sizes.buildScratchSize + vk.asScratchAlignment, 0, compute::Host::None));
        }
        catch (const std::exception& e)
        {
            log::error("render: {}", e.what());
            return false;
        }
        VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        aci.buffer = g.blasBuffer.buffer;
        aci.size = sizes.accelerationStructureSize;
        aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(vk.device, &aci, nullptr, &g.blas);
        const VkDeviceSize align = vk.asScratchAlignment;
        info.dstAccelerationStructure = g.blas;
        info.scratchData.deviceAddress = (f.scratch.back().address + align - 1) / align * align;
        const VkAccelerationStructureBuildRangeInfoKHR range{triangles, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &info, &ranges);
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, nullptr, g.blas};
        g.blasAddress = vkGetAccelerationStructureDeviceAddressKHR(vk.device, &ai);
        return true;
    }

    /**
     * Rebuilds the scene's top-level structure in place from the frame's shaded placements, building bottom-level ones
     * for meshes that have none. Deforming meshes are left out. False when there is nothing to trace or no ray tracing.
     */
    bool buildSceneTlas(VkCommandBuffer cmd, Frame& f)
    {
        if (!vk.rayQuery || (!rtaoPipeline && !bevelPipeline))
            return false;
        // Built once a frame however many passes trace it.
        if (frameTlasBuilt == frameCounter)
            return frameTlasOk;
        frameTlasBuilt = frameCounter;
        frameTlasOk = buildSceneTlasNow(cmd, f);
        return frameTlasOk;
    }

    uint64_t frameTlasBuilt = ~uint64_t(0);
    bool frameTlasOk = false;

    bool buildSceneTlasNow(VkCommandBuffer cmd, Frame& f)
    {
        LILA_ZONE();

        std::vector<VkAccelerationStructureInstanceKHR> instances;
        std::vector<TraceInstance> table;
        bool built = false;
        for (const DrawBatch& b : frameBatches.shaded)
        {
            const MeshDraw& d = *b.draw;
            if (!d.mesh || d.mesh->dynamic || d.mesh->indices.empty())
                continue;
            GpuMesh* g = gpuMesh(d.mesh);
            if (!g || g->indexCount < 3)
                continue;
            if (!g->blas)
            {
                const RenderMesh& source = d.mesh->shareVertices ? *d.mesh->shareVertices : *d.mesh;
                if (!buildBlas(cmd, f, *g, uint32_t(source.vertices.size())))
                    continue;
                built = true;
            }
            for (uint32_t i = 0; i < b.count; ++i)
            {
                const glm::mat4& w = frameBatches.instances[b.first + i].world;
                VkAccelerationStructureInstanceKHR inst{};
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c)
                        inst.transform.matrix[r][c] = w[c][r];
                inst.mask = 0xff;
                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                inst.accelerationStructureReference = g->blasAddress;
                inst.instanceCustomIndex = uint32_t(table.size()) & 0xffffff;
                instances.push_back(inst);
                TraceInstance ti;
                ti.vertices = bufferAddress(g->vertices.buffer);
                ti.indices = bufferAddress(g->indices.buffer);
                for (int r = 0; r < 3; ++r)
                    ti.rows[r] = glm::vec4(w[0][r], w[1][r], w[2][r], w[3][r]);
                table.push_back(ti);
            }
        }
        if (instances.empty())
            return false;
        if (built)
            compute::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                   VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

        const size_t bytes = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
        try
        {
            if (f.tlasInstances.size < bytes)
            {
                compute::destroy(vk, f.tlasInstances);
                f.tlasInstances = compute::makeBuffer(vk, bytes * 2, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                      compute::Host::Write);
            }
        }
        catch (const std::exception& e)
        {
            log::error("render: {}", e.what());
            return false;
        }
        std::memcpy(f.tlasInstances.mapped, instances.data(), bytes);
        vmaFlushAllocation(vk.allocator, f.tlasInstances.alloc, 0, VK_WHOLE_SIZE);
        try
        {
            const size_t tableBytes = table.size() * sizeof(TraceInstance);
            if (f.traceTable.size < tableBytes)
            {
                compute::destroy(vk, f.traceTable);
                f.traceTable = compute::makeBuffer(vk, tableBytes * 2, 0, compute::Host::Write);
            }
            std::memcpy(f.traceTable.mapped, table.data(), tableBytes);
            vmaFlushAllocation(vk.allocator, f.traceTable.alloc, 0, VK_WHOLE_SIZE);
            frameTraceTable = f.traceTable.address;
        }
        catch (const std::exception& e)
        {
            log::error("render: {}", e.what());
            frameTraceTable = 0;
        }

        VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geom.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
        geom.geometry.instances.data.deviceAddress = f.tlasInstances.address;
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = 1;
        info.pGeometries = &geom;
        const uint32_t count = uint32_t(instances.size());
        if (!tlas || count > tlasCapacity)
        {
            // A bigger structure: the sets reading the old one are rewritten before their next use.
            vkDeviceWaitIdle(vk.device);
            if (tlas)
                vkDestroyAccelerationStructureKHR(vk.device, tlas, nullptr);
            compute::destroy(vk, tlasBuffer);
            compute::destroy(vk, tlasScratch);
            tlas = VK_NULL_HANDLE;
            tlasCapacity = std::max(count * 2, 64u);
            VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            vkGetAccelerationStructureBuildSizesKHR(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &tlasCapacity, &sizes);
            try
            {
                tlasBuffer = compute::makeBuffer(vk, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                                 compute::Host::None);
                tlasScratch = compute::makeBuffer(vk, sizes.buildScratchSize + vk.asScratchAlignment, 0, compute::Host::None);
            }
            catch (const std::exception& e)
            {
                log::error("render: {}", e.what());
                tlasCapacity = 0;
                return false;
            }
            VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            aci.buffer = tlasBuffer.buffer;
            aci.size = sizes.accelerationStructureSize;
            aci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            vkCreateAccelerationStructureKHR(vk.device, &aci, nullptr, &tlas);
            ++tlasGeneration;
        }
        // The previous frame's traces finish before this rebuild overwrites the structure.
        compute::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
        const VkDeviceSize align = vk.asScratchAlignment;
        info.dstAccelerationStructure = tlas;
        info.scratchData.deviceAddress = (tlasScratch.address + align - 1) / align * align;
        const VkAccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &info, &ranges);
        compute::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        return true;
    }

    static bool anyBevel(const FrameState& s)
    {
        for (const MeshDraw& d : s.meshes)
            if (d.flags & DrawShaded)
                for (const Material& m : d.materials)
                    if (m.bevelRadius > 0.0f)
                        return true;
        return false;
    }

    /** aoFinal as unoccluded, for frames whose traced passes bind the AO set while AO itself is off. */
    void clearAo(VkCommandBuffer cmd, Viewport& v)
    {
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        constexpr auto transfer = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        const VkClearColorValue one{{1.0f, 1.0f, 1.0f, 1.0f}};
        const VkImageSubresourceRange range{color, 0, 1, 0, 1};
        barrier(cmd, v.aoFinal.image, VK_IMAGE_LAYOUT_UNDEFINED, general, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, transfer,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, color);
        vkCmdClearColorImage(cmd, v.aoFinal.image, general, &one, 1, &range);
        barrier(cmd, v.aoFinal.image, general, general, transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, color);
    }

    /** Traced visibility from the prepass into v.rtShadow, a channel for each light frameRtLights names. */
    bool recordRtShadow(VkCommandBuffer cmd, Frame& f, Viewport& v, const FrameState& s, const glm::mat4& viewProj)
    {
        if (!v.shadowSet || frameRtLights.empty() || !buildSceneTlas(cmd, f))
            return false;
        LILA_ZONE();
        if (v.shadowGeneration != tlasGeneration)
        {
            Frame& other = v.frames[(v.frameIndex + 1) % kFramesInFlight];
            vkWaitForFences(vk.device, 1, &other.fence, VK_TRUE, UINT64_MAX);
            VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            as.accelerationStructureCount = 1;
            as.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.pNext = &as;
            w.dstSet = v.shadowSet;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
            v.shadowGeneration = tlasGeneration;
        }
        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier(cmd, v.rtShadow.image, VK_IMAGE_LAYOUT_UNDEFINED, general, fragment, 0, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, color);
        struct
        {
            glm::mat4 invViewProj;
            glm::vec4 position[4];
            glm::vec4 size[4];
            glm::vec4 params;
            glm::vec4 extent;
        } p{};
        p.invViewProj = glm::inverse(viewProj);
        for (size_t c = 0; c < frameRtLights.size(); ++c)
        {
            const LightDraw& l = s.lights[frameRtLights[c] - 1];
            if (l.kind == LightDraw::Directional)
            {
                // The sun's angular radius: its source radius taken as degrees, the real sun's quarter degree by default.
                const float angle = glm::radians(l.radius > 0.0f ? std::min(l.radius, 10.0f) : 0.27f);
                p.position[c] = glm::vec4(-glm::normalize(l.direction), 0.0f);
                p.size[c] = glm::vec4(std::tan(angle), 0.0f, 0.0f, 0.0f);
            }
            else
            {
                p.position[c] = glm::vec4(l.position, 1.0f);
                p.size[c] = glm::vec4(std::max(l.radius, 0.5f), l.range, 0.0f, 0.0f);
            }
        }
        p.params = glm::vec4(float(frameRtLights.size()), 0.0f, float(frameCounter % 65536), s.accumulateOnly ? 4.0f : 2.0f);
        p.extent = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rtShadowPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rtShadowLayout, 0, 1, &v.shadowSet, 0, nullptr);
        vkCmdPushConstants(cmd, rtShadowLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, (v.extent.width + 7) / 8, (v.extent.height + 7) / 8, 1);
        barrier(cmd, v.rtShadow.image, general, general, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, fragment, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                color);
        return true;
    }

    /** The live bevel's rounded normals from the prepass into v.bevel. */
    bool recordBevel(VkCommandBuffer cmd, Frame& f, Viewport& v, const FrameState& s, const glm::mat4& viewProj)
    {
        if (!v.bevelSet || !buildSceneTlas(cmd, f) || !frameTraceTable)
            return false;
        LILA_ZONE();
        if (v.bevelGeneration != tlasGeneration)
        {
            Frame& other = v.frames[(v.frameIndex + 1) % kFramesInFlight];
            vkWaitForFences(vk.device, 1, &other.fence, VK_TRUE, UINT64_MAX);
            VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            as.accelerationStructureCount = 1;
            as.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.pNext = &as;
            w.dstSet = v.bevelSet;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
            v.bevelGeneration = tlasGeneration;
        }
        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier(cmd, v.bevel.image, VK_IMAGE_LAYOUT_UNDEFINED, general, fragment, 0, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, color);
        struct
        {
            glm::mat4 invViewProj;
            glm::vec4 params;
            glm::vec4 size;
            uint64_t instances;
            uint64_t pad;
        } p;
        p.invViewProj = glm::inverse(viewProj);
        p.params = glm::vec4(0.0f, 0.0f, float(frameCounter % 65536), s.accumulateOnly ? 32.0f : 16.0f);
        p.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        p.instances = frameTraceTable;
        p.pad = 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bevelPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bevelLayout, 0, 1, &v.bevelSet, 0, nullptr);
        vkCmdPushConstants(cmd, bevelLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, (v.extent.width + 7) / 8, (v.extent.height + 7) / 8, 1);
        barrier(cmd, v.bevel.image, general, general, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, fragment, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, color);
        return true;
    }

    /** Ray-traced AO from the prepass into aoRaw, then the shared denoise into aoFinal; false leaves it to GTAO. */
    bool recordRtao(VkCommandBuffer cmd, Frame& f, Viewport& v, const FrameState& s, const glm::mat4& viewProj)
    {
        if (!v.rtaoSet || !buildSceneTlas(cmd, f))
            return false;
        if (v.rtaoGeneration != tlasGeneration)
        {
            Frame& other = v.frames[(v.frameIndex + 1) % kFramesInFlight];
            vkWaitForFences(vk.device, 1, &other.fence, VK_TRUE, UINT64_MAX);
            VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            as.accelerationStructureCount = 1;
            as.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.pNext = &as;
            w.dstSet = v.rtaoSet;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
            v.rtaoGeneration = tlasGeneration;
        }
        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier(cmd, v.aoRaw.image, VK_IMAGE_LAYOUT_UNDEFINED, general, compute, 0, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, color);
        struct
        {
            glm::mat4 invViewProj;
            glm::vec4 params;
            glm::vec4 size;
        } p;
        p.invViewProj = glm::inverse(viewProj);
        p.params = glm::vec4(std::max(s.aoRadius, 0.01f), std::max(s.aoIntensity, 0.0f), float(frameCounter % 65536), s.accumulateOnly ? 8.0f : 4.0f);
        p.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rtaoPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rtaoLayout, 0, 1, &v.rtaoSet, 0, nullptr);
        vkCmdPushConstants(cmd, rtaoLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, (v.extent.width + 7) / 8, (v.extent.height + 7) / 8, 1);
        barrier(cmd, v.aoRaw.image, general, general, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, compute, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, color);
        recordDenoise(cmd, v, s);
        return true;
    }

    // ---------------------------------------------------------------- pipelines

    VkShaderModule module(const std::vector<uint32_t>& code) { return compute::shaderModule(vk, code); }

    /** A requested line width within what the device rasterizes. */
    float lineWidth(float w) const
    {
        if (!vk.wideLines)
            return 1.0f;
        const float* range = vk.properties.limits.lineWidthRange;
        return std::clamp(w, std::max(1.0f, range[0]), std::max(1.0f, range[1]));
    }

    VkPipeline buildPipeline(VkPipelineLayout layout, VkShaderModule mod, const char* vs, const char* ps, VkFormat format,
                             VkPrimitiveTopology topology, const VkPipelineVertexInputStateCreateInfo& vertexInput, bool depthWrite,
                             bool blend, bool depthTest = true, float depthBias = 0.0f, bool dynamicCull = false, bool singleSample = false,
                             bool dynamicLineWidth = false, const VkPipelineColorBlendAttachmentState* blendState = nullptr, bool withDepth = true,
                             bool withColor = true, const VkSpecializationInfo* spec = nullptr)
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, mod, vs, spec};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, mod, ps, spec};

        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topology;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        // Reversed-Z: a positive bias pulls toward the camera, which keeps wireframes on top of their own faces.
        rs.depthBiasEnable = depthBias != 0.0f;
        rs.depthBiasConstantFactor = depthBias;
        rs.depthBiasSlopeFactor = depthBias != 0.0f ? 1.0f : 0.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = singleSample ? VK_SAMPLE_COUNT_1_BIT : vk.msaa;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = depthTest;
        ds.depthWriteEnable = depthWrite;
        ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // reversed-Z
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = 0xF;
        if (blend)
        {
            att.blendEnable = VK_TRUE;
            att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            att.colorBlendOp = VK_BLEND_OP_ADD;
            att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            att.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        if (blendState)
            att = *blendState;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = withColor ? 1 : 0;
        cb.pAttachments = &att;
        VkDynamicState dyn[4] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        uint32_t dynCount = 2;
        if (dynamicCull)
            dyn[dynCount++] = VK_DYNAMIC_STATE_CULL_MODE;
        if (dynamicLineWidth)
            dyn[dynCount++] = VK_DYNAMIC_STATE_LINE_WIDTH;
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = dynCount;
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        ri.colorAttachmentCount = withColor ? 1 : 0;
        ri.pColorAttachmentFormats = &format;
        ri.depthAttachmentFormat = withDepth ? kDepthFormat : VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &ri;
        pci.stageCount = 2;
        pci.pStages = stages;
        pci.pVertexInputState = &vertexInput;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp;
        pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb;
        pci.pDynamicState = &dy;
        pci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pci, nullptr, &p) != VK_SUCCESS)
            log::error("render: pipeline creation failed ({} / {})", vs, ps);
        return p;
    }

    VkPipeline buildCompute(VkPipelineLayout layout, VkShaderModule mod, const char* entry)
    {
        VkPipeline p = compute::computePipeline(vk, layout, mod, entry);
        if (!p)
            log::error("render: compute pipeline creation failed ({})", entry);
        return p;
    }

    void createPipelines(VkFormat format)
    {
        LILA_ZONE();

        destroyPipelines();
        pipelineFormat = format;

        auto layoutFor = [&](uint32_t size, VkShaderStageFlags stages, VkDescriptorSetLayout set = VK_NULL_HANDLE) {
            return compute::pipelineLayout(vk, size, stages, set ? std::span(&set, 1) : std::span<const VkDescriptorSetLayout>());
        };
        gridLayout = layoutFor(sizeof(GridPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        linesLayout = layoutFor(sizeof(LinesPush), VK_SHADER_STAGE_VERTEX_BIT);
        pointsLayout = layoutFor(sizeof(PointsPush), VK_SHADER_STAGE_VERTEX_BIT);
        pointsCullLayout = layoutFor(sizeof(PointsCullPush), VK_SHADER_STAGE_COMPUTE_BIT);
        overlayTrisLayout = layoutFor(sizeof(glm::vec2), VK_SHADER_STAGE_VERTEX_BIT, atlasSetLayout);

        VkShaderModule grid = module(loadSpirv("grid.spv"));
        VkShaderModule lines = module(loadSpirv("lines.spv"));

        VkPipelineVertexInputStateCreateInfo none{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        if (grid)
        {
            gridPipeline = buildPipeline(gridLayout, grid, "gridVS", "gridPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none, false, true);
            gridFramePipeline = buildPipeline(gridLayout, grid, "gridVS", "gridPS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none, false, true);
        }

        VkVertexInputBindingDescription binding{0, sizeof(LineVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[2] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LineVertex, pos)},
            {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(LineVertex, color)},
        };
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vin.vertexBindingDescriptionCount = 1;
        vin.pVertexBindingDescriptions = &binding;
        vin.vertexAttributeDescriptionCount = 2;
        vin.pVertexAttributeDescriptions = attrs;
        if (lines)
        {
            linesPipeline = buildPipeline(linesLayout, lines, "linesVS", "linesPS", format, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, vin, true, false);
            overlayPipeline =
                buildPipeline(linesLayout, lines, "linesVS", "linesPS", format, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, vin, false, true, false);
            linesOnTopPipeline = buildPipeline(linesLayout, lines, "linesVS", "linesPS", format, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, vin, false,
                                               true, true, 0.0f, false, false, true);
            tris3dPipeline =
                buildPipeline(linesLayout, lines, "linesVS", "linesPS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, vin, false, true, true, 2.0f);
            linesOverPipeline = buildPipeline(linesLayout, lines, "linesVS", "linesPS", format, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, vin, false,
                                              true, false, 0.0f, false, false, true);
            if (VkShaderModule points = module(loadSpirv("points.spv")))
            {
                VkVertexInputBindingDescription pb{0, sizeof(PointVertex), VK_VERTEX_INPUT_RATE_VERTEX};
                VkVertexInputAttributeDescription pa[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PointVertex, pos)},
                                                           {1, 0, VK_FORMAT_R32_UINT, offsetof(PointVertex, state)}};
                VkPipelineVertexInputStateCreateInfo pvin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
                pvin.vertexBindingDescriptionCount = 1;
                pvin.pVertexBindingDescriptions = &pb;
                pvin.vertexAttributeDescriptionCount = 2;
                pvin.pVertexAttributeDescriptions = pa;
                pointsPipeline =
                    buildPipeline(pointsLayout, points, "pointsVS", "pointsPS", format, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, pvin, false, false, true);
                vkDestroyShaderModule(vk.device, points, nullptr);
            }
        }

        {
            const VkDescriptorSetLayout sets[3] = {envSetLayout, bindless ? bindlessLayout : materialSetLayout, aoSampleLayout};
            meshLayout = compute::pipelineLayout(vk, sizeof(MeshPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sets);
        }
        wireLayout = layoutFor(sizeof(WirePush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        if (VkShaderModule mesh = module(loadSpirv(bindless ? "mesh_bindless.spv" : "mesh.spv")))
        {
            VkVertexInputBindingDescription mb{0, sizeof(MeshVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription ma[6] = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, pos)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, normal)},
                {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshVertex, tangent)},
                {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(MeshVertex, uv)},
                {4, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(MeshVertex, uv1)},
                {5, 0, VK_FORMAT_R16G16B16A16_SFLOAT, offsetof(MeshVertex, color)},
            };
            VkPipelineVertexInputStateCreateInfo mvin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            mvin.vertexBindingDescriptionCount = 1;
            mvin.pVertexBindingDescriptions = &mb;
            mvin.vertexAttributeDescriptionCount = 6;
            mvin.pVertexAttributeDescriptions = ma;
            meshPipeline =
                buildPipeline(meshLayout, mesh, "meshVS", "meshPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin, true, false, true, 0.0f, true);
            const VkBool32 lean = VK_TRUE;
            const VkSpecializationMapEntry leanEntry{0, 0, sizeof(lean)};
            const VkSpecializationInfo leanSpec{1, &leanEntry, sizeof(lean), &lean};
            meshLeanPipeline = buildPipeline(meshLayout, mesh, "meshVS", "meshPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin, true, false,
                                             true, 0.0f, true, false, false, nullptr, true, true, &leanSpec);
            meshBlendPipeline =
                buildPipeline(meshLayout, mesh, "meshVS", "meshPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin, false, true, true, 0.0f, true);
            prepassPipeline = buildPipeline(meshLayout, mesh, "meshVS", "prepassPS", kAoNormalFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin, true,
                                            false, true, 0.0f, true, true);
            shadowPipeline = buildPipeline(meshLayout, mesh, "meshVS", "prepassPS", VK_FORMAT_UNDEFINED, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin, true,
                                           false, true, 0.0f, true, true, false, nullptr, true, false);
            if (vk.fragmentBarycentric)
                if (VkShaderModule wired = module(loadSpirv(bindless ? "mesh_bindless_wire.spv" : "mesh_wire.spv")))
                {
                    meshWirePipeline = buildPipeline(meshLayout, wired, "meshVS", "meshWirePS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin, true,
                                                     false, true, 0.0f, true);
                    meshWireLeanPipeline = buildPipeline(meshLayout, wired, "meshVS", "meshWirePS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin,
                                                         true, false, true, 0.0f, true, false, false, nullptr, true, true, &leanSpec);
                    meshWireBlendPipeline = buildPipeline(meshLayout, wired, "meshVS", "meshWirePS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, mvin,
                                                          false, true, true, 0.0f, true);
                    vkDestroyShaderModule(vk.device, wired, nullptr);
                }
            backgroundPipeline = buildPipeline(meshLayout, mesh, "backgroundVS", "backgroundPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                               none, false, false, false);
            vkDestroyShaderModule(vk.device, mesh, nullptr);
        }
        aoLayout = layoutFor(sizeof(AoPush), VK_SHADER_STAGE_COMPUTE_BIT, aoComputeLayout);
        if (rtaoSetLayout)
        {
            rtaoLayout = layoutFor(sizeof(glm::mat4) + 2 * sizeof(glm::vec4), VK_SHADER_STAGE_COMPUTE_BIT, rtaoSetLayout);
            if (VkShaderModule rt = module(loadSpirv("rtao.spv")))
            {
                rtaoPipeline = buildCompute(rtaoLayout, rt, "rtaoCS");
                vkDestroyShaderModule(vk.device, rt, nullptr);
            }
            bevelLayout = layoutFor(sizeof(glm::mat4) + 2 * sizeof(glm::vec4) + sizeof(uint64_t) * 2, VK_SHADER_STAGE_COMPUTE_BIT, rtaoSetLayout);
            if (VkShaderModule rt = module(loadSpirv("rtbevel.spv")))
            {
                bevelPipeline = buildCompute(bevelLayout, rt, "bevelCS");
                vkDestroyShaderModule(vk.device, rt, nullptr);
            }
            rtShadowLayout = layoutFor(sizeof(glm::mat4) + 10 * sizeof(glm::vec4), VK_SHADER_STAGE_COMPUTE_BIT, rtaoSetLayout);
            if (VkShaderModule rt = module(loadSpirv("rtshadow.spv")))
            {
                rtShadowPipeline = buildCompute(rtShadowLayout, rt, "shadowCS");
                vkDestroyShaderModule(vk.device, rt, nullptr);
            }
        }
        if (VkShaderModule cull = module(loadSpirv("points_cull.spv")))
        {
            pointsNearestPipeline = buildCompute(pointsCullLayout, cull, "pointsNearestCS");
            pointsKeepPipeline = buildCompute(pointsCullLayout, cull, "pointsKeepCS");
            vkDestroyShaderModule(vk.device, cull, nullptr);
        }
        if (VkShaderModule ao = module(loadSpirv("ao.spv")))
        {
            gtaoPipeline = buildCompute(aoLayout, ao, "gtaoCS");
            denoisePipeline = buildCompute(aoLayout, ao, "denoiseCS");
            vkDestroyShaderModule(vk.device, ao, nullptr);
        }
        aoApplyLayout = layoutFor(sizeof(glm::vec4), VK_SHADER_STAGE_FRAGMENT_BIT, aoSampleLayout);
        if (VkShaderModule apply = module(loadSpirv("ao_apply.spv")))
        {
            VkPipelineColorBlendAttachmentState multiply{};
            multiply.blendEnable = VK_TRUE;
            multiply.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
            multiply.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            multiply.colorBlendOp = VK_BLEND_OP_ADD;
            multiply.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            multiply.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            multiply.alphaBlendOp = VK_BLEND_OP_ADD;
            multiply.colorWriteMask = 0xF;
            aoApplyPipeline = buildPipeline(aoApplyLayout, apply, "applyVS", "applyPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none, false,
                                            false, false, 0.0f, false, false, false, &multiply);
            vkDestroyShaderModule(vk.device, apply, nullptr);
        }
        postLayout = layoutFor(sizeof(PostPush), VK_SHADER_STAGE_FRAGMENT_BIT, aoSampleLayout);
        if (VkShaderModule post = module(loadSpirv("post.spv")))
        {
            postPipeline = buildPipeline(postLayout, post, "postVS", "postPS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none, false, false, false);
            vkDestroyShaderModule(vk.device, post, nullptr);
        }
        taaLayout = layoutFor(sizeof(TaaPush), VK_SHADER_STAGE_COMPUTE_BIT, aoComputeLayout);
        if (VkShaderModule taa = module(loadSpirv("taa.spv")))
        {
            taaPipeline = buildCompute(taaLayout, taa, "taaCS");
            vkDestroyShaderModule(vk.device, taa, nullptr);
        }
        composeLayout = layoutFor(sizeof(ComposePush), VK_SHADER_STAGE_FRAGMENT_BIT, composeSetLayout);
        bloomLayout = layoutFor(sizeof(BloomPush), VK_SHADER_STAGE_COMPUTE_BIT, postComputeLayout);
        if (VkShaderModule bloom = module(loadSpirv("bloom.spv")))
        {
            bloomDownPipeline = buildCompute(bloomLayout, bloom, "downCS");
            bloomUpPipeline = buildCompute(bloomLayout, bloom, "upCS");
            vkDestroyShaderModule(vk.device, bloom, nullptr);
        }
        exposureLayout = layoutFor(sizeof(ExposurePush), VK_SHADER_STAGE_COMPUTE_BIT, postComputeLayout);
        if (VkShaderModule exposure = module(loadSpirv("exposure.spv")))
        {
            histogramPipeline = buildCompute(exposureLayout, exposure, "histogramCS");
            averagePipeline = buildCompute(exposureLayout, exposure, "averageCS");
            vkDestroyShaderModule(vk.device, exposure, nullptr);
        }
        if (VkShaderModule compose = module(loadSpirv("compose.spv")))
        {
            composePipeline = buildPipeline(composeLayout, compose, "composeVS", "composePS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none, false,
                                            false, false);
            composeSinglePipeline = buildPipeline(composeLayout, compose, "composeVS", "composePS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none,
                                                  false, false, false, 0.0f, false, true, false, nullptr, false);
            VkPipelineColorBlendAttachmentState over{};
            over.blendEnable = VK_TRUE;
            over.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            over.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            over.colorBlendOp = VK_BLEND_OP_ADD;
            over.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            over.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            over.alphaBlendOp = VK_BLEND_OP_ADD;
            over.colorWriteMask = 0xF;
            composeOverPipeline = buildPipeline(composeLayout, compose, "composeVS", "composePS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none,
                                                false, false, false, 0.0f, false, false, false, &over);
            vkDestroyShaderModule(vk.device, compose, nullptr);
        }
        contactLayout = layoutFor(sizeof(ContactPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        if (VkShaderModule contact = module(loadSpirv("contact.spv")))
        {
            contactPipeline = buildPipeline(contactLayout, contact, "contactVS", "contactPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none,
                                            false, true);
            vkDestroyShaderModule(vk.device, contact, nullptr);
        }
        if (VkShaderModule wire = module(loadSpirv("wire.spv")))
        {
            VkVertexInputBindingDescription wb{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription wa{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
            VkPipelineVertexInputStateCreateInfo wvin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            wvin.vertexBindingDescriptionCount = 1;
            wvin.pVertexBindingDescriptions = &wb;
            wvin.vertexAttributeDescriptionCount = 1;
            wvin.pVertexAttributeDescriptions = &wa;
            wirePipeline = buildPipeline(wireLayout, wire, "wireVS", "wirePS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, wvin, false, true, true, 0.0f,
                                         false, false, true);
            wireXrayPipeline = buildPipeline(wireLayout, wire, "wireVS", "wirePS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, wvin, false, true, false,
                                             0.0f, false, false, true);
            vkDestroyShaderModule(vk.device, wire, nullptr);
        }

        if (VkShaderModule overlay = module(loadSpirv("overlay.spv")))
        {
            VkVertexInputBindingDescription ob{0, sizeof(OverlayVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription oa[3] = {
                {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(OverlayVertex, pos)},
                {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(OverlayVertex, uv)},
                {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(OverlayVertex, color)},
            };
            VkPipelineVertexInputStateCreateInfo ovin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            ovin.vertexBindingDescriptionCount = 1;
            ovin.pVertexBindingDescriptions = &ob;
            ovin.vertexAttributeDescriptionCount = 3;
            ovin.pVertexAttributeDescriptions = oa;
            overlayTrisPipeline = buildPipeline(overlayTrisLayout, overlay, "overlayVS", "overlayPS", format,
                                                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, ovin, false, true, false);
            vkDestroyShaderModule(vk.device, overlay, nullptr);
        }

        imagePlaneLayout = layoutFor(sizeof(ImagePlanePush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, materialSetLayout);
        if (VkShaderModule plane = module(loadSpirv("imageplane.spv")))
        {
            auto build = [&](bool depthWrite, bool depthTest) {
                return buildPipeline(imagePlaneLayout, plane, "imagePlaneVS", "imagePlanePS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none,
                                     depthWrite, true, depthTest);
            };
            imageOverPipeline = build(false, false);
            imageDepthPipeline = build(true, true);
            imageBlendPipeline = build(false, true);
            imageBehindPipeline = build(false, true);
            vkDestroyShaderModule(vk.device, plane, nullptr);
        }
        heightfieldLayout = layoutFor(sizeof(HeightfieldPush), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, materialSetLayout);
        if (VkShaderModule hf = module(loadSpirv("heightfield.spv")))
        {
            heightfieldPipeline = buildPipeline(heightfieldLayout, hf, "heightfieldVS", "heightfieldPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                none, true, false, true);
            vkDestroyShaderModule(vk.device, hf, nullptr);
        }
        externalLayout = layoutFor(sizeof(ExternalPush), VK_SHADER_STAGE_FRAGMENT_BIT, materialSetLayout);
        if (VkShaderModule external = module(loadSpirv("external.spv")))
        {
            VkPipelineColorBlendAttachmentState noColor{};
            externalDepthPipeline = buildPipeline(externalLayout, external, "externalVS", "externalPS", sceneFormat, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  none, true, false, true, 0.0f, false, false, false, &noColor);
            externalPipeline = buildPipeline(externalLayout, external, "externalVS", "externalPS", format, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, none,
                                             true, true, true);
            vkDestroyShaderModule(vk.device, external, nullptr);
        }

        if (grid)
            vkDestroyShaderModule(vk.device, grid, nullptr);
        if (lines)
            vkDestroyShaderModule(vk.device, lines, nullptr);
    }

    void destroyPipelines()
    {
        for (VkPipeline* p : {&gridPipeline, &linesPipeline, &overlayPipeline, &overlayTrisPipeline, &meshPipeline, &meshBlendPipeline, &meshWirePipeline,
                              &meshWireBlendPipeline, &meshLeanPipeline, &meshWireLeanPipeline, &aoApplyPipeline, &pointsPipeline, &wirePipeline,
                              &wireXrayPipeline, &backgroundPipeline, &linesOnTopPipeline, &tris3dPipeline, &linesOverPipeline, &contactPipeline, &prepassPipeline, &gtaoPipeline, &pointsNearestPipeline, &pointsKeepPipeline,
                              &denoisePipeline, &imageOverPipeline, &imageDepthPipeline, &imageBlendPipeline, &imageBehindPipeline, &gridFramePipeline, &externalDepthPipeline, &heightfieldPipeline,
                              &composeOverPipeline, &postPipeline, &externalPipeline,
                              &taaPipeline, &composePipeline, &composeSinglePipeline,
                              &bloomDownPipeline, &bloomUpPipeline, &histogramPipeline, &averagePipeline, &shadowPipeline, &rtaoPipeline, &bevelPipeline, &rtShadowPipeline})
            if (*p)
            {
                vkDestroyPipeline(vk.device, *p, nullptr);
                *p = VK_NULL_HANDLE;
            }
        for (VkPipelineLayout* l : {&gridLayout, &linesLayout, &overlayTrisLayout, &meshLayout, &wireLayout, &contactLayout, &aoLayout, &aoApplyLayout, &pointsLayout, &pointsCullLayout,
                                    &imagePlaneLayout, &postLayout, &externalLayout, &heightfieldLayout, &taaLayout, &composeLayout, &bloomLayout,
                                    &exposureLayout, &rtaoLayout, &bevelLayout, &rtShadowLayout})
            if (*l)
            {
                vkDestroyPipelineLayout(vk.device, *l, nullptr);
                *l = VK_NULL_HANDLE;
            }
    }

    // ---------------------------------------------------------------- frame

    static void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 srcStage,
                        VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, VkImageAspectFlags aspect,
                        uint32_t mips = 1, uint32_t layers = 1)
    {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStage;
        b.srcAccessMask = srcAccess;
        b.dstStageMask = dstStage;
        b.dstAccessMask = dstAccess;
        b.oldLayout = from;
        b.newLayout = to;
        b.image = image;
        b.subresourceRange = {aspect, 0, mips, 0, layers};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    /** Switches every viewport to the frame's sample count: rebuilds the pipelines now, the targets on each next frame. */
    void applyMsaa(const FrameState& s)
    {
        auto want = VkSampleCountFlagBits(msaaSamples(true, s.msaaSamples, vk.msaaSupported));
        if (want == vk.msaa)
            return;
        LILA_ZONE();
        vkDeviceWaitIdle(vk.device);
        vk.msaa = want;
        ++msaaGeneration;
        if (pipelineFormat != VK_FORMAT_UNDEFINED)
            createPipelines(pipelineFormat);
        log::info("render: {}x MSAA", int(want));
    }

    /**
     * Mailbox where the surface has it: the display takes the newest finished frame at each vblank, so a camera drag
     * shows within a vblank of the input instead of behind FIFO's queue of older frames, and a view never blocks the
     * render thread (and the other views) waiting for its vblank. Drawing is on demand and held to the frame cap, so
     * the GPU only runs while something changes. VSync asks for FIFO, which always exists.
     */
    VkPresentModeKHR presentMode(VkSurfaceKHR surface, bool vsync)
    {
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical, surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical, surface, &n, modes.data());
        bool mailbox = !vsync && std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end();
        const VkPresentModeKHR mode = mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
        if (mode != loggedPresentMode)
        {
            loggedPresentMode = mode;
            log::info("render: {} presentation", mailbox ? "mailbox" : "fifo");
        }
        return mode;
    }
    VkPresentModeKHR loggedPresentMode = VK_PRESENT_MODE_MAX_ENUM_KHR;

    /** The display's refresh period, read again each second as windows move between monitors. */
    std::chrono::duration<double> refreshPeriod(Viewport& v)
    {
        if (v.refreshCheck <= std::chrono::steady_clock::now())
        {
            v.refreshInterval = std::chrono::duration<double>(1.0 / refreshHz(v.window));
            v.refreshCheck = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        }
        return v.refreshInterval;
    }

    /** The shortest time between two of the view's frames under its frame cap; zero when uncapped. */
    std::chrono::steady_clock::duration frameInterval(Viewport& v)
    {
        const int cap = v.state.frameCap;
        const std::chrono::duration<double> d = cap == kFrameCapRefresh ? refreshPeriod(v)
                                                : cap > 0               ? std::chrono::duration<double>(1.0 / cap)
                                                                        : std::chrono::duration<double>(0.0);
        return std::chrono::duration_cast<std::chrono::steady_clock::duration>(d);
    }

    /** Per-frame numbers for a performance trace, from the last frame the GPU finished. */
    void plotFrame(const Renderer::FrameStats& t)
    {
        if (!LILA_PROFILER_CONNECTED())
            return;
        LILA_PLOT("Viewport GPU ms", t.gpuMs);
        LILA_PLOT("Viewport AO GPU ms", t.aoMs);
        LILA_PLOT("Viewport record CPU ms", t.cpuMs);
        LILA_PLOT("Mesh draw calls", int64_t(t.meshDrawCalls));
        LILA_PLOT("Mesh instances", int64_t(t.meshInstances));
        if (vk.memoryBudget)
        {
            VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
            vmaGetHeapBudgets(vk.allocator, budgets);
            const VkPhysicalDeviceMemoryProperties* mem = nullptr;
            vmaGetMemoryProperties(vk.allocator, &mem);
            uint64_t used = 0;
            for (uint32_t h = 0; h < mem->memoryHeapCount; ++h)
                if (mem->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    used += budgets[h].usage;
            LILA_PLOT("VRAM used MB", double(used) / (1024.0 * 1024.0));
        }
    }

    /** With the frame's fence signalled: what its timestamps measured. */
    void readTimings(Viewport& v, Frame& f)
    {
        if (!f.timestamps || !f.timed)
            return;
        f.timed = false;
        uint64_t t[4] = {};
        if (vkGetQueryPoolResults(vk.device, f.timestamps, 0, 4, sizeof(t), t, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return;
        double toMs = double(vk.properties.limits.timestampPeriod) * 1e-6;
        v.timing.gpuMs = double(t[3] - t[0]) * toMs;
        v.timing.aoMs = f.aoTimed ? double(t[2] - t[1]) * toMs : 0.0;
    }

    void timestamp(VkCommandBuffer cmd, Frame& f, uint32_t query)
    {
        if (f.timestamps)
            vkCmdWriteTimestamp2(cmd, query == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.timestamps, query);
    }

    /**
     * True when the frame could not be shown and should be drawn again straight away. latch runs once the frame's
     * fence and swapchain image are in hand, and may swap in a newer state submitted while the thread was blocked.
     */
    bool render(Viewport& v, FrameState& s, const std::function<void(FrameState&)>& latch = {})
    {
        LILA_ZONE();

        applyMsaa(s);
        if (v.msaaGeneration != msaaGeneration)
            v.needsRecreate = true;
        if (v.vsync != s.vsync && !v.offscreen)
        {
            v.vsync = s.vsync;
            v.needsRecreate = true;
        }

        // Not every driver reports a child window's resize through OUT_OF_DATE, so compare against the window itself.
        if (v.swapchain && !v.needsRecreate && !v.offscreen)
        {
            VkSurfaceCapabilitiesKHR caps;
            if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical, v.surface, &caps) == VK_SUCCESS && caps.currentExtent.width != UINT32_MAX &&
                (caps.currentExtent.width != v.extent.width || caps.currentExtent.height != v.extent.height))
                v.needsRecreate = true;
        }
        if (v.needsRecreate.exchange(false) && !createSwapchain(v))
        {
            v.needsRecreate = true;
            return false;
        }

        Frame& f = v.frames[v.frameIndex];
        {
            LILA_ZONE_N("Wait for frame in flight");
            vkWaitForFences(vk.device, 1, &f.fence, VK_TRUE, UINT64_MAX);
        }
        f.feedback.collect();
        freeStaging(f);
        readTimings(v, f);
        gpuProfile::collect();
        GpuImage::collectGarbage(vk);

        uint32_t imageIndex = 0;
        VkResult r = VK_SUCCESS;
        if (!v.offscreen)
        {
            LILA_ZONE_N("Acquire swapchain image");
            r = vkAcquireNextImageKHR(vk.device, v.swapchain, UINT64_MAX, f.acquired, VK_NULL_HANDLE, &imageIndex);
        }
        if (r == VK_ERROR_OUT_OF_DATE_KHR)
        {
            v.needsRecreate = true;
            return true;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
        {
            log::error("render: acquire failed ({})", vkResultName(r));
            return false;
        }
        vkResetFences(vk.device, 1, &f.fence);
        // Waiting above can take a whole vblank; draw what the user is doing now, not what they did before it.
        if (latch)
            latch(s);
        if (s.externalScene)
            handOverScene(s);

        if (s.atlas && s.atlas.get() != uploadedAtlas)
            uploadAtlas(s.atlas);

        // One buffer per frame: lines, linesOnTop, tris3d, overlay and linesXray share the LineVertex stride, overlayTris follow.
        const std::vector<LineVertex>* lineSets[] = {&s.lines, &s.linesOnTop, &s.tris3d, &s.overlay, &s.linesXray, &s.selectedEdges, &s.seamLines,
                                                     &s.symmetryLines, &s.highlightEdges, &s.curveHalo, &s.curveLines, &s.curveHaloBold, &s.curveLinesBold};
        constexpr int kLineSets = int(std::size(lineSets));
        uint32_t lineFirst[kLineSets];
        size_t lineBytes = 0;
        for (int i = 0; i < kLineSets; ++i)
        {
            lineFirst[i] = uint32_t(lineBytes / sizeof(LineVertex));
            lineBytes += lineSets[i]->size() * sizeof(LineVertex);
        }
        // Debug drawing follows the fixed sets: each line batch, then its triangles.
        const size_t debugBatches = s.debugLines ? s.debugLines->size() : 0;
        std::vector<uint32_t> debugFirst(debugBatches + 1);
        for (size_t i = 0; i <= debugBatches; ++i)
        {
            debugFirst[i] = uint32_t(lineBytes / sizeof(LineVertex));
            const std::vector<LineVertex>* verts = i < debugBatches ? &(*s.debugLines)[i].vertices : s.debugTris.get();
            lineBytes += verts ? verts->size() * sizeof(LineVertex) : 0;
        }
        size_t trisBytes = s.overlayTris.size() * sizeof(OverlayVertex);
        if (lineBytes + trisBytes)
        {
            ensureLines(f, lineBytes + trisBytes);
            LineVertex* mapped = static_cast<LineVertex*>(f.linesMapped);
            const glm::dmat4 clipViewProj = s.proj * s.view;
            const glm::vec3 behind(s.eye + glm::dvec3(s.view[0][2], s.view[1][2], s.view[2][2]));
            for (int i = 0; i < kLineSets; ++i)
            {
                std::memcpy(mapped + lineFirst[i], lineSets[i]->data(), lineSets[i]->size() * sizeof(LineVertex));
                // Every set but tris3d (triangles) and overlay (already in NDC) is a world-space line list.
                if (i != 2 && i != 3)
                    clipLineList(lineSets[i]->data(), mapped + lineFirst[i], lineSets[i]->size(), clipViewProj, behind);
            }
            for (size_t i = 0; i <= debugBatches; ++i)
                if (const std::vector<LineVertex>* verts = i < debugBatches ? &(*s.debugLines)[i].vertices : s.debugTris.get())
                {
                    std::memcpy(mapped + debugFirst[i], verts->data(), verts->size() * sizeof(LineVertex));
                    if (i < debugBatches)
                        clipLineList(verts->data(), mapped + debugFirst[i], verts->size(), clipViewProj, behind);
                }
            std::memcpy(static_cast<char*>(f.linesMapped) + lineBytes, s.overlayTris.data(), trisBytes);
        }

        VkCommandBuffer cmd = f.cmd;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        cullCmd = VK_NULL_HANDLE;
        const auto cpuStart = std::chrono::steady_clock::now();
        if (f.timestamps)
        {
            vkCmdResetQueryPool(cmd, f.timestamps, 0, 4);
            timestamp(cmd, f, 0);
            f.timed = true;
        }
        gpuProfile::Sections gpuFrame(cmd), gpuPass(cmd);
        LILA_GPU_SECTION(gpuFrame, "Viewport frame");
        LILA_GPU_SECTION(gpuPass, "Uploads");
        ++frameCounter;
        const bool shaderWire = meshWirePipeline != VK_NULL_HANDLE;
        const bool sameMeshes = !v.offscreen && v.keptShaderWire == shaderWire && (s.meshes.sameList(v.keptMeshes) || sameDraws(s.meshes.list(), v.keptMeshes.list()));
        if (!sameMeshes && !v.offscreen)
        {
            LILA_ZONE_N("Keep mesh draws");
            v.keptMeshes = s.meshes;
            v.keptShaderWire = shaderWire;
            v.keptMapsValid = false;
            v.keptContactsValid = false;
            v.keptDynamic.clear();
            for (size_t i = 0; i < s.meshes.size(); ++i)
                if (s.meshes[i].mesh && s.meshes[i].mesh->dynamic)
                    v.keptDynamic.push_back(uint32_t(i));
        }
        prepareTextures(cmd, f, s, v.offscreen ? nullptr : &v);
        if (frameUsesVt && frameTableAddress && f.feedback.begin(vk, cmd, int(v.extent.width), int(v.extent.height), frameCounter))
        {
            const std::array<uint32_t, vt::kTableHeaderWords> header = f.feedback.header();
            std::memcpy(f.table.mapped, header.data(), sizeof(header));
            vmaFlushAllocation(vk.allocator, f.table.alloc, 0, sizeof(header));
        }
        else if (f.feedback.active())
            f.feedback.destroy(vk);
        prepareMeshes(cmd, f, s, sameMeshes);
        prepareDynamicMeshes(cmd, f, s, sameMeshes ? &v.keptDynamic : nullptr);
        writeInstances(f, s, v.offscreen ? nullptr : &v.keptBatches, sameMeshes);
        writeLights(f, s);
        prepareRecords(v, s, sameMeshes);
        frameMeshDraws = frameMeshInstances = 0;

        glm::dmat4 viewProj = s.proj * s.view;
        glm::mat4 vpF(viewProj);
        cullPoints(cmd, f, s, vpF, v.extent);
        const bool msaa = v.color.image != VK_NULL_HANDLE;
        // Temporal antialiasing: the scene draws with a sub-pixel jitter and blends into its history, while lines and
        // overlays draw over the result with the plain projection. An external renderer's image has no jitter to resolve.
        const bool taa = (s.taa || s.accumulateOnly) && taaPipeline && v.history[0].image && v.taaSets[0][0] && !s.externalScene;
        const bool unchanged = v.renderingSerial == v.drawnSerial;
        v.drawnSerial = v.renderingSerial;
        const bool sameCamera = v.historyValid && s.view == v.prevView && s.proj == v.prevProj;
        float taaWeight = 1.0f;
        bool taaClip = false;
        const bool taaReset = !v.historyValid;
        if (!taa)
            v.accumulated = 0;
        else if (taaReset)
        {
            v.accumulated = 1;
            v.jitterIndex = 0;
        }
        else if (s.accumulateOnly || (unchanged && sameCamera && v.uploadEpoch == uploadEpoch))
            // Holding still: a plain average of every jittered frame since, converging on a supersampled image.
            taaWeight = 1.0f / float(++v.accumulated);
        else
        {
            taaWeight = 0.1f;
            taaClip = true;
            v.accumulated = 2;
        }
        v.uploadEpoch = uploadEpoch;
        v.wantsMore = taa && v.accumulated < uint32_t(std::max(s.accumulate, 1));
        glm::dmat4 sceneViewProj = viewProj;
        if (taa)
        {
            const glm::vec2 o = jitterOffset(v.jitterIndex);
            v.jitterIndex = v.jitterIndex % 1023 + 1;
            glm::dmat4 jitter(1.0);
            jitter[3][0] = 2.0 * o.x / double(v.extent.width);
            jitter[3][1] = 2.0 * o.y / double(v.extent.height);
            sceneViewProj = jitter * viewProj;
            if (s.lensRadius > 0.0f && s.focusDistance > 0.0f && s.proj[3][3] == 0.0)
            {
                // Look from a point of the aperture, sheared so the focus plane stays where it was: x' = x - l (1 - depth / focus).
                const glm::vec2 l = apertureSample(v.jitterIndex, s.apertureBlades, s.apertureRotation) * s.lensRadius;
                glm::dmat4 lens(1.0);
                lens[2][0] = -double(l.x) / double(s.focusDistance);
                lens[2][1] = -double(l.y) / double(s.focusDistance);
                lens[3][0] = -double(l.x);
                lens[3][1] = -double(l.y);
                sceneViewProj = jitter * s.proj * lens * s.view;
            }
        }
        const glm::mat4 vpScene(sceneViewProj);

        bool ao = s.ambientOcclusion && s.aoIntensity > 0.0f && prepassPipeline && gtaoPipeline && denoisePipeline && v.gtaoSet && v.denoiseSet &&
                  v.aoSampleSet && anyShaded(s) && !s.xray;
        const bool post = (s.kuwahara || s.outline) && postPipeline && composeSinglePipeline && v.postSet && v.postColor.view;
        // Fast AO comes from the main pass's own depth after the meshes; the post pass reads the prepass, so it keeps it.
        const bool aoFast = ao && s.aoFast && !post && aoApplyPipeline;
        f.aoTimed = ao && !aoFast;
        const bool bevel = vk.rayQuery && bevelPipeline && rtaoSetLayout && !s.xray && !s.externalScene && anyBevel(s);
        const bool rtShadow = !frameRtLights.empty() && rtaoSetLayout && !s.xray && !s.externalScene && anyShaded(s);
        const bool prepass = (ao && !aoFast) || (post && s.outline && prepassPipeline && anyShaded(s) && !s.xray) || bevel || rtShadow;
        timestamp(cmd, f, 1);
        LILA_GPU_SECTION(gpuPass, "Shadow maps");
        recordShadows(cmd, s);
        const auto aoStart = std::chrono::steady_clock::now();
        frameBevel = false;
        frameRtShadow = false;
        if (prepass)
        {
            LILA_GPU_SECTION(gpuPass, "Prepass and AO");
            recordAo(cmd, f, v, s, vpScene, ao && !aoFast);
            if (bevel)
                frameBevel = recordBevel(cmd, f, v, s, vpScene);
            if (rtShadow)
                frameRtShadow = recordRtShadow(cmd, f, v, s, vpScene);
            // The traced passes come with the AO set: unoccluded when AO itself is off.
            if ((frameBevel || frameRtShadow) && !(ao && !aoFast))
                clearAo(cmd, v);
        }
        // The mesh shader reads the traced shadow only when the pass ran; otherwise that light keeps no shadow this frame.
        if (frameLightAddress && !frameRtLights.empty())
        {
            auto* header = static_cast<GpuFrame*>(f.lights.mapped);
            for (size_t c = 0; c < 4; ++c)
                header->shadowing[int(c)] = frameRtShadow && c < frameRtLights.size() ? float(frameRtLights[c]) : 0.0f;
            vmaFlushAllocation(vk.allocator, f.lights.alloc, 0, sizeof(GpuFrame));
        }
        else if (post)
        {
            // Nothing wrote them this frame; the post set still needs them in their sampled layouts.
            barrier(cmd, v.aoDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
            barrier(cmd, v.aoNormal.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        v.timing.aoCpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - aoStart).count();
        timestamp(cmd, f, 2);
        LILA_GPU_SECTION(gpuPass, "Scene");

        constexpr auto colorStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        constexpr auto colorWrite = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        constexpr auto depthStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        constexpr auto depthAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        constexpr auto computeStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto fragmentStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto sampledRead = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier(cmd, v.images[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, colorStage, 0, colorStage,
                colorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        if (v.color.image)
            barrier(cmd, v.color.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, colorStage, colorWrite, colorStage,
                    colorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        barrier(cmd, v.depth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, depthStage, depthAccess, depthStage,
                depthAccess, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (post)
            barrier(cmd, v.postColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, colorStage | fragmentStage, 0,
                    colorStage, colorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        if (msaa)
        {
            barrier(cmd, v.sceneMs.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, colorStage, colorWrite, colorStage,
                    colorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            barrier(cmd, v.sceneDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, computeStage, 0,
                    depthStage | colorStage, depthAccess | colorWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        barrier(cmd, v.scene.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, computeStage | fragmentStage, 0, colorStage,
                colorWrite, VK_IMAGE_ASPECT_COLOR_BIT);

        // An external image goes under the scene in the frame pass, so the scene keeps clear where it draws nothing.
        const bool clearScene = s.transparentBackground || s.externalScene;
        const glm::vec3 bg = clearScene ? glm::vec3(0.0f) : sceneFromDisplay(srgbToLinear(s.background), s.gamma);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = msaa ? v.sceneMs.view : v.scene.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{bg.r, bg.g, bg.b, clearScene ? 0.0f : 1.0f}};
        if (msaa)
        {
            color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            color.resolveImageView = v.scene.view;
            color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        // The depth stays for the lines and overlays drawn after the compose pass; its first sample feeds the TAA.
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = v.depth.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {0.0f, 0}; // reversed-Z
        if (msaa)
        {
            depth.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            depth.resolveImageView = v.sceneDepth.view;
            depth.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        }
        // Fast AO ends the pass after the meshes and picks it up again with these once the AO is applied.
        VkRenderingAttachmentInfo resumeColor = color, resumeDepth = depth;
        if (aoFast)
        {
            resumeColor.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            resumeDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color.resolveMode = VK_RESOLVE_MODE_NONE;
            color.resolveImageView = VK_NULL_HANDLE;
            barrier(cmd, v.aoDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, computeStage | fragmentStage, 0,
                    depthStage | colorStage, depthAccess | colorWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
            if (msaa)
                depth.resolveImageView = v.aoDepth.view;
            else
                depth.imageView = resumeDepth.imageView = v.aoDepth.view;
        }

        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, v.extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        ri.pDepthAttachment = &depth;
        // Many sections record on worker threads: the pass suspends around them, see drawMeshes.
        const bool sceneChunks = !s.xray && meshPipeline && !s.meshes.empty() && chunked();
        if (sceneChunks)
            ri.flags = VK_RENDERING_SUSPENDING_BIT;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport viewport{0, 0, float(v.extent.width), float(v.extent.height), 0, 1};
        VkRect2D scissor{{0, 0}, v.extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        if (backgroundPipeline && envLoaded && s.envBackground && !s.transparentBackground)
        {
            MeshPush p{};
            p.viewProj = glm::mat4(glm::inverse(sceneViewProj));
            p.base = glm::vec4(std::clamp(s.envBlur, 0.0f, 1.0f) * envMaxLod, std::clamp(s.envSaturation, 0.0f, 1.0f),
                               s.envBackgroundExposure, 0.0f);
            p.params = glm::vec4(s.exposure, s.zUp ? 1.0f : 0.0f, 0.0f, 1.0f);
            p.env = envParams(s);
            p.gamma = s.gamma;
            p.lights = frameLightAddress;
            p.lightCount = frameLightCount;
            p.eye = glm::vec4(glm::vec3(s.eye), 1.0f);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 0, 1, &envSet, 0, nullptr);
            vkCmdPushConstants(cmd, meshLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        if (s.externalScene)
            drawExternal(cmd, s, externalDepthPipeline);
        // Grid next: it writes no depth, so anything on the ground plane draws over it. An external image draws it in the
        // frame pass instead, against that image's depth.
        if (gridPipeline && s.grid && !s.externalScene)
        {
            GridPush gp;
            gp.viewProj = vpScene;
            gp.invViewProj = glm::mat4(glm::inverse(sceneViewProj));
            gp.eye = glm::vec4(glm::vec3(s.eye), 1.0f);
            gp.params = glm::vec4(s.gridSpacing, float(s.gridMajorEvery), s.zUp ? 1.0f : 0.0f, s.gamma);
            gp.color = glm::vec4(s.gridColor, 1.0f);
            gp.axisX = glm::vec4(s.gridAxisX, 1.0f);
            gp.axisZ = glm::vec4(s.gridAxisZ, 1.0f);
            gp.origin = glm::vec4(glm::vec3(s.gridOrigin), 0.0f);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gridPipeline);
            vkCmdPushConstants(cmd, gridLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GridPush), &gp);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        drawContactShadows(cmd, s, vpScene, v.offscreen ? nullptr : &v);
        drawHeightfields(cmd, s, vpScene);
        drawMeshes(cmd, s, vpScene, (ao && !aoFast) || frameBevel || frameRtShadow ? v.aoSampleSet : whiteAoSet, v.extent.height,
                   sceneChunks ? &f : nullptr, &ri);
        if (aoFast)
        {
            vkCmdEndRendering(cmd);
            recordFastAo(cmd, v, s);
            if (!msaa)
                barrier(cmd, v.aoDepth.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        computeStage, 0, depthStage, depthAccess, VK_IMAGE_ASPECT_DEPTH_BIT);
            ri.pColorAttachments = &resumeColor;
            ri.pDepthAttachment = &resumeDepth;
            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            applyAo(cmd, v);
        }
        vkCmdEndRendering(cmd);

        // The scene becomes the compose pass's input, through the temporal history when that is on.
        LILA_GPU_SECTION(gpuPass, "Temporal AA and post");
        const bool depthIsAo = !msaa && aoFast;
        const VkImage frameDepth = depthIsAo ? v.aoDepth.image : v.depth.image;
        const VkImageView frameDepthView = depthIsAo ? v.aoDepth.view : v.depth.view;
        barrier(cmd, v.scene.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorStage, colorWrite,
                computeStage | fragmentStage, sampledRead, VK_IMAGE_ASPECT_COLOR_BIT);
        VkDescriptorSet composeSet = v.composeSets[0];
        int composeInput = 0;
        if (taa)
        {
            const VkImage taaDepth = msaa ? v.sceneDepth.image : frameDepth;
            const int out = v.historyIndex ^ 1;
            barrier(cmd, taaDepth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, depthStage | colorStage,
                    depthAccess | colorWrite, computeStage, sampledRead, VK_IMAGE_ASPECT_DEPTH_BIT);
            barrier(cmd, v.history[out].image, v.historyValid ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    computeStage | fragmentStage, sampledRead, computeStage, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            if (!v.historyValid)
                barrier(cmd, v.history[out ^ 1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, computeStage, 0, computeStage, sampledRead,
                        VK_IMAGE_ASPECT_COLOR_BIT);
            TaaPush tp;
            tp.invViewProj = glm::mat4(glm::inverse(viewProj));
            tp.prevViewProj = glm::mat4(v.prevViewProj);
            tp.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
            tp.params = glm::vec4(taaWeight, taaClip ? 1.0f : 0.0f, taaReset ? 1.0f : 0.0f, s.accumulateOnly ? 1.0f : 0.0f);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, taaPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, taaLayout, 0, 1, &v.taaSets[out][depthIsAo ? 1 : 0], 0, nullptr);
            vkCmdPushConstants(cmd, taaLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tp), &tp);
            vkCmdDispatch(cmd, (v.extent.width + 7) / 8, (v.extent.height + 7) / 8, 1);
            barrier(cmd, v.history[out].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, computeStage, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    fragmentStage | computeStage, sampledRead, VK_IMAGE_ASPECT_COLOR_BIT);
            if (taaDepth == frameDepth)
                barrier(cmd, frameDepth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, computeStage, 0,
                        depthStage, depthAccess, VK_IMAGE_ASPECT_DEPTH_BIT);
            v.historyIndex = out;
            v.historyValid = true;
            composeSet = v.composeSets[1 + out];
            composeInput = 1 + out;
        }
        else
            v.historyValid = false;
        if (!taa || msaa)
            barrier(cmd, frameDepth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, depthStage, depthAccess,
                    depthStage, depthAccess, VK_IMAGE_ASPECT_DEPTH_BIT);
        v.prevViewProj = viewProj;
        v.prevView = s.view;
        v.prevProj = s.proj;
        const PostResult postResult = recordPostStack(cmd, f, v, s, composeInput);

        // The frame: the composed scene, then lines and overlays depth tested against the scene's depth.
        LILA_GPU_SECTION(gpuPass, "Compose and overlays");
        VkRenderingAttachmentInfo frameColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        frameColor.imageView = msaa ? v.color.view : v.views[imageIndex];
        frameColor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // The composed scene covers every pixel, but an external image leaves those its renderer drew nothing in.
        frameColor.loadOp = s.externalScene ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        const glm::vec3 frameBg = srgbToLinear(s.background);
        frameColor.clearValue.color = {{frameBg.r, frameBg.g, frameBg.b, 1.0f}};
        frameColor.storeOp = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        if (msaa)
        {
            frameColor.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            frameColor.resolveImageView = v.views[imageIndex];
            frameColor.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        VkRenderingAttachmentInfo overlayDepth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        overlayDepth.imageView = frameDepthView;
        overlayDepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        overlayDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        overlayDepth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ri.pColorAttachments = &frameColor;
        ri.pDepthAttachment = &overlayDepth;
        const ComposePush composePush = composeParams(v, s, postResult);
        auto drawCompose = [&](VkPipeline pipeline) {
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composeLayout, 0, 1, &composeSet, 0, nullptr);
            vkCmdPushConstants(cmd, composeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(composePush), &composePush);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        };
        if (post)
        {
            // The painterly filter and outlines read the composed scene, single-sampled, and write the frame.
            VkRenderingAttachmentInfo postTarget{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            postTarget.imageView = v.postColor.view;
            postTarget.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            postTarget.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            postTarget.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo pri = ri;
            pri.pColorAttachments = &postTarget;
            pri.pDepthAttachment = nullptr;
            vkCmdBeginRendering(cmd, &pri);
            drawCompose(composeSinglePipeline);
            vkCmdEndRendering(cmd);
            barrier(cmd, v.postColor.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colorStage, colorWrite,
                    fragmentStage, sampledRead, VK_IMAGE_ASPECT_COLOR_BIT);
            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            recordPost(cmd, v, s);
        }
        else
        {
            vkCmdBeginRendering(cmd, &ri);
            if (s.externalScene)
                drawUnderScene(cmd, s, vpF);
            drawCompose(s.externalScene && composeOverPipeline ? composeOverPipeline : composePipeline);
        }
        drawOverScene(cmd, s, vpF);

        if (linesPipeline && !s.lines.empty())
        {
            const LinesPush lp{vpF};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linesPipeline);
            vkCmdPushConstants(cmd, linesLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lp), &lp);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
            vkCmdDraw(cmd, uint32_t(s.lines.size()), 1, 0, 0);
        }

        // Selected edges draw between the other on-top lines and tris3d, so hover ribbons stay over them.
        struct Biased
        {
            VkPipeline pipeline;
            int set;
            float width;
        };
        const Biased biased[] = {{linesOnTopPipeline, 9, s.curveHaloWidth},     {linesOnTopPipeline, 10, s.curveWidth},
                                 {linesOnTopPipeline, 11, s.curveHaloBoldWidth}, {linesOnTopPipeline, 12, s.curveBoldWidth},
                                 {linesOnTopPipeline, 1, 1.0f}, {linesOnTopPipeline, 8, s.highlightEdgeWidth}, {linesOnTopPipeline, 6, s.seamLineWidth},
                                 {linesOnTopPipeline, 7, s.symmetryLineWidth}, {linesOnTopPipeline, 5, s.selectedEdgeWidth},
                                 {tris3dPipeline, 2, 0.0f}};
        drawPoints(cmd, f, s, vpF, v.extent.height);
        for (const Biased& b : biased)
        {
            const std::vector<LineVertex>& verts = *lineSets[b.set];
            if (!b.pipeline || verts.empty())
                continue;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, b.pipeline);
            if (b.width > 0.0f)
                vkCmdSetLineWidth(cmd, lineWidth(b.width));
            const LinesPush lp{vpF, depthPull(s.proj, v.extent.height, kLinePullPixels + 0.5 * lineWidth(std::max(b.width, 1.0f)))};
            vkCmdPushConstants(cmd, linesLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lp), &lp);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
            vkCmdDraw(cmd, uint32_t(verts.size()), 1, lineFirst[b.set], 0);
        }

        // Debug drawing: depth-tested lines and triangles here, the on-top lines over the x-ray joints.
        auto drawDebugLines = [&](bool onTop) {
            for (size_t i = 0; i < debugBatches; ++i)
            {
                const LineBatch& batch = (*s.debugLines)[i];
                VkPipeline pipeline = onTop ? linesOverPipeline : linesOnTopPipeline;
                if (batch.onTop != onTop || batch.vertices.empty() || !pipeline)
                    continue;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdSetLineWidth(cmd, lineWidth(batch.width));
                const LinesPush lp{vpF, onTop ? glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)
                                              : depthPull(s.proj, v.extent.height, kLinePullPixels + 0.5 * lineWidth(batch.width))};
                vkCmdPushConstants(cmd, linesLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lp), &lp);
                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
                vkCmdDraw(cmd, uint32_t(batch.vertices.size()), 1, debugFirst[i], 0);
            }
        };
        if (s.debugTris && !s.debugTris->empty() && tris3dPipeline)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tris3dPipeline);
            const LinesPush lp{vpF, depthPull(s.proj, v.extent.height, kLinePullPixels)};
            vkCmdPushConstants(cmd, linesLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lp), &lp);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
            vkCmdDraw(cmd, uint32_t(s.debugTris->size()), 1, debugFirst[debugBatches], 0);
        }
        drawDebugLines(false);

        if (overlayPipeline && !s.linesXray.empty())
        {
            const LinesPush lp{vpF};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline);
            vkCmdPushConstants(cmd, linesLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lp), &lp);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
            vkCmdDraw(cmd, uint32_t(s.linesXray.size()), 1, lineFirst[4], 0);
        }
        drawDebugLines(true);

        if (overlayPipeline && !s.overlay.empty())
        {
            const LinesPush lp{glm::mat4(1.0f)};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline);
            vkCmdPushConstants(cmd, linesLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(lp), &lp);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
            vkCmdDraw(cmd, uint32_t(s.overlay.size()), 1, lineFirst[3], 0);
        }

        if (overlayTrisPipeline && !s.overlayTris.empty())
        {
            glm::vec2 invSize(1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayTrisPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayTrisLayout, 0, 1, &atlasSet, 0, nullptr);
            vkCmdPushConstants(cmd, overlayTrisLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::vec2), &invSize);
            VkDeviceSize offset = lineBytes;
            vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines, &offset);
            vkCmdDraw(cmd, uint32_t(s.overlayTris.size()), 1, 0, 0);
        }

        vkCmdEndRendering(cmd);
        if (f.feedback.active())
            f.feedback.end(vk, cmd);
        gpuPass.end();
        gpuFrame.end();
        timestamp(cmd, f, 3);
        v.timing.cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();
        v.timing.meshDrawCalls = frameMeshDraws;
        v.timing.meshInstances = frameMeshInstances;
        v.timing.meshUploads = meshUploads;
        if (v.offscreen)
            return finishOffscreen(v, f, cmd);
        barrier(cmd, v.images[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, colorStage, colorWrite,
                VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0, VK_IMAGE_ASPECT_COLOR_BIT);
        vkEndCommandBuffer(cmd);

        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = f.acquired;
        wait.stageMask = colorStage;
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = v.rendered[imageIndex];
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbi.commandBuffer = cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.waitSemaphoreInfoCount = 1;
        si.pWaitSemaphoreInfos = &wait;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &signal;
        checkQueue(vkQueueSubmit2(vk.queue, 1, &si, f.fence), "submit");

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &v.rendered[imageIndex];
        pi.swapchainCount = 1;
        pi.pSwapchains = &v.swapchain;
        pi.pImageIndices = &imageIndex;
        {
            LILA_ZONE_N("Present");
            r = vkQueuePresentKHR(vk.queue, &pi);
        }
        checkQueue(r, "present");
        v.frameIndex = (v.frameIndex + 1) % kFramesInFlight;
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        {
            v.needsRecreate = true;
            return true;
        }
        return false;
    }

    /** A 3D LUT as an RGBA16F image, filled by a copy recorded into cmd; the staging buffer goes with the frame. */
    Image uploadLut(VkCommandBuffer cmd, Frame& f, int size, const float* rgb)
    {
        Image img;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_3D;
        ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = {uint32_t(size), uint32_t(size), uint32_t(size)};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(vk.allocator, &ici, &aci, &img.image, &img.alloc, nullptr) != VK_SUCCESS)
            return {};
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = ici.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(vk.device, &vci, nullptr, &img.view);

        const size_t count = size_t(size) * size * size;
        std::vector<uint16_t> half(count * 4);
        for (size_t i = 0; i < count; ++i)
        {
            for (int c = 0; c < 3; ++c)
                half[i * 4 + c] = glm::packHalf1x16(rgb[i * 3 + c]);
            half[i * 4 + 3] = glm::packHalf1x16(1.0f);
        }
        AllocatedBuffer staging = createBuffer(half.data(), half.size() * sizeof(uint16_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        f.staging.push_back(staging);
        barrier(cmd, img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = ici.extent;
        vkCmdCopyBufferToImage(cmd, staging.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);
        return img;
    }

    /** The view of a frame's LUT, uploaded the first time; the identity when it has none. */
    VkImageView lutView(VkCommandBuffer cmd, Frame& f, const std::shared_ptr<const Lut3D>& lut)
    {
        if (!identityLut.image)
        {
            float id[2 * 2 * 2 * 3];
            for (int b = 0, i = 0; b < 2; ++b)
                for (int g = 0; g < 2; ++g)
                    for (int r = 0; r < 2; ++r, ++i)
                    {
                        id[i * 3 + 0] = float(r);
                        id[i * 3 + 1] = float(g);
                        id[i * 3 + 2] = float(b);
                    }
            identityLut = uploadLut(cmd, f, 2, id);
        }
        if (!lut || lut->size < 2 || lut->rgb.size() < size_t(lut->size) * lut->size * lut->size * 3)
            return identityLut.view;
        auto it = luts.find(lut->key);
        if (it == luts.end())
            it = luts.emplace(lut->key, uploadLut(cmd, f, lut->size, lut->rgb.data())).first;
        return it->second.view ? it->second.view : identityLut.view;
    }

    struct PostResult
    {
        bool bloom = false;
        bool autoExposure = false;
        bool lut = false;
    };

    /**
     * The compute half of the post stack, between the TAA and the compose pass: auto exposure from a histogram of the
     * scene, the bloom chain, and the LUT the compose pass samples. input picks the scene (0) or a history (1, 2).
     */
    PostResult recordPostStack(VkCommandBuffer cmd, Frame& f, Viewport& v, const FrameState& s, int input)
    {
        PostResult r;
        if (!v.bloomDown.image || !v.composeSets[input])
            return r;
        constexpr auto compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto read = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        constexpr auto write = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        constexpr auto general = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto color = VK_IMAGE_ASPECT_COLOR_BIT;
        if (!v.postImagesReady)
        {
            barrier(cmd, v.bloomDown.image, VK_IMAGE_LAYOUT_UNDEFINED, general, compute, 0, compute, read | write, color, v.bloomLevels);
            barrier(cmd, v.bloomUp.image, VK_IMAGE_LAYOUT_UNDEFINED, general, compute, 0, compute, read | write, color, v.bloomLevels);
            barrier(cmd, v.exposureState.image, VK_IMAGE_LAYOUT_UNDEFINED, general, compute, 0, compute | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                    read | write | VK_ACCESS_2_TRANSFER_WRITE_BIT, color);
            v.postImagesReady = true;
        }

        // The LUT binding is part of the compose sets, so changing it waits for the frame still reading them.
        const VkImageView lut = lutView(cmd, f, s.post.effects ? s.post.lut : nullptr);
        if (uint64_t(lut) != v.lutKey)
        {
            Frame& other = v.frames[(v.frameIndex + 1) % kFramesInFlight];
            vkWaitForFences(vk.device, 1, &other.fence, VK_TRUE, UINT64_MAX);
            for (VkDescriptorSet set : v.composeSets)
                writeImage(set, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lut, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, linearSampler);
            v.lutKey = uint64_t(lut);
        }
        r.lut = lut != identityLut.view && s.post.lutIntensity > 0.0f;

        if (s.post.autoExposure && histogramPipeline && averagePipeline)
        {
            const VkClearColorValue zero{};
            const VkImageSubresourceRange range{color, 0, 1, 0, 1};
            barrier(cmd, v.histogram.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, compute, 0,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, color);
            vkCmdClearColorImage(cmd, v.histogram.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            barrier(cmd, v.histogram.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, general, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, compute, read | write, color);
            const auto now = std::chrono::steady_clock::now();
            float blend = 1.0f;
            if (!v.exposureValid)
            {
                vkCmdClearColorImage(cmd, v.exposureState.image, general, &zero, 1, &range);
                barrier(cmd, v.exposureState.image, general, general, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, compute,
                        read | write, color);
            }
            else if (s.post.exposureSpeed > 0.0f && !v.offscreen)
            {
                const float dt = std::min(std::chrono::duration<float>(now - v.lastExposure).count(), 0.25f);
                blend = 1.0f - std::exp(-dt * s.post.exposureSpeed);
            }
            v.lastExposure = now;
            v.exposureValid = true;
            ExposurePush ep;
            ep.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
            ep.range = glm::vec4(-10.0f, 8.0f, s.post.exposureMin, s.post.exposureMax);
            ep.adapt = glm::vec4(blend, s.post.exposureCompensation, 0.5f, 0.95f);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposureLayout, 0, 1, &v.exposureSets[input], 0, nullptr);
            vkCmdPushConstants(cmd, exposureLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ep), &ep);
            vkCmdDispatch(cmd, (v.extent.width / 4 + 15) / 16 + 1, (v.extent.height / 4 + 15) / 16 + 1, 1);
            barrier(cmd, v.histogram.image, general, general, compute, write, compute, read, color);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, averagePipeline);
            vkCmdDispatch(cmd, 1, 1, 1);
            barrier(cmd, v.exposureState.image, general, general, compute, write, fragment | compute, read, color);
            r.autoExposure = true;
        }
        else
            v.exposureValid = false;

        if (s.post.effects && s.post.bloom > 0.0f && v.bloomLevels > 1 && bloomDownPipeline && bloomUpPipeline)
        {
            auto levelSize = [&](uint32_t m) {
                return glm::uvec2(std::max(v.extent.width / 2 >> m, 1u), std::max(v.extent.height / 2 >> m, 1u));
            };
            auto dispatch = [&](VkPipeline pipeline, VkDescriptorSet set, glm::uvec2 size, glm::vec2 srcTexel, float first, float radius) {
                BloomPush bp;
                bp.size = glm::vec4(float(size.x), float(size.y), srcTexel.x, srcTexel.y);
                bp.params = glm::vec4(first, radius, 0.0f, 0.0f);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomLayout, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(cmd, bloomLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bp), &bp);
                vkCmdDispatch(cmd, (size.x + 7) / 8, (size.y + 7) / 8, 1);
            };
            barrier(cmd, v.bloomDown.image, general, general, fragment | compute, read, compute, write, color, v.bloomLevels);
            barrier(cmd, v.bloomUp.image, general, general, fragment | compute, read, compute, write, color, v.bloomLevels);
            for (uint32_t mip = 0; mip < v.bloomLevels; ++mip)
            {
                const glm::uvec2 src = mip == 0 ? glm::uvec2(v.extent.width, v.extent.height) : levelSize(mip - 1);
                dispatch(bloomDownPipeline, mip == 0 ? v.bloomSets[input] : v.bloomSets[3 + mip - 1], levelSize(mip), 1.0f / glm::vec2(src),
                         mip == 0 ? 1.0f : 0.0f, 1.0f);
                barrier(cmd, v.bloomDown.image, general, general, compute, write, compute, read, color, v.bloomLevels);
            }
            for (int mip = int(v.bloomLevels) - 2; mip >= 0; --mip)
            {
                dispatch(bloomUpPipeline, v.bloomSets[3 + kMaxBloomLevels - 1 + mip], levelSize(mip), 1.0f / glm::vec2(levelSize(mip + 1)), 0.0f,
                         std::max(s.post.bloomRadius, 0.1f));
                barrier(cmd, v.bloomUp.image, general, general, compute, write, compute | fragment, read, color, v.bloomLevels);
            }
            r.bloom = true;
        }
        return r;
    }

    /** Push constants for the compose pass from the frame's post settings and what recordPostStack produced. */
    ComposePush composeParams(const Viewport& v, const FrameState& s, const PostResult& r) const
    {
        const PostSettings& p = s.post;
        ComposePush c;
        c.params = glm::vec4(s.gamma, p.dither ? 1.0f : 0.0f, 1.0f, float(int(p.tonemapper)));
        c.grade = glm::vec4(p.contrast, p.saturation, p.effects ? 1.0f : 0.0f, float(p.agxLook));
        c.white = glm::vec4(p.whiteBalance, r.lut ? std::clamp(p.lutIntensity, 0.0f, 1.0f) : 0.0f);
        c.lift = glm::vec4(p.lift, std::max(p.vignette, 0.0f));
        c.gamma = glm::vec4(p.gamma, std::max(p.grain, 0.0f));
        c.gain = glm::vec4(p.gain, float(frameCounter % 4096));
        c.lens = glm::vec4(std::max(p.chromaticAberration, 0.0f), p.distortion, std::max(p.sharpen, 0.0f), r.bloom ? std::clamp(p.bloom, 0.0f, 1.0f) : 0.0f);
        c.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        c.misc = glm::vec4(r.autoExposure ? 1.0f : 0.0f, 1.0f / float(std::max(v.bloomLevels, 1u)), 0.0f, 0.0f);
        return c;
    }

    /** Filters postColor (the composed scene) over every pixel of the frame pass; lines and overlays then draw on top unfiltered. */
    void recordPost(VkCommandBuffer cmd, Viewport& v, const FrameState& s)
    {
        LILA_ZONE();

        PostPush p;
        p.size = glm::vec4(float(v.extent.width), float(v.extent.height), 1.0f / float(v.extent.width), 1.0f / float(v.extent.height));
        p.params = glm::vec4(s.kuwahara ? 1.0f : 0.0f, std::clamp(s.kuwaharaRadius, 1.0f, 12.0f), s.outline && !s.xray ? 1.0f : 0.0f,
                             std::clamp(s.outlineWidth, 0.5f, 8.0f));
        p.outline = glm::vec4(s.outlineColor, 1.0f);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postLayout, 0, 1, &v.postSet, 0, nullptr);
        vkCmdPushConstants(cmd, postLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    /** Copies the offscreen target into the readback buffer and waits for the GPU. Never asks for a retry. */
    bool finishOffscreen(Viewport& v, Frame& f, VkCommandBuffer cmd)
    {
        if (!v.readbackWanted)
        {
            // An intermediate frame of an accumulation: the next one waits for it through the frame's fence.
            vkEndCommandBuffer(cmd);
            VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            cbi.commandBuffer = cmd;
            VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            si.commandBufferInfoCount = 1;
            si.pCommandBufferInfos = &cbi;
            checkQueue(vkQueueSubmit2(vk.queue, 1, &si, f.fence), "submit");
            v.frameIndex = (v.frameIndex + 1) % kFramesInFlight;
            return false;
        }
        if (v.hdrWanted && v.hdrReadback.buffer)
        {
            // The light before tonemapping: the temporal history when it ran, else the resolved scene.
            const bool history = v.historyValid;
            const VkImage src = history ? v.history[v.historyIndex].image : v.scene.image;
            const VkImageLayout layout = history ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier(cmd, src, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            VkBufferImageCopy hr{};
            hr.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            hr.imageExtent = {v.extent.width, v.extent.height, 1};
            vkCmdCopyImageToBuffer(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, v.hdrReadback.buffer, 1, &hr);
            barrier(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        barrier(cmd, v.images[0], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {v.extent.width, v.extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, v.images[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, v.readback.buffer, 1, &region);
        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbi.commandBuffer = cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        checkQueue(vkQueueSubmit2(vk.queue, 1, &si, f.fence), "submit");
        vkWaitForFences(vk.device, 1, &f.fence, VK_TRUE, UINT64_MAX);
        f.feedback.collect();
        readTimings(v, f);
        vmaInvalidateAllocation(vk.allocator, v.readback.alloc, 0, VK_WHOLE_SIZE);
        if (v.hdrReadback.alloc)
            vmaInvalidateAllocation(vk.allocator, v.hdrReadback.alloc, 0, VK_WHOLE_SIZE);
        v.frameIndex = (v.frameIndex + 1) % kFramesInFlight;
        return false;
    }

    /** Render thread: one capture on a throwaway offscreen viewport. */
    void runOffscreen(OffscreenJob& job)
    {
        LILA_ZONE();

        Viewport v;
        v.offscreen = true;
        v.width = job.width;
        v.height = job.height;
        initFrames(v);
        v.needsRecreate = true;
        // Repeats reuse the targets, so their timings are steady-state frames like a live viewport's. Every frame draws the same
        // state, so with temporal antialiasing on they average into one supersampled image.
        v.renderingSerial = 1;
        v.hdrWanted = job.hdr;
        std::vector<Renderer::FrameStats> runs;
        if (job.frames.empty())
            job.frames.push_back(std::move(job.state));
        const int each = job.frames.size() > 1 ? std::max(job.samples, 1) : std::max({job.repeat, job.samples, 1});
        const size_t total = job.frames.size() * size_t(each);
        size_t drawn = 0;
        for (FrameState& frame : job.frames)
            for (int i = 0; i < each; ++i)
            {
                // Timing runs read every frame back, so each one's timestamps are collected.
                v.readbackWanted = ++drawn == total || job.repeat > 1;
                FrameState s = frame;
                render(v, s);
                runs.push_back(v.timing);
            }
        if (v.hdrMapped)
        {
            const size_t count = size_t(v.extent.width) * v.extent.height;
            job.hdrPixels.resize(count * 4);
            const auto* half = static_cast<const uint16_t*>(v.hdrMapped);
            tbb::parallel_for(size_t(0), count, [&](size_t i) {
                glm::vec4 c(glm::unpackHalf1x16(half[i * 4]), glm::unpackHalf1x16(half[i * 4 + 1]), glm::unpackHalf1x16(half[i * 4 + 2]),
                            glm::unpackHalf1x16(half[i * 4 + 3]));
                // Undo the scene target's compression (scene_color.slang), and the coverage premultiplying it.
                const float m = std::max({c.r, c.g, c.b});
                glm::vec3 rgb = glm::vec3(c) / std::max(1.0f - m, 1.0f / 65504.0f);
                if (c.a > 1e-4f && c.a < 1.0f)
                    rgb /= c.a;
                job.hdrPixels[i * 4] = rgb.r;
                job.hdrPixels[i * 4 + 1] = rgb.g;
                job.hdrPixels[i * 4 + 2] = rgb.b;
                job.hdrPixels[i * 4 + 3] = std::clamp(c.a, 0.0f, 1.0f);
            });
        }
        std::nth_element(runs.begin(), runs.begin() + runs.size() / 2, runs.end(),
                         [](const Renderer::FrameStats& a, const Renderer::FrameStats& b) { return a.gpuMs < b.gpuMs; });
        job.timing = runs[runs.size() / 2];
        job.timing.frames = runs.size();
        if (v.readbackMapped)
        {
            const auto* src = static_cast<const uint8_t*>(v.readbackMapped);
            job.pixels.assign(src, src + size_t(v.extent.width) * v.extent.height * 4);
            // The pipelines target whatever the swapchain picked; callers always get BGRA.
            if (v.format == VK_FORMAT_R8G8B8A8_SRGB || v.format == VK_FORMAT_R8G8B8A8_UNORM)
                for (size_t i = 0; i + 3 < job.pixels.size(); i += 4)
                    std::swap(job.pixels[i], job.pixels[i + 2]);
            job.ok = true;
        }
        else
            job.error = "offscreen target could not be created";
        destroyViewport(v);
    }

    glm::vec4 envParams(const FrameState& s) const
    {
        float r = glm::radians(s.envRotation);
        return {std::cos(r), std::sin(r), s.envIntensity, envMaxLod};
    }

    /**
     * Makes s an external renderer's frame: no shaded meshes, AO, post, contact shadows or backdrop, and the wires and
     * highlights that remain hide behind the external depth.
     */
    static void handOverScene(FrameState& s)
    {
        std::vector<MeshDraw>& meshes = s.meshes.edit();
        for (MeshDraw& d : meshes)
        {
            d.flags &= ~uint32_t(DrawShaded);
            if (drawsWire(d))
                d.flags |= DrawWireDepth;
        }
        std::erase_if(meshes, [](const MeshDraw& d) { return !drawsWire(d); });
        s.envBackground = false;
        s.ambientOcclusion = false;
        s.kuwahara = s.outline = false;
        s.groundShadow = 0.0f;
    }

    /**
     * An external renderer's image in the frame pass, under the composed scene (its wires and highlights): reference
     * images behind it, then the image, then the grid against its depth. Display space, so its colours come through exactly.
     */
    void drawUnderScene(VkCommandBuffer cmd, const FrameState& s, const glm::mat4& viewProj)
    {
        drawImagePlanes(cmd, s, viewProj, ImagePlaneDraw::Depth::Behind, false);
        drawExternal(cmd, s, externalPipeline);
        if (!gridFramePipeline || !s.grid || !s.externalDepth)
            return;
        GridPush gp;
        gp.viewProj = viewProj;
        gp.invViewProj = glm::inverse(viewProj);
        gp.eye = glm::vec4(glm::vec3(s.eye), 1.0f);
        gp.params = glm::vec4(s.gridSpacing, float(s.gridMajorEvery), s.zUp ? 1.0f : 0.0f, -1.0f);
        gp.color = glm::vec4(s.gridColor, 1.0f);
        gp.axisX = glm::vec4(s.gridAxisX, 1.0f);
        gp.axisZ = glm::vec4(s.gridAxisZ, 1.0f);
        gp.origin = glm::vec4(glm::vec3(s.gridOrigin), 0.0f);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gridFramePipeline);
        vkCmdPushConstants(cmd, gridLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GridPush), &gp);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    /** Reference images over the composed scene; behind ones only where the scene drew nothing. */
    void drawOverScene(VkCommandBuffer cmd, const FrameState& s, const glm::mat4& viewProj)
    {
        if (!s.externalScene)
            drawImagePlanes(cmd, s, viewProj, ImagePlaneDraw::Depth::Behind, false);
        // Opaque planes in the scene write depth; see-through ones blend over them afterwards.
        drawImagePlanes(cmd, s, viewProj, ImagePlaneDraw::Depth::Scene, false);
        drawImagePlanes(cmd, s, viewProj, ImagePlaneDraw::Depth::Scene, true);
        drawImagePlanes(cmd, s, viewProj, ImagePlaneDraw::Depth::Front, false);
    }

    /** The external renderer's image over the cleared background, writing its depth where it has some. */
    void drawExternal(VkCommandBuffer cmd, const FrameState& s, VkPipeline pipeline)
    {
        if (!pipeline || !s.externalColor || textureKey(s.externalColor, 0) == 0)
            return;
        LILA_ZONE();

        Material mat;
        mat.baseColorMap = s.externalColor;
        const bool depth = s.externalDepth && textureKey(s.externalDepth, 0) != 0;
        if (depth)
            mat.normalMap = s.externalDepth;
        VkDescriptorSet set = materialSet(&mat);
        if (!set)
            return;
        ExternalPush p;
        const float w = float(std::max(s.externalColor->width, 1)), h = float(std::max(s.externalColor->height, 1));
        p.size = glm::vec4(w, h, 1.0f / w, 1.0f / h);
        p.projZ = glm::vec4(float(s.proj[2][2]), float(s.proj[3][2]), float(s.proj[2][3]), float(s.proj[3][3]));
        p.params = glm::vec4(!depth ? 0.0f : s.externalDepthUnproject ? 2.0f : 1.0f, s.externalBottomUp ? 1.0f : 0.0f,
                             s.externalPremultiplied ? 1.0f : 0.0f, s.externalExposure);
        p.unproject = s.externalDepthUnproject.value_or(glm::vec4(0.0f));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, externalLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, externalLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    /** One depth group of image planes, in frame order. In the scene group, blended picks the see-through ones. */
    void drawImagePlanes(VkCommandBuffer cmd, const FrameState& s, const glm::mat4& viewProj, ImagePlaneDraw::Depth depth, bool blended)
    {
        if (s.imagePlanes.empty())
            return;
        LILA_ZONE();

        bool scene = depth == ImagePlaneDraw::Depth::Scene;
        const bool behind = depth == ImagePlaneDraw::Depth::Behind;
        VkPipeline pipeline = behind ? imageBehindPipeline : !scene ? imageOverPipeline : blended ? imageBlendPipeline : imageDepthPipeline;
        if (!pipeline)
            return;
        bool bound = false;
        for (const ImagePlaneDraw& p : s.imagePlanes)
        {
            if (p.depth != depth || !p.image || (scene && (p.opacity < 0.999f) != blended) || textureKey(p.image, 0) == 0)
                continue;
            Material material;
            material.baseColorMap = p.image;
            VkDescriptorSet set = materialSet(&material);
            if (!set)
                continue;
            if (!bound)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                bound = true;
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, imagePlaneLayout, 0, 1, &set, 0, nullptr);
            ImagePlanePush push;
            push.viewProj = viewProj;
            for (int i = 0; i < 4; ++i)
                push.corners[i] = glm::vec4(p.corners[i], p.screen ? 0.0f : 1.0f);
            const glm::vec2 rect[4] = {{p.uv.x, p.uv.y}, {p.uv.z, p.uv.y}, {p.uv.z, p.uv.w}, {p.uv.x, p.uv.w}};
            const glm::vec2* uv = p.cornerUvs ? p.cornerUv : rect;
            push.uv[0] = glm::vec4(uv[0], uv[1]);
            push.uv[1] = glm::vec4(uv[2], uv[3]);
            push.params = glm::vec4(std::clamp(p.opacity, 0.0f, 1.0f), p.brightness, p.contrast, p.grayscale ? 1.0f : 0.0f);
            push.texel = glm::vec4(0.5f / float(std::max(p.image->width, 1)), 0.5f / float(std::max(p.image->height, 1)), 0.0f, behind ? 1.0f : 0.0f);
            push.outline = glm::vec4(p.outlineColor, p.outline ? 1.0f : 0.0f);
            push.key = p.key;
            vkCmdPushConstants(cmd, imagePlaneLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(cmd, 6, 1, 0, 0);
        }
    }

    /** A soft blob under each shaded mesh that rests near the ground, fading out as it lifts off. */
    void drawContactShadows(VkCommandBuffer cmd, const FrameState& s, const glm::mat4& viewProj, Viewport* v = nullptr)
    {
        LILA_ZONE();

        int up = s.zUp ? 2 : 1;
        int a0 = 0, a1 = s.zUp ? 1 : 2;
        if (!contactPipeline || s.groundShadow <= 0.0f || s.eye[up] <= 0.0)
            return;
        // The blobs depend only on the draws: worked out once while the frame draws the same list.
        std::vector<ContactPush> scratch;
        std::vector<ContactPush>& blobs = v ? v->keptContacts : scratch;
        const glm::vec2 key(s.groundShadow, s.zUp ? 1.0f : 0.0f);
        if (!v || !v->keptContactsValid || v->keptContactKey != key)
        {
            blobs.clear();
            for (const MeshDraw& d : s.meshes)
            {
                if (!(d.flags & DrawShaded) || !d.mesh || d.mesh->indices.empty())
                    continue;
                glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
                auto extend = [&](const glm::mat4& world) {
                    for (int c = 0; c < 8; ++c)
                    {
                        glm::vec3 corner((c & 1) ? d.mesh->boundsMax.x : d.mesh->boundsMin.x, (c & 2) ? d.mesh->boundsMax.y : d.mesh->boundsMin.y,
                                         (c & 4) ? d.mesh->boundsMax.z : d.mesh->boundsMin.z);
                        glm::vec3 w(world * glm::vec4(corner, 1.0f));
                        lo = glm::min(lo, w);
                        hi = glm::max(hi, w);
                    }
                };
                if (d.instances && !d.instances->empty())
                    for (const glm::mat4& placement : *d.instances)
                        extend(d.world * placement);
                else
                    extend(d.world);
                glm::vec3 half = (hi - lo) * 0.5f;
                float size = std::max(half[a0], half[a1]);
                float gap = lo[up];
                if (size <= 0.0f || hi[up] <= 0.0f)
                    continue;
                float opacity = s.groundShadow * (1.0f - std::clamp(gap / size, 0.0f, 1.0f));
                if (opacity <= 0.0f)
                    continue;

                ContactPush& p = blobs.emplace_back();
                glm::vec3 center = (lo + hi) * 0.5f;
                center[up] = 0.0f;
                p.center = glm::vec4(center, opacity);
                // The blob reaches past the footprint by a fraction of the object's size so it reads as soft occlusion.
                glm::vec3 axisA(0.0f), axisB(0.0f);
                float spread = size * 0.6f;
                axisA[a0] = half[a0] + spread;
                axisB[a1] = half[a1] + spread;
                p.axisA = glm::vec4(axisA, size / (size + spread));
                p.axisB = glm::vec4(axisB, 0.0f);
            }
            if (v)
            {
                v->keptContactsValid = true;
                v->keptContactKey = key;
            }
        }
        if (blobs.empty())
            return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, contactPipeline);
        for (ContactPush& p : blobs)
        {
            p.viewProj = viewProj;
            vkCmdPushConstants(cmd, contactLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
            vkCmdDraw(cmd, 6, 1, 0, 0);
        }
    }

    /**
     * kNoHandle when a map is missing or none of it uploaded, else something other than kNoHandle: its table handle on
     * the bindless path.
     */
    uint32_t mapHandle(const TexturePtr& t, uint64_t placeholder)
    {
        if (!t || textureKey(t, 0) == 0)
            return kNoHandle;
        if (!bindless)
            return 0;
        if (!frameTableAddress)
            return kNoHandle;
        auto it = frameHandles.find({t->key, placeholder});
        return it != frameHandles.end() ? it->second : kNoHandle;
    }

    /** A draw's material sections, or one section of slot 0 covering the whole mesh. */
    static std::span<const MeshSection> sectionsOf(const MeshDraw& d, const GpuMesh& g, MeshSection& whole)
    {
        if (!d.mesh->sections.empty())
            return d.mesh->sections;
        whole = {0, 0, g.indexCount};
        return {&whole, 1};
    }

    /**
     * Points frameRecords at the view's section records, rebuilt unless the frame draws the same list under the same
     * settings as the last; only the per-frame addresses and the eye are refreshed then.
     */
    void prepareRecords(Viewport& v, const FrameState& s, bool sameMeshes)
    {
        LILA_ZONE();
        RecordKey key;
        const float lighting = s.lighting == FrameState::Lighting::Flat ? 2.0f : s.lighting == FrameState::Lighting::Environment && envLoaded ? 1.0f : 0.0f;
        key.params = glm::vec4(s.exposure, s.zUp ? 1.0f : 0.0f, 0.0f, lighting);
        key.env = envParams(s);
        key.backTint = s.backfaceTint;
        key.gamma = s.gamma;
        key.saturation = s.envLightSaturation;
        key.lightCount = frameLightCount;
        key.bevel = frameBevel;
        key.grey = s.greyBaseColor;
        key.twoSided = s.twoSidedLighting;
        key.xray = s.xray;
        key.table = frameTableAddress != 0;
        key.instances = frameInstanceAddress != 0;
        key.bindless = bindless;
        key.wire = meshWirePipeline != VK_NULL_HANDLE;
        key.uploads = uploadEpoch;
        key.evictions = meshEvictions;
        key.textureTable.assign(frameTable.begin() + std::min<size_t>(frameTable.size(), vt::kTableHeaderWords), frameTable.end());
        std::vector<SectionRecord>& records = v.offscreen ? offscreenRecords : v.keptRecords;
        frameRecords = &records;
        if (!v.offscreen && sameMeshes && v.keptRecordsValid && key == v.keptRecordKey)
        {
            const glm::vec4 eye(glm::vec3(s.eye), 1.0f);
            for (SectionRecord& r : records)
            {
                r.push.instances = frameInstanceAddress;
                r.push.lights = frameLightAddress;
                r.push.eye = eye;
                if (r.material)
                    r.push.textureTable = frameTableAddress;
            }
            // Nothing else looks these meshes up this frame, and evictMeshes frees what goes unused.
            for (GpuMesh* g : v.keptGpuMeshes)
                g->lastUsed = frameCounter;
            return;
        }
        records.clear();
        v.keptGpuMeshes.clear();
        if (frameInstanceAddress)
            for (uint32_t bi = 0; bi < frameBatches.shaded.size(); ++bi)
            {
                const DrawBatch& b = frameBatches.shaded[bi];
                if (b.count == 0)
                    continue;
                const MeshDraw& d = *b.draw;
                GpuMesh* g = gpuMesh(d.mesh);
                if (!g)
                    continue;
                v.keptGpuMeshes.push_back(g);
                if (d.mesh->shareVertices)
                    if (auto base = meshCache.find(d.mesh->shareVertices->key); base != meshCache.end())
                        v.keptGpuMeshes.push_back(&base->second);
                for (MeshSection whole; const MeshSection& sec : sectionsOf(d, *g, whole))
                {
                    SectionRecord r;
                    r.batch = bi;
                    r.section = sec;
                    r.mesh = g;
                    r.push = sectionPush(s, b, *g, sec, r.set);
                    const Material* mat = sec.material < d.materials.size() ? &d.materials[sec.material] : nullptr;
                    r.material = mat != nullptr;
                    r.translucent = mat && (mat->flags & MatTranslucent);
                    r.lean = leanPush(r.push.flags, r.push.style);
                    if (r.set)
                        records.push_back(r);
                }
            }
        v.keptRecordKey = std::move(key);
        v.keptRecordsValid = !v.offscreen;
    }

    /**
     * The push constants of one material section of a batch, everything but viewProj, and the descriptor set it samples
     * through (null when there is none, and nothing draws).
     */
    MeshPush sectionPush(const FrameState& s, const DrawBatch& b, const GpuMesh& g, const MeshSection& sec, VkDescriptorSet& set)
    {
        const MeshDraw& d = *b.draw;
        const Material* mat = sec.material < d.materials.size() ? &d.materials[sec.material] : nullptr;
        MeshPush p;
        p.instances = frameInstanceAddress;
        p.firstInstance = b.first;
        p.eye = glm::vec4(glm::vec3(s.eye), 1.0f);
        float lighting = s.lighting == FrameState::Lighting::Flat ? 2.0f
                         : s.lighting == FrameState::Lighting::Environment && envLoaded ? 1.0f
                                                                                        : 0.0f;
        p.params = glm::vec4(s.exposure, s.zUp ? 1.0f : 0.0f, 0.0f, lighting);
        p.env = envParams(s);
        p.gamma = s.gamma;
        p.lights = frameLightAddress;
        p.lightCount = frameLightCount;
        p.lightPad = d.lightingChannels;
        if (mat)
        {
            p.extra = glm::vec4(mat->bevelRadius, frameBevel ? 1.0f : 0.0f, 0.0f, 0.0f);
            p.base = glm::vec4(mat->baseColor, mat->roughness);
            p.surface = glm::vec4(mat->metallic, mat->specular, mat->opacity, mat->opacityClip);
            p.emissive = mat->emissive;
            p.flags = mat->flags;
            if (isTwoChannelNormal(mat->normalMap))
                p.flags |= kNormalXY;
            p.sampling = mat->sampling;
            // Maps whose upload failed fall back to the constants.
            const uint32_t base = mapHandle(mat->baseColorMap, kWhiteSrgbKey), normal = mapHandle(mat->normalMap, kFlatNormalKey);
            const uint32_t orm = mapHandle(mat->ormMap, kWhiteLinearKey), emissive = mapHandle(mat->emissiveMap, kWhiteSrgbKey);
            if (base == kNoHandle)
                p.flags &= ~MatBaseColorMap;
            if (normal == kNoHandle)
                p.flags &= ~(MatNormalMap | MatNormalGreenUp);
            if (orm == kNoHandle)
                p.flags &= ~(MatMetallicMap | MatRoughnessMap | MatOcclusionMap);
            if (emissive == kNoHandle)
                p.flags &= ~MatEmissiveMap;
            if (s.greyBaseColor)
            {
                p.base = glm::vec4(glm::vec3(0.5f), mat->roughness);
                p.flags |= kGreyBase;
            }
            p.maps[0] = base | normal << 16;
            p.maps[1] = orm | emissive << 16;
            p.textureTable = frameTableAddress;
            if (frameTableAddress)
                if (auto it = frameStyles.find({&d, sec.material}); it != frameStyles.end())
                    p.style = it->second;
        }
        else
        {
            p.base = glm::vec4(d.baseColor, d.roughness);
            p.surface = glm::vec4(0.0f, 0.5f, 1.0f, 0.3333f);
            p.flags = MatTwoSided;
        }
        p.extra.z = std::clamp(s.envLightSaturation, 0.0f, 1.0f);
        if (d.vertexColors != VertexColorMode::Off)
            p.flags |= (d.vertexColors == VertexColorMode::Unlit ? kVertexColorUnlit : kVertexColorShaded) |
                       (uint32_t(d.colorChannel) << kColorChannelShift);
        if (!s.twoSidedLighting)
            p.flags |= kOneSidedLighting;
        if (s.backfaceTint.a > 0.0f)
            p.backTint = glm::packUnorm4x8(glm::clamp(s.backfaceTint, 0.0f, 1.0f));
        if (s.xray)
        {
            p.flags |= MatTranslucent;
            p.surface.z *= kXrayOpacity;
            p.surface.w *= kXrayOpacity; // masked cutouts stay where they were
        }
        set = bindless ? bindlessSet : materialSet(mat);
        if (b.wire)
        {
            const uint32_t first = std::min(sec.firstIndex, g.indexCount);
            p.edgeMask = g.edgeMask.address;
            p.firstTriangle = first / 3;
            p.wireWidth = std::max(d.wireWidth, 1.0f);
        }
        return p;
    }

    /** Records one section draw of count placements from first with whichever mesh pipeline is bound. */
    void emitSection(VkCommandBuffer cmd, const FrameState& s, MeshPush& p, VkDescriptorSet set, const GpuMesh& g, const MeshSection& sec,
                     uint32_t count, const glm::mat4& viewProj, VkDescriptorSet& boundSet)
    {
        p.viewProj = viewProj;
        if (set != boundSet)
        {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 1, 1, &set, 0, nullptr);
            boundSet = set;
        }
        // The viewport toggle alone decides, as Maya's does: one-sided materials show their back faces with it off.
        const VkCullModeFlags cull = s.backfaceCulling ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        if (cmd != cullCmd || cull != cullMode)
        {
            vkCmdSetCullMode(cmd, cull);
            cullCmd = cmd;
            cullMode = cull;
        }
        vkCmdPushConstants(cmd, meshLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &g.vertices.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, g.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        const uint32_t first = std::min(sec.firstIndex, g.indexCount);
        vkCmdDrawIndexed(cmd, std::min(sec.indexCount, g.indexCount - first), count, first, 0, 0);
        ++frameMeshDraws;
        frameMeshInstances += count;
    }

    /** Where the opaque sections of a pass record into secondary command buffers, and with what. */
    struct ChunkPass
    {
        VkFormat color = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        VkRenderingFlags flags = 0; // of the rendering pass that executes them, less the secondary contents bit
        VkExtent2D extent{};
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipeline wirePipeline = VK_NULL_HANDLE; // for batches that draw their wire in the shader; null draws them plain
        VkPipeline leanPipeline = VK_NULL_HANDLE, leanWirePipeline = VK_NULL_HANDLE; // for lean sections; null uses the two above
        VkDescriptorSet aoSet = VK_NULL_HANDLE;
        glm::mat4 viewProj{1.0f};
        size_t firstChunk = 0; // into Frame::chunks, so passes of one frame never share a buffer
    };
    static constexpr size_t kChunkSections = 1024; // sections per worker below which recording stays on this thread
    static constexpr size_t kMaxChunks = 16;

    /** Enough opaque sections that recording them on worker threads beats doing it here. LILA_VK_NO_CHUNKS turns it off. */
    bool chunked() const
    {
        return frameInstanceAddress && frameRecords->size() >= 2 * kChunkSections && std::getenv("LILA_VK_NO_CHUNKS") == nullptr;
    }

    /**
     * Records the frame's opaque section records on worker threads, in order, one secondary command buffer per slice:
     * executed in sequence they draw exactly what emitSection would. Translucent sections are left to the caller.
     */
    std::vector<VkCommandBuffer> recordChunks(Frame& f, const FrameState& s, const ChunkPass& pass)
    {
        LILA_ZONE();
        const std::vector<SectionRecord>& records = *frameRecords;
        const size_t count = std::clamp<size_t>(records.size() / kChunkSections, 1, kMaxChunks);
        while (f.chunks.size() < pass.firstChunk + count)
        {
            Frame::Chunk& c = f.chunks.emplace_back();
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pci.queueFamilyIndex = vk.queueFamily;
            vkCreateCommandPool(vk.device, &pci, nullptr, &c.pool);
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = c.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(vk.device, &ai, &c.cmd);
        }
        VkCommandBufferInheritanceRenderingInfo rendering{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO};
        rendering.flags = pass.flags;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &pass.color;
        rendering.depthAttachmentFormat = kDepthFormat;
        rendering.rasterizationSamples = pass.samples;
        VkCommandBufferInheritanceInfo inherit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        inherit.pNext = &rendering;
        const VkCullModeFlags cull = s.backfaceCulling ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        const VkViewport viewport{0, 0, float(pass.extent.width), float(pass.extent.height), 0, 1};
        const VkRect2D scissor{{0, 0}, pass.extent};
        std::vector<VkCommandBuffer> out(count);
        std::vector<std::pair<uint64_t, uint64_t>> counts(count); // draws, placements
        tbb::parallel_for(size_t(0), count, [&](size_t c) {
            LILA_ZONE_N("Record section chunk");
            const Frame::Chunk& chunk = f.chunks[pass.firstChunk + c];
            vkResetCommandPool(vk.device, chunk.pool, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
            bi.pInheritanceInfo = &inherit;
            VkCommandBuffer cmd = chunk.cmd;
            vkBeginCommandBuffer(cmd, &bi);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdSetCullMode(cmd, cull);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 0, 1, &envSet, 0, nullptr);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 2, 1, &pass.aoSet, 0, nullptr);
            VkPipeline bound = pass.pipeline;
            VkDescriptorSet boundSet = VK_NULL_HANDLE;
            auto& [draws, placements] = counts[c];
            const size_t begin = records.size() * c / count, end = records.size() * (c + 1) / count;
            for (size_t i = begin; i < end; ++i)
            {
                const SectionRecord& r = records[i];
                if (r.translucent)
                    continue;
                const DrawBatch& b = frameBatches.shaded[r.batch];
                const bool wire = pass.wirePipeline && b.wire && r.mesh->edgeMask.address;
                const VkPipeline lean = r.lean ? (wire ? pass.leanWirePipeline : pass.leanPipeline) : VK_NULL_HANDLE;
                const VkPipeline want = lean ? lean : wire ? pass.wirePipeline : pass.pipeline;
                if (want != bound)
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bound = want);
                if (r.set != boundSet)
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 1, 1, &(boundSet = r.set), 0, nullptr);
                MeshPush p = r.push;
                p.viewProj = pass.viewProj;
                vkCmdPushConstants(cmd, meshLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
                const GpuMesh& g = *r.mesh;
                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &g.vertices.buffer, &offset);
                vkCmdBindIndexBuffer(cmd, g.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                const uint32_t first = std::min(r.section.firstIndex, g.indexCount);
                vkCmdDrawIndexed(cmd, std::min(r.section.indexCount, g.indexCount - first), b.count, first, 0, 0);
                ++draws;
                placements += b.count;
            }
            vkEndCommandBuffer(cmd);
            out[c] = cmd;
        });
        for (const auto& [draws, placements] : counts)
        {
            frameMeshDraws += draws;
            frameMeshInstances += placements;
        }
        return out;
    }

    /**
     * With chunks (the frame to record them in), the caller began the pass suspending: the opaque sections record on
     * worker threads and run in a resumed instance of it, and the rest continue in a third, which leaves ri's flags clear.
     */
    void drawMeshes(VkCommandBuffer cmd, const FrameState& s, const glm::mat4& viewProj, VkDescriptorSet aoSet, uint32_t heightPx,
                    Frame* chunks = nullptr, VkRenderingInfo* ri = nullptr)
    {
        LILA_ZONE();

        if (chunks)
        {
            ChunkPass pass;
            pass.color = sceneFormat;
            pass.samples = vk.msaa;
            pass.flags = VK_RENDERING_RESUMING_BIT | VK_RENDERING_SUSPENDING_BIT;
            pass.extent = ri->renderArea.extent;
            pass.pipeline = meshPipeline;
            pass.wirePipeline = meshWirePipeline;
            pass.leanPipeline = meshLeanPipeline;
            pass.leanWirePipeline = meshWireLeanPipeline;
            pass.aoSet = aoSet;
            pass.viewProj = viewProj;
            pass.firstChunk = kMaxChunks;
            const std::vector<VkCommandBuffer> recorded = recordChunks(*chunks, s, pass);
            vkCmdEndRendering(cmd);
            ri->flags = pass.flags | VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
            vkCmdBeginRendering(cmd, ri);
            vkCmdExecuteCommands(cmd, uint32_t(recorded.size()), recorded.data());
            vkCmdEndRendering(cmd);
            ri->flags = VK_RENDERING_RESUMING_BIT;
            vkCmdBeginRendering(cmd, ri);
            ri->flags = 0;
            const VkViewport viewport{0, 0, float(pass.extent.width), float(pass.extent.height), 0, 1};
            const VkRect2D scissor{{0, 0}, pass.extent};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            cullCmd = VK_NULL_HANDLE;
        }
        if (s.meshes.empty() || !meshPipeline)
            return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 0, 1, &envSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout, 2, 1, &aoSet, 0, nullptr);

        struct Pending
        {
            SectionRecord* record;
            uint32_t instance; // translucent placements sort individually
            float distance;
        };
        std::vector<Pending> translucent;
        VkDescriptorSet boundSet = VK_NULL_HANDLE;
        VkPipeline bound = meshPipeline;
        auto use = [&](VkPipeline p) {
            if (p && p != bound)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bound = p);
                cullCmd = VK_NULL_HANDLE;
            }
        };
        cullCmd = VK_NULL_HANDLE;

        for (SectionRecord& r : *frameRecords)
        {
            const DrawBatch& b = frameBatches.shaded[r.batch];
            const bool wire = b.wire && r.mesh->edgeMask.address && meshWirePipeline;
            if (chunks && !r.translucent)
                continue;
            if (s.xray || r.translucent)
            {
                const glm::vec4 centre((b.draw->mesh->boundsMin + b.draw->mesh->boundsMax) * 0.5f, 1.0f);
                for (uint32_t i = 0; i < b.count; ++i)
                {
                    const glm::vec3 c(frameBatches.instances[b.first + i].world * centre);
                    translucent.push_back({&r, b.first + i, glm::length(c - glm::vec3(s.eye))});
                }
                continue;
            }
            const VkPipeline lean = r.lean ? (wire ? meshWireLeanPipeline : meshLeanPipeline) : VK_NULL_HANDLE;
            use(lean ? lean : wire ? meshWirePipeline : meshPipeline);
            emitSection(cmd, s, r.push, r.set, *r.mesh, r.section, b.count, viewProj, boundSet);
        }
        // Translucency after everything opaque, far to near, testing depth without writing it.
        if (!translucent.empty() && meshBlendPipeline)
        {
            std::ranges::stable_sort(translucent, [](const Pending& a, const Pending& b) { return a.distance > b.distance; });
            for (const Pending& t : translucent)
            {
                SectionRecord& r = *t.record;
                use(frameBatches.shaded[r.batch].wire && r.mesh->edgeMask.address && meshWireBlendPipeline ? meshWireBlendPipeline : meshBlendPipeline);
                MeshPush p = r.push;
                p.firstInstance = t.instance;
                emitSection(cmd, s, p, r.set, *r.mesh, r.section, 1, viewProj, boundSet);
            }
        }
        evictTextures();

        if (!wirePipeline || !frameInstanceAddress)
            return;
        for (const DrawBatch& b : frameBatches.wire)
        {
            const MeshDraw& d = *b.draw;
            GpuMesh* g = gpuMesh(d.mesh);
            WirePush p;
            p.viewProj = viewProj;
            p.instances = frameInstanceAddress;
            p.firstInstance = b.first;
            bool shaded = d.flags & (DrawShaded | DrawWireDepth);
            if (shaded)
                p.depthPull = depthPull(s.proj, heightPx, kLinePullPixels + 0.5 * lineWidth(d.wireWidth));
            p.depthPull.z = s.gamma;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shaded ? wirePipeline : wireXrayPipeline);
            vkCmdSetLineWidth(cmd, lineWidth(d.wireWidth));
            vkCmdPushConstants(cmd, wireLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &g->edges.buffer, &offset);
            if (g->edgeIndexCount)
            {
                vkCmdBindIndexBuffer(cmd, g->edgeIndices.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, g->edgeIndexCount, b.count, 0, 0, 0);
            }
            else
                vkCmdDraw(cmd, g->edgePointCount, b.count, 0, 0);
        }
        evictMeshes();
    }

    /**
     * Thins the frame's dense dot clouds to their nearest point per screen cell, ahead of the passes that draw them: a
     * dot covers its cell, so millions of points become about one per cell and draw in a fraction of the time.
     */
    void cullPoints(VkCommandBuffer cmd, Frame& f, const FrameState& s, const glm::mat4& viewProj, VkExtent2D extent)
    {
        frameCulledPoints.assign(s.points.size(), -1);
        if (!pointsNearestPipeline || !pointsKeepPipeline || extent.width == 0 || extent.height == 0)
            return;
        LILA_ZONE();
        constexpr uint32_t kMinPoints = 200'000, kMaxClouds = 4, kGroups = 65535 * 256;
        std::vector<std::pair<size_t, const GpuPoints*>> clouds;
        uint32_t cellPx = 0;
        for (size_t i = 0; i < s.points.size() && clouds.size() < kMaxClouds; ++i)
        {
            const PointDraw& d = s.points[i];
            auto it = d.cloud ? pointCache.find(d.cloud->key) : pointCache.end();
            if (it == pointCache.end() || !it->second.address || it->second.count < kMinPoints || it->second.count > kGroups)
                continue;
            clouds.emplace_back(i, &it->second);
            // Half the smaller dot: neighbouring kept dots still overlap, so a dense surface stays solid.
            cellPx = std::max(cellPx, uint32_t(std::max(1.0f, std::floor(std::min(d.size, d.selectedSize) * 0.5f))));
        }
        if (clouds.empty())
            return;
        const uint32_t across = (extent.width + cellPx - 1) / cellPx, down = (extent.height + cellPx - 1) / cellPx;
        const VkDeviceSize cells = VkDeviceSize(across) * down;
        auto ensure = [&](compute::Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage) {
            if (b.size >= bytes)
                return true;
            compute::destroy(vk, b);
            try
            {
                b = compute::makeBuffer(vk, bytes, usage, compute::Host::None);
                return true;
            }
            catch (const std::exception& e)
            {
                log::error("render: {}", e.what());
                return false;
            }
        };
        const VkDeviceSize n = clouds.size();
        if (!ensure(f.pointCells, n * cells * 2 * sizeof(uint32_t), 0) ||
            !ensure(f.pointOut, n * cells * sizeof(PointVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
            !ensure(f.pointArgs, n * sizeof(VkDrawIndirectCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT))
            return;

        vkCmdFillBuffer(cmd, f.pointCells.buffer, 0, n * cells * 2 * sizeof(uint32_t), 0);
        std::vector<VkDrawIndirectCommand> args(clouds.size(), VkDrawIndirectCommand{0, 1, 0, 0});
        vkCmdUpdateBuffer(cmd, f.pointArgs.buffer, 0, args.size() * sizeof(VkDrawIndirectCommand), args.data());
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        auto run = [&](VkPipeline pipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            for (size_t k = 0; k < clouds.size(); ++k)
            {
                PointsCullPush p;
                p.viewProj = viewProj;
                p.grid = glm::uvec4(across, down, cellPx, clouds[k].second->count);
                p.viewport = glm::vec4(float(extent.width), float(extent.height), 0.0f, 0.0f);
                p.points = clouds[k].second->address;
                p.cells = f.pointCells.address + k * cells * 2 * sizeof(uint32_t);
                p.out = f.pointOut.address + k * cells * sizeof(PointVertex);
                p.args = f.pointArgs.address + k * sizeof(VkDrawIndirectCommand);
                vkCmdPushConstants(cmd, pointsCullLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                vkCmdDispatch(cmd, (clouds[k].second->count + 255) / 256, 1, 1);
            }
        };
        run(pointsNearestPipeline);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        run(pointsKeepPipeline);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                      VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        for (size_t k = 0; k < clouds.size(); ++k)
            frameCulledPoints[clouds[k].first] = int(k);
        frameCullCells = cells;
    }
    VkDeviceSize frameCullCells = 0;

    void drawPoints(VkCommandBuffer cmd, Frame& f, const FrameState& s, const glm::mat4& viewProj, uint32_t heightPx)
    {
        if (!pointsPipeline || s.points.empty())
            return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pointsPipeline);
        for (size_t i = 0; i < s.points.size(); ++i)
        {
            const PointDraw& d = s.points[i];
            auto it = d.cloud ? pointCache.find(d.cloud->key) : pointCache.end();
            if (it == pointCache.end())
                continue;
            it->second.lastUsed = frameCounter;
            PointsPush p;
            p.viewProj = viewProj;
            p.depthPull = depthPull(s.proj, heightPx, kLinePullPixels + 0.5 * std::max(d.size, d.selectedSize));
            for (int c = 0; c < 3; ++c)
                p.colors[c] = d.colors[c];
            const float* range = vk.properties.limits.pointSizeRange;
            p.sizes = vk.largePoints
                          ? glm::vec4(std::clamp(d.size, range[0], range[1]), std::clamp(d.selectedSize, range[0], range[1]), 0.0f, 0.0f)
                          : glm::vec4(1.0f, 1.0f, 0.0f, 0.0f);
            vkCmdPushConstants(cmd, pointsLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(p), &p);
            if (const int k = i < frameCulledPoints.size() ? frameCulledPoints[i] : -1; k >= 0)
            {
                const VkDeviceSize offset = VkDeviceSize(k) * frameCullCells * sizeof(PointVertex);
                vkCmdBindVertexBuffers(cmd, 0, 1, &f.pointOut.buffer, &offset);
                vkCmdDrawIndirect(cmd, f.pointArgs.buffer, VkDeviceSize(k) * sizeof(VkDrawIndirectCommand), 1, sizeof(VkDrawIndirectCommand));
                continue;
            }
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &it->second.buffer.buffer, &offset);
            vkCmdDraw(cmd, it->second.count, 1, 0, 0);
        }
        evictPoints();
        evictHeightfields();
    }

    /** One slice of a task; resets task when it finished or threw. */
    void runTaskSlice(std::unique_ptr<QueuedTask>& task)
    {
        LILA_ZONE();

        try
        {
            if (!task->step(vk))
                return;
            task->done.set_value();
        }
        catch (...)
        {
            task->done.set_exception(std::current_exception());
        }
        task.reset();
    }

    /** A new state once its frame cap allows, or the next frame of an accumulation. */
    static bool due(const Viewport& v, std::chrono::steady_clock::time_point now)
    {
        return v.hasState && ((v.pending && now >= v.drawAt) || (v.converging && now >= v.convergeAt));
    }

    void loop()
    {
        LILA_THREAD_NAME("Render");
        std::unique_lock lock(mutex);
        auto ready = [&] {
            if (quit || !toRemove.empty())
                return true;
            // Captures wait out an environment build in flight, so they are lit like the views.
            if (!offscreenJobs.empty() && !envBusy)
                return true;
            if (!tasks.empty())
                return true;
            // A viewport with no frame state yet can't draw, and waking for it would spin with the mutex held.
            const auto now = std::chrono::steady_clock::now();
            for (auto& [id, v] : viewports)
                if (due(*v, now))
                    return true;
            return false;
        };
        auto nextWake = [&] {
            std::optional<std::chrono::steady_clock::time_point> next;
            auto consider = [&](std::chrono::steady_clock::time_point t) {
                if (!next || t < *next)
                    next = t;
            };
            for (auto& [id, v] : viewports)
            {
                if (!v->hasState)
                    continue;
                if (v->converging)
                    consider(v->convergeAt);
                if (v->pending)
                    consider(v->drawAt);
            }
            return next;
        };
        while (true)
        {
            // Big uploads leave big staging buffers, slow to free: they go once the GPU is done with them, while the
            // thread is idle, instead of in front of the next frame drawn into the same slot.
            const auto wake = nextWake();
            if (reclaimStaging())
                cv.wait_for(lock, std::chrono::milliseconds(20), ready);
            else if (wake)
                cv.wait_until(lock, *wake, ready);
            else
                cv.wait(lock, ready);
            if (quit)
                break;

            for (ViewportId id : toRemove)
                if (auto it = viewports.find(id); it != viewports.end())
                {
                    destroyViewport(*it->second);
                    viewports.erase(it);
                }
            if (!toRemove.empty())
            {
                toRemove.clear();
                removed.notify_all();
            }

            while (!offscreenJobs.empty() && !envBusy)
            {
                std::shared_ptr<OffscreenJob> job = offscreenJobs.front();
                std::unique_ptr<EnvironmentData> env = pollEnvironment(job->envPath());
                if (!env && envBusy)
                    break;
                offscreenJobs.pop_front();
                lock.unlock();
                if (env)
                    uploadEnvironment(*env);
                runOffscreen(*job);
                lock.lock();
                job->done = true;
                offscreenDone.notify_all();
            }

            if (!tasks.empty())
            {
                std::unique_ptr<QueuedTask> task = std::move(tasks.front());
                tasks.pop_front();
                lock.unlock();
                runTaskSlice(task);
                lock.lock();
                if (task)
                    tasks.push_back(std::move(task));
            }

            for (auto& [id, vp] : viewports)
            {
                Viewport& v = *vp;
                if (!due(v, std::chrono::steady_clock::now()))
                    continue;
                v.pending = false;
                v.converging = false;
                {
                    // Waits wake late on Windows, so frames keep to the cap's schedule instead of restarting it from
                    // each late wake, which would land well under the cap. A view idle for longer starts afresh.
                    const auto now = std::chrono::steady_clock::now();
                    const auto interval = frameInterval(v);
                    v.drawAt = now - v.drawAt < interval ? v.drawAt + interval : now + interval;
                }
                if (!v.surface)
                {
                    v.surface = createSurface(vk.instance, v.surfaceWindow.handle);
                    if (v.surface)
                    {
                        VkBool32 present = VK_FALSE;
                        vkGetPhysicalDeviceSurfaceSupportKHR(vk.physical, vk.queueFamily, v.surface, &present);
                        if (!present)
                        {
                            log::error("render: the selected GPU queue cannot present to this window");
                            vkDestroySurfaceKHR(vk.instance, v.surface, nullptr);
                            v.surface = VK_NULL_HANDLE;
                        }
                        else
                            initFrames(v);
                    }
                }
                if (!v.surface)
                    continue;
                std::unique_ptr<EnvironmentData> env = pollEnvironment(v.state.envPath);
                // Until the first environment arrives there is nothing to light with; the worker re-queues every viewport.
                if (!env && !envLoaded)
                    continue;
                FrameState s = v.state;
                v.renderingSerial = v.submitted;
                lock.unlock();
                if (env)
                    uploadEnvironment(*env);
                auto latch = [&](FrameState& latest) {
                    std::scoped_lock relock(mutex);
                    // A new environment or sample count needs the setup that ran before the wait, so it waits a frame.
                    if (v.pending && v.state.envPath == latest.envPath && v.state.msaaSamples == latest.msaaSamples)
                    {
                        latest = v.state;
                        v.renderingSerial = v.submitted;
                        v.pending = false;
                    }
                };
                bool retry = render(v, s, latch);
                LILA_FRAME_MARK();
                plotFrame(v.timing);
                lock.lock();
                ++v.timing.frames;
                v.stats = v.timing;
                if (retry && v.retries++ < 3)
                {
                    v.pending = true;
                    v.drawAt = {};
                }
                // Still converging on the same state: draw it again a refresh (or the cap, if longer) later, unless a
                // new state comes first. Mailbox presentation never blocks, so without this the same image redraws
                // thousands of times a second.
                else if (v.wantsMore && !retry)
                {
                    v.converging = true;
                    v.convergeAt = std::chrono::steady_clock::now() +
                                   std::max(std::chrono::duration_cast<std::chrono::steady_clock::duration>(refreshPeriod(v)), frameInterval(v));
                }
            }
        }

        for (auto& t : tasks)
            t->done.set_exception(std::make_exception_ptr(std::runtime_error("the renderer stopped")));
        tasks.clear();
        for (auto& [id, v] : viewports)
            destroyViewport(*v);
        viewports.clear();
        // The worker takes the mutex when it finishes.
        lock.unlock();
        if (envWorker.joinable())
            envWorker.join();
        lock.lock();
        vkDeviceWaitIdle(vk.device);
        destroyMeshes();
        removed.notify_all();
    }
};

Renderer::Renderer() : m(std::make_unique<Impl>()) {}
Renderer::~Renderer() { stop(); }

Renderer& Renderer::get()
{
    static Renderer r;
    return r;
}

bool Renderer::start()
{
    if (m->started)
        return true;
#ifdef NDEBUG
    bool validation = std::getenv("LILA_VK_VALIDATION") != nullptr;
#else
    bool validation = true;
#endif
    if (!m->vk.init(validation))
        return false;
    if (m->vk.properties.limits.maxPushConstantsSize < 256)
        log::warn("render: GPU push constant limit is {} bytes; grid may not draw", m->vk.properties.limits.maxPushConstantsSize);
    m->initDescriptors();
    gpuProfile::start(m->vk);
    m->quit = false;
    m->started = true;
    m->thread = std::thread([this] { m->loop(); });
    return true;
}

void Renderer::stop()
{
    if (!m->started)
        return;
    {
        std::scoped_lock lock(m->mutex);
        m->quit = true;
    }
    m->cv.notify_all();
    m->thread.join();
    for (const auto& hook : m->stopHooks)
        hook(m->vk);
    gpuProfile::stop();
    compute::sharedPipelines().clear(m->vk);
    GpuImage::releaseAll(m->vk);
    m->destroyPipelines();
    m->destroyDescriptors();
    m->vk.shutdown();
    m->started = false;
}

void Renderer::addStopHook(std::function<void(VulkanContext& vk)> fn)
{
    m->stopHooks.push_back(std::move(fn));
}

bool Renderer::running() const { return m->started; }

ViewportId Renderer::addViewport(uintptr_t nativeWindow, int width, int height)
{
    SurfaceWindow surfaceWindow = prepareSurfaceWindow(nativeWindow, width, height);
    if (!surfaceWindow.handle)
        return 0;
    std::scoped_lock lock(m->mutex);
    auto v = std::make_unique<Viewport>();
    v->window = nativeWindow;
    v->surfaceWindow = std::move(surfaceWindow);
    v->width = width;
    v->height = height;
    ViewportId id = m->nextId++;
    m->viewports[id] = std::move(v);
    return id;
}

void Renderer::removeViewport(ViewportId id)
{
    if (!m->started)
        return;
    std::unique_lock lock(m->mutex);
    m->toRemove.push_back(id);
    m->cv.notify_all();
    m->removed.wait(lock, [&] { return !m->viewports.contains(id) || m->quit; });
}

void Renderer::removeAllViewports()
{
    if (!m->started)
        return;
    std::unique_lock lock(m->mutex);
    for (auto& [id, v] : m->viewports)
        m->toRemove.push_back(id);
    m->cv.notify_all();
    m->removed.wait(lock, [&] { return m->viewports.empty() || m->quit; });
}

void Renderer::resize(ViewportId id, int width, int height)
{
    SurfaceWindow surfaceWindow;
    uintptr_t nativeWindow = 0;
    {
        std::scoped_lock lock(m->mutex);
        auto it = m->viewports.find(id);
        if (it == m->viewports.end())
            return;
        Viewport& v = *it->second;
        surfaceWindow = v.surfaceWindow;
        nativeWindow = v.window;
        // The swapchain is rebuilt with the next submitted frame, which carries the matching projection.
        if (v.width != width || v.height != height)
            v.needsRecreate = true;
        v.width = width;
        v.height = height;
    }
    // AppKit work stays on the caller's GUI thread, outside the render mutex.
    resizeSurfaceWindow(surfaceWindow, nativeWindow, width, height);
}

void Renderer::submit(ViewportId id, FrameState state)
{
    {
        std::scoped_lock lock(m->mutex);
        auto it = m->viewports.find(id);
        if (it == m->viewports.end())
            return;
        it->second->state = std::move(state);
        ++it->second->submitted;
        it->second->pending = true;
        it->second->hasState = true;
        it->second->retries = 0;
    }
    m->cv.notify_all();
}

Renderer::FrameStats Renderer::stats(ViewportId id) const
{
    std::scoped_lock lock(m->mutex);
    auto it = m->viewports.find(id);
    return it == m->viewports.end() ? FrameStats{} : it->second->stats;
}

bool Renderer::renderOffscreenFrames(std::vector<FrameState> frames, int width, int height, int samples, bool hdr, OffscreenResult& out,
                                     std::string* err)
{
    if (frames.empty())
    {
        if (err)
            *err = "no frames";
        return false;
    }
    if (!m->started)
    {
        if (err)
            *err = "renderer not running";
        return false;
    }
    if (width < 1 || height < 1 || width > 16384 || height > 16384)
    {
        if (err)
            *err = "size out of range";
        return false;
    }
    auto job = std::make_shared<OffscreenJob>();
    job->frames = std::move(frames);
    job->width = width;
    job->height = height;
    job->samples = std::clamp(samples, 1, 4096);
    job->hdr = hdr;
    const size_t draws = job->frames.size() * size_t(job->samples);
    std::unique_lock lock(m->mutex);
    m->offscreenJobs.push_back(job);
    m->cv.notify_all();
    const auto limit = std::chrono::seconds(60) + std::chrono::milliseconds(250) * draws;
    if (!m->offscreenDone.wait_for(lock, limit, [&] { return job->done || m->quit; }) || !job->done)
    {
        std::erase(m->offscreenJobs, job);
        if (err)
            *err = "timed out waiting for the render thread";
        return false;
    }
    if (!job->ok)
    {
        if (err)
            *err = job->error;
        return false;
    }
    out.bgra = std::move(job->pixels);
    out.hdr = std::move(job->hdrPixels);
    out.timing = job->timing;
    return true;
}

bool Renderer::renderOffscreen(FrameState state, int width, int height, std::vector<uint8_t>& bgra, std::string* err, FrameStats* timing,
                               int repeat, int samples)
{
    auto fail = [&](const std::string& e) {
        if (err)
            *err = e;
        return false;
    };
    if (!m->started)
        return fail("renderer not running");
    if (width < 1 || height < 1 || width > 8192 || height > 8192)
        return fail("size out of range");
    auto job = std::make_shared<OffscreenJob>();
    job->state = std::move(state);
    job->width = width;
    job->height = height;
    job->repeat = std::clamp(repeat, 1, 1000);
    job->samples = std::clamp(samples, 1, 4096);
    std::unique_lock lock(m->mutex);
    m->offscreenJobs.push_back(job);
    m->cv.notify_all();
    const auto limit = std::chrono::seconds(60) + std::chrono::milliseconds(250) * job->samples;
    if (!m->offscreenDone.wait_for(lock, limit, [&] { return job->done || m->quit; }) || !job->done)
    {
        std::erase(m->offscreenJobs, job);
        return fail("timed out waiting for the render thread");
    }
    if (!job->ok)
        return fail(job->error);
    bgra = std::move(job->pixels);
    if (timing)
        *timing = job->timing;
    return true;
}

std::string Renderer::deviceName() const
{
    return m->vk.deviceName;
}

Renderer::GpuInfo Renderer::gpuInfo() const
{
    GpuInfo info;
    if (!running())
        return info;
    info.name = m->vk.deviceName;
    info.discrete = m->vk.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(m->vk.physical, &mem);
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
        if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            info.vramBytes += mem.memoryHeaps[i].size;
    return info;
}

std::future<void> Renderer::runTask(GpuTask task)
{
    auto t = std::make_unique<Impl::QueuedTask>();
    t->step = std::move(task);
    std::future<void> f = t->done.get_future();
    std::scoped_lock lock(m->mutex);
    if (!m->started || m->quit)
    {
        t->done.set_exception(std::make_exception_ptr(std::runtime_error("the renderer is not running")));
        return f;
    }
    m->tasks.push_back(std::move(t));
    m->cv.notify_all();
    return f;
}

VulkanContext* Renderer::context()
{
    return m->started ? &m->vk : nullptr;
}

std::shared_ptr<const EnvProbe> Renderer::environmentProbe(const std::string& path) const
{
    std::scoped_lock lock(m->probeMutex);
    return m->probePath == path ? m->probe : nullptr;
}
} // namespace lila::render
