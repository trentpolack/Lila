#pragma once

#include "core/Export.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lila::render
{
/** Instance, device and allocator shared by every viewport. Created once, used from the render thread. */
class LILA_UI_API VulkanContext
{
public:
    bool init(bool validation);
    void shutdown();

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_4_BIT; // what the viewport targets and pipelines use right now
    VkSampleCountFlags msaaSupported = VK_SAMPLE_COUNT_1_BIT;
    bool anisotropy = false; // samplerAnisotropy enabled
    bool bcTextures = false; // textureCompressionBC enabled
    bool wideLines = false;
    bool largePoints = false;
    std::string deviceName;

    // Optional features, each enabled only when the device supports it.
    /** VK_KHR_acceleration_structure, VK_KHR_ray_query and VK_KHR_deferred_host_operations. */
    bool rayQuery = false;
    /** Runtime-sized, partially bound, update-after-bind (also while pending) descriptor arrays (bindless). LILA_VK_NO_BINDLESS turns it off. */
    bool descriptorIndexing = false;
    bool shaderFloat16 = false;
    bool shaderInt64 = false;
    /** Double precision in shaders (the baker's raster matches the CPU's coverage tests with it). */
    bool shaderFloat64 = false;
    bool int64Atomics = false;
    /** Fragment shaders writing storage buffers (virtual texture feedback). */
    bool fragmentStores = false;
    /** VK_KHR_fragment_shader_barycentric: the mesh shader draws wireframes over shaded meshes itself. */
    bool fragmentBarycentric = false;
    /** VK_EXT_memory_budget: the allocator tracks per-heap budgets (vmaGetHeapBudgets). */
    bool memoryBudget = false;
    /** VK_KHR_ or VK_EXT_calibrated_timestamps (whichever the device has) with hostQueryReset, for the profiler's GPU zones; null without. */
    const char* calibratedTimestamps = nullptr;
    /** VK_EXT_debug_utils: command buffer labels name the viewport's passes in GPU profilers (Nsight, RenderDoc). */
    bool debugLabels = false;
    /** VK_KHR_external_memory_win32: GpuImages another Vulkan device in the process can import (GpuImageDesc::shared). */
    bool externalMemory = false;
    /** VkPhysicalDeviceIDProperties::deviceUUID: another device with the same one drives the same GPU. */
    std::array<uint8_t, 16> deviceUuid{};
    uint32_t asScratchAlignment = 256;

private:
    VkDebugUtilsMessengerEXT mMessenger = VK_NULL_HANDLE;
};

LILA_UI_API const char* vkResultName(VkResult r);
} // namespace lila::render
