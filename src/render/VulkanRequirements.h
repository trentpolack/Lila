#pragma once

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_beta.h>
#include <initializer_list>

namespace lila::render
{
/** Features used without a fallback by the viewport and compute/bake shaders. Null means usable. */
inline const char* missingRequiredFeature(const VkPhysicalDeviceVulkan11Features& f11, const VkPhysicalDeviceVulkan12Features& f12,
                                           const VkPhysicalDeviceVulkan13Features& f13,
                                           const VkPhysicalDevicePortabilitySubsetFeaturesKHR* portability = nullptr)
{
    if (!f11.shaderDrawParameters) return "shaderDrawParameters";
    if (!f12.timelineSemaphore) return "timelineSemaphore";
    if (!f12.bufferDeviceAddress) return "bufferDeviceAddress";
    if (!f13.dynamicRendering) return "dynamicRendering";
    if (!f13.synchronization2) return "synchronization2";
    if (!f13.shaderDemoteToHelperInvocation) return "shaderDemoteToHelperInvocation";
    if (portability && !portability->imageViewFormatSwizzle) return "imageViewFormatSwizzle";
    return nullptr;
}

inline VkPhysicalDeviceFeatures rasterFeatures(const VkPhysicalDeviceFeatures& supported)
{
    VkPhysicalDeviceFeatures enabled{};
    enabled.wideLines = supported.wideLines;
    enabled.largePoints = supported.largePoints;
    enabled.fillModeNonSolid = supported.fillModeNonSolid;
    enabled.samplerAnisotropy = supported.samplerAnisotropy;
    enabled.textureCompressionBC = supported.textureCompressionBC;
    return enabled;
}

inline VkCompositeAlphaFlagBitsKHR surfaceCompositeAlpha(VkCompositeAlphaFlagsKHR supported)
{
    for (auto mode : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                     VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
        if (supported & mode)
            return mode;
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Vulkan requires at least one supported mode.
}
} // namespace lila::render
