#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>

#include "core/SystemMemory.h"
#include "python/RuntimePaths.h"
#include "render/VulkanRequirements.h"

#include <chrono>
#include <filesystem>

using namespace lila;

TEST_CASE("Vulkan baseline does not require optional raster features")
{
    VkPhysicalDeviceVulkan11Features f11{};
    VkPhysicalDeviceVulkan12Features f12{};
    VkPhysicalDeviceVulkan13Features f13{};
    f11.shaderDrawParameters = VK_TRUE;
    f12.timelineSemaphore = f12.bufferDeviceAddress = VK_TRUE;
    f13.dynamicRendering = f13.synchronization2 = f13.shaderDemoteToHelperInvocation = VK_TRUE;
    CHECK(render::missingRequiredFeature(f11, f12, f13) == nullptr);
    SUBCASE("portability devices must permit the renderer's grayscale image swizzles")
    {
        VkPhysicalDevicePortabilitySubsetFeaturesKHR portability{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR};
        CHECK(render::missingRequiredFeature(f11, f12, f13, &portability) != nullptr);
        portability.imageViewFormatSwizzle = VK_TRUE;
        CHECK(render::missingRequiredFeature(f11, f12, f13, &portability) == nullptr);
    }
    SUBCASE("buffer addresses are mandatory for mesh and bake data")
    {
        f12.bufferDeviceAddress = VK_FALSE;
        CHECK(render::missingRequiredFeature(f11, f12, f13) != nullptr);
    }
    SUBCASE("dynamic rendering is mandatory")
    {
        f13.dynamicRendering = VK_FALSE;
        CHECK(render::missingRequiredFeature(f11, f12, f13) != nullptr);
    }
    SUBCASE("each remaining shader and synchronization requirement is checked")
    {
        for (VkBool32* feature : {&f11.shaderDrawParameters, &f12.timelineSemaphore, &f13.synchronization2,
                                  &f13.shaderDemoteToHelperInvocation})
        {
            *feature = VK_FALSE;
            CHECK(render::missingRequiredFeature(f11, f12, f13) != nullptr);
            *feature = VK_TRUE;
        }
    }
}

TEST_CASE("unsupported raster features are never requested")
{
    VkPhysicalDeviceFeatures supported{};
    supported.largePoints = VK_TRUE;
    supported.textureCompressionBC = VK_TRUE;
    const auto enabled = render::rasterFeatures(supported);
    CHECK(enabled.wideLines == VK_FALSE);
    CHECK(enabled.largePoints == VK_TRUE);
    CHECK(enabled.fillModeNonSolid == VK_FALSE);
    CHECK(enabled.samplerAnisotropy == VK_FALSE);
    CHECK(enabled.textureCompressionBC == VK_TRUE);
    supported.wideLines = supported.samplerAnisotropy = VK_TRUE;
    CHECK(render::rasterFeatures(supported).wideLines == VK_TRUE);
    CHECK(render::rasterFeatures(supported).samplerAnisotropy == VK_TRUE);
}

TEST_CASE("swapchains use an advertised composite alpha mode")
{
    CHECK(render::surfaceCompositeAlpha(VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) == VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
    CHECK(render::surfaceCompositeAlpha(VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) == VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);
    CHECK(render::surfaceCompositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR | VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
}

TEST_CASE("Python uses a complete bundled stdlib or the configured installation prefix")
{
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("lila_runtime_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup
    {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{dir};
    const auto configured = dir / "Python installation";
    const auto bundled = dir / "python-runtime";
    CHECK(python::runtimeHome(dir, configured, "lib/python3.14", nullptr) == configured);
    fs::create_directories(bundled / "lib/python3.14");
    CHECK(python::runtimeHome(dir, configured, "lib/python3.14", nullptr) == configured);
    fs::create_directories(bundled / "lib/python3.14/encodings");
    CHECK(python::runtimeHome(dir, configured, "lib/python3.14", "") == bundled);
    CHECK(python::runtimeHome(dir, configured, "Lib", nullptr) == configured);
    fs::create_directories(bundled / "Lib/encodings");
    CHECK(python::runtimeHome(dir, configured, "Lib", nullptr) == bundled);
    CHECK(python::runtimeHome(dir, configured, "Lib", "/custom/python") == fs::path("/custom/python"));
}

TEST_CASE("physical memory is available to cache budgeting")
{
    CHECK(physicalMemoryBytes() > 0);
}
