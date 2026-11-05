#include "render/Surface.h"
#include "core/Log.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CATransaction.h>
#include <vulkan/vulkan_metal.h>

#include <algorithm>

namespace lila::render
{
void resizeSurfaceWindow(const SurfaceWindow& window, uintptr_t nativeWindow, int width, int height)
{
    // The renderer may hold its mutex while the GUI waits for viewport removal. Never dispatch_sync to the GUI.
    if (!window.handle || !nativeWindow || ![NSThread isMainThread])
        return;
    NSView* view = reinterpret_cast<NSView*>(nativeWindow);
    CAMetalLayer* layer = reinterpret_cast<CAMetalLayer*>(window.handle);
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    layer.frame = view.bounds;
    layer.contentsScale = view.window ? view.window.backingScaleFactor : NSScreen.mainScreen.backingScaleFactor;
    layer.drawableSize = CGSizeMake(std::max(1, width), std::max(1, height));
    [CATransaction commit];
}

SurfaceWindow prepareSurfaceWindow(uintptr_t nativeWindow, int width, int height)
{
    if (!nativeWindow || ![NSThread isMainThread])
    {
        log::error("vulkan: Metal presentation must be prepared on the GUI thread");
        return {};
    }
    NSView* view = reinterpret_cast<NSView*>(nativeWindow);
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.opaque = YES;
    layer.framebufferOnly = NO; // Viewport capture reads the swapchain image.
    view.wantsLayer = YES;
    view.layer = layer;
    // Keep the layer alive until the render thread has destroyed the swapchain and surface, even after reparenting.
    SurfaceWindow window{reinterpret_cast<uintptr_t>(layer),
                         std::shared_ptr<void>([layer retain], [](void* p) { [static_cast<CAMetalLayer*>(p) release]; })};
    resizeSurfaceWindow(window, nativeWindow, width, height);
    return window;
}

VkSurfaceKHR createSurface(VkInstance instance, uintptr_t layer)
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const auto create = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(vkGetInstanceProcAddr(instance, "vkCreateMetalSurfaceEXT"));
    VkMetalSurfaceCreateInfoEXT info{VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT};
    info.pLayer = reinterpret_cast<CAMetalLayer*>(layer);
    if (!layer || !create || create(instance, &info, nullptr, &surface) != VK_SUCCESS)
        log::error("vulkan: failed to create a Metal window surface");
    return surface;
}

double refreshHz(uintptr_t)
{
    // Called on the render thread. Use the documented fallback without accessing AppKit views off the main thread.
    return 60.0;
}
} // namespace lila::render
