#pragma once

#include "core/Export.h"

#include <volk.h>

#include <cstdint>
#include <memory>

namespace lila::render
{
/** A presentation handle and its lifetime: HWND on Windows, a retained CAMetalLayer on macOS. */
struct SurfaceWindow
{
    uintptr_t handle = 0;
    std::shared_ptr<void> owner;
};
/** Prepare/update native views only on the GUI thread, before the render thread touches their surfaces. */
LILA_UI_API SurfaceWindow prepareSurfaceWindow(uintptr_t nativeWindow, int width, int height);
LILA_UI_API void resizeSurfaceWindow(const SurfaceWindow& window, uintptr_t nativeWindow, int width, int height);
/** Takes SurfaceWindow::handle, not an NSView. Safe on the render thread; never calls back to the GUI thread. */
LILA_UI_API VkSurfaceKHR createSurface(VkInstance instance, uintptr_t nativeWindow);
/** Refresh rate of the display showing the window; 60 when unknown. */
LILA_UI_API double refreshHz(uintptr_t nativeWindow);
} // namespace lila::render
