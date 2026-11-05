set(_lila_surface Surface.cpp)
if(APPLE)
    set(_lila_surface SurfaceMac.mm)
endif()
lila_module(lila_render UI
    Backdrop.cpp
    Baker.cpp
    BakerTrace.cpp
    BakerResident.cpp
    BakerKernels.cpp
    Compute.cpp
    GpuCompute.cpp
    CubeLut.cpp
    ClayGpu.cpp
    DeformGpu.cpp
    DrawBatches.cpp
    Environment.cpp
    GpuImage.cpp
    GpuProfile.cpp
    HdrImage.cpp
    MeshMaps.cpp
    PaintCommands.cpp
    PaintGpu.cpp
    Renderer.cpp
    ShadingModels.cpp
    SharedMemory.cpp
    ${_lila_surface}
    TexEval.cpp
    TexEvalState.cpp
    TexEvalRecord.cpp
    TexEvalPages.cpp
    TextureFormat.cpp
    VirtualTexture.cpp
    VulkanContext.cpp)

target_link_libraries(lila_render PUBLIC lila_core volk::volk Vulkan::Headers GPUOpen::VulkanMemoryAllocator)
# Vulkan headers still gate the portability subset's structure IDs behind this switch.
target_compile_definitions(lila_render PRIVATE VK_ENABLE_BETA_EXTENSIONS)
if(APPLE)
    find_library(LILA_MOLTENVK NAMES MoltenVK
        HINTS "$ENV{VULKAN_SDK}/lib" /opt/homebrew/lib /opt/homebrew/opt/molten-vk/lib REQUIRED)
    target_compile_definitions(lila_render PRIVATE LILA_MOLTENVK="${LILA_MOLTENVK}")
    target_link_libraries(lila_render PRIVATE "-framework AppKit" "-framework QuartzCore")
    set_source_files_properties(SurfaceMac.mm PROPERTIES COMPILE_OPTIONS "-fno-objc-arc" SKIP_UNITY_BUILD_INCLUSION ON)
endif()

lila_add_shaders(lila_render FILES
    shaders/ao.slang
    shaders/ao_apply.slang
    shaders/bake_dilate.slang
    shaders/bake_hw.slang
    shaders/bake_image.slang
    shaders/bake_raster32.slang
    shaders/bake_raster64.slang
    shaders/bake_sw.slang
    shaders/bloom.slang
    shaders/compose.slang
    shaders/clay.slang
    shaders/contact.slang
    shaders/deform.slang
    shaders/exposure.slang
    shaders/external.slang
    shaders/grid.slang
    shaders/heightfield.slang
    shaders/imageplane.slang
    shaders/lines.slang
    shaders/overlay.slang
    shaders/paint_dab.slang
    shaders/paint_depth.slang
    shaders/paint_gather.slang
    shaders/paint_screen.slang
    shaders/post.slang
    shaders/rtao.slang
    shaders/rtbevel.slang
    shaders/rtshadow.slang
    shaders/points.slang
    shaders/points_cull.slang
    shaders/mesh.slang
    shaders/mesh_bindless.slang
    shaders/mesh_wire.slang
    shaders/mesh_bindless_wire.slang
    shaders/tex_blur.slang
    shaders/tex_decode.slang
    shaders/tex_page.slang
    shaders/tex_store.slang
    shaders/tex_uvfill.slang
    shaders/taa.slang
    shaders/vt.slang
    shaders/wire.slang
    INCLUDES
    shaders/mesh.slang
    shaders/scene_color.slang
    shaders/stylised.slang
    shaders/bake_common.slang
    shaders/bake_maps.slang
    shaders/bake_raster.slang
    shaders/bake_trace.slang
    shaders/tex_formats.slang)
