lila_module(lila_tex Scene
    BrushPresets.cpp
    Commands.cpp
    Evaluate.cpp
    Generators.cpp
    Graph.cpp
    Natives.cpp
    GraphFiles.cpp
    Interop.cpp
    Interop.h
    LayerCommands.cpp
    Layers.cpp
    Library.cpp
    LibraryCommands.cpp
    Lint.cpp
    NodeDef.cpp
    PageStore.cpp
    PaintBuffer.cpp
    PaintParticles.cpp
    PaintStrokes.cpp
    PaintStrokes.h
    PaintSurface.cpp
    PaintSymmetry.cpp
    PaintSurface.h
    Plan.cpp
    Ports.cpp
    SetOps.cpp
    SlangRt.cpp
    SmartMaterial.cpp
    TextureSet.cpp
    Types.cpp)

find_package(slang CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
target_link_libraries(lila_tex PUBLIC lila_scene lila_geo PRIVATE slang::slang zstd::libzstd lila_uv)

# The CPU evaluator JIT-compiles through slang-llvm, which Slang loads at runtime (so app-local deployment misses it),
# along with its standard modules. Copy them beside the binaries.
if(WIN32)
    set(_slang_bin ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin)
    set(_slang_runtime ${_slang_bin}/slang.dll ${_slang_bin}/slang-compiler.dll ${_slang_bin}/slang-llvm.dll
        ${_slang_bin}/slang-glslang.dll ${_slang_bin}/slang-glsl-module.dll ${_slang_bin}/slang-rt.dll ${_slang_bin}/slang.slang)
else()
    set(_slang_bin ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib)
    file(GLOB _slang_runtime CONFIGURE_DEPENDS ${_slang_bin}/libslang*.dylib ${_slang_bin}/libslang*.so*
        ${_slang_bin}/*.slang)
    if(NOT _slang_runtime)
        message(FATAL_ERROR "No Slang runtime found in ${_slang_bin}")
    endif()
endif()
file(GLOB _slang_modules LIST_DIRECTORIES true ${_slang_bin}/slang-standard-module-*)
add_custom_target(lila_slang_runtime
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_slang_runtime}
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>
    COMMENT "Copying the Slang runtime")
foreach(_m ${_slang_modules})
    get_filename_component(_mn ${_m} NAME)
    add_custom_command(TARGET lila_slang_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_m} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/${_mn})
endforeach()
add_dependencies(lila_tex lila_slang_runtime)
