# The crash report and capture code, without Lila's dependencies, for the lilacrash watchdog and the lilawer WER helper.
# LilaFoundation compiles the same sources for itself.
set(LILA_CRASHKIT_SOURCES CrashReport.cpp CrashCapture.cpp CrashWatch.cpp CrashUpload.cpp)
set(LILA_CRASHKIT_LIBS nlohmann_json::nlohmann_json)
if(WIN32)
    set(LILA_CRASHKIT_LIBS nlohmann_json::nlohmann_json dbghelp version wevtapi shcore dxgi winhttp)
endif()
add_library(lila_crashkit STATIC ${LILA_CRASHKIT_SOURCES})
target_include_directories(lila_crashkit PUBLIC ${CMAKE_SOURCE_DIR}/src)
# The shared PCH sees TBB through the vcpkg include path; the helpers must not link it.
target_compile_definitions(lila_crashkit PUBLIC LILA_CRASHKIT_STATIC __TBB_NO_IMPLICIT_LINKAGE=1)
target_link_libraries(lila_crashkit PUBLIC ${LILA_CRASHKIT_LIBS})

lila_module(lila_core Foundation
    ${LILA_CRASHKIT_SOURCES}
    Achievements.cpp
    CrashHandler.cpp
    Dds.cpp
    Events.cpp
    FileBytes.cpp
    ExtensionContext.cpp
    ExtensionManifest.cpp
    ImageIO.cpp
    Log.cpp
    MainQueue.cpp
    Paths.cpp
    PrefStore.cpp
    Profiler.cpp
    Progress.cpp
    ProjectTextures.cpp
    Registry.cpp
    ResourceKey.cpp
    SystemMemory.cpp
    TextMatch.cpp
    UsdPlugins.cpp
    Version.cpp
    VideoWriter.cpp
    Workspace.cpp)

find_package(miniz CONFIG REQUIRED)
find_package(WebP CONFIG REQUIRED)
target_link_libraries(lila_core PRIVATE miniz::miniz WebP::webpdecoder)
target_include_directories(lila_core SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/bcdec)
if(WIN32)
    target_link_libraries(lila_core PRIVATE mfplat mfreadwrite mfuuid ole32 iphlpapi ws2_32)
endif()
target_include_directories(lila_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/sdk/include)
target_link_libraries(lila_core PUBLIC nlohmann_json::nlohmann_json TBB::tbb glm::glm ${LILA_CRASHKIT_LIBS})
if(APPLE)
    target_link_libraries(lila_core PUBLIC FastFloat::fast_float)
endif()
# Out of the unity batch: the stamp changes with every commit and should recompile only this file.
set_source_files_properties(Version.cpp PROPERTIES INCLUDE_DIRECTORIES ${LILA_BUILD_INFO_DIR} SKIP_UNITY_BUILD_INCLUSION ON SKIP_PRECOMPILE_HEADERS ON)
add_dependencies(lila_core lila_version)
# windows.h stays out of the unity batch.
set_source_files_properties(SystemMemory.cpp Profiler.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
if(LILA_TRACY)
    target_link_libraries(lila_core PUBLIC Tracy::TracyClient)
    target_compile_definitions(lila_core PUBLIC LILA_TRACY)
endif()
target_compile_definitions(lila_core PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL
    LILA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
