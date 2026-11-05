# Startup and shutdown shared by both hosts, without the UI.
lila_module(lila_host Scene Host.cpp Host.h)
target_link_libraries(lila_host PUBLIC lila_scene lila_io lila_python lila_mcp lila_extensions lila_vcs lila_link lila_stream lila_fbx lila_gltf lila_maya lila_blend lila_gn
    lila_uv_cmds lila_retopo_cmds lila_retarget lila_clay lila_groom lila_sculpt_cmds lila_vcolor lila_layouts lila_batch)

# lila is the GUI app (no console window); lilabatch is the console app for --batch from shells and CI, without the UI library.
add_executable(lila WIN32 main.cpp $<$<PLATFORM_ID:Windows>:lila.rc>)
target_link_libraries(lila PRIVATE LilaUI)
target_compile_definitions(lila PRIVATE LILA_GUI_SUBSYSTEM)
add_executable(lilabatch batch.cpp $<$<PLATFORM_ID:Windows>:lila.rc>)
target_link_libraries(lilabatch PRIVATE LilaScene)
target_compile_definitions(lilabatch PRIVATE LILA_UI_LIBRARY="$<TARGET_FILE_NAME:LilaUI>")
# Loaded at run time for the capture commands when it is there; lilabatch runs without it.
add_dependencies(lilabatch LilaUI)
add_dependencies(lila lilabatch)
# Opt-in for source builds: cmake --build build --target register_file_types (unregister_file_types to undo).
add_custom_target(register_file_types COMMAND lilabatch -c "registerFileTypes()" DEPENDS lila VERBATIM)
add_custom_target(unregister_file_types COMMAND lilabatch -c "unregisterFileTypes()" VERBATIM)
foreach(exe lila lilabatch)
    target_include_directories(${exe} PRIVATE ${LILA_BUILD_INFO_DIR})
    add_dependencies(${exe} lila_version)
endforeach()
if(WIN32)
    # The crash watchdog and the WER helper it relies on for fail-fast crashes; see src/core/CrashWatch.h.
    add_executable(lilacrash WIN32 lilacrash.cpp)
    target_link_libraries(lilacrash PRIVATE lila_crashkit)
    add_library(lilawer SHARED lilawer.cpp)
    target_link_libraries(lilawer PRIVATE lila_crashkit)
    add_dependencies(lilabatch lilacrash lilawer)
endif()

if(WIN32)
    # Python and Qt next to the exe so it runs without PATH setup.
    cmake_path(GET Python_EXECUTABLE PARENT_PATH _pyhome)
    add_custom_command(TARGET lila POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_pyhome}/python3.dll ${_pyhome}/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.dll $<TARGET_FILE_DIR:lila>
        COMMENT "Deploying the Python runtime")
    lila_deploy_qt(lila)
    lila_bundle_pyside(lila)
    # Help > Licenses reads the notices and license texts beside the exe; installers ship them from there.
    add_custom_command(TARGET lila POST_BUILD
        COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DSHARE_DIR=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share
            -DOUT_DIR=$<TARGET_FILE_DIR:lila> -P ${CMAKE_SOURCE_DIR}/cmake/LilaLicenses.cmake
        COMMENT "Deploying third-party notices and license texts")
    # Help > Record Performance Trace runs it against Lila over 127.0.0.1.
    if(LILA_TRACY AND VCPKG_INSTALLED_DIR)
        set(_tracy_tools ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/tracy)
        add_custom_command(TARGET lila POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tracy_tools}/tracy-capture.exe ${_tracy_tools}/capstone.dll ${_tracy_tools}/zstd.dll
                $<TARGET_FILE_DIR:lila>
            COMMENT "Deploying tracy-capture")
    endif()
endif()
