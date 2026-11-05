# lila_add_extension(<target> ID <Id> MANIFEST <extension.json> SOURCES ... [RESOURCES <dir>] [UI] [OUTPUT_DIR <dir>])
#
# Builds a C++ extension as a package: <OUTPUT_DIR>/<Id>.lilaext/ holding the manifest, the library and the resources.
# OUTPUT_DIR defaults to <bin>/<config>/extensions, where Lila looks. UI links LilaUI as well as LilaScene.
#
# Extensions may include only the SDK's <lila/...> headers and their own: the check below fails the configure step when
# a source includes Lila's internal headers or the target adds Lila's src/ folder to its include path.
function(lila_add_extension target)
    cmake_parse_arguments(ARG "UI" "ID;MANIFEST;RESOURCES;OUTPUT_DIR" "SOURCES" ${ARGN})
    if(NOT ARG_ID OR NOT ARG_MANIFEST OR NOT ARG_SOURCES)
        message(FATAL_ERROR "lila_add_extension(${target}): give ID, MANIFEST and SOURCES")
    endif()
    if(NOT ARG_OUTPUT_DIR AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(ARG_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/extensions)
    elseif(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR ${CMAKE_BINARY_DIR}/$<CONFIG>/extensions)
    endif()
    if(ARG_UI AND LILA_SDK_NO_QT)
        message(FATAL_ERROR "lila_add_extension(${target}): UI extensions need Qt ${LILA_QT_VERSION}; add it to CMAKE_PREFIX_PATH")
    endif()
    set(package ${ARG_OUTPUT_DIR}/${ARG_ID}.lilaext)

    add_library(${target} MODULE ${ARG_SOURCES})
    set_property(GLOBAL APPEND PROPERTY LILA_EXTENSION_TARGETS ${target})
    target_link_libraries(${target} PRIVATE LilaScene)
    if(ARG_UI)
        target_link_libraries(${target} PRIVATE LilaUI)
    endif()
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${ARG_ID}
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${package})
    if(APPLE)
        # CMake MODULE libraries otherwise use .so, while Lila's manifests and loader expect .dylib.
        set_target_properties(${target} PROPERTIES SUFFIX ".dylib")
    endif()
    cmake_path(ABSOLUTE_PATH ARG_MANIFEST BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE manifest)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${manifest} ${package}/extension.json
        COMMENT "Packaging ${ARG_ID}.lilaext")
    if(ARG_RESOURCES)
        cmake_path(ABSOLUTE_PATH ARG_RESOURCES BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE resources)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${resources} ${package}/resources)
    endif()

    lila_check_extension_sources(${target} ${ARG_SOURCES})
    # Deferred to the end of the directory so include paths added after this call are seen; EVAL bakes in the name.
    cmake_language(EVAL CODE "cmake_language(DEFER CALL lila_check_extension_includes [[${target}]])")
endfunction()

# lila_add_resource_extension(<target> ID <Id> MANIFEST <extension.json> RESOURCES <dir> [OUTPUT_DIR <dir>])
#
# Packages a resource-only extension (materials, nodes, presets; no library, no Python): <OUTPUT_DIR>/<Id>.lilaext/
# holding the manifest and the resources, refreshed on every build.
function(lila_add_resource_extension target)
    cmake_parse_arguments(ARG "" "ID;MANIFEST;RESOURCES;OUTPUT_DIR" "" ${ARGN})
    if(NOT ARG_ID OR NOT ARG_MANIFEST OR NOT ARG_RESOURCES)
        message(FATAL_ERROR "lila_add_resource_extension(${target}): give ID, MANIFEST and RESOURCES")
    endif()
    if(NOT ARG_OUTPUT_DIR AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(ARG_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/extensions)
    elseif(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR ${CMAKE_BINARY_DIR}/$<CONFIG>/extensions)
    endif()
    set(package ${ARG_OUTPUT_DIR}/${ARG_ID}.lilaext)
    cmake_path(ABSOLUTE_PATH ARG_MANIFEST BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE manifest)
    cmake_path(ABSOLUTE_PATH ARG_RESOURCES BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE resources)
    add_custom_target(${target} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${package}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${manifest} ${package}/extension.json
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${resources} ${package}/resources
        COMMENT "Packaging ${ARG_ID}.lilaext")
endfunction()

# An extension's sources may include <lila/...>, the standard library, third-party headers and their own files only.
function(lila_check_extension_sources target)
    set(internal core ops math scene geo deform io fbx uv gn link maya retopo sculpt vcolor vcs mcp python extensions render ui app)
    list(JOIN internal "|" internal_re)
    foreach(src ${ARGN})
        cmake_path(ABSOLUTE_PATH src BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE file)
        if(NOT EXISTS ${file})
            continue()
        endif()
        file(STRINGS ${file} includes REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"]")
        foreach(line ${includes})
            if(line MATCHES "[<\"](lila/internal/|(\\.\\./)*(src/)?(${internal_re})/)[^>\"]*[>\"]")
                message(FATAL_ERROR "${file}: extension ${target} includes Lila's internal header ${CMAKE_MATCH_0}; include only <lila/...> SDK headers")
            endif()
        endforeach()
    endforeach()
endfunction()

function(lila_check_extension_includes target)
    get_target_property(dirs ${target} INCLUDE_DIRECTORIES)
    if(NOT dirs)
        return()
    endif()
    cmake_path(SET src NORMALIZE ${CMAKE_SOURCE_DIR}/src)
    foreach(dir ${dirs})
        cmake_path(SET d NORMALIZE ${dir})
        if(d STREQUAL src OR d STREQUAL "${src}/")
            message(FATAL_ERROR "extension ${target} adds Lila's src/ to its include path; include only <lila/...> SDK headers")
        endif()
    endforeach()
endfunction()
