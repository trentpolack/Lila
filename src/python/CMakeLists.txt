nanobind_build_library(nanobind-static)
if(MSVC)
    # nanobind's own sources shadow names under /W4.
    target_compile_options(nanobind-static PRIVATE /wd4457 /wd4459)
endif()

lila_module(lila_python Scene
    Bindings.cpp
    MathBindings.cpp
    DebugDrawBindings.cpp
    SceneBindings.cpp
    MeshBindings.cpp
    ExtensionBindings.cpp
    RegistryBindings.cpp
    Python.cpp)

target_link_libraries(lila_python PUBLIC lila_scene lila_math lila_debug lila_vcolor lila_stream PRIVATE lila_rigmath nanobind-static Python::Python)
nanobind_compile_options(lila_python)
nanobind_link_options(lila_python)

target_compile_definitions(lila_python PRIVATE LILA_PYTHON_HOME="${LILA_PYTHON_HOME}"
    LILA_PYTHON_STDLIB_SUBDIR="${LILA_PYTHON_STDLIB_SUBDIR}")
