add_executable(lila_tests
    main.cpp
    test_graph.cpp
    test_prefs.cpp
    test_formats.cpp
    test_file_types.cpp
    test_scene_version.cpp
    test_references.cpp
    test_meshes.cpp
    test_modeling.cpp
    test_normal_frames.cpp
    test_orient_pivot.cpp
    test_knife.cpp
    test_step_history.cpp
    test_command_history.cpp
    test_conform.cpp
    test_modifiers.cpp
    test_array_modifier.cpp
    test_draw_batches.cpp
    test_modstack.cpp
    test_instancing.cpp
    test_duplicate.cpp
    test_user_attrs.cpp
    test_attr_authoring.cpp
    test_lighting_channels.cpp
    test_component_values.cpp
    test_scene_graph.cpp
    test_unreal_components.cpp
    test_array_group.cpp
    test_docs.cpp
    test_dds.cpp
    test_samples.cpp
    test_user_docs.cpp
    test_transfer.cpp
    test_boolean.cpp
    test_fracture.cpp
    test_gate.cpp
    test_autobevel.cpp
    test_measure.cpp
    test_debug_draw.cpp
    test_surface_snap.cpp
    test_geometry_snap.cpp
    test_imageplane.cpp
    test_hotkeys.cpp
    test_display.cpp
    test_display_aids.cpp
    test_icon_bars.cpp
    test_preselect.cpp
    test_layouts.cpp
    test_themes.cpp
    test_telemetry.cpp
    test_credits.cpp
    test_toolkit.cpp
    test_modeling_defaults.cpp
    test_symmetry.cpp
    test_radial_symmetry.cpp
    test_softselect.cpp
    test_gizmo.cpp
    test_history.cpp
    test_crash.cpp
    test_component_query.cpp
    test_outliner.cpp
    test_outliner_filter.cpp
    test_value_reset.cpp
    test_progress.cpp
    test_scene_scripts.cpp
    test_camera_history.cpp
    test_options.cpp
    test_extrude.cpp
    test_hud.cpp
    test_backdrop.cpp
    test_tearoff.cpp
    test_visibility.cpp
    test_display_layers.cpp
    test_object_sets.cpp
    test_turntable.cpp
    test_create_select.cpp
    test_primitive_axis.cpp
    test_camera.cpp
    test_light.cpp
    test_showcase.cpp
    test_locks.cpp
    test_paths.cpp
    test_project_textures.cpp
    test_text_match.cpp
    test_ui_registries.cpp
    test_graph_backends.cpp
    test_viewport_ext.cpp
    test_select_paths.cpp
    test_isolation.cpp
    test_diagnostics.cpp
    test_smart_join.cpp
    test_batch_rename.cpp
    test_change_stream.cpp
    test_batch_run.cpp
    test_node_paths.cpp
    test_select_strokes.cpp
    test_render_textures.cpp)
target_link_libraries(lila_tests PRIVATE LilaUI doctest::doctest)
find_package(miniz CONFIG REQUIRED)
target_link_libraries(lila_tests PRIVATE miniz::miniz)
# The batch run tests start real lilabatch workers.
add_dependencies(lila_tests lilabatch)
add_test(NAME lila_tests COMMAND lila_tests)
if(APPLE)
    add_test(NAME lila_macos_smoke COMMAND lilabatch --batch ${CMAKE_SOURCE_DIR}/python/tests/test_macos_smoke.py)
    set_tests_properties(lila_macos_smoke PROPERTIES LABELS "macos;gpu"
        ENVIRONMENT "LILA_CONFIG_DIR=${CMAKE_CURRENT_BINARY_DIR}/macos-smoke-config" TIMEOUT 300)
endif()
# Two sources of one target defining the same file-local name break unity builds once a chunk boundary moves.
add_test(NAME lila_unity_clashes COMMAND ${Python_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/scripts/unity_clashes.py --build ${CMAKE_BINARY_DIR})
# Fails when a Qt bump leaves the bundled PySide6 behind or it starts loading its own Qt.
if(LILA_PYSIDE AND WIN32)
    add_test(NAME lila_pyside_python COMMAND lilabatch --batch ${CMAKE_SOURCE_DIR}/python/tests/test_pyside.py
             CONFIGURATIONS Release RelWithDebInfo)
    set_tests_properties(lila_pyside_python PROPERTIES ENVIRONMENT "LILA_CONFIG_DIR=${CMAKE_CURRENT_BINARY_DIR}/pyside_cfg")
endif()
# The telemetry Worker's credits rules; needs Node 22.5+ for node:sqlite.
find_program(LILA_NODE node)
if(LILA_NODE)
    add_test(NAME lila_worker_tests COMMAND ${LILA_NODE} --test test/credits.test.js
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/services/telemetry-worker)
endif()

if(WIN32)
    # Real crashes of a helper process, checked through the watchdog's report; kept out of lila_tests for speed.
    add_executable(lila_crashtest crash_helper.cpp)
    target_link_libraries(lila_crashtest PRIVATE LilaFoundation)
    add_executable(lila_crash_tests test_crash_watch.cpp)
    target_link_libraries(lila_crash_tests PRIVATE LilaFoundation doctest::doctest)
    add_dependencies(lila_crash_tests lila_crashtest lilacrash lilawer)
    add_test(NAME lila_crash_tests COMMAND lila_crash_tests)
endif()

add_subdirectory(math)
add_subdirectory(portability)
add_subdirectory(geo)
add_subdirectory(curve)
add_subdirectory(vcs)
add_subdirectory(editor)
add_subdirectory(docks)
add_subdirectory(extensions)
add_subdirectory(terminal)
add_subdirectory(hk)
add_subdirectory(link)
add_subdirectory(fbx)
add_subdirectory(gltf)
add_subdirectory(seq)
add_subdirectory(bake)
add_subdirectory(tex)
add_subdirectory(anim)
add_subdirectory(media)
add_subdirectory(sim)
if(TARGET lila_puppeteer_extension)
    add_subdirectory(puppeteer_anim)
    add_subdirectory(puppeteer)
endif()
add_subdirectory(uv)
add_subdirectory(rig)
add_subdirectory(rigtemplates)
add_subdirectory(retarget)
add_subdirectory(rigmath)
add_subdirectory(gn)
if(LILA_VOLUMES)
    add_subdirectory(nanovdb)
endif()
add_subdirectory(goz)
add_subdirectory(print)
add_subdirectory(retopo)
add_subdirectory(clay)
add_subdirectory(groom)
add_subdirectory(sculpt)
add_subdirectory(vcolor)
add_subdirectory(brush)
add_subdirectory(deform)
add_subdirectory(materials)
add_subdirectory(stylised)
add_subdirectory(viewport_renderers)
add_subdirectory(fbxskin)
add_subdirectory(maya)
add_subdirectory(blend)
add_subdirectory(agentic)
add_subdirectory(foliage)
add_subdirectory(masonry)
add_subdirectory(tailor)
add_subdirectory(tailor_gpu)
add_subdirectory(trim)
add_subdirectory(stagehand)
add_subdirectory(settlement)
add_subdirectory(groundtiles)
add_subdirectory(terrain)
add_subdirectory(attributes)
if(LILA_USD)
    add_subdirectory(usd)
endif()
