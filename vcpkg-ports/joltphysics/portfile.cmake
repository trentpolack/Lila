# The upstream port with determinism on: results match bit for bit across machines and compilers. Static, so it links
# into the one extension that uses it and nothing else carries it. Baseline SSE2 and no config-dependent features, so
# the headers a consumer compiles agree with the library in every configuration.
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

# vcpkg may build with a newer MSVC than Lila links with; the newer STL's vectorized helpers would not resolve there.
if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    set(VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS} /D_USE_STD_VECTOR_ALGORITHMS=0")
    set(VCPKG_C_FLAGS "${VCPKG_C_FLAGS} /D_USE_STD_VECTOR_ALGORITHMS=0")
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO jrouwe/JoltPhysics
    REF "v${VERSION}"
    SHA512 bc6f2436bef91a6ffd09eee98186be645f6e9a9b3f65a7a645abf00432ac40ada03320c1fe946661817a94eda372cf8d88e4889991cdd25f1c16ecf9a4486677
    HEAD_REF master
)

string(COMPARE EQUAL "${VCPKG_CRT_LINKAGE}" "static" USE_STATIC_CRT)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/Build"
    OPTIONS
        -DTARGET_UNIT_TESTS=OFF
        -DTARGET_HELLO_WORLD=OFF
        -DTARGET_PERFORMANCE_TEST=OFF
        -DTARGET_SAMPLES=OFF
        -DTARGET_VIEWER=OFF
        -DCROSS_PLATFORM_DETERMINISTIC=ON
        -DINTERPROCEDURAL_OPTIMIZATION=OFF
        -DUSE_STATIC_MSVC_RUNTIME_LIBRARY=${USE_STATIC_CRT}
        -DENABLE_ALL_WARNINGS=OFF
        -DOVERRIDE_CXX_FLAGS=OFF
        -DCPP_RTTI_ENABLED=ON
        -DCPP_EXCEPTIONS_ENABLED=ON
        -DDEBUG_RENDERER_IN_DEBUG_AND_RELEASE=OFF
        -DPROFILER_IN_DEBUG_AND_RELEASE=OFF
        -DFLOATING_POINT_EXCEPTIONS_ENABLED=OFF
        -DUSE_SSE4_1=OFF
        -DUSE_SSE4_2=OFF
        -DUSE_AVX=OFF
        -DUSE_AVX2=OFF
        -DUSE_AVX512=OFF
        -DUSE_LZCNT=OFF
        -DUSE_TZCNT=OFF
        -DUSE_F16C=OFF
        -DUSE_FMADD=OFF
        -DJPH_USE_DX12=OFF
        -DJPH_USE_VK=OFF
        -DJPH_USE_MTL=OFF
    OPTIONS_RELEASE
        -DGENERATE_DEBUG_SYMBOLS=OFF
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_cmake_config_fixup(PACKAGE_NAME Jolt CONFIG_PATH "lib/cmake/Jolt")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
