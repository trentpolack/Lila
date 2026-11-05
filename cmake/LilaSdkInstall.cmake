# The extension SDK package: cmake --install <build> --component sdk --prefix <dir>
#   include/lila/...           <lila/...> headers, plus internal/ (the src headers they reach)
#   include/glm, nlohmann, tbb, oneapi
#   lib/Lila*.lib, the TBB import library
#   lib/cmake/LilaSDK/         LilaSDKConfig.cmake (lila_add_extension), version file
#   examples/                  sample extensions, buildable as one project

include(CMakePackageConfigHelpers)

file(STRINGS ${CMAKE_SOURCE_DIR}/src/core/ExtensionContext.h _sdk_major REGEX "#define LILA_SDK_VERSION_MAJOR")
file(STRINGS ${CMAKE_SOURCE_DIR}/src/core/ExtensionContext.h _sdk_minor REGEX "#define LILA_SDK_VERSION_MINOR")
string(REGEX MATCH "[0-9]+$" _sdk_major "${_sdk_major}")
string(REGEX MATCH "[0-9]+$" _sdk_minor "${_sdk_minor}")
set(LILA_SDK_VERSION ${_sdk_major}.${_sdk_minor})
set(LILA_QT_VERSION ${Qt6_VERSION})
set(LILA_SDK_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

get_target_property(_tbb_implib TBB::tbb IMPORTED_IMPLIB_RELEASE)
if(NOT _tbb_implib)
    get_target_property(_tbb_implib TBB::tbb IMPORTED_IMPLIB)
endif()
cmake_path(GET _tbb_implib FILENAME LILA_SDK_TBB_LIB)

set(_sdk_cmake ${CMAKE_BINARY_DIR}/sdk/cmake)
configure_file(${CMAKE_SOURCE_DIR}/cmake/LilaSDKConfig.cmake.in ${_sdk_cmake}/LilaSDKConfig.cmake @ONLY)
write_basic_package_version_file(${_sdk_cmake}/LilaSDKConfigVersion.cmake VERSION ${LILA_SDK_VERSION} COMPATIBILITY SameMajorVersion)

install(CODE "set(LILA_SDK_SOURCE \"${CMAKE_SOURCE_DIR}\")
set(LILA_SDK_DEST \"\${CMAKE_INSTALL_PREFIX}/include/lila\")
include(\"${CMAKE_SOURCE_DIR}/cmake/LilaSdkHeaders.cmake\")" COMPONENT sdk)

find_path(_third_party_inc glm/glm.hpp REQUIRED)
foreach(_dir glm nlohmann tbb oneapi)
    install(DIRECTORY ${_third_party_inc}/${_dir} DESTINATION include COMPONENT sdk)
endforeach()
if(APPLE)
    install(DIRECTORY ${_third_party_inc}/fast_float DESTINATION include COMPONENT sdk)
    install(FILES ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/fast-float/copyright
            DESTINATION share/licenses/fast-float COMPONENT sdk)
endif()

install(FILES $<TARGET_LINKER_FILE:LilaFoundation> $<TARGET_LINKER_FILE:LilaScene> $<TARGET_LINKER_FILE:LilaUI> ${_tbb_implib}
        DESTINATION lib COMPONENT sdk)
install(FILES ${_sdk_cmake}/LilaSDKConfig.cmake ${_sdk_cmake}/LilaSDKConfigVersion.cmake ${CMAKE_SOURCE_DIR}/cmake/LilaExtension.cmake
        ${CMAKE_SOURCE_DIR}/cmake/LilaShaders.cmake
        DESTINATION lib/cmake/LilaSDK COMPONENT sdk)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/extensions/examples/ DESTINATION examples COMPONENT sdk PATTERN __pycache__ EXCLUDE)
