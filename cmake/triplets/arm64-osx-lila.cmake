set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
# Qt plugins and Slang are loaded at runtime. Keep one shared Qt instance across the host and extensions.
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_DEPLOYMENT_TARGET "14.0")
