#pragma once

#include <filesystem>
#include <string_view>

namespace lila::python
{
/** A runtime override wins; otherwise use a bundled stdlib, then the build's actual Python prefix. */
inline std::filesystem::path runtimeHome(const std::filesystem::path& executableDir, const std::filesystem::path& configured,
                                         std::string_view stdlibSubdir, const char* overrideHome)
{
    if (overrideHome && *overrideHome)
        return std::filesystem::u8path(overrideHome);
    const auto bundled = executableDir / "python-runtime";
    std::error_code ec;
    return std::filesystem::is_directory(bundled / stdlibSubdir / "encodings", ec) ? bundled : configured;
}
} // namespace lila::python
