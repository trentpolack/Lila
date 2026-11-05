#pragma once

#include "core/Export.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace lila::paths
{
namespace fs = std::filesystem;

/** File timestamp as seconds since the Unix epoch, independent of the filesystem clock's epoch. */
inline int64_t unixSeconds(fs::file_time_type time)
{
#if defined(_WIN32)
    const auto systemTime = std::chrono::clock_cast<std::chrono::system_clock>(time);
#else
    const auto systemTime = fs::file_time_type::clock::to_sys(time);
#endif
    return std::chrono::duration_cast<std::chrono::seconds>(systemTime.time_since_epoch()).count();
}

/** %APPDATA%/Lila, ~/.config/Lila, ~/Library/Application Support/Lila. Created on first call. */
LILA_FOUNDATION_API fs::path userConfigDir();

LILA_FOUNDATION_API fs::path executableDir();

/** Read-only app data: resources/ and python/. Source tree in dev builds, next to the exe when installed. */
LILA_FOUNDATION_API fs::path resourcesDir();
LILA_FOUNDATION_API fs::path pythonDir();

LILA_FOUNDATION_API fs::path fromUtf8(std::string_view utf8);
/** Generic (forward slash) UTF-8 form. */
LILA_FOUNDATION_API std::string toUtf8(const fs::path& p);

/** Folder of the open scene file, empty for an untitled scene. io keeps it current. */
LILA_FOUNDATION_API const fs::path& currentSceneDir();
LILA_FOUNDATION_API void setCurrentSceneDir(const fs::path& dir);

/**
 * path relative to base, lexically, with ".." where it leaves base and case blind on Windows. Empty when no relative
 * form exists: either path is relative, or they sit on different drives / root names.
 */
LILA_FOUNDATION_API fs::path relativeTo(const fs::path& path, const fs::path& base);

/**
 * How a file path is written to a scene or workspace file: relative to the workspace root when one is open and on the
 * path's drive, else to sceneDir, ".." allowed, forward slashes. Absolute only when neither has a relative form
 * (another drive) or there is no base. A relative path is taken as stored for the open scene, so it is resolved first and re-expressed against
 * sceneDir, which is where the file being written lives. Empty and URL-like strings pass through.
 */
LILA_FOUNDATION_API std::string storedPath(const fs::path& path, const fs::path& sceneDir = currentSceneDir());

/**
 * A stored path made usable: absolute ones as written, relative ones against the workspace root, then sceneDir. The
 * first base where the file exists wins, then the first where its folder does (a sequence pattern names no file),
 * then the first base.
 */
LILA_FOUNDATION_API fs::path resolvePath(std::string_view stored, const fs::path& sceneDir = currentSceneDir());
} // namespace lila::paths
