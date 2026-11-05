#include "core/SystemMemory.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <cstdint>
#else
#include <unistd.h>
#endif

namespace lila
{
size_t physicalMemoryBytes()
{
#ifdef _WIN32
    MEMORYSTATUSEX mem{sizeof(mem)};
    return GlobalMemoryStatusEx(&mem) ? size_t(mem.ullTotalPhys) : 0;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0 ? size_t(bytes) : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && size > 0 ? size_t(pages) * size_t(size) : 0;
#endif
}
} // namespace lila
