#include "app/Host.h"
#include "core/CrashHandler.h"
#include "core/Paths.h"
#include "core/Version.h"
#include "extensions/Library.h"

#include <cstdio>

using namespace lila;

namespace
{
/**
 * The capture commands render through the UI library, which lilabatch does not link: load it when it sits next to
 * the exe, and run without those commands when it does not.
 */
void loadUiCommands()
{
    const auto dll = paths::executableDir() / LILA_UI_LIBRARY;
    std::string error;
    auto ui = extensions::Library::open(dll, &error);
    if (!ui)
    {
        std::fprintf(stderr, "lilabatch: %s did not load (%s): captureView, GPU bake and the other UI commands are unavailable\n",
                     dll.string().c_str(), error.c_str());
        return;
    }
    using RegisterFn = void (*)();
    // Registered commands and shutdown hooks outlive this local handle.
    ui->leak();
    if (auto fn = reinterpret_cast<RegisterFn>(ui->symbol("lilaUiRegisterBatchCommands")))
        fn();
    else
        std::fprintf(stderr, "lilabatch: %s has no batch command entry point\n", dll.string().c_str());
}
} // namespace

int main(int argc, char** argv)
{
    crash::install();
    host::Options opts = host::parse(argc, argv);
    opts.batch = true;
    if (opts.printVersion)
    {
        std::printf("Lila %s\n", version::full().c_str());
        return 0;
    }

    host::startCore();
    loadUiCommands();
    host::openStartupWorkspace(opts);
    const bool pythonOk = host::startPython(opts);
    const int rc = host::runBatch(opts, pythonOk);
    host::shutdown();
    return rc;
}
