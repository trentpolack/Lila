# Lila

> [!CAUTION]
> **Under construction: pre-alpha.** Expect crashes, missing features, lost work, and an excessive and rapid quantity of commits. Testers welcome: report issues on the [Lila Dev Discord](https://discord.gg/kx3jxkksmf) (open until release).
> <br>Access is currently only available via `git clone` until ready for alpha-testing at earliest (TBD).

3D app for game assets: Maya-style workflow, modern multithreaded core, Unreal-first pipeline.

## Get it running (Windows)

Needs Visual Studio 2022 (C++ workload with its default MSVC 14.44, CMake 3.28+, vcpkg). Qt comes from vcpkg with the other dependencies; no Qt installer or Qt account. An installed Python 3.14 is found in `C:\Python314` or python.org's default folders, otherwise the first build downloads it into `.deps`. Defaults live in `scripts/env.cmd`; set `LILA_VS`, `LILA_PYTHON_DIR` or `VCPKG_ROOT` before calling it to override. Lila and its vcpkg dependencies all build with the MSVC toolset pinned in `cmake/triplets/x64-windows.cmake`; `scripts\toolchain.cmd` shows what the build gets. sccache and lld-link are used when found, and are optional.

```
build.bat                              # configures on first run (vcpkg deps; Qt and USD make this long), then builds
build\bin\RelWithDebInfo\lila.exe      # or: lila.exe <folder>  /  lila.exe scene.lila
```

Don't create the `build` folder by hand; the first `build.bat` makes it. `build.bat <dir> <config>` builds into another folder or configuration.

To skip the heaviest dependencies, configure once before `build.bat` with `cmake --preset windows -DLILA_USD=OFF -DLILA_VOLUMES=OFF` (either or both). `LILA_USD=OFF` drops OpenUSD: USD files, live stages, MaterialX and Hydra viewport renderers; Lila's own viewport doesn't need it. `LILA_VOLUMES=OFF` drops OpenVDB and the VDB volume nodes.

To build an installer instead, also install Inno Setup 6 (`winget install JRSoftware.InnoSetup`), then:

```
scripts\build.cmd release
scripts\installer\build.cmd            # writes build-installer\Lila-<version>-Setup.exe
```

Tests: `scripts\t.cmd build RelWithDebInfo -LE slow`. `-L slow` runs the slow suites (every shipped material evaluated and linted, about 20 minutes).

### Unreal

1. Copy `ue/LilaLink` into `<Project>/Plugins/LilaLink` and build the project.
2. Open the editor; its log shows `LogLilaLink: Listening on 127.0.0.1:8741`.
3. In Lila: select meshes, File > Send to Unreal, enter content paths. From then on every save re-exports and reimports the mapped objects.

## Get it running (macOS, experimental)

Apple Silicon developer build: macOS 14+, a current Xcode/Command Line Tools installation, CMake 3.28+, Ninja,
vcpkg, ARM64 Python 3.14 and a current MoltenVK with Vulkan 1.3 support. Qt 6.11 comes from vcpkg. This is a
source-tree build; keep the checkout and dependency installations in place. The macOS port still needs hardware
validation before it can be considered supported.

Install the Homebrew dependencies, then build:

```sh
brew install cmake python@3.14 molten-vk

# From the Lila checkout; configure/build every target, including extensions.
./build.sh                             # dev build (RelWithDebInfo)
# ./build.sh --release                 # release build; executable in bin/Release instead
build/macos-arm64/bin/RelWithDebInfo/lila
```

`build.sh` uses `VCPKG_ROOT` when set, or reuses the configured checkout, `.deps/vcpkg`, `$HOME/dev/vcpkg` or
`$HOME/vcpkg`. Otherwise it clones and bootstraps vcpkg into the ignored `.deps/vcpkg` directory. It also fetches
Ninja through vcpkg if Ninja is missing from `PATH`. The first run needs network access and builds the dependencies.
The script requires Python 3.14 and, when needed, adds Homebrew's `python@3.14/libexec/bin` to its own `PATH` so
vcpkg/Meson uses it instead of macOS's system Python. The same interpreter is passed to Lila's CMake configuration.
It then configures with the `macos-arm64` preset and builds every target; subsequent builds reuse that configuration.
For custom dependency paths, configure manually with `-DPython_EXECUTABLE=...` and `-DLILA_MOLTENVK=...`, setting
`VCPKG_ROOT` to your checkout (or `$PWD/.deps/vcpkg`) first.

The preset uses dynamic ARM64 dependencies and initially disables USD, volumes, PySide and Tracy. A Vulkan SDK
installation can supply MoltenVK instead: set `LILA_MOLTENVK` to its `lib/libMoltenVK.dylib`. The renderer loads that
configured library directly, so a global Vulkan loader/ICD setup is unnecessary. This direct path does not load
Vulkan validation layers. Unsupported wide lines render at one pixel; unsupported ray queries use compute traversal.

First-milestone checks:

```sh
# An edited cube must survive save/reopen, produce a shaded capture, and complete a GPU compute bake.
# This fails on missing Vulkan or CPU fallback and prints a temporary directory containing the results.
ctest --preset macos-dev -R '^lila_macos_smoke$'

# Existing numerical GPU-vs-CPU bake comparison; require a GPU instead of silently skipping.
LILA_REQUIRE_GPU=1 LILA_VK_NO_RT=1 \
  build/macos-arm64/bin/RelWithDebInfo/lila_bake_gpu_tests --test-case='*bump plane*'

# Remaining default checks, excluding the repository's slow and known-failing suites.
ctest --preset macos-dev
```

Also open the smoke test's `edited-cube.lila` in the GUI. Confirm that the shaded cube appears, orbit and edit it,
save/reopen it, resize the window, move it between Retina/non-Retina displays if available, and undock/redock the
viewport. Confirm that drawing and selection remain aligned, then inspect `shaded-cube.png` and the baked maps.
An offscreen capture alone does not validate native window presentation. There is no distributable `.app` bundle,
macOS video encoder, or macOS crash watchdog in this milestone; use image sequences for video output.

## Troubleshooting

- **vcpkg fails with `Operation REMOVE_DUPLICATES not recognized`:** the vcpkg that ships with Visual Studio is older than Lila's pinned ports. Update Visual Studio 2022 in the Visual Studio Installer (or point `VCPKG_ROOT` at a fresh clone of github.com/microsoft/vcpkg after running its `bootstrap-vcpkg.bat`), delete the `build` folder and run `build.bat` again.
- **`could not open 'msvcrtd.lib'` / `oldnames.lib`, or the compiler shows as Clang:** `vcvars64.bat` failed, so CMake ran without MSVC. Run `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"` in a Command Prompt and read its `[ERROR...]` lines; usually the MSVC toolset is incomplete, fixed by Visual Studio Installer > Modify > "MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)", or Repair. Then delete the `build` folder and run `build.bat` again; `vcpkg_installed` is kept, so the deps don't rebuild.
- **`Could not set up Python` or vcpkg fails with `usd[python] needs a Python install`:** the Python download failed (network or proxy). Install Python 3.14 from python.org (not the Microsoft Store version; it installs alongside other versions) and `set LILA_PYTHON_DIR=<its folder>` before `build.bat`.
- **Link error LNK1168 while building:** Lila is running. Close it (or `taskkill /IM lila.exe /F`).
- **Viewport stays black:** check the Script Output for a `vulkan:` line. Lila needs a Vulkan 1.3 GPU and driver.
- **Python errors at startup:** `LILA_PYTHON_DIR` must point at the exact Python 3.14 the build used.
- **Script editor shows no completions:** the first open installs basedpyright and ruff into `%LOCALAPPDATA%\Lila\tools`; the status line says when it's ready. It needs pip and a network connection once.
- **Unreal doesn't connect:** run `unrealStatus(scan=True)` in the script editor; check Editor Preferences > Plugins > Lila Link is enabled; the plugin tries ports 8741 to 8748.
- **Save fails on a read-only file:** the file is under Perforce and not checked out; Lila asks to check it out (pref `sourceControl` `/autoCheckout`: `always`, `prompt`, `never`).
- **Crashes:** every crash, including heap corruption and other fail-fast crashes, writes `%APPDATA%\Lila\crashes\<time>_<pid>\report.txt` (what crashed, every thread's stack, the last log lines and commands, system and GPU) plus `minidump.dmp`. Help > Copy Last Crash Report puts the report on the clipboard. No report after a fail-fast crash means `lilawer.dll` or `lilacrash.exe` is missing next to `lila.exe`, or Windows Error Reporting is disabled.

## Tuning

Prefs are JSON in `%APPDATA%\Lila\prefs` (Help > Open Prefs Folder), written as you change things and reloaded when edited outside Lila. A workspace's `.lila/` folder overrides them per project.

- `general.json`: `/scene/upAxis` (`"y"` or `"z"`), `/scene/defaultFormat` (`"binary"` or `"ascii"`), `/autosave/intervalSeconds` (default 30), `/viewport/env/*` (HDRI path, rotation, blur, background), `/manipulator/*` (size, spaces, axis tints; `/manipulator/snap/{move,rotate,scale}/{enabled,step}` and `/manipulator/snap/move/absolute` for the viewport toolbar's increment snapping), `/softSelect/*`, `/references/autoReload`, `/terminal/shell`, `/crash/keepReports` (crash folders kept, default 20), `/crash/fullDump` (full-memory minidumps, default off).
- `hotkeys.json`: every binding, by context (Windows > Hotkey Editor edits it).
- `markingMenus/*.json`, `shelves/*.json`: menus and shelves (Windows > Marking Menu Editor; right-click a shelf).
- `naming.json` (per workspace): naming conventions; joints snake_case, everything else PascalCase by default.
- `sourceControl.json`: Perforce/git settings.
- `unreal.json` (per workspace): Unreal projects and asset mappings.
- MCP for Claude: toggle with the MCP button in the status bar; connection details in `%APPDATA%\Lila\mcp.json`.
