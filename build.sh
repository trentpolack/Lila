#!/usr/bin/env bash
# Configures on first run, then builds every macOS target.
set -euo pipefail

usage() {
    echo "Usage: $0 [--release]"
    echo "Builds macOS in RelWithDebInfo (dev) by default, or Release with --release."
    echo "Uses VCPKG_ROOT when set; otherwise sets up vcpkg in .deps/vcpkg."
}

config=RelWithDebInfo
for arg in "$@"; do
    case "$arg" in
        --release) config=Release ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "$(uname -s)" != Darwin ]]; then
    echo "[failed] build.sh requires macOS; use build.bat on Windows." >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "[failed] CMake is missing. Install it with: brew install cmake" >&2
    exit 1
fi

# vcpkg's Meson port searches for python3, not the versioned python3.14 command.
if ! python3 -I -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))' >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
        python_prefix=$(brew --prefix python@3.14 2>/dev/null || true)
        if [[ -n "$python_prefix" && -x "$python_prefix/libexec/bin/python3" ]]; then
            export PATH="$python_prefix/libexec/bin:$python_prefix/bin:$PATH"
        fi
    fi
fi
if ! python3 -I -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))' >/dev/null 2>&1; then
    echo "[failed] Python 3.14 is required. Install it with: brew install python@3.14" >&2
    echo "For another installation, put its Python 3.14 python3 executable first on PATH." >&2
    exit 1
fi
python_executable=$(command -v python3)
echo "[setup] Using $("$python_executable" --version) at $python_executable"

# Keep using the configured toolchain when building from a new terminal.
if [[ -z "${VCPKG_ROOT:-}" && -f build/macos-arm64/CMakeCache.txt ]]; then
    toolchain=$(sed -n 's/^CMAKE_TOOLCHAIN_FILE:[^=]*=//p' build/macos-arm64/CMakeCache.txt)
    candidate="${toolchain%/scripts/buildsystems/vcpkg.cmake}"
    if [[ -n "$candidate" && -f "$candidate/scripts/buildsystems/vcpkg.cmake" ]]; then
        VCPKG_ROOT="$candidate"
    fi
fi
if [[ -z "${VCPKG_ROOT:-}" ]]; then
    for candidate in "$PWD/.deps/vcpkg" "$HOME/dev/vcpkg" "$HOME/vcpkg"; do
        if [[ -f "$candidate/scripts/buildsystems/vcpkg.cmake" ]]; then
            VCPKG_ROOT="$candidate"
            break
        fi
    done
fi
if [[ -z "${VCPKG_ROOT:-}" ]]; then
    VCPKG_ROOT="$PWD/.deps/vcpkg"
    echo "[setup] Cloning vcpkg into $VCPKG_ROOT"
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
fi
if [[ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "[failed] VCPKG_ROOT is not a vcpkg checkout: $VCPKG_ROOT" >&2
    echo "Set it to your vcpkg checkout, or unset it to use .deps/vcpkg." >&2
    exit 1
fi
VCPKG_ROOT=$(cd -- "$VCPKG_ROOT" && pwd -P)
export VCPKG_ROOT
if [[ ! -x "$VCPKG_ROOT/vcpkg" ]]; then
    echo "[setup] Bootstrapping vcpkg"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
fi

# Match the Windows setup: use vcpkg's Ninja when none is on PATH.
if ! command -v ninja >/dev/null 2>&1; then
    echo "[setup] Fetching Ninja through vcpkg"
    ninja_path=$("$VCPKG_ROOT/vcpkg" fetch ninja --x-stderr-status --vcpkg-root="$VCPKG_ROOT")
    export PATH="$(dirname -- "$ninja_path"):$PATH"
fi

if [[ ! -f build/macos-arm64/build.ninja ]]; then
    cmake --preset macos-arm64 -DCMAKE_MAKE_PROGRAM="$(command -v ninja)" \
        -DPython_EXECUTABLE="$python_executable"
fi
cmake --build build/macos-arm64 --config "$config"
