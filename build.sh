#!/usr/bin/env bash
# Configures on first run, then builds every macOS target.
set -euo pipefail

usage() {
    echo "Usage: $0 [--release]"
    echo "Builds macOS in RelWithDebInfo (dev) by default, or Release with --release."
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

if [[ ! -f build/macos-arm64/build.ninja ]]; then
    cmake --preset macos-arm64
fi
cmake --build build/macos-arm64 --config "$config"
