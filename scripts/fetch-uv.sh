#!/usr/bin/env bash
# Put uv (the Python installer RapidRetouch uses to set up its AI engine on
# first launch) where the release build bundles it: src-tauri/binaries/uv-<target>.
# Usage: scripts/fetch-uv.sh <rust target triple>, e.g. x86_64-unknown-linux-gnu.
set -euo pipefail
UV_VERSION="0.12.19"  # the version the engine's setup is tested with
target="${1:?usage: fetch-uv.sh <target triple>}"
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$here/src-tauri/binaries"
mkdir -p "$out"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
case "$target" in
  *windows*)
    curl -fsSL "$base/uv-$target.zip" -o "$tmp/uv.zip"
    unzip -q "$tmp/uv.zip" -d "$tmp"
    cp "$(find "$tmp" -name uv.exe | head -1)" "$out/uv-$target.exe"
    ;;
  *)
    curl -fsSL "$base/uv-$target.tar.gz" -o "$tmp/uv.tar.gz"
    tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
    cp "$(find "$tmp" -type f -name uv | head -1)" "$out/uv-$target"
    chmod +x "$out/uv-$target"
    ;;
esac
echo "uv $UV_VERSION -> $out"
