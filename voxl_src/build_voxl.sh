#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLCHAIN=$SCRIPT_DIR/aarch64_toolchain.cmake

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_local_config.sh"

ACADOS_SRC=$ACADOS_SOURCE_DIR
ACADOS_BUILD=$ACADOS_BUILD_DIR

cd "$ACADOS_SRC"
rm -rf "$ACADOS_BUILD"
mkdir -p "$ACADOS_BUILD"
cd "$ACADOS_BUILD"
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DACADOS_WITH_QPOASES=OFF \
    -DACADOS_WITH_HPIPM=ON \
    -DBLASFEO_TARGET=ARMV8A_ARM_CORTEX_A57 \
    -DBUILD_SHARED_LIBS=OFF
make -j"$(nproc)"

cd "$SCRIPT_DIR"
make -f Makefile.voxl clean
make -f Makefile.voxl

make -f Makefile.voxl deploy && adb shell /usr/local/bin/nmpc_udp_solver --benchmark
