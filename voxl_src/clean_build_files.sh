#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_local_config.sh"

paths=(
    "$SCRIPT_DIR/nmpc_pipe_solver"
    "$SCRIPT_DIR/nmpc_pipe_solver.o"
    "$SCRIPT_DIR/nmpc_udp_solver"
    "$SCRIPT_DIR/nmpc_udp_solver.o"
    "$SCRIPT_DIR/motor_test_sender"
    "$SCRIPT_DIR/motor_test_sender.o"
    "$SCRIPT_DIR/single_motor_test_sender"
    "$SCRIPT_DIR/single_motor_test_sender.o"
)

if [ -n "${PX4_FIRMWARE_DIR:-}" ]; then
    paths+=("$PX4_FIRMWARE_DIR/build")
fi

if [ -n "${ACADOS_BUILD_DIR:-}" ]; then
    paths+=("$ACADOS_BUILD_DIR")
fi

for path in "${paths[@]}"; do
    if [ -e "$path" ]; then
        rm -rf "$path"
        echo "Deleted: $path"
    else
        echo "Missing: $path"
    fi
done
