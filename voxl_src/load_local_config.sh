#!/bin/bash

if [ -z "${SCRIPT_DIR:-}" ] || [ -z "${REPO_DIR:-}" ]; then
    echo "ERROR: load_local_config.sh requires SCRIPT_DIR and REPO_DIR to be set by the caller." >&2
    return 1 2>/dev/null || exit 1
fi

VOXL_LOCAL_CONFIG_PATH=${VOXL_LOCAL_CONFIG_PATH:-"$REPO_DIR/results/voxl_paths.conf"}

if [ -f "$VOXL_LOCAL_CONFIG_PATH" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$VOXL_LOCAL_CONFIG_PATH"
    set +a
fi

if [ -z "${VOXL_PX4_REPO:-}" ] && [ -n "${VOXL_PX4_REPO_DIR:-}" ]; then
    VOXL_PX4_REPO=$VOXL_PX4_REPO_DIR
fi

if [ -z "${ACADOS_SOURCE_DIR:-}" ]; then
    ACADOS_SOURCE_DIR="$SCRIPT_DIR/acados"
fi

if [ -z "${ACADOS_BUILD_DIR:-}" ]; then
    ACADOS_BUILD_DIR="$ACADOS_SOURCE_DIR/build_aarch64"
fi

if [ -z "${PX4_FIRMWARE_DIR:-}" ] && [ -n "${VOXL_PX4_REPO:-}" ]; then
    PX4_FIRMWARE_DIR="$VOXL_PX4_REPO/voxl-px4-dev/px4-firmware"
fi

export VOXL_LOCAL_CONFIG_PATH
export VOXL_PX4_REPO
export PX4_FIRMWARE_DIR
export ACADOS_SOURCE_DIR
export ACADOS_BUILD_DIR
