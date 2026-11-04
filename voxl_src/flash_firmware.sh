#!/bin/bash
# flash_firmware.sh - Build and flash PX4 firmware to VOXL via adb.
# Run from desktop with VOXL connected via USB.
# The whole script body lives in main() so bash parses the entire file before
# executing anything; otherwise editing this file during a long run makes bash
# misread it mid-execution.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOLVER_BIN=/usr/local/bin/nmpc_pipe_solver
SOLVER_UNIT=nmpc-pipe-solver.service
NMPC_CONFIG_LOCAL_PATH=
NMPC_CONFIG_REMOTE_PATH=/home/root/nmpc_controller_cfg.yaml
ROS_HELPER_DIR=
PYTHON_BIN=
CURRENT_STEP="initialization"

report_failure() {
    local exit_code=$?
    echo ""
    echo "========================================"
    echo "  FLASH FAILED"
    echo "========================================"
    echo "Step: $CURRENT_STEP"
    echo "Exit code: $exit_code"
    echo "The VOXL may still be running the previously flashed firmware."
    exit "$exit_code"
}

trap report_failure ERR

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_local_config.sh"

require_cmd() {
    local cmd_name=$1
    if ! command -v "$cmd_name" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd_name"
        exit 1
    fi
}

require_single_adb_device() {
    local devices
    devices="$(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')"

    local device_count
    device_count="$(printf '%s\n' "$devices" | sed '/^$/d' | wc -l)"

    if [ "$device_count" -ne 1 ]; then
        echo "ERROR: Expected exactly one connected ADB device, found $device_count."
        adb devices -l
        exit 1
    fi
}

detect_python_bin() {
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
        return
    fi

    echo "ERROR: Neither python nor python3 is available on PATH."
    exit 1
}

deploy_recovered_ros_helpers() {
    local helper_dir=$1

    if [ ! -d "$helper_dir" ]; then
        echo "ERROR: Recovered ROS helper directory does not exist: $helper_dir"
        exit 1
    fi

    local required_files=(
        mocap_pane1_init.sh
        qualisys_odom_to_mavros_odometry.py
        qualisys_odom_to_px4_mavlink.py
        record_qualisys_odom_csv.py
    )
    local file_name=
    for file_name in "${required_files[@]}"; do
        if [ ! -f "$helper_dir/$file_name" ]; then
            echo "ERROR: Missing recovered ROS helper file: $helper_dir/$file_name"
            exit 1
        fi
    done

    echo "Deploying recovered ROS helper scripts to VOXL..."
    adb shell "mkdir -p /home/root"

    adb push "$helper_dir/mocap_pane1_init.sh" /home/root/mocap_pane1_init.sh
    adb shell "chown root:root /home/root/mocap_pane1_init.sh && chmod 0644 /home/root/mocap_pane1_init.sh"

    adb push "$helper_dir/qualisys_odom_to_mavros_odometry.py" /home/root/qualisys_odom_to_mavros_odometry.py
    adb shell "id system >/dev/null 2>&1 && chown system:system /home/root/qualisys_odom_to_mavros_odometry.py && chmod 0755 /home/root/qualisys_odom_to_mavros_odometry.py"

    adb push "$helper_dir/qualisys_odom_to_px4_mavlink.py" /home/root/qualisys_odom_to_px4_mavlink.py
    adb shell "chown root:root /home/root/qualisys_odom_to_px4_mavlink.py && chmod 0755 /home/root/qualisys_odom_to_px4_mavlink.py"

    adb push "$helper_dir/record_qualisys_odom_csv.py" /home/root/record_qualisys_odom_csv.py
    adb shell "chown root:root /home/root/record_qualisys_odom_csv.py && chmod 0755 /home/root/record_qualisys_odom_csv.py"
}

require_px4_build_prereqs() {
    local build_script=$1

    if grep -Eq '(^|[[:space:]])docker([[:space:]]|$)' "$build_script"; then
        if ! command -v docker >/dev/null 2>&1; then
            echo "ERROR: PX4 build script requires Docker, but 'docker' is not available on PATH."
            echo "Install Docker or update $build_script to use your container runtime."
            exit 1
        fi
    fi
}

main() {

CURRENT_STEP="cleaning build files"
"$SCRIPT_DIR/clean_build_files.sh"

CURRENT_STEP="building acados and solver deps"
"$SCRIPT_DIR/build_voxl.sh"

require_cmd adb
require_cmd make
require_cmd aarch64-linux-gnu-gcc
detect_python_bin
ROS_HELPER_DIR="$SCRIPT_DIR/drone_ros_files"
NMPC_CONFIG_LOCAL_PATH="$REPO_DIR/configs/nmpc_controller_cfg.yaml"

if [ -z "${VOXL_PX4_REPO:-}" ]; then
    echo "ERROR: VOXL_PX4_REPO is not set."
    echo "Create $VOXL_LOCAL_CONFIG_PATH from voxl_src/voxl_paths.conf.example and set VOXL_PX4_REPO."
    exit 1
fi

if [ ! -d "$VOXL_PX4_REPO" ]; then
    echo "ERROR: VOXL_PX4_REPO does not exist: $VOXL_PX4_REPO"
    exit 1
fi

if [ -z "${PX4_FIRMWARE_DIR:-}" ] || [ ! -d "$PX4_FIRMWARE_DIR" ]; then
    echo "ERROR: PX4_FIRMWARE_DIR does not exist: ${PX4_FIRMWARE_DIR:-<unset>}"
    exit 1
fi

if [ ! -d "$ACADOS_SOURCE_DIR" ]; then
    echo "ERROR: ACADOS_SOURCE_DIR does not exist: $ACADOS_SOURCE_DIR"
    exit 1
fi

if [ ! -d "$ACADOS_BUILD_DIR" ]; then
    echo "ERROR: ACADOS_BUILD_DIR does not exist: $ACADOS_BUILD_DIR"
    echo "Build acados first, e.g. via voxl_src/build_voxl.sh."
    exit 1
fi

if [ ! -f "$NMPC_CONFIG_LOCAL_PATH" ]; then
    echo "ERROR: Missing NMPC config file: $NMPC_CONFIG_LOCAL_PATH"
    exit 1
fi

if [ ! -f "$VOXL_PX4_REPO/build_firmware.sh" ]; then
    echo "ERROR: Missing PX4 build script: $VOXL_PX4_REPO/build_firmware.sh"
    exit 1
fi

require_px4_build_prereqs "$VOXL_PX4_REPO/build_firmware.sh"

CURRENT_STEP="building PX4 firmware"
echo "Building PX4 firmware..."
cd "$VOXL_PX4_REPO"
./build_firmware.sh

CURRENT_STEP="generating NMPC solver code"
echo "Generating NMPC solver code..."
cd "$REPO_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/generate_nmpc_solver.py"

CURRENT_STEP="building nmpc_pipe_solver"
echo "Building nmpc_pipe_solver..."
make -C "$SCRIPT_DIR" -f Makefile.voxl pipe

CURRENT_STEP="stopping voxl-px4 on target"
echo "Stopping voxl-px4 on VOXL..."
require_single_adb_device
adb shell systemctl stop voxl-px4
sleep 5

CURRENT_STEP="flashing PX4 firmware to target"
echo "Flashing PX4 firmware to VOXL..."
require_single_adb_device
cd "$PX4_FIRMWARE_DIR"
bash -e boards/modalai/voxl2/scripts/install-voxl.sh

CURRENT_STEP="deploying nmpc_pipe_solver"
echo "Deploying nmpc_pipe_solver to VOXL..."
require_single_adb_device
adb shell "if systemctl is-active --quiet $SOLVER_UNIT; then systemctl stop $SOLVER_UNIT; fi"
adb shell "mkdir -p /usr/local/bin"
adb push "$SCRIPT_DIR/nmpc_pipe_solver" "$SOLVER_BIN"
adb shell "chmod +x $SOLVER_BIN"

CURRENT_STEP="deploying NMPC runtime config"
echo "Deploying NMPC runtime config to VOXL..."
adb shell "mkdir -p /home/root"
adb push "$NMPC_CONFIG_LOCAL_PATH" "$NMPC_CONFIG_REMOTE_PATH"
adb shell "chown root:root $NMPC_CONFIG_REMOTE_PATH && chmod 0644 $NMPC_CONFIG_REMOTE_PATH"

CURRENT_STEP="deploying ROS helper scripts"
deploy_recovered_ros_helpers "$ROS_HELPER_DIR"

CURRENT_STEP="starting voxl-px4 on target"
echo "Starting voxl-px4 on VOXL..."
adb shell systemctl start voxl-px4
trap - ERR

}

main "$@"
