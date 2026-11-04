#!/bin/bash
# propeller_off_test_PX4.sh - Flash PX4 firmware and run the HITL NMPC bench test.
# Run from desktop with VOXL connected via USB (adb).
#
#
# State packing for bench test (no MOCAP):
#   - Attitude / angular_velocity: real, from ICM42688p via EKF2
#   - Position / velocity:         EKF2 estimate from baro+IMU (starts at 0,0,0)
#   - Solver motor state x0[13:17]: internal first-order estimate inside nmpc_pipe_solver
#   - Setpoint:                    hold current position
#
# PX4 firmware path is provided by flash_firmware.sh via results/voxl_paths.conf.
# The mc_nmpc_control module lives under that PX4 firmware checkout.

set -euo pipefail

SOLVER_BIN=/usr/local/bin/nmpc_pipe_solver
SOLVER_UNIT=nmpc-pipe-solver.service
QSHELL_BIN=
PX4_PARAM_BIN=px4-param
DURATION=10
DO_ARM=1
NO_FLASH=0
BUILD_SOLVER=1
RT_PRIO=50
CPU_CORE=7
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLVER_STDOUT_LOG=/tmp/nmpc_solver.log
SOLVER_STDERR_LOG=/tmp/nmpc_solver.err.log
SOLVER_CSV_LOG=/tmp/nmpc_solver_data.csv
T5_NMPC_CONTROL_SAMPLE=
T5_ACTUATOR_MOTORS_SAMPLE=
T5_ACTUATOR_OUTPUTS_SAMPLE=
T5_ACTUATOR_OUTPUTS_DEBUG_SAMPLE=
T5_ESC_STATUS_SAMPLE=
T5_VEHICLE_STATUS_SAMPLE=
T5_VEHICLE_CONTROL_MODE_SAMPLE=
T5_VEHICLE_LAND_DETECTED_SAMPLE=
T5_FAILURE_DETECTOR_STATUS_SAMPLE=
NEURAL_CTRL_WAS_RUNNING=0
LOG_DIR=
HOST_SYNC_TIME_UTC=
HOST_SYNC_EPOCH=
VOXL_SYNC_TIME_UTC=
PX4_LOG_BASELINE=
PX4_LOG_BASELINE_SIZE=
PX4_LOG_SELECTED=
TEST_RESULT_OK=0
DISARM_EVENT_CAPTURED=0
DISARM_EVENT_TIME_S=
ARMED_STATE_SEEN=0
BENCH_PARAM_NAMES=(COM_DISARM_PRFLT COM_DISARM_LAND)
BENCH_PARAM_VALUES=(-1 -1)
BENCH_PARAM_ORIGINAL_VALUES=()
BENCH_PARAMS_APPLIED=0

print_t5_runtime_snapshots() {
    if [ -z "$T5_NMPC_CONTROL_SAMPLE" ] && [ -z "$T5_ACTUATOR_MOTORS_SAMPLE" ] && [ -z "$T5_ACTUATOR_OUTPUTS_SAMPLE" ] && [ -z "$T5_ACTUATOR_OUTPUTS_DEBUG_SAMPLE" ] && [ -z "$T5_ESC_STATUS_SAMPLE" ] && [ -z "$T5_VEHICLE_STATUS_SAMPLE" ] && [ -z "$T5_VEHICLE_CONTROL_MODE_SAMPLE" ] && [ -z "$T5_VEHICLE_LAND_DETECTED_SAMPLE" ] && [ -z "$T5_FAILURE_DETECTOR_STATUS_SAMPLE" ]; then
        return
    fi

    echo "  5s runtime snapshots:"
    if [ -n "$T5_NMPC_CONTROL_SAMPLE" ]; then
        echo "  nmpc_control_data @5s:"
        printf '%s\n' "$T5_NMPC_CONTROL_SAMPLE"
    fi
    if [ -n "$T5_ACTUATOR_MOTORS_SAMPLE" ]; then
        echo ""
        echo "  actuator_motors @5s:"
        printf '%s\n' "$T5_ACTUATOR_MOTORS_SAMPLE"
    fi
    if [ -n "$T5_ACTUATOR_OUTPUTS_SAMPLE" ]; then
        echo ""
        echo "  actuator_outputs @5s:"
        printf '%s\n' "$T5_ACTUATOR_OUTPUTS_SAMPLE"
    fi
    if [ -n "$T5_ACTUATOR_OUTPUTS_DEBUG_SAMPLE" ]; then
        echo ""
        echo "  actuator_outputs_debug @5s:"
        printf '%s\n' "$T5_ACTUATOR_OUTPUTS_DEBUG_SAMPLE"
    fi
    if [ -n "$T5_ESC_STATUS_SAMPLE" ]; then
        echo ""
        echo "  esc_status @5s:"
        printf '%s\n' "$T5_ESC_STATUS_SAMPLE"
    fi
    if [ -n "$T5_VEHICLE_STATUS_SAMPLE" ]; then
        echo ""
        echo "  vehicle_status @5s:"
        printf '%s\n' "$T5_VEHICLE_STATUS_SAMPLE"
    fi
    if [ -n "$T5_VEHICLE_CONTROL_MODE_SAMPLE" ]; then
        echo ""
        echo "  vehicle_control_mode @5s:"
        printf '%s\n' "$T5_VEHICLE_CONTROL_MODE_SAMPLE"
    fi
    if [ -n "$T5_VEHICLE_LAND_DETECTED_SAMPLE" ]; then
        echo ""
        echo "  vehicle_land_detected @5s:"
        printf '%s\n' "$T5_VEHICLE_LAND_DETECTED_SAMPLE"
    fi
    if [ -n "$T5_FAILURE_DETECTOR_STATUS_SAMPLE" ]; then
        echo ""
        echo "  failure_detector_status @5s:"
        printf '%s\n' "$T5_FAILURE_DETECTOR_STATUS_SAMPLE"
    fi
    echo ""
}

stop_conflicting_actuator_publishers() {
    if run_qshell "mc_neural_control status" >/dev/null 2>/dev/null; then
        echo "Stopping mc_neural_control to prevent actuator_motors publisher conflicts..."
        NEURAL_CTRL_WAS_RUNNING=1
        run_qshell "mc_neural_control stop" || true
        sleep 1
    fi
}

run_qshell() {
    adb shell "$QSHELL_BIN $*"
}

run_qshell_timeout() {
    local timeout_seconds=$1
    shift
    timeout "$timeout_seconds" adb shell "$QSHELL_BIN $*"
}

run_px4_param() {
    adb shell "$PX4_PARAM_BIN $*"
}

run_px4_param_timeout() {
    local timeout_seconds=$1
    shift
    timeout "$timeout_seconds" adb shell "$PX4_PARAM_BIN $*"
}

run_listener() {
    adb shell "px4-listener $*"
}

capture_listener_snapshot() {
    local topic_name=$1
    local output_path=$2
    adb shell "px4-listener $topic_name -n 1" > "$output_path" 2>&1 || true
}

capture_run_snapshots() {
    local suffix=$1
    capture_listener_snapshot "nmpc_control_data" "$LOG_DIR/nmpc_control_data${suffix}.txt"
    capture_listener_snapshot "actuator_motors" "$LOG_DIR/actuator_motors${suffix}.txt"
    capture_listener_snapshot "actuator_outputs" "$LOG_DIR/actuator_outputs${suffix}.txt"
    capture_listener_snapshot "actuator_outputs_debug" "$LOG_DIR/actuator_outputs_debug${suffix}.txt"
    capture_listener_snapshot "esc_status" "$LOG_DIR/esc_status${suffix}.txt"
    capture_listener_snapshot "vehicle_status" "$LOG_DIR/vehicle_status${suffix}.txt"
    capture_listener_snapshot "vehicle_control_mode" "$LOG_DIR/vehicle_control_mode${suffix}.txt"
    capture_listener_snapshot "vehicle_land_detected" "$LOG_DIR/vehicle_land_detected${suffix}.txt"
    capture_listener_snapshot "failure_detector_status" "$LOG_DIR/failure_detector_status${suffix}.txt"
}

run_boot_journal() {
    adb shell "journalctl -u voxl-px4 -b $* --no-pager"
}

remote_file_exists() {
    adb shell "if [ -f $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"
}

remote_exec_exists() {
    adb shell "if [ -x $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"
}

remote_cmd_exists() {
    adb shell "if command -v $1 >/dev/null 2>/dev/null; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"
}

remote_pipe_exists() {
    adb shell "if [ -p $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"
}

remote_file_size() {
    adb shell "wc -c < \"$1\"" | tr -d '\r\n '
}

remote_solver_pid() {
    adb shell "pidof $(basename "$SOLVER_BIN")" | tr -d '\r' | awk '{print $1}'
}

stop_solver_unit() {
    adb shell "systemctl is-active $SOLVER_UNIT >/dev/null 2>/dev/null && systemctl stop $SOLVER_UNIT || true"
}

boot_journal_has() {
    run_boot_journal "-n 200" | tr -d '\r' | grep -Fq "$1"
}

detect_qshell_bin() {
    QSHELL_BIN=$(adb shell "if [ -x /usr/bin/qshell ]; then echo /usr/bin/qshell; elif command -v qshell >/dev/null 2>/dev/null; then command -v qshell; elif command -v px4-qshell >/dev/null 2>/dev/null; then command -v px4-qshell; else echo __MISSING__; fi" | tr -d '\r\n')
}

get_param_quiet() {
    local param_name=$1
    local param_value

    param_value=$(run_px4_param_timeout 8 "show -q $param_name" | perl -pe 's/\e\[[0-9;]*[[:alpha:]]//g' | tr -d '\r\n ')

    if [ -z "$param_value" ]; then
        echo "ERROR: Could not read current value for $param_name."
        return 1
    fi

    printf '%s\n' "$param_value"
}

set_and_verify_param() {
    local param_name=$1
    local param_value=$2

    run_px4_param_timeout 8 "set $param_name $param_value"

    if ! run_px4_param_timeout 8 "compare $param_name $param_value"; then
        echo "ERROR: $param_name did not take value $param_value."
        return 1
    fi

    return 0
}

write_bench_param_manifest() {
    local manifest_path=$LOG_DIR/bench_param_overrides.txt

    : > "$manifest_path"

    for param_index in "${!BENCH_PARAM_NAMES[@]}"; do
        printf '%s original=%s override=%s\n' \
            "${BENCH_PARAM_NAMES[$param_index]}" \
            "${BENCH_PARAM_ORIGINAL_VALUES[$param_index]}" \
            "${BENCH_PARAM_VALUES[$param_index]}" >> "$manifest_path"
    done
}

apply_bench_test_params() {
    local param_name
    local param_value
    local current_value

    if [ $DO_ARM -eq 0 ]; then
        return 0
    fi

    echo "Disabling PX4 auto-disarm for bench testing..."

    for param_index in "${!BENCH_PARAM_NAMES[@]}"; do
        param_name=${BENCH_PARAM_NAMES[$param_index]}
        param_value=${BENCH_PARAM_VALUES[$param_index]}
        if ! current_value=$(get_param_quiet "$param_name"); then
            exit 1
        fi

        BENCH_PARAM_ORIGINAL_VALUES[$param_index]=$current_value

        if ! set_and_verify_param "$param_name" "$param_value"; then
            exit 1
        fi
    done

    BENCH_PARAMS_APPLIED=1
    write_bench_param_manifest
}

restore_bench_test_params() {
    local param_name
    local param_value

    if [ $BENCH_PARAMS_APPLIED -eq 0 ]; then
        return 0
    fi

    echo "Restoring PX4 bench-test params..."

    for param_index in "${!BENCH_PARAM_NAMES[@]}"; do
        param_name=${BENCH_PARAM_NAMES[$param_index]}
        param_value=${BENCH_PARAM_ORIGINAL_VALUES[$param_index]}

        if ! set_and_verify_param "$param_name" "$param_value"; then
            return 1
        fi
    done

    BENCH_PARAMS_APPLIED=0
    return 0
}

enable_esc_requested_rpm_logging() {
    local outputs_debug_sample

    echo "Enabling PX4 ESC requested-RPM logging..."
    run_px4_param_timeout 8 "set VOXL_ESC_VLOG 1"
    if ! run_px4_param_timeout 8 "compare VOXL_ESC_VLOG 1"; then
        echo "ERROR: VOXL_ESC_VLOG did not take value 1 after param set."
        exit 1
    fi

    echo "Restarting ESC driver so VOXL_ESC_VLOG takes effect..."
    run_qshell_timeout 8 "voxl_esc stop"
    sleep 1
    run_qshell_timeout 8 "voxl_esc start"
    sleep 2

    outputs_debug_sample=$(run_listener "actuator_outputs_debug -n 1" 2>/dev/null) || outputs_debug_sample=""
    if [ -z "$outputs_debug_sample" ] || printf '%s\n' "$outputs_debug_sample" | grep -q "never published"; then
        echo "ERROR: actuator_outputs_debug is still not publishing after enabling VOXL_ESC_VLOG and restarting ESC drivers."
        exit 1
    fi
}

sync_voxl_utc_clock() {
    HOST_SYNC_TIME_UTC=$(date -u '+%Y-%m-%d %H:%M:%S')
    HOST_SYNC_EPOCH=$(date -u +%s)

    echo "Synchronizing VOXL UTC clock..."
    echo "  Host UTC: $HOST_SYNC_TIME_UTC ($HOST_SYNC_EPOCH)"
    adb shell "date -u -s '$HOST_SYNC_TIME_UTC'" >/dev/null
    VOXL_SYNC_TIME_UTC=$(adb shell "date -u '+%Y-%m-%d %H:%M:%S %s'" | tr -d '\r\n')
    echo "  VOXL UTC: $VOXL_SYNC_TIME_UTC"

    cat > "$LOG_DIR/host_time_sync.txt" <<EOF
host_utc=$HOST_SYNC_TIME_UTC
host_epoch=$HOST_SYNC_EPOCH
EOF

    cat > "$LOG_DIR/voxl_time_sync.txt" <<EOF
voxl_utc=$VOXL_SYNC_TIME_UTC
EOF
}

record_px4_log_baseline() {
    PX4_LOG_BASELINE=$(adb shell "find /data/px4/log -type f -name '*.ulg' | sort | tail -n 1" | tr -d '\r\n')

    if [ -n "$PX4_LOG_BASELINE" ]; then
        PX4_LOG_BASELINE_SIZE=$(remote_file_size "$PX4_LOG_BASELINE")
        echo "PX4 ULog baseline: $PX4_LOG_BASELINE (size=${PX4_LOG_BASELINE_SIZE} bytes)"
    else
        PX4_LOG_BASELINE_SIZE=
        echo "PX4 ULog baseline: none"
    fi
}

resolve_px4_run_log() {
    local latest_ulg
    local latest_size

    latest_ulg=$(adb shell "find /data/px4/log -type f -name '*.ulg' | sort | tail -n 1" | tr -d '\r\n')

    if [ -z "$latest_ulg" ]; then
        echo "ERROR: No PX4 .ulg file found under /data/px4/log after the run."
        return 1
    fi

    latest_size=$(remote_file_size "$latest_ulg")

    if [ -n "$PX4_LOG_BASELINE" ] && [ "$latest_ulg" = "$PX4_LOG_BASELINE" ] && [ "$latest_size" = "$PX4_LOG_BASELINE_SIZE" ]; then
        echo "ERROR: Latest PX4 .ulg path and size are unchanged after the run: $latest_ulg"
        return 1
    fi

    PX4_LOG_SELECTED=$latest_ulg
    echo "Selected PX4 ULog: $PX4_LOG_SELECTED (size=${latest_size} bytes)"
    return 0
}

write_run_manifest() {
    cat > "$LOG_DIR/run_manifest.txt" <<EOF
host_sync_utc=$HOST_SYNC_TIME_UTC
host_sync_epoch=$HOST_SYNC_EPOCH
voxl_sync_utc=$VOXL_SYNC_TIME_UTC
px4_log_baseline=$PX4_LOG_BASELINE
px4_log_baseline_size=$PX4_LOG_BASELINE_SIZE
px4_log_selected=$PX4_LOG_SELECTED
local_px4_log=$LOG_DIR/px4_run.ulg
local_px4_motor_csv=$LOG_DIR/px4_motor_log.csv
local_solver_csv=$LOG_DIR/nmpc_solver_data.csv
EOF
}

dump_solver_diagnostics() {
    echo "Solver startup diagnostics:"
    echo "  Binary metadata:"
    adb shell "ls -l $SOLVER_BIN" || true
    echo "  Dynamic dependencies:"
    adb shell "ldd $SOLVER_BIN" || true
    echo "  Runtime tools:"
    adb shell "which taskset" || true
    adb shell "which chrt" || true
    echo "  Solver service:"
    adb shell "systemctl status $SOLVER_UNIT --no-pager -l" || true
    echo "  qshell binary:"
    if [ -n "$QSHELL_BIN" ] && [ "$QSHELL_BIN" != "__MISSING__" ]; then
        adb shell "ls -l $QSHELL_BIN" || true
    else
        echo "    (missing)"
    fi
    echo "  Solver stdout ($SOLVER_STDOUT_LOG):"
    adb shell "test -s $SOLVER_STDOUT_LOG && cat $SOLVER_STDOUT_LOG" || echo "    (empty or missing)"
    echo "  Solver stderr ($SOLVER_STDERR_LOG):"
    adb shell "test -s $SOLVER_STDERR_LOG && cat $SOLVER_STDERR_LOG" || echo "    (empty or missing)"
}

collect_crash_diagnostics() {
    echo ""
    echo "=== CRASH DIAGNOSTICS ==="
    echo ""
    echo "[1] systemctl status voxl-px4:"
    adb shell "systemctl status voxl-px4 --no-pager -l" || true
    echo ""
    echo "[2] journalctl -u voxl-px4 (last 200 lines):"
    adb shell "journalctl -u voxl-px4 -n 200 --no-pager" || true
    echo ""
    echo "[3] SLPI DSP status:"
    adb shell "pidof slpi_proc" && echo "  slpi_proc is running" || echo "  slpi_proc NOT running"
    adb shell "systemctl status init_slpi --no-pager -l" || true
    echo ""
    echo "[4] DSP syslog (last 50 lines):"
    adb shell "tail -n 50 /var/log/slpi.log" || echo "  /var/log/slpi.log not found"
    adb shell "cat /data/px4/log/latest/px4.log 2>/dev/null | tail -n 50" || true
    echo ""
    echo "[5] dmesg SLPI/DSP/fastrpc errors (last 30):"
    adb shell "dmesg | grep -iE 'slpi|adsp|fastrpc|subsys|err|crash|fault' | tail -n 30" || true
    echo ""
    echo "[6] PX4 process tree:"
    adb shell "ps aux | grep -E 'px4|qshell'" || true
    echo ""
    echo "[7] muORB / shared memory:"
    adb shell "ls -la /dev/shm/muorb*" || echo "  No muorb shared memory files found"
    adb shell "ls -la /data/px4/muorb*" || echo "  No muorb files in /data/px4/"
    echo ""
    echo "[8] SLPI crash dumps:"
    adb shell "ls -lt /data/vendor/ramdump/ 2>/dev/null | head -5" || echo "  No ramdump directory"
    adb shell "ls -lt /data/vendor/tombstones/ 2>/dev/null | head -5" || echo "  No tombstones"
    echo ""
    echo "=== END DIAGNOSTICS ==="
}

ensure_px4_running() {
    echo "Rebooting VOXL for clean state..."
    adb reboot
    echo "  Waiting for VOXL to go offline..."
    sleep 5
    echo "  Waiting for VOXL to come back (up to 120s)..."
    for i in $(seq 1 120); do
        if adb shell "echo __ALIVE__" 2>/dev/null | tr -d '\r' | grep -qx "__ALIVE__"; then
            echo "  VOXL is back after ${i}s."
            break
        fi
        [ $i -eq 120 ] && { echo "ERROR: VOXL did not come back after reboot."; exit 1; }
        sleep 1
    done

    echo "  Waiting 30s for all services to initialize..."
    sleep 30

    PX4_ACTIVE=$(adb shell "systemctl is-active voxl-px4" | tr -d '\r\n') || PX4_ACTIVE=""
    if [ "$PX4_ACTIVE" != "active" ]; then
        echo "ERROR: voxl-px4 not active after reboot (status: '$PX4_ACTIVE')."
        collect_crash_diagnostics
        exit 1
    fi
    echo "  voxl-px4 is active."

    echo "  Checking muORB shared memory..."
    MUORB_COUNT=$(adb shell "ls /dev/shm/muorb* 2>/dev/null | wc -l" | tr -d '\r\n') || MUORB_COUNT="0"
    if [ "$MUORB_COUNT" -eq 0 ]; then
        echo "WARNING: No muORB shared memory files found in /dev/shm/. This may be normal depending on the firmware version."
    else
        echo "  muORB shared memory OK ($MUORB_COUNT files)."
    fi

    echo "  Waiting 15s for PX4 topic graph to initialize..."
    sleep 15

    LISTENER_OUT=$(run_listener "vehicle_status -n 1" 2>/dev/null) || LISTENER_OUT=""
    if echo "$LISTENER_OUT" | grep -q "arming_state"; then
        echo "PX4 is ready."
    else
        echo "ERROR: PX4 did not become ready after reboot."
        echo "Last px4-listener output: $LISTENER_OUT"
        collect_crash_diagnostics
        exit 1
    fi
}

build_and_deploy_solver() {
    echo "Building nmpc_pipe_solver on host..."
    make -C "$SCRIPT_DIR" -f Makefile.voxl pipe
    echo "Deploying nmpc_pipe_solver to VOXL..."
    stop_solver_unit
    adb shell "mkdir -p /usr/local/bin"
    adb push "$SCRIPT_DIR/nmpc_pipe_solver" "$SOLVER_BIN"
    adb shell "chmod +x $SOLVER_BIN"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-arm)   DO_ARM=0;    shift ;;
        --no-flash) NO_FLASH=1;  shift ;;
        --no-build) BUILD_SOLVER=0; shift ;;
        --duration) DURATION=$2; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

LOG_DIR="$SCRIPT_DIR/../cache/workspace_local/voxl_hitl/$(date +%Y_%m_%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== HITL PX4 + NMPC Bench Test ==="
echo "  Solver:   $SOLVER_BIN"
echo "  Build:    $([ $BUILD_SOLVER -eq 1 ] && echo YES || echo NO)"
echo "  Duration: ${DURATION}s"
echo "  Arm:      $([ $DO_ARM -eq 1 ] && echo YES || echo NO)"
echo "  Log dir:  $LOG_DIR"
echo ""

# ── Flash firmware ─────────────────────────────────────────────────────────────
if [ $NO_FLASH -eq 0 ]; then
    bash "$SCRIPT_DIR/flash_firmware.sh"
    echo ""
fi

if [ $BUILD_SOLVER -eq 1 ]; then
    build_and_deploy_solver
    echo ""
fi

# ── Prerequisites (checked on VOXL via adb) ────────────────────────────────────
check_prereqs() {
    local fail=0
    detect_qshell_bin

    if [ $BUILD_SOLVER -eq 0 ] && ! remote_file_exists "$SOLVER_BIN"; then
        echo "ERROR: $SOLVER_BIN not found on VOXL."
        echo "  Either run without --no-build, or deploy manually:"
        echo "  make -f voxl_src/Makefile.voxl pipe && adb push voxl_src/nmpc_pipe_solver $SOLVER_BIN && adb shell chmod +x $SOLVER_BIN"
        fail=1
    fi
    if [ "$QSHELL_BIN" = "__MISSING__" ] || ! remote_exec_exists "$QSHELL_BIN"; then
        echo "ERROR: qshell binary not found on VOXL."
        echo "  Tried: /usr/bin/qshell, \\`command -v qshell\\`, \\`command -v px4-qshell\\`"
        fail=1
    else
        echo "Using qshell binary: $QSHELL_BIN"
    fi
    if ! remote_cmd_exists taskset; then
        echo "ERROR: taskset not found on VOXL."
        fail=1
    fi
    if ! remote_cmd_exists chrt; then
        echo "ERROR: chrt not found on VOXL."
        fail=1
    fi
    if ! remote_cmd_exists pidof; then
        echo "ERROR: pidof not found on VOXL."
        fail=1
    fi
    if ! remote_cmd_exists systemd-run; then
        echo "ERROR: systemd-run not found on VOXL."
        fail=1
    fi
    if ! remote_cmd_exists "$PX4_PARAM_BIN"; then
        echo "ERROR: $PX4_PARAM_BIN not found on VOXL."
        fail=1
    fi
    [ $fail -ne 0 ] && exit 1
    return 0
}
check_prereqs

if [ $DO_ARM -eq 1 ]; then
    echo "WARNING: Motors will spin. Ensure propellers are removed."
fi

cleanup() {
    echo ""
    echo "Cleaning up..."
    NEED_REBOOT=0
    ARTIFACT_FAILURE=0
    if [ -n "$QSHELL_BIN" ] && [ "$QSHELL_BIN" != "__MISSING__" ]; then
        run_qshell "mc_nmpc_control stop" || true
    fi
    if [ $DO_ARM -eq 1 ] && [ -n "$QSHELL_BIN" ] && [ "$QSHELL_BIN" != "__MISSING__" ]; then
        run_qshell "commander disarm -f" || true
        DISARMED=0
        for i in $(seq 1 10); do
            STATUS=$(run_listener "vehicle_status -n 1" 2>/dev/null) || STATUS=""
            if echo "$STATUS" | grep -q "arming_state: .*1"; then
                DISARMED=1
                break
            fi
            sleep 1
        done
        if [ $DISARMED -eq 0 ]; then
            echo "ERROR: vehicle did not disarm during cleanup. Will reboot VOXL after logs are copied."
            NEED_REBOOT=1
        fi
    fi
    if ! restore_bench_test_params; then
        echo "ERROR: Failed to restore one or more PX4 bench-test params."
        ARTIFACT_FAILURE=1
    fi
    stop_solver_unit
    adb shell "journalctl -u voxl-px4 -b -n 200 --no-pager > /tmp/voxl-px4.journal.log" || true

    echo "Copying logs to $LOG_DIR ..."
    adb pull "$SOLVER_STDOUT_LOG" "$LOG_DIR/nmpc_solver.log" || true
    adb pull "$SOLVER_STDERR_LOG" "$LOG_DIR/nmpc_solver.err.log" || true
    adb pull "$SOLVER_CSV_LOG" "$LOG_DIR/nmpc_solver_data.csv" || true
    adb pull /tmp/voxl-px4.journal.log "$LOG_DIR/voxl-px4.journal.log" || true
    capture_run_snapshots ""
    echo "Logs saved to: $LOG_DIR"

    if ! resolve_px4_run_log; then
        ARTIFACT_FAILURE=1
    else
        if ! adb pull "$PX4_LOG_SELECTED" "$LOG_DIR/px4_run.ulg"; then
            echo "ERROR: Failed to pull PX4 ULog from $PX4_LOG_SELECTED"
            ARTIFACT_FAILURE=1
        elif ! python3 "$SCRIPT_DIR/extract_px4_motor_log.py" "$LOG_DIR/px4_run.ulg" "$LOG_DIR/px4_motor_log.csv"; then
            echo "ERROR: Failed to extract PX4 motor CSV from $LOG_DIR/px4_run.ulg"
            ARTIFACT_FAILURE=1
        fi
    fi

    write_run_manifest

    if [ -f "$LOG_DIR/nmpc_solver_data.csv" ] && [ -f "$LOG_DIR/px4_motor_log.csv" ]; then
        echo "Generating diagnostic plots..."
        python3 "$SCRIPT_DIR/plot_hitl.py" "$LOG_DIR/nmpc_solver_data.csv" "$LOG_DIR/px4_motor_log.csv" "$LOG_DIR/hitl_diagnostics.pdf"
        echo "Plots saved to: $LOG_DIR/hitl_diagnostics.pdf"
        echo "Printing CSV values nearest t=5s..."
        python3 - "$LOG_DIR/nmpc_solver_data.csv" <<'PY'
import sys
import pandas as pd

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

if df.empty:
    raise RuntimeError(f"{csv_path} is empty")

t = (df["t_us"] - df["t_us"].iloc[0]) / 1e6
row_index = (t - 5.0).abs().idxmin()
row = df.loc[row_index]
t_row = t.loc[row_index]

print(f"  CSV snapshot near t=5s (row={row_index}, t={t_row:.6f}s):")
for i in range(4):
    command_rps = row[f'command_rps{i}']
    estimated_rps = row[f'estimated_rps{i}']
    print(
        f"    motor {i}: solver_u={row[f'solver_u{i}']:.6f}, "
        f"command_rps={command_rps:.6f}, estimated_rps={estimated_rps:.6f}, "
        f"delta_rps={command_rps - estimated_rps:.6f}"
    )
PY
    elif [ $TEST_RESULT_OK -eq 1 ]; then
        echo "ERROR: Missing CSV inputs required for HITL plotting."
        ARTIFACT_FAILURE=1
    fi

    if [ $NEED_REBOOT -eq 1 ]; then
        adb reboot || true
    fi

    if [ $ARTIFACT_FAILURE -ne 0 ] && [ $TEST_RESULT_OK -eq 1 ]; then
        exit 1
    fi
}
trap cleanup EXIT

# ── Ensure PX4 is running ──────────────────────────────────────────────────────
echo ""
ensure_px4_running
enable_esc_requested_rpm_logging
stop_conflicting_actuator_publishers
apply_bench_test_params
sync_voxl_utc_clock
record_px4_log_baseline

# ── Start nmpc_pipe_solver on VOXL ─────────────────────────────────────────────
echo "Stopping any stale solver..."
stop_solver_unit
adb shell "rm -f $SOLVER_STDOUT_LOG $SOLVER_STDERR_LOG $SOLVER_CSV_LOG"
sleep 1

echo "Starting nmpc_pipe_solver (CPU $CPU_CORE, RT prio $RT_PRIO)..."
adb shell "systemd-run --unit=${SOLVER_UNIT%.service} --collect --property=StandardOutput=file:$SOLVER_STDOUT_LOG --property=StandardError=file:$SOLVER_STDERR_LOG taskset -c $CPU_CORE chrt -f $RT_PRIO $SOLVER_BIN"
sleep 3
SOLVER_PID=$(remote_solver_pid) || SOLVER_PID=""
if [ -z "$SOLVER_PID" ]; then
    echo "ERROR: nmpc_pipe_solver exited immediately."
    dump_solver_diagnostics
    exit 1
fi
echo "Solver running (PID=$SOLVER_PID)"

# ── Ensure NMPC modules are up ────────────────────────────────────────────────
echo ""
echo "Checking NMPC modules..."
NMPC_CTRL_RUNNING=0
for attempt in $(seq 1 5); do
    if run_qshell "mc_nmpc_control status" >/dev/null 2>/dev/null; then
        echo "  mc_nmpc_control: running"
        NMPC_CTRL_RUNNING=1; break
    fi
    echo "  Starting mc_nmpc_control (attempt $attempt/5)..."
    run_qshell "mc_nmpc_control start" || true
    sleep 3
done
if [ $NMPC_CTRL_RUNNING -eq 0 ]; then
    if run_qshell "mc_nmpc_control status" >/dev/null 2>/dev/null; then
        echo "  mc_nmpc_control: running"
    else
        echo "ERROR: mc_nmpc_control failed to start after 5 attempts."
        exit 1
    fi
fi

BRIDGE_RUNNING=0
# Ensure alias exists even if skipping flash
adb shell "cd /usr/bin; /bin/ln -f -s px4 px4-nmpc_io_bridge" || true
for attempt in $(seq 1 5); do
    if remote_pipe_exists /run/mpa/nmpc_control; then
        echo "  nmpc_io_bridge: control pipe ready"
        BRIDGE_RUNNING=1; break
    fi
    echo "  Starting nmpc_io_bridge (attempt $attempt/5)..."
    adb shell "px4-nmpc_io_bridge start" || true
    sleep 3
done
if [ $BRIDGE_RUNNING -eq 0 ]; then
    if remote_pipe_exists /run/mpa/nmpc_control; then
        echo "  nmpc_io_bridge: control pipe ready"
    else
        echo "ERROR: nmpc_io_bridge failed to start after 5 attempts."
        exit 1
    fi
fi

# ── Wait for FIFOs to appear ──────────────────────────────────────────────────
echo ""
echo "Waiting for pipe FIFOs to appear..."
FIFO_OK=0
for i in $(seq 1 10); do
    STATE_OK=0; CTRL_OK=0
    adb shell "test -p /run/mpa/nmpc_state"   2>/dev/null && STATE_OK=1 || true
    adb shell "test -p /run/mpa/nmpc_control" 2>/dev/null && CTRL_OK=1  || true
    if [ $STATE_OK -eq 1 ] && [ $CTRL_OK -eq 1 ]; then
        echo "  FIFOs ready: /run/mpa/nmpc_state  /run/mpa/nmpc_control"
        FIFO_OK=1; break
    fi
    echo "  [$i/10] nmpc_state:$([ $STATE_OK -eq 1 ] && echo OK || echo missing)  nmpc_control:$([ $CTRL_OK -eq 1 ] && echo OK || echo missing)"
    sleep 2
done
[ $FIFO_OK -eq 0 ] && echo "WARNING: FIFOs not both present after 20s."

# ── Wait for EKF2 attitude ────────────────────────────────────────────────────
echo ""
echo "Waiting for EKF2 attitude estimate (up to 30s)..."
ATTITUDE_OK=0
for i in $(seq 1 30); do
    if run_listener "vehicle_attitude -n 1" 2>/dev/null | grep -q "q:"; then
        echo "  EKF2 attitude OK"
        ATTITUDE_OK=1; break
    fi
    printf "  [%d/30]\r" $i
    sleep 1
done
[ $ATTITUDE_OK -eq 0 ] && echo "  WARNING: EKF2 attitude not confirmed. IMU may not be ready."

# ── Arm + Offboard ────────────────────────────────────────────────────────────
if [ $DO_ARM -eq 1 ]; then
    echo ""
    echo "Stopping control_allocator so mc_nmpc_control gets actuator_motors instance 0..."
    run_qshell "control_allocator stop" || true
    sleep 1

    echo "Arming and switching to offboard mode via qshell commander..."
    timeout 8 adb shell "$QSHELL_BIN commander arm -f" || true
    timeout 8 adb shell "$QSHELL_BIN commander mode offboard" || true

    ARMED_OFFBOARD_OK=0
    for i in $(seq 1 10); do
        STATUS=$(run_listener "vehicle_status -n 1" 2>/dev/null) || STATUS=""
        CTRL_MODE=$(run_listener "vehicle_control_mode -n 1" 2>/dev/null) || CTRL_MODE=""
        if echo "$STATUS" | grep -q "arming_state: .*2" && echo "$CTRL_MODE" | grep -q "flag_control_offboard_enabled: .*True"; then
            echo "  Armed and OFFBOARD enabled"
            ARMED_OFFBOARD_OK=1; break
        fi
        sleep 1
    done
    if [ $ARMED_OFFBOARD_OK -eq 0 ]; then
        echo "ERROR: Failed to reach armed offboard state."
        echo "  vehicle_status:"
        printf '%s\n' "$STATUS"
        echo "  vehicle_control_mode:"
        printf '%s\n' "$CTRL_MODE"
        exit 1
    fi
fi

# ── Monitor NMPC output ───────────────────────────────────────────────────────
echo ""
echo "=== Monitoring NMPC for ${DURATION}s ==="
echo "  Current boot journal is checked for first nmpc_state_data / nmpc_control_data publication"
echo "  Solver log is checked for the first completed solve"
echo "  actuator_motors is checked for fresh output after offboard"
echo ""

TSTART=$(date +%s)
STATE_PUBLISHED=0
CONTROL_PUBLISHED=0
SOLVER_SOLVED=0
ACTUATOR_UPDATED=0
ACTUATOR_LAST_SAMPLE=""

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - TSTART))
    [ $ELAPSED -ge $DURATION ] && break

    if [ $STATE_PUBLISHED -eq 0 ] && boot_journal_has "Advertising remote topic nmpc_state_data"; then
        echo "  t=+$(printf "%02d" "$ELAPSED")s  nmpc_state_data published"
        STATE_PUBLISHED=1
    fi
    if [ $CONTROL_PUBLISHED -eq 0 ] && boot_journal_has "Advertising remote topic nmpc_control_data"; then
        echo "  t=+$(printf "%02d" "$ELAPSED")s  nmpc_control_data published"
        CONTROL_PUBLISHED=1
    fi
    if [ $SOLVER_SOLVED -eq 0 ] && adb shell "grep -q 'First solve seq=' $SOLVER_STDOUT_LOG"; then
        SOLVER_LINE=$(adb shell "grep 'First solve seq=' $SOLVER_STDOUT_LOG | head -1" | tr -d '\r') || SOLVER_LINE=""
        echo "  t=+$(printf "%02d" "$ELAPSED")s  $SOLVER_LINE"
        SOLVER_SOLVED=1
    fi
    if [ $ACTUATOR_UPDATED -eq 0 ]; then
        ACTUATOR_SAMPLE=$(run_listener "actuator_motors -n 1" 2>/dev/null) || ACTUATOR_SAMPLE=""
        if printf '%s\n' "$ACTUATOR_SAMPLE" | grep -q 'control:' && ! printf '%s\n' "$ACTUATOR_SAMPLE" | grep -qE 'control: \[0\.0+,'; then
            ACTUATOR_LAST_SAMPLE="$ACTUATOR_SAMPLE"
            echo "  t=+$(printf "%02d" "$ELAPSED")s  actuator_motors has nonzero control"
            ACTUATOR_UPDATED=1
        fi
    fi
    if [ $DO_ARM -eq 1 ] && [ $DISARM_EVENT_CAPTURED -eq 0 ]; then
        STATUS_SAMPLE=$(run_listener "vehicle_status -n 1" 2>/dev/null) || STATUS_SAMPLE=""
        if echo "$STATUS_SAMPLE" | grep -q "arming_state: .*2"; then
            ARMED_STATE_SEEN=1
        elif [ $ARMED_STATE_SEEN -eq 1 ]; then
            DISARM_EVENT_CAPTURED=1
            DISARM_EVENT_TIME_S=$ELAPSED
            echo "  t=+$(printf "%02d" "$ELAPSED")s  disarm detected, capturing event snapshots"
            capture_run_snapshots "_disarm_event"
        fi
    fi
    if [ $ELAPSED -ge 5 ] && [ -z "$T5_NMPC_CONTROL_SAMPLE" ]; then
        T5_NMPC_CONTROL_SAMPLE=$(run_listener "nmpc_control_data -n 1" 2>/dev/null) || T5_NMPC_CONTROL_SAMPLE=""
        T5_ACTUATOR_MOTORS_SAMPLE=$(run_listener "actuator_motors -n 1" 2>/dev/null) || T5_ACTUATOR_MOTORS_SAMPLE=""
        T5_ACTUATOR_OUTPUTS_SAMPLE=$(run_listener "actuator_outputs -n 1" 2>/dev/null) || T5_ACTUATOR_OUTPUTS_SAMPLE=""
        T5_ACTUATOR_OUTPUTS_DEBUG_SAMPLE=$(run_listener "actuator_outputs_debug -n 1" 2>/dev/null) || T5_ACTUATOR_OUTPUTS_DEBUG_SAMPLE=""
        T5_ESC_STATUS_SAMPLE=$(run_listener "esc_status -n 1" 2>/dev/null) || T5_ESC_STATUS_SAMPLE=""
        T5_VEHICLE_STATUS_SAMPLE=$(run_listener "vehicle_status -n 1" 2>/dev/null) || T5_VEHICLE_STATUS_SAMPLE=""
        T5_VEHICLE_CONTROL_MODE_SAMPLE=$(run_listener "vehicle_control_mode -n 1" 2>/dev/null) || T5_VEHICLE_CONTROL_MODE_SAMPLE=""
        T5_VEHICLE_LAND_DETECTED_SAMPLE=$(run_listener "vehicle_land_detected -n 1" 2>/dev/null) || T5_VEHICLE_LAND_DETECTED_SAMPLE=""
        T5_FAILURE_DETECTOR_STATUS_SAMPLE=$(run_listener "failure_detector_status -n 1" 2>/dev/null) || T5_FAILURE_DETECTOR_STATUS_SAMPLE=""
        echo "  t=+05s  captured runtime topic snapshots"
    fi
    sleep 1
done

NMPC_CONTROL_SAMPLE=$(run_listener "nmpc_control_data -n 1" 2>/dev/null) || NMPC_CONTROL_SAMPLE=""
ACTUATOR_OUTPUTS_SAMPLE=$(run_listener "actuator_outputs -n 1" 2>/dev/null) || ACTUATOR_OUTPUTS_SAMPLE=""
ESC_STATUS_SAMPLE=$(run_listener "esc_status -n 1" 2>/dev/null) || ESC_STATUS_SAMPLE=""

# ── Results ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Test Results ==="
echo "  Duration:              ${DURATION}s"
echo "  nmpc_state_data seen:  $([ $STATE_PUBLISHED -eq 1 ] && echo YES || echo NO)"
echo "  Solver solved:         $([ $SOLVER_SOLVED -eq 1 ] && echo YES || echo NO)"
echo "  nmpc_control_data seen: $([ $CONTROL_PUBLISHED -eq 1 ] && echo YES || echo NO)"
echo "  Actuator update seen:  $([ $ACTUATOR_UPDATED -eq 1 ] && echo YES || echo NO)"
echo "  Disarm event seen:     $([ $DISARM_EVENT_CAPTURED -eq 1 ] && echo YES || echo NO)"
if [ $DISARM_EVENT_CAPTURED -eq 1 ]; then
    echo "  Disarm event time:     ${DISARM_EVENT_TIME_S}s"
fi

if [ $CONTROL_PUBLISHED -eq 1 ]; then
    TEST_RESULT_OK=1
    echo ""
    echo "  PASS: NMPC loop reached the return path and published nmpc_control_data."
    echo ""
    print_t5_runtime_snapshots
    if [ -n "$NMPC_CONTROL_SAMPLE" ]; then
        echo "  nmpc_control_data:"
        printf '%s\n' "$NMPC_CONTROL_SAMPLE"
    fi
    if [ -n "$ACTUATOR_OUTPUTS_SAMPLE" ]; then
        echo ""
        echo "  actuator_outputs:"
        printf '%s\n' "$ACTUATOR_OUTPUTS_SAMPLE"
    fi
    if [ -n "$ESC_STATUS_SAMPLE" ]; then
        echo ""
        echo "  esc_status:"
        printf '%s\n' "$ESC_STATUS_SAMPLE"
    fi
    echo ""
    if [ -n "$ACTUATOR_LAST_SAMPLE" ]; then
        printf '%s\n' "$ACTUATOR_LAST_SAMPLE"
    else
        run_listener "actuator_motors -n 1" 2>/dev/null || echo "  (could not read actuator_motors)"
    fi
else
    echo ""
    echo "  FAIL: control return path was not confirmed."
    echo ""
    print_t5_runtime_snapshots
    echo "  Diagnostics:"
    echo "    1. Solver out:   adb shell cat $SOLVER_STDOUT_LOG"
    echo "    2. Solver err:   adb shell cat $SOLVER_STDERR_LOG"
    echo "    3. Bridge pipe:  adb shell ls -la /run/mpa/nmpc_control"
    echo "    4. Module:       adb shell $QSHELL_BIN mc_nmpc_control status"
    echo "    5. PX4 journal:  adb shell journalctl -u voxl-px4 -b -n 120 --no-pager"
    echo "    6. State topic:  adb shell journalctl -u voxl-px4 -b --no-pager | grep -F nmpc_state_data"
    echo "    7. Actuators:    adb shell px4-listener actuator_motors -n 1"
    echo "    8. FIFOs:        adb shell ls -la /run/mpa/nmpc_*"
    echo "    9. Vehicle mode: adb shell px4-listener vehicle_control_mode -n 1"
    echo "   10. Vehicle stat: adb shell px4-listener vehicle_status -n 1"
    exit 1
fi
