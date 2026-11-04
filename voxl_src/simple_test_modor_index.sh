#!/bin/bash

set -euo pipefail

SIMPLE_TEST_BIN=/usr/local/bin/single_motor_test_sender
SIMPLE_TEST_UNIT=simple-motor-test-sender.service
QSHELL_BIN=
PX4_PARAM_BIN=px4-param
TEST_RPS=500
TEST_DURATION_SECONDS=12
BUILD=1
RT_PRIO=50
CPU_CORE=7
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STDOUT_LOG=/tmp/simple_motor_test.log
STDERR_LOG=/tmp/simple_motor_test.err.log
LOG_DIR=
TEST_PARAM_NAMES=(COM_DISARM_PRFLT COM_DISARM_LAND VOXL_ESC_RPM_MIN VOXL_ESC_RPM_MAX)
TEST_PARAM_VALUES=(-1 -1 0 30000)
TEST_PARAM_ORIGINAL_VALUES=()
TEST_PARAMS_APPLIED=0
ESC_MOTOR_MAPPING=(0 1 2 3)
ESC_FUNC_PARAM_NAMES=(VOXL_ESC_FUNC1 VOXL_ESC_FUNC2 VOXL_ESC_FUNC3 VOXL_ESC_FUNC4)
ESC_FUNC_CURRENT_VALUES=()
ESC_FUNC_NEW_VALUES=()
ESC_FUNC_PARAMS_APPLIED=0
NMPC_TO_ESC_MAP=(0 0 0 0)

for esc_ch in 0 1 2 3; do
    NMPC_TO_ESC_MAP[${ESC_MOTOR_MAPPING[$esc_ch]}]=$esc_ch
done

run_qshell() { adb shell "$QSHELL_BIN $*"; }
run_qshell_timeout() { local timeout_seconds=$1; shift; timeout "$timeout_seconds" adb shell "$QSHELL_BIN $*"; }
run_px4_param_timeout() { local timeout_seconds=$1; shift; timeout "$timeout_seconds" adb shell "$PX4_PARAM_BIN $*"; }
run_listener() { adb shell "px4-listener $*"; }
remote_exec_exists() { adb shell "if [ -x $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"; }
remote_cmd_exists() { adb shell "if command -v $1 >/dev/null 2>/dev/null; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"; }
remote_pipe_exists() { adb shell "if [ -p $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"; }
remote_solver_pid() { adb shell "pidof nmpc_pipe_solver" | tr -d '\r' | awk '{print $1}'; }

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
}

write_test_param_manifest() {
    local manifest_path=$LOG_DIR/test_param_overrides.txt
    : > "$manifest_path"

    for param_index in "${!TEST_PARAM_NAMES[@]}"; do
        printf '%s original=%s override=%s\n' \
            "${TEST_PARAM_NAMES[$param_index]}" \
            "${TEST_PARAM_ORIGINAL_VALUES[$param_index]}" \
            "${TEST_PARAM_VALUES[$param_index]}" >> "$manifest_path"
    done
}

apply_test_params() {
    local param_name
    local param_value
    local current_value

    echo "Applying PX4 single-motor-test params..."

    for param_index in "${!TEST_PARAM_NAMES[@]}"; do
        param_name=${TEST_PARAM_NAMES[$param_index]}
        param_value=${TEST_PARAM_VALUES[$param_index]}

        if ! current_value=$(get_param_quiet "$param_name"); then
            exit 1
        fi
        TEST_PARAM_ORIGINAL_VALUES[$param_index]=$current_value

        if ! set_and_verify_param "$param_name" "$param_value"; then
            exit 1
        fi
    done

    TEST_PARAMS_APPLIED=1
    write_test_param_manifest
}

restore_test_params() {
    local param_name
    local param_value

    if [ $TEST_PARAMS_APPLIED -eq 0 ]; then
        return 0
    fi

    echo "Restoring PX4 single-motor-test params..."

    for param_index in "${!TEST_PARAM_NAMES[@]}"; do
        param_name=${TEST_PARAM_NAMES[$param_index]}
        param_value=${TEST_PARAM_ORIGINAL_VALUES[$param_index]}
        if ! set_and_verify_param "$param_name" "$param_value"; then
            return 1
        fi
    done

    TEST_PARAMS_APPLIED=0
}

write_esc_motor_mapping_manifest() {
    local manifest_path=$LOG_DIR/esc_motor_mapping_overrides.txt
    : > "$manifest_path"

    for param_index in "${!ESC_FUNC_PARAM_NAMES[@]}"; do
        printf '%s original=%s override=%s\n' \
            "${ESC_FUNC_PARAM_NAMES[$param_index]}" \
            "${ESC_FUNC_CURRENT_VALUES[$param_index]}" \
            "${ESC_FUNC_NEW_VALUES[$param_index]}" >> "$manifest_path"
    done
}

configure_esc_motor_mapping() {
    local esc_ch
    local current_value
    local nmpc_motor
    local param_name
    local param_value

    if [ ${#ESC_MOTOR_MAPPING[@]} -ne 4 ]; then
        echo "ERROR: ESC_MOTOR_MAPPING must contain 4 values."
        exit 1
    fi

    ESC_FUNC_CURRENT_VALUES=()
    ESC_FUNC_NEW_VALUES=()

    for esc_ch in 0 1 2 3; do
        param_name=${ESC_FUNC_PARAM_NAMES[$esc_ch]}
        if ! current_value=$(get_param_quiet "$param_name"); then
            exit 1
        fi
        ESC_FUNC_CURRENT_VALUES[$esc_ch]=$current_value

        nmpc_motor=${ESC_MOTOR_MAPPING[$esc_ch]}
        ESC_FUNC_NEW_VALUES[$esc_ch]=$((101 + nmpc_motor))
    done

    echo "Applying ESC motor mapping (VOXL_ESC_FUNC1-4):"
    for esc_ch in 0 1 2 3; do
        echo "  ESC channel $esc_ch -> NMPC motor ${ESC_MOTOR_MAPPING[$esc_ch]} (${ESC_FUNC_PARAM_NAMES[$esc_ch]}=${ESC_FUNC_NEW_VALUES[$esc_ch]})"
    done

    for esc_ch in 0 1 2 3; do
        param_name=${ESC_FUNC_PARAM_NAMES[$esc_ch]}
        param_value=${ESC_FUNC_NEW_VALUES[$esc_ch]}
        if ! set_and_verify_param "$param_name" "$param_value"; then
            exit 1
        fi
    done

    run_px4_param_timeout 8 "save"
    ESC_FUNC_PARAMS_APPLIED=1
    write_esc_motor_mapping_manifest
}

collect_crash_diagnostics() {
    echo ""
    echo "=== CRASH DIAGNOSTICS ==="
    echo "[1] systemctl status voxl-px4:"
    adb shell "systemctl status voxl-px4 --no-pager -l"
    echo ""
    echo "[2] journalctl -u voxl-px4 (last 100 lines):"
    adb shell "journalctl -u voxl-px4 -n 100 --no-pager"
    echo "=== END DIAGNOSTICS ==="
}

cleanup() {
    local exit_code=$?

    trap - EXIT
    set +e
    echo ""
    echo "Cleaning up..."
    run_qshell "mc_nmpc_control stop"
    run_qshell "commander disarm -f"
    adb shell "systemctl stop ${SIMPLE_TEST_UNIT%.service}"
    adb shell "systemctl start nmpc-pipe-solver" 2>/dev/null || true

    if [ $TEST_PARAMS_APPLIED -eq 1 ]; then
        restore_test_params
    fi

    echo "Copying logs to $LOG_DIR ..."
    adb pull "$STDOUT_LOG" "$LOG_DIR/simple_motor_test.log" 2>/dev/null
    adb pull "$STDERR_LOG" "$LOG_DIR/simple_motor_test.err.log" 2>/dev/null
    adb shell "journalctl -u voxl-px4 -b -n 200 --no-pager > /tmp/voxl-px4.simple_motor_test.journal.log"
    adb pull /tmp/voxl-px4.simple_motor_test.journal.log "$LOG_DIR/voxl-px4.journal.log" 2>/dev/null
    echo "Logs saved to: $LOG_DIR"

    exit $exit_code
}

if [ $# -ne 0 ]; then
    echo "ERROR: simple_test_modor_index.sh does not accept CLI arguments."
    exit 1
fi

TOTAL_DURATION=$((TEST_DURATION_SECONDS + 10))
LOG_DIR="$SCRIPT_DIR/../cache/workspace_local/voxl_simple_motor_test/$(date +%Y_%m_%d_%H%M%S)"
mkdir -p "$LOG_DIR"
trap cleanup EXIT

echo "=== Simple Motor Index Test ==="
echo "  RPS:       $TEST_RPS"
echo "  Duration:  ${TEST_DURATION_SECONDS}s"
echo "  Total:     ${TOTAL_DURATION}s"
echo "  Build:     $([ $BUILD -eq 1 ] && echo YES || echo NO)"
echo "  Log dir:   $LOG_DIR"
echo ""
echo "  Will spin NMPC motor 0 only."
echo "  Observe which physical motor spins."
echo ""

if [ $BUILD -eq 1 ]; then
    echo "Building single_motor_test_sender..."
    make -C "$SCRIPT_DIR" -f Makefile.voxl simple-motor-test
    echo "Deploying to VOXL..."
    adb shell "mkdir -p /usr/local/bin"
    adb push "$SCRIPT_DIR/single_motor_test_sender" "$SIMPLE_TEST_BIN"
    adb shell "chmod +x $SIMPLE_TEST_BIN"
    echo ""
fi

detect_qshell_bin
if [ "$QSHELL_BIN" = "__MISSING__" ] || ! remote_exec_exists "$QSHELL_BIN"; then
    echo "ERROR: qshell binary not found on VOXL."
    exit 1
fi
echo "Using qshell binary: $QSHELL_BIN"

if [ $BUILD -eq 0 ] && ! remote_exec_exists "$SIMPLE_TEST_BIN"; then
    echo "ERROR: $SIMPLE_TEST_BIN not found on VOXL."
    exit 1
fi

for cmd in taskset chrt pidof systemd-run; do
    if ! remote_cmd_exists "$cmd"; then
        echo "ERROR: $cmd not found on VOXL."
        exit 1
    fi
done

if ! remote_cmd_exists "$PX4_PARAM_BIN"; then
    echo "ERROR: $PX4_PARAM_BIN not found on VOXL."
    exit 1
fi

echo "WARNING: Motors will spin. Ensure propellers are removed."
echo ""

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

    if [ $i -eq 120 ]; then
        echo "ERROR: VOXL did not come back after reboot."
        exit 1
    fi

    sleep 1
done

echo "  Waiting 30s for services to initialize..."
sleep 30

PX4_ACTIVE=$(adb shell "systemctl is-active voxl-px4" | tr -d '\r\n')
if [ "$PX4_ACTIVE" != "active" ]; then
    echo "ERROR: voxl-px4 not active after reboot (status: '$PX4_ACTIVE')."
    collect_crash_diagnostics
    exit 1
fi
echo "  voxl-px4 is active."

echo "  Waiting 15s for PX4 topic graph..."
sleep 15

LISTENER_OUT=$(run_listener "vehicle_status -n 1" 2>/dev/null)
if ! echo "$LISTENER_OUT" | grep -q "arming_state"; then
    echo "ERROR: PX4 did not become ready after reboot."
    collect_crash_diagnostics
    exit 1
fi
echo "PX4 is ready."

echo ""
configure_esc_motor_mapping
echo ""
echo "Disabling auto-disarm and motor RPM minima for single-motor test..."
apply_test_params

echo ""
if run_qshell "mc_neural_control status" >/dev/null 2>/dev/null; then
    echo "Stopping mc_neural_control..."
    run_qshell "mc_neural_control stop"
    sleep 1
fi

echo "Starting NMPC modules..."
for attempt in $(seq 1 5); do
    if run_qshell "mc_nmpc_control status" >/dev/null 2>/dev/null; then
        echo "  mc_nmpc_control: running"
        break
    fi

    echo "  Starting mc_nmpc_control (attempt $attempt/5)..."
    run_qshell "mc_nmpc_control start"
    sleep 3
done

if ! run_qshell "mc_nmpc_control status" >/dev/null 2>/dev/null; then
    echo "ERROR: mc_nmpc_control failed to start."
    exit 1
fi

adb shell "cd /usr/bin; /bin/ln -f -s px4 px4-nmpc_io_bridge"
for attempt in $(seq 1 5); do
    if remote_pipe_exists /run/mpa/nmpc_control; then
        echo "  nmpc_io_bridge: pipe ready"
        break
    fi

    echo "  Starting nmpc_io_bridge (attempt $attempt/5)..."
    adb shell "px4-nmpc_io_bridge start"
    sleep 3
done

if ! remote_pipe_exists /run/mpa/nmpc_control; then
    echo "ERROR: nmpc_io_bridge failed to start."
    exit 1
fi

echo "Waiting for FIFOs..."
FIFO_OK=0
for i in $(seq 1 10); do
    STATE_OK=0
    CTRL_OK=0
    if adb shell "test -p /run/mpa/nmpc_state" 2>/dev/null; then
        STATE_OK=1
    fi
    if adb shell "test -p /run/mpa/nmpc_control" 2>/dev/null; then
        CTRL_OK=1
    fi

    if [ $STATE_OK -eq 1 ] && [ $CTRL_OK -eq 1 ]; then
        echo "  FIFOs ready."
        FIFO_OK=1
        break
    fi

    sleep 2
done

if [ $FIFO_OK -eq 0 ]; then
    echo "ERROR: FIFOs not present after 20s."
    exit 1
fi

echo ""
echo "Stopping nmpc_pipe_solver to prevent competing motor commands..."
adb shell "systemctl stop nmpc-pipe-solver"
sleep 1
SOLVER_PID=$(remote_solver_pid) || SOLVER_PID=""
if [ -n "$SOLVER_PID" ]; then
    echo "ERROR: nmpc_pipe_solver is still running (PID=$SOLVER_PID)."
    exit 1
fi

adb shell "rm -f $STDOUT_LOG $STDERR_LOG"
echo "Starting single_motor_test_sender (motor=0, rps=$TEST_RPS, duration=${TEST_DURATION_SECONDS}s)..."
adb shell "systemd-run --unit=${SIMPLE_TEST_UNIT%.service} --collect --property=StandardOutput=file:$STDOUT_LOG --property=StandardError=file:$STDERR_LOG taskset -c $CPU_CORE chrt -f $RT_PRIO $SIMPLE_TEST_BIN $TEST_RPS $TEST_DURATION_SECONDS"
sleep 3
SENDER_PID=$(adb shell "pidof single_motor_test_sender" | tr -d '\r' | awk '{print $1}')
if [ -z "$SENDER_PID" ]; then
    echo "ERROR: single_motor_test_sender exited immediately."
    echo "  stdout:"
    adb shell "cat $STDOUT_LOG"
    echo "  stderr:"
    adb shell "cat $STDERR_LOG"
    exit 1
fi
echo "  single_motor_test_sender running (PID=$SENDER_PID)"

echo "Waiting for EKF2 attitude..."
for i in $(seq 1 30); do
    if run_listener "vehicle_attitude -n 1" 2>/dev/null | grep -q "q:"; then
        echo "  EKF2 attitude OK"
        break
    fi

    if [ $i -eq 30 ]; then
        echo "ERROR: EKF2 attitude not confirmed."
        exit 1
    fi

    sleep 1
done

echo ""
echo "Stopping control_allocator so mc_nmpc_control gets actuator_motors instance 0..."
run_qshell "control_allocator stop"
sleep 1

echo "Arming and switching to offboard..."
timeout 8 adb shell "$QSHELL_BIN commander arm -f"
timeout 8 adb shell "$QSHELL_BIN commander mode offboard"

ARMED_OK=0
for i in $(seq 1 10); do
    STATUS=$(run_listener "vehicle_status -n 1" 2>/dev/null)
    CTRL_MODE=$(run_listener "vehicle_control_mode -n 1" 2>/dev/null)
    if echo "$STATUS" | grep -q "arming_state: .*2" && echo "$CTRL_MODE" | grep -q "flag_control_offboard_enabled: .*True"; then
        echo "  Armed and OFFBOARD."
        ARMED_OK=1
        break
    fi
    sleep 1
done

if [ $ARMED_OK -eq 0 ]; then
    echo "ERROR: Failed to reach armed offboard state."
    echo "  vehicle_status:"
    printf '%s\n' "$STATUS"
    echo "  vehicle_control_mode:"
    printf '%s\n' "$CTRL_MODE"
    exit 1
fi

echo ""
echo "=== Simple Motor Test Running ==="
echo "  NMPC motor 0 -> ESC $((NMPC_TO_ESC_MAP[0] + 1)) (ch${NMPC_TO_ESC_MAP[0]})"
TSTART=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - TSTART))
    if [ $ELAPSED -ge $TOTAL_DURATION ]; then
        break
    fi

    SENDER_ALIVE=$(adb shell "pidof single_motor_test_sender" | tr -d '\r' | awk '{print $1}')
    if [ -z "$SENDER_ALIVE" ]; then
        echo "  t=+$(printf "%02d" "$ELAPSED")s  single_motor_test_sender finished."
        break
    fi

    ESC_SAMPLE=$(run_listener "esc_status -n 1" 2>/dev/null)
    if [ -n "$ESC_SAMPLE" ]; then
        ESC_RPMS=$(printf '%s\n' "$ESC_SAMPLE" | tr -d '\r' | awk '
            /esc_rpm:/ {
                for (i = 1; i <= NF; i++) {
                    if ($i == "esc_rpm:") {
                        print $(i + 1)
                    }
                }
            }
        ' | tr '\n' ' ')
        if [ -n "$ESC_RPMS" ]; then
            echo "  t=+$(printf "%02d" "$ELAPSED")s  esc_rpm: [$ESC_RPMS]"
        fi
    fi

    sleep 2
done

echo ""
echo "=== single_motor_test_sender log ==="
adb shell "cat $STDOUT_LOG"

echo ""
echo "=== Simple Motor Test Complete ==="
echo "  NMPC motor 0 -> ESC $((NMPC_TO_ESC_MAP[0] + 1)) (ch${NMPC_TO_ESC_MAP[0]})"
