#!/bin/bash
# test_motor_indexes.sh - Spin each NMPC motor (0..3) one at a time via the pipe protocol.
# Verifies which physical motor spins and its direction for each NMPC index.
# Run from desktop with VOXL connected via USB (adb).
#
# Requires: motor_test_sender binary (built via: make -C voxl_src -f Makefile.voxl motor-test)
#
# The motor_test_sender replaces the NMPC solver: it reads state packets from
# /run/mpa/nmpc_state and writes control packets to /run/mpa/nmpc_control with
# only one motor nonzero at a time. mc_nmpc_control passes commands straight through
# to the ESCs (motor mapping is done via VOXL_ESC_FUNC1-4 params).

set -euo pipefail

MOTOR_TEST_BIN=/usr/local/bin/motor_test_sender
MOTOR_TEST_UNIT=motor-test-sender.service
QSHELL_BIN=
PX4_PARAM_BIN=px4-param
TEST_RPS=50
SECONDS_PER_MOTOR=3
GAP_SECONDS=2
INITIAL_DELAY_SECONDS=4
NO_FLASH=1
BUILD=1
REBOOT=1
RT_PRIO=50
CPU_CORE=7
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STDOUT_LOG=/tmp/motor_test.log
STDERR_LOG=/tmp/motor_test.err.log
LOG_DIR=
ACTUATOR_ACTIVE_THRESHOLD=0.01
ESC_ACTIVE_RPM_THRESHOLD=500
TEST_PARAM_NAMES=(COM_DISARM_PRFLT COM_DISARM_LAND VOXL_ESC_RPM_MIN)
TEST_PARAM_VALUES=(-1 -1 0)
TEST_PARAM_ORIGINAL_VALUES=()
TEST_PARAMS_APPLIED=0
SPIN_DIR_PARAM_NAMES=(VOXL_ESC_SDIR1 VOXL_ESC_SDIR2 VOXL_ESC_SDIR3 VOXL_ESC_SDIR4)
SPIN_DIR_CURRENT_VALUES=()
SPIN_DIR_NEW_VALUES=()
SPIN_DIR_PARAMS_APPLIED=0
VOXL_ESC_SPIN_DIRECTIONS=(normal normal normal inverted)  # per ESC channel (0-3). normal or inverted.
SPIN_DIRECTION_SOURCE="script defaults"
ESC_MOTOR_MAPPING=(0 1 2 3)  # ESC_MOTOR_MAPPING[esc_channel] = nmpc_motor_index. Sets VOXL_ESC_FUNC1-4.
ESC_FUNC_PARAM_NAMES=(VOXL_ESC_FUNC1 VOXL_ESC_FUNC2 VOXL_ESC_FUNC3 VOXL_ESC_FUNC4)
ESC_FUNC_CURRENT_VALUES=()
ESC_FUNC_NEW_VALUES=()
ESC_FUNC_PARAMS_APPLIED=0


# PREV CA_ROTOR params (PX4 control allocator geometry):
#   CA_ROTOR0: PX=0.16 PY=-0.16 KM=-0.05
#   CA_ROTOR1: PX=-0.16 PY=-0.16 KM=0.05
#   CA_ROTOR2: PX=-0.16 PY=0.16 KM=-0.05
#   CA_ROTOR3: PX=0.16 PY=0.16 KM=0.05
# + CA_ROTOR0_PX: prev: -0.1200 -> new: 0.1600
# + CA_ROTOR0_PY: prev: -0.0700 -> new: -0.1600
#   CA_ROTOR0_KM: prev: 0.0500 -> new: -0.0500
#   CA_ROTOR1_PX: prev: 0.0000 -> new: -0.1600
# + CA_ROTOR1_PY: prev: 0.1400 -> new: -0.1600
# + CA_ROTOR2_PX: prev: 0.1200 -> new: -0.1600
# + CA_ROTOR2_PY: prev: -0.0700 -> new: 0.1600
#   CA_ROTOR2_KM: prev: 0.0500 -> new: -0.0500
# + CA_ROTOR3_PX: prev: -0.1200 -> new: 0.1600
# + CA_ROTOR3_PY: prev: 0.0700 -> new: 0.1600
# + CA_ROTOR3_KM: prev: -0.0500 -> new: 0.0500
# + CA_ROTOR_COUNT: prev: 6 -> new: 4



# CA_ROTOR geometry for PX4 stabilized mode.
# Position and axis are PX4 FRD, converted from the NMPC FLU airframe geometry.
CA_ROTOR_PX=(0.12681146607985921 0.0029042659595838248 -0.12681236851557717 -0.0028654005636986474)
CA_ROTOR_PY=(0.0 -0.10363315076185076 0.0 0.11169994528478353)
CA_ROTOR_PZ=(0.0 0.078686766054808108 0.0 0.060148329284921358)
CA_ROTOR_AX=(0.16655232523072261 -0.0019413838124934801 -0.1665674759845783 0.001957027819017243)
CA_ROTOR_AY=(0.0052755595681712869 -0.3989773984220194 -0.0052907433326074838 0.39899377456418594)
CA_ROTOR_AZ=(-0.9860185046090576 -0.91695870494656939 -0.9860158639591533 -0.91695154610324936)
CA_ROTOR_KM=(0.0099999997764825821 -0.0099999997764825821 0.0099999997764825821 -0.0099999997764825821)
CA_ROTOR_PARAM_NAMES=()
CA_ROTOR_CURRENT_VALUES=()
CA_ROTOR_NEW_VALUES=()
CA_ROTOR_PARAMS_APPLIED=0

run_qshell() { adb shell "$QSHELL_BIN $*"; }
run_qshell_timeout() { local t=$1; shift; timeout "$t" adb shell "$QSHELL_BIN $*"; }
run_px4_param() { adb shell "$PX4_PARAM_BIN $*"; }
run_px4_param_timeout() { local t=$1; shift; timeout "$t" adb shell "$PX4_PARAM_BIN $*"; }
run_listener() { adb shell "px4-listener $*"; }
remote_exec_exists() { adb shell "if [ -x $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"; }
remote_cmd_exists() { adb shell "if command -v $1 >/dev/null 2>/dev/null; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"; }
remote_pipe_exists() { adb shell "if [ -p $1 ]; then echo __OK__; else echo __NO__; fi" | tr -d '\r' | grep -qx "__OK__"; }
remote_solver_pid() { adb shell "pidof nmpc_pipe_solver" | tr -d '\r' | awk '{print $1}'; }

NMPC_TO_ESC_MAP=(0 0 0 0)
for _esc_ch in 0 1 2 3; do NMPC_TO_ESC_MAP[${ESC_MOTOR_MAPPING[$_esc_ch]}]=$_esc_ch; done

ensure_voxl_px4_active() {
    local px4_active

    px4_active=$(adb shell "systemctl is-active voxl-px4" | tr -d '\r\n') || px4_active=""
    if [ "$px4_active" = "active" ]; then
        return 0
    fi

    echo "  voxl-px4 status is '$px4_active'; enabling and starting it..."
    adb shell "systemctl enable voxl-px4 >/dev/null 2>&1 || true; systemctl start voxl-px4 || true"

    for i in $(seq 1 20); do
        px4_active=$(adb shell "systemctl is-active voxl-px4" | tr -d '\r\n') || px4_active=""
        if [ "$px4_active" = "active" ]; then
            return 0
        fi
        sleep 1
    done

    return 1
}

load_spin_direction_defaults() {
    local params_file="$SCRIPT_DIR/voxl_mocap_scripts/px4_parameters"
    local loaded_values=()
    local param_name
    local raw_value

    if [ ! -f "$params_file" ]; then
        return 0
    fi

    for esc_ch in 0 1 2 3; do
        param_name="VOXL_ESC_SDIR$((esc_ch + 1))"
        raw_value=$(awk -v name="$param_name" '$1 == name { print $2; exit }' "$params_file")

        case "$raw_value" in
            0) loaded_values+=(normal) ;;
            1) loaded_values+=(inverted) ;;
            "") return 0 ;;
            *)
                echo "ERROR: Invalid $param_name value '$raw_value' in $params_file (expected 0 or 1)."
                exit 1
                ;;
        esac
    done

    if [ ${#loaded_values[@]} -eq 4 ]; then
        VOXL_ESC_SPIN_DIRECTIONS=("${loaded_values[@]}")
        SPIN_DIRECTION_SOURCE="$params_file"
    fi
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

write_spin_direction_manifest() {
    local manifest_path=$LOG_DIR/spin_direction_overrides.txt

    : > "$manifest_path"

    for param_index in "${!SPIN_DIR_PARAM_NAMES[@]}"; do
        printf '%s original=%s override=%s\n' \
            "${SPIN_DIR_PARAM_NAMES[$param_index]}" \
            "${SPIN_DIR_CURRENT_VALUES[$param_index]}" \
            "${SPIN_DIR_NEW_VALUES[$param_index]}" >> "$manifest_path"
    done
}

apply_test_params() {
    local param_name
    local param_value
    local current_value

    echo "Applying PX4 motor-test params..."

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

    echo "Restoring PX4 motor-test params..."

    for param_index in "${!TEST_PARAM_NAMES[@]}"; do
        param_name=${TEST_PARAM_NAMES[$param_index]}
        param_value=${TEST_PARAM_ORIGINAL_VALUES[$param_index]}

        if ! set_and_verify_param "$param_name" "$param_value"; then
            return 1
        fi
    done

    TEST_PARAMS_APPLIED=0
    return 0
}

read_spin_direction_params() {
    local param_name
    local current_value

    SPIN_DIR_CURRENT_VALUES=()

    for param_index in "${!SPIN_DIR_PARAM_NAMES[@]}"; do
        param_name=${SPIN_DIR_PARAM_NAMES[$param_index]}
        if ! current_value=$(get_param_quiet "$param_name"); then
            exit 1
        fi

        SPIN_DIR_CURRENT_VALUES[$param_index]=$current_value
    done
}

print_spin_direction_params() {
    read_spin_direction_params

    echo "Current VOXL ESC spin-direction params:"
    for param_index in "${!SPIN_DIR_PARAM_NAMES[@]}"; do
        echo "  ${SPIN_DIR_PARAM_NAMES[$param_index]}=${SPIN_DIR_CURRENT_VALUES[$param_index]}"
    done
}

configure_spin_direction_params() {
    local esc_ch param_name param_value spin_direction

    if [ ${#VOXL_ESC_SPIN_DIRECTIONS[@]} -ne 4 ]; then
        echo "ERROR: VOXL_ESC_SPIN_DIRECTIONS must contain 4 values."
        exit 1
    fi

    SPIN_DIR_NEW_VALUES=()

    for esc_ch in 0 1 2 3; do
        spin_direction=${VOXL_ESC_SPIN_DIRECTIONS[$esc_ch]}

        case "$spin_direction" in
            normal|inverted) ;;
            *) echo "ERROR: VOXL_ESC_SPIN_DIRECTIONS[$esc_ch] must be normal or inverted."; exit 1 ;;
        esac

        if [ "$spin_direction" = "normal" ]; then
            SPIN_DIR_NEW_VALUES[$esc_ch]=0
        else
            SPIN_DIR_NEW_VALUES[$esc_ch]=1
        fi
    done

    print_spin_direction_params
    echo "Applying VOXL ESC spin-direction params:"
    for esc_ch in 0 1 2 3; do
        echo "  ESC $esc_ch -> ${SPIN_DIR_PARAM_NAMES[$esc_ch]} spin=${VOXL_ESC_SPIN_DIRECTIONS[$esc_ch]} override=${SPIN_DIR_NEW_VALUES[$esc_ch]}"
    done

    for esc_ch in 0 1 2 3; do
        param_name=${SPIN_DIR_PARAM_NAMES[$esc_ch]}
        param_value=${SPIN_DIR_NEW_VALUES[$esc_ch]}

        if ! set_and_verify_param "$param_name" "$param_value"; then
            exit 1
        fi
    done

    run_px4_param_timeout 8 "save"
    SPIN_DIR_PARAMS_APPLIED=1
    write_spin_direction_manifest
    echo ""
    print_spin_direction_params
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
    local esc_ch param_name param_value nmpc_motor

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
        ESC_FUNC_NEW_VALUES[$esc_ch]=$(( 101 + nmpc_motor ))
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

write_ca_rotor_manifest() {
    local manifest_path=$LOG_DIR/ca_rotor_overrides.txt

    : > "$manifest_path"

    for param_index in "${!CA_ROTOR_PARAM_NAMES[@]}"; do
        printf '%s original=%s override=%s\n' \
            "${CA_ROTOR_PARAM_NAMES[$param_index]}" \
            "${CA_ROTOR_CURRENT_VALUES[$param_index]}" \
            "${CA_ROTOR_NEW_VALUES[$param_index]}" >> "$manifest_path"
    done
}

configure_ca_rotor_params() {
    local rotor_idx param_name current_value

    CA_ROTOR_PARAM_NAMES=()
    CA_ROTOR_CURRENT_VALUES=()
    CA_ROTOR_NEW_VALUES=()

    for rotor_idx in 0 1 2 3; do
        for suffix in PX PY PZ AX AY AZ KM; do
            CA_ROTOR_PARAM_NAMES+=("CA_ROTOR${rotor_idx}_${suffix}")
            case "$suffix" in
                PX) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_PX[$rotor_idx]}") ;;
                PY) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_PY[$rotor_idx]}") ;;
                PZ) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_PZ[$rotor_idx]}") ;;
                AX) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_AX[$rotor_idx]}") ;;
                AY) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_AY[$rotor_idx]}") ;;
                AZ) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_AZ[$rotor_idx]}") ;;
                KM) CA_ROTOR_NEW_VALUES+=("${CA_ROTOR_KM[$rotor_idx]}") ;;
            esac
        done
    done

    echo "Reading current CA_ROTOR params..."
    for param_index in "${!CA_ROTOR_PARAM_NAMES[@]}"; do
        param_name=${CA_ROTOR_PARAM_NAMES[$param_index]}
        if ! current_value=$(get_param_quiet "$param_name"); then
            exit 1
        fi
        CA_ROTOR_CURRENT_VALUES[$param_index]=$current_value
    done

    echo "Applying CA_ROTOR params (PX4 control allocator geometry):"
    for rotor_idx in 0 1 2 3; do
        echo "  CA_ROTOR${rotor_idx}: PX=${CA_ROTOR_PX[$rotor_idx]} PY=${CA_ROTOR_PY[$rotor_idx]} PZ=${CA_ROTOR_PZ[$rotor_idx]} AX=${CA_ROTOR_AX[$rotor_idx]} AY=${CA_ROTOR_AY[$rotor_idx]} AZ=${CA_ROTOR_AZ[$rotor_idx]} KM=${CA_ROTOR_KM[$rotor_idx]}"
    done

    for param_index in "${!CA_ROTOR_PARAM_NAMES[@]}"; do
        param_name=${CA_ROTOR_PARAM_NAMES[$param_index]}
        param_value=${CA_ROTOR_NEW_VALUES[$param_index]}

        if ! set_and_verify_param "$param_name" "$param_value"; then
            exit 1
        fi
    done

    if ! set_and_verify_param "CA_ROTOR_COUNT" "4"; then
        exit 1
    fi

    run_px4_param_timeout 8 "save"
    CA_ROTOR_PARAMS_APPLIED=1
    write_ca_rotor_manifest
}

collect_crash_diagnostics() {
    echo ""
    echo "=== CRASH DIAGNOSTICS ==="
    echo "[1] systemctl status voxl-px4:"
    adb shell "systemctl status voxl-px4 --no-pager -l" || true
    echo ""
    echo "[2] journalctl -u voxl-px4 (last 100 lines):"
    adb shell "journalctl -u voxl-px4 -n 100 --no-pager" || true
    echo "=== END DIAGNOSTICS ==="
}

usage() {
    cat <<'EOF'
Usage: test_motor_indexes.sh [--no-reboot] [--rps <value>] [--gap-seconds <value>] [--initial-delay-seconds <value>] [--help]

Options:
  --no-reboot          Skip the initial VOXL reboot and readiness wait.
  --rps VALUE          Override the per-motor test command in RPS.
  --gap-seconds VALUE  Zero-command gap between motor phases.
  --initial-delay-seconds VALUE
                       Zero-command wait before motor 0 starts.
  --help               Show this help text.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-reboot)
            REBOOT=0
            ;;
        --rps)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --rps requires a value."
                usage
                exit 1
            fi
            TEST_RPS=$1
            ;;
        --gap-seconds)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --gap-seconds requires a value."
                usage
                exit 1
            fi
            GAP_SECONDS=$1
            ;;
        --initial-delay-seconds)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --initial-delay-seconds requires a value."
                usage
                exit 1
            fi
            INITIAL_DELAY_SECONDS=$1
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

load_spin_direction_defaults

TOTAL_DURATION=$(awk "BEGIN { printf \"%.0f\", ($SECONDS_PER_MOTOR * 4) + ($GAP_SECONDS * 3) + $INITIAL_DELAY_SECONDS + 10 }")
LOG_DIR="$SCRIPT_DIR/../cache/workspace_local/voxl_motor_test/$(date +%Y_%m_%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Motor Index Test ==="
echo "  RPS:        $TEST_RPS"
echo "  Per motor:  ${SECONDS_PER_MOTOR}s"
echo "  Gap:        ${GAP_SECONDS}s"
echo "  Initial:    ${INITIAL_DELAY_SECONDS}s"
echo "  Total:      ${TOTAL_DURATION}s"
echo "  Build:      $([ $BUILD -eq 1 ] && echo YES || echo NO)"
echo "  Reboot:     $([ $REBOOT -eq 1 ] && echo YES || echo NO)"
echo "  Spin dirs:  ${VOXL_ESC_SPIN_DIRECTIONS[*]} ($SPIN_DIRECTION_SOURCE)"
echo "  Log dir:    $LOG_DIR"
echo ""
echo "  Will spin NMPC motors 0, 1, 2, 3 one at a time."
echo "  Observe which physical motor spins and its direction."
echo ""

# ── Flash firmware ────────────────────────────────────────────────────────────
if [ $NO_FLASH -eq 0 ]; then
    bash "$SCRIPT_DIR/flash_firmware.sh"
    echo ""
fi

# ── Build and deploy motor_test_sender ────────────────────────────────────────
if [ $BUILD -eq 1 ]; then
    echo "Building motor_test_sender..."
    make -C "$SCRIPT_DIR" -f Makefile.voxl motor-test
    echo "Deploying to VOXL..."
    adb shell "mkdir -p /usr/local/bin"
    adb push "$SCRIPT_DIR/motor_test_sender" "$MOTOR_TEST_BIN"
    adb shell "chmod +x $MOTOR_TEST_BIN"
    echo ""
fi

# ── Prerequisites ─────────────────────────────────────────────────────────────
detect_qshell_bin
if [ "$QSHELL_BIN" = "__MISSING__" ] || ! remote_exec_exists "$QSHELL_BIN"; then
    echo "ERROR: qshell binary not found on VOXL."; exit 1
fi
echo "Using qshell binary: $QSHELL_BIN"

if [ $BUILD -eq 0 ] && ! remote_exec_exists "$MOTOR_TEST_BIN"; then
    echo "ERROR: $MOTOR_TEST_BIN not found on VOXL. Run without --no-build."; exit 1
fi
for cmd in taskset chrt pidof systemd-run; do
    remote_cmd_exists "$cmd" || { echo "ERROR: $cmd not found on VOXL."; exit 1; }
done
remote_cmd_exists "$PX4_PARAM_BIN" || { echo "ERROR: $PX4_PARAM_BIN not found on VOXL."; exit 1; }

echo "WARNING: Motors will spin. Ensure propellers are removed."
echo ""

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Exit offboard cleanly before stopping mc_nmpc_control.  If mc_nmpc_control
    # is stopped while the vehicle is armed in offboard, PX4 raises an
    # offboard-loss failsafe that latches and breaks subsequent --no-reboot runs.
    run_qshell "commander mode stabilized" 2>/dev/null || true
    sleep 1
    run_qshell "commander disarm -f" || true
    run_qshell "mc_nmpc_control stop" || true
    run_qshell "control_allocator start" >/dev/null 2>/dev/null || true
    if ! restore_test_params; then
        echo "ERROR: Failed to restore one or more PX4 motor-test params."
    fi
    adb shell "systemctl stop ${MOTOR_TEST_UNIT%.service} 2>/dev/null || true"

    echo "Copying logs to $LOG_DIR ..."
    adb shell "test -f $STDOUT_LOG" 2>/dev/null && adb pull "$STDOUT_LOG" "$LOG_DIR/motor_test.log" 2>/dev/null || true
    adb shell "test -f $STDERR_LOG" 2>/dev/null && adb pull "$STDERR_LOG" "$LOG_DIR/motor_test.err.log" 2>/dev/null || true
    adb shell "journalctl -u voxl-px4 -b -n 200 --no-pager > /tmp/voxl-px4.journal.log" || true
    adb pull /tmp/voxl-px4.journal.log "$LOG_DIR/voxl-px4.journal.log" 2>/dev/null || true
    echo "Logs saved to: $LOG_DIR"
}
trap cleanup EXIT

# ── Reach clean/ready VOXL state ─────────────────────────────────────────────
if [ $REBOOT -eq 1 ]; then
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

    echo "  Waiting 30s for services to initialize..."
    sleep 30
else
    echo "Skipping initial reboot. Verifying current VOXL session..."
fi

if ! ensure_voxl_px4_active; then
    PX4_ACTIVE=$(adb shell "systemctl is-active voxl-px4" | tr -d '\r\n') || PX4_ACTIVE=""
    if [ $REBOOT -eq 1 ]; then
        echo "ERROR: voxl-px4 not active after reboot (status: '$PX4_ACTIVE')."
    else
        echo "ERROR: voxl-px4 is not active in the current session (status: '$PX4_ACTIVE')."
    fi
    collect_crash_diagnostics
    exit 1
fi
echo "  voxl-px4 is active."

echo "  Waiting 15s for PX4 topic graph..."
sleep 15

LISTENER_OUT=$(run_listener "vehicle_status -n 1" 2>/dev/null) || LISTENER_OUT=""
if ! echo "$LISTENER_OUT" | grep -q "arming_state"; then
    if [ $REBOOT -eq 1 ]; then
        echo "ERROR: PX4 did not become ready after reboot."
    else
        echo "ERROR: PX4 is not ready in the current session."
    fi
    collect_crash_diagnostics
    exit 1
fi
echo "PX4 is ready."

echo ""
configure_esc_motor_mapping
echo ""
configure_spin_direction_params
echo ""
configure_ca_rotor_params
echo ""

CURRENT_THR_MDL_FAC=$(get_param_quiet "THR_MDL_FAC") || exit 1
CURRENT_ESC_RPM_MAX=$(get_param_quiet "VOXL_ESC_RPM_MAX") || exit 1
echo "Current PX4 actuator mapping inputs:"
echo "  THR_MDL_FAC=$CURRENT_THR_MDL_FAC"
echo "  VOXL_ESC_RPM_MAX=$CURRENT_ESC_RPM_MAX"
echo "  Requested test command = ${TEST_RPS} rps ($(awk "BEGIN { printf \"%.0f\", $TEST_RPS * 60.0 }") rpm)"
echo ""

# ── Stop conflicting publishers ──────────────────────────────────────────────
if run_qshell "mc_neural_control status" >/dev/null 2>/dev/null; then
    echo "Stopping mc_neural_control..."
    run_qshell "mc_neural_control stop" || true
    sleep 1
fi

# ── Apply motor-test params ──────────────────────────────────────────────────
echo "Disabling auto-disarm and motor RPM minima for motor-index test..."
apply_test_params

# mc_nmpc_control caches VOXL_ESC_RPM_MIN at startup. If it was already running
# (started by PX4 at boot) it holds the old value (typically 4980 rpm). Any
# test RPS below rpm_min/60 would map to control≈0 → active motor off, inactive
# motors at idle floor. Force a restart so it re-reads the updated param (0).
echo "Restarting mc_nmpc_control to reload updated ESC params..."
run_qshell "mc_nmpc_control stop" 2>/dev/null || true
sleep 2

# actuator_motors instance 0 must be free before mc_nmpc_control starts. If
# control_allocator advertises it first, mc_nmpc_control can end up publishing
# on another uORB instance while the ESC driver continues following instance 0.
echo "Stopping control_allocator so mc_nmpc_control gets actuator_motors instance 0..."
run_qshell "control_allocator stop" || true
sleep 1

# ── Ensure NMPC modules are up (creates the FIFOs) ──────────────────────────
echo ""
echo "Starting NMPC modules..."
for attempt in $(seq 1 5); do
    STATUS_OUT=$(run_qshell "mc_nmpc_control status" 2>&1 || true)
    if ! echo "$STATUS_OUT" | grep -qi "Command failed"; then
        echo "  mc_nmpc_control: running"
        break
    fi
    echo "  Starting mc_nmpc_control (attempt $attempt/5)..."
    run_qshell "mc_nmpc_control start" || true
    sleep 3
done
STATUS_OUT=$(run_qshell "mc_nmpc_control status" 2>&1 || true)
if echo "$STATUS_OUT" | grep -qi "Command failed"; then
    echo "ERROR: mc_nmpc_control failed to start."
    exit 1
fi

adb shell "cd /usr/bin; /bin/ln -f -s px4 px4-nmpc_io_bridge" || true
for attempt in $(seq 1 5); do
    if remote_pipe_exists /run/mpa/nmpc_control; then
        echo "  nmpc_io_bridge: pipe ready"; break
    fi
    echo "  Starting nmpc_io_bridge (attempt $attempt/5)..."
    adb shell "px4-nmpc_io_bridge start" || true
    sleep 3
done
remote_pipe_exists /run/mpa/nmpc_control || { echo "ERROR: nmpc_io_bridge failed to start."; exit 1; }

echo "Stopping nmpc_pipe_solver to prevent competing control packets..."
adb shell "systemctl stop nmpc-pipe-solver"
sleep 1
SOLVER_PID=$(remote_solver_pid) || SOLVER_PID=""
if [ -n "$SOLVER_PID" ]; then
    echo "ERROR: nmpc_pipe_solver is still running (PID=$SOLVER_PID)."
    exit 1
fi

# ── Wait for FIFOs ───────────────────────────────────────────────────────────
echo "Waiting for FIFOs..."
FIFO_OK=0
for i in $(seq 1 10); do
    STATE_OK=0; CTRL_OK=0
    adb shell "test -p /run/mpa/nmpc_state"   2>/dev/null && STATE_OK=1 || true
    adb shell "test -p /run/mpa/nmpc_control" 2>/dev/null && CTRL_OK=1 || true
    if [ $STATE_OK -eq 1 ] && [ $CTRL_OK -eq 1 ]; then
        echo "  FIFOs ready."; FIFO_OK=1; break
    fi
    sleep 2
done
[ $FIFO_OK -eq 0 ] && { echo "ERROR: FIFOs not present after 20s."; exit 1; }

# ── Start motor_test_sender ─────────────────────────────────────────────────
echo ""
adb shell "rm -f $STDOUT_LOG $STDERR_LOG"
echo "Starting motor_test_sender (rps=$TEST_RPS, ${SECONDS_PER_MOTOR}s/motor, ${GAP_SECONDS}s gap, ${INITIAL_DELAY_SECONDS}s initial delay)..."
adb shell "systemd-run --unit=${MOTOR_TEST_UNIT%.service} --collect --property=StandardOutput=file:$STDOUT_LOG --property=StandardError=file:$STDERR_LOG taskset -c $CPU_CORE chrt -f $RT_PRIO $MOTOR_TEST_BIN $TEST_RPS $SECONDS_PER_MOTOR $GAP_SECONDS $INITIAL_DELAY_SECONDS"
sleep 3
SENDER_PID=$(adb shell "pidof motor_test_sender" | tr -d '\r' | awk '{print $1}') || SENDER_PID=""
if [ -z "$SENDER_PID" ]; then
    echo "ERROR: motor_test_sender exited immediately."
    echo "  stdout:"
    adb shell "cat $STDOUT_LOG" || true
    echo "  stderr:"
    adb shell "cat $STDERR_LOG" || true
    exit 1
fi
echo "  motor_test_sender running (PID=$SENDER_PID)"

# ── Wait for EKF2 attitude ──────────────────────────────────────────────────
echo "Waiting for EKF2 attitude..."
for i in $(seq 1 30); do
    if run_listener "vehicle_attitude -n 1" 2>/dev/null | grep -q "q:"; then
        echo "  EKF2 attitude OK"; break
    fi
    [ $i -eq 30 ] && echo "  WARNING: EKF2 attitude not confirmed."
    sleep 1
done

# ── Arm + Offboard ───────────────────────────────────────────────────────────
echo ""
echo "Arming and switching to offboard..."
timeout 8 adb shell "$QSHELL_BIN commander arm -f" || true
timeout 8 adb shell "$QSHELL_BIN commander mode offboard" || true

ARMED_OK=0
for i in $(seq 1 10); do
    STATUS=$(run_listener "vehicle_status -n 1" 2>/dev/null) || STATUS=""
    CTRL_MODE=$(run_listener "vehicle_control_mode -n 1" 2>/dev/null) || CTRL_MODE=""
    if echo "$STATUS" | grep -q "arming_state: .*2" && echo "$CTRL_MODE" | grep -q "flag_control_offboard_enabled: .*True"; then
        echo "  Armed and OFFBOARD."
        ARMED_OK=1; break
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

# ── Monitor motor test ──────────────────────────────────────────────────────
echo ""
echo "=== Motor Test Running ==="
TSTART=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - TSTART))
    [ $ELAPSED -ge $TOTAL_DURATION ] && break

    STATUS=$(run_listener "vehicle_status -n 1" 2>/dev/null) || STATUS=""
    if [ -n "$STATUS" ] && ! echo "$STATUS" | grep -q "arming_state: .*2"; then
        echo "ERROR: PX4 disarmed during motor test."
        echo "  vehicle_status:"
        printf '%s\n' "$STATUS"
        collect_crash_diagnostics
        exit 1
    fi

    SENDER_ALIVE=$(adb shell "pidof motor_test_sender" | tr -d '\r' | awk '{print $1}') || SENDER_ALIVE=""
    if [ -z "$SENDER_ALIVE" ]; then
        echo "  t=+$(printf "%02d" "$ELAPSED")s  motor_test_sender finished."
        break
    fi

    ESC_SAMPLE=$(run_listener "esc_status -n 1" 2>/dev/null) || ESC_SAMPLE=""
    ACTUATOR_SAMPLE=$(run_listener "actuator_motors -n 1" 2>/dev/null) || ACTUATOR_SAMPLE=""
    ACTUATOR_CONTROLS=""
    ACTUATOR_ACTIVE_COUNT=""
    ESC_ACTIVE_COUNT=""

    if [ -n "$ACTUATOR_SAMPLE" ]; then
        ACTUATOR_CONTROLS=$(printf '%s\n' "$ACTUATOR_SAMPLE" | tr -d '\r' | sed -n 's/.*control: \[\(.*\)\].*/\1/p' | head -n 1 | cut -d',' -f1-4 | tr -d ' ')

        if [ -n "$ACTUATOR_CONTROLS" ]; then
            ACTUATOR_ACTIVE_COUNT=$(printf '%s\n' "$ACTUATOR_CONTROLS" | awk -F',' -v threshold="$ACTUATOR_ACTIVE_THRESHOLD" '
                function abs(v) { return v < 0 ? -v : v }
                {
                    count = 0
                    for (i = 1; i <= NF; i++) {
                        if ($i != "nan" && abs($i + 0.0) > threshold) {
                            count++
                        }
                    }
                    print count
                }')
        fi
    fi

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
            ESC_ACTIVE_COUNT=$(printf '%s\n' "$ESC_RPMS" | awk -v threshold="$ESC_ACTIVE_RPM_THRESHOLD" '
                {
                    count = 0
                    for (i = 1; i <= NF; i++) {
                        if (($i + 0) > threshold) {
                            count++
                        }
                    }
                    print count
                }')
            echo "  t=+$(printf "%02d" "$ELAPSED")s  esc_rpm: [$ESC_RPMS]"
        else
            echo "  t=+$(printf "%02d" "$ELAPSED")s  esc_status received, no esc_rpm fields."
        fi
    fi

    if [ -n "$ACTUATOR_CONTROLS" ]; then
        echo "  t=+$(printf "%02d" "$ELAPSED")s  actuator_cmd: [$ACTUATOR_CONTROLS]"
    fi

    if [ -n "$ACTUATOR_ACTIVE_COUNT" ] && [ "$ACTUATOR_ACTIVE_COUNT" -gt 1 ]; then
        echo "  WARNING: actuator_motors has $ACTUATOR_ACTIVE_COUNT active channels (> $ACTUATOR_ACTIVE_THRESHOLD)."
    fi

    if [ -n "$ESC_ACTIVE_COUNT" ] && [ "$ESC_ACTIVE_COUNT" -gt 1 ]; then
        echo "  WARNING: esc_status reports $ESC_ACTIVE_COUNT motors above ${ESC_ACTIVE_RPM_THRESHOLD} rpm."
    fi

    sleep 2
done

# ── Report ───────────────────────────────────────────────────────────────────
echo ""
echo "=== motor_test_sender log ==="
adb shell "cat $STDOUT_LOG" || true

echo ""
echo "=== Motor Test Complete ==="
echo ""
echo "  ESC_MOTOR_MAPPING = (${ESC_MOTOR_MAPPING[*]})"
echo "  NMPC motor spin order -> ESC channel:"
for _nmpc_idx in 0 1 2 3; do
    _esc_ch=${NMPC_TO_ESC_MAP[$_nmpc_idx]}
    echo "    Phase $((_nmpc_idx+1)): NMPC motor $_nmpc_idx -> ESC $((_esc_ch+1)) (ch$_esc_ch) spin=${VOXL_ESC_SPIN_DIRECTIONS[$_esc_ch]}"
done
