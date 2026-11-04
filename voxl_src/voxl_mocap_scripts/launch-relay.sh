#!/bin/bash
# Launch mocap relay pipeline in a terminator window with tiled panes.
# Each pane SSHs into the VOXL, then runs its command.

set -euo pipefail

load_voxl_connection_config() {
    VOXL_USER="root"
    VOXL_IP="192.168.5.2"
    VOXL_PASS="oelinux123"
}

run_local_gui_command_clean() {
    (
        local snap_var
        for snap_var in $(compgen -A variable SNAP); do
            unset "${snap_var}"
        done

        unset \
            LD_LIBRARY_PATH \
            LD_PRELOAD \
            PYTHONHOME \
            PYTHONPATH \
            PYTHONUSERBASE \
            GIO_EXTRA_MODULES \
            GIO_MODULE_DIR \
            GI_TYPELIB_PATH \
            GTK_PATH \
            GDK_PIXBUF_MODULEDIR \
            GDK_PIXBUF_MODULE_FILE \
            XDG_CACHE_HOME \
            XDG_CONFIG_HOME \
            XDG_DATA_HOME

        export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        exec "$@"
    )
}

apply_px4_parameters() {
    local parameter_file="$1"
    local remote_parameter_file="$2"

    if [ ! -f "$parameter_file" ]; then
        echo "ERROR: PX4 parameter file does not exist: $parameter_file"
        exit 1
    fi

    ${SCP} "$parameter_file" "${VOXL_USER}@${VOXL_IP}:${remote_parameter_file}"

    ${SSH_BATCH} "PARAM_FILE='${remote_parameter_file}' bash -s" << 'EOF'
set -euo pipefail

if [ ! -f "$PARAM_FILE" ]; then
    echo "ERROR: Remote PX4 parameter file does not exist: $PARAM_FILE"
    exit 1
fi

systemctl is-active --quiet voxl-px4

fail_if_armed() {
    local armed_state
    armed_state=$(px4-listener actuator_armed 1)
    if printf '%s\n' "$armed_state" | grep -q "armed: True"; then
        echo "ERROR: PX4 is armed. Refusing to apply parameters."
        exit 1
    fi
}

params_changed=0
fail_if_armed

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
        ""|\#*) continue ;;
    esac

    field_count=$(printf '%s\n' "$line" | awk '{print NF}')
    if [ "$field_count" -ne 2 ]; then
        echo "ERROR: Invalid PX4 parameter line: $line"
        exit 1
    fi

    param_name=$(printf '%s\n' "$line" | awk '{print $1}')
    param_value=$(printf '%s\n' "$line" | awk '{print $2}')

    if px4-param compare "$param_name" "$param_value"; then
        echo "[OK] $param_name is already $param_value"
    else
        echo "[INFO] Setting $param_name=$param_value"
        px4-param set "$param_name" "$param_value"
        px4-param compare "$param_name" "$param_value"
        params_changed=1
    fi
done < "$PARAM_FILE"

if [ "$params_changed" -eq 0 ]; then
    echo "[INFO] PX4 parameters are already in the desired state; keeping voxl-px4 running"
    exit 0
fi

fail_if_armed
px4-param save
sync
echo "[INFO] PX4 parameters changed"
exit 10
EOF
}

sync_nmpc_config() {
    local config_file="$1"
    local remote_config_file="$2"
    local local_hash
    local remote_hash
    local verified_hash

    if [ ! -f "$config_file" ]; then
        echo "ERROR: NMPC config file does not exist: $config_file"
        exit 1
    fi

    local_hash=$(sha256sum "$config_file" | awk '{print $1}')

    if ${SSH_BATCH} "test -f '${remote_config_file}'"; then
        remote_hash=$(${SSH_BATCH} "sha256sum '${remote_config_file}'" | awk '{print $1}' | tr -d '\r\n')
        if [ "$remote_hash" = "$local_hash" ]; then
            echo "[OK] Remote NMPC config matches ${config_file}"
            return 0
        fi
        echo "[INFO] Remote NMPC config differs from ${config_file}"
    else
        echo "[INFO] Remote NMPC config is missing: ${remote_config_file}"
    fi

    echo "Deploying NMPC config from ${config_file}..."
    ${SCP} "$config_file" "${VOXL_USER}@${VOXL_IP}:${remote_config_file}"
    verified_hash=$(${SSH_BATCH} "sync; sha256sum '${remote_config_file}'" | awk '{print $1}' | tr -d '\r\n')
    if [ "$verified_hash" != "$local_hash" ]; then
        echo "ERROR: Remote NMPC config hash mismatch after deploy"
        echo "  local:  $local_hash"
        echo "  remote: $verified_hash"
        exit 1
    fi

    echo "[INFO] NMPC config changed"
    return 10
}

generate_nmpc_px4_parameter_file() {
    local config_file="$1"
    local output_file="$2"

    python3 - "$config_file" "$output_file" << 'PY'
import math
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
repo_root = config_path.parent.parent
sys.path.insert(0, str(repo_root / "voxl_src"))

from trajectory_reference import parse_circle_config

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
trajectory = cfg["trajectory"]
active_trajectory = trajectory["active"]
collision = trajectory["collision"]

control_modes = {
    "NMPC_DIRECT_ACTUATOR": 0,
    "PX4_POSITION_CONTROL": 1,
}
nmpc_modes = {
    "NMPC_FLIGHT_MODE": 0,
    "NMPC_RECOVERY_MODE": 1,
    "NMPC_COAST_MODE": 2,
}

def require_vector3(name, value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list")
    parsed = [float(item) for item in value]
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"{name} contains non-finite values")
    return parsed

def format_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric YAML value: {value}")
    return f"{parsed:.9g}"

def format_optional_float_has_value(value):
    if value is None:
        return "0", "0"
    return "1", format_float(value)

def build_collision_setpoints():
    setpoints = collision["setpoints"]
    if not setpoints:
        raise ValueError("trajectory.collision.setpoints must not be empty")
    return setpoints

def build_circle_setpoints():
    circle = parse_circle_config(trajectory["circle"])
    setpoint = {
        "position_rel_gap": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "setpoint_time": circle["total_duration_s"],
        "waypoint_x_limit_rel_gap": None,
        "control_mode": "NMPC_DIRECT_ACTUATOR",
        "nmpc_mode": "NMPC_FLIGHT_MODE",
    }
    return [setpoint]

if active_trajectory == "collision":
    setpoints = build_collision_setpoints()
elif active_trajectory == "circle":
    setpoints = build_circle_setpoints()
else:
    raise ValueError(f"unsupported trajectory.active: {active_trajectory}")

if len(setpoints) > 7:
    raise ValueError(f"trajectory.{active_trajectory}.setpoints supports at most 7 setpoints in PX4 parameters")

if active_trajectory == "collision":
    repeat_count = collision["repeat_count"]
elif active_trajectory == "circle":
    repeat_count = 1

if not isinstance(repeat_count, int) or isinstance(repeat_count, bool) or repeat_count <= 0:
    raise ValueError(f"trajectory.{active_trajectory}.repeat_count must be a positive integer")

if repeat_count > 1 and setpoints[-1]["setpoint_time"] is None:
    raise ValueError(f"trajectory.{active_trajectory}.setpoints[-1].setpoint_time is required when repeat_count > 1")

gap_ref1 = require_vector3("trajectory.collision.gap_ref1_mm", collision["gap_ref1_mm"])
gap_ref2 = require_vector3("trajectory.collision.gap_ref2_mm", collision["gap_ref2_mm"])

lines = [
    "",
    f"NMPC_G1_XMM {format_float(gap_ref1[0])}",
    f"NMPC_G1_YMM {format_float(gap_ref1[1])}",
    f"NMPC_G1_ZMM {format_float(gap_ref1[2])}",
    f"NMPC_G2_XMM {format_float(gap_ref2[0])}",
    f"NMPC_G2_YMM {format_float(gap_ref2[1])}",
    f"NMPC_G2_ZMM {format_float(gap_ref2[2])}",
    f"NMPC_SP_COUNT {len(setpoints)}",
    f"NMPC_SP_REPEAT {repeat_count}",
]

for index, setpoint in enumerate(setpoints):
    pos = require_vector3(f"trajectory.{active_trajectory}.setpoints[{index}].position_rel_gap", setpoint["position_rel_gap"])
    vel = require_vector3(f"trajectory.{active_trajectory}.setpoints[{index}].velocity", setpoint["velocity"])
    control_mode = setpoint["control_mode"]
    nmpc_mode = setpoint["nmpc_mode"]
    has_time, setpoint_time = format_optional_float_has_value(setpoint["setpoint_time"])
    has_x_limit, waypoint_x_limit = format_optional_float_has_value(setpoint["waypoint_x_limit_rel_gap"])
    if control_mode not in control_modes:
        raise ValueError(f"invalid control_mode at setpoint {index}: {control_mode}")
    if nmpc_mode not in nmpc_modes:
        raise ValueError(f"invalid nmpc_mode at setpoint {index}: {nmpc_mode}")

    prefix = f"NMPC_S{index}"
    lines.extend([
        f"{prefix}_PX {format_float(pos[0])}",
        f"{prefix}_PY {format_float(pos[1])}",
        f"{prefix}_PZ {format_float(pos[2])}",
        f"{prefix}_VX {format_float(vel[0])}",
        f"{prefix}_VY {format_float(vel[1])}",
        f"{prefix}_VZ {format_float(vel[2])}",
        f"{prefix}_HAS_T {has_time}",
        f"{prefix}_T {setpoint_time}",
        f"{prefix}_HAS_X {has_x_limit}",
        f"{prefix}_XLIM {waypoint_x_limit}",
        f"{prefix}_CTRL {control_modes[control_mode]}",
        f"{prefix}_MODE {nmpc_modes[nmpc_mode]}",
    ])

output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

print_battery_estimate() {
    local battery_status_output
    local battery_summary

    battery_status_output="$(${SSH_BATCH} "px4-listener battery_status 1")"

    battery_summary="$(BATTERY_STATUS_OUTPUT="$battery_status_output" python3 << 'PY'
import os
import re

text = os.environ["BATTERY_STATUS_OUTPUT"]

voltage_match = re.search(r"voltage_v:\s*([0-9.]+)", text)
current_match = re.search(r"current_a:\s*([0-9.]+)", text)
cell_count_match = re.search(r"cell_count:\s*(\d+)", text)
capacity_match = re.search(r"capacity:\s*(\d+)", text)

if voltage_match is None:
    raise SystemExit("ERROR: Could not parse voltage_v from PX4 battery_status")
if current_match is None:
    raise SystemExit("ERROR: Could not parse current_a from PX4 battery_status")
if cell_count_match is None:
    raise SystemExit("ERROR: Could not parse cell_count from PX4 battery_status")
voltage_v = float(voltage_match.group(1))
current_a = float(current_match.group(1))
cell_count = int(cell_count_match.group(1))

if cell_count != 4:
    raise SystemExit(f"ERROR: Expected 4S battery, got {cell_count}S")

per_cell_voltage = voltage_v / cell_count

curve = [
    (4.350, 100.0),
    (4.280, 95.0),
    (4.220, 90.0),
    (4.150, 85.0),
    (4.050, 83.0),
    (4.000, 75.0),
    (3.950, 65.0),
    (3.900, 55.0),
    (3.850, 45.0),
    (3.800, 35.0),
    (3.750, 25.0),
    (3.700, 18.0),
    (3.650, 12.0),
    (3.600, 8.0),
    (3.550, 4.0),
    (3.500, 0.0),
]

if per_cell_voltage >= curve[0][0]:
    percent = curve[0][1]
elif per_cell_voltage <= curve[-1][0]:
    percent = curve[-1][1]
else:
    percent = None
    for (v_hi, p_hi), (v_lo, p_lo) in zip(curve, curve[1:]):
        if v_hi >= per_cell_voltage >= v_lo:
            alpha = (per_cell_voltage - v_lo) / (v_hi - v_lo)
            percent = p_lo + alpha * (p_hi - p_lo)
            break
    if percent is None:
        raise SystemExit("ERROR: Failed to interpolate LiHV battery percentage")

print(f"Battery estimate: {percent:.1f}% | voltage={voltage_v:.3f} V | per_cell={per_cell_voltage:.3f} V | current={current_a:.3f} A | battery=4S LiHV 1100 mAh")
PY
)"

    printf '%s\n' "$battery_summary"
}

main() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    load_voxl_connection_config
    SSH="sshpass -p ${VOXL_PASS} ssh -tt -o StrictHostKeyChecking=no ${VOXL_USER}@${VOXL_IP}"
    SSH_BATCH="sshpass -p ${VOXL_PASS} ssh -T -o StrictHostKeyChecking=no ${VOXL_USER}@${VOXL_IP}"
    SCP="sshpass -p ${VOXL_PASS} scp -o StrictHostKeyChecking=no"

    ODOM_RELAY_REMOTE_PATH="/home/root/qualisys_odom_to_px4_mavlink.py"
    ODOM_RECORDER_REMOTE_PATH="/home/root/record_qualisys_odom_csv.py"
    PANE1_INIT_REMOTE_PATH="/home/root/mocap_pane1_init.sh"
    PANE2_INIT_REMOTE_PATH="/home/root/mocap_pane2_init.sh"
    RAW_QUALISYS_CSV_REMOTE_PATH="/tmp/qualisys_odom.csv"
    BATTERY_RECORDER_REMOTE_PATH="/home/root/record_battery_status_csv.py"
    BATTERY_LOG_REMOTE_PATH="/tmp/battery_status_log.csv"
    MAVROS_READY_REMOTE_PATH="/tmp/mavros_ready"
    PX4_PARAMETER_FILE="${SCRIPT_DIR}/px4_parameters"
    PX4_PARAMETER_REMOTE_PATH="/home/root/px4_parameters"
    NMPC_CONFIG_FILE="$(cd "${SCRIPT_DIR}/../.." && pwd)/configs/nmpc_controller_cfg.yaml"
    NMPC_CONFIG_REMOTE_PATH="/home/root/nmpc_controller_cfg.yaml"
    NMPC_PX4_PARAMETER_LOCAL_PATH="/tmp/nmpc_px4_parameters"
    PX4_COMBINED_PARAMETER_LOCAL_PATH="/tmp/px4_parameters_with_nmpc"
    nmpc_changed=0
    px4_parameters_changed=0
    if [ ! -f "$NMPC_CONFIG_FILE" ]; then
        echo "ERROR: NMPC config file does not exist: $NMPC_CONFIG_FILE"
        exit 1
    fi
    print_battery_estimate
    if sync_nmpc_config "$NMPC_CONFIG_FILE" "$NMPC_CONFIG_REMOTE_PATH"; then
        true
    else
        sync_status="$?"
        if [ "$sync_status" -eq 10 ]; then
            nmpc_changed=1
        else
            exit "$sync_status"
        fi
    fi
    echo "Generating PX4 NMPC parameters from ${NMPC_CONFIG_FILE}..."
    generate_nmpc_px4_parameter_file "$NMPC_CONFIG_FILE" "$NMPC_PX4_PARAMETER_LOCAL_PATH"
    cat "$PX4_PARAMETER_FILE" "$NMPC_PX4_PARAMETER_LOCAL_PATH" > "$PX4_COMBINED_PARAMETER_LOCAL_PATH"
    echo "Syncing VOXL clock to local machine..."
    ${SSH_BATCH} "date -s @$(date +%s)"
    ${SSH_BATCH} "rm -f /etc/systemd/system/voxl-px4.service.d/nmpc-env.conf /home/root/nmpc_px4_env; systemctl daemon-reload"
    echo "Applying PX4 parameters from ${PX4_COMBINED_PARAMETER_LOCAL_PATH}..."
    if apply_px4_parameters "$PX4_COMBINED_PARAMETER_LOCAL_PATH" "$PX4_PARAMETER_REMOTE_PATH"; then
        true
    else
        apply_status="$?"
        if [ "$apply_status" -eq 10 ]; then
            px4_parameters_changed=1
        else
            exit "$apply_status"
        fi
    fi
    if [ "$nmpc_changed" -eq 1 ] || [ "$px4_parameters_changed" -eq 1 ]; then
        echo "VOXL reboot requested because NMPC config or PX4 parameters changed; exiting without launching relay."
        ${SSH_BATCH} "nohup sh -c 'sleep 1; reboot' >/dev/null 2>&1 &"
        exit 0
    fi
    ${SCP} "${SCRIPT_DIR}/qualisys_odom_to_px4_mavlink.py" "${VOXL_USER}@${VOXL_IP}:${ODOM_RELAY_REMOTE_PATH}"
    ${SCP} "${SCRIPT_DIR}/record_qualisys_odom_csv.py" "${VOXL_USER}@${VOXL_IP}:${ODOM_RECORDER_REMOTE_PATH}"
    ${SCP} "${SCRIPT_DIR}/record_battery_status_csv.py" "${VOXL_USER}@${VOXL_IP}:${BATTERY_RECORDER_REMOTE_PATH}"
    cat > /tmp/mocap_pane1_init.sh << EOF
#!/bin/bash

wait_for_service() {
    local service_name="\$1"
    local timeout_s="\$2"
    local deadline_s=\$((\$(date +%s) + timeout_s))
    while [ \$(date +%s) -le "\$deadline_s" ]; do
        if systemctl is-active --quiet "\$service_name"; then
            echo "[OK] \$service_name is active"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] \$service_name did not become active within \${timeout_s}s"
    systemctl status --no-pager "\$service_name" || true
    return 1
}

check_background_process() {
    local process_label="\$1"
    local process_pid="\$2"
    if ps -p "\$process_pid" >/dev/null 2>&1; then
        echo "[OK] \$process_label is running with pid \$process_pid"
        return 0
    fi
    echo "[ERROR] \$process_label failed to stay alive"
    return 1
}

wait_for_ros_master() {
    local timeout_s="\$1"
    local deadline_s=\$((\$(date +%s) + timeout_s))
    while [ \$(date +%s) -le "\$deadline_s" ]; do
        if rostopic list >/dev/null 2>&1; then
            echo "[OK] ROS master is reachable"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] ROS master did not become reachable within \${timeout_s}s"
    return 1
}

wait_for_service_inactive() {
    local service_name="\$1"
    local timeout_s="\$2"
    local deadline_s=\$((\$(date +%s) + timeout_s))
    while [ \$(date +%s) -le "\$deadline_s" ]; do
        if ! systemctl is-active --quiet "\$service_name"; then
            echo "[OK] \$service_name is inactive"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] \$service_name did not stop within \${timeout_s}s"
    systemctl status --no-pager "\$service_name" || true
    return 1
}

mocap_pane1_setup() {
    set -eo pipefail
    source /home/root/catkin_ws/devel/setup.bash
    set -u

    echo "[INFO] Preparing Qualisys recorder and MAVLink relay on \$(hostname)"
    chmod +x ${ODOM_RELAY_REMOTE_PATH} ${ODOM_RECORDER_REMOTE_PATH} ${BATTERY_RECORDER_REMOTE_PATH}
    rm -f ${RAW_QUALISYS_CSV_REMOTE_PATH}
    rm -f ${BATTERY_LOG_REMOTE_PATH}
    rm -f ${MAVROS_READY_REMOTE_PATH}
    export ROS_MASTER_URI=http://${VOXL_IP}:11311/
    wait_for_ros_master 30

    echo "[INFO] Stopping any previous recorder, relay, or MAVROS processes"
    pkill -f "[r]ecord_qualisys_odom_csv.py" || true
    pkill -f "[r]ecord_battery_status_csv.py" || true
    pkill -f "[q]ualisys_odom_to_px4_mavlink.py" || true
    pkill -f "[m]avros_node" || true
    pkill -f "roslaunch mavros px4.launch" || true
    pkill -f "roslaunch hexbug mavros.launch" || true
    sleep 1

    echo "[INFO] Stopping voxl-mavlink-server so MAVROS can bind 127.0.0.1:14557"
    systemctl stop voxl-mavlink-server
    wait_for_service_inactive voxl-mavlink-server 20

    echo "[INFO] Starting raw Qualisys recorder -> ${RAW_QUALISYS_CSV_REMOTE_PATH}"
    python ${ODOM_RECORDER_REMOTE_PATH} --source-topic /qualisys/prog_morphy/odom --output-csv ${RAW_QUALISYS_CSV_REMOTE_PATH} >/tmp/qualisys_odom_record.log 2>&1 &
    QUALISYS_RECORDER_PID=\$!
    sleep 1
    check_background_process "Raw Qualisys recorder" "\$QUALISYS_RECORDER_PID"

    echo "[INFO] Starting MAVLink ODOMETRY relay -> /mavlink/to"
    python ${ODOM_RELAY_REMOTE_PATH} --source-topic /qualisys/prog_morphy/odom --target-topic /mavlink/to --system-id 1 --component-id 197 --quality 0 >/tmp/qualisys_odom_relay.log 2>&1 &
    QUALISYS_RELAY_PID=\$!
    sleep 1
    check_background_process "Qualisys MAVLink relay" "\$QUALISYS_RELAY_PID"

    echo "[INFO] Restarting init_slpi"
    systemctl restart init_slpi
    wait_for_service init_slpi 20

    echo "[INFO] Restarting voxl-imu-server"
    systemctl restart voxl-imu-server
    wait_for_service voxl-imu-server 20

    echo "[INFO] Starting voxl-px4"
    systemctl start voxl-px4
    wait_for_service voxl-px4 20

    sleep 5

    echo "[INFO] Starting battery status recorder -> ${BATTERY_LOG_REMOTE_PATH}"
    python3 ${BATTERY_RECORDER_REMOTE_PATH} --sample-interval-s 0.5 --output-csv ${BATTERY_LOG_REMOTE_PATH} >/tmp/battery_status_log_record.log 2>&1 &
    BATTERY_RECORDER_PID=\$!
    sleep 1
    check_background_process "Battery status recorder" "\$BATTERY_RECORDER_PID"

    echo "[INFO] Qualisys recorder log: /tmp/qualisys_odom_record.log"
    echo "[INFO] Qualisys relay log: /tmp/qualisys_odom_relay.log"
    echo "[INFO] Raw Qualisys CSV: ${RAW_QUALISYS_CSV_REMOTE_PATH}"
    echo "[INFO] Battery status recorder log: /tmp/battery_status_log_record.log"
    echo "[INFO] Battery status CSV: ${BATTERY_LOG_REMOTE_PATH}"
    touch ${MAVROS_READY_REMOTE_PATH}
    echo "[INFO] Pane 1 setup complete"
}

if mocap_pane1_setup; then
    echo "[INFO] Remote shell is ready"
else
    setup_status=\$?
    echo "[ERROR] Pane 1 setup failed with status \$setup_status"
fi

unset -f mocap_pane1_setup
unset -f wait_for_service
unset -f wait_for_service_inactive
unset -f check_background_process
unset -f wait_for_ros_master
EOF
    ${SCP} /tmp/mocap_pane1_init.sh "${VOXL_USER}@${VOXL_IP}:${PANE1_INIT_REMOTE_PATH}"

    cat > /tmp/mocap_pane2_init.sh << EOF
#!/bin/bash

mocap_pane2_run() {
    set -eo pipefail
    source /home/root/catkin_ws/devel/setup.bash
    set -u
    export ROS_MASTER_URI=http://${VOXL_IP}:11311/

    while [ ! -f ${MAVROS_READY_REMOTE_PATH} ]; do
        sleep 0.5
    done

    echo "[INFO] Starting MAVROS via hexbug launch"
    if roslaunch hexbug mavros.launch \
        fcu_url:=udp://127.0.0.1:14557@:14557 \
        gcs_url:=udp://@192.168.5.152 \
        tgt_system:=1 \
        tgt_component:=1 \
        log_output:=screen \
        fcu_protocol:=v2.0 \
        respawn_mavros:=false \
        namespace:=mavros; then
        echo "[INFO] MAVROS exited normally"
    else
        launch_status=\$?
        echo "[ERROR] MAVROS launch exited with status \$launch_status"
    fi
}

if mocap_pane2_run; then
    echo "[INFO] Remote shell is ready"
else
    setup_status=\$?
    echo "[ERROR] Pane 2 setup failed with status \$setup_status"
fi

unset -f mocap_pane2_run
EOF
    ${SCP} /tmp/mocap_pane2_init.sh "${VOXL_USER}@${VOXL_IP}:${PANE2_INIT_REMOTE_PATH}"

    cat > /tmp/mocap_pane0.sh << EOF
#!/bin/bash
${SSH} 'source /home/root/catkin_ws/devel/setup.bash; systemctl stop voxl-px4 || true; export ROS_MASTER_URI=http://${VOXL_IP}:11311/; roscore; bash'
EOF

    rm -f /tmp/mocap_go

    cat > /tmp/mocap_pane1.sh << EOF
#!/bin/bash
read -p "1) Kill the mocap process in the windows PC, 2) run 'roslaunch mocap_quialisys qualisys.launch' in the windows PC, 3) press Enter to continue..."
touch /tmp/mocap_go
exec ${SSH} "bash --init-file ${PANE1_INIT_REMOTE_PATH} -i"
EOF

    cat > /tmp/mocap_pane2.sh << EOF
#!/bin/bash
until [ -f /tmp/mocap_go ]; do sleep 0.2; done
exec ${SSH} "bash --init-file ${PANE2_INIT_REMOTE_PATH} -i"
EOF

    cat > /tmp/mocap_pane3.sh << EOF
#!/bin/bash
until [ -f /tmp/mocap_go ]; do sleep 0.2; done
${SSH} 'source /home/root/catkin_ws/devel/setup.bash; \
while [ ! -f ${MAVROS_READY_REMOTE_PATH} ]; do sleep 0.5; done; \
sleep 4 && rostopic echo /mavros/local_position/pose'
EOF

    chmod +x /tmp/mocap_pane{0,1,2,3}.sh

    cat > /tmp/terminator_mocap.cfg << 'EOF'
[global_config]
[keybindings]
[profiles]
  [[default]]
[layouts]
  [[mocap]]
    [[[root]]]
      type = Window
      parent = ""
    [[[paned_h]]]
      type = HPaned
      parent = root
      ratio = 0.5
    [[[paned_v0]]]
      type = VPaned
      parent = paned_h
      ratio = 0.5
    [[[pane0]]]
      type = Terminal
      parent = paned_v0
      command = /tmp/mocap_pane0.sh
    [[[pane2]]]
      type = Terminal
      parent = paned_v0
      command = /tmp/mocap_pane2.sh
    [[[paned_v1]]]
      type = VPaned
      parent = paned_h
      ratio = 0.5
    [[[pane1]]]
      type = Terminal
      parent = paned_v1
      command = /tmp/mocap_pane1.sh
    [[[pane3]]]
      type = Terminal
      parent = paned_v1
      command = /tmp/mocap_pane3.sh
[plugins]
EOF

    run_local_gui_command_clean terminator --config /tmp/terminator_mocap.cfg --layout mocap
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
