#!/bin/bash

wait_for_service() {
    local service_name="$1"
    local timeout_s="$2"
    local deadline_s=$(($(date +%s) + timeout_s))
    while [ $(date +%s) -le "$deadline_s" ]; do
        if systemctl is-active --quiet "$service_name"; then
            echo "[OK] $service_name is active"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] $service_name did not become active within ${timeout_s}s"
    systemctl status --no-pager "$service_name" || true
    return 1
}

check_background_process() {
    local process_label="$1"
    local process_pid="$2"
    if ps -p "$process_pid" >/dev/null 2>&1; then
        echo "[OK] $process_label is running with pid $process_pid"
        return 0
    fi
    echo "[ERROR] $process_label failed to stay alive"
    return 1
}

wait_for_ros_master() {
    local timeout_s="$1"
    local deadline_s=$(($(date +%s) + timeout_s))
    while [ $(date +%s) -le "$deadline_s" ]; do
        if rostopic list >/dev/null 2>&1; then
            echo "[OK] ROS master is reachable"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] ROS master did not become reachable within ${timeout_s}s"
    return 1
}

mocap_pane1_setup() {
    set -eo pipefail
    source /home/root/catkin_ws/devel/setup.bash
    set -u

    echo "[INFO] Preparing Qualisys recorder and MAVLink relay on $(hostname)"
    chmod +x /home/root/qualisys_odom_to_px4_mavlink.py /home/root/record_qualisys_odom_csv.py
    rm -f /tmp/qualisys_odom.csv
    export ROS_MASTER_URI=http://192.168.5.5:11311/
    wait_for_ros_master 30

    echo "[INFO] Stopping any previous recorder or relay processes"
    pkill -f "[r]ecord_qualisys_odom_csv.py" || true
    pkill -f "[q]ualisys_odom_to_px4_mavlink.py" || true
    sleep 1

    echo "[INFO] Starting raw Qualisys recorder -> /tmp/qualisys_odom.csv"
    python /home/root/record_qualisys_odom_csv.py --source-topic /qualisys/prog_morphy/odom --output-csv /tmp/qualisys_odom.csv >/tmp/qualisys_odom_record.log 2>&1 &
    QUALISYS_RECORDER_PID=$!
    sleep 1
    check_background_process "Raw Qualisys recorder" "$QUALISYS_RECORDER_PID"

    echo "[INFO] Starting MAVLink ODOMETRY relay -> /mavlink/to"
    python /home/root/qualisys_odom_to_px4_mavlink.py --source-topic /qualisys/prog_morphy/odom --target-topic /mavlink/to --system-id 1 --component-id 197 --quality 0 >/tmp/qualisys_odom_relay.log 2>&1 &
    QUALISYS_RELAY_PID=$!
    sleep 1
    check_background_process "Qualisys MAVLink relay" "$QUALISYS_RELAY_PID"

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

    echo "[INFO] Qualisys recorder log: /tmp/qualisys_odom_record.log"
    echo "[INFO] Qualisys relay log: /tmp/qualisys_odom_relay.log"
    echo "[INFO] Raw Qualisys CSV: /tmp/qualisys_odom.csv"
    echo "[INFO] Pane 1 setup complete"
}

if mocap_pane1_setup; then
    echo "[INFO] Remote shell is ready"
else
    setup_status=$?
    echo "[ERROR] Pane 1 setup failed with status $setup_status"
fi

unset -f mocap_pane1_setup
unset -f wait_for_service
unset -f check_background_process
unset -f wait_for_ros_master
