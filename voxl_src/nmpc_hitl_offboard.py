#!/usr/bin/env python3
"""
nmpc_hitl_offboard.py - Arm drone and switch to offboard mode via MAVLink.

Runs on the VOXL (or any host with network access to PX4 mavlink port).
PX4 maintains offboard mode because mc_nmpc_control publishes
offboard_control_mode_s.actuator=true every IMU cycle (no external
trajectory setpoints needed once this module is running).

Usage:
    python3 nmpc_hitl_offboard.py [--duration SEC] [--no-disarm] [--port PORT]

    --duration SEC    How long to hold offboard mode (default: 10)
    --no-disarm       Do not disarm at the end
    --port PORT       PX4 mavlink UDP port (default: 14556)
"""
import sys
import time
import argparse

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed. Run: pip3 install pymavlink")
    sys.exit(1)

PX4_OFFBOARD_MODE = 6  # MAV_CUSTOM_MAIN_MODE_OFFBOARD


def wait_ack(master, cmd, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.5)
        if msg and msg.command == cmd:
            return msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    return False


def run(duration, disarm_after, port):
    conn_str = f'udpin:0.0.0.0:{port + 1}'
    print(f"Connecting to PX4 mavlink (PX4 sends to port {port + 1}, we send to port {port})...")
    master = mavutil.mavlink_connection(conn_str)

    # Send a heartbeat so PX4 learns our address and replies to us
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0)

    print("Waiting for PX4 heartbeat...")
    hb = master.wait_heartbeat(timeout=10)
    if hb is None:
        print("ERROR: No heartbeat from PX4. Is voxl-px4 running?")
        sys.exit(1)
    print(f"PX4 heartbeat: system={master.target_system} component={master.target_component}")

    # Arm with force (bypasses RC and some safety checks for bench testing)
    print("Sending ARM command (force)...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,       # arm
        21196,   # force arm magic number
        0, 0, 0, 0, 0)
    if wait_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM):
        print("Armed OK")
    else:
        print("WARNING: ARM ack not received or rejected. Continuing anyway.")

    time.sleep(0.5)

    # Switch to offboard mode
    print("Switching to OFFBOARD mode...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_OFFBOARD_MODE,
        0, 0, 0, 0, 0)
    if wait_ack(master, mavutil.mavlink.MAV_CMD_DO_SET_MODE):
        print("OFFBOARD mode OK")
    else:
        print("WARNING: SET_MODE ack not received. mc_nmpc_control may not be publishing keepalive yet.")

    print(f"\nHolding for {duration}s. mc_nmpc_control handles offboard keepalive internally.")
    print("Monitor: qshell listener nmpc_control_data -n 5")
    print("Monitor: qshell listener actuator_motors -n 5")

    t0 = time.time()
    while time.time() - t0 < duration:
        # Send GCS heartbeat to keep the link alive
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0)
        time.sleep(1.0)
        elapsed = time.time() - t0
        print(f"  t={elapsed:.0f}s / {duration}s", end='\r', flush=True)

    print()

    if disarm_after:
        print("Disarming...")
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,  # disarm
            0, 0, 0, 0, 0, 0)
        if wait_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM):
            print("Disarmed OK")
        else:
            print("WARNING: DISARM ack not received")
    else:
        print("--no-disarm set: vehicle remains armed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float)
    parser.add_argument('--no-disarm', action='store_true')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    run(
        duration=args.duration,
        disarm_after=not args.no_disarm,
        port=args.port,
    )
