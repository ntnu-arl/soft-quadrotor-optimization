#!/usr/bin/env python3
"""Estimate the tilt of the mocap frame relative to gravity from a training run.

Usage:

    python3 voxl_src/calibrate_mocap_orientation.py <run-directory>

The run directory must be an extracted mocap run containing a circular-flight
segment (for example results/real_world_experiments/voxl_mocap/<timestamp>).
The script reports whether the Qualisys world frame is misaligned with
gravity and by how much. A significant tilt means the mocap system itself
needs recalibrating; a near-zero tilt means the mocap calibration is good.

Method: during circular flight the horizontal acceleration is a large,
high-SNR signal at the lap frequency (the rotating centripetal acceleration).
If the mocap frame is tilted by a small rotation theta relative to the
gravity-aligned frame, the vertical acceleration measured by the IMU (rotated
to world with the PX4 EKF attitude, which is gravity-aligned) differs from
the vertical acceleration of the double-differentiated mocap positions by
the horizontal acceleration leaking through the tilt:

    a_gravity = (I + [theta]x) a_mocap        (small-angle)
    a_gravity_z - a_mocap_z = theta_x * a_mocap_y - theta_y * a_mocap_x

All signals are reduced to their complex amplitude (phasor) at the lap
frequency before solving, so the estimate only uses the frequency where the
centripetal excitation dominates and is immune to accelerometer bias, slow
drifts, and the broadband noise that double differentiation of mocap
positions produces.

The fit returns the TOTAL mocap-to-gravity tilt: it uses the raw Qualisys
CSV and the PX4 EKF attitude (which is gravity-aligned through the
accelerometer). After the mocap system is recalibrated correctly, refitting
on a new flight returns a tilt near zero.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pyulog import ULog
except ModuleNotFoundError as exc:
    raise SystemExit("pyulog is required to calibrate the mocap orientation") from exc

GRAVITY_M_S2 = 9.81
MIN_REFERENCE_SPEED_M_S = 0.5
REFERENCE_SPEED_WINDOW_RATIO = 0.95
WINDOW_EDGE_MARGIN_S = 0.5
MIN_HORIZONTAL_ACCEL_AMPLITUDE_M_S2 = 0.5
MAX_HORIZONTAL_AMPLITUDE_MISMATCH_RATIO = 0.15
MAX_HORIZONTAL_PHASE_GAP_ERROR_RAD = 0.10
MAX_IMU_MOCAP_HORIZONTAL_MISMATCH_RATIO = 0.10
MIN_FIT_LAPS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the mocap-to-gravity tilt from a circular-flight training run"
    )
    parser.add_argument("run_dir")
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise RuntimeError(f"required file does not exist: {path}")
    return path


def get_single_topic(ulog, topic_name):
    matches = [dataset for dataset in ulog.data_list if dataset.name == topic_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one '{topic_name}' dataset, found {len(matches)}")
    return matches[0]


def load_base_timestamp_us(flight_normalized_path):
    flight_df = pd.read_csv(flight_normalized_path, usecols=["time_s", "timestamp_us"])
    if flight_df.empty:
        raise RuntimeError(f"{flight_normalized_path} is empty")
    return float(flight_df["timestamp_us"].iloc[0]) - float(flight_df["time_s"].iloc[0]) * 1e6


def detect_circle_window(run_dir):
    solver_df = pd.read_csv(
        require_file(run_dir / "nmpc_solver_data_combined_normalized.csv"),
        usecols=["time_s", "sp_vx", "sp_vy"],
    )
    reference_speed = np.hypot(solver_df["sp_vx"].to_numpy(float), solver_df["sp_vy"].to_numpy(float))
    peak_speed = reference_speed.max()
    if peak_speed < MIN_REFERENCE_SPEED_M_S:
        raise RuntimeError(
            f"peak reference speed is {peak_speed:.3f} m/s; the run does not contain a "
            f"circular trajectory segment usable for calibration"
        )
    cruise_times = solver_df["time_s"].to_numpy(float)[reference_speed >= REFERENCE_SPEED_WINDOW_RATIO * peak_speed]
    return cruise_times.min() + WINDOW_EDGE_MARGIN_S, cruise_times.max() - WINDOW_EDGE_MARGIN_S


def load_mocap_positions_time_s(run_dir, base_us):
    raw_df = pd.read_csv(
        require_file(run_dir / "qualisys_odom.csv"),
        usecols=["timestamp_ns", "position_x", "position_y", "position_z"],
    )
    align_df = pd.read_csv(require_file(run_dir / "qualisys_px4_time_alignment.csv"))
    raw_us = (raw_df["timestamp_ns"].to_numpy(np.int64) // 1000).astype(np.float64)
    px4_us = np.interp(
        raw_us,
        align_df["qualisys_timestamp_us"].to_numpy(np.float64),
        align_df["px4_timestamp_us"].to_numpy(np.float64),
    )
    time_s = (px4_us - base_us) / 1e6
    positions_enu = raw_df[["position_x", "position_y", "position_z"]].to_numpy(np.float64)
    return time_s, positions_enu


def load_imu_accel_gravity_enu(run_dir, base_us, t_start, t_end):
    ulog = ULog(str(require_file(run_dir / "px4_run.ulg")), ["sensor_combined", "vehicle_attitude"])
    sensor = get_single_topic(ulog, "sensor_combined")
    attitude = get_single_topic(ulog, "vehicle_attitude")

    imu_time_s = (sensor.data["timestamp"].astype(np.float64) - base_us) / 1e6
    specific_force_body = np.column_stack(
        [sensor.data[f"accelerometer_m_s2[{axis}]"].astype(np.float64) for axis in range(3)]
    )
    attitude_time_s = (attitude.data["timestamp_sample"].astype(np.float64) - base_us) / 1e6
    quaternions = np.column_stack([attitude.data[f"q[{axis}]"].astype(np.float64) for axis in range(4)])

    mask = (imu_time_s >= t_start) & (imu_time_s <= t_end)
    if not np.any(mask):
        raise RuntimeError("no IMU samples inside the fit window")
    imu_time_s = imu_time_s[mask]
    specific_force_body = specific_force_body[mask]

    interpolated = np.column_stack(
        [np.interp(imu_time_s, attitude_time_s, quaternions[:, axis]) for axis in range(4)]
    )
    interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)
    qw, qx, qy, qz = interpolated.T
    rotation_ned_frd = np.empty((len(imu_time_s), 3, 3))
    rotation_ned_frd[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    rotation_ned_frd[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rotation_ned_frd[:, 0, 2] = 2 * (qx * qz + qw * qy)
    rotation_ned_frd[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rotation_ned_frd[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    rotation_ned_frd[:, 1, 2] = 2 * (qy * qz - qw * qx)
    rotation_ned_frd[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rotation_ned_frd[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rotation_ned_frd[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)

    accel_ned = np.einsum("nij,nj->ni", rotation_ned_frd, specific_force_body)
    accel_ned[:, 2] += GRAVITY_M_S2
    accel_enu = np.column_stack([accel_ned[:, 1], accel_ned[:, 0], -accel_ned[:, 2]])
    return imu_time_s, accel_enu


def fit_phasor(theta, values):
    regressors = np.column_stack(
        [
            np.ones_like(theta),
            theta,
            np.cos(theta),
            np.sin(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(regressors, values, rcond=None)
    return coefficients[2] - 1j * coefficients[3]


def solve_tilt(accel_x_phasor, accel_y_phasor, vertical_inconsistency_phasor):
    matrix = np.array(
        [
            [accel_y_phasor.real, -accel_x_phasor.real],
            [accel_y_phasor.imag, -accel_x_phasor.imag],
        ]
    )
    rhs = np.array([vertical_inconsistency_phasor.real, vertical_inconsistency_phasor.imag])
    theta_x, theta_y = np.linalg.solve(matrix, rhs)
    return theta_x, theta_y


def estimate_window(mocap_theta, mocap_positions, imu_theta, imu_accel_up, lap_omega):
    position_phasors = [fit_phasor(mocap_theta, mocap_positions[:, axis]) for axis in range(3)]
    accel_phasors_mocap = [-(lap_omega ** 2) * phasor for phasor in position_phasors]
    accel_up_phasor_imu = fit_phasor(imu_theta, imu_accel_up)
    inconsistency = accel_up_phasor_imu - accel_phasors_mocap[2]
    theta_x, theta_y = solve_tilt(accel_phasors_mocap[0], accel_phasors_mocap[1], inconsistency)
    return theta_x, theta_y, accel_phasors_mocap, accel_up_phasor_imu, inconsistency


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise RuntimeError(f"run directory does not exist: {run_dir}")

    base_us = load_base_timestamp_us(require_file(run_dir / "px4_flight_log_normalized.csv"))
    fit_start_s, fit_end_s = detect_circle_window(run_dir)
    mocap_time_s, mocap_positions = load_mocap_positions_time_s(run_dir, base_us)

    window = (mocap_time_s >= fit_start_s) & (mocap_time_s <= fit_end_s)
    if window.sum() < 100:
        raise RuntimeError(f"only {window.sum()} mocap samples inside the fit window")
    mocap_time_s = mocap_time_s[window]
    mocap_positions = mocap_positions[window]

    center_xy = mocap_positions[:, :2].mean(axis=0)
    mocap_theta = np.unwrap(
        np.arctan2(mocap_positions[:, 1] - center_xy[1], mocap_positions[:, 0] - center_xy[0])
    )

    # Trim the window end to an integer number of laps to minimize spectral leakage.
    whole_laps = int(abs(mocap_theta - mocap_theta[0]).max() // (2 * np.pi))
    if whole_laps < MIN_FIT_LAPS:
        raise RuntimeError(f"fit window covers only {whole_laps} whole laps, need at least {MIN_FIT_LAPS}")
    inside_whole_laps = abs(mocap_theta - mocap_theta[0]) <= whole_laps * 2 * np.pi
    last_index = int(np.flatnonzero(inside_whole_laps).max())
    mocap_time_s = mocap_time_s[: last_index + 1]
    mocap_positions = mocap_positions[: last_index + 1]
    mocap_theta = mocap_theta[: last_index + 1]
    fit_end_s = float(mocap_time_s[-1])
    lap_omega = np.polyfit(mocap_time_s, mocap_theta, 1)[0]

    imu_time_s, imu_accel_enu = load_imu_accel_gravity_enu(run_dir, base_us, fit_start_s, fit_end_s)
    imu_theta = np.interp(imu_time_s, mocap_time_s, mocap_theta)

    theta_x, theta_y, accel_phasors_mocap, accel_up_phasor_imu, inconsistency = estimate_window(
        mocap_theta, mocap_positions, imu_theta, imu_accel_enu[:, 2], lap_omega
    )

    amp_x = abs(accel_phasors_mocap[0])
    amp_y = abs(accel_phasors_mocap[1])
    if min(amp_x, amp_y) < MIN_HORIZONTAL_ACCEL_AMPLITUDE_M_S2:
        raise RuntimeError(
            f"horizontal centripetal acceleration too small for a tilt fit: "
            f"|ax|={amp_x:.3f} |ay|={amp_y:.3f} m/s^2 (need a circular-flight segment)"
        )
    if abs(amp_x - amp_y) / max(amp_x, amp_y) > MAX_HORIZONTAL_AMPLITUDE_MISMATCH_RATIO:
        raise RuntimeError(f"horizontal accel amplitudes inconsistent with a circle: |ax|={amp_x:.3f} |ay|={amp_y:.3f}")
    phase_gap = np.angle(accel_phasors_mocap[1] / accel_phasors_mocap[0])
    if abs(abs(phase_gap) - np.pi / 2) > MAX_HORIZONTAL_PHASE_GAP_ERROR_RAD:
        raise RuntimeError(f"horizontal accel phase gap inconsistent with a circle: {phase_gap:+.3f} rad")
    for axis, label in ((0, "x"), (1, "y")):
        imu_phasor = fit_phasor(imu_theta, imu_accel_enu[:, axis])
        mismatch = abs(imu_phasor - accel_phasors_mocap[axis]) / abs(accel_phasors_mocap[axis])
        if mismatch > MAX_IMU_MOCAP_HORIZONTAL_MISMATCH_RATIO:
            raise RuntimeError(
                f"IMU and mocap horizontal accel disagree on axis {label} by {mismatch * 100:.1f} percent; "
                f"attitude or time alignment is not trustworthy"
            )

    lap_period_s = 2 * np.pi / abs(lap_omega)
    per_lap = []
    lap_start = fit_start_s
    while lap_start + lap_period_s <= fit_end_s + 1e-9:
        lap_mask_mocap = (mocap_time_s >= lap_start) & (mocap_time_s <= lap_start + lap_period_s)
        lap_mask_imu = (imu_time_s >= lap_start) & (imu_time_s <= lap_start + lap_period_s)
        lap_tx, lap_ty, *_ = estimate_window(
            mocap_theta[lap_mask_mocap],
            mocap_positions[lap_mask_mocap],
            imu_theta[lap_mask_imu],
            imu_accel_enu[lap_mask_imu, 2],
            lap_omega,
        )
        per_lap.append((lap_tx, lap_ty))
        lap_start += lap_period_s
    per_lap = np.array(per_lap)

    print(
        f"fit window: {fit_start_s:.2f} s to {fit_end_s:.2f} s "
        f"({whole_laps} whole laps, period {lap_period_s:.3f} s)"
    )
    print(f"circle check: |ax|={amp_x:.3f} |ay|={amp_y:.3f} m/s^2, phase gap {phase_gap:+.3f} rad")
    print(f"vertical inconsistency at lap frequency: {abs(inconsistency) * 100:.2f} cm/s^2")
    print(f"IMU vertical accel phasor at lap frequency: {abs(accel_up_phasor_imu) * 100:.2f} cm/s^2")
    print()
    tilt_deg = np.degrees(np.hypot(theta_x, theta_y))
    print(f"fitted TOTAL mocap-to-gravity rotation vector (ENU, rad): ({theta_x:+.6f}, {theta_y:+.6f}, 0.0)")
    print(f"fitted tilt: {tilt_deg:.3f} deg")
    print("per-lap estimates (stability check):")
    for index, (lap_tx, lap_ty) in enumerate(per_lap):
        print(f"  lap {index}: ({lap_tx:+.6f}, {lap_ty:+.6f}) rad = {np.degrees(np.hypot(lap_tx, lap_ty)):.3f} deg")
    scatter = per_lap.std(axis=0)
    print(f"per-lap scatter: ({scatter[0]:.6f}, {scatter[1]:.6f}) rad")
    print()
    print(
        "a tilt well above the per-lap scatter means the mocap frame is misaligned "
        "with gravity and the mocap system needs recalibrating"
    )


if __name__ == "__main__":
    main()
