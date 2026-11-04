from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nmpc_controller import (
    PersistentDisturbanceEstimator,
    _body_velocity_drag_force_np,
)


FLAG_REINIT = 0x01

PARAM_MASS = 0
PARAM_IXX = 1
PARAM_IXY = 2
PARAM_IXZ = 3
PARAM_IYY = 4
PARAM_IYZ = 5
PARAM_IZZ = 6
PARAM_GRAVITY = 7
PARAM_ALLOC = 23
PARAM_KF = 47
PARAM_COM = 55
PARAM_DIST_FORCE = 58
PARAM_DIST_TORQUE = 61
PARAM_U_PREV = 64
NUM_PARAMS = 68
NUM_MOTORS = 4

DIST_FORCE_BOUND = PersistentDisturbanceEstimator.FORCE_BOUND.astype(float, copy=True)
DIST_TORQUE_BOUND = PersistentDisturbanceEstimator.TORQUE_BOUND.astype(float, copy=True)
DIST_INITIAL_SIGMA_RATIO = float(PersistentDisturbanceEstimator.INITIAL_SIGMA_RATIO)
DIST_READY_SIGMA_RATIO = float(PersistentDisturbanceEstimator.READY_SIGMA_RATIO)
DIST_FORCE_VARIANCE_FLOOR_SIGMA_RATIO = float(PersistentDisturbanceEstimator.FORCE_VARIANCE_FLOOR_SIGMA_RATIO)
DIST_TORQUE_VARIANCE_FLOOR_SIGMA_RATIO = float(PersistentDisturbanceEstimator.TORQUE_VARIANCE_FLOOR_SIGMA_RATIO)
DIST_FORCE_PROCESS_SIGMA_RATIO = float(PersistentDisturbanceEstimator.FORCE_PROCESS_SIGMA_RATIO)
DIST_TORQUE_PROCESS_SIGMA_RATIO = float(PersistentDisturbanceEstimator.TORQUE_PROCESS_SIGMA_RATIO)
DIST_FORCE_MEAS_SIGMA_RATIO = float(PersistentDisturbanceEstimator.FORCE_MEAS_SIGMA_RATIO)
DIST_TORQUE_MEAS_SIGMA_RATIO = float(PersistentDisturbanceEstimator.TORQUE_MEAS_SIGMA_RATIO)
DIST_FORCE_ABS_GATE_RATIO = float(PersistentDisturbanceEstimator.FORCE_ABS_GATE_RATIO)
DIST_TORQUE_ABS_GATE_RATIO = float(PersistentDisturbanceEstimator.TORQUE_ABS_GATE_RATIO)
DIST_FORCE_INNOVATION_GATE_NIS = float(PersistentDisturbanceEstimator.FORCE_INNOVATION_GATE_NIS)
DIST_TORQUE_INNOVATION_GATE_NIS = float(PersistentDisturbanceEstimator.TORQUE_INNOVATION_GATE_NIS)
DIST_WARMUP_TIME_S = float(PersistentDisturbanceEstimator.WARMUP_TIME_S)
DIST_WARMUP_MIN_SAMPLES = int(PersistentDisturbanceEstimator.WARMUP_MIN_SAMPLES)

FILTER_FLAG_RESET = 1 << 0
FILTER_FLAG_WARMUP = 1 << 1
FILTER_FLAG_NOT_READY = 1 << 2
FILTER_FLAG_GATED = 1 << 3
FILTER_FLAG_SATURATED = 1 << 4
FILTER_FLAG_VALID = 1 << 5
FILTER_FLAG_MEASUREMENT_READY = 1 << 6


REQUIRED_SOLVER_COLUMNS = [
    "t_us",
    "seq",
    "packet_flags",
    "sample_timestamp_us",
    "position_velocity_timestamp_us",
    "attitude_timestamp_us",
    "angular_velocity_timestamp_us",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
    "qw",
    "qx",
    "qy",
    "qz",
    "wx",
    "wy",
    "wz",
    "estimated_rps0",
    "estimated_rps1",
    "estimated_rps2",
    "estimated_rps3",
    "fixed_dist_force_bx",
    "fixed_dist_force_by",
    "fixed_dist_force_bz",
    "fixed_dist_torque_bx",
    "fixed_dist_torque_by",
    "fixed_dist_torque_bz",
]
REQUIRED_SOLVER_COLUMNS.extend([f"p{param_index:02d}" for param_index in range(NUM_PARAMS)])


def require_columns(dataframe, column_names, dataframe_name):
    missing_columns = [column_name for column_name in column_names if column_name not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in {dataframe_name}: {missing_columns}")


def _clip_scalar(value, lower, upper):
    return min(max(value, lower), upper)


def _cross3(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=float,
    )


def _quat_to_rotmat(quaternion):
    quat = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quat)

    if not np.isfinite(norm) or not norm > 1e-9:
        raise RuntimeError(f"Invalid quaternion norm {norm}")

    qw, qx, qy, qz = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _compute_body_wrench_from_rps(params, motor_rps):
    actual_forces = np.zeros(NUM_MOTORS, dtype=float)
    for motor_index in range(NUM_MOTORS):
        kf = params[PARAM_KF + motor_index]
        if not np.isfinite(kf) or not kf > 0.0:
            raise RuntimeError(f"Invalid thrust coefficient p[{PARAM_KF + motor_index}]={kf}")
        actual_forces[motor_index] = kf * motor_rps[motor_index] * motor_rps[motor_index]

    force_body = np.zeros(3, dtype=float)
    torque_body = np.zeros(3, dtype=float)
    for axis in range(3):
        for motor_index in range(NUM_MOTORS):
            alloc_base = PARAM_ALLOC + 6 * motor_index
            force_body[axis] += params[alloc_base + axis] * actual_forces[motor_index]
            torque_body[axis] += params[alloc_base + 3 + axis] * actual_forces[motor_index]
    return force_body, torque_body


def compute_body_velocity_drag_force(velocity_body):
    return _body_velocity_drag_force_np(velocity_body)


def _compute_disturbance_bounds(params):
    del params
    return DIST_FORCE_BOUND.copy(), DIST_TORQUE_BOUND.copy()


def _build_inertia_matrix(params):
    return np.array(
        [
            [params[PARAM_IXX], params[PARAM_IXY], params[PARAM_IXZ]],
            [params[PARAM_IXY], params[PARAM_IYY], params[PARAM_IYZ]],
            [params[PARAM_IXZ], params[PARAM_IYZ], params[PARAM_IZZ]],
        ],
        dtype=float,
    )


def _extract_params(row):
    return np.array([float(row[f"p{param_index:02d}"]) for param_index in range(NUM_PARAMS)], dtype=float)


def _build_attitude_lookup(solver_df):
    attitude_df = solver_df[
        ["attitude_timestamp_us", "qw", "qx", "qy", "qz"]
    ].drop_duplicates(subset=["attitude_timestamp_us"], keep="last").sort_values("attitude_timestamp_us")

    timestamps = attitude_df["attitude_timestamp_us"].to_numpy(dtype=np.int64)
    quaternions = attitude_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)

    if len(timestamps) == 0:
        raise RuntimeError("No attitude samples available for offline disturbance replay")

    return timestamps, quaternions


def _lookup_attitude(attitude_timestamps_us, attitude_quaternions, target_timestamp_us):
    attitude_index = np.searchsorted(attitude_timestamps_us, target_timestamp_us, side="right") - 1
    if attitude_index < 0:
        raise RuntimeError(
            f"No attitude sample available at or before timestamp {int(target_timestamp_us)} for offline replay"
        )
    return int(attitude_timestamps_us[attitude_index]), attitude_quaternions[attitude_index]


@dataclass
class ChannelState:
    estimated: np.ndarray
    variance: np.ndarray
    bound: np.ndarray
    last_timestamp_us: int
    sample_count: int
    warmup_elapsed_s: float
    valid: bool
    last_flags: int


def _initialize_channel_state(bound, timestamp_us):
    sigma = DIST_INITIAL_SIGMA_RATIO * bound
    return ChannelState(
        estimated=np.zeros(3, dtype=float),
        variance=sigma * sigma,
        bound=np.array(bound, dtype=float),
        last_timestamp_us=int(timestamp_us),
        sample_count=0,
        warmup_elapsed_s=0.0,
        valid=False,
        last_flags=FILTER_FLAG_RESET | FILTER_FLAG_WARMUP | FILTER_FLAG_NOT_READY,
    )


def _update_channel_state(
    state,
    measurement,
    bound,
    dt_s,
    process_sigma_ratio,
    measurement_sigma_ratio,
    abs_gate_ratio,
    variance_floor_sigma_ratio,
    innovation_gate_nis,
):
    process_var = np.square(process_sigma_ratio * bound) * dt_s
    meas_var = np.square(measurement_sigma_ratio * bound)
    min_var = np.square(variance_floor_sigma_ratio * bound)
    max_var = np.square(bound)

    state.bound = np.array(bound, dtype=float)
    state.warmup_elapsed_s += dt_s
    state.sample_count += 1

    state.variance = np.clip(state.variance + process_var, min_var, max_var)
    innovation = measurement - state.estimated
    innovation_cov = state.variance + meas_var
    innovation_nis = float(np.sum(np.square(innovation) / innovation_cov))
    gated = bool(np.any(np.abs(measurement) > abs_gate_ratio * bound) or innovation_nis > innovation_gate_nis)

    if not gated:
        gain = state.variance / innovation_cov
        state.estimated = state.estimated + gain * innovation
        state.variance = np.clip((1.0 - gain) * state.variance, min_var, max_var)

    estimated_before_clip = state.estimated.copy()
    state.estimated = np.clip(state.estimated, -bound, bound)
    saturated = bool(np.any(np.abs(state.estimated - estimated_before_clip) > 1e-12))

    ready = bool(np.all(np.sqrt(state.variance) <= DIST_READY_SIGMA_RATIO * bound))
    state.valid = bool(
        state.warmup_elapsed_s >= DIST_WARMUP_TIME_S
        and state.sample_count >= DIST_WARMUP_MIN_SAMPLES
        and ready
    )

    confidence = float(1.0 - np.mean(np.sqrt(state.variance) / bound))

    flags = FILTER_FLAG_MEASUREMENT_READY
    if not state.valid:
        flags |= FILTER_FLAG_WARMUP | FILTER_FLAG_NOT_READY
    if gated:
        flags |= FILTER_FLAG_GATED
    if saturated:
        flags |= FILTER_FLAG_SATURATED
    if state.valid:
        flags |= FILTER_FLAG_VALID
    if state.sample_count == 1:
        flags |= FILTER_FLAG_RESET

    state.last_flags = flags

    return {
        "measurement": measurement,
        "innovation_nis": innovation_nis,
        "confidence": confidence,
        "flags": flags,
        "valid": state.valid,
    }


def estimate_offline_disturbance(solver_df):
    require_columns(solver_df, REQUIRED_SOLVER_COLUMNS, "solver_df")

    solver_df = solver_df.sort_values(["sample_timestamp_us", "seq"]).reset_index(drop=True)
    attitude_timestamps_us, attitude_quaternions = _build_attitude_lookup(solver_df)

    rows = []
    force_state = None
    torque_state = None
    previous_velocity = None
    previous_omega = None

    for row_index, row in solver_df.iterrows():
        params = _extract_params(row)
        if not np.all(np.isfinite(params)):
            raise RuntimeError(f"Non-finite parameter values at solver row {row_index}")

        force_bound, torque_bound = _compute_disturbance_bounds(params)
        mass = params[PARAM_MASS]
        gravity_vec = params[PARAM_GRAVITY:PARAM_GRAVITY + 3]
        inertia = _build_inertia_matrix(params)
        r_com = params[PARAM_COM:PARAM_COM + 3]
        motor_rps = row[[f"estimated_rps{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
        force_body, torque_body = _compute_body_wrench_from_rps(params, motor_rps)

        position_velocity_timestamp_us = int(row["position_velocity_timestamp_us"])
        angular_velocity_timestamp_us = int(row["angular_velocity_timestamp_us"])
        packet_flags = int(row["packet_flags"])
        velocity = row[["vx", "vy", "vz"]].to_numpy(dtype=float)
        omega = row[["wx", "wy", "wz"]].to_numpy(dtype=float)

        reset_now = force_state is None or torque_state is None or (packet_flags & FLAG_REINIT) != 0
        if reset_now:
            force_state = _initialize_channel_state(force_bound, position_velocity_timestamp_us)
            torque_state = _initialize_channel_state(torque_bound, angular_velocity_timestamp_us)
            previous_velocity = velocity.copy()
            previous_omega = omega.copy()

        force_measurement_valid = False
        torque_measurement_valid = False
        force_measurement = np.full(3, np.nan, dtype=float)
        torque_measurement = np.full(3, np.nan, dtype=float)
        force_dt_s = np.nan
        torque_dt_s = np.nan
        force_attitude_timestamp_us = np.nan
        torque_attitude_timestamp_us = np.nan
        force_innovation_nis = np.nan
        torque_innovation_nis = np.nan
        force_confidence = float(1.0 - np.mean(np.sqrt(force_state.variance) / force_state.bound))
        torque_confidence = float(1.0 - np.mean(np.sqrt(torque_state.variance) / torque_state.bound))
        force_flags = force_state.last_flags & ~FILTER_FLAG_MEASUREMENT_READY
        torque_flags = torque_state.last_flags & ~FILTER_FLAG_MEASUREMENT_READY

        if not reset_now and position_velocity_timestamp_us < force_state.last_timestamp_us:
            raise RuntimeError(
                "Position/velocity timestamp moved backwards during offline replay: "
                f"current={position_velocity_timestamp_us} previous={force_state.last_timestamp_us}"
            )
        if not reset_now and angular_velocity_timestamp_us < torque_state.last_timestamp_us:
            raise RuntimeError(
                "Angular-velocity timestamp moved backwards during offline replay: "
                f"current={angular_velocity_timestamp_us} previous={torque_state.last_timestamp_us}"
            )

        if not reset_now and position_velocity_timestamp_us > force_state.last_timestamp_us:
            force_measurement_valid = True
            force_dt_s = (position_velocity_timestamp_us - force_state.last_timestamp_us) * 1e-6
            force_attitude_timestamp_us, force_quat = _lookup_attitude(
                attitude_timestamps_us,
                attitude_quaternions,
                position_velocity_timestamp_us,
            )
            force_rotation = _quat_to_rotmat(force_quat)
            measured_world_acc = (velocity - previous_velocity) / force_dt_s
            velocity_body = force_rotation.T @ velocity
            measured_body_acc = force_rotation.T @ measured_world_acc
            gravity_body = force_rotation.T @ gravity_vec
            measured_body_force = mass * (measured_body_acc - gravity_body)
            drag_force_body = compute_body_velocity_drag_force(velocity_body)
            force_measurement = measured_body_force - force_body - drag_force_body
            force_update = _update_channel_state(
                force_state,
                force_measurement,
                force_bound,
                force_dt_s,
                DIST_FORCE_PROCESS_SIGMA_RATIO,
                DIST_FORCE_MEAS_SIGMA_RATIO,
                DIST_FORCE_ABS_GATE_RATIO,
                DIST_FORCE_VARIANCE_FLOOR_SIGMA_RATIO,
                DIST_FORCE_INNOVATION_GATE_NIS,
            )
            force_innovation_nis = force_update["innovation_nis"]
            force_confidence = force_update["confidence"]
            force_flags = force_update["flags"]
            previous_velocity = velocity.copy()
            force_state.last_timestamp_us = position_velocity_timestamp_us

        if not reset_now and angular_velocity_timestamp_us > torque_state.last_timestamp_us:
            torque_measurement_valid = True
            torque_dt_s = (angular_velocity_timestamp_us - torque_state.last_timestamp_us) * 1e-6
            torque_attitude_timestamp_us, torque_quat = _lookup_attitude(
                attitude_timestamps_us,
                attitude_quaternions,
                angular_velocity_timestamp_us,
            )
            torque_rotation = _quat_to_rotmat(torque_quat)
            omega_dot = (omega - previous_omega) / torque_dt_s
            inertia_omega_dot = inertia @ omega_dot
            inertia_omega = inertia @ omega
            omega_cross_inertia_omega = _cross3(omega, inertia_omega)
            gravity_force_body = torque_rotation.T @ (mass * gravity_vec)
            torque_gravity = _cross3(r_com, gravity_force_body)
            torque_measurement = inertia_omega_dot - torque_body - torque_gravity + omega_cross_inertia_omega
            torque_update = _update_channel_state(
                torque_state,
                torque_measurement,
                torque_bound,
                torque_dt_s,
                DIST_TORQUE_PROCESS_SIGMA_RATIO,
                DIST_TORQUE_MEAS_SIGMA_RATIO,
                DIST_TORQUE_ABS_GATE_RATIO,
                DIST_TORQUE_VARIANCE_FLOOR_SIGMA_RATIO,
                DIST_TORQUE_INNOVATION_GATE_NIS,
            )
            torque_innovation_nis = torque_update["innovation_nis"]
            torque_confidence = torque_update["confidence"]
            torque_flags = torque_update["flags"]
            previous_omega = omega.copy()
            torque_state.last_timestamp_us = angular_velocity_timestamp_us

        row_out = {
            "t_us": float(row["t_us"]),
            "seq": int(row["seq"]),
            "packet_flags": packet_flags,
            "sample_timestamp_us": int(row["sample_timestamp_us"]),
            "position_velocity_timestamp_us": position_velocity_timestamp_us,
            "attitude_timestamp_us": int(row["attitude_timestamp_us"]),
            "angular_velocity_timestamp_us": angular_velocity_timestamp_us,
            "force_measurement_valid": int(force_measurement_valid),
            "force_dt_s": force_dt_s,
            "force_attitude_timestamp_us": force_attitude_timestamp_us,
            "force_measurement_bx": force_measurement[0],
            "force_measurement_by": force_measurement[1],
            "force_measurement_bz": force_measurement[2],
            "estimated_force_bx": force_state.estimated[0],
            "estimated_force_by": force_state.estimated[1],
            "estimated_force_bz": force_state.estimated[2],
            "force_var_x": force_state.variance[0],
            "force_var_y": force_state.variance[1],
            "force_var_z": force_state.variance[2],
            "force_bound_x": force_state.bound[0],
            "force_bound_y": force_state.bound[1],
            "force_bound_z": force_state.bound[2],
            "force_confidence": force_confidence,
            "force_innovation_nis": force_innovation_nis,
            "force_flags": force_flags,
            "force_valid": int(force_state.valid),
            "force_warmup_s": force_state.warmup_elapsed_s,
            "force_samples": force_state.sample_count,
            "torque_measurement_valid": int(torque_measurement_valid),
            "torque_dt_s": torque_dt_s,
            "torque_attitude_timestamp_us": torque_attitude_timestamp_us,
            "torque_measurement_bx": torque_measurement[0],
            "torque_measurement_by": torque_measurement[1],
            "torque_measurement_bz": torque_measurement[2],
            "estimated_torque_bx": torque_state.estimated[0],
            "estimated_torque_by": torque_state.estimated[1],
            "estimated_torque_bz": torque_state.estimated[2],
            "torque_var_x": torque_state.variance[0],
            "torque_var_y": torque_state.variance[1],
            "torque_var_z": torque_state.variance[2],
            "torque_bound_x": torque_state.bound[0],
            "torque_bound_y": torque_state.bound[1],
            "torque_bound_z": torque_state.bound[2],
            "torque_confidence": torque_confidence,
            "torque_innovation_nis": torque_innovation_nis,
            "torque_flags": torque_flags,
            "torque_valid": int(torque_state.valid),
            "torque_warmup_s": torque_state.warmup_elapsed_s,
            "torque_samples": torque_state.sample_count,
            "combined_valid": int(force_state.valid and torque_state.valid),
            "fixed_dist_force_bx": float(row["fixed_dist_force_bx"]),
            "fixed_dist_force_by": float(row["fixed_dist_force_by"]),
            "fixed_dist_force_bz": float(row["fixed_dist_force_bz"]),
            "fixed_dist_torque_bx": float(row["fixed_dist_torque_bx"]),
            "fixed_dist_torque_by": float(row["fixed_dist_torque_by"]),
            "fixed_dist_torque_bz": float(row["fixed_dist_torque_bz"]),
        }
        rows.append(row_out)

    offline_df = pd.DataFrame(rows)
    summary = summarize_offline_disturbance(offline_df)
    return offline_df, summary


def summarize_offline_disturbance(offline_df):
    force_valid_updates = offline_df[
        (offline_df["force_measurement_valid"] == 1) & (offline_df["force_valid"] == 1)
    ]
    torque_valid_updates = offline_df[
        (offline_df["torque_measurement_valid"] == 1) & (offline_df["torque_valid"] == 1)
    ]

    if force_valid_updates.empty:
        recommended_force = np.array([np.nan, np.nan, np.nan], dtype=float)
        last_force = np.array([np.nan, np.nan, np.nan], dtype=float)
    else:
        recommended_force = force_valid_updates[
            ["estimated_force_bx", "estimated_force_by", "estimated_force_bz"]
        ].mean().to_numpy(dtype=float)
        last_force = force_valid_updates[
            ["estimated_force_bx", "estimated_force_by", "estimated_force_bz"]
        ].iloc[-1].to_numpy(dtype=float)

    if torque_valid_updates.empty:
        recommended_torque = np.array([np.nan, np.nan, np.nan], dtype=float)
        last_torque = np.array([np.nan, np.nan, np.nan], dtype=float)
    else:
        recommended_torque = torque_valid_updates[
            ["estimated_torque_bx", "estimated_torque_by", "estimated_torque_bz"]
        ].mean().to_numpy(dtype=float)
        last_torque = torque_valid_updates[
            ["estimated_torque_bx", "estimated_torque_by", "estimated_torque_bz"]
        ].iloc[-1].to_numpy(dtype=float)

    return {
        "force_valid_updates": int(len(force_valid_updates)),
        "torque_valid_updates": int(len(torque_valid_updates)),
        "recommended_force": recommended_force,
        "recommended_torque": recommended_torque,
        "last_force": last_force,
        "last_torque": last_torque,
        "fixed_force": offline_df[["fixed_dist_force_bx", "fixed_dist_force_by", "fixed_dist_force_bz"]].iloc[-1].to_numpy(dtype=float),
        "fixed_torque": offline_df[["fixed_dist_torque_bx", "fixed_dist_torque_by", "fixed_dist_torque_bz"]].iloc[-1].to_numpy(dtype=float),
    }


def format_recommended_disturbance(summary):
    force = summary["recommended_force"]
    torque = summary["recommended_torque"]
    lines = [
        "Offline disturbance recommendation",
        f"  valid force updates:  {summary['force_valid_updates']}",
        f"  valid torque updates: {summary['torque_valid_updates']}",
        "",
        f"  force  mean valid estimate [N]:  bx={force[0]:.9f}  by={force[1]:.9f}  bz={force[2]:.9f}",
        f"  torque mean valid estimate [Nm]: bx={torque[0]:.9f}  by={torque[1]:.9f}  bz={torque[2]:.9f}",
        "",
        "PX4 fixed disturbance block",
        f"static constexpr double NMPC_DIST_FORCE_BX  = {force[0]:.9f};",
        f"static constexpr double NMPC_DIST_FORCE_BY  = {force[1]:.9f};",
        f"static constexpr double NMPC_DIST_FORCE_BZ  = {force[2]:.9f};",
        f"static constexpr double NMPC_DIST_TORQUE_BX = {torque[0]:.9f};",
        f"static constexpr double NMPC_DIST_TORQUE_BY = {torque[1]:.9f};",
        f"static constexpr double NMPC_DIST_TORQUE_BZ = {torque[2]:.9f};",
    ]
    return "\n".join(lines)
