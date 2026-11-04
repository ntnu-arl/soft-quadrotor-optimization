import os
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from acados_template import AcadosOcpSolver
from nmpc_controller import ACADOS_JSON_FILE, PARAM_COUNT, _reinitialize_acados_solver, _solve_acados_problem


NUM_MOTORS = 4
REPLAY_TIME_S = 11.4
END_TIME_S = REPLAY_TIME_S + 1.5
PHYSICS_DT_S = 0.01
PX4_CPP_PATH = (
    Path(os.environ["VOXL_PX4_REPO"])
    / "voxl-px4-dev/px4-firmware/src/modules/mc_nmpc_control/MulticopterNmpcControl.cpp"
)
PARAM_MASS = 0
PARAM_IXX = 1
PARAM_IXY = 2
PARAM_IXZ = 3
PARAM_IYY = 4
PARAM_IYZ = 5
PARAM_IZZ = 6
PARAM_GRAVITY = 7
PARAM_POS_REF = 10
PARAM_VEL_REF = 13
PARAM_ACC_REF = 16
PARAM_QUAT_REF = 19
PARAM_ALLOC = 23
PARAM_ALLOC_SIZE = 6 * NUM_MOTORS
PARAM_KF = PARAM_ALLOC + PARAM_ALLOC_SIZE
PARAM_TAU = PARAM_KF + NUM_MOTORS
PARAM_COM = PARAM_TAU + NUM_MOTORS
PARAM_DIST_FORCE = PARAM_COM + 3
PARAM_DIST_TORQUE = PARAM_DIST_FORCE + 3
PARAM_U_PREV = PARAM_DIST_TORQUE + 3
STATE13_COLUMNS = [
    "position_x",
    "position_y",
    "position_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
    "orientation_w",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
]
PARAM_COLUMNS = [f"p{i:02d}" for i in range(PARAM_COUNT)]
OFFLINE_FORCE_COLUMNS = ["estimated_force_bx", "estimated_force_by", "estimated_force_bz"]
OFFLINE_TORQUE_COLUMNS = ["estimated_torque_bx", "estimated_torque_by", "estimated_torque_bz"]
MOTOR_MEASURED_RPM_COLUMNS = ["measured_rpm0", "measured_rpm1", "measured_rpm2", "measured_rpm3"]
SOLVER_MEASURED_RPS_COLUMNS = ["measured_rps0", "measured_rps1", "measured_rps2", "measured_rps3"]
SOLVER_COMMAND_RPM_COLUMNS = ["command_rps0", "command_rps1", "command_rps2", "command_rps3"]
ESTIMATOR_FLAG_RESET = 1 << 0
ESTIMATOR_FLAG_WARMUP = 1 << 1
ESTIMATOR_FLAG_NOT_READY = 1 << 2
ESTIMATOR_FLAG_FORCE_GATED = 1 << 3
ESTIMATOR_FLAG_TORQUE_GATED = 1 << 4
ESTIMATOR_FLAG_FORCE_SATURATED = 1 << 5
ESTIMATOR_FLAG_TORQUE_SATURATED = 1 << 6
ESTIMATOR_FLAG_VALID = 1 << 7
ESTIMATOR_FLAG_MEASUREMENT_READY = 1 << 8
DIST_WARMUP_TIME_S = 2.0
DIST_WARMUP_MIN_SAMPLES = 40
DIST_INITIAL_SIGMA_RATIO = 0.50
DIST_READY_SIGMA_RATIO = 0.15
DIST_FORCE_VARIANCE_FLOOR_SIGMA_RATIO = 0.005
DIST_TORQUE_VARIANCE_FLOOR_SIGMA_RATIO = 0.05
DIST_FORCE_PROCESS_SIGMA_RATIO = 0.03
DIST_TORQUE_PROCESS_SIGMA_RATIO = 0.03
DIST_FORCE_MEAS_SIGMA_RATIO = 0.03
DIST_TORQUE_MEAS_SIGMA_RATIO = 0.20
DIST_FORCE_ABS_GATE_RATIO = 2.0
DIST_TORQUE_ABS_GATE_RATIO = 2.0
DIST_FORCE_INNOVATION_GATE_NIS = 1000.0
DIST_TORQUE_INNOVATION_GATE_NIS = 25.0
DIST_SLEW_RATE_RATIO_PER_S = 0.75
DIST_FORCE_HORIZON_DECAY_TAU_S = 0.2
DIST_FORCE_BOUND = np.array([1.0, 1.0, 1.0], dtype=np.float64)
DIST_TORQUE_BOUND = np.array([0.1, 0.1, 0.02], dtype=np.float64)
MOTOR_RPS_MEAS_TIMEOUT_US = 50000
MOTOR_RPS_CORRECTION_GAIN_MIN = 0.15
MOTOR_RPS_CORRECTION_GAIN_MAX = 0.60


@dataclass
class Px4PlantParameters:
    mass: float
    inertia: np.ndarray
    allocation_matrix: np.ndarray
    thrust_coefficients: np.ndarray
    motor_time_constants: np.ndarray
    com_offset: np.ndarray
    body_drag_coefficients: np.ndarray
    fixed_disturbance_force: np.ndarray
    fixed_disturbance_torque: np.ndarray
    gravity: np.ndarray


@dataclass
class ReplaySignals:
    source_segment_start_timestamp_us: int
    source_segment_end_timestamp_us: int
    solver_time_s: np.ndarray
    solver_sample_timestamp_us: np.ndarray
    position_velocity_timestamp_us: np.ndarray
    attitude_timestamp_us: np.ndarray
    angular_velocity_timestamp_us: np.ndarray
    packet_flags: np.ndarray
    actual_time_s: np.ndarray
    actual_state13: np.ndarray
    packet_motor_measured_rps: np.ndarray
    packet_motor_timestamp_us: np.ndarray
    packet_motor_valid_mask: np.ndarray
    logged_solver_status: np.ndarray
    logged_solver_forces: np.ndarray
    logged_commanded_rps: np.ndarray
    logged_estimated_rps: np.ndarray
    logged_applied_force: np.ndarray
    logged_applied_torque: np.ndarray
    motor_time_s: np.ndarray
    motor_measured_rps: np.ndarray
    params_time_series: np.ndarray
    offline_time_s: np.ndarray
    offline_force_body: np.ndarray
    offline_torque_body: np.ndarray
    replay_end_time_s: float


def clip_scalar(value, lower, upper):
    return min(max(value, lower), upper)


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if not np.isfinite(quat_norm) or quat_norm <= 1e-12:
        raise ValueError(f"invalid quaternion norm {quat_norm}")
    return quat / quat_norm


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_rotmat(quat):
    qw, qx, qy, qz = normalize_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def normalize_series(values):
    centered = values - np.mean(values)
    std = np.std(centered)
    if not std > 0.0:
        raise ValueError("cannot normalize zero-variance series")
    return centered / std


def interpolate_columns(time_s, sample_time_s, sample_matrix):
    return np.column_stack(
        [np.interp(time_s, sample_time_s, sample_matrix[:, col_idx]) for col_idx in range(sample_matrix.shape[1])]
    )


def interpolate_state13(time_s, sample_time_s, sample_state13):
    interpolated = interpolate_columns(time_s, sample_time_s, sample_state13)
    interpolated[:, 6:10] = np.array([normalize_quat(quat) for quat in interpolated[:, 6:10]], dtype=np.float64)
    return interpolated


def interpolate_state13_at(time_s, sample_time_s, sample_state13):
    state13 = interpolate_vector(time_s, sample_time_s, sample_state13)
    state13[6:10] = normalize_quat(state13[6:10])
    return state13


def find_best_offset(reference_time_s, reference_values, target_time_s, target_values, resample_dt_s):
    max_window_s = min(reference_time_s[-1], target_time_s[-1])
    resampled_time_s = np.arange(0.0, max_window_s, resample_dt_s)
    if len(resampled_time_s) < 2:
        raise ValueError("reference series is too short for alignment")

    reference_resampled = np.interp(resampled_time_s, reference_time_s, reference_values)
    reference_norm = normalize_series(reference_resampled)
    max_offset_s = target_time_s[-1] - resampled_time_s[-1]
    if max_offset_s < 0.0:
        raise ValueError("target series is too short for alignment")

    candidate_offsets_s = np.linspace(0.0, max_offset_s, 5000)
    best_score = None
    best_offset_s = None

    for offset_s in candidate_offsets_s:
        target_resampled = np.interp(resampled_time_s + offset_s, target_time_s, target_values)
        target_norm = normalize_series(target_resampled)
        score = float(np.mean(reference_norm * target_norm))
        if best_score is None or score > best_score:
            best_score = score
            best_offset_s = float(offset_s)

    if best_offset_s is None:
        raise RuntimeError("alignment failed")
    return best_offset_s, best_score


def find_motor_solver_offset_s(solver_time_s, solver_command_rpm, motor_time_s, motor_expected_rpm):
    max_window_s = min(solver_time_s[-1], motor_time_s[-1])
    solver_mask = solver_time_s <= max_window_s
    solver_time_window_s = solver_time_s[solver_mask]
    solver_command_window_rpm = solver_command_rpm[solver_mask]
    max_offset_s = motor_time_s[-1] - solver_time_window_s[-1]
    if max_offset_s < 0.0:
        raise ValueError("motor log is too short for alignment")

    candidate_offsets_s = np.linspace(0.0, max_offset_s, 5000)
    best_score = None
    best_offset_s = None

    for offset_s in candidate_offsets_s:
        score = 0.0
        for motor_idx in range(NUM_MOTORS):
            solver_series = normalize_series(solver_command_window_rpm[:, motor_idx])
            motor_series = np.interp(solver_time_window_s + offset_s, motor_time_s, motor_expected_rpm[:, motor_idx])
            motor_series = normalize_series(motor_series)
            score += float(np.mean(solver_series * motor_series))
        score /= float(NUM_MOTORS)
        if best_score is None or score > best_score:
            best_score = score
            best_offset_s = float(offset_s)

    if best_offset_s is None:
        raise RuntimeError("motor-to-solver alignment failed")
    return best_offset_s, best_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    return parser.parse_args()


def resolve_input_dir(results_dir):
    results_dir = results_dir.resolve()
    candidate_dirs = []

    if has_required_input_files(results_dir):
        candidate_dirs.append(results_dir)

    child_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    for child_dir in child_dirs:
        if has_required_input_files(child_dir):
            candidate_dirs.append(child_dir)

    if len(candidate_dirs) != 1:
        raise ValueError(
            f"expected exactly one replay input directory inside {results_dir}, found {len(candidate_dirs)}"
        )
    return candidate_dirs[0]


def has_required_input_files(candidate_dir):
    solver_candidates = [
        path for path in candidate_dir.glob("nmpc_solver_data_*.csv") if "_offline_disturbance" not in path.name
    ]
    offline_candidates = list(candidate_dir.glob("nmpc_solver_data_*_offline_disturbance.csv"))
    return (
        len(solver_candidates) == 1
        and len(offline_candidates) == 1
        and (candidate_dir / "qualisys_odom.csv").is_file()
        and (candidate_dir / "px4_motor_log.csv").is_file()
    )


def find_replay_segment_slice(solver_sample_timestamp_us, motor_timestamp_us):
    solver_sample_timestamp_us = np.asarray(solver_sample_timestamp_us, dtype=np.int64)
    motor_timestamp_us = np.asarray(motor_timestamp_us, dtype=np.int64)
    if len(solver_sample_timestamp_us) == 0:
        raise ValueError("solver log contains no samples")
    if len(motor_timestamp_us) == 0:
        raise ValueError("motor log contains no samples")
    if np.any(np.diff(solver_sample_timestamp_us) < 0):
        raise ValueError("solver sample timestamps moved backwards")

    motor_start_us = int(motor_timestamp_us[0])
    motor_end_us = int(motor_timestamp_us[-1])
    if motor_end_us < motor_start_us:
        raise ValueError("motor timestamps moved backwards")

    gap_indices = np.flatnonzero(np.diff(solver_sample_timestamp_us) > int(1.0e6))
    segment_starts = np.concatenate(([0], gap_indices + 1))
    segment_ends = np.concatenate((gap_indices + 1, [len(solver_sample_timestamp_us)]))
    overlapping_segments = []

    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment_start_us = int(solver_sample_timestamp_us[segment_start])
        segment_end_us = int(solver_sample_timestamp_us[segment_end - 1])
        overlap_start_us = max(segment_start_us, motor_start_us)
        overlap_end_us = min(segment_end_us, motor_end_us)
        if overlap_end_us >= overlap_start_us:
            overlapping_segments.append((int(segment_start), int(segment_end)))

    if len(overlapping_segments) != 1:
        raise ValueError(
            "expected exactly one continuous solver timestamp segment overlapping the motor log, "
            f"found {len(overlapping_segments)}"
        )

    return slice(overlapping_segments[0][0], overlapping_segments[0][1])


def parse_cpp_scalar(source_text, symbol_name):
    pattern = re.compile(rf"static constexpr [^=]+ {re.escape(symbol_name)}\s*=\s*([^;]+);")
    match = pattern.search(source_text)
    if match is None:
        raise ValueError(f"could not find {symbol_name} in {PX4_CPP_PATH}")
    raw_value = match.group(1).strip().replace("f", "")
    return float(raw_value)


def parse_cpp_array(source_text, symbol_name, expected_count):
    pattern = re.compile(
        rf"static constexpr [^=]+ {re.escape(symbol_name)}\s*\[\s*{expected_count}\s*\]\s*=\s*\{{(.*?)\}};",
        re.DOTALL,
    )
    match = pattern.search(source_text)
    if match is None:
        raise ValueError(f"could not find {symbol_name}[{expected_count}] in {PX4_CPP_PATH}")
    body = re.sub(r"//.*", "", match.group(1))
    tokens = [token.strip() for token in body.replace("\n", " ").split(",") if token.strip()]
    values = np.array([float(token.replace("f", "")) for token in tokens], dtype=np.float64)
    if len(values) != expected_count:
        raise ValueError(f"{symbol_name} expected {expected_count} values, found {len(values)}")
    return values


def load_px4_plant_parameters():
    source_text = PX4_CPP_PATH.read_text()
    alloc_flat = parse_cpp_array(source_text, "ALLOC_MATRIX_COLMAJOR", 24)
    inertia = np.array(
        [
            [parse_cpp_scalar(source_text, "NMPC_IXX"), parse_cpp_scalar(source_text, "NMPC_IXY"), parse_cpp_scalar(source_text, "NMPC_IXZ")],
            [parse_cpp_scalar(source_text, "NMPC_IXY"), parse_cpp_scalar(source_text, "NMPC_IYY"), parse_cpp_scalar(source_text, "NMPC_IYZ")],
            [parse_cpp_scalar(source_text, "NMPC_IXZ"), parse_cpp_scalar(source_text, "NMPC_IYZ"), parse_cpp_scalar(source_text, "NMPC_IZZ")],
        ],
        dtype=np.float64,
    )
    return Px4PlantParameters(
        mass=parse_cpp_scalar(source_text, "NMPC_MASS"),
        inertia=inertia,
        allocation_matrix=alloc_flat.reshape(6, NUM_MOTORS, order="F"),
        thrust_coefficients=np.array(
            [parse_cpp_scalar(source_text, f"NMPC_KF{i}") for i in range(1, NUM_MOTORS + 1)],
            dtype=np.float64,
        ),
        motor_time_constants=np.array(
            [parse_cpp_scalar(source_text, f"NMPC_TC{i}") for i in range(1, NUM_MOTORS + 1)],
            dtype=np.float64,
        ),
        com_offset=np.array(
            [
                parse_cpp_scalar(source_text, "NMPC_COM_X"),
                parse_cpp_scalar(source_text, "NMPC_COM_Y"),
                parse_cpp_scalar(source_text, "NMPC_COM_Z"),
            ],
            dtype=np.float64,
        ),
        body_drag_coefficients=np.array(
            [
                parse_cpp_scalar(source_text, "NMPC_BODY_DRAG_COEFF_BX"),
                parse_cpp_scalar(source_text, "NMPC_BODY_DRAG_COEFF_BY"),
                parse_cpp_scalar(source_text, "NMPC_BODY_DRAG_COEFF_BZ"),
            ],
            dtype=np.float64,
        ),
        fixed_disturbance_force=np.array(
            [
                parse_cpp_scalar(source_text, "NMPC_DIST_FORCE_BX"),
                parse_cpp_scalar(source_text, "NMPC_DIST_FORCE_BY"),
                parse_cpp_scalar(source_text, "NMPC_DIST_FORCE_BZ"),
            ],
            dtype=np.float64,
        ),
        fixed_disturbance_torque=np.array(
            [
                parse_cpp_scalar(source_text, "NMPC_DIST_TORQUE_BX"),
                parse_cpp_scalar(source_text, "NMPC_DIST_TORQUE_BY"),
                parse_cpp_scalar(source_text, "NMPC_DIST_TORQUE_BZ"),
            ],
            dtype=np.float64,
        ),
        gravity=np.array([0.0, 0.0, -9.81], dtype=np.float64),
    )


def validate_solver_artifacts(control_dt_s):
    config_path = REPO_ROOT / "configs" / "nmpc_controller_cfg.yaml"
    config_text = config_path.read_text()
    import yaml

    cfg = yaml.safe_load(config_text)["nmpc"]
    if not math.isclose(float(cfg["nmpc_dt"]), control_dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"config nmpc_dt {cfg['nmpc_dt']} does not match requested control dt {control_dt_s}")

    solver_json = json.loads(Path(ACADOS_JSON_FILE).read_text())
    horizon = int(solver_json["dims"]["N"])
    tf = float(solver_json["solver_options"]["tf"])
    json_dt = tf / horizon
    if horizon != int(cfg["N"]):
        raise ValueError(f"generated acados horizon {horizon} does not match config N {cfg['N']}")
    if not math.isclose(json_dt, control_dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"generated acados dt {json_dt} does not match config nmpc_dt {cfg['nmpc_dt']}")


def load_replay_signals(input_dir):
    solver_path = next(path for path in input_dir.glob("nmpc_solver_data_*.csv") if "_offline_disturbance" not in path.name)
    offline_path = next(input_dir.glob("nmpc_solver_data_*_offline_disturbance.csv"))
    qualisys_path = input_dir / "qualisys_odom.csv"
    motor_path = input_dir / "px4_motor_log.csv"

    solver_df = pd.read_csv(solver_path)
    offline_df = pd.read_csv(offline_path)
    motor_df = pd.read_csv(motor_path)

    full_solver_sample_timestamp_us = solver_df["sample_timestamp_us"].to_numpy(dtype=np.int64)
    motor_timestamp_us = motor_df["timestamp_us"].to_numpy(dtype=np.int64)
    segment_slice = find_replay_segment_slice(full_solver_sample_timestamp_us, motor_timestamp_us)
    solver_df = solver_df.iloc[segment_slice].reset_index(drop=True)
    offline_df = offline_df.iloc[segment_slice].reset_index(drop=True)

    solver_sample_timestamp_us = solver_df["sample_timestamp_us"].to_numpy(dtype=np.int64)
    offline_sample_timestamp_us = offline_df["sample_timestamp_us"].to_numpy(dtype=np.int64)
    if len(solver_sample_timestamp_us) != len(offline_sample_timestamp_us):
        raise ValueError("solver and offline disturbance segment lengths differ")
    if not np.array_equal(solver_sample_timestamp_us, offline_sample_timestamp_us):
        raise ValueError("solver and offline disturbance segment timestamps differ")

    solver_time_s = (solver_sample_timestamp_us.astype(np.float64) - float(solver_sample_timestamp_us[0])) / 1.0e6
    actual_state13 = solver_df[["px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]].to_numpy(
        dtype=np.float64
    )
    solver_command_rpm = solver_df[SOLVER_COMMAND_RPM_COLUMNS].to_numpy(dtype=np.float64) * 60.0
    packet_motor_measured_rps = solver_df[SOLVER_MEASURED_RPS_COLUMNS].to_numpy(dtype=np.float64)
    packet_motor_timestamp_us = solver_df[[f"motor_rps_timestamp_us{i}" for i in range(NUM_MOTORS)]].to_numpy(
        dtype=np.int64
    )
    packet_motor_valid_mask = solver_df["motor_rps_valid_mask"].to_numpy(dtype=np.uint8)

    motor_time_s = (motor_timestamp_us.astype(np.float64) - float(motor_timestamp_us[0])) / 1.0e6
    motor_expected_rpm = motor_df[[f"expected_requested_rpm{i}" for i in range(NUM_MOTORS)]].to_numpy(dtype=np.float64)
    motor_measured_rps = motor_df[MOTOR_MEASURED_RPM_COLUMNS].to_numpy(dtype=np.float64) / 60.0

    motor_to_solver_offset_s, _ = find_motor_solver_offset_s(solver_time_s, solver_command_rpm, motor_time_s, motor_expected_rpm)

    motor_local_time_s = motor_time_s - motor_to_solver_offset_s
    offline_time_s = (
        offline_df["sample_timestamp_us"].to_numpy(dtype=np.float64) - float(solver_sample_timestamp_us[0])
    ) / 1.0e6
    params_time_series = solver_df[PARAM_COLUMNS].to_numpy(dtype=np.float64)
    replay_end_time_s = min(solver_time_s[-1], offline_time_s[-1], END_TIME_S)

    if replay_end_time_s <= REPLAY_TIME_S:
        raise ValueError(f"replay overlap ends at {replay_end_time_s:.3f}s, before REPLAY_TIME={REPLAY_TIME_S:.3f}s")

    return ReplaySignals(
        source_segment_start_timestamp_us=int(solver_sample_timestamp_us[0]),
        source_segment_end_timestamp_us=int(solver_sample_timestamp_us[-1]),
        solver_time_s=solver_time_s,
        solver_sample_timestamp_us=solver_sample_timestamp_us,
        position_velocity_timestamp_us=solver_df["position_velocity_timestamp_us"].to_numpy(dtype=np.int64),
        attitude_timestamp_us=solver_df["attitude_timestamp_us"].to_numpy(dtype=np.int64),
        angular_velocity_timestamp_us=solver_df["angular_velocity_timestamp_us"].to_numpy(dtype=np.int64),
        packet_flags=solver_df["packet_flags"].to_numpy(dtype=np.uint8),
        actual_time_s=solver_time_s,
        actual_state13=actual_state13,
        packet_motor_measured_rps=packet_motor_measured_rps,
        packet_motor_timestamp_us=packet_motor_timestamp_us,
        packet_motor_valid_mask=packet_motor_valid_mask,
        logged_solver_status=solver_df["status"].to_numpy(dtype=np.int32),
        logged_solver_forces=solver_df[["solver_u0", "solver_u1", "solver_u2", "solver_u3"]].to_numpy(dtype=np.float64),
        logged_commanded_rps=solver_df[SOLVER_COMMAND_RPM_COLUMNS].to_numpy(dtype=np.float64),
        logged_estimated_rps=solver_df[["estimated_rps0", "estimated_rps1", "estimated_rps2", "estimated_rps3"]].to_numpy(
            dtype=np.float64
        ),
        logged_applied_force=solver_df[["fixed_dist_force_bx", "fixed_dist_force_by", "fixed_dist_force_bz"]].to_numpy(
            dtype=np.float64
        ),
        logged_applied_torque=solver_df[
            ["fixed_dist_torque_bx", "fixed_dist_torque_by", "fixed_dist_torque_bz"]
        ].to_numpy(dtype=np.float64),
        motor_time_s=motor_local_time_s,
        motor_measured_rps=motor_measured_rps,
        params_time_series=params_time_series,
        offline_time_s=offline_time_s,
        offline_force_body=offline_df[OFFLINE_FORCE_COLUMNS].to_numpy(dtype=np.float64),
        offline_torque_body=offline_df[OFFLINE_TORQUE_COLUMNS].to_numpy(dtype=np.float64),
        replay_end_time_s=float(replay_end_time_s),
    )


class DisturbanceEstimator:
    def __init__(self):
        self.estimated_force = np.zeros(3, dtype=np.float64)
        self.estimated_torque = np.zeros(3, dtype=np.float64)
        self.applied_force = np.zeros(3, dtype=np.float64)
        self.applied_torque = np.zeros(3, dtype=np.float64)
        self.force_variance = np.zeros(3, dtype=np.float64)
        self.torque_variance = np.zeros(3, dtype=np.float64)
        self.prev_velocity = np.zeros(3, dtype=np.float64)
        self.prev_omega = np.zeros(3, dtype=np.float64)
        self.last_force_timestamp_us = 0
        self.last_torque_timestamp_us = 0
        self.warmup_force_s = 0.0
        self.warmup_torque_s = 0.0
        self.sample_count_force = 0
        self.sample_count_torque = 0
        self.initialized = False
        self.force_innovation_nis = np.nan
        self.torque_innovation_nis = np.nan
        self.flags = ESTIMATOR_FLAG_RESET | ESTIMATOR_FLAG_WARMUP | ESTIMATOR_FLAG_NOT_READY

    def initialize(self, position_velocity_timestamp_us, angular_velocity_timestamp_us, x0):
        self.estimated_force[:] = 0.0
        self.estimated_torque[:] = 0.0
        self.applied_force[:] = 0.0
        self.applied_torque[:] = 0.0
        self.force_variance[:] = np.square(DIST_INITIAL_SIGMA_RATIO * DIST_FORCE_BOUND)
        self.torque_variance[:] = np.square(DIST_INITIAL_SIGMA_RATIO * DIST_TORQUE_BOUND)
        self.prev_velocity[:] = x0[3:6]
        self.prev_omega[:] = x0[10:13]
        self.last_force_timestamp_us = int(position_velocity_timestamp_us)
        self.last_torque_timestamp_us = int(angular_velocity_timestamp_us)
        self.warmup_force_s = 0.0
        self.warmup_torque_s = 0.0
        self.sample_count_force = 0
        self.sample_count_torque = 0
        self.initialized = True
        self.force_innovation_nis = np.nan
        self.torque_innovation_nis = np.nan
        self.flags = ESTIMATOR_FLAG_RESET | ESTIMATOR_FLAG_WARMUP | ESTIMATOR_FLAG_NOT_READY

    def _update_channel_state(
        self,
        estimated,
        applied,
        variance,
        measurement,
        bound,
        dt_s,
        process_sigma_ratio,
        measurement_sigma_ratio,
        abs_gate_ratio,
        variance_floor_sigma_ratio,
        innovation_gate_nis,
        sample_count,
        warmup_s,
    ):
        innovation_nis = 0.0
        gated = False
        ready = True

        for axis in range(3):
            process_var = np.square(process_sigma_ratio * bound[axis]) * dt_s
            meas_var = np.square(measurement_sigma_ratio * bound[axis])
            min_var = np.square(variance_floor_sigma_ratio * bound[axis])
            max_var = np.square(bound[axis])
            variance[axis] = clip_scalar(variance[axis] + process_var, min_var, max_var)
            innovation_nis += np.square(measurement[axis] - estimated[axis]) / (variance[axis] + meas_var)
            if abs(measurement[axis]) > abs_gate_ratio * bound[axis]:
                gated = True

        if innovation_nis > innovation_gate_nis:
            gated = True

        if not gated:
            for axis in range(3):
                meas_var = np.square(measurement_sigma_ratio * bound[axis])
                min_var = np.square(variance_floor_sigma_ratio * bound[axis])
                max_var = np.square(bound[axis])
                innovation = measurement[axis] - estimated[axis]
                innovation_cov = variance[axis] + meas_var
                gain = variance[axis] / innovation_cov
                estimated[axis] += gain * innovation
                variance[axis] = clip_scalar((1.0 - gain) * variance[axis], min_var, max_var)

        saturated = False
        for axis in range(3):
            before = estimated[axis]
            estimated[axis] = clip_scalar(estimated[axis], -bound[axis], bound[axis])
            saturated = saturated or abs(estimated[axis] - before) > 1e-12
            if math.sqrt(variance[axis]) > DIST_READY_SIGMA_RATIO * bound[axis]:
                ready = False

        if not (warmup_s >= DIST_WARMUP_TIME_S and sample_count >= DIST_WARMUP_MIN_SAMPLES and ready):
            ready = False

        if ready:
            for axis in range(3):
                step = DIST_SLEW_RATE_RATIO_PER_S * bound[axis] * dt_s
                delta = clip_scalar(estimated[axis] - applied[axis], -step, step)
                applied[axis] = clip_scalar(applied[axis] + delta, -bound[axis], bound[axis])
        else:
            applied[:] = 0.0

        return innovation_nis, gated, saturated, ready

    def update(self, sample_timestamp_us, position_velocity_timestamp_us, angular_velocity_timestamp_us, x0, solver_params):
        if not self.initialized:
            raise RuntimeError("disturbance estimator is not initialized")

        rotation = quat_to_rotmat(x0[6:10])
        actual_force_body, actual_torque_body = compute_body_wrench_from_rps(solver_params, x0[13:17])
        flags = ESTIMATOR_FLAG_MEASUREMENT_READY
        force_ready = False
        torque_ready = False

        if position_velocity_timestamp_us < self.last_force_timestamp_us:
            raise ValueError("position/velocity timestamp moved backwards")
        if angular_velocity_timestamp_us < self.last_torque_timestamp_us:
            raise ValueError("angular velocity timestamp moved backwards")

        if position_velocity_timestamp_us > self.last_force_timestamp_us:
            dt_s = (position_velocity_timestamp_us - self.last_force_timestamp_us) * 1e-6
            gravity_body = rotation.T @ solver_params[PARAM_GRAVITY:PARAM_GRAVITY + 3]
            measured_world_acc = (x0[3:6] - self.prev_velocity) / dt_s
            measured_body_acc = rotation.T @ measured_world_acc
            velocity_body = rotation.T @ x0[3:6]
            drag_force_body = compute_body_velocity_drag_force(velocity_body, self.plant_params.body_drag_coefficients)
            disturbance_force = (
                solver_params[PARAM_MASS] * (measured_body_acc - gravity_body) - actual_force_body - drag_force_body
            )
            self.force_innovation_nis, force_gated, force_saturated, force_ready = self._update_channel_state(
                self.estimated_force,
                self.applied_force,
                self.force_variance,
                disturbance_force,
                DIST_FORCE_BOUND,
                dt_s,
                DIST_FORCE_PROCESS_SIGMA_RATIO,
                DIST_FORCE_MEAS_SIGMA_RATIO,
                DIST_FORCE_ABS_GATE_RATIO,
                DIST_FORCE_VARIANCE_FLOOR_SIGMA_RATIO,
                DIST_FORCE_INNOVATION_GATE_NIS,
                self.sample_count_force + 1,
                self.warmup_force_s + dt_s,
            )
            self.prev_velocity[:] = x0[3:6]
            self.last_force_timestamp_us = int(position_velocity_timestamp_us)
            self.sample_count_force += 1
            self.warmup_force_s += dt_s
            if force_gated:
                flags |= ESTIMATOR_FLAG_FORCE_GATED
            if force_saturated:
                flags |= ESTIMATOR_FLAG_FORCE_SATURATED

        if angular_velocity_timestamp_us > self.last_torque_timestamp_us:
            dt_s = (angular_velocity_timestamp_us - self.last_torque_timestamp_us) * 1e-6
            inertia = build_inertia_matrix(solver_params)
            gravity_force_body = rotation.T @ (solver_params[PARAM_MASS] * solver_params[PARAM_GRAVITY:PARAM_GRAVITY + 3])
            torque_gravity = np.cross(solver_params[PARAM_COM:PARAM_COM + 3], gravity_force_body)
            omega = x0[10:13]
            omega_dot = (omega - self.prev_omega) / dt_s
            disturbance_torque = inertia @ omega_dot - actual_torque_body - torque_gravity + np.cross(omega, inertia @ omega)
            self.torque_innovation_nis, torque_gated, torque_saturated, torque_ready = self._update_channel_state(
                self.estimated_torque,
                self.applied_torque,
                self.torque_variance,
                disturbance_torque,
                DIST_TORQUE_BOUND,
                dt_s,
                DIST_TORQUE_PROCESS_SIGMA_RATIO,
                DIST_TORQUE_MEAS_SIGMA_RATIO,
                DIST_TORQUE_ABS_GATE_RATIO,
                DIST_TORQUE_VARIANCE_FLOOR_SIGMA_RATIO,
                DIST_TORQUE_INNOVATION_GATE_NIS,
                self.sample_count_torque + 1,
                self.warmup_torque_s + dt_s,
            )
            self.prev_omega[:] = omega
            self.last_torque_timestamp_us = int(angular_velocity_timestamp_us)
            self.sample_count_torque += 1
            self.warmup_torque_s += dt_s
            if torque_gated:
                flags |= ESTIMATOR_FLAG_TORQUE_GATED
            if torque_saturated:
                flags |= ESTIMATOR_FLAG_TORQUE_SATURATED

        valid = force_ready and torque_ready
        if valid:
            flags |= ESTIMATOR_FLAG_VALID
        else:
            flags |= ESTIMATOR_FLAG_WARMUP | ESTIMATOR_FLAG_NOT_READY
        self.flags = flags
        solver_params[PARAM_DIST_FORCE:PARAM_DIST_FORCE + 3] = self.applied_force
        solver_params[PARAM_DIST_TORQUE:PARAM_DIST_TORQUE + 3] = self.applied_torque


class OnboardNmpcReplica:
    def __init__(self, plant_params, control_dt_s, horizon_steps):
        self.plant_params = plant_params
        self.control_dt_s = control_dt_s
        self.horizon_steps = horizon_steps
        self.solver = AcadosOcpSolver(None, json_file=ACADOS_JSON_FILE, build=False, generate=False, verbose=False)
        self.motor_rps_hat = np.zeros(NUM_MOTORS, dtype=np.float64)
        self.commanded_rps_prev = np.zeros(NUM_MOTORS, dtype=np.float64)
        self.previous_horizon_controls = np.zeros((horizon_steps, NUM_MOTORS), dtype=np.float64)
        self.last_sample_timestamp_us = 0
        self.initialized = False
        self.disturbance_estimator = DisturbanceEstimator()
        self.commanded_force_prev = np.zeros(NUM_MOTORS, dtype=np.float64)

    def close(self):
        del self.solver

    def get_hover_force(self):
        return self.plant_params.mass * np.linalg.norm(self.plant_params.gravity) / NUM_MOTORS

    def get_hover_rps(self):
        return np.sqrt(self.get_hover_force() / self.plant_params.thrust_coefficients)

    def initialize_motor_estimator(self, sample_timestamp_us):
        hover_rps = self.get_hover_rps()
        self.motor_rps_hat[:] = hover_rps
        self.commanded_rps_prev[:] = hover_rps
        self.commanded_force_prev[:] = self.get_hover_force()
        self.previous_horizon_controls[:] = self.get_hover_force()
        self.last_sample_timestamp_us = int(sample_timestamp_us)
        self.initialized = True

    def seed_from_replay(
        self,
        sample_timestamp_us,
        position_velocity_timestamp_us,
        angular_velocity_timestamp_us,
        measured_state13,
        estimated_motor_rps,
        commanded_rps,
        commanded_forces,
        applied_disturbance_force,
        applied_disturbance_torque,
    ):
        x0 = np.zeros(17, dtype=np.float64)
        x0[0:13] = measured_state13
        x0[6:10] = normalize_quat(x0[6:10])
        x0[13:17] = estimated_motor_rps

        self.motor_rps_hat[:] = estimated_motor_rps
        self.commanded_rps_prev[:] = commanded_rps
        self.commanded_force_prev[:] = commanded_forces
        self.previous_horizon_controls[:] = commanded_forces
        self.last_sample_timestamp_us = int(sample_timestamp_us)
        self.initialized = True
        _reinitialize_acados_solver(self.solver, self.horizon_steps, x0, self.get_hover_force())

        self.disturbance_estimator.initialize(position_velocity_timestamp_us, angular_velocity_timestamp_us, x0)
        self.disturbance_estimator.estimated_force[:] = applied_disturbance_force
        self.disturbance_estimator.applied_force[:] = applied_disturbance_force
        self.disturbance_estimator.estimated_torque[:] = applied_disturbance_torque
        self.disturbance_estimator.applied_torque[:] = applied_disturbance_torque

    def propagate_motor_estimator(self, sample_timestamp_us):
        if sample_timestamp_us < self.last_sample_timestamp_us:
            raise ValueError("sample timestamp moved backwards")
        dt_s = (sample_timestamp_us - self.last_sample_timestamp_us) * 1e-6
        decay = np.exp(-dt_s / self.plant_params.motor_time_constants)
        self.motor_rps_hat[:] = self.commanded_rps_prev + (self.motor_rps_hat - self.commanded_rps_prev) * decay
        self.last_sample_timestamp_us = int(sample_timestamp_us)

    def correct_with_motor_measurements(
        self,
        sample_timestamp_us,
        motor_rps_measured,
        motor_timestamp_us,
        motor_rps_valid_mask,
        prefer_measurement_seed,
    ):
        for motor_idx in range(NUM_MOTORS):
            if (int(motor_rps_valid_mask) & (1 << motor_idx)) == 0:
                continue

            timestamp_us = int(motor_timestamp_us[motor_idx])
            if timestamp_us == 0:
                continue

            age_us = int(sample_timestamp_us - timestamp_us)
            if age_us < 0:
                continue
            if age_us > MOTOR_RPS_MEAS_TIMEOUT_US:
                continue

            measured_rps = float(motor_rps_measured[motor_idx])
            if not np.isfinite(measured_rps):
                continue

            if prefer_measurement_seed:
                self.motor_rps_hat[motor_idx] = measured_rps
            else:
                freshness = 1.0 - (age_us / MOTOR_RPS_MEAS_TIMEOUT_US)
                gain = clip_scalar(
                    MOTOR_RPS_CORRECTION_GAIN_MAX * freshness,
                    MOTOR_RPS_CORRECTION_GAIN_MIN,
                    MOTOR_RPS_CORRECTION_GAIN_MAX,
                )
                self.motor_rps_hat[motor_idx] += gain * (measured_rps - self.motor_rps_hat[motor_idx])

    def build_solver_params(self, setpoint_params):
        solver_params = np.zeros(PARAM_COUNT, dtype=np.float64)
        solver_params[PARAM_MASS] = self.plant_params.mass
        solver_params[PARAM_IXX] = self.plant_params.inertia[0, 0]
        solver_params[PARAM_IXY] = self.plant_params.inertia[0, 1]
        solver_params[PARAM_IXZ] = self.plant_params.inertia[0, 2]
        solver_params[PARAM_IYY] = self.plant_params.inertia[1, 1]
        solver_params[PARAM_IYZ] = self.plant_params.inertia[1, 2]
        solver_params[PARAM_IZZ] = self.plant_params.inertia[2, 2]
        solver_params[PARAM_GRAVITY:PARAM_GRAVITY + 3] = self.plant_params.gravity
        solver_params[PARAM_POS_REF:PARAM_POS_REF + 3] = setpoint_params[PARAM_POS_REF:PARAM_POS_REF + 3]
        solver_params[PARAM_VEL_REF:PARAM_VEL_REF + 3] = setpoint_params[PARAM_VEL_REF:PARAM_VEL_REF + 3]
        solver_params[PARAM_ACC_REF:PARAM_ACC_REF + 3] = setpoint_params[PARAM_ACC_REF:PARAM_ACC_REF + 3]
        solver_params[PARAM_QUAT_REF:PARAM_QUAT_REF + 4] = normalize_quat(setpoint_params[PARAM_QUAT_REF:PARAM_QUAT_REF + 4])
        solver_params[PARAM_ALLOC:PARAM_ALLOC + PARAM_ALLOC_SIZE] = self.plant_params.allocation_matrix.flatten(order="F")
        solver_params[PARAM_KF:PARAM_KF + NUM_MOTORS] = self.plant_params.thrust_coefficients
        solver_params[PARAM_TAU:PARAM_TAU + NUM_MOTORS] = self.plant_params.motor_time_constants
        solver_params[PARAM_COM:PARAM_COM + 3] = self.plant_params.com_offset
        solver_params[PARAM_DIST_FORCE:PARAM_DIST_FORCE + 3] = self.plant_params.fixed_disturbance_force
        solver_params[PARAM_DIST_TORQUE:PARAM_DIST_TORQUE + 3] = self.plant_params.fixed_disturbance_torque
        solver_params[PARAM_U_PREV:PARAM_U_PREV + NUM_MOTORS] = setpoint_params[PARAM_U_PREV:PARAM_U_PREV + NUM_MOTORS]
        return solver_params

    def step(
        self,
        measured_state13,
        sample_timestamp_us,
        position_velocity_timestamp_us,
        angular_velocity_timestamp_us,
        setpoint_params,
        motor_rps_measured,
        motor_timestamp_us,
        motor_rps_valid_mask,
        reset_controller,
    ):
        if reset_controller or not self.initialized:
            self.initialize_motor_estimator(sample_timestamp_us)
        else:
            self.propagate_motor_estimator(sample_timestamp_us)

        self.correct_with_motor_measurements(
            sample_timestamp_us,
            motor_rps_measured,
            motor_timestamp_us,
            motor_rps_valid_mask,
            reset_controller,
        )

        x0 = np.zeros(17, dtype=np.float64)
        x0[0:13] = measured_state13
        x0[6:10] = normalize_quat(x0[6:10])
        x0[13:17] = self.motor_rps_hat

        solver_params = self.build_solver_params(setpoint_params)
        if reset_controller or not self.disturbance_estimator.initialized:
            self.disturbance_estimator.initialize(position_velocity_timestamp_us, angular_velocity_timestamp_us, x0)
            solver_params[PARAM_DIST_FORCE:PARAM_DIST_FORCE + 3] = self.disturbance_estimator.applied_force
            solver_params[PARAM_DIST_TORQUE:PARAM_DIST_TORQUE + 3] = self.disturbance_estimator.applied_torque
        else:
            self.disturbance_estimator.update(
                sample_timestamp_us,
                position_velocity_timestamp_us,
                angular_velocity_timestamp_us,
                x0,
                solver_params,
            )

        params_by_stage = np.repeat(solver_params[np.newaxis, :], self.horizon_steps + 1, axis=0)
        for stage_idx in range(1, self.horizon_steps + 1):
            params_by_stage[stage_idx, PARAM_U_PREV:PARAM_U_PREV + NUM_MOTORS] = self.previous_horizon_controls[stage_idx - 1]
        apply_stage_disturbance_decay(params_by_stage, self.control_dt_s)

        solve_result = _solve_acados_problem(
            self.solver,
            self.horizon_steps,
            x0,
            params_by_stage,
            self.get_hover_force(),
            bool(reset_controller),
        )

        motor_forces = np.array(solve_result["u_opt"], dtype=np.float64, copy=True)
        commanded_rps = np.sqrt(np.maximum(motor_forces, 0.0) / self.plant_params.thrust_coefficients)
        self.commanded_force_prev[:] = motor_forces
        self.commanded_rps_prev[:] = commanded_rps
        self.previous_horizon_controls[:] = solve_result["horizon_controls"]

        return {
            "motor_forces": motor_forces,
            "commanded_rps": commanded_rps,
            "estimated_motor_rps": self.motor_rps_hat.copy(),
            "solver_status": int(solve_result["status"]),
            "solver_qp_status": int(solve_result["qp_status"]),
            "solver_cost": float(solve_result["cost"]),
            "predicted_quat_next": normalize_quat(solve_result["x1"][6:10]),
            "applied_disturbance_force": self.disturbance_estimator.applied_force.copy(),
            "applied_disturbance_torque": self.disturbance_estimator.applied_torque.copy(),
            "estimated_disturbance_force": self.disturbance_estimator.estimated_force.copy(),
            "estimated_disturbance_torque": self.disturbance_estimator.estimated_torque.copy(),
        }


def build_inertia_matrix(params):
    return np.array(
        [
            [params[PARAM_IXX], params[PARAM_IXY], params[PARAM_IXZ]],
            [params[PARAM_IXY], params[PARAM_IYY], params[PARAM_IYZ]],
            [params[PARAM_IXZ], params[PARAM_IYZ], params[PARAM_IZZ]],
        ],
        dtype=np.float64,
    )


def compute_body_wrench_from_rps(params, motor_rps):
    actual_forces = params[PARAM_KF:PARAM_KF + NUM_MOTORS] * np.square(motor_rps)
    wrench = params[PARAM_ALLOC:PARAM_ALLOC + PARAM_ALLOC_SIZE].reshape(6, NUM_MOTORS, order="F") @ actual_forces
    return wrench[0:3], wrench[3:6]


def compute_body_velocity_drag_force(velocity_body, body_drag_coefficients):
    return -body_drag_coefficients * velocity_body


def apply_stage_disturbance_decay(params_by_stage, control_dt_s):
    stage_time_s = np.arange(params_by_stage.shape[0], dtype=np.float64) * control_dt_s
    params_by_stage[:, PARAM_DIST_FORCE:PARAM_DIST_FORCE + 3] *= np.exp(-stage_time_s / DIST_FORCE_HORIZON_DECAY_TAU_S)[:, None]


def compute_observed_plant_disturbance(time_s, state13, motor_rps, plant_params):
    velocity_dot_world = np.gradient(state13[:, 3:6], time_s, axis=0)
    omega_dot_body = np.gradient(state13[:, 10:13], time_s, axis=0)
    observed_force_body = np.zeros((len(time_s), 3), dtype=np.float64)
    observed_torque_body = np.zeros((len(time_s), 3), dtype=np.float64)

    for sample_idx in range(len(time_s)):
        quat = normalize_quat(state13[sample_idx, 6:10])
        rotation_world_from_body = quat_to_rotmat(quat)
        actual_forces = plant_params.thrust_coefficients * np.square(motor_rps[sample_idx])
        wrench_body = plant_params.allocation_matrix @ actual_forces
        force_body = wrench_body[0:3]
        torque_body = wrench_body[3:6]
        gravity_body = rotation_world_from_body.T @ plant_params.gravity
        measured_body_acc = rotation_world_from_body.T @ velocity_dot_world[sample_idx]
        velocity_body = rotation_world_from_body.T @ state13[sample_idx, 3:6]
        drag_force_body = compute_body_velocity_drag_force(velocity_body, plant_params.body_drag_coefficients)
        observed_force_body[sample_idx] = plant_params.mass * (measured_body_acc - gravity_body) - force_body - drag_force_body

        omega = state13[sample_idx, 10:13]
        inertia_omega = plant_params.inertia @ omega
        gravity_force_body = rotation_world_from_body.T @ (plant_params.mass * plant_params.gravity)
        torque_gravity = np.cross(plant_params.com_offset, gravity_force_body)
        observed_torque_body[sample_idx] = (
            plant_params.inertia @ omega_dot_body[sample_idx]
            - torque_body
            - torque_gravity
            + np.cross(omega, inertia_omega)
        )

    return observed_force_body, observed_torque_body


def compute_modeled_motor_rps(time_s, command_forces, plant_params):
    modeled_motor_rps = np.zeros((len(time_s), NUM_MOTORS), dtype=np.float64)
    modeled_motor_rps[0] = np.sqrt(np.maximum(command_forces[0], 0.0) / plant_params.thrust_coefficients)

    for sample_idx in range(1, len(time_s)):
        dt_s = time_s[sample_idx] - time_s[sample_idx - 1]
        if dt_s <= 0.0:
            raise ValueError("motor model time grid must be strictly increasing")
        desired_rps = np.sqrt(np.maximum(command_forces[sample_idx - 1], 0.0) / plant_params.thrust_coefficients)
        decay = np.exp(-dt_s / plant_params.motor_time_constants)
        modeled_motor_rps[sample_idx] = desired_rps + (modeled_motor_rps[sample_idx - 1] - desired_rps) * decay

    return modeled_motor_rps


def rigid_body_rhs(state17, motor_force_cmd, plant_params, external_force_body, external_torque_body):
    position_world = state17[0:3]
    velocity_world = state17[3:6]
    quat = normalize_quat(state17[6:10])
    omega_body = state17[10:13]
    motor_rps = state17[13:17]
    del position_world

    actual_forces = plant_params.thrust_coefficients * np.square(motor_rps)
    wrench_body = plant_params.allocation_matrix @ actual_forces
    force_body = wrench_body[0:3]
    torque_body = wrench_body[3:6]
    rotation_world_from_body = quat_to_rotmat(quat)
    velocity_body = rotation_world_from_body.T @ velocity_world
    drag_force_body = compute_body_velocity_drag_force(velocity_body, plant_params.body_drag_coefficients)
    velocity_dot_world = (
        plant_params.gravity
        + rotation_world_from_body @ (force_body + drag_force_body + external_force_body) / plant_params.mass
    )
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=np.float64)
    quat_dot = 0.5 * quat_multiply(quat, omega_quat)
    gravity_body = rotation_world_from_body.T @ (plant_params.mass * plant_params.gravity)
    gravity_torque_body = np.cross(plant_params.com_offset, gravity_body)
    inertia_omega = plant_params.inertia @ omega_body
    omega_dot_body = np.linalg.solve(
        plant_params.inertia,
        torque_body + gravity_torque_body + external_torque_body - np.cross(omega_body, inertia_omega),
    )
    desired_rps = np.sqrt(np.maximum(motor_force_cmd, 0.0) / plant_params.thrust_coefficients)
    motor_rps_dot = (desired_rps - motor_rps) / plant_params.motor_time_constants
    return np.concatenate([velocity_world, velocity_dot_world, quat_dot, omega_dot_body, motor_rps_dot])


def rk4_step(state17, motor_force_cmd, plant_params, disturbance_force_body, disturbance_torque_body, dt_s):
    k1 = rigid_body_rhs(state17, motor_force_cmd, plant_params, disturbance_force_body, disturbance_torque_body)
    k2 = rigid_body_rhs(state17 + 0.5 * dt_s * k1, motor_force_cmd, plant_params, disturbance_force_body, disturbance_torque_body)
    k3 = rigid_body_rhs(state17 + 0.5 * dt_s * k2, motor_force_cmd, plant_params, disturbance_force_body, disturbance_torque_body)
    k4 = rigid_body_rhs(state17 + dt_s * k3, motor_force_cmd, plant_params, disturbance_force_body, disturbance_torque_body)
    next_state17 = state17 + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    next_state17[6:10] = normalize_quat(next_state17[6:10])
    return next_state17


def interpolate_vector(time_s, sample_time_s, sample_matrix):
    return np.array(
        [np.interp(time_s, sample_time_s, sample_matrix[:, axis_idx]) for axis_idx in range(sample_matrix.shape[1])],
        dtype=np.float64,
    )


def build_setpoint_lookup(replay_signals):
    return interpolate_columns(replay_signals.solver_time_s, replay_signals.solver_time_s, replay_signals.params_time_series)


def simulate_replica(input_dir):
    import yaml

    nmpc_cfg = yaml.safe_load((REPO_ROOT / "configs" / "nmpc_controller_cfg.yaml").read_text())["nmpc"]
    control_dt_s = float(nmpc_cfg["nmpc_dt"])
    horizon_steps = int(nmpc_cfg["N"])

    validate_solver_artifacts(control_dt_s)
    plant_params = load_px4_plant_parameters()
    replay_signals = load_replay_signals(input_dir)
    actual_time_grid_s = np.arange(0.0, replay_signals.replay_end_time_s + 0.5 * PHYSICS_DT_S, PHYSICS_DT_S)
    if actual_time_grid_s[-1] > replay_signals.replay_end_time_s:
        actual_time_grid_s = actual_time_grid_s[:-1]
    if len(actual_time_grid_s) < 2:
        raise ValueError("time grid is too short")

    actual_state13 = interpolate_state13(actual_time_grid_s, replay_signals.actual_time_s, replay_signals.actual_state13)
    setpoint_params_interp = interpolate_columns(actual_time_grid_s, replay_signals.solver_time_s, replay_signals.params_time_series)
    logged_force_interp = interpolate_columns(
        actual_time_grid_s,
        replay_signals.solver_time_s,
        replay_signals.logged_solver_forces,
    )
    motor_rps_interp = compute_modeled_motor_rps(actual_time_grid_s, logged_force_interp, plant_params)
    observed_force_body = interpolate_columns(
        actual_time_grid_s,
        replay_signals.offline_time_s,
        replay_signals.offline_force_body,
    )
    observed_torque_body = interpolate_columns(
        actual_time_grid_s,
        replay_signals.offline_time_s,
        replay_signals.offline_torque_body,
    )

    controller = OnboardNmpcReplica(plant_params, control_dt_s, horizon_steps)

    simulated_state13 = np.zeros((len(actual_time_grid_s), 13), dtype=np.float64)
    simulated_motor_rps = np.zeros((len(actual_time_grid_s), NUM_MOTORS), dtype=np.float64)
    controller_forces = np.zeros((len(actual_time_grid_s), NUM_MOTORS), dtype=np.float64)
    controller_commanded_rps = np.zeros((len(actual_time_grid_s), NUM_MOTORS), dtype=np.float64)
    controller_estimated_rps = np.zeros((len(actual_time_grid_s), NUM_MOTORS), dtype=np.float64)
    controller_applied_force = np.zeros((len(actual_time_grid_s), 3), dtype=np.float64)
    controller_applied_torque = np.zeros((len(actual_time_grid_s), 3), dtype=np.float64)
    solver_status = np.full(len(actual_time_grid_s), -1, dtype=np.int32)
    solver_qp_status = np.full(len(actual_time_grid_s), -1, dtype=np.int32)
    solver_cost = np.full(len(actual_time_grid_s), np.nan, dtype=np.float64)
    divergence_time_s = np.nan
    divergence_reason = ""
    failure_events = []

    handoff_index = int(round(REPLAY_TIME_S / PHYSICS_DT_S))
    if not math.isclose(actual_time_grid_s[handoff_index], REPLAY_TIME_S, rel_tol=0.0, abs_tol=1e-9):
        handoff_index = int(np.argmin(np.abs(actual_time_grid_s - REPLAY_TIME_S)))
    handoff_time_s = float(actual_time_grid_s[handoff_index])

    solver_event_mask = replay_signals.solver_time_s <= replay_signals.replay_end_time_s
    solver_event_indices = np.flatnonzero(solver_event_mask)
    if len(solver_event_indices) == 0:
        raise ValueError("no solver events overlap replay interval")

    current_state17 = np.zeros(17, dtype=np.float64)
    current_state17[0:13] = interpolate_vector(0.0, actual_time_grid_s, actual_state13)
    current_state17[6:10] = normalize_quat(current_state17[6:10])
    current_state17[13:17] = interpolate_vector(0.0, actual_time_grid_s, motor_rps_interp)
    current_force_cmd = np.full(NUM_MOTORS, plant_params.mass * np.linalg.norm(plant_params.gravity) / NUM_MOTORS, dtype=np.float64)
    current_time_s = 0.0
    measured_sensor_state13 = current_state17[0:13].copy()
    last_position_velocity_timestamp_us = int(replay_signals.position_velocity_timestamp_us[0])
    last_attitude_timestamp_us = int(replay_signals.attitude_timestamp_us[0])
    sim_history_time_s = [0.0]
    sim_history_motor_rps = [current_state17[13:17].copy()]
    last_controller_result = None
    controller_needs_reinit = False

    def set_state_from_reality(time_s):
        current_state17[0:13] = interpolate_vector(time_s, actual_time_grid_s, actual_state13)
        current_state17[6:10] = normalize_quat(current_state17[6:10])
        current_state17[13:17] = interpolate_vector(time_s, actual_time_grid_s, motor_rps_interp)

    def append_sim_history():
        if len(sim_history_time_s) > 0 and math.isclose(sim_history_time_s[-1], current_time_s, rel_tol=0.0, abs_tol=1e-12):
            sim_history_motor_rps[-1] = current_state17[13:17].copy()
        else:
            sim_history_time_s.append(float(current_time_s))
            sim_history_motor_rps.append(current_state17[13:17].copy())

    def motor_rps_at_timestamp_us(timestamp_us):
        local_time_s = (float(timestamp_us) - float(replay_signals.solver_sample_timestamp_us[0])) / 1.0e6
        if local_time_s <= handoff_time_s:
            return interpolate_vector(local_time_s, actual_time_grid_s, motor_rps_interp)

        history_matrix = np.asarray(sim_history_motor_rps, dtype=np.float64)
        history_time_s = np.asarray(sim_history_time_s, dtype=np.float64)
        if local_time_s > history_time_s[-1] + 1e-9:
            raise ValueError("motor measurement timestamp is newer than simulated history")
        return interpolate_vector(local_time_s, history_time_s, history_matrix)

    def logged_controller_result(event_idx):
        return {
            "motor_forces": replay_signals.logged_solver_forces[event_idx].copy(),
            "commanded_rps": replay_signals.logged_commanded_rps[event_idx].copy(),
            "estimated_motor_rps": replay_signals.logged_estimated_rps[event_idx].copy(),
            "solver_status": int(replay_signals.logged_solver_status[event_idx]),
            "solver_qp_status": np.nan,
            "solver_cost": np.nan,
            "applied_disturbance_force": replay_signals.logged_applied_force[event_idx].copy(),
            "applied_disturbance_torque": replay_signals.logged_applied_torque[event_idx].copy(),
        }

    def advance_to(target_time_s):
        nonlocal current_time_s, current_state17, divergence_time_s, divergence_reason
        if target_time_s < current_time_s - 1e-12:
            raise ValueError("simulation time moved backwards")

        if target_time_s <= handoff_time_s + 1e-12:
            set_state_from_reality(target_time_s)
            current_time_s = float(target_time_s)
            return

        if current_time_s < handoff_time_s:
            set_state_from_reality(handoff_time_s)
            current_time_s = handoff_time_s
            append_sim_history()

        while current_time_s + 1e-12 < target_time_s:
            dt_s = min(PHYSICS_DT_S, target_time_s - current_time_s)
            disturbance_force = interpolate_vector(current_time_s, actual_time_grid_s, observed_force_body)
            disturbance_torque = interpolate_vector(current_time_s, actual_time_grid_s, observed_torque_body)
            current_state17 = rk4_step(
                current_state17,
                current_force_cmd,
                plant_params,
                disturbance_force,
                disturbance_torque,
                dt_s,
            )
            current_time_s += dt_s
            append_sim_history()

            if not np.all(np.isfinite(current_state17)):
                divergence_time_s = current_time_s
                divergence_reason = "state became non-finite"
                return

            if (
                np.linalg.norm(current_state17[0:3]) > 100.0
                or np.linalg.norm(current_state17[3:6]) > 100.0
                or np.linalg.norm(current_state17[10:13]) > 1000.0
            ):
                divergence_time_s = current_time_s
                divergence_reason = "state norm exceeded explicit divergence threshold"
                return

    try:
        event_cursor = 0
        for step_idx, output_time_s in enumerate(actual_time_grid_s):
            while event_cursor < len(solver_event_indices):
                event_idx = int(solver_event_indices[event_cursor])
                event_time_s = float(replay_signals.solver_time_s[event_idx])
                if event_time_s > output_time_s + 1e-12:
                    break

                advance_to(event_time_s)
                if np.isfinite(divergence_time_s):
                    break

                sample_timestamp_us = int(replay_signals.solver_sample_timestamp_us[event_idx])
                position_velocity_timestamp_us = int(replay_signals.position_velocity_timestamp_us[event_idx])
                attitude_timestamp_us = int(replay_signals.attitude_timestamp_us[event_idx])
                angular_velocity_timestamp_us = int(replay_signals.angular_velocity_timestamp_us[event_idx])
                reset_controller = (not controller.initialized) or (
                    int(replay_signals.packet_flags[event_idx]) & ESTIMATOR_FLAG_RESET
                ) or controller_needs_reinit

                if event_time_s <= handoff_time_s + 1e-12:
                    controller_needs_reinit = False
                    last_controller_result = logged_controller_result(event_idx)
                    current_force_cmd[:] = last_controller_result["motor_forces"]
                    replay_state13 = interpolate_vector(event_time_s, actual_time_grid_s, actual_state13)
                    replay_state13[6:10] = normalize_quat(replay_state13[6:10])
                    measured_sensor_state13[:] = replay_state13
                    controller.seed_from_replay(
                        sample_timestamp_us,
                        position_velocity_timestamp_us,
                        angular_velocity_timestamp_us,
                        replay_state13,
                        replay_signals.logged_estimated_rps[event_idx],
                        replay_signals.logged_commanded_rps[event_idx],
                        replay_signals.logged_solver_forces[event_idx],
                        replay_signals.logged_applied_force[event_idx],
                        replay_signals.logged_applied_torque[event_idx],
                    )
                    last_position_velocity_timestamp_us = position_velocity_timestamp_us
                    last_attitude_timestamp_us = attitude_timestamp_us
                    event_cursor += 1
                    continue
                else:
                    if position_velocity_timestamp_us < last_position_velocity_timestamp_us:
                        raise ValueError("position/velocity timestamp moved backwards")
                    if attitude_timestamp_us < last_attitude_timestamp_us:
                        raise ValueError("attitude timestamp moved backwards")

                    if position_velocity_timestamp_us > last_position_velocity_timestamp_us:
                        measured_sensor_state13[0:6] = current_state17[0:6]
                        last_position_velocity_timestamp_us = position_velocity_timestamp_us
                    if attitude_timestamp_us > last_attitude_timestamp_us:
                        measured_sensor_state13[6:10] = current_state17[6:10]
                        last_attitude_timestamp_us = attitude_timestamp_us
                    measured_sensor_state13[10:13] = current_state17[10:13]
                    measured_state13 = measured_sensor_state13.copy()
                    setpoint_params = replay_signals.params_time_series[event_idx].copy()
                    setpoint_params[PARAM_U_PREV:PARAM_U_PREV + NUM_MOTORS] = current_force_cmd

                    motor_timestamp_us = replay_signals.packet_motor_timestamp_us[event_idx].copy()
                    motor_rps_valid_mask = int(replay_signals.packet_motor_valid_mask[event_idx])
                    motor_rps_measured = np.zeros(NUM_MOTORS, dtype=np.float64)
                    for motor_idx in range(NUM_MOTORS):
                        if (motor_rps_valid_mask & (1 << motor_idx)) != 0 and motor_timestamp_us[motor_idx] != 0:
                            motor_rps_measured[motor_idx] = motor_rps_at_timestamp_us(motor_timestamp_us[motor_idx])[
                                motor_idx
                            ]

                controller_result = controller.step(
                    measured_state13=measured_state13,
                    sample_timestamp_us=sample_timestamp_us,
                    position_velocity_timestamp_us=position_velocity_timestamp_us,
                    angular_velocity_timestamp_us=angular_velocity_timestamp_us,
                    setpoint_params=setpoint_params,
                    motor_rps_measured=motor_rps_measured,
                    motor_timestamp_us=motor_timestamp_us,
                    motor_rps_valid_mask=motor_rps_valid_mask,
                    reset_controller=reset_controller,
                )
                last_controller_result = controller_result
                if controller_result["solver_status"] != 0 or controller_result["solver_qp_status"] != 0:
                    failure_events.append(
                        {
                            "time_s": event_time_s,
                            "x_m": float(current_state17[0]),
                            "y_m": float(current_state17[1]),
                            "z_m": float(current_state17[2]),
                            "solver_status": int(controller_result["solver_status"]),
                            "qp_status": int(controller_result["solver_qp_status"]),
                        }
                    )
                if controller_result["solver_status"] == 0:
                    current_force_cmd[:] = controller_result["motor_forces"]
                    controller_needs_reinit = False
                else:
                    controller_needs_reinit = True
                event_cursor += 1

            if np.isfinite(divergence_time_s):
                break

            advance_to(float(output_time_s))
            if np.isfinite(divergence_time_s):
                break

            if last_controller_result is None:
                raise RuntimeError("no controller result available for output sample")

            simulated_state13[step_idx] = current_state17[0:13]
            simulated_motor_rps[step_idx] = current_state17[13:17]
            solver_status[step_idx] = last_controller_result["solver_status"]
            if np.isnan(last_controller_result["solver_qp_status"]):
                solver_qp_status[step_idx] = -1
            else:
                solver_qp_status[step_idx] = int(last_controller_result["solver_qp_status"])
            solver_cost[step_idx] = last_controller_result["solver_cost"]
            controller_forces[step_idx] = last_controller_result["motor_forces"]
            controller_commanded_rps[step_idx] = last_controller_result["commanded_rps"]
            controller_estimated_rps[step_idx] = last_controller_result["estimated_motor_rps"]
            controller_applied_force[step_idx] = last_controller_result["applied_disturbance_force"]
            controller_applied_torque[step_idx] = last_controller_result["applied_disturbance_torque"]
    finally:
        controller.close()

    valid_steps = len(actual_time_grid_s)
    if np.isfinite(divergence_time_s):
        valid_steps = int(np.searchsorted(actual_time_grid_s, divergence_time_s, side="left"))
        actual_time_grid_s = actual_time_grid_s[:valid_steps]
        actual_state13 = actual_state13[:valid_steps]
        setpoint_params_interp = setpoint_params_interp[:valid_steps]
        observed_force_body = observed_force_body[:valid_steps]
        observed_torque_body = observed_torque_body[:valid_steps]
        simulated_state13 = simulated_state13[:valid_steps]
        simulated_motor_rps = simulated_motor_rps[:valid_steps]
        controller_forces = controller_forces[:valid_steps]
        controller_commanded_rps = controller_commanded_rps[:valid_steps]
        controller_estimated_rps = controller_estimated_rps[:valid_steps]
        controller_applied_force = controller_applied_force[:valid_steps]
        controller_applied_torque = controller_applied_torque[:valid_steps]
        solver_status = solver_status[:valid_steps]
        solver_qp_status = solver_qp_status[:valid_steps]
        solver_cost = solver_cost[:valid_steps]

    if len(actual_time_grid_s) == 0:
        raise RuntimeError("replica produced no valid trajectory samples")

    replay_df = pd.DataFrame(
        {
            "time_s": actual_time_grid_s,
            "time_after_handoff_s": actual_time_grid_s - handoff_time_s,
            "actual_px": actual_state13[:, 0],
            "actual_py": actual_state13[:, 1],
            "actual_pz": actual_state13[:, 2],
            "actual_vx": actual_state13[:, 3],
            "actual_vy": actual_state13[:, 4],
            "actual_vz": actual_state13[:, 5],
            "actual_qw": actual_state13[:, 6],
            "actual_qx": actual_state13[:, 7],
            "actual_qy": actual_state13[:, 8],
            "actual_qz": actual_state13[:, 9],
            "actual_wx": actual_state13[:, 10],
            "actual_wy": actual_state13[:, 11],
            "actual_wz": actual_state13[:, 12],
            "sim_px": simulated_state13[:, 0],
            "sim_py": simulated_state13[:, 1],
            "sim_pz": simulated_state13[:, 2],
            "sim_vx": simulated_state13[:, 3],
            "sim_vy": simulated_state13[:, 4],
            "sim_vz": simulated_state13[:, 5],
            "sim_qw": simulated_state13[:, 6],
            "sim_qx": simulated_state13[:, 7],
            "sim_qy": simulated_state13[:, 8],
            "sim_qz": simulated_state13[:, 9],
            "sim_wx": simulated_state13[:, 10],
            "sim_wy": simulated_state13[:, 11],
            "sim_wz": simulated_state13[:, 12],
            "setpoint_px": setpoint_params_interp[:, PARAM_POS_REF + 0],
            "setpoint_py": setpoint_params_interp[:, PARAM_POS_REF + 1],
            "setpoint_pz": setpoint_params_interp[:, PARAM_POS_REF + 2],
            "setpoint_vx": setpoint_params_interp[:, PARAM_VEL_REF + 0],
            "setpoint_vy": setpoint_params_interp[:, PARAM_VEL_REF + 1],
            "setpoint_vz": setpoint_params_interp[:, PARAM_VEL_REF + 2],
            "plant_dist_force_bx": observed_force_body[:, 0],
            "plant_dist_force_by": observed_force_body[:, 1],
            "plant_dist_force_bz": observed_force_body[:, 2],
            "plant_dist_torque_bx": observed_torque_body[:, 0],
            "plant_dist_torque_by": observed_torque_body[:, 1],
            "plant_dist_torque_bz": observed_torque_body[:, 2],
            "controller_u0": controller_forces[:, 0],
            "controller_u1": controller_forces[:, 1],
            "controller_u2": controller_forces[:, 2],
            "controller_u3": controller_forces[:, 3],
            "controller_command_rps0": controller_commanded_rps[:, 0],
            "controller_command_rps1": controller_commanded_rps[:, 1],
            "controller_command_rps2": controller_commanded_rps[:, 2],
            "controller_command_rps3": controller_commanded_rps[:, 3],
            "controller_estimated_rps0": controller_estimated_rps[:, 0],
            "controller_estimated_rps1": controller_estimated_rps[:, 1],
            "controller_estimated_rps2": controller_estimated_rps[:, 2],
            "controller_estimated_rps3": controller_estimated_rps[:, 3],
            "sim_motor_rps0": simulated_motor_rps[:, 0],
            "sim_motor_rps1": simulated_motor_rps[:, 1],
            "sim_motor_rps2": simulated_motor_rps[:, 2],
            "sim_motor_rps3": simulated_motor_rps[:, 3],
            "controller_applied_dist_force_bx": controller_applied_force[:, 0],
            "controller_applied_dist_force_by": controller_applied_force[:, 1],
            "controller_applied_dist_force_bz": controller_applied_force[:, 2],
            "controller_applied_dist_torque_bx": controller_applied_torque[:, 0],
            "controller_applied_dist_torque_by": controller_applied_torque[:, 1],
            "controller_applied_dist_torque_bz": controller_applied_torque[:, 2],
            "solver_status": solver_status,
            "solver_qp_status": solver_qp_status,
            "solver_cost": solver_cost,
        }
    )

    failure_events_df = pd.DataFrame(
        failure_events,
        columns=["time_s", "x_m", "y_m", "z_m", "solver_status", "qp_status"],
    )

    after_handoff_mask = actual_time_grid_s >= handoff_time_s
    position_error = simulated_state13[after_handoff_mask, 0:3] - actual_state13[after_handoff_mask, 0:3]
    velocity_error = simulated_state13[after_handoff_mask, 3:6] - actual_state13[after_handoff_mask, 3:6]
    angular_rate_error = simulated_state13[after_handoff_mask, 10:13] - actual_state13[after_handoff_mask, 10:13]
    summary = {
        "input_dir": str(input_dir),
        "actual_state_source": "solver_log_state",
        "px4_parameter_source": str(PX4_CPP_PATH),
        "source_segment_start_timestamp_us": replay_signals.source_segment_start_timestamp_us,
        "source_segment_end_timestamp_us": replay_signals.source_segment_end_timestamp_us,
        "replay_time_s": REPLAY_TIME_S,
        "end_time_s": END_TIME_S,
        "physics_dt_s": PHYSICS_DT_S,
        "controller_dt_s": control_dt_s,
        "horizon_steps": horizon_steps,
        "horizon_duration_s": control_dt_s * horizon_steps,
        "handoff_time_s": handoff_time_s,
        "replay_end_time_s": float(actual_time_grid_s[-1]),
        "post_handoff_duration_s": float(actual_time_grid_s[-1] - handoff_time_s),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "angular_rate_rmse_radps": float(np.sqrt(np.mean(np.square(angular_rate_error)))),
        "final_position_error_m": position_error[-1].tolist(),
        "final_velocity_error_mps": velocity_error[-1].tolist(),
        "final_angular_rate_error_radps": angular_rate_error[-1].tolist(),
        "failure_event_count": int(len(failure_events_df)),
        "failure_solver_status_count": int((failure_events_df["solver_status"] != 0).sum()),
        "failure_qp_status_count": int((failure_events_df["qp_status"] != 0).sum()),
        "diverged": bool(np.isfinite(divergence_time_s)),
        "divergence_time_s": None if not np.isfinite(divergence_time_s) else float(divergence_time_s),
        "divergence_reason": divergence_reason,
    }
    return replay_df, failure_events_df, summary


def write_outputs(input_dir, replay_df, failure_events_df, summary):
    output_dir = input_dir / "real_flight_replica"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "replica_timeseries.csv"
    json_path = output_dir / "replica_summary.json"
    txt_path = output_dir / "replica_summary.txt"
    top_view_path = output_dir / "trajectory_top_view.png"
    side_view_path = output_dir / "trajectory_side_view_xz.png"
    replay_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary_lines = [f"{key}: {value}" for key, value in summary.items()]
    txt_path.write_text("\n".join(summary_lines) + "\n")
    render_trajectory_plots(replay_df, failure_events_df, summary, top_view_path, side_view_path)
    return csv_path, json_path, txt_path, top_view_path, side_view_path


def render_trajectory_plots(replay_df, failure_events_df, summary, top_view_path, side_view_path):
    handoff_time_s = float(summary["handoff_time_s"])
    diverged = bool(summary["diverged"])
    divergence_time_s = summary["divergence_time_s"]
    status_values = {
        0,
        1,
        4,
    }
    status_values.update(int(value) for value in failure_events_df["solver_status"].tolist())
    status_values.update(int(value) for value in failure_events_df["qp_status"].tolist())
    status_values.discard(0)
    status_ticks = sorted(status_values)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.5]},
    )
    ax = axes[0]
    ax_status = axes[1]
    ax.plot(replay_df["actual_px"], replay_df["actual_py"], label="Real", linewidth=2.0)
    ax.plot(replay_df["sim_px"], replay_df["sim_py"], label="Simulated", linewidth=2.0, linestyle="--")
    ax.scatter(
        [float(replay_df.loc[replay_df["time_s"].sub(handoff_time_s).abs().idxmin(), "actual_px"])],
        [float(replay_df.loc[replay_df["time_s"].sub(handoff_time_s).abs().idxmin(), "actual_py"])],
        label="Handoff",
        marker="o",
        s=60.0,
    )
    if diverged and divergence_time_s is not None:
        divergence_idx = replay_df["time_s"].sub(float(divergence_time_s)).abs().idxmin()
        ax.scatter(
            [float(replay_df.loc[divergence_idx, "sim_px"])],
            [float(replay_df.loc[divergence_idx, "sim_py"])],
            label="Divergence Stop",
            marker="x",
            s=80.0,
        )
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Top View")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if not failure_events_df.empty:
        solver_failure_mask = failure_events_df["solver_status"] != 0
        qp_failure_mask = failure_events_df["qp_status"] != 0
        if solver_failure_mask.any():
            ax_status.scatter(
                failure_events_df.loc[solver_failure_mask, "x_m"],
                failure_events_df.loc[solver_failure_mask, "solver_status"],
                label="Solve Status",
                marker="o",
                s=22.0,
                color="tab:red",
            )
        if qp_failure_mask.any():
            ax_status.scatter(
                failure_events_df.loc[qp_failure_mask, "x_m"],
                failure_events_df.loc[qp_failure_mask, "qp_status"],
                label="QP Status",
                marker="x",
                s=36.0,
                color="tab:orange",
                linewidths=1.4,
            )
    ax_status.set_xlabel("x [m]")
    ax_status.set_ylabel("status")
    ax_status.set_yticks(status_ticks)
    ax_status.grid(True, alpha=0.3)
    ax_status.legend()
    fig.tight_layout()
    fig.savefig(top_view_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.5]},
    )
    ax = axes[0]
    ax_status = axes[1]
    ax.plot(replay_df["actual_px"], replay_df["actual_pz"], label="Real", linewidth=2.0)
    ax.plot(replay_df["sim_px"], replay_df["sim_pz"], label="Simulated", linewidth=2.0, linestyle="--")
    ax.scatter(
        [float(replay_df.loc[replay_df["time_s"].sub(handoff_time_s).abs().idxmin(), "actual_px"])],
        [float(replay_df.loc[replay_df["time_s"].sub(handoff_time_s).abs().idxmin(), "actual_pz"])],
        label="Handoff",
        marker="o",
        s=60.0,
    )
    if diverged and divergence_time_s is not None:
        divergence_idx = replay_df["time_s"].sub(float(divergence_time_s)).abs().idxmin()
        ax.scatter(
            [float(replay_df.loc[divergence_idx, "sim_px"])],
            [float(replay_df.loc[divergence_idx, "sim_pz"])],
            label="Divergence Stop",
            marker="x",
            s=80.0,
        )
    ax.set_ylabel("z [m]")
    ax.set_title("Trajectory Side View (x-z)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if not failure_events_df.empty:
        solver_failure_mask = failure_events_df["solver_status"] != 0
        qp_failure_mask = failure_events_df["qp_status"] != 0
        if solver_failure_mask.any():
            ax_status.scatter(
                failure_events_df.loc[solver_failure_mask, "x_m"],
                failure_events_df.loc[solver_failure_mask, "solver_status"],
                label="Solve Status",
                marker="o",
                s=22.0,
                color="tab:red",
            )
        if qp_failure_mask.any():
            ax_status.scatter(
                failure_events_df.loc[qp_failure_mask, "x_m"],
                failure_events_df.loc[qp_failure_mask, "qp_status"],
                label="QP Status",
                marker="x",
                s=36.0,
                color="tab:orange",
                linewidths=1.4,
            )
    ax_status.set_xlabel("x [m]")
    ax_status.set_ylabel("status")
    ax_status.set_yticks(status_ticks)
    ax_status.grid(True, alpha=0.3)
    ax_status.legend()
    fig.tight_layout()
    fig.savefig(side_view_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = resolve_input_dir(args.results_dir)
    replay_df, failure_events_df, summary = simulate_replica(input_dir)
    csv_path, json_path, txt_path, top_view_path, side_view_path = write_outputs(
        input_dir,
        replay_df,
        failure_events_df,
        summary,
    )
    print(f"timeseries_csv= {csv_path}")
    print(f"summary_json= {json_path}")
    print(f"summary_txt= {txt_path}")
    print(f"top_view_png= {top_view_path}")
    print(f"side_view_png= {side_view_path}")
