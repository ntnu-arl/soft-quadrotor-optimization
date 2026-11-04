import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


NUM_MOTORS = 4
SYNC_START_TIME_S = 5.0
SIM_DT_S = 0.01
ALIGNMENT_RESAMPLE_DT_S = 0.02
NEW_INERTIA_MATRIX = np.array(
    [
        [0.001085435, 1.59528e-05, 3.50167e-06],
        [1.59528e-05, 0.00128329 * 1.0, -4.13557e-05],
        [3.50167e-06, -4.13557e-05, 0.001572009],
    ],
    dtype=np.float64,
)

PARAM_ALLOC_START = 20
PARAM_ALLOC_SIZE = 6 * NUM_MOTORS
PARAM_KF_START = PARAM_ALLOC_START + PARAM_ALLOC_SIZE
PARAM_TAU_START = PARAM_KF_START + NUM_MOTORS
PARAM_COM_START = PARAM_TAU_START + NUM_MOTORS
PARAM_DIST_FORCE_START = PARAM_COM_START + 3
PARAM_DIST_TORQUE_START = PARAM_DIST_FORCE_START + 3

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
SOLVER_COMMAND_RPM_COLUMNS = ["command_rps0", "command_rps1", "command_rps2", "command_rps3"]
MOTOR_EXPECTED_RPM_COLUMNS = [
    "expected_requested_rpm0",
    "expected_requested_rpm1",
    "expected_requested_rpm2",
    "expected_requested_rpm3",
]
MOTOR_MEASURED_RPM_COLUMNS = ["measured_rpm0", "measured_rpm1", "measured_rpm2", "measured_rpm3"]
PARAM_COLUMNS = [f"p{i:02d}" for i in range(65)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def has_required_input_files(candidate_dir):
    solver_candidates = [
        path
        for path in candidate_dir.glob("nmpc_solver_data_*.csv")
        if "_offline_disturbance" not in path.name
    ]
    return (
        len(solver_candidates) == 1
        and (candidate_dir / "qualisys_odom.csv").is_file()
        and (candidate_dir / "px4_motor_log.csv").is_file()
    )


def resolve_input_output_dirs(experiment_dir):
    candidate_input_dirs = []
    if has_required_input_files(experiment_dir):
        candidate_input_dirs.append(experiment_dir)

    child_dirs = sorted(path for path in experiment_dir.iterdir() if path.is_dir() and path.name != "nmpc_diagnostics")
    for child_dir in child_dirs:
        if has_required_input_files(child_dir):
            candidate_input_dirs.append(child_dir)

    if len(candidate_input_dirs) != 1:
        candidate_paths = ", ".join(str(path) for path in candidate_input_dirs)
        raise ValueError(
            f"expected exactly one input directory inside {experiment_dir}, found {len(candidate_input_dirs)}"
            f" candidates: {candidate_paths}"
        )

    input_dir = candidate_input_dirs[0]
    output_dir = experiment_dir / "nmpc_diagnostics"
    output_dir.mkdir(exist_ok=True)
    return input_dir, output_dir


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
    qw, qx, qy, qz = quat
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def normalize_quat(quat):
    quat_norm = np.linalg.norm(quat)
    if quat_norm <= 0.0 or not np.isfinite(quat_norm):
        raise ValueError("quaternion norm is invalid")
    return quat / quat_norm


def compute_pitch_deg(quat_array):
    quat_array = np.asarray(quat_array, dtype=np.float64)
    sin_pitch = 2.0 * (quat_array[..., 0] * quat_array[..., 2] - quat_array[..., 3] * quat_array[..., 1])
    return np.degrees(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))


def inertia_from_params(params):
    return np.array(
        [
            [params[1], params[2], params[3]],
            [params[2], params[4], params[5]],
            [params[3], params[5], params[6]],
        ],
        dtype=np.float64,
    )


def set_inertia_in_params(params, inertia_matrix):
    params_out = np.array(params, dtype=np.float64, copy=True)
    params_out[1] = inertia_matrix[0, 0]
    params_out[2] = inertia_matrix[0, 1]
    params_out[3] = inertia_matrix[0, 2]
    params_out[4] = inertia_matrix[1, 1]
    params_out[5] = inertia_matrix[1, 2]
    params_out[6] = inertia_matrix[2, 2]
    return params_out


def normalize_series(values):
    centered = values - np.mean(values)
    std = np.std(centered)
    if std <= 0.0:
        raise ValueError("cannot normalize a zero-variance series")
    return centered / std


def load_logs(input_dir):
    solver_candidates = sorted(
        path
        for path in input_dir.glob("nmpc_solver_data_*.csv")
        if "_offline_disturbance" not in path.name
    )
    if len(solver_candidates) != 1:
        raise ValueError(f"expected exactly one solver csv in {input_dir}, found {len(solver_candidates)}")

    motor_log_path = input_dir / "px4_motor_log.csv"
    solver_log_path = solver_candidates[0]
    qualisys_log_path = input_dir / "qualisys_odom.csv"
    solver_df = pd.read_csv(solver_log_path)
    qualisys_df = pd.read_csv(qualisys_log_path)
    motor_df = pd.read_csv(motor_log_path)
    return solver_df, qualisys_df, motor_df, solver_log_path, motor_log_path, qualisys_log_path


def find_best_offset(reference_time_s, reference_values, target_time_s, target_values):
    resampled_time_s = np.arange(0.0, reference_time_s[-1], ALIGNMENT_RESAMPLE_DT_S)
    if len(resampled_time_s) < 2:
        raise ValueError("reference series is too short for alignment")

    reference_resampled = np.interp(resampled_time_s, reference_time_s, reference_values)
    reference_norm = normalize_series(reference_resampled)

    max_offset_s = target_time_s[-1] - resampled_time_s[-1]
    if max_offset_s < 0.0:
        raise ValueError("target series is shorter than the reference series")

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

    if best_score is None or best_offset_s is None:
        raise RuntimeError("alignment failed")

    return best_offset_s, best_score


def find_motor_solver_offset_s(solver_time_s, solver_command_rpm, motor_time_s, motor_expected_rpm):
    if motor_time_s[-1] < solver_time_s[-1]:
        raise ValueError("motor log is shorter than the solver log")

    candidate_offsets_s = np.linspace(0.0, motor_time_s[-1] - solver_time_s[-1], 5000)
    best_score = None
    best_offset_s = None

    for offset_s in candidate_offsets_s:
        score = 0.0
        for motor_idx in range(NUM_MOTORS):
            solver_series = normalize_series(solver_command_rpm[:, motor_idx])
            motor_series = np.interp(solver_time_s + offset_s, motor_time_s, motor_expected_rpm[:, motor_idx])
            motor_series = normalize_series(motor_series)
            score += float(np.mean(solver_series * motor_series))
        score /= float(NUM_MOTORS)
        if best_score is None or score > best_score:
            best_score = score
            best_offset_s = float(offset_s)

    if best_score is None or best_offset_s is None:
        raise RuntimeError("motor-to-solver alignment failed")

    return best_offset_s, best_score


def interpolate_columns(time_s, sample_time_s, sample_matrix):
    return np.column_stack(
        [np.interp(time_s, sample_time_s, sample_matrix[:, col_idx]) for col_idx in range(sample_matrix.shape[1])]
    )


def interpolate_state13(time_s, sample_time_s, sample_state13):
    interpolated = interpolate_columns(time_s, sample_time_s, sample_state13)
    interpolated[:, 6:10] = np.array([normalize_quat(quat) for quat in interpolated[:, 6:10]])
    return interpolated


def rigid_body_dynamics(state13, motor_rps, params):
    position_world = state13[0:3]
    velocity_world = state13[3:6]
    quat = state13[6:10]
    omega_body = state13[10:13]
    del position_world

    mass = params[0]
    gravity_world = params[7:10]
    inertia = inertia_from_params(params)
    allocation_matrix = params[PARAM_ALLOC_START:PARAM_ALLOC_START + PARAM_ALLOC_SIZE].reshape(6, NUM_MOTORS, order="F")
    thrust_coefficients = params[PARAM_KF_START:PARAM_KF_START + NUM_MOTORS]
    com_offset = params[PARAM_COM_START:PARAM_COM_START + 3]
    disturbance_force_body = params[PARAM_DIST_FORCE_START:PARAM_DIST_FORCE_START + 3]
    disturbance_torque_body = params[PARAM_DIST_TORQUE_START:PARAM_DIST_TORQUE_START + 3]

    actual_forces = thrust_coefficients * motor_rps ** 2
    wrench_body = allocation_matrix @ actual_forces
    force_body = wrench_body[0:3]
    torque_body = wrench_body[3:6]

    rotation_world_from_body = quat_to_rotmat(quat)
    velocity_dot_world = gravity_world + rotation_world_from_body @ (force_body + disturbance_force_body) / mass

    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]], dtype=np.float64)
    quat_dot = 0.5 * quat_multiply(quat, omega_quat)

    gravity_body = rotation_world_from_body.T @ (mass * gravity_world)
    gravity_torque_body = np.cross(com_offset, gravity_body)
    inertia_omega = inertia @ omega_body
    omega_dot_body = np.linalg.solve(
        inertia,
        torque_body + gravity_torque_body + disturbance_torque_body - np.cross(omega_body, inertia_omega),
    )

    return np.concatenate([velocity_world, velocity_dot_world, quat_dot, omega_dot_body])


def motor_rps_at_time(time_s, motor_time_s, motor_rps_samples):
    return np.array(
        [np.interp(time_s, motor_time_s, motor_rps_samples[:, motor_idx]) for motor_idx in range(NUM_MOTORS)],
        dtype=np.float64,
    )


def rk4_step_time_varying(state13, t0_s, dt_s, motor_time_s, motor_rps_samples, params):
    def f(eval_time_s, eval_state13):
        return rigid_body_dynamics(eval_state13, motor_rps_at_time(eval_time_s, motor_time_s, motor_rps_samples), params)

    k1 = f(t0_s, state13)
    k2 = f(t0_s + 0.5 * dt_s, state13 + 0.5 * dt_s * k1)
    k3 = f(t0_s + 0.5 * dt_s, state13 + 0.5 * dt_s * k2)
    k4 = f(t0_s + dt_s, state13 + dt_s * k3)

    next_state13 = state13 + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    next_state13[6:10] = normalize_quat(next_state13[6:10])
    return next_state13


def simulate_from_sync_time(time_grid_s, initial_state13, motor_time_s, motor_rps_samples, params):
    simulated_state13 = np.zeros((len(time_grid_s), 13), dtype=np.float64)
    simulated_motor_rps = np.zeros((len(time_grid_s), NUM_MOTORS), dtype=np.float64)
    simulated_state13[0] = initial_state13
    simulated_motor_rps[0] = motor_rps_at_time(time_grid_s[0], motor_time_s, motor_rps_samples)

    for idx in range(len(time_grid_s) - 1):
        simulated_state13[idx + 1] = rk4_step_time_varying(
            simulated_state13[idx],
            time_grid_s[idx],
            time_grid_s[idx + 1] - time_grid_s[idx],
            motor_time_s,
            motor_rps_samples,
            params,
        )
        simulated_motor_rps[idx + 1] = motor_rps_at_time(time_grid_s[idx + 1], motor_time_s, motor_rps_samples)

    return simulated_state13, simulated_motor_rps


def compute_pitch_torque_body_y_nm(motor_rps_series, params):
    allocation_matrix = params[PARAM_ALLOC_START:PARAM_ALLOC_START + PARAM_ALLOC_SIZE].reshape(6, NUM_MOTORS, order="F")
    thrust_coefficients = params[PARAM_KF_START:PARAM_KF_START + NUM_MOTORS]
    torques_y = np.zeros(motor_rps_series.shape[0], dtype=np.float64)

    for idx in range(motor_rps_series.shape[0]):
        actual_forces = thrust_coefficients * motor_rps_series[idx] ** 2
        wrench_body = allocation_matrix @ actual_forces
        torques_y[idx] = wrench_body[4]

    return torques_y


def find_first_threshold_time_s(time_s, values_deg, threshold_deg):
    indices = np.where(values_deg <= threshold_deg)[0]
    if len(indices) == 0:
        return np.nan
    return float(time_s[indices[0]])


def write_summary(output_summary_path, summary_lines):
    output_summary_path.write_text("\n".join(summary_lines) + "\n")


def render_pdf(
    output_pdf_path,
    summary_lines,
    time_after_sync_s,
    actual_pitch_deg,
    simulated_pitch_deg,
    new_inertia_pitch_deg,
    actual_pitch_rate_deg_s,
    simulated_pitch_rate_deg_s,
    torque_y_nm,
    measured_rpm,
):
    with PdfPages(output_pdf_path) as pdf:
        fig = plt.figure(figsize=(11.0, 8.5))
        fig.suptitle("Synchronized Plant Replay Diagnostics")
        fig.text(0.05, 0.95, "\n".join(summary_lines), va="top", family="monospace", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(11.0, 8.5), sharex=True)
        fig.suptitle("Replay From The Synchronized 5 s Mark")

        axes[0].plot(time_after_sync_s, actual_pitch_deg, label="Measured pitch", linewidth=1.7)
        axes[0].plot(time_after_sync_s, simulated_pitch_deg, label="Plant replay pitch", linewidth=1.7, linestyle="--")
        axes[0].plot(time_after_sync_s, new_inertia_pitch_deg, label="Plant replay with new inertia", linewidth=1.7, linestyle=":")
        axes[0].set_ylabel("Pitch [deg]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(time_after_sync_s, actual_pitch_rate_deg_s, label="Measured pitch rate", linewidth=1.7)
        axes[1].plot(time_after_sync_s, simulated_pitch_rate_deg_s, label="Plant replay pitch rate", linewidth=1.7, linestyle="--")
        axes[1].set_ylabel("Pitch rate [deg/s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        axes[2].plot(time_after_sync_s, torque_y_nm, color="tab:red", linewidth=1.5)
        axes[2].set_ylabel("Body pitch torque [N m]")
        axes[2].grid(True, alpha=0.3)

        for motor_idx in range(NUM_MOTORS):
            axes[3].plot(time_after_sync_s, measured_rpm[:, motor_idx], label=f"motor {motor_idx}", linewidth=1.2)
        axes[3].set_xlabel("Time after synchronized 5 s mark [s]")
        axes[3].set_ylabel("Measured RPM")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper right")

        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    input_dir, output_dir = resolve_input_output_dirs(experiment_dir)

    output_pdf_path = output_dir / "pitch_response_diagnostics.pdf"
    output_csv_path = output_dir / "pitch_response_replay.csv"
    output_summary_path = output_dir / "pitch_response_summary.txt"

    solver_df, qualisys_df, motor_df, solver_log_path, motor_log_path, qualisys_log_path = load_logs(input_dir)

    solver_time_s = (solver_df["t_us"].to_numpy(dtype=np.float64) - solver_df["t_us"].iloc[0]) / 1.0e6
    solver_pitch_deg = compute_pitch_deg(solver_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64))
    solver_command_rpm = solver_df[SOLVER_COMMAND_RPM_COLUMNS].to_numpy(dtype=np.float64) * 60.0

    qualisys_time_s = (qualisys_df["timestamp_ns"].to_numpy(dtype=np.float64) - qualisys_df["timestamp_ns"].iloc[0]) / 1.0e9
    qualisys_pitch_deg = compute_pitch_deg(
        qualisys_df[["orientation_w", "orientation_x", "orientation_y", "orientation_z"]].to_numpy(dtype=np.float64)
    )
    qualisys_state13 = qualisys_df[STATE13_COLUMNS].to_numpy(dtype=np.float64)

    motor_time_s = (motor_df["timestamp_us"].to_numpy(dtype=np.float64) - motor_df["timestamp_us"].iloc[0]) / 1.0e6
    motor_expected_rpm = motor_df[MOTOR_EXPECTED_RPM_COLUMNS].to_numpy(dtype=np.float64)
    motor_measured_rpm = motor_df[MOTOR_MEASURED_RPM_COLUMNS].to_numpy(dtype=np.float64)
    motor_measured_rps = motor_measured_rpm / 60.0
    motor_measured_rpm_sum = np.sum(motor_measured_rpm, axis=1)

    solver_to_qualisys_offset_s, solver_to_qualisys_score = find_best_offset(
        solver_time_s,
        solver_pitch_deg,
        qualisys_time_s,
        qualisys_pitch_deg,
    )
    motor_to_solver_offset_s, motor_to_solver_score = find_motor_solver_offset_s(
        solver_time_s,
        solver_command_rpm,
        motor_time_s,
        motor_expected_rpm,
    )

    qualisys_local_time_s = qualisys_time_s - solver_to_qualisys_offset_s
    motor_local_time_s = motor_time_s - motor_to_solver_offset_s

    active_motor_indices = np.where(motor_measured_rpm_sum > 1000.0)[0]
    if len(active_motor_indices) == 0:
        raise ValueError("motor log does not contain any active measured RPM samples")
    active_motor_end_time_s = motor_local_time_s[active_motor_indices[-1]]

    replay_end_time_s = min(qualisys_local_time_s[-1], active_motor_end_time_s)
    if replay_end_time_s <= SYNC_START_TIME_S:
        raise ValueError("there is no synchronized overlap after the requested 5 s mark")

    replay_time_s = np.arange(SYNC_START_TIME_S, replay_end_time_s + 0.5 * SIM_DT_S, SIM_DT_S)
    if replay_time_s[-1] > replay_end_time_s:
        replay_time_s = replay_time_s[:-1]
    if len(replay_time_s) < 2:
        raise ValueError("replay grid is too short")

    actual_state13 = interpolate_state13(replay_time_s, qualisys_local_time_s, qualisys_state13)
    actual_pitch_deg = compute_pitch_deg(actual_state13[:, 6:10])
    actual_pitch_rate_deg_s = np.degrees(actual_state13[:, 11])

    initial_state13 = actual_state13[0].copy()
    solver_sync_idx = int(np.argmin(np.abs(solver_time_s - SYNC_START_TIME_S)))
    params_sync = solver_df.loc[solver_sync_idx, PARAM_COLUMNS].to_numpy(dtype=np.float64)

    simulated_state13, simulated_motor_rps = simulate_from_sync_time(
        replay_time_s,
        initial_state13,
        motor_local_time_s,
        motor_measured_rps,
        params_sync,
    )
    params_new_inertia = set_inertia_in_params(params_sync, NEW_INERTIA_MATRIX)
    simulated_state13_new_inertia, _ = simulate_from_sync_time(
        replay_time_s,
        initial_state13,
        motor_local_time_s,
        motor_measured_rps,
        params_new_inertia,
    )
    simulated_pitch_deg = compute_pitch_deg(simulated_state13[:, 6:10])
    simulated_pitch_deg_new_inertia = compute_pitch_deg(simulated_state13_new_inertia[:, 6:10])
    simulated_pitch_rate_deg_s = np.degrees(simulated_state13[:, 11])

    replay_motor_rpm = interpolate_columns(replay_time_s, motor_local_time_s, motor_measured_rpm)
    replay_torque_y_nm = compute_pitch_torque_body_y_nm(simulated_motor_rps, params_sync)

    replay_df = pd.DataFrame(
        {
            "time_after_sync_5s_mark_s": replay_time_s - SYNC_START_TIME_S,
            "absolute_synced_time_s": replay_time_s,
            "actual_pitch_deg": actual_pitch_deg,
            "simulated_pitch_deg": simulated_pitch_deg,
            "simulated_pitch_deg_new_inertia": simulated_pitch_deg_new_inertia,
            "actual_pitch_rate_deg_s": actual_pitch_rate_deg_s,
            "simulated_pitch_rate_deg_s": simulated_pitch_rate_deg_s,
            "measured_rpm0": replay_motor_rpm[:, 0],
            "measured_rpm1": replay_motor_rpm[:, 1],
            "measured_rpm2": replay_motor_rpm[:, 2],
            "measured_rpm3": replay_motor_rpm[:, 3],
            "simulated_torque_body_y_nm": replay_torque_y_nm,
        }
    )
    replay_df.to_csv(output_csv_path, index=False)

    pitch_rmse_deg = float(np.sqrt(np.mean((simulated_pitch_deg - actual_pitch_deg) ** 2)))
    pitch_rate_rmse_deg_s = float(np.sqrt(np.mean((simulated_pitch_rate_deg_s - actual_pitch_rate_deg_s) ** 2)))

    threshold_lines = []
    for threshold_deg in [-5.0, -10.0, -20.0, -40.0]:
        actual_time_s = find_first_threshold_time_s(replay_time_s - SYNC_START_TIME_S, actual_pitch_deg, threshold_deg)
        simulated_time_s = find_first_threshold_time_s(replay_time_s - SYNC_START_TIME_S, simulated_pitch_deg, threshold_deg)
        threshold_lines.append(
            f"threshold {threshold_deg:>5.1f} deg: actual={actual_time_s:>7.3f} s, simulated={simulated_time_s:>7.3f} s"
        )

    divergence_direction = "slower"
    if np.nanmax(np.abs(simulated_pitch_deg - actual_pitch_deg)) > 0.0:
        idx_max = int(np.nanargmax(np.abs(simulated_pitch_deg - actual_pitch_deg)))
        if simulated_pitch_deg[idx_max] < actual_pitch_deg[idx_max]:
            divergence_direction = "faster forward"
        else:
            divergence_direction = "slower forward"

    summary_lines = [
        f"experiment_dir: {experiment_dir}",
        f"input_dir: {input_dir}",
        f"solver_log: {solver_log_path.name}",
        f"motor_log: {motor_log_path.name}",
        f"qualisys_log: {qualisys_log_path.name}",
        f"solver_to_qualisys_offset_s: {solver_to_qualisys_offset_s:.6f}",
        f"solver_to_qualisys_alignment_score: {solver_to_qualisys_score:.6f}",
        f"motor_to_solver_offset_s: {motor_to_solver_offset_s:.6f}",
        f"motor_to_solver_alignment_score: {motor_to_solver_score:.6f}",
        f"sync_start_time_s: {SYNC_START_TIME_S:.3f}",
        f"replay_duration_after_sync_start_s: {replay_time_s[-1] - replay_time_s[0]:.3f}",
        f"plant_mass_kg: {params_sync[0]:.6f}",
        f"plant_inertia_yy_kg_m2: {params_sync[4]:.7f}",
        f"new_inertia_yy_kg_m2: {NEW_INERTIA_MATRIX[1, 1]:.7f}",
        f"initial_pitch_deg: {actual_pitch_deg[0]:.3f}",
        f"final_actual_pitch_deg: {actual_pitch_deg[-1]:.3f}",
        f"final_simulated_pitch_deg: {simulated_pitch_deg[-1]:.3f}",
        f"final_simulated_pitch_deg_new_inertia: {simulated_pitch_deg_new_inertia[-1]:.3f}",
        f"pitch_rmse_deg: {pitch_rmse_deg:.3f}",
        f"pitch_rate_rmse_deg_s: {pitch_rate_rmse_deg_s:.3f}",
        f"measured_peak_forward_pitch_deg: {np.min(actual_pitch_deg):.3f}",
        f"simulated_peak_forward_pitch_deg: {np.min(simulated_pitch_deg):.3f}",
        f"measured_peak_forward_pitch_rate_deg_s: {np.min(actual_pitch_rate_deg_s):.3f}",
        f"simulated_peak_forward_pitch_rate_deg_s: {np.min(simulated_pitch_rate_deg_s):.3f}",
        f"conclusion: from the synchronized 5 s state, the plant replay diverges {divergence_direction} than the real vehicle under the same measured motor outputs.",
    ]
    summary_lines.extend(threshold_lines)

    write_summary(output_summary_path, summary_lines)
    render_pdf(
        output_pdf_path,
        summary_lines,
        replay_time_s - SYNC_START_TIME_S,
        actual_pitch_deg,
        simulated_pitch_deg,
        simulated_pitch_deg_new_inertia,
        actual_pitch_rate_deg_s,
        simulated_pitch_rate_deg_s,
        replay_torque_y_nm,
        replay_motor_rpm,
    )


if __name__ == "__main__":
    main()
