import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from offline_disturbance_estimator import (
    NUM_PARAMS,
    PARAM_ALLOC,
    PARAM_COM,
    PARAM_DIST_FORCE,
    PARAM_KF,
    REQUIRED_SOLVER_COLUMNS,
    compute_body_velocity_drag_force,
    estimate_offline_disturbance,
    format_recommended_disturbance,
    require_columns,
)


NUM_MOTORS = 4
PARAM_ACC_REF = 16
MOTOR_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
PLOTLY_MOTOR_COLORS = ["blue", "orange", "green", "red"]
RESIDUAL_COLUMNS = ["res_stat", "res_eq", "res_ineq", "res_comp"]
RESIDUAL_LABELS = ["stationarity", "equality", "inequality", "complementarity"]
RESIDUAL_COLORS = ["blue", "orange", "green", "red"]
OPTIMIZER_COST_COLUMNS = [
    "optimizer_cost_total",
    "optimizer_cost_stage_total",
    "optimizer_cost_terminal_total",
    "optimizer_cost_stage_pos",
    "optimizer_cost_stage_vel",
    "optimizer_cost_stage_att",
    "optimizer_cost_stage_omega",
    "optimizer_cost_stage_u",
    "optimizer_cost_stage_delta_u",
    "optimizer_cost_terminal_pos",
    "optimizer_cost_terminal_vel",
    "optimizer_cost_terminal_att",
    "optimizer_cost_terminal_omega",
]
COST_PLOT_YMIN = 1e-6
COST_PLOT_YMAX = 1e2
DOMINANT_COST_PLOT_YMIN = 1e-12
FLAG_RECOVERY = 0x02
NMPC_FLIGHT_MODE_LABEL = "NMPC_FLIGHT_MODE"
NMPC_RECOVERY_MODE_LABEL = "NMPC_RECOVERY_MODE"
ACTUATOR_ACTIVE_RPM_THRESHOLD = 100.0
NMPC_SETPOINT_COLUMNS = [
    "sp_px",
    "sp_py",
    "sp_pz",
    "sp_vx",
    "sp_vy",
    "sp_vz",
    "sp_qw",
    "sp_qx",
    "sp_qy",
    "sp_qz",
]


def get_residual_plot_series(solver_df):
    if not all(column_name in solver_df.columns for column_name in RESIDUAL_COLUMNS):
        return None

    residual_data = {}
    for column_name in RESIDUAL_COLUMNS:
        residual_values = solver_df[column_name].to_numpy(dtype=float)
        if np.any(~np.isfinite(residual_values)):
            raise RuntimeError(f"Non-finite residual values in column {column_name}")
        residual_data[column_name] = np.maximum(residual_values, 1e-16)

    return residual_data


def get_optimizer_cost_plot_series(solver_df):
    if not all(column_name in solver_df.columns for column_name in OPTIMIZER_COST_COLUMNS):
        return None

    cost_data = {}
    for column_name in OPTIMIZER_COST_COLUMNS:
        cost_values = solver_df[column_name].to_numpy(dtype=float)
        if np.any(~np.isfinite(cost_values)):
            raise RuntimeError(f"Non-finite optimizer cost values in column {column_name}")
        if np.any(cost_values < 0.0):
            raise RuntimeError(f"Negative optimizer cost values in column {column_name}")
        cost_data[column_name] = np.clip(cost_values, COST_PLOT_YMIN, COST_PLOT_YMAX)

    return cost_data


def find_first_motor_activity_timestamp_us(px4_df):
    measured_columns = [f"measured_rpm{motor_index}" for motor_index in range(NUM_MOTORS)]
    require_columns(px4_df, ["timestamp_us"] + measured_columns, "PX4 motor CSV")

    measured_rpm = px4_df[measured_columns].to_numpy(dtype=float)
    active_rows = np.any(measured_rpm > ACTUATOR_ACTIVE_RPM_THRESHOLD, axis=1)
    active_indices = np.flatnonzero(active_rows)
    if active_indices.size == 0:
        raise RuntimeError(
            f"No measured motor RPM exceeds {ACTUATOR_ACTIVE_RPM_THRESHOLD:g}; "
            "cannot identify actuator-on plot window"
        )

    return int(px4_df["timestamp_us"].iloc[int(active_indices[0])])


def trim_logs_to_activity_window(solver_df, px4_df, px4_flight_df):
    activity_start_us = find_first_motor_activity_timestamp_us(px4_df)

    solver_df = solver_df[solver_df["sample_timestamp_us"] >= activity_start_us].reset_index(drop=True)
    px4_df = px4_df[px4_df["timestamp_us"] >= activity_start_us].reset_index(drop=True)
    px4_flight_df = px4_flight_df[px4_flight_df["timestamp_us"] >= activity_start_us].reset_index(drop=True)

    if len(solver_df) < 2:
        raise RuntimeError(
            "Fewer than 2 solver samples remain after trimming to first measured motor activity"
        )
    if px4_df.empty:
        raise RuntimeError("No PX4 motor samples remain after trimming to first measured motor activity")
    if px4_flight_df.empty:
        raise RuntimeError("No PX4 flight samples remain after trimming to first measured motor activity")

    return solver_df, px4_df, px4_flight_df, activity_start_us


def build_cost_profile(nmpc_cfg):
    return {
        "stage_scaling": 1.0,
        "terminal_scaling": 1.0,
        "Q_pos": np.asarray(nmpc_cfg["Q_pos"], dtype=float),
        "Q_vel": np.asarray(nmpc_cfg["Q_vel"], dtype=float),
        "Q_acc": np.asarray(nmpc_cfg["Q_acc"], dtype=float),
        "Q_att": np.asarray(nmpc_cfg["Q_att"], dtype=float),
        "Q_omega": np.asarray(nmpc_cfg["Q_omega"], dtype=float),
        "R": float(nmpc_cfg["R"]),
        "R_delta_u": float(nmpc_cfg["R_delta_u"]),
        "terminal_factor": float(nmpc_cfg["terminal_factor"]),
    }


def load_nmpc_cost_weights(config_path, recovery_mode_present):
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nmpc_cfg = cfg["nmpc"]
    flight_profile = build_cost_profile(nmpc_cfg)

    if recovery_mode_present and "recovery" not in nmpc_cfg:
        raise RuntimeError(
            "Solver log contains NMPC_RECOVERY_MODE samples, but the NMPC config has no nmpc.recovery cost block"
        )

    if "recovery" in nmpc_cfg:
        recovery_cfg = dict(nmpc_cfg["recovery"])
        recovery_profile = build_cost_profile(recovery_cfg)
    else:
        recovery_profile = None

    return {
        NMPC_FLIGHT_MODE_LABEL: flight_profile,
        NMPC_RECOVERY_MODE_LABEL: recovery_profile,
    }


def get_recovery_mode_mask(solver_df):
    packet_flags = solver_df["packet_flags"].to_numpy(dtype=np.uint8)
    return (packet_flags & FLAG_RECOVERY) != 0


def get_nmpc_mode_segments(solver_df, solver_t):
    recovery_mask = get_recovery_mode_mask(solver_df)
    change_indices = np.flatnonzero(recovery_mask[1:] != recovery_mask[:-1]) + 1
    segment_edges = np.concatenate(([0], change_indices, [len(solver_df)]))

    segments = []
    for segment_start, segment_end in zip(segment_edges[:-1], segment_edges[1:]):
        is_recovery = bool(recovery_mask[segment_start])
        label = NMPC_RECOVERY_MODE_LABEL if is_recovery else NMPC_FLIGHT_MODE_LABEL
        segments.append(
            {
                "label": label,
                "is_recovery": is_recovery,
                "start_t": float(solver_t.iloc[segment_start]),
                "end_t": float(solver_t.iloc[segment_end - 1]),
                "count": int(segment_end - segment_start),
            }
        )

    return segments


def summarize_nmpc_modes(solver_df, solver_t):
    segments = get_nmpc_mode_segments(solver_df, solver_t)
    recovery_mask = get_recovery_mode_mask(solver_df)
    flight_count = int((~recovery_mask).sum())
    recovery_count = int(recovery_mask.sum())
    total_count = len(solver_df)

    if recovery_count == 0:
        return f"NMPC mode: {NMPC_FLIGHT_MODE_LABEL} only ({flight_count}/{total_count} samples)"

    if flight_count == 0:
        return f"NMPC mode: {NMPC_RECOVERY_MODE_LABEL} only ({recovery_count}/{total_count} samples)"

    segment_text = "; ".join(
        f"{segment['label'].replace('NMPC_', '').replace('_MODE', '')} "
        f"{segment['start_t']:.2f}-{segment['end_t']:.2f}s"
        for segment in segments
    )
    return (
        f"NMPC mode: {len(segments)} segments; recovery {recovery_count}/{total_count} samples; "
        f"{segment_text}"
    )


def get_nmpc_mode_transition_times(solver_df, solver_t):
    segments = get_nmpc_mode_segments(solver_df, solver_t)
    return np.asarray([segment["start_t"] for segment in segments[1:]], dtype=float)


def get_quaternion_error_series(solver_df):
    quat = solver_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    quat_ref = solver_df[["sp_qw", "sp_qx", "sp_qy", "sp_qz"]].to_numpy(dtype=float)
    quat_err = np.zeros((len(solver_df), 4), dtype=float)

    for row_index in range(len(solver_df)):
        q_err = np.empty(4, dtype=float)
        quaternion_error(quat[row_index], quat_ref[row_index], q_err)
        quat_err[row_index] = q_err

    return quat_err


def get_rotation_vector_error_series(solver_df):
    quat = solver_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    quat_ref = solver_df[["sp_qw", "sp_qx", "sp_qy", "sp_qz"]].to_numpy(dtype=float)
    rotation_error = np.zeros((len(solver_df), 3), dtype=float)

    for row_index in range(len(solver_df)):
        q_ref_inv = np.array([quat_ref[row_index, 0], -quat_ref[row_index, 1], -quat_ref[row_index, 2], -quat_ref[row_index, 3]], dtype=float)
        q_err = np.empty(4, dtype=float)
        quat_multiply(q_ref_inv, quat[row_index], q_err)
        if q_err[0] < 0.0:
            q_err *= -1.0
        q_err_vec = q_err[1:4]
        q_err_vec_norm = np.linalg.norm(q_err_vec)
        if q_err_vec_norm > 1e-9:
            rotation_angle = 2.0 * np.arctan2(q_err_vec_norm, q_err[0])
            rotation_error[row_index] = (rotation_angle / q_err_vec_norm) * q_err_vec
        else:
            rotation_error[row_index] = 2.0 * q_err_vec

    return rotation_error


def get_instant_cost_plot_series(solver_df, config_path):
    recovery_mask = get_recovery_mode_mask(solver_df)
    cost_profiles = load_nmpc_cost_weights(config_path, bool(np.any(recovery_mask)))
    flight_weights = cost_profiles[NMPC_FLIGHT_MODE_LABEL]
    recovery_weights = cost_profiles[NMPC_RECOVERY_MODE_LABEL]
    num_samples = len(solver_df)

    stage_scaling = np.full(num_samples, flight_weights["stage_scaling"], dtype=float)
    terminal_scaling = np.full(num_samples, flight_weights["terminal_scaling"], dtype=float)
    terminal_factor = np.full(num_samples, flight_weights["terminal_factor"], dtype=float)
    Q_pos = np.tile(flight_weights["Q_pos"], (num_samples, 1))
    Q_vel = np.tile(flight_weights["Q_vel"], (num_samples, 1))
    Q_acc = np.tile(flight_weights["Q_acc"], (num_samples, 1))
    Q_att = np.tile(flight_weights["Q_att"], (num_samples, 1))
    Q_omega = np.tile(flight_weights["Q_omega"], (num_samples, 1))
    R = np.full(num_samples, flight_weights["R"], dtype=float)
    R_delta_u = np.full(num_samples, flight_weights["R_delta_u"], dtype=float)

    if np.any(recovery_mask):
        Q_pos[recovery_mask] = recovery_weights["Q_pos"]
        Q_vel[recovery_mask] = recovery_weights["Q_vel"]
        Q_acc[recovery_mask] = recovery_weights["Q_acc"]
        Q_att[recovery_mask] = recovery_weights["Q_att"]
        Q_omega[recovery_mask] = recovery_weights["Q_omega"]
        R[recovery_mask] = recovery_weights["R"]
        R_delta_u[recovery_mask] = recovery_weights["R_delta_u"]

    ex = solver_df["px"].to_numpy(dtype=float) - solver_df["sp_px"].to_numpy(dtype=float)
    ey = solver_df["py"].to_numpy(dtype=float) - solver_df["sp_py"].to_numpy(dtype=float)
    ez = solver_df["pz"].to_numpy(dtype=float) - solver_df["sp_pz"].to_numpy(dtype=float)
    evx = solver_df["vx"].to_numpy(dtype=float) - solver_df["sp_vx"].to_numpy(dtype=float)
    evy = solver_df["vy"].to_numpy(dtype=float) - solver_df["sp_vy"].to_numpy(dtype=float)
    evz = solver_df["vz"].to_numpy(dtype=float) - solver_df["sp_vz"].to_numpy(dtype=float)
    att_err = get_rotation_vector_error_series(solver_df)
    omega = solver_df[["wx", "wy", "wz"]].to_numpy(dtype=float)
    solver_u = solver_df[[f"solver_u{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
    pred_acc = compute_predicted_world_acceleration_from_solver_commands(solver_df)
    acc_ref = solver_df[[f"p{PARAM_ACC_REF + axis_index:02d}" for axis_index in range(3)]].to_numpy(dtype=float)
    e_acc = pred_acc - acc_ref
    prev_command_rps = solver_df[[f"prev_command_rps{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
    kf = solver_df[[f"p{PARAM_KF + motor_index:02d}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
    prev_u = kf * prev_command_rps * prev_command_rps
    delta_u = solver_u - prev_u

    stage_pos = 0.5 * stage_scaling[:, None] * np.column_stack([
        Q_pos[:, 0] * ex * ex,
        Q_pos[:, 1] * ey * ey,
        Q_pos[:, 2] * ez * ez,
    ])
    terminal_pos = 0.5 * (terminal_scaling * terminal_factor)[:, None] * np.column_stack([
        Q_pos[:, 0] * ex * ex,
        Q_pos[:, 1] * ey * ey,
        Q_pos[:, 2] * ez * ez,
    ])
    stage_vel = 0.5 * stage_scaling[:, None] * np.column_stack([
        Q_vel[:, 0] * evx * evx,
        Q_vel[:, 1] * evy * evy,
        Q_vel[:, 2] * evz * evz,
    ])
    stage_acc = 0.5 * stage_scaling[:, None] * Q_acc * e_acc * e_acc
    terminal_vel = 0.5 * (terminal_scaling * terminal_factor)[:, None] * np.column_stack([
        Q_vel[:, 0] * evx * evx,
        Q_vel[:, 1] * evy * evy,
        Q_vel[:, 2] * evz * evz,
    ])
    stage_att = 0.5 * stage_scaling[:, None] * Q_att * att_err * att_err
    terminal_att = 0.5 * (terminal_scaling * terminal_factor)[:, None] * Q_att * att_err * att_err
    stage_omega = 0.5 * stage_scaling[:, None] * Q_omega * omega * omega
    terminal_omega = 0.5 * (terminal_scaling * terminal_factor)[:, None] * Q_omega * omega * omega
    stage_u = 0.5 * stage_scaling[:, None] * R[:, None] * solver_u * solver_u
    stage_delta_u = 0.5 * stage_scaling[:, None] * R_delta_u[:, None] * delta_u * delta_u

    series = {
        "instant_stage_pos_xyz": stage_pos,
        "instant_terminal_pos_xyz": terminal_pos,
        "instant_stage_vel_xyz": stage_vel,
        "instant_stage_acc_xyz": stage_acc,
        "instant_terminal_vel_xyz": terminal_vel,
        "instant_stage_att_xyz": stage_att,
        "instant_terminal_att_xyz": terminal_att,
        "instant_stage_omega_xyz": stage_omega,
        "instant_terminal_omega_xyz": terminal_omega,
        "instant_stage_u_motors": stage_u,
        "instant_stage_delta_u_motors": stage_delta_u,
    }
    series["instant_stage_total"] = (
        stage_pos.sum(axis=1)
        + stage_vel.sum(axis=1)
        + stage_acc.sum(axis=1)
        + stage_att.sum(axis=1)
        + stage_omega.sum(axis=1)
        + stage_u.sum(axis=1)
        + stage_delta_u.sum(axis=1)
    )
    series["instant_terminal_total"] = (
        terminal_pos.sum(axis=1)
        + terminal_vel.sum(axis=1)
        + terminal_att.sum(axis=1)
        + terminal_omega.sum(axis=1)
    )
    series["instant_total"] = series["instant_stage_total"] + series["instant_terminal_total"]

    for key, values in series.items():
        series[key] = np.clip(values, COST_PLOT_YMIN, COST_PLOT_YMAX)

    return series


def build_instant_cost_component_matrix(instant_cost_series):
    component_specs = [
        ("instant_stage_pos_xyz", ["stage pos x", "stage pos y", "stage pos z"]),
        ("instant_terminal_pos_xyz", ["terminal pos x", "terminal pos y", "terminal pos z"]),
        ("instant_stage_vel_xyz", ["stage vel x", "stage vel y", "stage vel z"]),
        ("instant_terminal_vel_xyz", ["terminal vel x", "terminal vel y", "terminal vel z"]),
        ("instant_stage_acc_xyz", ["stage acc x", "stage acc y", "stage acc z"]),
        ("instant_stage_att_xyz", ["stage att roll", "stage att pitch", "stage att yaw"]),
        ("instant_terminal_att_xyz", ["terminal att roll", "terminal att pitch", "terminal att yaw"]),
        ("instant_stage_omega_xyz", ["stage omega x", "stage omega y", "stage omega z"]),
        ("instant_terminal_omega_xyz", ["terminal omega x", "terminal omega y", "terminal omega z"]),
        ("instant_stage_u_motors", ["stage u0", "stage u1", "stage u2", "stage u3"]),
        ("instant_stage_delta_u_motors", ["stage du0", "stage du1", "stage du2", "stage du3"]),
    ]

    labels = []
    columns = []
    for series_key, series_labels in component_specs:
        values = np.asarray(instant_cost_series[series_key], dtype=float)
        if values.ndim != 2:
            raise RuntimeError(f"{series_key} must be a 2D array")
        if values.shape[1] != len(series_labels):
            raise RuntimeError(
                f"{series_key} has {values.shape[1]} columns but {len(series_labels)} labels were configured"
            )
        for column_index, label in enumerate(series_labels):
            labels.append(label)
            columns.append(values[:, column_index])

    matrix = np.column_stack(columns)
    if np.any(~np.isfinite(matrix)):
        raise RuntimeError("Non-finite values found in reconstructed instant cost components")
    if np.any(matrix < 0.0):
        raise RuntimeError("Negative values found in reconstructed instant cost components")

    return labels, matrix


def build_dominant_cost_series(solver_t, instant_cost_series, window_start_s, window_end_s, top_count):
    if not window_end_s > window_start_s:
        raise RuntimeError(
            f"Dominant cost window end ({window_end_s}) must be greater than start ({window_start_s})"
        )
    if top_count < 1:
        raise RuntimeError(f"Dominant cost top count must be at least 1, got {top_count}")

    labels, component_matrix = build_instant_cost_component_matrix(instant_cost_series)
    solver_t_values = solver_t.to_numpy(dtype=float)
    window_mask = (solver_t_values >= window_start_s) & (solver_t_values <= window_end_s)
    if not np.any(window_mask):
        available_start_s = float(solver_t_values.min())
        available_end_s = float(solver_t_values.max())
        print(
            f"Dominant cost window {window_start_s:.3f}s to {window_end_s:.3f}s has no overlap with "
            f"available solver samples ({available_start_s:.3f}s to {available_end_s:.3f}s); "
            f"using the full available range instead"
        )
        window_start_s = available_start_s
        window_end_s = available_end_s
        window_mask = (solver_t_values >= window_start_s) & (solver_t_values <= window_end_s)

    window_t = solver_t_values[window_mask]
    window_components = component_matrix[window_mask]
    total = np.sum(window_components, axis=1)
    if np.any(total <= 0.0):
        raise RuntimeError(
            f"Non-positive reconstructed instant total cost inside {window_start_s:.3f}s to {window_end_s:.3f}s"
        )

    component_percentages = 100.0 * window_components / total[:, None]
    component_percentage_area = np.trapz(component_percentages, window_t, axis=0)
    sorted_indices = np.argsort(component_percentage_area)[::-1]
    selected_indices = sorted_indices[:top_count]
    other_indices = sorted_indices[top_count:]

    selected_labels = [labels[index] for index in selected_indices]
    raw_values = window_components[:, selected_indices]
    if other_indices.size > 0:
        selected_labels.append("others")
        raw_values = np.column_stack((raw_values, np.sum(window_components[:, other_indices], axis=1)))

    percentages = 100.0 * raw_values / total[:, None]
    dominant_indices = np.argmax(window_components, axis=1)
    dominant_labels = np.asarray(labels, dtype=object)[dominant_indices]
    dominant_values = window_components[np.arange(window_components.shape[0]), dominant_indices]
    dominant_percentages = 100.0 * dominant_values / total

    return {
        "t": window_t,
        "window_mask": window_mask,
        "labels": selected_labels,
        "raw_values": raw_values,
        "percentages": percentages,
        "dominant_labels": dominant_labels,
        "dominant_values": dominant_values,
        "dominant_percentages": dominant_percentages,
        "window_start_s": float(window_start_s),
        "window_end_s": float(window_end_s),
        "actual_window_start_s": float(window_t[0]),
        "actual_window_end_s": float(window_t[-1]),
    }


def get_dominant_cost_window_solver_data(solver_df, dominant_cost_series, gap_center):
    window_solver_df = solver_df[dominant_cost_series["window_mask"]]
    if window_solver_df.empty:
        raise RuntimeError("Dominant cost window mask produced no solver rows")

    roll, pitch, yaw = quaternion_to_euler_deg(
        window_solver_df["qw"].to_numpy(dtype=float),
        window_solver_df["qx"].to_numpy(dtype=float),
        window_solver_df["qy"].to_numpy(dtype=float),
        window_solver_df["qz"].to_numpy(dtype=float),
    )

    position = window_solver_df[["px", "py", "pz"]].to_numpy(dtype=float)
    position[:, 0] -= float(gap_center[0])
    position[:, 1] -= float(gap_center[1])

    return {
        "position": position,
        "velocity": window_solver_df[["vx", "vy", "vz"]].to_numpy(dtype=float),
        "orientation_deg": np.column_stack((roll, pitch, yaw)),
    }


def write_dominant_cost_distribution_log(log_path, solver_df, dominant_cost_series, slice_count, gap_center):
    if slice_count < 2:
        raise RuntimeError(f"Dominant cost distribution slice count must be at least 2, got {slice_count}")

    t = dominant_cost_series["t"]
    target_times = np.linspace(
        dominant_cost_series["window_start_s"],
        dominant_cost_series["window_end_s"],
        slice_count,
    )
    sample_indices = []
    seen_sample_indices = set()
    for target_time in target_times:
        if target_time > dominant_cost_series["actual_window_end_s"]:
            continue
        sample_index = int(np.argmin(np.abs(t - target_time)))
        if sample_index in seen_sample_indices:
            continue
        sample_indices.append(sample_index)
        seen_sample_indices.add(sample_index)

    window_data = get_dominant_cost_window_solver_data(solver_df, dominant_cost_series, gap_center)
    labels = dominant_cost_series["labels"]
    raw_values = dominant_cost_series["raw_values"]

    with Path(log_path).open("w", encoding="utf-8") as f:
        header = [
            "slice",
            "time_s",
            "position_x_m",
            "position_y_m",
            "position_z_m",
            "velocity_x_mps",
            "velocity_y_mps",
            "velocity_z_mps",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ]
        header.extend(f"cost_{label.replace(' ', '_')}" for label in labels)
        f.write(",".join(header) + "\n")

        for output_index, sample_index in enumerate(sample_indices):
            row = [
                str(output_index + 1),
                f"{t[sample_index]:.3f}",
                f"{window_data['position'][sample_index, 0]:.3f}",
                f"{window_data['position'][sample_index, 1]:.3f}",
                f"{window_data['position'][sample_index, 2]:.3f}",
                f"{window_data['velocity'][sample_index, 0]:.3f}",
                f"{window_data['velocity'][sample_index, 1]:.3f}",
                f"{window_data['velocity'][sample_index, 2]:.3f}",
                f"{window_data['orientation_deg'][sample_index, 0]:.3f}",
                f"{window_data['orientation_deg'][sample_index, 1]:.3f}",
                f"{window_data['orientation_deg'][sample_index, 2]:.3f}",
            ]
            row.extend(f"{value:.3f}" for value in raw_values[sample_index])
            f.write(",".join(row) + "\n")


def summarize_series(series):
    clean_series = pd.Series(series).dropna()

    if clean_series.empty:
        raise RuntimeError("Cannot summarize an empty series")

    return {
        "mean": float(clean_series.mean()),
        "p50": float(clean_series.quantile(0.50)),
        "p90": float(clean_series.quantile(0.90)),
        "p99": float(clean_series.quantile(0.99)),
        "max": float(clean_series.max()),
    }


def quaternion_to_euler_deg(qw, qx, qy, qz):
    roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
    return roll, pitch, yaw


def quat_multiply(q1, q2, out):
    out[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    out[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    out[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    out[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]


def quaternion_error(q, q_ref, out):
    q_ref_inv = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]], dtype=float)
    q_err = np.empty(4, dtype=float)
    quat_multiply(q_ref_inv, q, q_err)
    out[0] = q_err[0] - 1.0
    out[1] = q_err[1]
    out[2] = q_err[2]
    out[3] = q_err[3]


def load_angle_offset_and_gap_center(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    angle_offset_deg = float(cfg["nmpc"]["angle_offset_deg"])
    collision_cfg = cfg["trajectory"]["collision"]
    gap_ref1_mm = np.asarray(collision_cfg["gap_ref1_mm"], dtype=float)
    gap_ref2_mm = np.asarray(collision_cfg["gap_ref2_mm"], dtype=float)
    gap_center = 0.5 * (gap_ref1_mm[:2] + gap_ref2_mm[:2]) / 1000.0
    return angle_offset_deg, gap_center


def rotate_believed_setpoints_to_ground_truth(solver_df, angle_offset_deg, gap_center):
    theta = np.deg2rad(angle_offset_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cx = float(gap_center[0])
    cy = float(gap_center[1])

    dx = solver_df["sp_px"].to_numpy(dtype=float) - cx
    dy = solver_df["sp_py"].to_numpy(dtype=float) - cy
    solver_df["sp_px"] = cos_t * dx - sin_t * dy + cx
    solver_df["sp_py"] = sin_t * dx + cos_t * dy + cy

    vx = solver_df["sp_vx"].to_numpy(dtype=float)
    vy = solver_df["sp_vy"].to_numpy(dtype=float)
    solver_df["sp_vx"] = cos_t * vx - sin_t * vy
    solver_df["sp_vy"] = sin_t * vx + cos_t * vy

    quat = solver_df[["sp_qw", "sp_qx", "sp_qy", "sp_qz"]].to_numpy(dtype=float)
    q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)], dtype=float)
    rotated = np.empty_like(quat)
    out = np.empty(4, dtype=float)
    for row_index in range(len(quat)):
        quat_multiply(q_yaw, quat[row_index], out)
        rotated[row_index] = out
    solver_df["sp_qw"] = rotated[:, 0]
    solver_df["sp_qx"] = rotated[:, 1]
    solver_df["sp_qy"] = rotated[:, 2]
    solver_df["sp_qz"] = rotated[:, 3]


def quat_to_rotmat(quaternion):
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


def cross3(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=float,
    )


def build_inertia_matrix(params):
    return np.array(
        [
            [params[1], params[2], params[3]],
            [params[2], params[4], params[5]],
            [params[3], params[5], params[6]],
        ],
        dtype=float,
    )


def compute_body_wrench_from_motor_forces(params, motor_forces):
    force_body = np.zeros(3, dtype=float)
    torque_body = np.zeros(3, dtype=float)
    for axis in range(3):
        for motor_index in range(NUM_MOTORS):
            alloc_base = PARAM_ALLOC + 6 * motor_index
            force_body[axis] += params[alloc_base + axis] * motor_forces[motor_index]
            torque_body[axis] += params[alloc_base + 3 + axis] * motor_forces[motor_index]
    return force_body, torque_body


def compute_body_wrench_from_rps(params, motor_rps):
    motor_forces = np.zeros(NUM_MOTORS, dtype=float)
    for motor_index in range(NUM_MOTORS):
        kf = params[PARAM_KF + motor_index]
        if not np.isfinite(kf) or not kf > 0.0:
            raise RuntimeError(f"Invalid thrust coefficient p[{PARAM_KF + motor_index}]={kf}")
        motor_forces[motor_index] = kf * motor_rps[motor_index] * motor_rps[motor_index]
    return compute_body_wrench_from_motor_forces(params, motor_forces)


def compute_predicted_world_acceleration_from_solver_commands(solver_df):
    acceleration = np.zeros((len(solver_df), 3), dtype=float)
    for row_index, row in solver_df.iterrows():
        params = extract_params(row)
        mass = params[0]
        if not np.isfinite(mass) or not mass > 0.0:
            raise RuntimeError(f"Invalid mass p[0]={mass} at solver row {row_index}")

        gravity_vec = params[7:10]
        quat = row[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
        rotation = quat_to_rotmat(quat)
        velocity = row[["vx", "vy", "vz"]].to_numpy(dtype=float)
        velocity_body = rotation.T @ velocity
        drag_force_body = compute_body_velocity_drag_force(velocity_body)
        motor_forces = row[[f"solver_u{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
        force_body, _ = compute_body_wrench_from_motor_forces(params, motor_forces)
        disturbance_force_body = params[PARAM_DIST_FORCE:PARAM_DIST_FORCE + 3]
        acceleration[row_index] = (
            gravity_vec
            + rotation @ ((force_body + drag_force_body + disturbance_force_body) / mass)
        )
    return acceleration


def compute_modeled_body_acceleration(rotation_world_from_body, gravity_vec, force_body, drag_force_body, disturbance_force_body, mass):
    world_acceleration = (
        gravity_vec
        + rotation_world_from_body @ ((force_body + drag_force_body + disturbance_force_body) / mass)
    )
    return rotation_world_from_body.T @ world_acceleration


def extract_params(row):
    return np.array([float(row[f"p{param_index:02d}"]) for param_index in range(NUM_PARAMS)], dtype=float)


def compute_acceleration_traces(solver_df):
    num_samples = len(solver_df)
    nan3 = np.full((num_samples, 3), np.nan, dtype=float)
    traces = {
        "command_linear": nan3.copy(),
        "expected_linear": nan3.copy(),
        "observed_linear": nan3.copy(),
        "command_angular": nan3.copy(),
        "expected_angular": nan3.copy(),
        "observed_angular": nan3.copy(),
    }

    previous_velocity = None
    previous_omega = None
    previous_position_velocity_timestamp_us = None
    previous_angular_velocity_timestamp_us = None

    for row_index, row in solver_df.iterrows():
        params = extract_params(row)
        mass = params[0]
        if not np.isfinite(mass) or not mass > 0.0:
            raise RuntimeError(f"Invalid mass p[0]={mass} at solver row {row_index}")

        gravity_vec = params[7:10]
        inertia = build_inertia_matrix(params)
        inertia_inv = np.linalg.inv(inertia)
        r_com = params[PARAM_COM:PARAM_COM + 3]
        quat = row[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
        rotation = quat_to_rotmat(quat)
        gravity_force_body = rotation.T @ (mass * gravity_vec)
        torque_gravity = cross3(r_com, gravity_force_body)
        omega = row[["wx", "wy", "wz"]].to_numpy(dtype=float)
        omega_cross_inertia_omega = cross3(omega, inertia @ omega)
        velocity = row[["vx", "vy", "vz"]].to_numpy(dtype=float)
        velocity_body = rotation.T @ velocity
        drag_force_body = compute_body_velocity_drag_force(velocity_body)
        disturbance_force_body = params[PARAM_DIST_FORCE:PARAM_DIST_FORCE + 3]

        command_motor_forces = row[[f"solver_u{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
        command_force_body, command_torque_body = compute_body_wrench_from_motor_forces(params, command_motor_forces)
        traces["command_linear"][row_index] = compute_modeled_body_acceleration(
            rotation,
            gravity_vec,
            command_force_body,
            drag_force_body,
            disturbance_force_body,
            mass,
        )
        traces["command_angular"][row_index] = inertia_inv @ (
            command_torque_body + torque_gravity - omega_cross_inertia_omega
        )

        expected_motor_rps = row[[f"estimated_rps{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=float)
        expected_force_body, expected_torque_body = compute_body_wrench_from_rps(params, expected_motor_rps)
        traces["expected_linear"][row_index] = compute_modeled_body_acceleration(
            rotation,
            gravity_vec,
            expected_force_body,
            drag_force_body,
            disturbance_force_body,
            mass,
        )
        traces["expected_angular"][row_index] = inertia_inv @ (
            expected_torque_body + torque_gravity - omega_cross_inertia_omega
        )

        position_velocity_timestamp_us = int(row["position_velocity_timestamp_us"])
        if (
            previous_velocity is not None
            and previous_position_velocity_timestamp_us is not None
            and position_velocity_timestamp_us > previous_position_velocity_timestamp_us
        ):
            dt_s = (position_velocity_timestamp_us - previous_position_velocity_timestamp_us) * 1e-6
            measured_world_acc = (velocity - previous_velocity) / dt_s
            traces["observed_linear"][row_index] = rotation.T @ measured_world_acc
        previous_velocity = velocity.copy()
        previous_position_velocity_timestamp_us = position_velocity_timestamp_us

        angular_velocity_timestamp_us = int(row["angular_velocity_timestamp_us"])
        if (
            previous_omega is not None
            and previous_angular_velocity_timestamp_us is not None
            and angular_velocity_timestamp_us > previous_angular_velocity_timestamp_us
        ):
            dt_s = (angular_velocity_timestamp_us - previous_angular_velocity_timestamp_us) * 1e-6
            traces["observed_angular"][row_index] = (omega - previous_omega) / dt_s
        previous_omega = omega.copy()
        previous_angular_velocity_timestamp_us = angular_velocity_timestamp_us

    return traces


def get_fresh_position_velocity_mask(solver_df):
    position_velocity_timestamp_us = solver_df["position_velocity_timestamp_us"].to_numpy(dtype=np.int64)
    if position_velocity_timestamp_us.size == 0:
        raise RuntimeError("Solver dataframe is empty")

    fresh_mask = np.zeros(position_velocity_timestamp_us.shape[0], dtype=bool)
    fresh_mask[0] = True
    fresh_mask[1:] = np.diff(position_velocity_timestamp_us) > 0
    return fresh_mask


def add_reference_line_to_axis(ax, reference_times):
    if not ax.axison:
        return
    for reference_time in reference_times:
        ax.axvline(reference_time, color="black", linestyle="--", linewidth=0.8, alpha=0.9)


def add_reference_lines_to_axes(axes, reference_times):
    for ax in np.asarray(axes, dtype=object).ravel():
        add_reference_line_to_axis(ax, reference_times)


def masked_mode_series(values, mode_mask, selected_mode_mask):
    values_array = np.asarray(values, dtype=float)
    selected = mode_mask == selected_mode_mask
    return np.where(selected, values_array, np.nan)


def get_nmpc_px4_transition_times(px4_flight_df, px4_flight_t):
    px4_active = px4_flight_df["px4_position_control_active"].to_numpy(dtype=bool)
    nmpc_active = px4_flight_df["nmpc_direct_actuator_active"].to_numpy(dtype=bool)
    transition_mask = (px4_active & np.r_[False, nmpc_active[:-1]]) | (nmpc_active & np.r_[False, px4_active[:-1]])
    return px4_flight_t.to_numpy(dtype=float)[transition_mask]


def get_nmpc_setpoint_change_times(solver_df, solver_t):
    require_columns(solver_df, NMPC_SETPOINT_COLUMNS, "solver CSV")
    setpoints = solver_df[NMPC_SETPOINT_COLUMNS].to_numpy(dtype=float)
    finite_mask = np.isfinite(setpoints)
    if not np.all(finite_mask):
        bad_row, bad_col = np.argwhere(~finite_mask)[0]
        raise RuntimeError(
            f"Non-finite NMPC setpoint in column {NMPC_SETPOINT_COLUMNS[int(bad_col)]} "
            f"at solver row {int(bad_row)}"
        )

    changed = np.any(setpoints[1:] != setpoints[:-1], axis=1)
    change_indices = np.flatnonzero(changed) + 1
    return solver_t.to_numpy(dtype=float)[change_indices]


def add_plotly_reference_lines(figure, reference_times, subplot_positions):
    for reference_time in reference_times:
        for row, col in subplot_positions:
            figure.add_vline(
                x=reference_time,
                line_color="black",
                line_dash="dash",
                line_width=0.8,
                opacity=0.9,
                row=row,
                col=col,
            )


def generate_offline_disturbance_pdf(offline_df, pdf_path):
    if offline_df.empty:
        raise RuntimeError("Offline disturbance replay is empty")

    t = (offline_df["sample_timestamp_us"] - offline_df["sample_timestamp_us"].iloc[0]) / 1e6
    force_est_cols = ["estimated_force_bx", "estimated_force_by", "estimated_force_bz"]
    torque_est_cols = ["estimated_torque_bx", "estimated_torque_by", "estimated_torque_bz"]
    force_meas_cols = ["force_measurement_bx", "force_measurement_by", "force_measurement_bz"]
    torque_meas_cols = ["torque_measurement_bx", "torque_measurement_by", "torque_measurement_bz"]
    fixed_force_cols = ["fixed_dist_force_bx", "fixed_dist_force_by", "fixed_dist_force_bz"]
    fixed_torque_cols = ["fixed_dist_torque_bx", "fixed_dist_torque_by", "fixed_dist_torque_bz"]
    force_var_cols = ["force_var_x", "force_var_y", "force_var_z"]
    torque_var_cols = ["torque_var_x", "torque_var_y", "torque_var_z"]
    force_bound_cols = ["force_bound_x", "force_bound_y", "force_bound_z"]
    torque_bound_cols = ["torque_bound_x", "torque_bound_y", "torque_bound_z"]
    axis_labels = ["x", "y", "z"]
    axis_colors = ["tab:blue", "tab:orange", "tab:green"]

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
        for axis_index, axis_label in enumerate(axis_labels):
            ax = axes[axis_index, 0]
            sigma = np.sqrt(offline_df[force_var_cols[axis_index]].values)
            ax.fill_between(t, offline_df[force_est_cols[axis_index]] - sigma, offline_df[force_est_cols[axis_index]] + sigma,
                            alpha=0.2, color=axis_colors[axis_index], label="estimate +/- 1 sigma")
            ax.plot(t, offline_df[force_est_cols[axis_index]], color=axis_colors[axis_index], linewidth=0.9, label="estimate")
            ax.plot(t, offline_df[force_meas_cols[axis_index]], color="gray", linewidth=0.6, alpha=0.6, label="measurement")
            ax.plot(t, offline_df[fixed_force_cols[axis_index]], color="black", linewidth=0.8, linestyle="--", label="fixed flight value")
            ax.plot(t, offline_df[force_bound_cols[axis_index]], color="red", linewidth=0.6, linestyle=":", label="bound")
            ax.plot(t, -offline_df[force_bound_cols[axis_index]], color="red", linewidth=0.6, linestyle=":")
            ax.set_ylabel(f"F{axis_label} [N]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

            ax = axes[axis_index, 1]
            sigma = np.sqrt(offline_df[torque_var_cols[axis_index]].values)
            ax.fill_between(t, offline_df[torque_est_cols[axis_index]] - sigma, offline_df[torque_est_cols[axis_index]] + sigma,
                            alpha=0.2, color=axis_colors[axis_index], label="estimate +/- 1 sigma")
            ax.plot(t, offline_df[torque_est_cols[axis_index]], color=axis_colors[axis_index], linewidth=0.9, label="estimate")
            ax.plot(t, offline_df[torque_meas_cols[axis_index]], color="gray", linewidth=0.6, alpha=0.6, label="measurement")
            ax.plot(t, offline_df[fixed_torque_cols[axis_index]], color="black", linewidth=0.8, linestyle="--", label="fixed flight value")
            ax.plot(t, offline_df[torque_bound_cols[axis_index]], color="red", linewidth=0.6, linestyle=":", label="bound")
            ax.plot(t, -offline_df[torque_bound_cols[axis_index]], color="red", linewidth=0.6, linestyle=":")
            ax.set_ylabel(f"T{axis_label} [Nm]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        axes[-1, 0].set_xlabel("Time [s]")
        axes[-1, 1].set_xlabel("Time [s]")
        add_reference_lines_to_axes(axes, [])
        fig.suptitle("Offline Disturbance Replay", fontsize=14, y=0.995)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        axes[0].plot(t, offline_df["force_confidence"], linewidth=0.9, label="force confidence")
        axes[0].plot(t, offline_df["torque_confidence"], linewidth=0.9, label="torque confidence")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel("score")
        axes[0].set_title("Offline estimator confidence")

        axes[1].plot(t, offline_df["force_innovation_nis"], linewidth=0.8, label="force NIS")
        axes[1].plot(t, offline_df["torque_innovation_nis"], linewidth=0.8, label="torque NIS")
        axes[1].axhline(25.0, color="red", linestyle="--", linewidth=0.6, label="gate")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel("NIS")
        axes[1].set_title("Offline estimator gating metrics")

        axes[2].plot(t, offline_df["force_measurement_valid"], linewidth=0.8, label="force update")
        axes[2].plot(t, offline_df["torque_measurement_valid"], linewidth=0.8, label="torque update")
        axes[2].plot(t, offline_df["combined_valid"], linewidth=0.8, label="combined valid")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylabel("flag")
        axes[2].set_title("Offline replay update availability")

        axes[3].plot(t, offline_df["position_velocity_timestamp_us"].diff() / 1e3, linewidth=0.8, label="position_velocity dt")
        axes[3].plot(t, offline_df["angular_velocity_timestamp_us"].diff() / 1e3, linewidth=0.8, label="angular_velocity dt")
        axes[3].plot(t, offline_df["attitude_timestamp_us"].diff() / 1e3, linewidth=0.8, label="attitude dt")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylabel("ms")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_title("Source timestamp increments used by offline replay")

        add_reference_lines_to_axes(axes, [])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def add_plotly_series(figure, row, col, x, y, name, color, dash=None, fill=None, showlegend=True):
    line = {"color": color}
    if dash is not None:
        line["dash"] = dash
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=line,
            fill=fill,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def add_dominant_cost_traces_to_plotly(figure, solver_df, dominant_cost_series, raw_row, percent_row, reference_col, gap_center):
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    t = dominant_cost_series["t"]
    raw_values = dominant_cost_series["raw_values"]
    percentages = dominant_cost_series["percentages"]
    labels = dominant_cost_series["labels"]

    for label_index, label in enumerate(labels):
        color = colors[label_index % len(colors)]
        figure.add_trace(
            go.Scatter(
                x=t,
                y=np.maximum(raw_values[:, label_index], DOMINANT_COST_PLOT_YMIN),
                mode="lines",
                name=f"{label} raw",
                line={"color": color},
            ),
            row=raw_row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=t,
                y=percentages[:, label_index],
                mode="lines",
                name=f"{label} percent",
                line={"color": color},
                stackgroup="dominant_cost_percent",
                showlegend=False,
            ),
            row=percent_row,
            col=1,
        )

    window_data = get_dominant_cost_window_solver_data(solver_df, dominant_cost_series, gap_center)
    for axis_index, axis_label in enumerate(["x", "y", "z"]):
        add_plotly_series(
            figure,
            raw_row,
            reference_col,
            t,
            window_data["position"][:, axis_index],
            f"p{axis_label}",
            colors[axis_index],
        )
    for axis_index, axis_label in enumerate(["roll", "pitch", "yaw"]):
        add_plotly_series(
            figure,
            percent_row,
            reference_col,
            t,
            window_data["orientation_deg"][:, axis_index],
            axis_label,
            colors[axis_index],
        )


def add_dominant_cost_pdf_page(pdf, solver_df, dominant_cost_series, nmpc_mode_summary, gap_center):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    t = dominant_cost_series["t"]
    labels = dominant_cost_series["labels"]
    raw_values = dominant_cost_series["raw_values"]
    percentages = dominant_cost_series["percentages"]
    window_data = get_dominant_cost_window_solver_data(solver_df, dominant_cost_series, gap_center)

    for label_index, label in enumerate(labels):
        axes[0].plot(
            t,
            np.maximum(raw_values[:, label_index], DOMINANT_COST_PLOT_YMIN),
            linewidth=0.9,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylim(
        DOMINANT_COST_PLOT_YMIN,
        max(COST_PLOT_YMAX, float(np.max(np.sum(raw_values, axis=1))) * 1.2),
    )
    axes[0].set_ylabel("weighted cost")
    axes[0].set_title("Highest reconstructed NMPC cost contributors, raw values")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].stackplot(t, percentages.T, labels=labels, alpha=0.8)
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_ylabel("% of reconstructed instant cost")
    axes[1].set_title("Highest reconstructed NMPC cost contributors, percentage of total")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(True, alpha=0.3)

    for axis_index, axis_label in enumerate(["x", "y", "z"]):
        axes[2].plot(t, window_data["position"][:, axis_index], linewidth=0.9, label=f"p{axis_label}")
    axes[2].set_ylabel("m")
    axes[2].set_title("Position relative to gap center during dominant-cost window")
    axes[2].legend(fontsize=8, ncol=3)
    axes[2].grid(True, alpha=0.3)

    for axis_index, axis_label in enumerate(["roll", "pitch", "yaw"]):
        axes[3].plot(t, window_data["orientation_deg"][:, axis_index], linewidth=0.9, label=axis_label)
    axes[3].set_ylabel("deg")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title("Orientation reference during dominant-cost window")
    axes[3].legend(fontsize=8, ncol=3)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(
        (
            "Dominant NMPC Cost Contributions "
            f"({dominant_cost_series['actual_window_start_s']:.2f}s to {dominant_cost_series['actual_window_end_s']:.2f}s)\n"
            f"{nmpc_mode_summary}"
        ),
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_binary_mode_lane(ax, t, active, lane, label, color):
    t_values = np.asarray(t, dtype=float)
    active_values = np.asarray(active, dtype=float)
    lane_low = np.full(len(t_values), lane - 0.35, dtype=float)
    lane_high = np.full(len(t_values), lane + 0.35, dtype=float)
    ax.fill_between(
        t_values,
        lane_low,
        lane_high,
        where=active_values > 0.5,
        step="post",
        alpha=0.25,
        color=color,
        label=label,
    )
    ax.step(t_values, lane + 0.3 * active_values, where="post", linewidth=0.9, color=color)


def add_mode_background_to_axis(ax, mode_t, px4_mode_active, nmpc_mode_active, solver_t, recovery_mode_active):
    recovery_active = np.asarray(recovery_mode_active, dtype=float)
    flight_active = 1.0 - recovery_active
    add_binary_mode_lane(ax, mode_t, px4_mode_active, 0.0, "PX4 position control", "tab:orange")
    add_binary_mode_lane(ax, mode_t, nmpc_mode_active, 1.0, "PX4 offboard actuator", "tab:blue")
    add_binary_mode_lane(ax, solver_t, flight_active, 2.0, NMPC_FLIGHT_MODE_LABEL, "tab:green")
    add_binary_mode_lane(ax, solver_t, recovery_active, 3.0, NMPC_RECOVERY_MODE_LABEL, "tab:red")
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["PX4 pos", "actuator", "flight", "recovery"])
    ax.set_ylabel("mode lane")
    ax.set_title("Offboard control mode and active NMPC solver mode")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def generate_offline_disturbance_html(offline_df, html_path):
    if offline_df.empty:
        raise RuntimeError("Offline disturbance replay is empty")

    t = ((offline_df["sample_timestamp_us"] - offline_df["sample_timestamp_us"].iloc[0]) / 1e6).to_numpy(dtype=float)
    force_est_cols = ["estimated_force_bx", "estimated_force_by", "estimated_force_bz"]
    torque_est_cols = ["estimated_torque_bx", "estimated_torque_by", "estimated_torque_bz"]
    force_meas_cols = ["force_measurement_bx", "force_measurement_by", "force_measurement_bz"]
    torque_meas_cols = ["torque_measurement_bx", "torque_measurement_by", "torque_measurement_bz"]
    fixed_force_cols = ["fixed_dist_force_bx", "fixed_dist_force_by", "fixed_dist_force_bz"]
    fixed_torque_cols = ["fixed_dist_torque_bx", "fixed_dist_torque_by", "fixed_dist_torque_bz"]
    force_var_cols = ["force_var_x", "force_var_y", "force_var_z"]
    torque_var_cols = ["torque_var_x", "torque_var_y", "torque_var_z"]
    force_bound_cols = ["force_bound_x", "force_bound_y", "force_bound_z"]
    torque_bound_cols = ["torque_bound_x", "torque_bound_y", "torque_bound_z"]
    axis_labels = ["x", "y", "z"]
    axis_colors = ["blue", "orange", "green"]

    figure = make_subplots(
        rows=7,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=(
            "Force x", "Torque x",
            "Force y", "Torque y",
            "Force z", "Torque z",
            "Estimator confidence", "Estimator gating metrics",
            "Replay update availability", "Source timestamp increments",
            "", "",
            "", "",
        ),
    )

    for axis_index, axis_label in enumerate(axis_labels):
        est_force = offline_df[force_est_cols[axis_index]].to_numpy(dtype=float)
        est_torque = offline_df[torque_est_cols[axis_index]].to_numpy(dtype=float)
        meas_force = offline_df[force_meas_cols[axis_index]].to_numpy(dtype=float)
        meas_torque = offline_df[torque_meas_cols[axis_index]].to_numpy(dtype=float)
        fixed_force = offline_df[fixed_force_cols[axis_index]].to_numpy(dtype=float)
        fixed_torque = offline_df[fixed_torque_cols[axis_index]].to_numpy(dtype=float)
        force_sigma = np.sqrt(offline_df[force_var_cols[axis_index]].to_numpy(dtype=float))
        torque_sigma = np.sqrt(offline_df[torque_var_cols[axis_index]].to_numpy(dtype=float))
        force_bound = offline_df[force_bound_cols[axis_index]].to_numpy(dtype=float)
        torque_bound = offline_df[torque_bound_cols[axis_index]].to_numpy(dtype=float)

        add_plotly_series(figure, axis_index + 1, 1, t, est_force + force_sigma, f"F{axis_label} +1 sigma", axis_colors[axis_index], showlegend=False)
        add_plotly_series(figure, axis_index + 1, 1, t, est_force - force_sigma, f"F{axis_label} -1 sigma", axis_colors[axis_index], fill="tonexty", showlegend=False)
        add_plotly_series(figure, axis_index + 1, 1, t, est_force, f"F{axis_label} estimate", axis_colors[axis_index])
        add_plotly_series(figure, axis_index + 1, 1, t, meas_force, f"F{axis_label} measurement", "gray")
        add_plotly_series(figure, axis_index + 1, 1, t, fixed_force, f"F{axis_label} fixed flight", "black", dash="dash")
        add_plotly_series(figure, axis_index + 1, 1, t, force_bound, f"F{axis_label} bound", "red", dash="dot")
        add_plotly_series(figure, axis_index + 1, 1, t, -force_bound, f"-F{axis_label} bound", "red", dash="dot", showlegend=False)

        add_plotly_series(figure, axis_index + 1, 2, t, est_torque + torque_sigma, f"T{axis_label} +1 sigma", axis_colors[axis_index], showlegend=False)
        add_plotly_series(figure, axis_index + 1, 2, t, est_torque - torque_sigma, f"T{axis_label} -1 sigma", axis_colors[axis_index], fill="tonexty", showlegend=False)
        add_plotly_series(figure, axis_index + 1, 2, t, est_torque, f"T{axis_label} estimate", axis_colors[axis_index])
        add_plotly_series(figure, axis_index + 1, 2, t, meas_torque, f"T{axis_label} measurement", "gray")
        add_plotly_series(figure, axis_index + 1, 2, t, fixed_torque, f"T{axis_label} fixed flight", "black", dash="dash")
        add_plotly_series(figure, axis_index + 1, 2, t, torque_bound, f"T{axis_label} bound", "red", dash="dot")
        add_plotly_series(figure, axis_index + 1, 2, t, -torque_bound, f"-T{axis_label} bound", "red", dash="dot", showlegend=False)

    add_plotly_series(figure, 4, 1, t, offline_df["force_confidence"], "force confidence", "blue")
    add_plotly_series(figure, 4, 1, t, offline_df["torque_confidence"], "torque confidence", "orange")
    add_plotly_series(figure, 4, 2, t, offline_df["force_innovation_nis"], "force NIS", "blue")
    add_plotly_series(figure, 4, 2, t, offline_df["torque_innovation_nis"], "torque NIS", "orange")
    add_plotly_series(figure, 4, 2, t, np.full(len(t), 25.0, dtype=float), "gate", "red", dash="dash")

    add_plotly_series(figure, 5, 1, t, offline_df["force_measurement_valid"], "force update", "blue")
    add_plotly_series(figure, 5, 1, t, offline_df["torque_measurement_valid"], "torque update", "orange")
    add_plotly_series(figure, 5, 1, t, offline_df["combined_valid"], "combined valid", "green")
    add_plotly_series(figure, 5, 2, t, offline_df["position_velocity_timestamp_us"].diff() / 1e3, "position_velocity dt", "blue")
    add_plotly_series(figure, 5, 2, t, offline_df["angular_velocity_timestamp_us"].diff() / 1e3, "angular_velocity dt", "orange")
    add_plotly_series(figure, 5, 2, t, offline_df["attitude_timestamp_us"].diff() / 1e3, "attitude dt", "green")

    figure.update_yaxes(title_text="Force [N]", row=1, col=1)
    figure.update_yaxes(title_text="Torque [Nm]", row=1, col=2)
    figure.update_yaxes(title_text="Force [N]", row=2, col=1)
    figure.update_yaxes(title_text="Torque [Nm]", row=2, col=2)
    figure.update_yaxes(title_text="Force [N]", row=3, col=1)
    figure.update_yaxes(title_text="Torque [Nm]", row=3, col=2)
    figure.update_yaxes(title_text="score", row=4, col=1)
    figure.update_yaxes(title_text="NIS", row=4, col=2)
    figure.update_yaxes(title_text="flag", row=5, col=1)
    figure.update_yaxes(title_text="ms", row=5, col=2)
    figure.update_xaxes(title_text="Time [s]", row=5, col=1)
    figure.update_xaxes(title_text="Time [s]", row=5, col=2)

    figure.update_layout(
        title="Offline Disturbance Replay",
        height=2200,
        width=1400,
        hovermode="x unified",
    )
    figure.write_html(html_path, include_plotlyjs=True)


def generate_diagnostics_html(
    solver_df,
    px4_df,
    px4_flight_df,
    offline_df,
    offline_summary,
    html_path,
    solver_t,
    offline_t,
    px4_t,
    px4_flight_t,
    reference_times,
    solve_time_ms,
    solver_input_dt_ms,
    solver_posvel_dt_ms,
    solver_att_dt_ms,
    solver_ang_dt_ms,
    acceleration_traces,
    optimizer_cost_series,
    instant_cost_series,
    dominant_cost_series,
    nmpc_mode_summary,
    gap_center,
):
    roll, pitch, yaw = quaternion_to_euler_deg(
        solver_df["qw"].values,
        solver_df["qx"].values,
        solver_df["qy"].values,
        solver_df["qz"].values,
    )
    sp_roll, sp_pitch, sp_yaw = quaternion_to_euler_deg(
        solver_df["sp_qw"].values,
        solver_df["sp_qx"].values,
        solver_df["sp_qy"].values,
        solver_df["sp_qz"].values,
    )

    ex = solver_df["px"].to_numpy(dtype=float) - solver_df["sp_px"].to_numpy(dtype=float)
    ey = solver_df["py"].to_numpy(dtype=float) - solver_df["sp_py"].to_numpy(dtype=float)
    ez = solver_df["pz"].to_numpy(dtype=float) - solver_df["sp_pz"].to_numpy(dtype=float)
    e_norm = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    fresh_position_velocity_mask = get_fresh_position_velocity_mask(solver_df)
    error_t = solver_t[fresh_position_velocity_mask]
    error_ex = ex[fresh_position_velocity_mask]
    error_ey = ey[fresh_position_velocity_mask]
    error_ez = ez[fresh_position_velocity_mask]
    error_e_norm = e_norm[fresh_position_velocity_mask]
    recovery_mode_mask = get_recovery_mode_mask(solver_df)
    gap_x = float(gap_center[0])
    gap_y = float(gap_center[1])
    px_rel = solver_df["px"] - gap_x
    py_rel = solver_df["py"] - gap_y
    sp_px_rel = solver_df["sp_px"] - gap_x
    sp_py_rel = solver_df["sp_py"] - gap_y
    px4_x_rel = px4_flight_df["px4_y"] - gap_x
    px4_y_rel = px4_flight_df["px4_x"] - gap_y
    px4_sp_x_rel = px4_flight_df["px4_sp_y"] - gap_x
    px4_sp_y_rel = px4_flight_df["px4_sp_x"] - gap_y

    metrics_mask = fresh_position_velocity_mask & (solver_t >= solver_t.iloc[-1] - 20.0).to_numpy()
    metrics_t = solver_t[metrics_mask]
    metrics_ex = ex[metrics_mask]
    metrics_ey = ey[metrics_mask]
    metrics_ez = ez[metrics_mask]
    metrics_e_norm = np.sqrt(metrics_ex ** 2 + metrics_ey ** 2 + metrics_ez ** 2)

    rmse_x = np.sqrt(np.mean(metrics_ex ** 2)) * 100
    rmse_y = np.sqrt(np.mean(metrics_ey ** 2)) * 100
    rmse_z = np.sqrt(np.mean(metrics_ez ** 2)) * 100
    rmse_3d = np.sqrt(np.mean(metrics_e_norm ** 2)) * 100
    max_x = np.max(np.abs(metrics_ex)) * 100
    max_y = np.max(np.abs(metrics_ey)) * 100
    max_z = np.max(np.abs(metrics_ez)) * 100
    max_3d = np.max(metrics_e_norm) * 100
    mean_x = np.mean(metrics_ex) * 100
    mean_y = np.mean(metrics_ey) * 100
    mean_z = np.mean(metrics_ez) * 100
    mean_3d = np.mean(metrics_e_norm) * 100
    std_x = np.std(metrics_ex) * 100
    std_y = np.std(metrics_ey) * 100
    std_z = np.std(metrics_ez) * 100
    std_3d = np.std(metrics_e_norm) * 100
    p95_x = np.percentile(np.abs(metrics_ex), 95) * 100
    p95_y = np.percentile(np.abs(metrics_ey), 95) * 100
    p95_z = np.percentile(np.abs(metrics_ez), 95) * 100
    p95_3d = np.percentile(metrics_e_norm, 95) * 100

    figure = make_subplots(
        rows=14,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "domain"}],
        ],
        subplot_titles=(
            "Solver timing and per-signal source timestamps", "Real ESC RPS vs NMPC commanded RPS",
            "Real ESC RPS vs NMPC estimated RPS", "Solver motor forces",
            "Position relative to gap center + active NMPC setpoint", "Velocity",
            "Orientation (Euler) + setpoint", "Linear acceleration tracking",
            "Angular acceleration tracking", "Top view",
            "Position error magnitude", "Position error x",
            "Position error y", "Position error z",
            "Offline disturbance force estimate vs fixed in-flight value", "Offline disturbance torque estimate vs fixed in-flight value",
            "Optimizer objective totals", "Optimizer objective term breakdown",
            "Instant cost by axis", "Instant cost by component",
            "Dominant NMPC cost contributors, raw", "Position relative to gap center during dominant-cost window",
            "Dominant NMPC cost contributors, percentage", "Orientation during dominant-cost window",
            "Offline replay confidence and update availability", "",
            "Offboard control mode", "Summary",
        ),
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )

    add_plotly_series(figure, 1, 1, solver_t, solve_time_ms, "solve time", "blue")
    add_plotly_series(figure, 1, 1, solver_t.iloc[1:], solver_input_dt_ms.iloc[1:], "sample dt", "orange")
    add_plotly_series(figure, 1, 1, solver_t.iloc[1:], solver_posvel_dt_ms.iloc[1:], "position/velocity dt", "green")
    add_plotly_series(figure, 1, 1, solver_t.iloc[1:], solver_att_dt_ms.iloc[1:], "attitude dt", "red")
    add_plotly_series(figure, 1, 1, solver_t.iloc[1:], solver_ang_dt_ms.iloc[1:], "angular velocity dt", "purple")
    add_plotly_series(figure, 1, 1, solver_t, np.full(len(solver_t), 10.0, dtype=float), "10ms (100Hz)", "black", dash="dash")

    for motor_index in range(NUM_MOTORS):
        color = PLOTLY_MOTOR_COLORS[motor_index]
        add_plotly_series(figure, 1, 2, solver_t, solver_df[f"command_rps{motor_index}"], f"command_rps{motor_index}", color)
        add_plotly_series(figure, 1, 2, px4_t, px4_df[f"measured_rpm{motor_index}"] / 60.0, f"real_rps{motor_index}", color, dash="dash")
        add_plotly_series(figure, 2, 1, solver_t, solver_df[f"estimated_rps{motor_index}"], f"estimated_rps{motor_index}", color)
        add_plotly_series(figure, 2, 1, px4_t, px4_df[f"measured_rpm{motor_index}"] / 60.0, f"real_rps{motor_index}", color, dash="dash")
        add_plotly_series(figure, 2, 2, solver_t, solver_df[f"solver_u{motor_index}"], f"solver_u{motor_index}", color, dash="dash")

    for values, label, color in [(px_rel, "x", "blue"), (py_rel, "y", "orange"), (solver_df["pz"], "z", "green")]:
        add_plotly_series(figure, 3, 1, solver_t, values, label, color)
    for values, label, color in [(sp_px_rel, "sp_x", "blue"), (sp_py_rel, "sp_y", "orange"), (solver_df["sp_pz"], "sp_z", "green")]:
        add_plotly_series(
            figure,
            3,
            1,
            solver_t,
            masked_mode_series(values, recovery_mode_mask, False),
            f"{label} flight",
            color,
            dash="dash",
        )
        add_plotly_series(
            figure,
            3,
            1,
            solver_t,
            masked_mode_series(values, recovery_mode_mask, True),
            f"{label} recovery",
            color,
            dash="dot",
        )
    add_plotly_series(figure, 3, 1, px4_flight_t, px4_x_rel, "PX4 x", "gray", dash="dot")
    add_plotly_series(figure, 3, 1, px4_flight_t, px4_y_rel, "PX4 y", "black", dash="dot")
    add_plotly_series(figure, 3, 1, px4_flight_t, -px4_flight_df["px4_z"], "PX4 z", "brown", dash="dot")
    add_plotly_series(figure, 3, 1, px4_flight_t, px4_sp_x_rel, "PX4 sp_x", "gray", dash="dash")
    add_plotly_series(figure, 3, 1, px4_flight_t, px4_sp_y_rel, "PX4 sp_y", "black", dash="dash")
    add_plotly_series(figure, 3, 1, px4_flight_t, -px4_flight_df["px4_sp_z"], "PX4 sp_z", "brown", dash="dash")

    for column_name, label, color in [("vx", "vx", "blue"), ("vy", "vy", "orange"), ("vz", "vz", "green")]:
        add_plotly_series(figure, 3, 2, solver_t, solver_df[column_name], label, color)

    for values, label, color, dash in [
        (roll, "roll", "blue", None),
        (pitch, "pitch", "orange", None),
        (yaw, "yaw", "green", None),
        (sp_roll, "sp_roll", "blue", "dash"),
        (sp_pitch, "sp_pitch", "orange", "dash"),
        (sp_yaw, "sp_yaw", "green", "dash"),
    ]:
        add_plotly_series(figure, 4, 1, solver_t, values, label, color, dash=dash)

    axis_labels = ["x", "y", "z"]
    for axis_index, axis_label in enumerate(axis_labels):
        add_plotly_series(figure, 4, 2, solver_t, acceleration_traces["command_linear"][:, axis_index], f"{axis_label} commanded", PLOTLY_MOTOR_COLORS[axis_index])
        add_plotly_series(figure, 4, 2, solver_t, acceleration_traces["expected_linear"][:, axis_index], f"{axis_label} expected", PLOTLY_MOTOR_COLORS[axis_index], dash="dash")
        add_plotly_series(figure, 4, 2, solver_t, acceleration_traces["observed_linear"][:, axis_index], f"{axis_label} observed", PLOTLY_MOTOR_COLORS[axis_index], dash="dot")
        add_plotly_series(figure, 5, 1, solver_t, acceleration_traces["command_angular"][:, axis_index], f"{axis_label} commanded", PLOTLY_MOTOR_COLORS[axis_index])
        add_plotly_series(figure, 5, 1, solver_t, acceleration_traces["expected_angular"][:, axis_index], f"{axis_label} expected", PLOTLY_MOTOR_COLORS[axis_index], dash="dash")
        add_plotly_series(figure, 5, 1, solver_t, acceleration_traces["observed_angular"][:, axis_index], f"{axis_label} observed", PLOTLY_MOTOR_COLORS[axis_index], dash="dot")

    figure.add_trace(
        go.Scatter(
            x=py_rel,
            y=px_rel,
            mode="markers",
            marker={"color": solver_t, "colorscale": "Viridis", "size": 4, "colorbar": {"title": "Time [s]"}},
            name="actual",
        ),
        row=5,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=masked_mode_series(sp_py_rel, recovery_mode_mask, False),
            y=masked_mode_series(sp_px_rel, recovery_mode_mask, False),
            mode="lines",
            line={"color": "red", "dash": "dash"},
            name="flight setpoint",
        ),
        row=5,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=masked_mode_series(sp_py_rel, recovery_mode_mask, True),
            y=masked_mode_series(sp_px_rel, recovery_mode_mask, True),
            mode="lines",
            line={"color": "magenta", "dash": "dot"},
            name="recovery setpoint",
        ),
        row=5,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=px4_y_rel, y=px4_x_rel, mode="lines", line={"color": "black", "dash": "dot"}, name="PX4 actual full offboard"),
        row=5,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=px4_sp_y_rel, y=px4_sp_x_rel, mode="lines", line={"color": "orange", "dash": "dash"}, name="PX4 setpoint full offboard"),
        row=5,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=[py_rel.iloc[0]], y=[px_rel.iloc[0]], mode="markers", marker={"color": "green", "size": 10, "symbol": "circle"}, name="start"),
        row=5,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=[py_rel.iloc[-1]], y=[px_rel.iloc[-1]], mode="markers", marker={"color": "red", "size": 10, "symbol": "triangle-up"}, name="end"),
        row=5,
        col=2,
    )

    add_plotly_series(figure, 6, 1, error_t, error_e_norm * 100.0, "||e|| [cm]", "purple", fill="tozeroy")
    add_plotly_series(figure, 6, 2, error_t, error_ex * 100.0, "error x [cm]", "blue", fill="tozeroy")
    add_plotly_series(figure, 7, 1, error_t, error_ey * 100.0, "error y [cm]", "orange", fill="tozeroy")
    add_plotly_series(figure, 7, 2, error_t, error_ez * 100.0, "error z [cm]", "green", fill="tozeroy")
    add_plotly_series(figure, 6, 2, error_t, np.zeros(len(error_t), dtype=float), "zero", "black", dash="dash")
    add_plotly_series(figure, 7, 1, error_t, np.zeros(len(error_t), dtype=float), "zero", "black", dash="dash")
    add_plotly_series(figure, 7, 2, error_t, np.zeros(len(error_t), dtype=float), "zero", "black", dash="dash")

    for column_name, label, color in [
        ("estimated_force_bx", "offline est Fx", "blue"),
        ("estimated_force_by", "offline est Fy", "orange"),
        ("estimated_force_bz", "offline est Fz", "green"),
        ("fixed_dist_force_bx", "fixed flight Fx", "blue"),
        ("fixed_dist_force_by", "fixed flight Fy", "orange"),
        ("fixed_dist_force_bz", "fixed flight Fz", "green"),
    ]:
        dash = "dash" if label.startswith("fixed") else None
        add_plotly_series(figure, 8, 1, offline_t, offline_df[column_name], label, color, dash=dash)

    for column_name, label, color in [
        ("estimated_torque_bx", "offline est Tx", "blue"),
        ("estimated_torque_by", "offline est Ty", "orange"),
        ("estimated_torque_bz", "offline est Tz", "green"),
        ("fixed_dist_torque_bx", "fixed flight Tx", "blue"),
        ("fixed_dist_torque_by", "fixed flight Ty", "orange"),
        ("fixed_dist_torque_bz", "fixed flight Tz", "green"),
    ]:
        dash = "dash" if label.startswith("fixed") else None
        add_plotly_series(figure, 8, 2, offline_t, offline_df[column_name], label, color, dash=dash)

    if optimizer_cost_series is not None:
        add_plotly_series(figure, 9, 1, solver_t, optimizer_cost_series["optimizer_cost_total"], "total cost", "black")
        add_plotly_series(figure, 9, 1, solver_t, optimizer_cost_series["optimizer_cost_stage_total"], "stage total", "blue")
        add_plotly_series(figure, 9, 1, solver_t, optimizer_cost_series["optimizer_cost_terminal_total"], "terminal total", "red")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_stage_pos"], "stage pos", "blue")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_stage_vel"], "stage vel", "orange")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_stage_att"], "stage att", "green")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_stage_omega"], "stage omega", "red")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_stage_u"], "stage u", "purple")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_stage_delta_u"], "stage delta_u", "brown")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_terminal_pos"], "terminal pos", "blue", dash="dash")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_terminal_vel"], "terminal vel", "orange", dash="dash")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_terminal_att"], "terminal att", "green", dash="dash")
        add_plotly_series(figure, 9, 2, solver_t, optimizer_cost_series["optimizer_cost_terminal_omega"], "terminal omega", "red", dash="dash")

    axis_colors = ["blue", "orange", "green"]
    axis_labels = ["x", "y", "z"]
    if instant_cost_series is not None:
        for axis_index, axis_label in enumerate(axis_labels):
            add_plotly_series(
                figure, 10, 1, solver_t,
                instant_cost_series["instant_stage_pos_xyz"][:, axis_index],
                f"stage pos {axis_label}", axis_colors[axis_index],
            )
            add_plotly_series(
                figure, 10, 1, solver_t,
                instant_cost_series["instant_stage_vel_xyz"][:, axis_index],
                f"stage vel {axis_label}", axis_colors[axis_index], dash="dash",
            )
            add_plotly_series(
                figure, 10, 1, solver_t,
                instant_cost_series["instant_stage_acc_xyz"][:, axis_index],
                f"stage acc {axis_label}", axis_colors[axis_index], dash="dot",
            )
            add_plotly_series(
                figure, 10, 1, solver_t,
                instant_cost_series["instant_stage_omega_xyz"][:, axis_index],
                f"stage omega {axis_label}", axis_colors[axis_index], dash="dashdot",
            )
            add_plotly_series(
                figure, 10, 1, solver_t,
                instant_cost_series["instant_terminal_pos_xyz"][:, axis_index],
                f"terminal pos {axis_label}", axis_colors[axis_index], dash="longdash",
            )

        att_labels = ["roll", "pitch", "yaw"]
        for quat_index, quat_label in enumerate(att_labels):
            add_plotly_series(
                figure, 10, 2, solver_t,
                instant_cost_series["instant_stage_att_xyz"][:, quat_index],
                f"stage att {quat_label}", PLOTLY_MOTOR_COLORS[quat_index],
            )
        for motor_index in range(NUM_MOTORS):
            add_plotly_series(
                figure, 10, 2, solver_t,
                instant_cost_series["instant_stage_u_motors"][:, motor_index],
                f"stage u{motor_index}", PLOTLY_MOTOR_COLORS[motor_index], dash="dash",
            )
            add_plotly_series(
                figure, 10, 2, solver_t,
                instant_cost_series["instant_stage_delta_u_motors"][:, motor_index],
                f"stage du{motor_index}", PLOTLY_MOTOR_COLORS[motor_index], dash="dot",
            )

    add_dominant_cost_traces_to_plotly(figure, solver_df, dominant_cost_series, 11, 12, 2, gap_center)

    add_plotly_series(figure, 13, 1, offline_t, offline_df["force_confidence"], "force confidence", "blue")
    add_plotly_series(figure, 13, 1, offline_t, offline_df["torque_confidence"], "torque confidence", "orange")
    add_plotly_series(figure, 13, 1, offline_t, offline_df["force_measurement_valid"], "force update", "blue", dash="dash")
    add_plotly_series(figure, 13, 1, offline_t, offline_df["torque_measurement_valid"], "torque update", "orange", dash="dash")
    add_plotly_series(figure, 13, 1, offline_t, offline_df["combined_valid"], "combined valid", "green", dash="dot")

    add_plotly_series(figure, 14, 1, px4_flight_t, px4_flight_df["px4_position_control_active"], "PX4 position control", "orange")
    add_plotly_series(figure, 14, 1, px4_flight_t, px4_flight_df["nmpc_direct_actuator_active"], "NMPC direct actuator", "blue")
    add_plotly_series(figure, 14, 1, solver_t, (~recovery_mode_mask).astype(float), NMPC_FLIGHT_MODE_LABEL, "green", dash="dot")
    add_plotly_series(figure, 14, 1, solver_t, recovery_mode_mask.astype(float), NMPC_RECOVERY_MODE_LABEL, "red", dash="dash")
    add_plotly_reference_lines(
        figure,
        reference_times,
        [
            (1, 1), (1, 2),
            (2, 1), (2, 2),
            (3, 1), (3, 2),
            (4, 1), (4, 2),
            (5, 1), (5, 2),
            (6, 1), (6, 2),
            (7, 1), (7, 2),
            (8, 1), (8, 2),
            (9, 1), (9, 2),
            (10, 1), (10, 2),
            (11, 1),
            (12, 1),
            (13, 1),
            (14, 1),
        ],
    )

    summary_text = (
        f"{nmpc_mode_summary}<br>"
        f"Metrics window: {metrics_t.iloc[0]:.2f}s to {metrics_t.iloc[-1]:.2f}s<br>"
        "Error is computed against the active NMPC setpoint; recovery mode sets x/y setpoints to current x/y.<br>"
        f"RMSE [cm]: x={rmse_x:.2f}, y={rmse_y:.2f}, z={rmse_z:.2f}, 3D={rmse_3d:.2f}<br>"
        f"Max |error| [cm]: x={max_x:.2f}, y={max_y:.2f}, z={max_z:.2f}, 3D={max_3d:.2f}<br>"
        f"Mean error [cm]: x={mean_x:.2f}, y={mean_y:.2f}, z={mean_z:.2f}, 3D={mean_3d:.2f}<br>"
        f"Std dev [cm]: x={std_x:.2f}, y={std_y:.2f}, z={std_z:.2f}, 3D={std_3d:.2f}<br>"
        f"95th pct |error| [cm]: x={p95_x:.2f}, y={p95_y:.2f}, z={p95_z:.2f}, 3D={p95_3d:.2f}<br><br>"
        f"{format_recommended_disturbance(offline_summary).replace(chr(10), '<br>')}"
    )
    figure.add_trace(
        go.Table(
            header={"values": ["Summary"], "align": "left"},
            cells={"values": [[summary_text]], "align": "left", "height": 260},
        ),
        row=14,
        col=2,
    )

    figure.update_layout(
        title=(
            f"HITL NMPC Diagnostics ({len(solver_df)} solves, {solver_t.iloc[-1]:.1f}s solver window)<br>"
            f"{nmpc_mode_summary}"
        ),
        height=5000,
        width=1600,
        hovermode="x unified",
    )
    figure.update_yaxes(type="log", range=[np.log10(COST_PLOT_YMIN), np.log10(COST_PLOT_YMAX)], title_text="cost", row=9, col=1)
    figure.update_yaxes(type="log", range=[np.log10(COST_PLOT_YMIN), np.log10(COST_PLOT_YMAX)], title_text="cost", row=9, col=2)
    figure.update_yaxes(type="log", range=[np.log10(COST_PLOT_YMIN), np.log10(COST_PLOT_YMAX)], title_text="cost", row=10, col=1)
    figure.update_yaxes(type="log", range=[np.log10(COST_PLOT_YMIN), np.log10(COST_PLOT_YMAX)], title_text="cost", row=10, col=2)
    figure.update_yaxes(type="log", title_text="weighted cost", row=11, col=1)
    figure.update_yaxes(title_text="m", row=11, col=2)
    figure.update_xaxes(title_text="y - gap_y [m]", row=5, col=2)
    figure.update_yaxes(title_text="x - gap_x [m]", row=5, col=2)
    figure.update_yaxes(range=[0.0, 100.0], title_text="% of instant cost", row=12, col=1)
    figure.update_yaxes(title_text="deg", row=12, col=2)
    figure.update_yaxes(title_text="score", row=13, col=1)
    figure.update_yaxes(title_text="active", row=14, col=1)
    figure.update_xaxes(title_text="Time [s]", row=12, col=1)
    figure.update_xaxes(title_text="Time [s]", row=12, col=2)
    figure.update_xaxes(title_text="Time [s]", row=13, col=1)
    figure.update_xaxes(title_text="Time [s]", row=14, col=1)
    figure.write_html(html_path, include_plotlyjs=True)


def load_inputs_from_raw_run_dir(run_dir, qualisys_match_max_position_error_m):
    """Build the normalized input dataframes fully in memory from a raw run directory.

    Replicates combine_nmpc_solver_logs.py, extract_px4_flight_log.py,
    extract_px4_motor_log.py and normalize_mocap_logs.py without writing any
    intermediate CSV files.
    """
    import normalize_mocap_logs as normalize_mocap_logs_module
    from extract_px4_flight_log import build_flight_log_dataframe
    from extract_px4_motor_log import build_motor_log_dataframe

    solver_csv_paths = sorted(
        path
        for path in run_dir.glob("nmpc_solver_data_*.csv")
        if path.name != "nmpc_solver_data_combined.csv"
        and not path.name.endswith("_normalized.csv")
        and not path.name.endswith("_offline_disturbance.csv")
    )
    if not solver_csv_paths:
        raise RuntimeError(f"No raw nmpc_solver_data_*.csv files found in {run_dir}")

    frames = []
    expected_columns = None
    for solver_csv_path in solver_csv_paths:
        solver_frame = pd.read_csv(solver_csv_path)
        if solver_frame.empty:
            raise RuntimeError(f"Input solver CSV is empty: {solver_csv_path}")
        if expected_columns is None:
            expected_columns = list(solver_frame.columns)
        elif list(solver_frame.columns) != expected_columns:
            raise RuntimeError(f"Input solver CSV header differs from the first file: {solver_csv_path}")
        solver_frame["source_solver_csv"] = solver_csv_path.name
        frames.append(solver_frame)
    solver_raw_df = pd.concat(frames, ignore_index=True)
    solver_raw_df.drop_duplicates(subset=["sample_timestamp_us", "seq"], keep="last", inplace=True)
    solver_raw_df.sort_values(["sample_timestamp_us", "seq"], inplace=True)
    solver_raw_df.reset_index(drop=True, inplace=True)

    ulog_path = run_dir / "px4_run.ulg"
    if not ulog_path.is_file():
        raise RuntimeError(f"Missing ULog file: {ulog_path}")
    px4_df = build_motor_log_dataframe(ulog_path)
    px4_flight_df = build_flight_log_dataframe(ulog_path)

    qualisys_path = run_dir / "qualisys_odom.csv"
    if not qualisys_path.is_file():
        raise RuntimeError(f"Missing Qualisys CSV: {qualisys_path}")
    qualisys_df = pd.read_csv(qualisys_path)

    battery_path = run_dir / "battery_status_log.csv"
    battery_raw_df = pd.read_csv(battery_path) if battery_path.is_file() else None

    active_start_us, active_end_us = normalize_mocap_logs_module.find_nmpc_active_window_us(px4_flight_df)
    matched_qualisys_us, matched_px4_us, matched_position_errors = normalize_mocap_logs_module.build_qualisys_px4_time_map(
        ulog_path,
        qualisys_df,
        qualisys_match_max_position_error_m,
    )
    solver_df = normalize_mocap_logs_module.add_time_s_and_trim(solver_raw_df, "sample_timestamp_us", active_start_us, active_end_us)
    px4_df = normalize_mocap_logs_module.add_time_s_and_trim(px4_df, "timestamp_us", active_start_us, active_end_us)
    px4_flight_df = normalize_mocap_logs_module.add_time_s_and_trim(px4_flight_df, "timestamp_us", active_start_us, active_end_us)
    if battery_raw_df is not None:
        battery_df = normalize_mocap_logs_module.normalize_battery_df(
            battery_raw_df,
            matched_qualisys_us,
            matched_px4_us,
            active_start_us,
            active_end_us,
        )
    else:
        battery_df = None

    print(
        "Built inputs in memory from raw logs in "
        f"{run_dir}: solver={len(solver_df)} rows, px4_motor={len(px4_df)} rows, "
        f"px4_flight={len(px4_flight_df)} rows, battery={'yes' if battery_df is not None else 'no'}"
    )
    return solver_df, px4_df, px4_flight_df, battery_df


def main():
    parser = argparse.ArgumentParser(description="Plot HITL NMPC diagnostics and replay disturbances offline from solver logs")
    parser.add_argument("solver_csv_path", nargs="?", default=None, help="CSV exported by nmpc_pipe_solver")
    parser.add_argument("px4_motor_csv_path", nargs="?", default=None, help="CSV exported by extract_px4_motor_log.py")
    parser.add_argument("px4_flight_csv_path", nargs="?", default=None, help="CSV exported by extract_px4_flight_log.py")
    parser.add_argument("pdf_path", help="Output PDF path")
    parser.add_argument("--battery-csv-path", dest="battery_csv_path", help="Normalized battery status CSV recorded during the flight; omit to plot a run without a battery status log")
    parser.add_argument("--raw-run-dir", dest="raw_run_dir", default=None, help="Run directory containing only raw logs; builds combined solver data, PX4 extractions and normalization in memory without writing intermediate CSVs")
    parser.add_argument("--qualisys-match-max-position-error-m", dest="qualisys_match_max_position_error_m", type=float, default=0.0001, help="Max position error for Qualisys/PX4 visual-odometry matching in --raw-run-dir mode")
    parser.add_argument("--nmpc-config-path", dest="nmpc_config_path", default=None, help="NMPC YAML config copied with this log; optional in --raw-run-dir mode when the run contains nmpc_controller_cfg.yaml")
    parser.add_argument("--html-path", dest="html_path", help="Output HTML path for interactive diagnostics")
    parser.add_argument("--offline-disturbance-csv", dest="offline_disturbance_csv_path", help="Output CSV path for offline disturbance replay")
    parser.add_argument("--offline-disturbance-pdf", dest="offline_disturbance_pdf_path", help="Output PDF path for offline disturbance replay")
    parser.add_argument("--offline-disturbance-html", dest="offline_disturbance_html_path", help="Output HTML path for offline disturbance replay")
    parser.add_argument("--dominant-cost-window-start-s", dest="dominant_cost_window_start_s", type=float, required=True, help="Start time for dominant NMPC cost page, relative to solver plot start")
    parser.add_argument("--dominant-cost-window-end-s", dest="dominant_cost_window_end_s", type=float, required=True, help="End time for dominant NMPC cost page, relative to solver plot start")
    parser.add_argument("--dominant-cost-top-count", dest="dominant_cost_top_count", type=int, required=True, help="Number of largest NMPC cost contributors to plot before aggregating others")
    parser.add_argument("--dominant-cost-distribution-log", dest="dominant_cost_distribution_log_path", required=True, help="Output text log path for sliced dominant NMPC cost distribution")
    parser.add_argument("--dominant-cost-distribution-slice-count", dest="dominant_cost_distribution_slice_count", type=int, required=True, help="Number of time slices written to the dominant NMPC cost distribution log")
    args = parser.parse_args()

    raw_run_source = None
    if args.raw_run_dir is not None:
        if args.solver_csv_path is not None or args.px4_motor_csv_path is not None or args.px4_flight_csv_path is not None:
            raise SystemExit("Pass either positional CSV paths or --raw-run-dir, not both")
        if args.battery_csv_path is not None:
            raise SystemExit("--battery-csv-path must not be combined with --raw-run-dir; the battery log is normalized in memory from the raw run directory")
        from run_archive import resolve_run_source

        raw_run_source = resolve_run_source(Path(args.raw_run_dir))
        solver_df, px4_df, px4_flight_df, battery_df = load_inputs_from_raw_run_dir(
            Path(raw_run_source),
            args.qualisys_match_max_position_error_m,
        )
    else:
        if args.solver_csv_path is None or args.px4_motor_csv_path is None or args.px4_flight_csv_path is None:
            raise SystemExit(
                "solver_csv_path, px4_motor_csv_path and px4_flight_csv_path are required unless --raw-run-dir is given"
            )
        solver_df = pd.read_csv(args.solver_csv_path)
        px4_df = pd.read_csv(args.px4_motor_csv_path)
        px4_flight_df = pd.read_csv(args.px4_flight_csv_path)
        battery_df = pd.read_csv(args.battery_csv_path) if args.battery_csv_path else None

    if args.nmpc_config_path is None or not Path(args.nmpc_config_path).is_file():
        config_fallback = None
        if raw_run_source is not None:
            candidate = Path(raw_run_source) / "nmpc_controller_cfg.yaml"
            if candidate.is_file():
                config_fallback = str(candidate)
        if config_fallback is None:
            raise SystemExit(
                "--nmpc-config-path is required and must exist (or the --raw-run-dir run must contain nmpc_controller_cfg.yaml)"
            )
        args.nmpc_config_path = config_fallback

    if solver_df.empty:
        raise RuntimeError(f"{args.solver_csv_path} is empty")
    if px4_df.empty:
        raise RuntimeError(f"{args.px4_motor_csv_path} is empty")
    if px4_flight_df.empty:
        raise RuntimeError(f"{args.px4_flight_csv_path} is empty")
    if battery_df is not None and battery_df.empty:
        raise RuntimeError(f"{args.battery_csv_path} is empty")
    if len(solver_df) < 2:
        raise RuntimeError(
            f"{args.solver_csv_path} contains fewer than 2 solver samples; "
            "the NMPC solver likely exited before producing a usable log"
        )

    require_columns(solver_df, REQUIRED_SOLVER_COLUMNS, args.solver_csv_path)

    angle_offset_deg, gap_center = load_angle_offset_and_gap_center(args.nmpc_config_path)
    if angle_offset_deg != 0.0:
        rotate_believed_setpoints_to_ground_truth(solver_df, angle_offset_deg, gap_center)
        print(
            f"Rotated believed setpoints to ground truth frame: "
            f"angle_offset_deg={angle_offset_deg} gap_center=[{gap_center[0]:.6f}, {gap_center[1]:.6f}]"
        )

    px4_required_columns = ["timestamp_us"]
    px4_required_columns.extend([f"measured_rpm{motor_index}" for motor_index in range(NUM_MOTORS)])
    require_columns(px4_df, px4_required_columns, args.px4_motor_csv_path)
    require_columns(
        px4_flight_df,
        [
            "timestamp_us",
            "px4_position_control_active",
            "nmpc_direct_actuator_active",
            "px4_x",
            "px4_y",
            "px4_z",
            "px4_vx",
            "px4_vy",
            "px4_vz",
            "px4_sp_x",
            "px4_sp_y",
            "px4_sp_z",
        ],
        args.px4_flight_csv_path,
    )
    require_columns(
        solver_df,
        [f"solver_u{motor_index}" for motor_index in range(NUM_MOTORS)],
        args.solver_csv_path,
    )
    if battery_df is not None:
        require_columns(
            battery_df,
            ["timestamp_us", "voltage_v", "per_cell_voltage_v", "percent_estimate"],
            args.battery_csv_path,
        )

    solver_df, px4_df, px4_flight_df, activity_start_us = trim_logs_to_activity_window(
        solver_df,
        px4_df,
        px4_flight_df,
    )
    print(f"Trimmed logs to first measured motor activity at PX4 timestamp_us={activity_start_us}")

    offline_df, offline_summary = estimate_offline_disturbance(solver_df)
    print(format_recommended_disturbance(offline_summary))

    if args.offline_disturbance_csv_path:
        offline_df.to_csv(args.offline_disturbance_csv_path, index=False)
        print(f"Saved {args.offline_disturbance_csv_path}")

    if args.offline_disturbance_pdf_path:
        generate_offline_disturbance_pdf(offline_df, args.offline_disturbance_pdf_path)
        print(f"Saved {args.offline_disturbance_pdf_path}")
    if args.offline_disturbance_html_path:
        generate_offline_disturbance_html(offline_df, args.offline_disturbance_html_path)
        print(f"Saved {args.offline_disturbance_html_path}")

    solver_timebase_us = solver_df["sample_timestamp_us"]
    px4_timebase_us = px4_df["timestamp_us"]
    px4_flight_timebase_us = px4_flight_df["timestamp_us"]
    plot_start_us = int(solver_timebase_us.iloc[0])
    plot_end_us = int(solver_timebase_us.iloc[-1])

    if plot_end_us <= plot_start_us:
        raise RuntimeError("Solver log does not have a positive timestamp span")

    px4_mask = (px4_timebase_us >= plot_start_us) & (px4_timebase_us <= plot_end_us)
    px4_flight_mask = (px4_flight_timebase_us >= plot_start_us) & (px4_flight_timebase_us <= plot_end_us)
    px4_df = px4_df[px4_mask].reset_index(drop=True)
    px4_flight_df = px4_flight_df[px4_flight_mask].reset_index(drop=True)

    if battery_df is not None:
        battery_timebase_us = battery_df["timestamp_us"]
        battery_mask = (battery_timebase_us >= plot_start_us) & (battery_timebase_us <= plot_end_us)
        battery_df = battery_df[battery_mask].reset_index(drop=True)

    if px4_df.empty:
        raise RuntimeError("No PX4 motor samples overlap the solver time window")
    if px4_flight_df.empty:
        raise RuntimeError("No PX4 flight samples overlap the solver time window")
    if battery_df is not None and battery_df.empty:
        raise RuntimeError("No battery status samples overlap the solver time window")

    solver_t = (solver_df["sample_timestamp_us"] - plot_start_us) / 1e6
    offline_t = (offline_df["sample_timestamp_us"] - plot_start_us) / 1e6
    px4_t = (px4_df["timestamp_us"] - plot_start_us) / 1e6
    px4_flight_t = (px4_flight_df["timestamp_us"] - plot_start_us) / 1e6
    battery_t = (battery_df["timestamp_us"] - plot_start_us) / 1e6 if battery_df is not None else None
    nmpc_mode_summary = summarize_nmpc_modes(solver_df, solver_t)
    recovery_mode_mask = get_recovery_mode_mask(solver_df)
    reference_times = get_nmpc_setpoint_change_times(solver_df, solver_t)
    print(f"NMPC setpoint changes marked with vertical lines: {len(reference_times)}")
    solve_time_ms = solver_df["solve_us"] / 1e3
    solver_input_dt_ms = solver_df["sample_timestamp_us"].diff() / 1e3
    solver_posvel_dt_ms = solver_df["position_velocity_timestamp_us"].diff() / 1e3
    solver_att_dt_ms = solver_df["attitude_timestamp_us"].diff() / 1e3
    solver_ang_dt_ms = solver_df["angular_velocity_timestamp_us"].diff() / 1e3
    optimizer_cost_series = get_optimizer_cost_plot_series(solver_df)
    instant_cost_series = get_instant_cost_plot_series(solver_df, args.nmpc_config_path)
    dominant_cost_series = build_dominant_cost_series(
        solver_t,
        instant_cost_series,
        args.dominant_cost_window_start_s,
        args.dominant_cost_window_end_s,
        args.dominant_cost_top_count,
    )
    write_dominant_cost_distribution_log(
        args.dominant_cost_distribution_log_path,
        solver_df,
        dominant_cost_series,
        args.dominant_cost_distribution_slice_count,
        gap_center,
    )
    print(f"Saved {args.dominant_cost_distribution_log_path}")
    solve_time_stats = summarize_series(solve_time_ms)
    solver_input_dt_stats = summarize_series(solver_input_dt_ms.iloc[1:])
    solver_posvel_dt_stats = summarize_series(solver_posvel_dt_ms.iloc[1:])
    solver_att_dt_stats = summarize_series(solver_att_dt_ms.iloc[1:])
    solver_ang_dt_stats = summarize_series(solver_ang_dt_ms.iloc[1:])

    solver_duration = (plot_end_us - plot_start_us) / 1e6
    px4_flight_start_s = float(px4_flight_t.iloc[0])
    px4_flight_end_s = float(px4_flight_t.iloc[-1])
    print(
        f"Solver time window: {solver_duration:.2f}s; "
        f"PX4 offboard overlap: {px4_flight_start_s:.2f}s to {px4_flight_end_s:.2f}s"
    )
    print(nmpc_mode_summary)
    print(
        "Timing stats:"
        f" solve_ms(mean/p99/max)={solve_time_stats['mean']:.3f}/{solve_time_stats['p99']:.3f}/{solve_time_stats['max']:.3f},"
        f" sample_dt_ms={solver_input_dt_stats['mean']:.3f}/{solver_input_dt_stats['p99']:.3f}/{solver_input_dt_stats['max']:.3f},"
        f" posvel_dt_ms={solver_posvel_dt_stats['mean']:.3f}/{solver_posvel_dt_stats['p99']:.3f}/{solver_posvel_dt_stats['max']:.3f},"
        f" att_dt_ms={solver_att_dt_stats['mean']:.3f}/{solver_att_dt_stats['p99']:.3f}/{solver_att_dt_stats['max']:.3f},"
        f" ang_dt_ms={solver_ang_dt_stats['mean']:.3f}/{solver_ang_dt_stats['p99']:.3f}/{solver_ang_dt_stats['max']:.3f}"
    )

    acceleration_traces = compute_acceleration_traces(solver_df)

    solver_motor_df = solver_df[
        ["sample_timestamp_us"]
        + [f"command_rps{motor_index}" for motor_index in range(NUM_MOTORS)]
        + [f"estimated_rps{motor_index}" for motor_index in range(NUM_MOTORS)]
    ].sort_values("sample_timestamp_us")
    px4_motor_df = px4_df[
        ["timestamp_us"] + [f"measured_rpm{motor_index}" for motor_index in range(NUM_MOTORS)]
    ].sort_values("timestamp_us")
    aligned_motor_df = pd.merge_asof(
        solver_motor_df,
        px4_motor_df,
        left_on="sample_timestamp_us",
        right_on="timestamp_us",
        direction="nearest",
        tolerance=5000,
    )

    for motor_index in range(NUM_MOTORS):
        alignment_columns = [
            f"command_rps{motor_index}",
            f"estimated_rps{motor_index}",
            f"measured_rpm{motor_index}",
        ]
        aligned_samples = aligned_motor_df.dropna(subset=alignment_columns)
        if aligned_samples.empty:
            raise RuntimeError(f"Motor {motor_index}: no aligned solver/PX4 samples within 5 ms")

        command_rpm = aligned_samples[f"command_rps{motor_index}"] * 60.0
        estimated_rpm = aligned_samples[f"estimated_rps{motor_index}"] * 60.0
        measured_rpm = aligned_samples[f"measured_rpm{motor_index}"]
        print(
            f"Motor {motor_index}: "
            f"|command-measured| mean={np.mean(np.abs(command_rpm - measured_rpm)):.1f} rpm, "
            f"|estimated-measured| mean={np.mean(np.abs(estimated_rpm - measured_rpm)):.1f} rpm "
            f"over {len(aligned_samples)} aligned samples"
        )

    diagnostics_title = (
        f"HITL NMPC Diagnostics ({len(solver_df)} solves, {solver_t.iloc[-1]:.1f}s solver window)\n"
        f"{nmpc_mode_summary}"
    )
    gap_x = float(gap_center[0])
    gap_y = float(gap_center[1])
    px_rel = solver_df["px"] - gap_x
    py_rel = solver_df["py"] - gap_y
    sp_px_rel = solver_df["sp_px"] - gap_x
    sp_py_rel = solver_df["sp_py"] - gap_y
    px4_x_rel = px4_flight_df["px4_y"] - gap_x
    px4_y_rel = px4_flight_df["px4_x"] - gap_y
    px4_sp_x_rel = px4_flight_df["px4_sp_y"] - gap_x
    px4_sp_y_rel = px4_flight_df["px4_sp_x"] - gap_y

    with PdfPages(args.pdf_path) as pdf:
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        add_mode_background_to_axis(
            axes[0],
            px4_flight_t,
            px4_flight_df["px4_position_control_active"].to_numpy(dtype=float),
            px4_flight_df["nmpc_direct_actuator_active"].to_numpy(dtype=float),
            solver_t,
            recovery_mode_mask.astype(float),
        )
        axes[1].plot(px4_flight_t, px4_x_rel, linewidth=0.8, label="x")
        axes[1].plot(px4_flight_t, px4_y_rel, linewidth=0.8, label="y")
        axes[1].plot(px4_flight_t, -px4_flight_df["px4_z"], linewidth=0.8, label="z")
        axes[1].plot(px4_flight_t, px4_sp_x_rel, linewidth=0.8, linestyle="--", label="sp_x")
        axes[1].plot(px4_flight_t, px4_sp_y_rel, linewidth=0.8, linestyle="--", label="sp_y")
        axes[1].plot(px4_flight_t, -px4_flight_df["px4_sp_z"], linewidth=0.8, linestyle="--", label="sp_z")
        axes[1].set_ylabel("m")
        axes[1].set_title("PX4 full offboard position relative to gap center + setpoint")
        axes[1].legend(fontsize=8, ncol=3)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(px4_flight_t, px4_flight_df["px4_vy"], linewidth=0.8, label="vx")
        axes[2].plot(px4_flight_t, px4_flight_df["px4_vx"], linewidth=0.8, label="vy")
        axes[2].plot(px4_flight_t, -px4_flight_df["px4_vz"], linewidth=0.8, label="vz")
        axes[2].set_ylabel("m/s")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_title("PX4 full offboard velocity")
        axes[2].legend(fontsize=8, ncol=3)
        axes[2].grid(True, alpha=0.3)

        for motor_index in range(NUM_MOTORS):
            axes[3].plot(
                px4_t,
                px4_df[f"measured_rpm{motor_index}"] / 60.0,
                linewidth=0.8,
                color=MOTOR_COLORS[motor_index],
                label=f"measured_rps{motor_index}",
            )
            if f"requested_rpm{motor_index}" in px4_df.columns:
                axes[3].plot(
                    px4_t,
                    px4_df[f"requested_rpm{motor_index}"] / 60.0,
                    linewidth=0.6,
                    linestyle="--",
                    color=MOTOR_COLORS[motor_index],
                    alpha=0.8,
                    label=f"requested_rps{motor_index}",
                )
        axes[3].set_ylabel("RPS")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_title("PX4 full offboard motor RPS")
        axes[3].legend(fontsize=8, ncol=4)
        axes[3].grid(True, alpha=0.3)
        fig.suptitle(f"PX4 Full Offboard Trajectory and Control Mode\n{nmpc_mode_summary}", fontsize=13, y=0.995)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(7, 1, figsize=(12, 22), sharex=False)

        axes[0].plot(solver_t, solve_time_ms, linewidth=0.8, label="solve time")
        axes[0].plot(solver_t.values[1:], solver_input_dt_ms.iloc[1:], linewidth=0.8, alpha=0.8, label="sample dt")
        axes[0].plot(solver_t.values[1:], solver_posvel_dt_ms.iloc[1:], linewidth=0.8, alpha=0.8, label="position/velocity dt")
        axes[0].plot(solver_t.values[1:], solver_att_dt_ms.iloc[1:], linewidth=0.8, alpha=0.8, label="attitude dt")
        axes[0].plot(solver_t.values[1:], solver_ang_dt_ms.iloc[1:], linewidth=0.8, alpha=0.8, label="angular velocity dt")
        axes[0].axhline(10.0, color="r", linestyle="--", linewidth=0.5, label="10ms (100Hz)")
        axes[0].set_ylabel("ms")
        axes[0].set_title("Solver timing and per-signal source timestamps")
        axes[0].legend(fontsize=8, ncol=3)
        axes[0].grid(True, alpha=0.3)

        for motor_index in range(NUM_MOTORS):
            axes[1].plot(solver_t, solver_df[f"command_rps{motor_index}"], linewidth=0.8, color=MOTOR_COLORS[motor_index], label=f"command_rps{motor_index}")
            axes[1].plot(px4_t, px4_df[f"measured_rpm{motor_index}"] / 60.0, linewidth=0.8, color=MOTOR_COLORS[motor_index], linestyle="--", alpha=0.8, label=f"real_rps{motor_index}")
        axes[1].set_ylabel("RPS")
        axes[1].set_title("Real ESC RPS vs NMPC commanded RPS")
        axes[1].legend(fontsize=8, ncol=4)
        axes[1].grid(True, alpha=0.3)

        for motor_index in range(NUM_MOTORS):
            axes[2].plot(solver_t, solver_df[f"estimated_rps{motor_index}"], linewidth=0.8, color=MOTOR_COLORS[motor_index], label=f"estimated_rps{motor_index}")
            axes[2].plot(px4_t, px4_df[f"measured_rpm{motor_index}"] / 60.0, linewidth=0.8, color=MOTOR_COLORS[motor_index], linestyle="--", alpha=0.8, label=f"real_rps{motor_index}")
        axes[2].set_ylabel("RPS")
        axes[2].set_title("Real ESC RPS vs NMPC estimated RPS")
        axes[2].legend(fontsize=8, ncol=4)
        axes[2].grid(True, alpha=0.3)

        for motor_index in range(NUM_MOTORS):
            axes[3].plot(solver_t, solver_df[f"solver_u{motor_index}"], linewidth=0.8, linestyle="--", alpha=0.8, label=f"solver_u{motor_index}")
        axes[3].set_ylabel("N")
        axes[3].set_title("Solver motor forces")
        axes[3].legend(fontsize=8, ncol=2)
        axes[3].grid(True, alpha=0.3)

        for values, label in [(px_rel, "x"), (py_rel, "y"), (solver_df["pz"], "z")]:
            axes[4].plot(solver_t, values, linewidth=0.8, label=label)
        for values, label in [(sp_px_rel, "sp_x"), (sp_py_rel, "sp_y"), (solver_df["sp_pz"], "sp_z")]:
            axes[4].plot(
                solver_t,
                masked_mode_series(values, recovery_mode_mask, False),
                linewidth=0.8,
                linestyle="--",
                alpha=0.75,
                label=f"{label} flight",
            )
            axes[4].plot(
                solver_t,
                masked_mode_series(values, recovery_mode_mask, True),
                linewidth=0.8,
                linestyle=":",
                alpha=0.95,
                label=f"{label} recovery",
            )
        axes[4].plot(px4_flight_t, px4_x_rel, linewidth=0.7, linestyle=":", alpha=0.8, label="PX4 x full")
        axes[4].plot(px4_flight_t, px4_y_rel, linewidth=0.7, linestyle=":", alpha=0.8, label="PX4 y full")
        axes[4].plot(px4_flight_t, -px4_flight_df["px4_z"], linewidth=0.7, linestyle=":", alpha=0.8, label="PX4 z full")
        axes[4].set_ylabel("m")
        axes[4].set_title("Position relative to gap center + active NMPC setpoint")
        axes[4].legend(fontsize=8, ncol=2)
        axes[4].grid(True, alpha=0.3)

        for column_name, label in [("vx", "vx"), ("vy", "vy"), ("vz", "vz")]:
            axes[5].plot(solver_t, solver_df[column_name], linewidth=0.8, label=label)
        axes[5].set_ylabel("m/s")
        axes[5].set_title("Velocity")
        axes[5].legend(fontsize=8)
        axes[5].grid(True, alpha=0.3)

        roll, pitch, yaw = quaternion_to_euler_deg(
            solver_df["qw"].values,
            solver_df["qx"].values,
            solver_df["qy"].values,
            solver_df["qz"].values,
        )
        sp_roll, sp_pitch, sp_yaw = quaternion_to_euler_deg(
            solver_df["sp_qw"].values,
            solver_df["sp_qx"].values,
            solver_df["sp_qy"].values,
            solver_df["sp_qz"].values,
        )
        axes[6].plot(solver_t, roll, linewidth=0.8, label="roll")
        axes[6].plot(solver_t, pitch, linewidth=0.8, label="pitch")
        axes[6].plot(solver_t, yaw, linewidth=0.8, label="yaw")
        axes[6].plot(solver_t, sp_roll, linewidth=0.8, linestyle="--", alpha=0.6, label="sp_roll")
        axes[6].plot(solver_t, sp_pitch, linewidth=0.8, linestyle="--", alpha=0.6, label="sp_pitch")
        axes[6].plot(solver_t, sp_yaw, linewidth=0.8, linestyle="--", alpha=0.6, label="sp_yaw")
        axes[6].set_ylabel("deg")
        axes[6].set_title("Orientation (Euler) + setpoint")
        axes[6].legend(fontsize=8, ncol=2)
        axes[6].set_xlabel("Time [s]")
        axes[6].grid(True, alpha=0.3)

        add_reference_lines_to_axes(axes, reference_times)
        fig.suptitle(diagnostics_title, fontsize=14, y=0.995)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
        axis_labels = ["x", "y", "z"]
        command_linear = acceleration_traces["command_linear"]
        expected_linear = acceleration_traces["expected_linear"]
        observed_linear = acceleration_traces["observed_linear"]
        command_angular = acceleration_traces["command_angular"]
        expected_angular = acceleration_traces["expected_angular"]
        observed_angular = acceleration_traces["observed_angular"]

        for axis_index, axis_label in enumerate(axis_labels):
            ax = axes[axis_index, 0]
            ax.plot(solver_t, command_linear[:, axis_index], linewidth=0.8, label="commanded")
            ax.plot(solver_t, expected_linear[:, axis_index], linewidth=0.8, linestyle="--", label="expected from estimated rpm")
            ax.plot(solver_t, observed_linear[:, axis_index], linewidth=0.8, alpha=0.8, label="observed")
            ax.set_ylabel("m/s^2")
            ax.set_title(f"Linear acceleration body {axis_label}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            ax = axes[axis_index, 1]
            ax.plot(solver_t, command_angular[:, axis_index], linewidth=0.8, label="commanded")
            ax.plot(solver_t, expected_angular[:, axis_index], linewidth=0.8, linestyle="--", label="expected from estimated rpm")
            ax.plot(solver_t, observed_angular[:, axis_index], linewidth=0.8, alpha=0.8, label="observed")
            ax.set_ylabel("rad/s^2")
            ax.set_title(f"Angular acceleration body {axis_label}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[-1, 0].set_xlabel("Time [s]")
        axes[-1, 1].set_xlabel("Time [s]")
        add_reference_lines_to_axes(axes, reference_times)
        fig.suptitle("Acceleration Tracking", fontsize=13, y=0.995)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        ex = solver_df["px"].values - solver_df["sp_px"].values
        ey = solver_df["py"].values - solver_df["sp_py"].values
        ez = solver_df["pz"].values - solver_df["sp_pz"].values
        e_norm = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
        fresh_position_velocity_mask = get_fresh_position_velocity_mask(solver_df)
        error_t = solver_t[fresh_position_velocity_mask]
        error_ex = ex[fresh_position_velocity_mask]
        error_ey = ey[fresh_position_velocity_mask]
        error_ez = ez[fresh_position_velocity_mask]
        error_e_norm = e_norm[fresh_position_velocity_mask]

        fig = plt.figure(figsize=(14, 20))
        gs = fig.add_gridspec(
            5, 2,
            height_ratios=[4, 1.2, 1.2, 1.2, 1.6],
            hspace=0.45, wspace=0.35,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        sc = ax_top.scatter(
            py_rel, px_rel,
            c=solver_t, cmap="viridis", s=3, zorder=3, label="actual",
        )
        plt.colorbar(sc, ax=ax_top, label="Time [s]", shrink=0.85)
        ax_top.plot(
            masked_mode_series(sp_py_rel, recovery_mode_mask, False),
            masked_mode_series(sp_px_rel, recovery_mode_mask, False),
            linestyle="--", color="red", linewidth=1.0, alpha=0.75, label="flight setpoint",
        )
        ax_top.plot(
            masked_mode_series(sp_py_rel, recovery_mode_mask, True),
            masked_mode_series(sp_px_rel, recovery_mode_mask, True),
            linestyle=":", color="magenta", linewidth=1.0, alpha=0.9, label="recovery setpoint",
        )
        ax_top.plot(
            px4_y_rel, px4_x_rel,
            linestyle=":", color="black", linewidth=1.0, alpha=0.75, label="PX4 actual full offboard",
        )
        ax_top.plot(
            px4_sp_y_rel, px4_sp_x_rel,
            linestyle="--", color="orange", linewidth=1.0, alpha=0.75, label="PX4 setpoint full offboard",
        )
        ax_top.plot(
            py_rel.iloc[0], px_rel.iloc[0],
            "go", markersize=8, zorder=4, label="start",
        )
        ax_top.plot(
            py_rel.iloc[-1], px_rel.iloc[-1],
            "r^", markersize=8, zorder=4, label="end",
        )
        ax_top.set_xlabel("y - gap_y [m]")
        ax_top.set_ylabel("x - gap_x [m]")
        ax_top.set_title("Top view")
        ax_top.set_aspect("equal")
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_enorm = fig.add_subplot(gs[0, 1])
        ax_enorm.plot(error_t, error_e_norm * 100, linewidth=0.8, color="purple")
        ax_enorm.fill_between(error_t, 0, error_e_norm * 100, alpha=0.18, color="purple")
        ax_enorm.set_xlabel("Time [s]")
        ax_enorm.set_ylabel("||e|| [cm]")
        ax_enorm.set_title("Position error magnitude (fresh position/velocity samples only)")
        ax_enorm.grid(True, alpha=0.3)

        axis_specs = [
            (error_ex, "x", "tab:blue", gs[1, :]),
            (error_ey, "y", "tab:orange", gs[2, :]),
            (error_ez, "z", "tab:green", gs[3, :]),
        ]
        position_error_axes = []
        for i, (err, label, color, cell) in enumerate(axis_specs):
            ax_e = fig.add_subplot(cell)
            position_error_axes.append(ax_e)
            ax_e.plot(error_t, err * 100, linewidth=0.8, color=color)
            ax_e.fill_between(error_t, 0, err * 100, alpha=0.15, color=color)
            ax_e.axhline(0, color="k", linestyle="--", linewidth=0.5)
            ax_e.set_ylabel("error [cm]")
            ax_e.set_title(f"Position error - {label} (fresh position/velocity samples only)")
            ax_e.grid(True, alpha=0.3)
            if i == len(axis_specs) - 1:
                ax_e.set_xlabel("Time [s]")

        ax_txt = fig.add_subplot(gs[4, :])
        ax_txt.axis("off")

        metrics_mask = fresh_position_velocity_mask & (solver_t >= solver_t.iloc[-1] - 20.0).to_numpy()
        metrics_t = solver_t[metrics_mask]
        metrics_ex = ex[metrics_mask]
        metrics_ey = ey[metrics_mask]
        metrics_ez = ez[metrics_mask]
        metrics_e_norm = np.sqrt(metrics_ex ** 2 + metrics_ey ** 2 + metrics_ez ** 2)

        rmse_x = np.sqrt(np.mean(metrics_ex ** 2)) * 100
        rmse_y = np.sqrt(np.mean(metrics_ey ** 2)) * 100
        rmse_z = np.sqrt(np.mean(metrics_ez ** 2)) * 100
        rmse_3d = np.sqrt(np.mean(metrics_e_norm ** 2)) * 100
        max_x = np.max(np.abs(metrics_ex)) * 100
        max_y = np.max(np.abs(metrics_ey)) * 100
        max_z = np.max(np.abs(metrics_ez)) * 100
        max_3d = np.max(metrics_e_norm) * 100
        mean_x = np.mean(metrics_ex) * 100
        mean_y = np.mean(metrics_ey) * 100
        mean_z = np.mean(metrics_ez) * 100
        mean_3d = np.mean(metrics_e_norm) * 100
        std_x = np.std(metrics_ex) * 100
        std_y = np.std(metrics_ey) * 100
        std_z = np.std(metrics_ez) * 100
        std_3d = np.std(metrics_e_norm) * 100
        p95_x = np.percentile(np.abs(metrics_ex), 95) * 100
        p95_y = np.percentile(np.abs(metrics_ey), 95) * 100
        p95_z = np.percentile(np.abs(metrics_ez), 95) * 100
        p95_3d = np.percentile(metrics_e_norm, 95) * 100

        W0, W = 26, 11
        lines = [
            f"Metrics window: {metrics_t.iloc[0]:.2f}s to {metrics_t.iloc[-1]:.2f}s",
            "Error uses only fresh position/velocity samples.",
            f"Fresh position/velocity samples: {np.count_nonzero(fresh_position_velocity_mask)} / {len(fresh_position_velocity_mask)}",
            "Error is computed against the active NMPC setpoint; recovery mode sets x/y setpoints to current x/y.",
            "",
            f"{'Metric':<{W0}}{'x':>{W}}{'y':>{W}}{'z':>{W}}{'3D':>{W}}",
            "─" * (W0 + 4 * W),
            f"{'RMSE [cm]':<{W0}}{rmse_x:>{W}.2f}{rmse_y:>{W}.2f}{rmse_z:>{W}.2f}{rmse_3d:>{W}.2f}",
            f"{'Max |error| [cm]':<{W0}}{max_x:>{W}.2f}{max_y:>{W}.2f}{max_z:>{W}.2f}{max_3d:>{W}.2f}",
            f"{'Mean error [cm]':<{W0}}{mean_x:>{W}.2f}{mean_y:>{W}.2f}{mean_z:>{W}.2f}{mean_3d:>{W}.2f}",
            f"{'Std dev [cm]':<{W0}}{std_x:>{W}.2f}{std_y:>{W}.2f}{std_z:>{W}.2f}{std_3d:>{W}.2f}",
            f"{'95th pct |error| [cm]':<{W0}}{p95_x:>{W}.2f}{p95_y:>{W}.2f}{p95_z:>{W}.2f}{p95_3d:>{W}.2f}",
        ]
        ax_txt.text(
            0.02, 0.95, "\n".join(lines),
            transform=ax_txt.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6),
        )

        add_reference_line_to_axis(ax_enorm, reference_times)
        add_reference_lines_to_axes(position_error_axes, reference_times)
        fig.suptitle(f"Trajectory Analysis\n{nmpc_mode_summary}", fontsize=13, y=0.998)
        fig.subplots_adjust(top=0.93)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        for column_name, label in [
            ("estimated_force_bx", "offline est Fx"),
            ("estimated_force_by", "offline est Fy"),
            ("estimated_force_bz", "offline est Fz"),
            ("fixed_dist_force_bx", "fixed flight Fx"),
            ("fixed_dist_force_by", "fixed flight Fy"),
            ("fixed_dist_force_bz", "fixed flight Fz"),
        ]:
            linestyle = "--" if label.startswith("fixed") else "-"
            axes[0].plot(offline_t, offline_df[column_name], linewidth=0.8, linestyle=linestyle, label=label)
        axes[0].set_ylabel("N")
        axes[0].set_title("Offline disturbance force estimate vs fixed in-flight value")
        axes[0].legend(fontsize=8, ncol=3)
        axes[0].grid(True, alpha=0.3)

        for column_name, label in [
            ("estimated_torque_bx", "offline est Tx"),
            ("estimated_torque_by", "offline est Ty"),
            ("estimated_torque_bz", "offline est Tz"),
            ("fixed_dist_torque_bx", "fixed flight Tx"),
            ("fixed_dist_torque_by", "fixed flight Ty"),
            ("fixed_dist_torque_bz", "fixed flight Tz"),
        ]:
            linestyle = "--" if label.startswith("fixed") else "-"
            axes[1].plot(offline_t, offline_df[column_name], linewidth=0.8, linestyle=linestyle, label=label)
        axes[1].set_ylabel("Nm")
        axes[1].set_title("Offline disturbance torque estimate vs fixed in-flight value")
        axes[1].legend(fontsize=8, ncol=3)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(offline_t, offline_df["force_confidence"], linewidth=0.9, label="force confidence")
        axes[2].plot(offline_t, offline_df["torque_confidence"], linewidth=0.9, label="torque confidence")
        axes[2].plot(offline_t, offline_df["force_measurement_valid"], linewidth=0.8, linestyle="--", label="force update")
        axes[2].plot(offline_t, offline_df["torque_measurement_valid"], linewidth=0.8, linestyle="--", label="torque update")
        axes[2].plot(offline_t, offline_df["combined_valid"], linewidth=0.8, linestyle=":", label="combined valid")
        axes[2].set_ylabel("score")
        axes[2].set_title("Offline replay confidence and update availability")
        axes[2].legend(fontsize=8, ncol=3)
        axes[2].grid(True, alpha=0.3)

        axes[3].axis("off")
        axes[3].text(
            0.02,
            0.95,
            format_recommended_disturbance(offline_summary),
            transform=axes[3].transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6),
        )
        add_reference_lines_to_axes(axes[:3], reference_times)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if optimizer_cost_series is not None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            axes[0].plot(solver_t, optimizer_cost_series["optimizer_cost_total"], linewidth=0.9, color="black", label="total cost")
            axes[0].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_total"], linewidth=0.8, color="tab:blue", label="stage total")
            axes[0].plot(solver_t, optimizer_cost_series["optimizer_cost_terminal_total"], linewidth=0.8, color="tab:red", label="terminal total")
            axes[0].set_yscale("log")
            axes[0].set_ylim(COST_PLOT_YMIN, COST_PLOT_YMAX)
            axes[0].set_ylabel("cost")
            axes[0].set_title("Optimizer objective totals")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_pos"], linewidth=0.8, color="tab:blue", label="stage pos")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_vel"], linewidth=0.8, color="tab:orange", label="stage vel")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_att"], linewidth=0.8, color="tab:green", label="stage att")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_omega"], linewidth=0.8, color="tab:red", label="stage omega")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_u"], linewidth=0.8, color="tab:purple", label="stage u")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_stage_delta_u"], linewidth=0.8, color="tab:brown", label="stage delta_u")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_terminal_pos"], linewidth=0.8, color="tab:blue", linestyle="--", label="terminal pos")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_terminal_vel"], linewidth=0.8, color="tab:orange", linestyle="--", label="terminal vel")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_terminal_att"], linewidth=0.8, color="tab:green", linestyle="--", label="terminal att")
            axes[1].plot(solver_t, optimizer_cost_series["optimizer_cost_terminal_omega"], linewidth=0.8, color="tab:red", linestyle="--", label="terminal omega")
            axes[1].set_yscale("log")
            axes[1].set_ylim(COST_PLOT_YMIN, COST_PLOT_YMAX)
            axes[1].set_ylabel("cost")
            axes[1].set_xlabel("Time [s]")
            axes[1].set_title("Optimizer objective term breakdown")
            axes[1].legend(fontsize=8, ncol=2)
            axes[1].grid(True, alpha=0.3)

            add_reference_lines_to_axes(axes, reference_times)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axis_labels = ["x", "y", "z"]
        axis_colors = ["tab:blue", "tab:orange", "tab:green"]

        for axis_index, axis_label in enumerate(axis_labels):
            axes[0, 0].plot(
                solver_t,
                instant_cost_series["instant_stage_pos_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                label=f"stage {axis_label}",
            )
            axes[0, 0].plot(
                solver_t,
                instant_cost_series["instant_terminal_pos_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                linestyle="--",
                label=f"terminal {axis_label}",
            )
            axes[0, 1].plot(
                solver_t,
                instant_cost_series["instant_stage_vel_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                label=f"vel stage {axis_label}",
            )
            axes[0, 1].plot(
                solver_t,
                instant_cost_series["instant_stage_acc_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                linestyle=":",
                label=f"acc stage {axis_label}",
            )
            axes[0, 1].plot(
                solver_t,
                instant_cost_series["instant_terminal_vel_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                linestyle="--",
                label=f"vel terminal {axis_label}",
            )
            axes[1, 0].plot(
                solver_t,
                instant_cost_series["instant_stage_omega_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                label=f"stage {axis_label}",
            )
            axes[1, 0].plot(
                solver_t,
                instant_cost_series["instant_terminal_omega_xyz"][:, axis_index],
                linewidth=0.8,
                color=axis_colors[axis_index],
                linestyle="--",
                label=f"terminal {axis_label}",
            )

        att_labels = ["roll", "pitch", "yaw"]
        for quat_index, quat_label in enumerate(att_labels):
            axes[1, 1].plot(
                solver_t,
                instant_cost_series["instant_stage_att_xyz"][:, quat_index],
                linewidth=0.8,
                color=MOTOR_COLORS[quat_index],
                label=f"att {quat_label}",
            )
        for motor_index in range(NUM_MOTORS):
            axes[1, 1].plot(
                solver_t,
                instant_cost_series["instant_stage_u_motors"][:, motor_index],
                linewidth=0.8,
                color=MOTOR_COLORS[motor_index],
                linestyle="--",
                label=f"u{motor_index}",
            )
            axes[1, 1].plot(
                solver_t,
                instant_cost_series["instant_stage_delta_u_motors"][:, motor_index],
                linewidth=0.8,
                color=MOTOR_COLORS[motor_index],
                linestyle=":",
                label=f"du{motor_index}",
            )

        axes[0, 0].set_title("Instant position cost by axis")
        axes[0, 1].set_title("Instant velocity / acceleration cost by axis")
        axes[1, 0].set_title("Instant angular-rate cost by axis")
        axes[1, 1].set_title("Instant attitude / control component cost")
        for row_axes in axes:
            for ax in row_axes:
                ax.set_yscale("log")
                ax.set_ylim(COST_PLOT_YMIN, COST_PLOT_YMAX)
                ax.set_ylabel("cost")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, ncol=2)
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 1].set_xlabel("Time [s]")
        add_reference_lines_to_axes(axes, reference_times)
        fig.suptitle(f"Instant NMPC Cost Reconstruction from Logged Sample\n{nmpc_mode_summary}", fontsize=13, y=0.995)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        add_dominant_cost_pdf_page(pdf, solver_df, dominant_cost_series, nmpc_mode_summary, gap_center)

        if battery_df is not None:
            fig, ax_voltage = plt.subplots(figsize=(12, 5))
            ax_voltage.plot(battery_t, battery_df["voltage_v"], linewidth=1.0, color="tab:blue", label="pack voltage")
            ax_voltage.plot(battery_t, battery_df["per_cell_voltage_v"], linewidth=1.0, color="tab:cyan", linestyle="--", label="per-cell voltage")
            ax_voltage.set_xlabel("Time [s]")
            ax_voltage.set_ylabel("Voltage [V]")
            ax_voltage.grid(True, alpha=0.3)
            ax_percent = ax_voltage.twinx()
            ax_percent.plot(battery_t, battery_df["percent_estimate"], linewidth=1.0, color="tab:orange", label="percent estimate")
            ax_percent.set_ylabel("Estimated charge [%]")
            ax_percent.set_ylim(0.0, 100.0)
            voltage_handles, voltage_labels = ax_voltage.get_legend_handles_labels()
            percent_handles, percent_labels = ax_percent.get_legend_handles_labels()
            ax_voltage.legend(voltage_handles + percent_handles, voltage_labels + percent_labels, fontsize=8, loc="lower left")
            add_reference_line_to_axis(ax_voltage, reference_times)
            fig.suptitle(f"Battery Voltage and Estimated Charge\n{nmpc_mode_summary}", fontsize=13, y=0.995)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {args.pdf_path}")

    if args.html_path:
        generate_diagnostics_html(
            solver_df=solver_df,
            px4_df=px4_df,
            px4_flight_df=px4_flight_df,
            offline_df=offline_df,
            offline_summary=offline_summary,
            html_path=args.html_path,
            solver_t=solver_t,
            offline_t=offline_t,
            px4_t=px4_t,
            px4_flight_t=px4_flight_t,
            reference_times=reference_times,
            solve_time_ms=solve_time_ms,
            solver_input_dt_ms=solver_input_dt_ms,
            solver_posvel_dt_ms=solver_posvel_dt_ms,
            solver_att_dt_ms=solver_att_dt_ms,
            solver_ang_dt_ms=solver_ang_dt_ms,
            acceleration_traces=acceleration_traces,
            optimizer_cost_series=optimizer_cost_series,
            instant_cost_series=instant_cost_series,
            dominant_cost_series=dominant_cost_series,
            nmpc_mode_summary=nmpc_mode_summary,
            gap_center=gap_center,
        )
        print(f"Saved {args.html_path}")


if __name__ == "__main__":
    main()
