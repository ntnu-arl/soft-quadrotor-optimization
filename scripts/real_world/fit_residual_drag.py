import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_circle_model_mismatch import compute_body_thrust_from_rps, quat_to_rotmat


FIT_START_S = 5.0
FIT_END_S = 24.0
HOVER_START_S = 5.0
HOVER_END_S = 10.0
CIRCLE_START_S = 10.0
CIRCLE_END_S = 24.0
MOVE_SPEED_THRESHOLD_MPS = 0.05
SMOOTHING_WINDOW_SAMPLES = 50
NMPC_SOLVER_SOURCE_PATH = Path(__file__).resolve().parents[2] / "voxl_src" / "nmpc_pipe_solver.c"
NUM_MOTORS = 4
NUM_PARAMS = 68
BODY_DRAG_DEFINE_NAMES = [
    "BODY_DRAG_COEFF_BX",
    "BODY_DRAG_COEFF_BY",
    "BODY_DRAG_COEFF_BZ",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def smooth_series(values):
    finite_mask = np.isfinite(values)
    smoothed = values.copy()
    if np.count_nonzero(finite_mask) < SMOOTHING_WINDOW_SAMPLES:
        return smoothed

    kernel = np.ones(SMOOTHING_WINDOW_SAMPLES, dtype=np.float64) / float(SMOOTHING_WINDOW_SAMPLES)
    smoothed[finite_mask] = np.convolve(values[finite_mask], kernel, mode="same")
    return smoothed


def load_current_drag_configuration():
    source_text = NMPC_SOLVER_SOURCE_PATH.read_text(encoding="utf-8")
    configuration = {}
    for define_name in BODY_DRAG_DEFINE_NAMES:
        match = re.search(
            rf"^\s*#define\s+{re.escape(define_name)}\s+([-+0-9.eE]+)\s*$",
            source_text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"missing #define {define_name} in {NMPC_SOLVER_SOURCE_PATH}")
        configuration[define_name] = float(match.group(1))
    return configuration


def load_run_data(run_dir):
    solver_path = run_dir / "nmpc_solver_data_combined.csv"
    if not solver_path.is_file():
        raise FileNotFoundError(f"missing solver log: {solver_path}")

    solver_df = pd.read_csv(solver_path)
    params = np.array([float(solver_df[f"p{param_index:02d}"].iloc[0]) for param_index in range(NUM_PARAMS)], dtype=np.float64)
    time_s = (solver_df["t_us"].to_numpy(dtype=np.float64) - float(solver_df["t_us"].iloc[0])) * 1.0e-6
    quat = solver_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    velocity_world = solver_df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    reference_velocity_world = solver_df[["sp_vx", "sp_vy", "sp_vz"]].to_numpy(dtype=np.float64)
    corrected_rps = solver_df[[f"corrected_rps{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=np.float64)
    position_velocity_timestamp_us = solver_df["position_velocity_timestamp_us"].to_numpy(dtype=np.int64)
    packet_flags = solver_df["packet_flags"].to_numpy(dtype=np.int64)
    fixed_disturbance_force_body = solver_df[
        ["fixed_dist_force_bx", "fixed_dist_force_by", "fixed_dist_force_bz"]
    ].to_numpy(dtype=np.float64)

    rotation_world_from_body = np.array([quat_to_rotmat(row_quat) for row_quat in quat], dtype=np.float64)
    thrust_body = np.array([compute_body_thrust_from_rps(params, row_rps) for row_rps in corrected_rps], dtype=np.float64)
    measured_world_acc = np.full_like(velocity_world, np.nan)
    valid_force_measurement = np.zeros(len(solver_df), dtype=bool)
    previous_velocity = None
    previous_timestamp_us = None

    for row_index in range(len(solver_df)):
        reset_now = previous_velocity is None or previous_timestamp_us is None or (packet_flags[row_index] & 0x01) != 0
        if reset_now:
            previous_velocity = velocity_world[row_index].copy()
            previous_timestamp_us = position_velocity_timestamp_us[row_index]
            continue

        if position_velocity_timestamp_us[row_index] > previous_timestamp_us:
            dt_s = (position_velocity_timestamp_us[row_index] - previous_timestamp_us) * 1.0e-6
            measured_world_acc[row_index] = (velocity_world[row_index] - previous_velocity) / dt_s
            valid_force_measurement[row_index] = True
            previous_velocity = velocity_world[row_index].copy()
            previous_timestamp_us = position_velocity_timestamp_us[row_index]

    reference_speed = np.linalg.norm(reference_velocity_world[:, 0:2], axis=1)
    moving_mask = reference_speed > MOVE_SPEED_THRESHOLD_MPS
    tangent_world = np.zeros_like(reference_velocity_world)
    tangent_world[moving_mask, 0:2] = reference_velocity_world[moving_mask, 0:2] / reference_speed[moving_mask, None]

    return {
        "solver_df": solver_df,
        "params": params,
        "time_s": time_s,
        "velocity_world": velocity_world,
        "rotation_world_from_body": rotation_world_from_body,
        "fixed_disturbance_force_body": fixed_disturbance_force_body,
        "position_velocity_timestamp_us": position_velocity_timestamp_us,
        "packet_flags": packet_flags,
        "thrust_body": thrust_body,
        "measured_world_acc": measured_world_acc,
        "valid_force_measurement": valid_force_measurement,
        "moving_mask": moving_mask,
        "tangent_world": tangent_world,
    }


def reconstruct_residual(run_data, drag_coeff_bx, drag_coeff_by):
    params = run_data["params"]
    mass = float(params[0])
    gravity_world = params[7:10]
    row_count = len(run_data["time_s"])
    residual_body = np.full((row_count, 3), np.nan, dtype=np.float64)
    residual_world = np.full((row_count, 3), np.nan, dtype=np.float64)
    residual_tangent = np.full(row_count, np.nan, dtype=np.float64)
    velocity_body = np.full((row_count, 3), np.nan, dtype=np.float64)

    for row_index in np.where(run_data["valid_force_measurement"])[0]:
        rotation_world_from_body = run_data["rotation_world_from_body"][row_index]
        velocity_body[row_index] = rotation_world_from_body.T @ run_data["velocity_world"][row_index]
        measured_body_acc = rotation_world_from_body.T @ run_data["measured_world_acc"][row_index]
        gravity_body = rotation_world_from_body.T @ gravity_world
        drag_force_body = np.array(
            [
                -drag_coeff_bx * velocity_body[row_index, 0],
                -drag_coeff_by * velocity_body[row_index, 1],
                0.0,
            ],
            dtype=np.float64,
        )
        residual_body[row_index] = (
            mass * (measured_body_acc - gravity_body)
            - run_data["thrust_body"][row_index]
            - drag_force_body
        )
        residual_world[row_index] = rotation_world_from_body @ residual_body[row_index]
        residual_tangent[row_index] = float(np.dot(residual_world[row_index], run_data["tangent_world"][row_index]))

    return residual_body, residual_world, residual_tangent, velocity_body


def fit_drag_coefficients(run_data, fit_mask):
    baseline_body, _, _, velocity_body = reconstruct_residual(run_data, 0.0, 0.0)

    fit_velocity_x = velocity_body[fit_mask, 0]
    fit_velocity_y = velocity_body[fit_mask, 1]
    fit_baseline_x = baseline_body[fit_mask, 0]
    fit_baseline_y = baseline_body[fit_mask, 1]

    if not np.any(np.abs(fit_velocity_x) > 0.0):
        raise RuntimeError("body x velocity is identically zero in the fit window")
    if not np.any(np.abs(fit_velocity_y) > 0.0):
        raise RuntimeError("body y velocity is identically zero in the fit window")

    design_x = fit_velocity_x[:, None]
    design_y = fit_velocity_y[:, None]
    drag_coeff_bx = -float(np.linalg.lstsq(design_x, fit_baseline_x, rcond=None)[0][0])
    drag_coeff_by = -float(np.linalg.lstsq(design_y, fit_baseline_y, rcond=None)[0][0])

    if not np.isfinite(drag_coeff_bx) or not np.isfinite(drag_coeff_by):
        raise RuntimeError("fitted drag coefficients are non-finite")

    return drag_coeff_bx, drag_coeff_by


def compute_metrics(run_data, residual_body, residual_world, residual_tangent):
    hover_mask = (
        run_data["valid_force_measurement"]
        & (run_data["time_s"] >= HOVER_START_S)
        & (run_data["time_s"] <= HOVER_END_S)
    )
    circle_mask = (
        run_data["valid_force_measurement"]
        & run_data["moving_mask"]
        & (run_data["time_s"] >= CIRCLE_START_S)
        & (run_data["time_s"] <= CIRCLE_END_S)
    )
    fit_mask = (
        run_data["valid_force_measurement"]
        & (run_data["time_s"] >= FIT_START_S)
        & (run_data["time_s"] <= FIT_END_S)
    )

    def residual_norm_rmse(mask):
        return float(np.sqrt(np.mean(np.sum(residual_body[mask] ** 2, axis=1))))

    return {
        "fit_count": int(np.count_nonzero(fit_mask)),
        "hover_count": int(np.count_nonzero(hover_mask)),
        "circle_count": int(np.count_nonzero(circle_mask)),
        "fit_planar_rmse_N": float(
            np.sqrt(
                np.mean(
                    residual_body[fit_mask, 0] * residual_body[fit_mask, 0]
                    + residual_body[fit_mask, 1] * residual_body[fit_mask, 1]
                )
            )
        ),
        "fit_world_fz_rmse_N": float(np.sqrt(np.mean(residual_world[fit_mask, 2] * residual_world[fit_mask, 2]))),
        "fit_world_fz_mean_N": float(np.mean(residual_world[fit_mask, 2])),
        "hover_force_rmse_norm_N": residual_norm_rmse(hover_mask),
        "circle_force_rmse_norm_N": residual_norm_rmse(circle_mask),
        "circle_tangent_mean_N": float(np.mean(residual_tangent[circle_mask])),
        "circle_tangent_std_N": float(np.std(residual_tangent[circle_mask])),
    }


def plot_before_after(run_data, current_outputs, fitted_outputs, output_png):
    current_body, current_world, current_tangent, _ = current_outputs
    fitted_body, fitted_world, fitted_tangent, _ = fitted_outputs
    time_s = run_data["time_s"]
    valid_mask = run_data["valid_force_measurement"]
    circle_mask = valid_mask & run_data["moving_mask"] & (time_s >= CIRCLE_START_S) & (time_s <= CIRCLE_END_S)

    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(time_s[valid_mask], smooth_series(current_body[:, 0])[valid_mask], label="current", linewidth=1.2)
    axes[0].plot(time_s[valid_mask], smooth_series(fitted_body[:, 0])[valid_mask], label="fitted", linewidth=1.2)
    axes[0].set_ylabel("body Fx [N]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s[valid_mask], smooth_series(current_body[:, 1])[valid_mask], linewidth=1.2)
    axes[1].plot(time_s[valid_mask], smooth_series(fitted_body[:, 1])[valid_mask], linewidth=1.2)
    axes[1].set_ylabel("body Fy [N]")

    axes[2].plot(time_s[valid_mask], smooth_series(current_world[:, 2])[valid_mask], linewidth=1.2)
    axes[2].plot(time_s[valid_mask], smooth_series(fitted_world[:, 2])[valid_mask], linewidth=1.2)
    axes[2].set_ylabel("world Fz [N]")

    axes[3].plot(time_s[circle_mask], smooth_series(current_tangent)[circle_mask], linewidth=1.2)
    axes[3].plot(time_s[circle_mask], smooth_series(fitted_tangent)[circle_mask], linewidth=1.2)
    axes[3].set_ylabel("tangent F [N]")
    axes[3].set_xlabel("time [s]")

    for axis in axes:
        axis.axvspan(HOVER_START_S, HOVER_END_S, color="0.92")
        axis.axvspan(CIRCLE_START_S, CIRCLE_END_S, color="0.96")
        axis.grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_png, dpi=200)
    plt.close(figure)


def write_summary(output_txt, current_drag_configuration, fitted_drag_coeff_bx, fitted_drag_coeff_by, current_metrics, fitted_metrics):
    with output_txt.open("w", encoding="ascii") as output_file:
        output_file.write(f"current_drag_coeff_bx={current_drag_configuration['BODY_DRAG_COEFF_BX']:.9f}\n")
        output_file.write(f"current_drag_coeff_by={current_drag_configuration['BODY_DRAG_COEFF_BY']:.9f}\n")
        output_file.write(f"current_drag_coeff_bz={current_drag_configuration['BODY_DRAG_COEFF_BZ']:.9f}\n")
        output_file.write(f"fitted_drag_coeff_bx={fitted_drag_coeff_bx:.9f}\n")
        output_file.write(f"fitted_drag_coeff_by={fitted_drag_coeff_by:.9f}\n")
        for key, value in current_metrics.items():
            output_file.write(f"current_{key}={value:.9f}\n")
        for key, value in fitted_metrics.items():
            output_file.write(f"fitted_{key}={value:.9f}\n")


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_data = load_run_data(run_dir)
    current_drag_configuration = load_current_drag_configuration()
    fit_mask = (
        run_data["valid_force_measurement"]
        & (run_data["time_s"] >= FIT_START_S)
        & (run_data["time_s"] <= FIT_END_S)
    )

    if not np.any(fit_mask):
        raise RuntimeError("no valid force samples in the requested fit window")

    fitted_drag_coeff_bx, fitted_drag_coeff_by = fit_drag_coefficients(run_data, fit_mask)
    current_outputs = reconstruct_residual(
        run_data,
        current_drag_configuration["BODY_DRAG_COEFF_BX"],
        current_drag_configuration["BODY_DRAG_COEFF_BY"],
    )
    fitted_outputs = reconstruct_residual(run_data, fitted_drag_coeff_bx, fitted_drag_coeff_by)
    current_metrics = compute_metrics(run_data, current_outputs[0], current_outputs[1], current_outputs[2])
    fitted_metrics = compute_metrics(run_data, fitted_outputs[0], fitted_outputs[1], fitted_outputs[2])

    output_png = run_dir / "fit_residual_drag_before_after.png"
    output_txt = run_dir / "fit_residual_drag_summary.txt"
    plot_before_after(run_data, current_outputs, fitted_outputs, output_png)
    write_summary(
        output_txt,
        current_drag_configuration,
        fitted_drag_coeff_bx,
        fitted_drag_coeff_by,
        current_metrics,
        fitted_metrics,
    )

    print(f"run_dir = {run_dir}")
    print(f"current_drag_coeff_bx = {current_drag_configuration['BODY_DRAG_COEFF_BX']:.9f}")
    print(f"current_drag_coeff_by = {current_drag_configuration['BODY_DRAG_COEFF_BY']:.9f}")
    print(f"fitted_drag_coeff_bx = {fitted_drag_coeff_bx:.9f}")
    print(f"fitted_drag_coeff_by = {fitted_drag_coeff_by:.9f}")
    print(f"current_fit_planar_rmse_N = {current_metrics['fit_planar_rmse_N']:.9f}")
    print(f"fitted_fit_planar_rmse_N = {fitted_metrics['fit_planar_rmse_N']:.9f}")
    print(f"summary_txt = {output_txt}")
    print(f"figure_png = {output_png}")


if __name__ == "__main__":
    main()
