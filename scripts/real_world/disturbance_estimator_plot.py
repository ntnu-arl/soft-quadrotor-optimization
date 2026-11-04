import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd


FIT_START_S = 5.0
FIT_END_S = 24.0
PLOT_Y_LIMIT_N = 0.03
NUM_MOTORS = 4
PARAM_ALLOC = 23
PARAM_KF = 47
SMOOTHING_WINDOW_SAMPLES = 50
NMPC_SOLVER_SOURCE_PATH = Path(__file__).resolve().parents[2] / "voxl_src" / "nmpc_pipe_solver.c"
NMPC_DEFINE_NAMES = [
    "DIST_INITIAL_SIGMA_RATIO",
    "DIST_READY_SIGMA_RATIO",
    "DIST_VARIANCE_FLOOR_SIGMA_RATIO",
    "DIST_FORCE_PROCESS_SIGMA_RATIO",
    "DIST_FORCE_MEAS_SIGMA_RATIO",
    "DIST_FORCE_ABS_GATE_RATIO",
    "DIST_INNOVATION_GATE_NIS",
    "DIST_SLEW_RATE_RATIO_PER_S",
    "DIST_FORCE_BOUND_X",
    "DIST_FORCE_BOUND_Y",
    "DIST_FORCE_BOUND_Z",
]
SLIDER_SPECS = [
    ("DIST_INITIAL_SIGMA_RATIO", "initial sigma", 0.01, 1.50, "%.4f"),
    ("DIST_READY_SIGMA_RATIO", "ready sigma", 0.01, 0.80, "%.4f"),
    ("DIST_VARIANCE_FLOOR_SIGMA_RATIO", "variance floor", 0.001, 0.40, "%.4f"),
    ("DIST_FORCE_PROCESS_SIGMA_RATIO", "process sigma", 0.001, 0.30, "%.4f"),
    ("DIST_FORCE_MEAS_SIGMA_RATIO", "measurement sigma", 0.01, 1.00, "%.4f"),
    ("DIST_FORCE_ABS_GATE_RATIO", "abs gate", 0.10, 8.00, "%.3f"),
    ("DIST_INNOVATION_GATE_NIS", "innovation gate", 1.00, 100.00, "%.2f"),
    ("DIST_SLEW_RATE_RATIO_PER_S", "slew rate", 0.01, 5.00, "%.4f"),
    ("DIST_FORCE_BOUND_X", "bound x [N]", 0.05, 4.00, "%.3f"),
    ("DIST_FORCE_BOUND_Y", "bound y [N]", 0.05, 4.00, "%.3f"),
    ("DIST_FORCE_BOUND_Z", "bound z [N]", 0.05, 4.00, "%.3f"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def load_nmpc_force_disturbance_configuration():
    source_text = NMPC_SOLVER_SOURCE_PATH.read_text(encoding="utf-8")
    configuration = {}
    for define_name in NMPC_DEFINE_NAMES:
        match = re.search(
            rf"^\s*#define\s+{re.escape(define_name)}\s+([-+0-9.eE]+)\s*$",
            source_text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"missing #define {define_name} in {NMPC_SOLVER_SOURCE_PATH}")
        configuration[define_name] = float(match.group(1))
    return configuration


def quat_to_rotmat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if not np.isfinite(quat_norm) or quat_norm <= 0.0:
        raise RuntimeError(f"invalid quaternion norm={quat_norm}")
    quat = quat / quat_norm
    qw, qx, qy, qz = quat
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def compute_body_thrust_from_rps(params, motor_rps):
    actual_forces = np.zeros(NUM_MOTORS, dtype=np.float64)
    for motor_index in range(NUM_MOTORS):
        thrust_coeff = params[PARAM_KF + motor_index]
        if not np.isfinite(thrust_coeff) or thrust_coeff <= 0.0:
            raise RuntimeError(f"invalid thrust coefficient p{PARAM_KF + motor_index:02d}={thrust_coeff}")
        actual_forces[motor_index] = thrust_coeff * motor_rps[motor_index] * motor_rps[motor_index]

    force_body = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        for motor_index in range(NUM_MOTORS):
            alloc_base = PARAM_ALLOC + 6 * motor_index
            force_body[axis] += params[alloc_base + axis] * actual_forces[motor_index]
    return force_body


def smooth_series(values):
    finite_mask = np.isfinite(values)
    smoothed = values.copy()
    if np.count_nonzero(finite_mask) < SMOOTHING_WINDOW_SAMPLES:
        return smoothed

    kernel = np.ones(SMOOTHING_WINDOW_SAMPLES, dtype=np.float64) / float(SMOOTHING_WINDOW_SAMPLES)
    finite_values = values[finite_mask]
    smoothed[finite_mask] = np.convolve(finite_values, kernel, mode="same")
    return smoothed


def load_run_data(run_dir):
    solver_path = run_dir / "nmpc_solver_data_combined.csv"
    if not solver_path.is_file():
        raise FileNotFoundError(f"missing solver log: {solver_path}")

    solver_df = pd.read_csv(solver_path)
    params = np.array([float(solver_df[f"p{param_index:02d}"].iloc[0]) for param_index in range(68)], dtype=np.float64)
    time_s = (solver_df["t_us"].to_numpy(dtype=np.float64) - float(solver_df["t_us"].iloc[0])) * 1.0e-6
    quat = solver_df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    velocity_world = solver_df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    corrected_rps = solver_df[[f"corrected_rps{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=np.float64)
    position_velocity_timestamp_us = solver_df["position_velocity_timestamp_us"].to_numpy(dtype=np.int64)
    packet_flags = solver_df["packet_flags"].to_numpy(dtype=np.int64)

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

    return {
        "params": params,
        "time_s": time_s,
        "velocity_world": velocity_world,
        "rotation_world_from_body": rotation_world_from_body,
        "position_velocity_timestamp_us": position_velocity_timestamp_us,
        "packet_flags": packet_flags,
        "thrust_body": thrust_body,
        "measured_world_acc": measured_world_acc,
        "valid_force_measurement": valid_force_measurement,
    }


def reconstruct_baseline_disturbance(run_data):
    params = run_data["params"]
    mass = float(params[0])
    gravity_world = params[7:10]
    row_count = len(run_data["time_s"])
    disturbance_body = np.full((row_count, 3), np.nan, dtype=np.float64)
    disturbance_world = np.full((row_count, 3), np.nan, dtype=np.float64)

    for row_index in np.where(run_data["valid_force_measurement"])[0]:
        rotation_world_from_body = run_data["rotation_world_from_body"][row_index]
        measured_body_acc = rotation_world_from_body.T @ run_data["measured_world_acc"][row_index]
        gravity_body = rotation_world_from_body.T @ gravity_world
        disturbance_body[row_index] = (
            mass * (measured_body_acc - gravity_body)
            - run_data["thrust_body"][row_index]
        )
        disturbance_world[row_index] = rotation_world_from_body @ disturbance_body[row_index]

    return disturbance_body, disturbance_world


def clip_scalar(value, lower, upper):
    return min(max(value, lower), upper)


def update_force_filter_state(estimated, applied, variance, measurement, force_bound, dt_s, configuration):
    innovation_nis = 0.0
    gated = False
    ready = True

    for axis in range(3):
        process_sigma = configuration["DIST_FORCE_PROCESS_SIGMA_RATIO"] * force_bound[axis]
        meas_sigma = configuration["DIST_FORCE_MEAS_SIGMA_RATIO"] * force_bound[axis]
        min_sigma = configuration["DIST_VARIANCE_FLOOR_SIGMA_RATIO"] * force_bound[axis]
        min_var = min_sigma * min_sigma
        max_var = force_bound[axis] * force_bound[axis]
        variance[axis] = clip_scalar(variance[axis] + process_sigma * process_sigma * dt_s, min_var, max_var)
        innovation = measurement[axis] - estimated[axis]
        innovation_nis += innovation * innovation / (variance[axis] + meas_sigma * meas_sigma)
        if abs(measurement[axis]) > configuration["DIST_FORCE_ABS_GATE_RATIO"] * force_bound[axis]:
            gated = True

    if innovation_nis > configuration["DIST_INNOVATION_GATE_NIS"]:
        gated = True

    if not gated:
        for axis in range(3):
            meas_sigma = configuration["DIST_FORCE_MEAS_SIGMA_RATIO"] * force_bound[axis]
            min_sigma = configuration["DIST_VARIANCE_FLOOR_SIGMA_RATIO"] * force_bound[axis]
            min_var = min_sigma * min_sigma
            max_var = force_bound[axis] * force_bound[axis]
            innovation = measurement[axis] - estimated[axis]
            innovation_cov = variance[axis] + meas_sigma * meas_sigma
            gain = variance[axis] / innovation_cov
            estimated[axis] += gain * innovation
            variance[axis] = clip_scalar((1.0 - gain) * variance[axis], min_var, max_var)

    for axis in range(3):
        estimated[axis] = clip_scalar(estimated[axis], -force_bound[axis], force_bound[axis])
        if np.sqrt(variance[axis]) > configuration["DIST_READY_SIGMA_RATIO"] * force_bound[axis]:
            ready = False

    if ready:
        for axis in range(3):
            step = configuration["DIST_SLEW_RATE_RATIO_PER_S"] * force_bound[axis] * dt_s
            delta = clip_scalar(estimated[axis] - applied[axis], -step, step)
            applied[axis] = clip_scalar(applied[axis] + delta, -force_bound[axis], force_bound[axis])
    else:
        applied[:] = 0.0


def simulate_online_force_disturbance(run_data, disturbance_body, configuration):
    row_count = len(run_data["time_s"])
    force_bound = np.array(
        [
            configuration["DIST_FORCE_BOUND_X"],
            configuration["DIST_FORCE_BOUND_Y"],
            configuration["DIST_FORCE_BOUND_Z"],
        ],
        dtype=np.float64,
    )
    estimated = np.zeros(3, dtype=np.float64)
    applied = np.zeros(3, dtype=np.float64)
    variance = np.square(configuration["DIST_INITIAL_SIGMA_RATIO"] * force_bound)
    applied_force_body = np.zeros((row_count, 3), dtype=np.float64)
    initialized = False
    last_force_timestamp_us = 0

    for row_index in range(row_count):
        reset_now = (run_data["packet_flags"][row_index] & 0x01) != 0
        if reset_now or not initialized:
            estimated[:] = 0.0
            applied[:] = 0.0
            variance[:] = np.square(configuration["DIST_INITIAL_SIGMA_RATIO"] * force_bound)
            last_force_timestamp_us = run_data["position_velocity_timestamp_us"][row_index]
            initialized = True
            applied_force_body[row_index] = applied
            continue

        force_timestamp_us = run_data["position_velocity_timestamp_us"][row_index]
        if force_timestamp_us < last_force_timestamp_us:
            raise RuntimeError(
                "position/velocity timestamp moved backwards: "
                f"current={force_timestamp_us} previous={last_force_timestamp_us}"
            )

        if force_timestamp_us > last_force_timestamp_us:
            if not run_data["valid_force_measurement"][row_index]:
                raise RuntimeError(f"advanced timestamp without valid force measurement at row_index={row_index}")
            dt_s = float(force_timestamp_us - last_force_timestamp_us) * 1.0e-6
            update_force_filter_state(
                estimated,
                applied,
                variance,
                disturbance_body[row_index],
                force_bound,
                dt_s,
                configuration,
            )
            last_force_timestamp_us = force_timestamp_us

        applied_force_body[row_index] = applied

    return applied_force_body


def compute_leftover_force(run_data, disturbance_body, disturbance_world, configuration):
    applied_force_body = simulate_online_force_disturbance(run_data, disturbance_body, configuration)
    applied_force_world = np.einsum("nij,nj->ni", run_data["rotation_world_from_body"], applied_force_body)
    smoothed_disturbance_body = np.column_stack(
        [
            smooth_series(disturbance_body[:, 0]),
            smooth_series(disturbance_body[:, 1]),
            smooth_series(disturbance_body[:, 2]),
        ]
    )
    smoothed_disturbance_world = np.column_stack(
        [
            smooth_series(disturbance_world[:, 0]),
            smooth_series(disturbance_world[:, 1]),
            smooth_series(disturbance_world[:, 2]),
        ]
    )
    leftover_body = smoothed_disturbance_body - applied_force_body
    leftover_world = smoothed_disturbance_world - applied_force_world
    return leftover_body, leftover_world, applied_force_body


def compute_metrics(fit_mask, leftover_body, leftover_world):
    combined_residual = np.column_stack(
        [
            leftover_body[fit_mask, 0],
            leftover_body[fit_mask, 1],
            leftover_world[fit_mask, 2],
        ]
    )
    planar_residual = combined_residual[:, 0:2]
    return {
        "planar_rmse_N": float(np.sqrt(np.mean(np.sum(planar_residual ** 2, axis=1)))),
        "combined_rmse_N": float(np.sqrt(np.mean(np.sum(combined_residual ** 2, axis=1)))),
        "mean_N": np.mean(combined_residual, axis=0),
    }


def slider_configuration(sliders):
    configuration = {}
    for define_name, _, _, _, _ in SLIDER_SPECS:
        configuration[define_name] = float(sliders[define_name].val)
    return configuration


def build_metrics_text(metrics, applied_force_body, fit_mask):
    applied_abs_max = np.max(np.abs(applied_force_body[fit_mask]), axis=0)
    mean_force = metrics["mean_N"]
    return (
        f"leftover RMSE xy = {metrics['planar_rmse_N']:.5f} N\n"
        f"leftover RMSE xyz = {metrics['combined_rmse_N']:.5f} N\n"
        f"leftover mean [body x, body y, world z] = "
        f"[{mean_force[0]:.5f}, {mean_force[1]:.5f}, {mean_force[2]:.5f}] N\n"
        f"applied max abs body force = "
        f"[{applied_abs_max[0]:.5f}, {applied_abs_max[1]:.5f}, {applied_abs_max[2]:.5f}] N"
    )


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_data = load_run_data(run_dir)
    fit_mask = (
        run_data["valid_force_measurement"]
        & (run_data["time_s"] >= FIT_START_S)
        & (run_data["time_s"] <= FIT_END_S)
    )
    if not np.any(fit_mask):
        raise RuntimeError("no valid force samples in the requested fit window")

    base_configuration = load_nmpc_force_disturbance_configuration()
    disturbance_body, disturbance_world = reconstruct_baseline_disturbance(run_data)
    leftover_body, leftover_world, applied_force_body = compute_leftover_force(
        run_data,
        disturbance_body,
        disturbance_world,
        base_configuration,
    )
    metrics = compute_metrics(fit_mask, leftover_body, leftover_world)

    figure, axis = plt.subplots(figsize=(14, 9))
    figure.subplots_adjust(left=0.08, right=0.76, top=0.91, bottom=0.08)
    time_s = run_data["time_s"]
    plot_time_s = time_s[fit_mask]
    body_x_line, = axis.plot(plot_time_s, leftover_body[fit_mask, 0], color="tab:blue", linewidth=1.2, label="body Fx")
    body_y_line, = axis.plot(plot_time_s, leftover_body[fit_mask, 1], color="tab:orange", linewidth=1.2, label="body Fy")
    world_z_line, = axis.plot(plot_time_s, leftover_world[fit_mask, 2], color="tab:green", linewidth=1.2, label="world Fz")
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("leftover online disturbance [N]")
    axis.set_title(f"{run_dir.name} online force disturbance leftover")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")

    axis.set_ylim(-PLOT_Y_LIMIT_N, PLOT_Y_LIMIT_N)

    metrics_text = figure.text(
        0.08,
        0.965,
        build_metrics_text(metrics, applied_force_body, fit_mask),
        ha="left",
        va="top",
        family="monospace",
        fontsize=9.0,
    )

    sliders = {}
    slider_height = 0.025
    slider_gap = 0.012
    slider_top = 0.86
    for slider_index, (define_name, label, slider_min, slider_max, value_format) in enumerate(SLIDER_SPECS):
        slider_bottom = slider_top - slider_index * (slider_height + slider_gap)
        slider_axis = figure.add_axes([0.80, slider_bottom, 0.17, slider_height])
        sliders[define_name] = Slider(
            slider_axis,
            label,
            slider_min,
            slider_max,
            valinit=base_configuration[define_name],
            valfmt=value_format,
        )

    def update_plot(_value):
        configuration = slider_configuration(sliders)
        updated_leftover_body, updated_leftover_world, updated_applied_force_body = compute_leftover_force(
            run_data,
            disturbance_body,
            disturbance_world,
            configuration,
        )
        updated_metrics = compute_metrics(fit_mask, updated_leftover_body, updated_leftover_world)
        body_x_line.set_ydata(updated_leftover_body[fit_mask, 0])
        body_y_line.set_ydata(updated_leftover_body[fit_mask, 1])
        world_z_line.set_ydata(updated_leftover_world[fit_mask, 2])
        axis.set_ylim(-PLOT_Y_LIMIT_N, PLOT_Y_LIMIT_N)
        metrics_text.set_text(build_metrics_text(updated_metrics, updated_applied_force_body, fit_mask))
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update_plot)

    reset_axis = figure.add_axes([0.80, 0.03, 0.17, 0.04])
    reset_button = Button(reset_axis, "reset")

    def reset_sliders(_event):
        for define_name, _, _, _, _ in SLIDER_SPECS:
            sliders[define_name].reset()

    reset_button.on_clicked(reset_sliders)
    plt.show()


if __name__ == "__main__":
    main()
