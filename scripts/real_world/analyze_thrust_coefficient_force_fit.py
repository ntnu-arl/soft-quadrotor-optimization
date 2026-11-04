import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np
import pandas as pd


NUM_MOTORS = 4
FORCE_COMPONENTS = ["x", "y", "z"]
FORCE_COLORS = ["tab:blue", "tab:orange", "tab:green"]
FORCE_MEASUREMENT_COLUMNS = ["force_measurement_bx", "force_measurement_by", "force_measurement_bz"]
ESTIMATED_RPS_COLUMNS = ["estimated_rps0", "estimated_rps1", "estimated_rps2", "estimated_rps3"]
PARAM_ALLOC_COLUMNS = [f"p{i:02d}" for i in range(20, 44)]
PARAM_KF_COLUMNS = [f"p{i:02d}" for i in range(44, 48)]


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
    offline_candidates = list(candidate_dir.glob("nmpc_solver_data_*_offline_disturbance.csv"))
    return len(solver_candidates) == 1 and len(offline_candidates) == 1


def resolve_input_dir(experiment_dir):
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

    return candidate_input_dirs[0]


def load_logs(input_dir):
    solver_candidates = sorted(
        path
        for path in input_dir.glob("nmpc_solver_data_*.csv")
        if "_offline_disturbance" not in path.name
    )
    offline_candidates = sorted(input_dir.glob("nmpc_solver_data_*_offline_disturbance.csv"))

    if len(solver_candidates) != 1:
        raise ValueError(f"expected exactly one solver csv in {input_dir}, found {len(solver_candidates)}")
    if len(offline_candidates) != 1:
        raise ValueError(f"expected exactly one offline disturbance csv in {input_dir}, found {len(offline_candidates)}")

    solver_log_path = solver_candidates[0]
    offline_log_path = offline_candidates[0]
    solver_df = pd.read_csv(solver_log_path)
    offline_df = pd.read_csv(offline_log_path)
    return solver_df, offline_df, solver_log_path, offline_log_path


def extract_constant_matrix(df, columns, rows, cols, name):
    values = df[columns].to_numpy(dtype=np.float64)
    reference = values[0]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(values, reference, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} is not constant across the log")
    return reference.reshape(rows, cols, order="F")


def extract_constant_vector(df, columns, name):
    values = df[columns].to_numpy(dtype=np.float64)
    reference = values[0]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(values, reference, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} is not constant across the log")
    return reference


def validate_aligned_logs(solver_df, offline_df):
    if len(solver_df) != len(offline_df):
        raise ValueError(f"solver and offline logs have different lengths: {len(solver_df)} vs {len(offline_df)}")

    for column in ["t_us", "sample_timestamp_us", "position_velocity_timestamp_us", "attitude_timestamp_us", "angular_velocity_timestamp_us"]:
        if not solver_df[column].equals(offline_df[column]):
            raise ValueError(f"solver and offline logs are not aligned on column {column}")


def compute_predicted_force_body(allocation_matrix, thrust_coefficients, motor_rps):
    actual_forces = thrust_coefficients[np.newaxis, :] * motor_rps ** 2
    wrench_body = actual_forces @ allocation_matrix.T
    return wrench_body[:, 0:3]


def compute_rmse(force_residual_body):
    if force_residual_body.shape[0] == 0:
        raise ValueError("cannot compute RMSE with zero valid force samples")
    return np.sqrt(np.mean(force_residual_body ** 2, axis=0))


def build_plot(
    time_s,
    observed_force_body,
    logged_predicted_force_body,
    selected_predicted_force_body,
    logged_disturbance_force_body,
    selected_disturbance_force_body,
    logged_kf,
    update_selected_force_callback,
):
    figure, axes = plt.subplots(2, 1, figsize=(15.0, 9.0), sharex=True)
    plt.subplots_adjust(left=0.08, right=0.82, bottom=0.18, top=0.90, hspace=0.22)

    force_axis = axes[0]
    residual_axis = axes[1]

    force_axis.set_title("Observed vs predicted body force")
    force_axis.set_ylabel("Force [N]")
    force_axis.grid(True, alpha=0.3)

    residual_axis.set_title("Residual disturbance force = observed - predicted")
    residual_axis.set_xlabel("Time [s]")
    residual_axis.set_ylabel("Force [N]")
    residual_axis.grid(True, alpha=0.3)

    source_handles = [
        force_axis.plot([], [], color="black", linewidth=1.8, linestyle="-", label="observed body force")[0],
        force_axis.plot([], [], color="black", linewidth=1.3, linestyle="--", label="predicted with logged coefficient")[0],
        force_axis.plot([], [], color="black", linewidth=1.3, linestyle=":", label="predicted with selected coefficient")[0],
    ]
    component_handles = [
        force_axis.plot([], [], color=FORCE_COLORS[force_index], linewidth=1.8, linestyle="-", label=f"body {force_name}")[0]
        for force_index, force_name in enumerate(FORCE_COMPONENTS)
    ]

    residual_handles = [
        residual_axis.plot([], [], color="black", linewidth=1.3, linestyle="--", label="residual with logged coefficient")[0],
        residual_axis.plot([], [], color="black", linewidth=1.3, linestyle=":", label="residual with selected coefficient")[0],
    ]

    axis_lines = {}
    for force_index, force_name in enumerate(FORCE_COMPONENTS):
        observed_line, = force_axis.plot(
            time_s,
            observed_force_body[:, force_index],
            color=FORCE_COLORS[force_index],
            linewidth=1.8,
            linestyle="-",
        )
        logged_predicted_line, = force_axis.plot(
            time_s,
            logged_predicted_force_body[:, force_index],
            color=FORCE_COLORS[force_index],
            linewidth=1.3,
            linestyle="--",
        )
        selected_predicted_line, = force_axis.plot(
            time_s,
            selected_predicted_force_body[:, force_index],
            color=FORCE_COLORS[force_index],
            linewidth=1.3,
            linestyle=":",
        )
        logged_residual_line, = residual_axis.plot(
            time_s,
            logged_disturbance_force_body[:, force_index],
            color=FORCE_COLORS[force_index],
            linewidth=1.3,
            linestyle="--",
        )
        selected_residual_line, = residual_axis.plot(
            time_s,
            selected_disturbance_force_body[:, force_index],
            color=FORCE_COLORS[force_index],
            linewidth=1.3,
            linestyle=":",
        )
        axis_lines[force_index] = (
            observed_line,
            logged_predicted_line,
            selected_predicted_line,
            logged_residual_line,
            selected_residual_line,
        )

    force_axis.legend(handles=source_handles + component_handles, loc="upper left")
    residual_axis.legend(handles=residual_handles, loc="upper left")

    button_axis = figure.add_axes([0.845, 0.48, 0.13, 0.18])
    check_buttons = CheckButtons(button_axis, [f"body {axis_name}" for axis_name in FORCE_COMPONENTS], [True] * len(FORCE_COMPONENTS))
    button_axis.set_title("Visible")

    slider_axis = figure.add_axes([0.10, 0.08, 0.62, 0.04])
    kf_min = float(np.min(logged_kf) * 0.25)
    kf_max = float(np.max(logged_kf) * 4.0)
    kf_slider = Slider(slider_axis, "thrust coefficient [N/RPS^2]", kf_min, kf_max, valinit=float(logged_kf[0]), valfmt="%.8e")

    info_text = figure.text(0.08, 0.95, "", ha="left", va="top")

    def sync_visibility(_label):
        for force_index, active in enumerate(check_buttons.get_status()):
            for line in axis_lines[force_index]:
                line.set_visible(active)
        figure.canvas.draw_idle()

    def update_kf(_value):
        rmse_logged, rmse_selected = update_selected_force_callback(kf_slider.val)
        info_text.set_text(
            "logged kf = "
            f"{logged_kf[0]:.8e}   selected kf = {kf_slider.val:.8e}   "
            f"RMSE logged xyz = [{rmse_logged[0]:.3f}, {rmse_logged[1]:.3f}, {rmse_logged[2]:.3f}] N   "
            f"RMSE selected xyz = [{rmse_selected[0]:.3f}, {rmse_selected[1]:.3f}, {rmse_selected[2]:.3f}] N"
        )
        figure.canvas.draw_idle()

    check_buttons.on_clicked(sync_visibility)
    kf_slider.on_changed(update_kf)
    update_kf(logged_kf[0])

    figure._analyze_thrust_coefficient_widgets = {
        "check_buttons": check_buttons,
        "kf_slider": kf_slider,
        "info_text": info_text,
    }

    return figure, axis_lines


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    input_dir = resolve_input_dir(experiment_dir)
    solver_df, offline_df, solver_log_path, offline_log_path = load_logs(input_dir)
    validate_aligned_logs(solver_df, offline_df)

    allocation_matrix = extract_constant_matrix(solver_df, PARAM_ALLOC_COLUMNS, 6, NUM_MOTORS, "allocation matrix")
    logged_kf = extract_constant_vector(solver_df, PARAM_KF_COLUMNS, "thrust coefficients")
    if np.any(logged_kf <= 0.0):
        raise ValueError(f"logged thrust coefficients must be strictly positive, got {logged_kf}")

    valid_force_mask = offline_df["force_measurement_valid"].to_numpy(dtype=np.int64) == 1
    if not np.any(valid_force_mask):
        raise ValueError("offline disturbance log does not contain any valid force measurements")

    time_s = (
        solver_df.loc[valid_force_mask, "t_us"].to_numpy(dtype=np.float64)
        - solver_df.loc[valid_force_mask, "t_us"].iloc[0]
    ) * 1.0e-6
    motor_rps = solver_df.loc[valid_force_mask, ESTIMATED_RPS_COLUMNS].to_numpy(dtype=np.float64)
    logged_disturbance_force_body = offline_df.loc[valid_force_mask, FORCE_MEASUREMENT_COLUMNS].to_numpy(dtype=np.float64)
    logged_predicted_force_body = compute_predicted_force_body(allocation_matrix, logged_kf, motor_rps)
    observed_force_body = logged_disturbance_force_body + logged_predicted_force_body
    selected_predicted_force_body = logged_predicted_force_body.copy()
    selected_disturbance_force_body = logged_disturbance_force_body.copy()
    logged_rmse = compute_rmse(logged_disturbance_force_body)

    print(f"experiment_dir={experiment_dir}")
    print(f"input_dir={input_dir}")
    print(f"solver_log={solver_log_path}")
    print(f"offline_disturbance_log={offline_log_path}")
    print(f"valid_force_samples={int(np.count_nonzero(valid_force_mask))}")
    print(f"logged_thrust_coefficients={logged_kf.tolist()}")
    print(f"logged_residual_rmse_xyz_n={logged_rmse.tolist()}")

    plot_state = {}

    def update_selected_force_callback(selected_kf_scalar):
        selected_kf = np.full(NUM_MOTORS, float(selected_kf_scalar), dtype=np.float64)
        updated_predicted_force_body = compute_predicted_force_body(allocation_matrix, selected_kf, motor_rps)
        updated_disturbance_force_body = observed_force_body - updated_predicted_force_body
        selected_predicted_force_body[:, :] = updated_predicted_force_body
        selected_disturbance_force_body[:, :] = updated_disturbance_force_body

        axis_lines = plot_state.get("axis_lines")
        if axis_lines is not None:
            for force_index in range(len(FORCE_COMPONENTS)):
                axis_lines[force_index][2].set_ydata(selected_predicted_force_body[:, force_index])
                axis_lines[force_index][4].set_ydata(selected_disturbance_force_body[:, force_index])

        selected_rmse = compute_rmse(selected_disturbance_force_body)
        return logged_rmse, selected_rmse

    figure, axis_lines = build_plot(
        time_s,
        observed_force_body,
        logged_predicted_force_body,
        selected_predicted_force_body,
        logged_disturbance_force_body,
        selected_disturbance_force_body,
        logged_kf,
        update_selected_force_callback,
    )
    plot_state["axis_lines"] = axis_lines
    figure.canvas.manager.set_window_title("analyze_thrust_coefficient_force_fit")
    plt.show()


if __name__ == "__main__":
    main()
