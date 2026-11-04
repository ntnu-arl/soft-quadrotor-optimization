import argparse
from dataclasses import dataclass
from pathlib import Path

import cma
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import numpy as np
import pandas as pd


NUM_MOTORS = 4
SOLVER_COMMAND_RPS_COLUMNS = [f"command_rps{i}" for i in range(NUM_MOTORS)]
SOLVER_ESTIMATED_RPS_COLUMNS = [f"estimated_rps{i}" for i in range(NUM_MOTORS)]
SOLVER_PREV_COMMAND_RPS_COLUMNS = [f"prev_command_rps{i}" for i in range(NUM_MOTORS)]
MOTOR_EXPECTED_RPM_COLUMNS = [f"expected_requested_rpm{i}" for i in range(NUM_MOTORS)]
MOTOR_MEASURED_RPM_COLUMNS = [f"measured_rpm{i}" for i in range(NUM_MOTORS)]
PARAM_TAU_COLUMNS = ["p48", "p49", "p50", "p51"]
MOTOR_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


@dataclass
class RunAnalysis:
    input_dir: Path
    solver_log_path: Path
    motor_log_path: Path
    solver_time_s: np.ndarray
    solver_sample_timestamp_us: np.ndarray
    solver_prev_command_rps: np.ndarray
    measured_rps_aligned: np.ndarray
    logged_estimated_rps: np.ndarray
    recomputed_open_loop_rps: np.ndarray
    motor_to_solver_offset_s: float
    motor_to_solver_score: float
    logged_tau_s: float
    motor_rps_valid_mask: np.ndarray | None
    motor_rps_correction_mask: np.ndarray | None
    motor_rps_age_us: np.ndarray | None


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
    return len(solver_candidates) == 1 and (candidate_dir / "px4_motor_log.csv").is_file()


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


def normalize_series(values):
    centered = values - np.mean(values)
    std = np.std(centered)
    if std <= 0.0:
        raise ValueError("cannot normalize a zero-variance series")
    return centered / std


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


def load_logs(input_dir):
    solver_candidates = sorted(
        path
        for path in input_dir.glob("nmpc_solver_data_*.csv")
        if "_offline_disturbance" not in path.name
    )
    if len(solver_candidates) != 1:
        raise ValueError(f"expected exactly one solver csv in {input_dir}, found {len(solver_candidates)}")

    solver_log_path = solver_candidates[0]
    motor_log_path = input_dir / "px4_motor_log.csv"
    solver_df = pd.read_csv(solver_log_path)
    motor_df = pd.read_csv(motor_log_path)
    return solver_df, motor_df, solver_log_path, motor_log_path


def extract_logged_tau_s(solver_df):
    tau_values = solver_df[PARAM_TAU_COLUMNS].to_numpy(dtype=np.float64)
    reference = tau_values[0, 0]
    if not np.all(np.isfinite(tau_values)):
        raise ValueError("solver log contains non-finite motor time constants")
    if not np.allclose(tau_values, reference, rtol=0.0, atol=1.0e-12):
        raise ValueError("solver log does not use a single constant motor time constant across motors and samples")
    if not reference > 0.0:
        raise ValueError(f"logged motor time constant must be positive, got {reference}")
    return float(reference)


def recompute_estimated_rps(sample_timestamp_us, prev_command_rps, initial_estimated_rps, tau_s):
    if not tau_s > 0.0:
        raise ValueError(f"motor time constant must be positive, got {tau_s}")

    estimated_rps = np.empty_like(prev_command_rps)
    estimated_rps[0] = initial_estimated_rps

    for sample_index in range(1, len(sample_timestamp_us)):
        dt_s = (sample_timestamp_us[sample_index] - sample_timestamp_us[sample_index - 1]) * 1.0e-6
        if dt_s < 0.0:
            raise ValueError("sample timestamps moved backwards in solver log")
        decay = np.exp(-dt_s / tau_s)
        estimated_rps[sample_index] = (
            prev_command_rps[sample_index]
            + (estimated_rps[sample_index - 1] - prev_command_rps[sample_index]) * decay
        )

    return estimated_rps


def recompute_estimated_rps_smooth_asymmetric(
    sample_timestamp_us,
    prev_command_rps,
    initial_estimated_rps,
    tau_up_s,
    tau_down_s,
    eps_rps,
):
    if not tau_up_s > 0.0:
        raise ValueError(f"motor acceleration time constant must be positive, got {tau_up_s}")
    if not tau_down_s > 0.0:
        raise ValueError(f"motor deceleration time constant must be positive, got {tau_down_s}")
    if not eps_rps > 0.0:
        raise ValueError(f"motor smoothing factor must be positive, got {eps_rps}")

    estimated_rps = np.empty_like(prev_command_rps)
    estimated_rps[0] = initial_estimated_rps

    for sample_index in range(1, len(sample_timestamp_us)):
        dt_s = (sample_timestamp_us[sample_index] - sample_timestamp_us[sample_index - 1]) * 1.0e-6
        if dt_s < 0.0:
            raise ValueError("sample timestamps moved backwards in solver log")

        diff_rps = prev_command_rps[sample_index] - estimated_rps[sample_index - 1]
        signal = 0.5 * (1.0 + np.tanh(diff_rps / eps_rps))
        tau_inv = signal / tau_up_s + (1.0 - signal) / tau_down_s
        estimated_rps[sample_index] = estimated_rps[sample_index - 1] + dt_s * tau_inv * diff_rps

    return estimated_rps


def recompute_estimated_rps_quadratic_drag(
    sample_timestamp_us,
    prev_command_rps,
    initial_estimated_rps,
    drive_gain,
    linear_damping,
    quadratic_drag,
    transition_resistance,
    transition_eps_rps,
):
    if not drive_gain > 0.0:
        raise ValueError(f"motor drive gain must be positive, got {drive_gain}")
    if not linear_damping >= 0.0:
        raise ValueError(f"motor linear damping must be non-negative, got {linear_damping}")
    if not quadratic_drag >= 0.0:
        raise ValueError(f"motor quadratic drag must be non-negative, got {quadratic_drag}")
    if not transition_resistance >= 0.0:
        raise ValueError(f"motor transition resistance must be non-negative, got {transition_resistance}")
    if not transition_eps_rps > 0.0:
        raise ValueError(f"motor transition smoothing must be positive, got {transition_eps_rps}")

    estimated_rps = np.empty_like(prev_command_rps)
    estimated_rps[0] = initial_estimated_rps

    for sample_index in range(1, len(sample_timestamp_us)):
        dt_s = (sample_timestamp_us[sample_index] - sample_timestamp_us[sample_index - 1]) * 1.0e-6
        if dt_s < 0.0:
            raise ValueError("sample timestamps moved backwards in solver log")

        previous_rps = estimated_rps[sample_index - 1]
        control_input = prev_command_rps[sample_index]
        diff_rps = control_input - previous_rps
        transition_term = transition_resistance * np.tanh(diff_rps / transition_eps_rps)
        rps_dot = (
            drive_gain * control_input
            - linear_damping * previous_rps
            - quadratic_drag * previous_rps**2
            - transition_term
        )
        estimated_rps[sample_index] = previous_rps + dt_s * rps_dot

    return estimated_rps


def analyze_run(input_dir):
    solver_df, motor_df, solver_log_path, motor_log_path = load_logs(input_dir)

    solver_time_s = (solver_df["t_us"].to_numpy(dtype=np.float64) - solver_df["t_us"].iloc[0]) * 1.0e-6
    solver_command_rpm = solver_df[SOLVER_COMMAND_RPS_COLUMNS].to_numpy(dtype=np.float64) * 60.0
    logged_estimated_rps = solver_df[SOLVER_ESTIMATED_RPS_COLUMNS].to_numpy(dtype=np.float64)
    solver_prev_command_rps = solver_df[SOLVER_PREV_COMMAND_RPS_COLUMNS].to_numpy(dtype=np.float64)
    solver_sample_timestamp_us = solver_df["sample_timestamp_us"].to_numpy(dtype=np.int64)

    motor_time_s = (motor_df["timestamp_us"].to_numpy(dtype=np.float64) - motor_df["timestamp_us"].iloc[0]) * 1.0e-6
    motor_expected_rpm = motor_df[MOTOR_EXPECTED_RPM_COLUMNS].to_numpy(dtype=np.float64)
    motor_measured_rps = motor_df[MOTOR_MEASURED_RPM_COLUMNS].to_numpy(dtype=np.float64) / 60.0

    motor_to_solver_offset_s, motor_to_solver_score = find_motor_solver_offset_s(
        solver_time_s,
        solver_command_rpm,
        motor_time_s,
        motor_expected_rpm,
    )
    motor_local_time_s = motor_time_s - motor_to_solver_offset_s
    measured_rps_aligned = np.column_stack(
        [
            np.interp(solver_time_s, motor_local_time_s, motor_measured_rps[:, motor_index])
            for motor_index in range(NUM_MOTORS)
        ]
    )

    logged_tau_s = extract_logged_tau_s(solver_df)
    recomputed_open_loop_rps = recompute_estimated_rps(
        solver_sample_timestamp_us,
        solver_prev_command_rps,
        logged_estimated_rps[0],
        logged_tau_s,
    )

    motor_rps_valid_mask = None
    motor_rps_correction_mask = None
    motor_rps_age_us = None
    if "motor_rps_valid_mask" in solver_df.columns:
        motor_rps_valid_mask = solver_df["motor_rps_valid_mask"].to_numpy(dtype=np.int64)
        motor_rps_correction_mask = solver_df["motor_rps_correction_mask"].to_numpy(dtype=np.int64)
        motor_rps_age_us = solver_df[[f"motor_rps_age_us{i}" for i in range(NUM_MOTORS)]].to_numpy(dtype=np.float64)

    return RunAnalysis(
        input_dir=input_dir,
        solver_log_path=solver_log_path,
        motor_log_path=motor_log_path,
        solver_time_s=solver_time_s,
        solver_sample_timestamp_us=solver_sample_timestamp_us,
        solver_prev_command_rps=solver_prev_command_rps,
        measured_rps_aligned=measured_rps_aligned,
        logged_estimated_rps=logged_estimated_rps,
        recomputed_open_loop_rps=recomputed_open_loop_rps,
        motor_to_solver_offset_s=motor_to_solver_offset_s,
        motor_to_solver_score=motor_to_solver_score,
        logged_tau_s=logged_tau_s,
        motor_rps_valid_mask=motor_rps_valid_mask,
        motor_rps_correction_mask=motor_rps_correction_mask,
        motor_rps_age_us=motor_rps_age_us,
    )


def compute_metrics(estimate_rps, measured_rps):
    error = estimate_rps - measured_rps
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    correlation = np.array(
        [np.corrcoef(estimate_rps[:, motor_index], measured_rps[:, motor_index])[0, 1] for motor_index in range(NUM_MOTORS)]
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(mae)),
        "correlation_mean": float(np.mean(correlation)),
    }


def format_metric_row(label, metrics):
    return (
        f"{label:<33}"
        f" rmse_mean={metrics['rmse_mean']:6.2f}"
        f" mae_mean={metrics['mae_mean']:6.2f}"
        f" corr_mean={metrics['correlation_mean']:6.3f}"
    )


def build_metrics_text(metrics):
    lines = [
        format_metric_row("simulated plant", metrics),
        "",
    ]
    for motor_index in range(NUM_MOTORS):
        lines.append(
            f"motor {motor_index}: "
            f"rmse={metrics['rmse'][motor_index]:6.2f} "
            f"mae={metrics['mae'][motor_index]:6.2f} "
            f"corr={metrics['correlation'][motor_index]:6.3f}"
        )
    return "\n".join(lines)


def build_interactive_plot(run_analysis):
    default_drive_gain = 1.0 / run_analysis.logged_tau_s
    default_linear_damping = 1.0 / run_analysis.logged_tau_s
    default_quadratic_drag = 5.0e-3
    default_transition_resistance = 0.0
    default_transition_eps_rps = 10.0

    model_state = {
        "model_name": "first_order",
        "tau_s": run_analysis.logged_tau_s,
        "tau_up_s": run_analysis.logged_tau_s,
        "tau_down_s": run_analysis.logged_tau_s,
        "eps_rps": 1.0,
        "drive_gain": default_drive_gain,
        "linear_damping": default_linear_damping,
        "quadratic_drag": default_quadratic_drag,
        "transition_resistance": default_transition_resistance,
        "transition_eps_rps": default_transition_eps_rps,
    }
    interaction_state = {
        "programmatic_update": False,
    }

    simulated_rps = run_analysis.recomputed_open_loop_rps.copy()
    simulated_metrics = compute_metrics(simulated_rps, run_analysis.measured_rps_aligned)

    figure, axis = plt.subplots(figsize=(15.0, 8.0))
    plt.subplots_adjust(left=0.08, right=0.82, bottom=0.30, top=0.88)

    axis.set_title("Measured motor RPS vs simulated plant motor RPS")
    axis.set_xlabel("Solver time [s]")
    axis.set_ylabel("Motor speed [RPS]")
    axis.grid(True, alpha=0.3)

    source_handles = [
        axis.plot([], [], color="black", linewidth=1.8, linestyle="-", label="measured")[0],
        axis.plot([], [], color="black", linewidth=1.2, linestyle="--", label="simulated plant")[0],
    ]

    motor_lines = {}
    for motor_index in range(NUM_MOTORS):
        measured_line, = axis.plot(
            run_analysis.solver_time_s,
            run_analysis.measured_rps_aligned[:, motor_index],
            color=MOTOR_COLORS[motor_index],
            linewidth=1.8,
            linestyle="-",
        )
        simulated_line, = axis.plot(
            run_analysis.solver_time_s,
            simulated_rps[:, motor_index],
            color=MOTOR_COLORS[motor_index],
            linewidth=1.2,
            linestyle="--",
        )
        motor_lines[motor_index] = (measured_line, simulated_line)

    axis.legend(handles=source_handles, loc="upper left")

    model_axis = figure.add_axes([0.10, 0.91, 0.22, 0.06])
    model_buttons = RadioButtons(model_axis, ("first_order", "smooth_asymmetric", "quadratic_drag"))
    model_axis.set_title("Motor model")
    optimize_axis = figure.add_axes([0.36, 0.915, 0.16, 0.045])
    optimize_button = Button(optimize_axis, "cma-es 10x10")

    button_axis = figure.add_axes([0.845, 0.48, 0.13, 0.22])
    check_buttons = CheckButtons(
        button_axis,
        [f"motor {motor_index}" for motor_index in range(NUM_MOTORS)],
        [True] * NUM_MOTORS,
    )
    button_axis.set_title("Visible")

    tau_slider_axis = figure.add_axes([0.10, 0.14, 0.62, 0.035])
    tau_slider_min = max(run_analysis.logged_tau_s * 0.1, 1.0e-4)
    tau_slider_max = run_analysis.logged_tau_s * 5.0
    tau_slider = Slider(
        tau_slider_axis,
        "motor constant [s]",
        tau_slider_min,
        tau_slider_max,
        valinit=run_analysis.logged_tau_s,
        valfmt="%.6f",
    )

    tau_up_slider_axis = figure.add_axes([0.10, 0.09, 0.62, 0.035])
    tau_up_slider = Slider(
        tau_up_slider_axis,
        "tau up [s]",
        tau_slider_min,
        tau_slider_max,
        valinit=run_analysis.logged_tau_s,
        valfmt="%.6f",
    )

    tau_down_slider_axis = figure.add_axes([0.10, 0.04, 0.62, 0.035])
    tau_down_slider = Slider(
        tau_down_slider_axis,
        "tau down [s]",
        tau_slider_min,
        tau_slider_max,
        valinit=run_analysis.logged_tau_s,
        valfmt="%.6f",
    )

    input_gain_slider_axis = figure.add_axes([0.10, 0.20, 0.62, 0.03])
    input_gain_slider = Slider(
        input_gain_slider_axis,
        "drive gain k0 [1/s]",
        max(default_drive_gain * 0.1, 1.0e-4),
        default_drive_gain * 5.0,
        valinit=default_drive_gain,
        valfmt="%.6f",
    )

    linear_friction_slider_axis = figure.add_axes([0.10, 0.155, 0.62, 0.03])
    linear_friction_slider = Slider(
        linear_friction_slider_axis,
        "linear damping k1 [1/s]",
        0.0,
        default_linear_damping * 5.0,
        valinit=default_linear_damping,
        valfmt="%.6f",
    )

    quadratic_drag_slider_axis = figure.add_axes([0.10, 0.11, 0.62, 0.03])
    quadratic_drag_slider = Slider(
        quadratic_drag_slider_axis,
        "quadratic drag k2 [1/rps s]",
        0.0,
        max(default_quadratic_drag * 50.0, 1.0e-2),
        valinit=default_quadratic_drag,
        valfmt="%.6f",
    )

    transition_resistance_slider_axis = figure.add_axes([0.10, 0.065, 0.62, 0.03])
    transition_resistance_slider = Slider(
        transition_resistance_slider_axis,
        "transition resistance k3 [rps/s]",
        0.0,
        max(default_drive_gain * np.max(run_analysis.solver_prev_command_rps), 1.0),
        valinit=default_transition_resistance,
        valfmt="%.6f",
    )

    transition_eps_slider_axis = figure.add_axes([0.10, 0.02, 0.62, 0.03])
    transition_eps_slider = Slider(
        transition_eps_slider_axis,
        "transition smooth eps [rps]",
        0.1,
        100.0,
        valinit=default_transition_eps_rps,
        valfmt="%.6f",
    )

    info_text = figure.text(
        0.08,
        0.95,
        (
            f"run = {run_analysis.input_dir}\n"
            f"alignment offset = {run_analysis.motor_to_solver_offset_s:.6f} s   "
            f"alignment score = {run_analysis.motor_to_solver_score:.6f}\n"
            f"logged motor constant = {run_analysis.logged_tau_s:.6f} s"
        ),
        ha="left",
        va="top",
        family="monospace",
    )
    metrics_text = figure.text(
        0.845,
        0.38,
        build_metrics_text(simulated_metrics),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10.0,
    )

    def set_model_control_visibility():
        use_first_order = model_state["model_name"] == "first_order"
        use_smooth_asymmetric = model_state["model_name"] == "smooth_asymmetric"
        use_quadratic_drag = model_state["model_name"] == "quadratic_drag"

        tau_slider_axis.set_visible(use_first_order)
        tau_up_slider_axis.set_visible(use_smooth_asymmetric)
        tau_down_slider_axis.set_visible(use_smooth_asymmetric)
        input_gain_slider_axis.set_visible(use_quadratic_drag)
        linear_friction_slider_axis.set_visible(use_quadratic_drag)
        quadratic_drag_slider_axis.set_visible(use_quadratic_drag)
        transition_resistance_slider_axis.set_visible(use_quadratic_drag)
        transition_eps_slider_axis.set_visible(use_quadratic_drag)

    def compute_selected_model_rps():
        if model_state["model_name"] == "first_order":
            return recompute_estimated_rps(
                run_analysis.solver_sample_timestamp_us,
                run_analysis.solver_prev_command_rps,
                run_analysis.logged_estimated_rps[0],
                model_state["tau_s"],
            )
        if model_state["model_name"] == "smooth_asymmetric":
            return recompute_estimated_rps_smooth_asymmetric(
                run_analysis.solver_sample_timestamp_us,
                run_analysis.solver_prev_command_rps,
                run_analysis.logged_estimated_rps[0],
                model_state["tau_up_s"],
                model_state["tau_down_s"],
                model_state["eps_rps"],
            )
        return recompute_estimated_rps_quadratic_drag(
            run_analysis.solver_sample_timestamp_us,
            run_analysis.solver_prev_command_rps,
            run_analysis.logged_estimated_rps[0],
            model_state["drive_gain"],
            model_state["linear_damping"],
            model_state["quadratic_drag"],
            model_state["transition_resistance"],
            model_state["transition_eps_rps"],
        )

    def build_info_text():
        if model_state["model_name"] == "first_order":
            return (
                f"run = {run_analysis.input_dir}\n"
                f"alignment offset = {run_analysis.motor_to_solver_offset_s:.6f} s   "
                f"alignment score = {run_analysis.motor_to_solver_score:.6f}\n"
                f"logged motor constant = {run_analysis.logged_tau_s:.6f} s   "
                f"selected first-order tau = {model_state['tau_s']:.6f} s"
            )
        if model_state["model_name"] == "smooth_asymmetric":
            return (
                f"run = {run_analysis.input_dir}\n"
                f"alignment offset = {run_analysis.motor_to_solver_offset_s:.6f} s   "
                f"alignment score = {run_analysis.motor_to_solver_score:.6f}\n"
                f"logged motor constant = {run_analysis.logged_tau_s:.6f} s   "
                f"selected smooth-asymmetric tau_up = {model_state['tau_up_s']:.6f} s   "
                f"tau_down = {model_state['tau_down_s']:.6f} s   "
                f"eps = {model_state['eps_rps']:.3f} RPS"
            )
        return (
            f"run = {run_analysis.input_dir}\n"
            f"alignment offset = {run_analysis.motor_to_solver_offset_s:.6f} s   "
            f"alignment score = {run_analysis.motor_to_solver_score:.6f}\n"
            f"logged motor constant = {run_analysis.logged_tau_s:.6f} s   "
            f"selected quadratic model k0 = {model_state['drive_gain']:.6f}  "
            f"k1 = {model_state['linear_damping']:.6f}  "
            f"k2 = {model_state['quadratic_drag']:.6f}  "
            f"k3 = {model_state['transition_resistance']:.6f}  "
            f"eps = {model_state['transition_eps_rps']:.6f}"
        )

    def refresh_simulation():
        updated_rps = compute_selected_model_rps()
        updated_metrics = compute_metrics(updated_rps, run_analysis.measured_rps_aligned)
        info_text.set_text(build_info_text())
        metrics_text.set_text(build_metrics_text(updated_metrics))
        for motor_index in range(NUM_MOTORS):
            motor_lines[motor_index][1].set_ydata(updated_rps[:, motor_index])
        figure.canvas.draw_idle()

    def sync_visibility(_label):
        for motor_index, active in enumerate(check_buttons.get_status()):
            for line in motor_lines[motor_index]:
                line.set_visible(active)
        figure.canvas.draw_idle()

    def set_slider_values(slider_value_pairs):
        interaction_state["programmatic_update"] = True
        try:
            for slider, value in slider_value_pairs:
                slider.set_val(value)
        finally:
            interaction_state["programmatic_update"] = False

    def update_model_parameters_from_sliders():
        model_state["tau_s"] = tau_slider.val
        model_state["tau_up_s"] = tau_up_slider.val
        model_state["tau_down_s"] = tau_down_slider.val
        model_state["drive_gain"] = input_gain_slider.val
        model_state["linear_damping"] = linear_friction_slider.val
        model_state["quadratic_drag"] = quadratic_drag_slider.val
        model_state["transition_resistance"] = transition_resistance_slider.val
        model_state["transition_eps_rps"] = transition_eps_slider.val

    def score_candidate(candidate_rps):
        metrics = compute_metrics(candidate_rps, run_analysis.measured_rps_aligned)
        return metrics["rmse_mean"]

    def build_model_objective(candidate_vector):
        if model_state["model_name"] == "first_order":
            candidate_rps = recompute_estimated_rps(
                run_analysis.solver_sample_timestamp_us,
                run_analysis.solver_prev_command_rps,
                run_analysis.logged_estimated_rps[0],
                float(candidate_vector[0]),
            )
            return score_candidate(candidate_rps)
        if model_state["model_name"] == "smooth_asymmetric":
            candidate_rps = recompute_estimated_rps_smooth_asymmetric(
                run_analysis.solver_sample_timestamp_us,
                run_analysis.solver_prev_command_rps,
                run_analysis.logged_estimated_rps[0],
                float(candidate_vector[0]),
                float(candidate_vector[1]),
                model_state["eps_rps"],
            )
            return score_candidate(candidate_rps)
        candidate_rps = recompute_estimated_rps_quadratic_drag(
            run_analysis.solver_sample_timestamp_us,
            run_analysis.solver_prev_command_rps,
            run_analysis.logged_estimated_rps[0],
            float(candidate_vector[0]),
            float(candidate_vector[1]),
            float(candidate_vector[2]),
            float(candidate_vector[3]),
            float(candidate_vector[4]),
        )
        return score_candidate(candidate_rps)

    def optimize_current_model(_event):
        if model_state["model_name"] == "first_order":
            sliders = [tau_slider]
        elif model_state["model_name"] == "smooth_asymmetric":
            sliders = [tau_up_slider, tau_down_slider]
        else:
            sliders = [
                input_gain_slider,
                linear_friction_slider,
                quadratic_drag_slider,
                transition_resistance_slider,
                transition_eps_slider,
            ]

        x0 = [slider.val for slider in sliders]
        lower_bounds = [slider.valmin for slider in sliders]
        upper_bounds = [slider.valmax for slider in sliders]
        slider_ranges = [slider.valmax - slider.valmin for slider in sliders]
        sigma0 = 0.2 * max(slider_ranges)
        if not sigma0 > 0.0:
            raise ValueError("cma-es requires slider ranges with positive width")

        es = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                "bounds": [lower_bounds, upper_bounds],
                "CMA_stds": [0.5 * slider_range for slider_range in slider_ranges],
                "popsize": 20,
                "verbose": -9,
            },
        )

        for _ in range(20):
            candidates = es.ask()
            scores = [build_model_objective(candidate) for candidate in candidates]
            es.tell(candidates, scores)

        best_values = [(slider, value) for slider, value in zip(sliders, es.result.xbest)]
        set_slider_values(best_values)
        update_model_parameters_from_sliders()
        refresh_simulation()

    def update_model(model_name):
        model_state["model_name"] = model_name
        set_model_control_visibility()
        refresh_simulation()

    def update_tau(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["tau_s"] = tau_slider.val
        if model_state["model_name"] == "first_order":
            refresh_simulation()

    def update_tau_up(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["tau_up_s"] = tau_up_slider.val
        if model_state["model_name"] == "smooth_asymmetric":
            refresh_simulation()

    def update_tau_down(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["tau_down_s"] = tau_down_slider.val
        if model_state["model_name"] == "smooth_asymmetric":
            refresh_simulation()

    def update_input_gain(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["drive_gain"] = input_gain_slider.val
        if model_state["model_name"] == "quadratic_drag":
            refresh_simulation()

    def update_linear_friction(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["linear_damping"] = linear_friction_slider.val
        if model_state["model_name"] == "quadratic_drag":
            refresh_simulation()

    def update_quadratic_drag(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["quadratic_drag"] = quadratic_drag_slider.val
        if model_state["model_name"] == "quadratic_drag":
            refresh_simulation()

    def update_transition_resistance(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["transition_resistance"] = transition_resistance_slider.val
        if model_state["model_name"] == "quadratic_drag":
            refresh_simulation()

    def update_transition_eps(_value):
        if interaction_state["programmatic_update"]:
            return
        model_state["transition_eps_rps"] = transition_eps_slider.val
        if model_state["model_name"] == "quadratic_drag":
            refresh_simulation()

    check_buttons.on_clicked(sync_visibility)
    model_buttons.on_clicked(update_model)
    optimize_button.on_clicked(optimize_current_model)
    tau_slider.on_changed(update_tau)
    tau_up_slider.on_changed(update_tau_up)
    tau_down_slider.on_changed(update_tau_down)
    input_gain_slider.on_changed(update_input_gain)
    linear_friction_slider.on_changed(update_linear_friction)
    quadratic_drag_slider.on_changed(update_quadratic_drag)
    transition_resistance_slider.on_changed(update_transition_resistance)
    transition_eps_slider.on_changed(update_transition_eps)

    set_model_control_visibility()
    info_text.set_text(build_info_text())

    figure._motor_response_model_buttons = model_buttons
    figure._motor_response_optimize_button = optimize_button
    figure._motor_response_check_buttons = check_buttons
    figure._motor_response_tau_slider = tau_slider
    figure._motor_response_tau_up_slider = tau_up_slider
    figure._motor_response_tau_down_slider = tau_down_slider
    figure._motor_response_input_gain_slider = input_gain_slider
    figure._motor_response_linear_friction_slider = linear_friction_slider
    figure._motor_response_quadratic_drag_slider = quadratic_drag_slider
    figure._motor_response_transition_resistance_slider = transition_resistance_slider
    figure._motor_response_transition_eps_slider = transition_eps_slider
    figure._motor_response_lines = motor_lines
    figure._motor_response_info_text = info_text
    figure._motor_response_metrics_text = metrics_text

    return figure


def print_run_summary(label, analysis, metrics):
    print(f"{label}: {analysis.input_dir}")
    print(f"  solver_log={analysis.solver_log_path}")
    print(f"  motor_log={analysis.motor_log_path}")
    print(f"  motor_to_solver_offset_s={analysis.motor_to_solver_offset_s:.6f}")
    print(f"  motor_to_solver_alignment_score={analysis.motor_to_solver_score:.6f}")
    print(f"  logged_motor_constant_s={analysis.logged_tau_s:.6f}")
    print(f"  rmse_mean_rps={metrics['rmse_mean']:.6f}")
    print(f"  mae_mean_rps={metrics['mae_mean']:.6f}")
    print(f"  correlation_mean={metrics['correlation_mean']:.6f}")


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    current_input_dir = resolve_input_dir(experiment_dir)
    current_analysis = analyze_run(current_input_dir)
    simulated_metrics = compute_metrics(current_analysis.recomputed_open_loop_rps, current_analysis.measured_rps_aligned)

    print_run_summary("current_run_simulated_plant", current_analysis, simulated_metrics)

    figure = build_interactive_plot(current_analysis)
    figure.canvas.manager.set_window_title("analyze_motor_response")
    plt.show()


if __name__ == "__main__":
    main()
