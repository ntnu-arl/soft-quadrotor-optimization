#!/usr/bin/env python3

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rotor_dynamics import manufacturerComponentData


@dataclass
class StepWindow:
    start_index: int
    end_index: int
    direction: str
    target_rpm: float
    initial_rpm: float
    time_s: np.ndarray
    actual_rpm: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive tuning for an asymmetric second-order motor model.")
    parser.add_argument("--csv", required=True, help="CSV file produced by the ESC step-response run.")
    parser.add_argument("--esc-id", required=True, type=int, help="ESC ID to analyze.")
    parser.add_argument("--step-threshold-rpm", required=True, type=float, help="Minimum target-RPM jump required to detect a step.")
    parser.add_argument("--min-step-samples", required=True, type=int, help="Minimum number of samples required to keep a detected step.")
    return parser.parse_args()


def load_step_response(csv_path, esc_id):
    rows = []
    with open(csv_path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if int(row["esc_id"]) != esc_id:
                continue
            rows.append(
                (
                    float(row["time_s"]),
                    float(row["target_rpm"]),
                    float(row["actual_rpm"]),
                )
            )
    if not rows:
        raise ValueError(f"No samples found for esc_id={esc_id} in {csv_path}.")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 2]


def extract_step_windows(time_s, target_rpm, actual_rpm, step_threshold_rpm, min_step_samples):
    step_change_indices = np.where(np.abs(np.diff(target_rpm)) >= step_threshold_rpm)[0]
    if len(step_change_indices) == 0:
        raise ValueError("No target-RPM steps detected.")

    windows = []
    for step_index_position, step_index in enumerate(step_change_indices):
        start_index = step_index + 1
        if step_index_position + 1 == len(step_change_indices):
            end_index = len(time_s)
        else:
            end_index = step_change_indices[step_index_position + 1] + 1
        if end_index - start_index < min_step_samples:
            continue

        target_before = target_rpm[step_index]
        target_after = target_rpm[start_index]
        direction = "up" if target_after > target_before else "down"
        initial_rpm = float(actual_rpm[step_index])
        time_window = time_s[start_index:end_index] - time_s[step_index]
        actual_window = actual_rpm[start_index:end_index]
        windows.append(
            StepWindow(
                start_index=start_index,
                end_index=end_index,
                direction=direction,
                target_rpm=float(target_after),
                initial_rpm=initial_rpm,
                time_s=time_window,
                actual_rpm=actual_window,
            )
        )

    if not windows:
        raise ValueError("Step detection succeeded, but all windows were filtered out.")
    return windows


def simulate_first_order_step(step_window, tau):
    predicted_rpm = np.empty_like(step_window.actual_rpm)
    current_rpm = step_window.initial_rpm
    previous_time = 0.0
    for sample_index, current_time in enumerate(step_window.time_s):
        dt = current_time - previous_time
        mixing_factor = 1.0 - np.exp(-dt / tau)
        current_rpm = current_rpm + mixing_factor * (step_window.target_rpm - current_rpm)
        predicted_rpm[sample_index] = current_rpm
        previous_time = current_time
    return predicted_rpm


def simulate_asymmetric_second_order_step(step_window, natural_freq_up, natural_freq_down, damping_ratio_up, damping_ratio_down):
    predicted_rpm = np.empty_like(step_window.actual_rpm)
    current_rpm = step_window.initial_rpm
    current_rpm_rate = 0.0
    previous_time = 0.0
    for sample_index, current_time in enumerate(step_window.time_s):
        dt = current_time - previous_time
        if dt <= 0.0:
            predicted_rpm[sample_index] = current_rpm
            continue
        error_rpm = step_window.target_rpm - current_rpm
        if error_rpm >= 0.0:
            natural_freq = natural_freq_up
            damping_ratio = damping_ratio_up
        else:
            natural_freq = natural_freq_down
            damping_ratio = damping_ratio_down
        rpm_acceleration = natural_freq ** 2 * error_rpm - 2.0 * damping_ratio * natural_freq * current_rpm_rate
        current_rpm_rate = current_rpm_rate + rpm_acceleration * dt
        current_rpm = current_rpm + current_rpm_rate * dt
        predicted_rpm[sample_index] = current_rpm
        previous_time = current_time
    return predicted_rpm


def evaluate_first_order(windows, tau):
    predictions = [simulate_first_order_step(step_window, tau) for step_window in windows]
    residuals = np.concatenate(
        [predicted_rpm - step_window.actual_rpm for step_window, predicted_rpm in zip(windows, predictions)]
    )
    return predictions, residuals


def evaluate_asymmetric_second_order(windows, params):
    natural_freq_up, natural_freq_down, damping_ratio_up, damping_ratio_down = params
    predictions = [
        simulate_asymmetric_second_order_step(
            step_window,
            natural_freq_up,
            natural_freq_down,
            damping_ratio_up,
            damping_ratio_down,
        )
        for step_window in windows
    ]
    residuals = np.concatenate(
        [predicted_rpm - step_window.actual_rpm for step_window, predicted_rpm in zip(windows, predictions)]
    )
    return predictions, residuals


def extract_transition_residuals(windows, predictions, transition_horizon_s):
    transition_residuals = []
    for step_window, predicted_rpm in zip(windows, predictions):
        in_transition = step_window.time_s <= transition_horizon_s
        if not np.any(in_transition):
            raise ValueError(
                f"Step window starting at index {step_window.start_index} has no samples within {transition_horizon_s} s."
            )
        transition_residuals.append(predicted_rpm[in_transition] - step_window.actual_rpm[in_transition])
    return np.concatenate(transition_residuals)


def compute_transition_error_area(step_window, predicted_rpm, transition_horizon_s):
    in_transition = step_window.time_s <= transition_horizon_s
    if not np.any(in_transition):
        raise ValueError(
            f"Step window starting at index {step_window.start_index} has no samples within {transition_horizon_s} s."
        )

    transition_time = step_window.time_s[in_transition]
    transition_abs_error = np.abs(predicted_rpm[in_transition] - step_window.actual_rpm[in_transition])
    if transition_time[0] > 0.0:
        transition_time = np.concatenate(([0.0], transition_time))
        transition_abs_error = np.concatenate(([transition_abs_error[0]], transition_abs_error))
    return float(np.trapz(transition_abs_error, transition_time))


def build_full_time_prediction(time_s, actual_rpm, windows, step_predictions):
    full_prediction = np.full_like(actual_rpm, np.nan)
    for step_window, predicted_rpm in zip(windows, step_predictions):
        full_prediction[step_window.start_index:step_window.end_index] = predicted_rpm
    for index in range(len(full_prediction)):
        if np.isnan(full_prediction[index]):
            full_prediction[index] = actual_rpm[index]
    return full_prediction


def build_interactive_plot(
    time_s,
    target_rpm,
    actual_rpm,
    baseline_prediction,
    windows,
    baseline_steps,
    initial_tau,
    initial_second_order_params,
):
    tuned_model_order = "second_order"
    fitted_steps, _ = evaluate_asymmetric_second_order(windows, initial_second_order_params)
    fitted_prediction = build_full_time_prediction(time_s, actual_rpm, windows, fitted_steps)

    figure, axes = plt.subplots(2, 1, figsize=(13, 10))
    figure.subplots_adjust(left=0.1, right=0.78, top=0.95, bottom=0.32, hspace=0.32)

    axes[0].plot(time_s, target_rpm, color="black", linestyle="--", label="target_rpm")
    axes[0].plot(time_s, actual_rpm, color="royalblue", label="actual_rpm")
    axes[0].plot(time_s, baseline_prediction, color="firebrick", label="simulation_first_order")
    fitted_full_line, = axes[0].plot(
        time_s,
        fitted_prediction,
        color="seagreen",
        label="manual_asymmetric_second_order",
    )
    axes[0].set_title("Full run")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("rpm")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    step_line_groups = []
    for step_window, baseline_step_prediction, fitted_step_prediction in zip(windows, baseline_steps, fitted_steps):
        actual_color = "royalblue" if step_window.direction == "up" else "darkorange"
        actual_line, = axes[1].plot(step_window.time_s, step_window.actual_rpm, color=actual_color, alpha=0.7)
        baseline_line, = axes[1].plot(step_window.time_s, baseline_step_prediction, color="firebrick", linestyle=":", alpha=0.8)
        fitted_line, = axes[1].plot(step_window.time_s, fitted_step_prediction, color="seagreen", alpha=0.8)
        step_line_groups.append((actual_line, baseline_line, fitted_line))
    axes[1].set_title("Aligned step windows")
    axes[1].set_xlabel("time since step [s]")
    axes[1].set_ylabel("rpm")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    model_order_axis = figure.add_axes([0.81, 0.83, 0.17, 0.09])
    tau_axis = figure.add_axes([0.1, 0.235, 0.82, 0.03])
    natural_freq_up_axis = figure.add_axes([0.1, 0.19, 0.82, 0.03])
    natural_freq_down_axis = figure.add_axes([0.1, 0.145, 0.82, 0.03])
    damping_ratio_up_axis = figure.add_axes([0.1, 0.10, 0.82, 0.03])
    damping_ratio_down_axis = figure.add_axes([0.1, 0.055, 0.82, 0.03])
    checkbox_axis_height = min(0.78, 0.035 * len(windows) + 0.02)
    checkbox_axis = figure.add_axes([0.81, 0.40, 0.17, checkbox_axis_height])

    model_order_buttons = RadioButtons(model_order_axis, ("first_order", "second_order"), active=1)
    tau_slider = Slider(tau_axis, "tau [s]", 0.001, 1.0, valinit=initial_tau, valfmt="%.5f")
    natural_freq_up_slider = Slider(natural_freq_up_axis, "wn_up [rad/s]", 1.0, 300.0, valinit=initial_second_order_params[0], valfmt="%.3f")
    natural_freq_down_slider = Slider(natural_freq_down_axis, "wn_down [rad/s]", 1.0, 300.0, valinit=initial_second_order_params[1], valfmt="%.3f")
    damping_ratio_up_slider = Slider(damping_ratio_up_axis, "zeta_up [-]", 0.01, 3.0, valinit=initial_second_order_params[2], valfmt="%.4f")
    damping_ratio_down_slider = Slider(damping_ratio_down_axis, "zeta_down [-]", 0.01, 3.0, valinit=initial_second_order_params[3], valfmt="%.4f")
    checkbox_labels = [
        f"{step_index + 1}: {step_window.initial_rpm:.0f}->{step_window.target_rpm:.0f}"
        for step_index, step_window in enumerate(windows)
    ]
    step_checkboxes = CheckButtons(checkbox_axis, checkbox_labels, [True] * len(windows))
    model_order_axis.set_title("Tuned model", fontsize=10)
    checkbox_axis.set_title("Steps", fontsize=10)

    def update_plot(updated_steps):
        updated_prediction = build_full_time_prediction(time_s, actual_rpm, windows, updated_steps)
        fitted_full_line.set_ydata(updated_prediction)
        for (_, _, fitted_line), updated_step in zip(step_line_groups, updated_steps):
            fitted_line.set_ydata(updated_step)
        figure.canvas.draw_idle()

    def sync_slider_visibility():
        tau_axis.set_visible(tuned_model_order == "first_order")
        natural_freq_up_axis.set_visible(tuned_model_order == "second_order")
        natural_freq_down_axis.set_visible(tuned_model_order == "second_order")
        damping_ratio_up_axis.set_visible(tuned_model_order == "second_order")
        damping_ratio_down_axis.set_visible(tuned_model_order == "second_order")

    def update(_):
        if tuned_model_order == "first_order":
            updated_steps, _ = evaluate_first_order(windows, tau_slider.val)
            fitted_full_line.set_label("manual_first_order")
        else:
            current_params = np.asarray(
                [
                    natural_freq_up_slider.val,
                    natural_freq_down_slider.val,
                    damping_ratio_up_slider.val,
                    damping_ratio_down_slider.val,
                ],
                dtype=np.float64,
            )
            updated_steps, _ = evaluate_asymmetric_second_order(windows, current_params)
            fitted_full_line.set_label("manual_asymmetric_second_order")
        axes[0].legend(loc="upper left")
        update_plot(updated_steps)

    def toggle_model_order(label):
        nonlocal tuned_model_order
        tuned_model_order = label
        sync_slider_visibility()
        update(None)

    def toggle_step(label):
        step_index = checkbox_labels.index(label)
        current_visibility = step_line_groups[step_index][0].get_visible()
        next_visibility = not current_visibility
        for line in step_line_groups[step_index]:
            line.set_visible(next_visibility)
        figure.canvas.draw_idle()

    sync_slider_visibility()
    model_order_buttons.on_clicked(toggle_model_order)
    tau_slider.on_changed(update)
    natural_freq_up_slider.on_changed(update)
    natural_freq_down_slider.on_changed(update)
    damping_ratio_up_slider.on_changed(update)
    damping_ratio_down_slider.on_changed(update)
    step_checkboxes.on_clicked(toggle_step)
    plt.show()

    return {
        "model_order": tuned_model_order,
        "tau": float(tau_slider.val),
        "second_order_params": np.asarray(
            [
                natural_freq_up_slider.val,
                natural_freq_down_slider.val,
                damping_ratio_up_slider.val,
                damping_ratio_down_slider.val,
            ],
            dtype=np.float64,
        ),
    }


def main():
    args = parse_args()
    time_s, target_rpm, actual_rpm = load_step_response(args.csv, args.esc_id)
    windows = extract_step_windows(
        time_s,
        target_rpm,
        actual_rpm,
        args.step_threshold_rpm,
        args.min_step_samples,
    )

    component_data = manufacturerComponentData("cpu", [5])
    baseline_tau = float(component_data.motor_time_constant[0].item())
    initial_second_order_params = [
        float(component_data.natural_freq_up[0].item()),
        float(component_data.natural_freq_down[0].item()),
        float(component_data.damping_ratio_up[0].item()),
        float(component_data.damping_ratio_down[0].item()),
    ]

    baseline_steps, _ = evaluate_first_order(windows, baseline_tau)
    baseline_prediction = build_full_time_prediction(time_s, actual_rpm, windows, baseline_steps)
    final_params = build_interactive_plot(
        time_s,
        target_rpm,
        actual_rpm,
        baseline_prediction,
        windows,
        baseline_steps,
        baseline_tau,
        initial_second_order_params,
    )
    print(f"model_order: {final_fit['model_order']}")
    if final_fit["model_order"] == "first_order":
        print(f"motor_time_constant: {final_fit['tau']:.9f}")
    else:
        print(f"natural_freq_up: {final_fit['second_order_params'][0]:.9f}")
        print(f"natural_freq_down: {final_fit['second_order_params'][1]:.9f}")
        print(f"damping_ratio_up: {final_fit['second_order_params'][2]:.9f}")
        print(f"damping_ratio_down: {final_fit['second_order_params'][3]:.9f}")


if __name__ == "__main__":
    main()
