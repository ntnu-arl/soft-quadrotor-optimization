from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_CSV_NAME = "morphy_tracking_error_per_ax_in_circle.csv"
PROCESSING_START_TIME_S = 15.0
PROGRAMMABLE_SOFT_QUAD_SMOOTHING_WINDOW = 20
USE_ABSOLUTE_ERROR = True
SHIFT_PROGRAMABLE_TIME = -1.2
Y_TICKS = [3, 6]
AXES = ("x", "y", "z")
DISPLAY_NAME_BY_DRONE = {
    "Soft Morphy": "Morphy",
    "Morphy": "Morphy",
    "Programmable Soft Quad": "Programmable Soft Quad",
}
LINE_STYLE_BY_DRONE = {
    "Morphy": {"color": "#8a8a8a", "linestyle": "--", "linewidth": 2.0},
    "Programmable Soft Quad": {"color": "#355c7d", "linestyle": "-", "linewidth": 2.2},
}
PLOTTED_DRONE_NAMES = (
    "Morphy",
    "Programmable Soft Quad",
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "xtick.labelsize": 10.4,
        "ytick.labelsize": 10.4,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def compute_rmse(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot compute RMSE with zero samples")
    return float(np.sqrt(np.mean(np.square(array))))


def compute_centered_moving_average(values, window_size):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot smooth zero samples")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    return (
        pd.Series(array)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def maybe_convert_to_absolute_error(values):
    array = np.asarray(values, dtype=float)
    if USE_ABSOLUTE_ERROR:
        return np.abs(array)
    return array


def convert_meters_to_centimeters(values):
    array = np.asarray(values, dtype=float)
    return 100.0 * array


def load_baseline_metrics(baseline_csv_path):
    baseline_df = pd.read_csv(baseline_csv_path)
    required_columns = {"drone_type", "time", "x", "y", "z"}
    missing_columns = required_columns.difference(baseline_df.columns)
    if missing_columns:
        raise ValueError(
            f"baseline CSV {baseline_csv_path} is missing required columns: {sorted(missing_columns)}"
        )

    durations_s = []
    trajectories = {}
    for drone_type, group_df in baseline_df.groupby("drone_type", sort=False):
        group_df = group_df.sort_values("time").reset_index(drop=True)
        time_values = group_df["time"].to_numpy(dtype=float)
        duration_s = float(np.max(time_values) - np.min(time_values))
        durations_s.append(duration_s)
        elapsed_time_s = time_values - time_values[0]
        display_name = DISPLAY_NAME_BY_DRONE.get(drone_type, drone_type)
        trajectories[display_name] = {
            "time_s": elapsed_time_s,
            **{
                axis_name: convert_meters_to_centimeters(
                    maybe_convert_to_absolute_error(group_df[axis_name].to_numpy(dtype=float))
                )
                for axis_name in AXES
            },
        }

    if not durations_s:
        raise ValueError(f"baseline CSV {baseline_csv_path} does not contain any drone groups")

    return trajectories, min(durations_s)


def load_programmable_soft_quad_metrics(experiment_dir, window_start_s, window_duration_s):
    solver_csv_path = experiment_dir / "nmpc_solver_data_combined_normalized.csv"
    solver_df = pd.read_csv(solver_csv_path)
    required_columns = {"time_s", "px", "py", "pz", "sp_px", "sp_py", "sp_pz"}
    missing_columns = required_columns.difference(solver_df.columns)
    if missing_columns:
        raise ValueError(
            f"solver CSV {solver_csv_path} is missing required columns: {sorted(missing_columns)}"
        )

    start_candidates_df = solver_df.loc[solver_df["time_s"] >= window_start_s].copy()
    if start_candidates_df.empty:
        raise ValueError(f"no solver samples found at or after {window_start_s} s for {solver_csv_path}")

    sampled_window_start_s = float(start_candidates_df["time_s"].iloc[0])
    sampled_start_index = int(start_candidates_df.index[0])
    extra_duration_s = max(0.0, -SHIFT_PROGRAMABLE_TIME)
    target_window_end_s = sampled_window_start_s + window_duration_s + extra_duration_s
    candidate_time_values = solver_df.loc[sampled_start_index:, "time_s"].to_numpy(dtype=float)
    end_offset_index = int(np.argmin(np.abs(candidate_time_values - target_window_end_s)))
    sampled_end_index = sampled_start_index + end_offset_index
    window_df = solver_df.iloc[sampled_start_index : sampled_end_index + 1].copy()

    actual_duration_s = float(window_df["time_s"].iloc[-1] - window_df["time_s"].iloc[0])
    sample_steps_s = np.diff(window_df["time_s"].to_numpy(dtype=float))
    sample_tolerance_s = float(np.max(sample_steps_s)) if sample_steps_s.size > 0 else 0.0
    if actual_duration_s + sample_tolerance_s + 1e-9 < window_duration_s:
        raise ValueError(
            f"solver window duration {actual_duration_s:.6f} s is shorter than requested {window_duration_s:.6f} s"
        )

    time_values = window_df["time_s"].to_numpy(dtype=float)
    error_df = pd.DataFrame(
        {
            "x": window_df["px"].to_numpy(dtype=float) - window_df["sp_px"].to_numpy(dtype=float),
            "y": window_df["py"].to_numpy(dtype=float) - window_df["sp_py"].to_numpy(dtype=float),
            "z": window_df["pz"].to_numpy(dtype=float) - window_df["sp_pz"].to_numpy(dtype=float),
        }
    )
    trajectory = {
        "time_s": time_values - time_values[0] + SHIFT_PROGRAMABLE_TIME,
        **{
            axis_name: convert_meters_to_centimeters(
                compute_centered_moving_average(
                    maybe_convert_to_absolute_error(error_df[axis_name].to_numpy(dtype=float)),
                    PROGRAMMABLE_SOFT_QUAD_SMOOTHING_WINDOW,
                )
            )
            for axis_name in AXES
        },
    }
    summary_metrics = {
        axis_name: compute_rmse(
            convert_meters_to_centimeters(
                maybe_convert_to_absolute_error(error_df[axis_name].to_numpy(dtype=float))
            )
        )
        for axis_name in AXES
    }
    return trajectory, summary_metrics, float(window_df["time_s"].iloc[0]), float(window_df["time_s"].iloc[-1])


def build_plot(trajectories_by_drone, output_png_path, output_pdf_path, window_start_s, window_end_s, baseline_duration_s):
    drone_names = [drone_name for drone_name in PLOTTED_DRONE_NAMES if drone_name in trajectories_by_drone]
    if not drone_names:
        raise ValueError("no plotted drone trajectories are available")
    fig, axes = plt.subplots(3, 1, figsize=(3.1, 3.45), sharex=True, constrained_layout=True)
    ylabel_prefix = "Error"
    y_values = []
    for drone_name in drone_names:
        trajectory = trajectories_by_drone[drone_name]
        for axis_name in AXES:
            y_values.append(np.asarray(trajectory[axis_name], dtype=float))
    if not y_values:
        raise ValueError("no y-values available for plotting")
    shared_y_min = min(float(np.min(values)) for values in y_values)
    shared_y_max = max(float(np.max(values)) for values in y_values)
    if np.isclose(shared_y_min, shared_y_max):
        y_padding = 0.5 if np.isclose(shared_y_min, 0.0) else 0.05 * abs(shared_y_min)
    else:
        y_padding = 0.05 * (shared_y_max - shared_y_min)
    shared_ylim = (shared_y_min - y_padding, shared_y_max + y_padding)

    for axis_index, axis_name in enumerate(AXES):
        subplot = axes[axis_index]
        for drone_name in drone_names:
            trajectory = trajectories_by_drone[drone_name]
            line_style = LINE_STYLE_BY_DRONE[drone_name]
            subplot.plot(
                trajectory["time_s"],
                trajectory[axis_name],
                color=line_style["color"],
                linestyle=line_style["linestyle"],
                linewidth=line_style["linewidth"],
                label=drone_name,
            )
        subplot.set_ylabel(f"{ylabel_prefix} {axis_name} [cm]")
        subplot.set_ylim(shared_ylim)
        subplot.set_yticks(Y_TICKS)
        subplot.grid(axis="y", alpha=0.3)
        subplot.set_axisbelow(True)

    axes[2].legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.03),
        frameon=False,
        handlelength=2.2,
        borderpad=0.2,
        labelspacing=0.3,
    )
    axes[-1].set_xlabel("Time [s]")
    fig.savefig(output_png_path, dpi=200)
    fig.savefig(output_pdf_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare circle-trajectory per-axis RMSE for Morphy baselines and the Programmable Soft Quad."
    )
    parser.add_argument(
        "experiment_dir",
        help="Experiment directory containing nmpc_solver_data_combined_normalized.csv",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    baseline_csv_path = Path(__file__).resolve().with_name(BASELINE_CSV_NAME)

    baseline_trajectories, baseline_duration_s = load_baseline_metrics(baseline_csv_path)
    programmable_trajectory, programmable_summary_metrics, window_start_s, window_end_s = load_programmable_soft_quad_metrics(
        experiment_dir,
        PROCESSING_START_TIME_S,
        baseline_duration_s,
    )

    trajectories_by_drone = dict(baseline_trajectories)
    trajectories_by_drone["Programmable Soft Quad"] = programmable_trajectory

    output_png_path = experiment_dir / "plot_vs_morphy_rmse_per_axis_circle.png"
    output_pdf_path = experiment_dir / "plot_vs_morphy_rmse_per_axis_circle.pdf"
    build_plot(
        trajectories_by_drone,
        output_png_path,
        output_pdf_path,
        window_start_s,
        window_end_s,
        baseline_duration_s,
    )

    metrics_df = pd.DataFrame(index=list(trajectories_by_drone.keys()), columns=list(AXES), dtype=float)
    metrics_df.loc["Programmable Soft Quad", list(AXES)] = [
        programmable_summary_metrics[axis_name] for axis_name in AXES
    ]
    for drone_name in baseline_trajectories:
        metrics_df.loc[drone_name, list(AXES)] = [
            compute_rmse(trajectories_by_drone[drone_name][axis_name]) for axis_name in AXES
        ]
    print("RMSE summary over plotted window:")
    print(metrics_df.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved {output_png_path}")
    print(f"Saved {output_pdf_path}")


if __name__ == "__main__":
    main()
