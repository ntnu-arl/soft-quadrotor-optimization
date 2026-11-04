"""Create optimization progress animations for behaviour-based NMPC CMA-ES runs."""

import argparse
import json
import os
import selectors
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_DIR = Path(__file__).parent
FPS = 30
VIDEO_DURATION_SECONDS = 10
FRAME_COUNT = FPS * VIDEO_DURATION_SECONDS
RENDER_TIMEOUT_SECONDS = 300
RENDER_NO_OUTPUT_TIMEOUT_SECONDS = 180
CAMERA_DISTANCE = 0.45
CAMERA_HEIGHT = 0.65
CAMERA_AZIMUTH = 45.0
TORQUE_DOMAIN_LIMIT = float(np.pi / 3.0 * 0.98)
TORQUE_SAMPLE_COUNT = 500
PHASE_SOBOL = "SOBOL"
PHASE_BO = "BO"
PHASE_CMAES = "CMAES"
PHASE_ORDER = [PHASE_SOBOL, PHASE_BO, PHASE_CMAES]
PHASE_PROGRESS_COLORS = {
    PHASE_SOBOL: "#6c757d",
    PHASE_BO: "#6c757d",
    PHASE_CMAES: "#1f77b4",
}
PHASE_PROGRESS_LABELS = {
    PHASE_SOBOL: "BO",
    PHASE_BO: "BO",
    PHASE_CMAES: "CMA-ES",
}


@dataclass(frozen=True)
class AnimationPaths:
    results_dir: Path
    trials_csv: Path
    run_config_json: Path
    frame_dir: Path
    render_dir: Path
    mp4_path: Path


def generate_f_over_time_animation(results_dir_text: str, skip_current_best: bool) -> Path:
    paths = build_animation_paths(results_dir_text)
    validate_inputs(paths)
    run_config = load_run_config(paths.run_config_json)
    df = load_trials(paths.trials_csv, run_config)
    prepared_df = prepare_progress_dataframe(df)
    torque_ylim, torque_xlim = calculate_torque_plot_limits(prepared_df)
    prepare_output_dirs(paths)
    render_rows_for_animation(prepared_df, paths, skip_current_best)
    write_animation_frames(prepared_df, paths, torque_ylim, torque_xlim)
    encode_mp4(paths)
    return paths.mp4_path


def build_animation_paths(results_dir_text: str) -> AnimationPaths:
    results_dir = Path(results_dir_text).expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir = results_dir.resolve()
    return AnimationPaths(
        results_dir=results_dir,
        trials_csv=results_dir / "trials.csv",
        run_config_json=results_dir / "run_config.json",
        frame_dir=results_dir / "f_over_time_frames",
        render_dir=results_dir / "f_over_time_renders",
        mp4_path=results_dir / "f_over_time.mp4",
    )


def validate_inputs(paths: AnimationPaths) -> None:
    if not paths.results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {paths.results_dir}")
    if not paths.trials_csv.is_file():
        raise FileNotFoundError(f"Trials CSV not found: {paths.trials_csv}")
    if not paths.run_config_json.is_file():
        raise FileNotFoundError(f"Run config JSON not found: {paths.run_config_json}")


def load_run_config(run_config_json: Path) -> dict:
    with open(run_config_json, "r") as f:
        return json.load(f)


def load_trials(trials_csv: Path, run_config: dict) -> pd.DataFrame:
    df = pd.read_csv(trials_csv)
    if len(df) == 0:
        raise ValueError(f"No rows in trials CSV: {trials_csv}")

    required_columns = [
        "generation",
        "individual_id",
        "phase",
        "success_rate",
        "mean_position_error",
        "workspace_id",
        "torque_fn_right",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required trial columns: {missing_columns}")
    unknown_phases = sorted(set(df["phase"]) - set(PHASE_ORDER))
    if unknown_phases:
        raise ValueError(f"Unknown optimization phases in trials CSV: {unknown_phases}")

    for i, _ in enumerate(run_config["param_names"]):
        column = f"x_{i}"
        if column not in df.columns:
            raise ValueError(f"Missing required optimization parameter column: {column}")

    return df


def prepare_progress_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = (
        df.sort_values(["generation", "individual_id"], kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    prepared_df["evaluation"] = np.arange(1, len(prepared_df) + 1)
    best_so_far_rows = []
    for row_index in prepared_df.index:
        best_so_far_rows.append(
            prepared_df.loc[:row_index]
            .sort_values(
                ["success_rate", "mean_position_error", "generation", "individual_id"],
                ascending=[False, True, True, True],
            )
            .iloc[0]
        )
    best_so_far_df = pd.DataFrame(best_so_far_rows).reset_index(drop=True)
    best_so_far_columns = [
        "evaluation",
        "generation",
        "individual_id",
        "phase",
        "success_rate",
        "mean_position_error",
        "workspace_id",
        "torque_fn_right",
    ]
    for column in best_so_far_columns:
        prepared_df[f"best_so_far_{column}"] = best_so_far_df[column].to_numpy()
    return prepared_df


def prepare_output_dirs(paths: AnimationPaths) -> None:
    if paths.frame_dir.exists():
        shutil.rmtree(paths.frame_dir)
    paths.frame_dir.mkdir(parents=True)
    paths.render_dir.mkdir(parents=True, exist_ok=True)


def render_rows_for_animation(
    prepared_df: pd.DataFrame,
    paths: AnimationPaths,
    skip_current_best: bool,
) -> None:
    frame_source_indices = frame_source_row_indices(prepared_df)
    best_so_far_jobs = []
    current_jobs = []
    for render_index in sorted(set(frame_source_indices)):
        row = prepared_df.loc[render_index]
        current_jobs.append(
            (
                "current evaluation",
                int(row["generation"]),
                int(row["individual_id"]),
                str(row["workspace_id"]),
                render_path_for_current_row(paths, row),
            )
        )
        best_so_far_jobs.append(
            (
                "best so far",
                int(row["best_so_far_generation"]),
                int(row["best_so_far_individual_id"]),
                str(row["best_so_far_workspace_id"]),
                render_path_for_best_so_far_row(paths, row),
            )
        )
    unique_best_so_far_jobs = list(dict.fromkeys(best_so_far_jobs))
    unique_current_jobs = list(dict.fromkeys(current_jobs))
    best_so_far_render_paths = {render_job[4] for render_job in unique_best_so_far_jobs}

    print(f"[Animation] Rendering {len(unique_best_so_far_jobs)} best-so-far airframe screenshots")
    for render_number, render_job in enumerate(unique_best_so_far_jobs, start=1):
        label, generation, individual_id, workspace_id, render_path = render_job
        if render_path.exists():
            print(
                f"[Animation] Skip render {render_number}/{len(unique_best_so_far_jobs)}: "
                f"{label}, gen {generation}, ind {individual_id}, existing {render_path}"
            )
            continue
        print(
            f"[Animation] Render {render_number}/{len(unique_best_so_far_jobs)}: "
            f"{label}, gen {generation}, ind {individual_id}"
        )
        render_workspace(workspace_id, render_path)

    if skip_current_best:
        print("[Animation] Copying current-evaluation renders from best-so-far renders")
        for row_index in sorted(set(frame_source_indices)):
            row = prepared_df.loc[row_index]
            current_path = render_path_for_current_row(paths, row)
            best_so_far_path = render_path_for_best_so_far_row(paths, row)
            if current_path in best_so_far_render_paths:
                continue
            if not best_so_far_path.is_file():
                raise FileNotFoundError(f"Best-so-far render missing before copy: {best_so_far_path}")
            shutil.copy2(best_so_far_path, current_path)
        return

    print(f"[Animation] Rendering {len(unique_current_jobs)} current-evaluation airframe screenshots")
    for render_number, render_job in enumerate(unique_current_jobs, start=1):
        label, generation, individual_id, workspace_id, render_path = render_job
        if render_path.exists():
            print(
                f"[Animation] Skip render {render_number}/{len(unique_current_jobs)}: "
                f"{label}, gen {generation}, ind {individual_id}, existing {render_path}"
            )
            continue
        print(
            f"[Animation] Render {render_number}/{len(unique_current_jobs)}: "
            f"{label}, gen {generation}, ind {individual_id}"
        )
        render_workspace(workspace_id, render_path)


def frame_source_row_indices(prepared_df: pd.DataFrame) -> list[int]:
    if len(prepared_df) <= FRAME_COUNT:
        sampled = np.linspace(0, len(prepared_df) - 1, FRAME_COUNT)
        return [int(round(value)) for value in sampled]
    sampled = np.linspace(0, len(prepared_df) - 1, FRAME_COUNT)
    return [int(round(value)) for value in sampled]


def render_path_for_candidate(paths: AnimationPaths, generation: int, individual_id: int) -> Path:
    return paths.render_dir / f"gen{generation:06d}_ind{individual_id:06d}.png"


def render_path_for_current_row(paths: AnimationPaths, row: pd.Series) -> Path:
    return render_path_for_candidate(paths, int(row["generation"]), int(row["individual_id"]))


def render_path_for_best_so_far_row(paths: AnimationPaths, row: pd.Series) -> Path:
    return render_path_for_candidate(
        paths,
        int(row["best_so_far_generation"]),
        int(row["best_so_far_individual_id"]),
    )


def workspace_dir_from_workspace_id(workspace_id: str) -> Path:
    workspace_dir = Path(workspace_id)
    if not workspace_dir.is_absolute():
        workspace_dir = PROJECT_ROOT / workspace_dir
    workspace_dir = workspace_dir.resolve()
    usd_path = workspace_dir / "morphy_prog.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"Recorded workspace USD not found: {usd_path}")
    return workspace_dir


def render_workspace(workspace_id: str, render_path: Path) -> None:
    workspace_dir = workspace_dir_from_workspace_id(workspace_id)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "_render_airframe_frame_impl.py"),
        "--workspace_dir",
        str(workspace_dir),
        "--output_path",
        str(render_path),
        "--camera_distance",
        str(CAMERA_DISTANCE),
        "--camera_height",
        str(CAMERA_HEIGHT),
        "--camera_azimuth",
        str(CAMERA_AZIMUTH),
        "--headless",
        "--enable_cameras",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_lines = []
    frame_saved = False
    deadline = time.monotonic() + RENDER_TIMEOUT_SECONDS
    no_output_deadline = time.monotonic() + RENDER_NO_OUTPUT_TIMEOUT_SECONDS
    selector = selectors.DefaultSelector()
    if process.stdout is None:
        raise RuntimeError("Render subprocess stdout pipe was not created")
    selector.register(process.stdout, selectors.EVENT_READ)

    try:
        while process.poll() is None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0.0:
                terminate_render_process(process)
                raise TimeoutError(
                    f"Airframe render timed out after {RENDER_TIMEOUT_SECONDS}s for {workspace_dir}"
                )
            if time.monotonic() >= no_output_deadline:
                terminate_render_process(process)
                raise TimeoutError(
                    f"Airframe render produced no output for {RENDER_NO_OUTPUT_TIMEOUT_SECONDS}s "
                    f"for {workspace_dir}\nOUTPUT:\n{''.join(output_lines)}"
                )

            events = selector.select(min(1.0, remaining_seconds))
            for key, _ in events:
                line = key.fileobj.readline()
                if line == "":
                    continue
                no_output_deadline = time.monotonic() + RENDER_NO_OUTPUT_TIMEOUT_SECONDS
                output_lines.append(line)
                print(line, end="")
                if "[INFO] Frame saved:" in line:
                    frame_saved = True
                    print("[Animation] Frame captured, terminating Isaac Sim renderer")
                    terminate_render_process(process)
                    break
                if "[INFO] Closing simulation app..." in line:
                    print("[Animation] Isaac Sim reached shutdown, terminating renderer process group")
                    terminate_render_process(process)
                    break

        for line in process.stdout.readlines():
            output_lines.append(line)
            print(line, end="")
    finally:
        selector.close()

    output_text = "".join(output_lines)
    if process.returncode not in [0, -15, -9] or "Traceback (most recent call last):" in output_text:
        raise RuntimeError(
            f"Airframe render failed for {workspace_dir} with return code {process.returncode}\n"
            f"OUTPUT:\n{output_text}"
        )
    if not render_path.is_file():
        raise RuntimeError(
            f"Render command completed but output PNG was not created: {render_path}\n"
            f"OUTPUT:\n{output_text}"
        )
    if not frame_saved and process.returncode in [-15, -9]:
        raise RuntimeError(
            f"Render subprocess was terminated before reporting that the frame was saved: {render_path}\n"
            f"OUTPUT:\n{output_text}"
        )
    side_effect_mp4 = render_path.with_suffix(".mp4")
    if side_effect_mp4.exists():
        side_effect_mp4.unlink()


def terminate_render_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def write_animation_frames(
    prepared_df: pd.DataFrame,
    paths: AnimationPaths,
    torque_ylim: tuple[float, float],
    torque_xlim: tuple[float, float],
) -> None:
    frame_source_indices = frame_source_row_indices(prepared_df)
    for frame_number, row_index in enumerate(frame_source_indices):
        row = prepared_df.loc[row_index]
        frame_path = paths.frame_dir / f"frame_{frame_number:06d}.png"
        write_frame(prepared_df, row, paths, frame_path, torque_ylim, torque_xlim)


def write_frame(
    prepared_df: pd.DataFrame,
    row: pd.Series,
    paths: AnimationPaths,
    frame_path: Path,
    torque_ylim: tuple[float, float],
    torque_xlim: tuple[float, float],
) -> None:
    current_image = plt.imread(render_path_for_current_row(paths, row))
    best_so_far_image = plt.imread(render_path_for_best_so_far_row(paths, row))

    fig = plt.figure(figsize=(16, 12), dpi=120)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.9, 1.0], width_ratios=[1.0, 1.0])

    ax_current = fig.add_subplot(grid[0, 0])
    ax_current.imshow(current_image)
    ax_current.set_title(
        f"Evaluation {int(row['evaluation'])}: {row['phase']}, gen {int(row['generation'])}, "
        f"ind {int(row['individual_id'])}, success {float(row['success_rate']):.1f}%",
        fontsize=12,
    )
    ax_current.axis("off")

    ax_best_so_far = fig.add_subplot(grid[0, 1])
    ax_best_so_far.imshow(best_so_far_image)
    ax_best_so_far.set_title(
        f"Best so far at evaluation {int(row['evaluation'])}: "
        f"gen {int(row['best_so_far_generation'])}, ind {int(row['best_so_far_individual_id'])}, "
        f"success {float(row['best_so_far_success_rate']):.1f}%",
        fontsize=12,
    )
    ax_best_so_far.axis("off")

    ax_progress = fig.add_subplot(grid[1, :])
    plot_progress(ax_progress, prepared_df, row)

    ax_torque = fig.add_subplot(grid[2, :])
    plot_torque_response(ax_torque, row, torque_ylim, torque_xlim)

    fig.suptitle("Behaviour-based NMPC CMA-ES optimization progress", fontsize=15)
    fig.tight_layout()
    fig.savefig(frame_path)
    plt.close(fig)


def plot_progress(ax: plt.Axes, prepared_df: pd.DataFrame, row: pd.Series) -> None:
    current_eval = int(row["evaluation"])
    visible_df = prepared_df[prepared_df["evaluation"] <= current_eval]
    shown_labels = set()
    for phase in PHASE_ORDER:
        phase_df = visible_df[visible_df["phase"] == phase]
        if len(phase_df) == 0:
            continue
        label = PHASE_PROGRESS_LABELS[phase]
        if label in shown_labels:
            label = None
        else:
            shown_labels.add(label)
        ax.scatter(
            phase_df["evaluation"],
            phase_df["success_rate"],
            s=12,
            alpha=0.45,
            color=PHASE_PROGRESS_COLORS[phase],
            label=label,
        )
    ax.plot(
        visible_df["evaluation"],
        visible_df["best_so_far_success_rate"],
        linewidth=2.5,
        color="black",
        label="Best so far",
    )
    ax.scatter(
        [current_eval],
        [float(row["success_rate"])],
        s=70,
        color="#d62828",
        zorder=5,
        label="Current",
    )
    ax.scatter(
        [float(row["best_so_far_evaluation"])],
        [float(row["best_so_far_success_rate"])],
        s=70,
        color="black",
        zorder=5,
        label="Best so far candidate",
    )
    evaluation_min = int(prepared_df["evaluation"].min())
    evaluation_max = int(prepared_df["evaluation"].max())
    if evaluation_min == evaluation_max:
        ax.set_xlim(evaluation_min - 0.5, evaluation_max + 0.5)
    else:
        ax.set_xlim(evaluation_min, evaluation_max)
    ax.set_ylim(0, max(100.0, float(prepared_df["success_rate"].max()) * 1.05))
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Fitness over time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)


def plot_torque_response(
    ax: plt.Axes, row: pd.Series, torque_ylim: tuple[float, float], torque_xlim: tuple[float, float]
) -> None:
    generation_angle_rad, generation_torque = evaluate_right_torque_curve(str(row["torque_fn_right"]))
    best_so_far_angle_rad, best_so_far_torque = evaluate_right_torque_curve(
        str(row["best_so_far_torque_fn_right"])
    )

    generation_angle_deg = np.degrees(generation_angle_rad)
    best_so_far_angle_deg = np.degrees(best_so_far_angle_rad)

    ax.plot(
        generation_angle_deg,
        generation_torque,
        color="#6c757d",
        linewidth=2.0,
        linestyle="--",
        label="Current evaluation",
    )
    ax.plot(
        best_so_far_angle_deg,
        best_so_far_torque,
        color="#006d77",
        linewidth=2.5,
        linestyle="-",
        label="Best so far",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylim(torque_ylim)
    ax.set_xlim(torque_xlim)
    ax.set_xlabel("Right arm angle x (deg)")
    ax.set_ylabel("Torque (N*m)")
    ax.set_title("Right arm torque response function")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def calculate_torque_plot_limits(prepared_df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    torque_arrays = []
    angle_arrays_deg = []
    for _, row in prepared_df.iterrows():
        generation_angle_rad, generation_torque = evaluate_right_torque_curve(
            str(row["torque_fn_right"])
        )
        best_so_far_angle_rad, best_so_far_torque = evaluate_right_torque_curve(
            str(row["best_so_far_torque_fn_right"])
        )
        torque_arrays.append(generation_torque)
        torque_arrays.append(best_so_far_torque)
        angle_arrays_deg.append(np.degrees(generation_angle_rad))
        angle_arrays_deg.append(np.degrees(best_so_far_angle_rad))

    all_torque = np.concatenate(torque_arrays)
    finite_torque = all_torque[np.isfinite(all_torque)]
    if len(finite_torque) != len(all_torque):
        raise ValueError("Torque response contains non-finite values")
    torque_min = float(finite_torque.min())
    torque_max = float(finite_torque.max())
    if torque_min == torque_max:
        torque_min -= 1.0
        torque_max += 1.0
    else:
        margin = (torque_max - torque_min) * 0.05
        torque_min -= margin
        torque_max += margin

    all_angles_deg = np.concatenate(angle_arrays_deg)
    angle_min_deg = float(all_angles_deg.min())
    angle_max_deg = float(all_angles_deg.max())
    if angle_min_deg == angle_max_deg:
        angle_min_deg -= 1.0
        angle_max_deg += 1.0
    else:
        margin = (angle_max_deg - angle_min_deg) * 0.05
        angle_min_deg -= margin
        angle_max_deg += margin

    return (torque_min, torque_max), (angle_min_deg, angle_max_deg)


def evaluate_right_torque_curve(torque_fn_right_string: str) -> tuple[np.ndarray, np.ndarray]:
    torque_fn_right = execute_torque_function_string(torque_fn_right_string, "right")
    angle_min = float(torque_fn_right.erc_angle_min_rad)
    angle_max = float(torque_fn_right.erc_angle_max_rad)
    if angle_min >= angle_max:
        raise ValueError(f"Invalid right torque function domain: {angle_min} >= {angle_max}")
    angle = torch.linspace(angle_min, angle_max, TORQUE_SAMPLE_COUNT, dtype=torch.float64)
    with torch.no_grad():
        torque = torque_fn_right(angle)
    angle_np = angle.detach().cpu().numpy()
    torque_np = torque.detach().cpu().numpy()
    return angle_np, torque_np


def execute_torque_function_string(torque_fn_string: str, side: str):
    namespace = {"torch": torch}
    exec(torque_fn_string, namespace)
    fn_name = f"torque_fn_{side}"
    if fn_name not in namespace:
        raise RuntimeError(f"Torque function string did not define {fn_name}")
    torque_fn = namespace[fn_name]
    if not callable(torque_fn):
        raise RuntimeError(f"{fn_name} is not callable")
    if not hasattr(torque_fn, "erc_angle_min_rad") or not hasattr(torque_fn, "erc_angle_max_rad"):
        raise RuntimeError(f"{fn_name} does not expose ERC angle-domain attributes")
    return torque_fn


def encode_mp4(paths: AnimationPaths) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(paths.frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(paths.mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if not paths.mp4_path.is_file():
        raise RuntimeError(f"ffmpeg completed but MP4 was not created: {paths.mp4_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create objective-over-time animation for behaviour-based NMPC optimization"
    )
    parser.add_argument("results_dir")
    parser.add_argument(
        "--skip_current_best",
        action="store_true",
        help="Copy current-candidate renders from best-so-far renders for faster debug videos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp4_path = generate_f_over_time_animation(args.results_dir, args.skip_current_best)
    print(f"[Animation] Saved to {mp4_path}")


if __name__ == "__main__":
    main()
