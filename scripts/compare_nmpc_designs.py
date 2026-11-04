# python scripts/compare_nmpc_designs.py --results_dir results/behaviour_based_nmpc_cma_es/behaviour_erc_nmpc20260601_112908 --seed 43 --num_envs 200 --num_episodes_eval 20 --num_episodes_video 1 --playback_speed 0.25 --gap_min 0.1 --gap_max 0.3 --video_gap_widths 0.16 0.18 0.20 0.22 0.24
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_metrics_file(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        with open(metrics_path) as metrics_file:
            metrics = json.load(metrics_file)
    except json.JSONDecodeError:
        return False
    required_keys = [
        "gap_statistics",
        "total_episodes",
        "total_successes",
        "total_timeouts",
        "total_crashes",
        "success_percentage",
    ]
    missing = [key for key in required_keys if key not in metrics]
    if missing:
        raise RuntimeError(f"Metrics file is missing keys {missing}: {metrics_path}")
    return True


def run_command_to_completion(command: list[str], process_label: str) -> None:
    print(f">> {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Subprocess stdout pipe was not created")

    for line in process.stdout:
        print(line, end="", flush=True)
    process.stdout.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"{process_label} failed with return code {process.returncode}: {' '.join(command)}")


def resolve_results_dir(results_dir_text: str) -> Path:
    results_dir = Path(results_dir_text).expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {results_dir}")
    return results_dir


def load_best_design(results_dir: Path) -> tuple[dict, Path, Path]:
    best_solution_json = results_dir / "best_solution.json"
    if not best_solution_json.is_file():
        raise FileNotFoundError(f"Missing best solution JSON: {best_solution_json}")
    with open(best_solution_json) as json_file:
        best_solution = json.load(json_file)
    workspace_dir = Path(best_solution["workspace_id"])
    if not workspace_dir.is_absolute():
        workspace_dir = PROJECT_ROOT / workspace_dir
    workspace_dir = workspace_dir.resolve()
    usd_path = workspace_dir / "morphy_prog.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"Missing best design USD: {usd_path}")
    return best_solution, best_solution_json, workspace_dir


def build_output_dir(results_dir: Path, workspace_dir: Path) -> Path:
    output_dir = results_dir / f"collision_sweep_{workspace_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_metadata(best_solution: dict, best_solution_json: Path, workspace_dir: Path, output_dir: Path) -> None:
    metadata = {
        "workspace_dir": str(workspace_dir),
        "best_solution_json": str(best_solution_json),
        "generation": best_solution["generation"],
        "individual_id": best_solution["individual_id"],
        "success_rate_from_run": best_solution["success_rate"],
        "mean_position_error_from_run": best_solution["mean_position_error"],
    }
    with open(output_dir / "design_metadata.json", "w") as json_file:
        json.dump(metadata, json_file, indent=2)


def rows_from_metrics(metrics_path: Path, workspace_dir: Path) -> list[dict]:
    with open(metrics_path) as metrics_file:
        metrics = json.load(metrics_file)
    rows = []
    for gap_stat in metrics["gap_statistics"]:
        rows.append(
            {
                "workspace": workspace_dir.name,
                "gap_width": float(gap_stat["gap_width"]),
                "success_percentage": 100.0 * float(gap_stat["success_rate"]),
                "num_episodes_at_width": int(gap_stat["num_episodes"]),
                "num_successes_at_width": int(gap_stat["num_successes"]),
                "overall_success_percentage": float(metrics["success_percentage"]),
                "overall_timeout_percentage": float(metrics["timeout_percentage"]),
                "overall_crash_percentage": float(metrics["crash_percentage"]),
                "total_episodes": int(metrics["total_episodes"]),
                "total_successes": int(metrics["total_successes"]),
                "total_timeouts": int(metrics["total_timeouts"]),
                "total_crashes": int(metrics["total_crashes"]),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def copy_summary_plot(metrics_path: Path, output_dir: Path) -> None:
    source_plot = metrics_path.with_suffix(".pdf")
    if not source_plot.is_file():
        raise FileNotFoundError(f"Expected NMPC summary plot not found: {source_plot}")
    shutil.copy2(source_plot, output_dir / "success_rate_vs_gap_width.pdf")


def evaluate_design(workspace_dir: Path, output_dir: Path, gap_min: float, gap_max: float, seed: int, num_envs: int, num_episodes: int) -> list[dict]:
    metrics_path = output_dir / f"gap_{gap_min:.2f}_{gap_max:.2f}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/nmpc_enjoy.py",
        "--headless",
        "--num_envs",
        str(num_envs),
        "--num_episodes",
        str(num_episodes),
        "--seed",
        str(seed),
        "--metrics_output",
        str(metrics_path),
        "--usd_path",
        str(workspace_dir / "morphy_prog.usd"),
        "--config-override",
        f"environment.min_gap_width={gap_min}",
        "--config-override",
        f"environment.max_gap_width={gap_max}",
        "--config-override",
        "log_metrics=True",
    ]
    run_command_to_completion(command, f"evaluate:{workspace_dir.name}")
    if not valid_metrics_file(metrics_path):
        raise RuntimeError(f"Evaluation did not produce a complete metrics file: {metrics_path}")
    copy_summary_plot(metrics_path, output_dir)
    return rows_from_metrics(metrics_path, workspace_dir)


def generate_video(workspace_dir: Path, output_dir: Path, gap_width: float, seed: int, num_episodes: int, playback_speed: float) -> None:
    destination = output_dir / "collision_videos" / f"gap_{gap_width:.2f}.mp4"
    if destination.exists():
        print(f"[{workspace_dir.name}] Removing existing video file before regeneration: {destination}", flush=True)
        destination.unlink()

    before = set((PROJECT_ROOT / "results").glob("*_nmpc_*_composed.mp4"))
    command = [
        sys.executable,
        "scripts/generate_collision_video_nmpc.py",
        "--workspace_dir",
        str(workspace_dir),
        "--num_episodes",
        str(num_episodes),
        "--playback_speed",
        str(playback_speed),
        "--gap_width",
        str(gap_width),
        "--seed",
        str(seed),
    ]

    run_command_to_completion(command, f"video:{workspace_dir.name}:gap_{gap_width:.2f}")
    after = set((PROJECT_ROOT / "results").glob("*_nmpc_*_composed.mp4"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one generated composed video, found {len(created)}")
    created_video_path = created[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(created_video_path), destination)


def write_summary(rows: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "success_rate_vs_width.csv"
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sorted_rows = sorted(rows, key=lambda row: row["gap_width"])
    ax.plot(
        [row["gap_width"] for row in sorted_rows],
        [row["success_percentage"] for row in sorted_rows],
        marker="o",
    )
    ax.set_xlabel("Gap width (m)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "success_rate_vs_width_from_metrics.pdf")
    fig.savefig(output_dir / "success_rate_vs_width_from_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--num_episodes_eval", type=int, required=True)
    parser.add_argument("--num_episodes_video", type=int, required=True)
    parser.add_argument("--playback_speed", type=float, required=True)
    parser.add_argument("--gap_min", type=float, required=True)
    parser.add_argument("--gap_max", type=float, required=True)
    parser.add_argument("--video_gap_widths", type=float, nargs="+", required=True)
    args = parser.parse_args()

    results_dir = resolve_results_dir(args.results_dir)
    best_solution, best_solution_json, workspace_dir = load_best_design(results_dir)
    output_dir = build_output_dir(results_dir, workspace_dir)
    write_metadata(best_solution, best_solution_json, workspace_dir, output_dir)

    rows = evaluate_design(
        workspace_dir,
        output_dir,
        args.gap_min,
        args.gap_max,
        args.seed,
        args.num_envs,
        args.num_episodes_eval,
    )
    write_summary(rows, output_dir)
    write_plot(rows, output_dir)

    for gap_width in args.video_gap_widths:
        generate_video(
            workspace_dir,
            output_dir,
            gap_width,
            args.seed,
            args.num_episodes_video,
            args.playback_speed,
        )


if __name__ == "__main__":
    main()
