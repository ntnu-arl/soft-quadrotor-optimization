"""Run and resume three behavior-based optimization seeds sequentially."""

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import signal
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZATION_SCRIPT = PROJECT_ROOT / "scripts" / "optimize_design_behaviour_based_nmpc_cma_es.py"
RESULTS_ROOT = PROJECT_ROOT / "results" / "behaviour_based_nmpc_cma_es"
BASELINE_EXPERIMENT_NAME = "behaviour_erc_nmpc20260604_004258"
BASELINE_SEED = 20260603
NEW_SEEDS = (20260604, 20260605, 20260606)
EXPERIMENT_NAME_PREFIX = "behaviour_erc_nmpc_seed_"
STATUS_PATH = RESULTS_ROOT / "different_seeds_progress.json"

POPULATION_SIZE = 20
MAX_GENERATIONS = 100
INITIAL_SIGMA = 0.05
MAX_SIGMA = 0.05
LIMIT_CHANGE_TYPE = "clip_sigma"
EVAL_SEED = 43
NUM_ENVS_EVAL = 100
NUM_EPISODES = 1
DEVICE = "cuda:0"
WALL_GAP_MIN = 0.15
WALL_GAP_MAX = 0.25
N_EVALS_SOBOL = 50
N_EVALS_BO = 350
N_EVALS_CMAES = 400
EXPECTED_PHASE_COUNTS = {
    "SOBOL": N_EVALS_SOBOL,
    "BO": N_EVALS_BO,
    "CMAES": N_EVALS_CMAES,
}
EXPECTED_EVALUATIONS_PER_RUN = sum(EXPECTED_PHASE_COUNTS.values())
WORKER_TYPES = ("local",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run three additional behavior-based optimizations with distinct seeds."
    )
    parser.add_argument(
        "--worker_type_list",
        nargs="+",
        required=True,
        choices=WORKER_TYPES,
        help="Worker services passed explicitly to every optimization process",
    )
    parser.add_argument(
        "--progress_interval_seconds",
        type=float,
        required=True,
        help="Seconds between progress and ETA updates while an optimization is running",
    )
    args = parser.parse_args()
    if args.progress_interval_seconds <= 0.0:
        raise ValueError(
            f"progress_interval_seconds must be > 0, got {args.progress_interval_seconds}"
        )
    return args


def experiment_name(seed):
    return f"{EXPERIMENT_NAME_PREFIX}{seed}"


def results_dir_for_seed(seed):
    return RESULTS_ROOT / experiment_name(seed)


def expected_config(seed, name):
    return {
        "experiment_name": name,
        "population_size": POPULATION_SIZE,
        "max_generations": MAX_GENERATIONS,
        "initial_sigma": INITIAL_SIGMA,
        "max_sigma": MAX_SIGMA,
        "limit_change_type": LIMIT_CHANGE_TYPE,
        "seed": seed,
        "eval_seed": EVAL_SEED,
        "num_envs_eval": NUM_ENVS_EVAL,
        "num_episodes": NUM_EPISODES,
        "device": DEVICE,
        "wall_gap_size_tuple": [WALL_GAP_MIN, WALL_GAP_MAX],
        "n_evals_sobol": N_EVALS_SOBOL,
        "n_evals_bo": N_EVALS_BO,
        "n_evals_cmaes": N_EVALS_CMAES,
    }


def validate_run_config(results_dir, seed, name):
    config_path = results_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with config_path.open("r") as handle:
        actual = json.load(handle)
    for key, expected_value in expected_config(seed, name).items():
        if key not in actual:
            raise KeyError(f"{config_path} does not contain {key}")
        if actual[key] != expected_value:
            raise ValueError(
                f"Configuration mismatch in {config_path} for {key}: "
                f"expected {expected_value!r}, got {actual[key]!r}"
            )


def read_progress(results_dir):
    trials_path = results_dir / "trials.csv"
    if not trials_path.exists():
        return {
            "evaluations": 0,
            "phase_counts": {phase: 0 for phase in EXPECTED_PHASE_COUNTS},
            "recorded_seconds": 0.0,
        }

    phase_counts = {phase: 0 for phase in EXPECTED_PHASE_COUNTS}
    generation_seconds = {}
    with trials_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"phase", "generation", "generation_time_seconds"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Missing required progress columns in {trials_path}")
        evaluations = 0
        for row in reader:
            phase = row["phase"]
            if phase not in phase_counts:
                raise ValueError(f"Unexpected optimization phase {phase!r} in {trials_path}")
            phase_counts[phase] += 1
            evaluations += 1
            generation = int(row["generation"])
            elapsed = float(row["generation_time_seconds"])
            if generation in generation_seconds and generation_seconds[generation] != elapsed:
                raise ValueError(
                    f"Generation {generation} has inconsistent durations in {trials_path}"
                )
            generation_seconds[generation] = elapsed

    for phase, count in phase_counts.items():
        if count > EXPECTED_PHASE_COUNTS[phase]:
            raise ValueError(
                f"{trials_path} contains {count} {phase} evaluations; "
                f"expected at most {EXPECTED_PHASE_COUNTS[phase]}"
            )
    return {
        "evaluations": evaluations,
        "phase_counts": phase_counts,
        "recorded_seconds": sum(generation_seconds.values()),
    }


def is_complete(progress):
    return progress["phase_counts"] == EXPECTED_PHASE_COUNTS


def format_duration(seconds):
    return str(timedelta(seconds=max(0, round(seconds))))


def collect_campaign_status(active_seed):
    runs = []
    baseline_dir = RESULTS_ROOT / BASELINE_EXPERIMENT_NAME
    validate_run_config(
        baseline_dir,
        BASELINE_SEED,
        BASELINE_EXPERIMENT_NAME,
    )
    baseline_progress = read_progress(baseline_dir)
    if not is_complete(baseline_progress):
        raise RuntimeError(f"Baseline optimization is not complete: {baseline_dir}")
    runs.append(
        {
            "seed": BASELINE_SEED,
            "experiment_name": BASELINE_EXPERIMENT_NAME,
            "role": "existing_baseline",
            "state": "complete",
            **baseline_progress,
        }
    )

    for seed in NEW_SEEDS:
        results_dir = results_dir_for_seed(seed)
        if not results_dir.exists():
            progress = {
                "evaluations": 0,
                "phase_counts": {phase: 0 for phase in EXPECTED_PHASE_COUNTS},
                "recorded_seconds": 0.0,
            }
            state = "pending"
        else:
            validate_run_config(results_dir, seed, experiment_name(seed))
            progress = read_progress(results_dir)
            state = "complete" if is_complete(progress) else "interrupted"
        if seed == active_seed:
            state = "running"
        runs.append(
            {
                "seed": seed,
                "experiment_name": experiment_name(seed),
                "role": "additional_seed",
                "state": state,
                **progress,
            }
        )

    completed_evaluations = sum(run["evaluations"] for run in runs)
    recorded_seconds = sum(run["recorded_seconds"] for run in runs)
    total_evaluations = EXPECTED_EVALUATIONS_PER_RUN * len(runs)
    remaining_evaluations = total_evaluations - completed_evaluations
    if completed_evaluations <= 0 or recorded_seconds <= 0.0:
        raise RuntimeError("Cannot estimate completion time without recorded evaluation durations")
    seconds_per_evaluation = recorded_seconds / completed_evaluations
    eta_seconds = seconds_per_evaluation * remaining_evaluations
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_evaluations": completed_evaluations,
        "total_evaluations": total_evaluations,
        "completion_percentage": 100.0 * completed_evaluations / total_evaluations,
        "eta_seconds": eta_seconds,
        "estimated_completion_at": (
            datetime.now().astimezone() + timedelta(seconds=eta_seconds)
        ).isoformat(timespec="seconds"),
        "runs": runs,
    }


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = STATUS_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as handle:
        json.dump(status, handle, indent=2)
    temporary_path.replace(STATUS_PATH)


def report_status(active_seed):
    status = collect_campaign_status(active_seed)
    write_status(status)
    print(
        f"[Campaign] {status['completed_evaluations']}/{status['total_evaluations']} "
        f"evaluations ({status['completion_percentage']:.1f}%) | "
        f"ETA {format_duration(status['eta_seconds'])} | "
        f"estimated completion {status['estimated_completion_at']}",
        flush=True,
    )
    for run in status["runs"]:
        print(
            f"  seed {run['seed']}: {run['state']}, "
            f"{run['evaluations']}/{EXPECTED_EVALUATIONS_PER_RUN} evaluations, "
            f"phases={run['phase_counts']}",
            flush=True,
        )
    return status


def solve_command(seed, worker_type_list):
    return [
        sys.executable,
        str(OPTIMIZATION_SCRIPT),
        "solve",
        "--worker_type_list",
        *worker_type_list,
        "--experiment_name",
        experiment_name(seed),
        "--population_size",
        str(POPULATION_SIZE),
        "--max_generations",
        str(MAX_GENERATIONS),
        "--initial_sigma",
        str(INITIAL_SIGMA),
        "--max_sigma",
        str(MAX_SIGMA),
        "--limit_change_type",
        LIMIT_CHANGE_TYPE,
        "--seed",
        str(seed),
        "--eval_seed",
        str(EVAL_SEED),
        "--num_envs_eval",
        str(NUM_ENVS_EVAL),
        "--num_episodes",
        str(NUM_EPISODES),
        "--device",
        DEVICE,
        "--wall_gap_min",
        str(WALL_GAP_MIN),
        "--wall_gap_max",
        str(WALL_GAP_MAX),
    ]


def resume_command(seed, worker_type_list):
    return [
        sys.executable,
        str(OPTIMIZATION_SCRIPT),
        "resume",
        "--results_dir",
        str(results_dir_for_seed(seed)),
        "--worker_type_list",
        *worker_type_list,
    ]


def command_for_seed(seed, worker_type_list):
    results_dir = results_dir_for_seed(seed)
    if not results_dir.exists():
        return solve_command(seed, worker_type_list), "start"
    validate_run_config(results_dir, seed, experiment_name(seed))
    progress = read_progress(results_dir)
    if is_complete(progress):
        return None, "complete"
    state_path = results_dir / "cma_state.pkl"
    if not state_path.is_file():
        trials_path = results_dir / "trials.csv"
        if trials_path.exists():
            raise FileNotFoundError(
                f"Run has results but no optimizer checkpoint: {state_path}"
            )
    return resume_command(seed, worker_type_list), "resume"


def run_child_with_progress(command, seed, progress_interval_seconds):
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        while True:
            try:
                return_code = process.wait(timeout=progress_interval_seconds)
                break
            except subprocess.TimeoutExpired:
                report_status(seed)
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        report_status(None)
        raise KeyboardInterrupt from None
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    args = parse_args()
    if not OPTIMIZATION_SCRIPT.is_file():
        raise FileNotFoundError(f"Optimization script not found: {OPTIMIZATION_SCRIPT}")

    report_status(None)
    for seed in NEW_SEEDS:
        command, action = command_for_seed(seed, args.worker_type_list)
        if action == "complete":
            print(f"[Seed {seed}] Already complete; skipping", flush=True)
            continue
        print(f"[Seed {seed}] {action}: {' '.join(command)}", flush=True)
        run_child_with_progress(command, seed, args.progress_interval_seconds)
        progress = read_progress(results_dir_for_seed(seed))
        if not is_complete(progress):
            raise RuntimeError(
                f"Optimization process exited successfully without completing seed {seed}"
            )
        report_status(None)

    final_status = report_status(None)
    if final_status["completed_evaluations"] != final_status["total_evaluations"]:
        raise RuntimeError("Dispatcher stopped before all four seed runs were complete")
    print(f"[Campaign] All four seeds complete. Status: {STATUS_PATH}", flush=True)


if __name__ == "__main__":
    main()
