"""Width optimization across different control authority loss values and constraint configurations.

This script:
1. Tests multiple configurations for acceleration constraints (coupled, semidecoupled, decoupled)
2. For each configuration and control authority loss value, generates N width-optimized airframe designs
3. Saves results to CSV with width and design parameters
4. Generates visualization comparing folded width vs control authority loss across configurations

Usage:
python scripts/airframe_width_optimization_diff_loss_authority.py --solve
python scripts/airframe_width_optimization_diff_loss_authority.py --plot
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from airframe_encoding import (
    Airframe,
    PARAM_BOUNDS,
)
from airframe_validations import (
    repair_solution_acceleration,
    build_collision_torque_response,
)
from optimization_algorithms import (
    CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA,
    CMA_VERBOSE,
    OBJECTIVE_SENSE_MINIMIZE,
    WIDTH_CMA_SIGMA_MAX,
    WIDTH_CMA_SIGMA_MIN,
    WIDTH_CMA_TOLX,
    SigmaClippedCMAES,
)
from utils import generate_width_loss_plot_with_top_views


CONTROL_AUTHORITY_CONSTRAINT_CONFIGURATIONS = {
    "config1_coupled": [
        [   0.0,    0.0, "+lin_z",       0.0,       0.0,      0.0],
        ["free", "free",     9.81,  "+ang_x",       0.0,      0.0],
        ["free", "free",     9.81,  "-ang_x",       0.0,      0.0],
        ["free", "free",     9.81,       0.0,  "+ang_y",      0.0],
        ["free", "free",     9.81,       0.0,  "-ang_y",      0.0],
        ["free", "free",   "free",       0.0,       0.0, "+ang_z"],
        ["free", "free",   "free",       0.0,       0.0, "-ang_z"],
    ],
    "config2_semidecoupled": [
        [     0.0,      0.0, "+lin_z",       0.0,       0.0,      0.0],
        [     0.0,   "free",     9.81,  "+ang_x",       0.0,      0.0],
        [     0.0,   "free",     9.81,  "-ang_x",       0.0,      0.0],
        [  "free",      0.0,     9.81,       0.0,  "+ang_y",      0.0],
        [  "free",      0.0,     9.81,       0.0,  "-ang_y",      0.0],
        [     0.0,      0.0,   "free",       0.0,       0.0, "+ang_z"],
        [     0.0,      0.0,   "free",       0.0,       0.0, "-ang_z"],
    ],
    "config3_decoupled": [
        [     0.0,      0.0, "+lin_z",       0.0,       0.0,      0.0],
        [     0.0,      0.0,     9.81,  "+ang_x",       0.0,      0.0],
        [     0.0,      0.0,     9.81,  "-ang_x",       0.0,      0.0],
        [     0.0,      0.0,     9.81,       0.0,  "+ang_y",      0.0],
        [     0.0,      0.0,     9.81,       0.0,  "-ang_y",      0.0],
        [     0.0,      0.0,   "free",       0.0,       0.0, "+ang_z"],
        [     0.0,      0.0,   "free",       0.0,       0.0, "-ang_z"],
    ],
}

SELECTED_HIGHLIGHT_DESIGNS = [
    {"config_name": "config1_coupled",       "loss_value": 0.00, "label": "Loss 0.00"},
    {"config_name": "config1_coupled",       "loss_value": 0.03, "label": "Loss 0.03"},
    {"config_name": "config1_coupled",       "loss_value": 0.05, "label": "Loss 0.05"},
    {"config_name": "config1_coupled",       "loss_value": 0.10, "label": "Loss 0.10"},
    {"config_name": "config2_semidecoupled", "loss_value": 0.00, "label": "Loss 0.00"},
    {"config_name": "config2_semidecoupled", "loss_value": 0.03, "label": "Loss 0.03"},
    {"config_name": "config2_semidecoupled", "loss_value": 0.05, "label": "Loss 0.05"},
    {"config_name": "config2_semidecoupled", "loss_value": 0.10, "label": "Loss 0.10"},
    {"config_name": "config3_decoupled",     "loss_value": 0.00, "label": "Loss 0.00"},
    {"config_name": "config3_decoupled",     "loss_value": 0.03, "label": "Loss 0.03"},
    {"config_name": "config3_decoupled",     "loss_value": 0.05, "label": "Loss 0.05"},
    {"config_name": "config3_decoupled",     "loss_value": 0.10, "label": "Loss 0.10"},
]


@dataclass
class Config:
    experiment_name: str = f"width_loss_authority_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    n_airframes_per_config: int = 10
    width_cma_population_size: int = 16
    width_cma_max_generations: int = 100
    width_cma_initial_sigma: float = 0.2
    target_control_authority_loss: list = field(default_factory=lambda: [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    minimum_arm_torque: float = 0.02
    seed: int = 20260305
    multistart: int = 16

    fixed_params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.results_dir = Path(__file__).parent.parent / "results/width_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = self.results_dir / f"{self.experiment_name}.csv"
        self.comparison_pdf = self.results_dir / f"{self.experiment_name}_comparison.pdf"
        self.param_names = list(PARAM_BOUNDS.keys())
        self.param_bounds = list(PARAM_BOUNDS.values())


def get_optimizable_indices(cfg: Config):
    return [i for i, name in enumerate(cfg.param_names) if name not in cfg.fixed_params]


def apply_fixed_params(x_opt: np.ndarray, cfg: Config) -> np.ndarray:
    x_full = np.zeros(len(cfg.param_names))
    opt_idx = 0
    for i, name in enumerate(cfg.param_names):
        if name in cfg.fixed_params:
            min_val, max_val = cfg.param_bounds[i]
            x_full[i] = (cfg.fixed_params[name] - min_val) / (max_val - min_val)
        else:
            x_full[i] = x_opt[opt_idx]
            opt_idx += 1
    return x_full


def compute_width(airframe: Airframe):
    torque_response = build_collision_torque_response(airframe, delta_deg=0.01, verbose=False)
    return torque_response["aabb_info"]["max_width"], torque_response["aabb_info"]


def evaluate_width_solution(x_opt: np.ndarray, cfg: Config, configuration: list, loss_value: float):
    x_full = apply_fixed_params(x_opt, cfg)
    airframe = Airframe.from_x_0_1(x_full)
    target_accel_ratios = {'linear': loss_value, 'angular': loss_value}
    repaired_airframe = repair_solution_acceleration(
        airframe,
        target_accel_ratios=target_accel_ratios,
        verbose=False,
        minimum_arm_torque=cfg.minimum_arm_torque,
        configuration=configuration
    )
    width, aabb_info = compute_width(repaired_airframe)
    return width, repaired_airframe.x_0_1, aabb_info


def create_width_cma_optimizer(n_opt_params: int, population_size: int, initial_sigma: float, seed: int):
    x0 = np.array([0.5] * n_opt_params)
    return SigmaClippedCMAES(
        x0=x0.tolist(),
        sigma0=initial_sigma,
        max_sigma=WIDTH_CMA_SIGMA_MAX,
        opts={
            "popsize": population_size,
            "seed": seed,
            "verbose": CMA_VERBOSE,
            "tolx": WIDTH_CMA_TOLX,
        },
        param_names=[f"x_{i}" for i in range(n_opt_params)],
        param_bounds=[(0.0, 1.0)] * n_opt_params,
        fixed_params={},
        log_params=[],
        limit_change_type=CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA,
        objective_sense=OBJECTIVE_SENSE_MINIMIZE,
        sigma_min=WIDTH_CMA_SIGMA_MIN,
        use_generation_only_stop_options=False,
    )


def _run_single_width_optimization(
    evaluate_fn,
    n_opt_params: int,
    population_size: int,
    max_generations: int,
    initial_sigma: float,
    seed: int,
    label: str,
    max_infeasible_ratio: float,
):
    print(f"\n{'='*80}")
    print(f"Width Optimization for {label} (algorithm: cma_es)")
    print(f"{'='*80}")

    optimizer = create_width_cma_optimizer(n_opt_params, population_size, initial_sigma, seed)

    best_width_ever = np.inf
    best_repaired_full_ever = None
    best_aabb_info_ever = None

    total_infeasible_count = 0
    max_infeasible_total = int(max_infeasible_ratio * population_size * max_generations)

    for gen in range(max_generations):
        gen_start_time = time.time()
        solutions_opt = optimizer.ask()
        widths = []
        repaired_full_solutions = []
        aabb_infos = []
        infeasible_indices = []

        for i, x_opt in enumerate(solutions_opt):
            try:
                width, x_repaired_full, aabb_info = evaluate_fn(np.array(x_opt))
                widths.append(width)
                repaired_full_solutions.append(x_repaired_full)
                aabb_infos.append(aabb_info)
            except RuntimeError as e:
                if "Infeasible_Problem_Detected" in str(e):
                    total_infeasible_count += 1
                    infeasible_indices.append(i)
                    widths.append(None)
                    repaired_full_solutions.append(None)
                    aabb_infos.append(None)

                    if total_infeasible_count > max_infeasible_total:
                        raise RuntimeError(
                            f"Too many infeasible solutions: {total_infeasible_count} > {max_infeasible_total} "
                            f"({max_infeasible_ratio*100:.0f}% of {population_size}*{max_generations})"
                        )
                else:
                    raise

        valid_widths = [w for w in widths if w is not None]
        if valid_widths:
            worst_valid_fitness = max(valid_widths)
            for i in infeasible_indices:
                widths[i] = worst_valid_fitness
        else:
            for i in infeasible_indices:
                widths[i] = 1e6

        optimizer.tell([list(x) for x in solutions_opt], widths)

        if valid_widths:
            gen_best_width = min(valid_widths)
            best_idx = widths.index(gen_best_width)

            if gen_best_width < best_width_ever:
                best_width_ever = gen_best_width
                best_repaired_full_ever = repaired_full_solutions[best_idx]
                best_aabb_info_ever = aabb_infos[best_idx]
        else:
            gen_best_width = np.inf

        n_infeasible_gen = len(infeasible_indices)
        infeasible_str = f", infeasible={n_infeasible_gen}" if n_infeasible_gen > 0 else ""
        gen_time = time.time() - gen_start_time
        print(f"  Gen {gen:3d}: best={gen_best_width:.4f}m, best_ever={best_width_ever:.4f}m, sigma={optimizer.sigma:.4f}{infeasible_str}, time={gen_time:.2f}s")

        if optimizer.stop:
            print(f"  CMA-ES converged at gen {gen}: {optimizer.stop}")
            break

    if best_repaired_full_ever is None:
        raise RuntimeError(f"{label}: No valid solution found after {max_generations} generations")

    width_reduction = (1 - best_width_ever / best_aabb_info_ever['unfolded_width']) * 100

    print(f"\n{label} complete:")
    print(f"  Folded width: {best_width_ever:.4f}m ({best_width_ever*100:.2f}cm)")
    print(f"  Unfolded width: {best_aabb_info_ever['unfolded_width']:.4f}m")
    print(f"  Width reduction: {width_reduction:.1f}%")
    if total_infeasible_count > 0:
        print(f"  Total infeasible solutions: {total_infeasible_count}")

    return {
        'x_0_1': list(best_repaired_full_ever),
        'folded_width': best_width_ever,
        'unfolded_width': best_aabb_info_ever['unfolded_width'],
        'width_reduction_percent': width_reduction,
        'aabb_info': best_aabb_info_ever
    }


def run_width_optimization(
    evaluate_fn,
    n_opt_params: int,
    population_size: int,
    max_generations: int,
    initial_sigma: float,
    seed: int,
    label: str,
    max_infeasible_ratio: float,
    multistart: int,
):
    best_result = None
    for restart_idx in range(multistart):
        restart_seed = seed + restart_idx * 99991
        restart_label = f"{label} [restart {restart_idx+1}/{multistart}]" if multistart > 1 else label
        result = _run_single_width_optimization(
            evaluate_fn,
            n_opt_params,
            population_size,
            max_generations,
            initial_sigma,
            restart_seed,
            restart_label,
            max_infeasible_ratio,
        )
        if best_result is None or result['folded_width'] < best_result['folded_width']:
            best_result = result

    if multistart > 1:
        print(f"\n{'='*80}")
        print(f"Multistart ({multistart} restarts) best for {label}:")
        print(f"  Folded width: {best_result['folded_width']:.4f}m ({best_result['folded_width']*100:.2f}cm)")
        print(f"  Width reduction: {best_result['width_reduction_percent']:.1f}%")
        print(f"{'='*80}")

    return best_result


def optimize_width_for_design(cfg: Config, config_name: str, configuration: list, loss_value: float, seed: int, design_id: int):
    n_opt_params = len(get_optimizable_indices(cfg))

    def evaluate_fn(x_opt):
        return evaluate_width_solution(x_opt, cfg, configuration, loss_value)

    result = run_width_optimization(
        evaluate_fn=evaluate_fn,
        n_opt_params=n_opt_params,
        population_size=cfg.width_cma_population_size,
        max_generations=cfg.width_cma_max_generations,
        initial_sigma=cfg.width_cma_initial_sigma,
        seed=seed,
        label=f"{config_name} Loss={loss_value} Design {design_id}",
        max_infeasible_ratio=0.2,
        multistart=cfg.multistart,
    )
    result['config_name'] = config_name
    result['loss_value'] = loss_value
    return result


def _worker_task(task_args):
    """Worker function for local parallel execution."""
    cfg, config_name, configuration, loss_value, design_id, seed = task_args
    design_info = optimize_width_for_design(
        cfg, config_name, configuration, loss_value, seed, design_id
    )
    result = {
        'config_name': config_name,
        'loss_value': loss_value,
        'design_id': design_id,
        'folded_width': design_info['folded_width'],
        'unfolded_width': design_info['unfolded_width'],
        'width_reduction_percent': design_info['width_reduction_percent'],
        'torque_fn_left': design_info['aabb_info']['torque_fn_left'],
        'torque_fn_right': design_info['aabb_info']['torque_fn_right'],
        **{f'x_{i}': design_info['x_0_1'][i] for i in range(len(design_info['x_0_1']))}
    }
    return result


class JobQueue:
    """Job queue that dispatches width-optimization tasks to local worker processes."""

    LOCAL_N_WORKERS = 6
    POLL_INTERVAL = 60

    def __init__(self, local_results_dir: Path, combined_csv: Path):
        self.local_results_dir = local_results_dir
        self.combined_csv = combined_csv
        self.local_results_dir.mkdir(parents=True, exist_ok=True)
        self._tasks = []

    def add_task(self, task_id: str, local_fn_args: tuple):
        self._tasks.append({
            "task_id": task_id,
            "local_fn_args": local_fn_args,
        })

    def run(self):
        if not self._tasks:
            print("No tasks queued.")
            return

        total = len(self._tasks)
        print(f"\n{'='*80}")
        print(f"JobQueue: {total} tasks")
        print(f"{'='*80}\n")

        pending = list(self._tasks)
        local_executor = ProcessPoolExecutor(max_workers=self.LOCAL_N_WORKERS)
        local_futures = {}
        start_time = time.time()

        while True:
            # Dispatch pending tasks to available slots
            self._dispatch(pending, local_executor, local_futures)

            # Collect completed local results
            newly_done = [f for f in list(local_futures) if f.done()]
            for f in newly_done:
                task_id = local_futures.pop(f)
                out = self.local_results_dir / f"{task_id}.csv"
                if not out.exists():
                    pd.DataFrame([f.result()]).to_csv(out, index=False)

            completed = self._collect_results()
            self._print_progress(completed, total, start_time)

            if completed >= total:
                break

            time.sleep(self.POLL_INTERVAL)

        local_executor.shutdown(wait=False)

        print(f"\n{'='*80}")
        print("ALL JOBS COMPLETE")
        print(f"Results: {self.combined_csv}")
        print(f"{'='*80}\n")

    def _dispatch(self, pending, local_executor, local_futures):
        if not pending:
            return

        local_running = len(local_futures)
        local_slots = max(0, self.LOCAL_N_WORKERS - local_running)
        for _ in range(min(local_slots, len(pending))):
            task = pending.pop(0)
            future = local_executor.submit(_worker_task, task["local_fn_args"])
            local_futures[future] = task["task_id"]

    def _collect_results(self):
        part_files = sorted(self.local_results_dir.glob("*.csv"))
        if part_files:
            df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
            df.to_csv(self.combined_csv, index=False)
        return len(part_files)

    @staticmethod
    def _fmt_duration(s):
        d, rem = divmod(int(s), 86400)
        h, rem = divmod(rem, 3600)
        m, sec = divmod(rem, 60)
        return f"{d}d {h:02d}:{m:02d}:{sec:02d}" if d else f"{h:02d}:{m:02d}:{sec:02d}"

    def _print_progress(self, completed, total, start_time):
        elapsed = time.time() - start_time
        pct = completed / total * 100
        msg = f"Progress: {completed}/{total} ({pct:.1f}%) | Elapsed: {self._fmt_duration(elapsed)}"
        if completed > 0:
            time_left = elapsed / completed * (total - completed)
            finish = datetime.now() + timedelta(seconds=time_left)
            eta = finish.strftime("%H:%M:%S") if finish.date() == datetime.now().date() else finish.strftime("%Y-%m-%d %H:%M:%S")
            msg += f" | Remaining: {self._fmt_duration(time_left)} | ETA: {eta}"
        print(f"\n{'='*60}\n{msg}")


def generate_comparison_plot(cfg: Config, df: pd.DataFrame | None):
    """Generate comparison plot of folded width vs control authority loss with top-view renders."""
    if df is None:
        if not cfg.results_csv.exists():
            print(f"Results file not found: {cfg.results_csv}")
            return
        df = pd.read_csv(cfg.results_csv)

    print(f"\nGenerating comparison plot with top views...")

    planar_csv = cfg.results_dir / "planar_width.csv"
    if not planar_csv.exists():
        print(f"Warning: Planar CSV not found: {planar_csv}")
        return

    config_color_map = {name: plt.cm.tab10(i) for i, name in enumerate(CONTROL_AUTHORITY_CONSTRAINT_CONFIGURATIONS)}
    for design in SELECTED_HIGHLIGHT_DESIGNS:
        design['color'] = config_color_map[design['config_name']]

    generate_width_loss_plot_with_top_views(df, planar_csv, SELECTED_HIGHLIGHT_DESIGNS, cfg.comparison_pdf)
    print(f"Plot saved to: {cfg.comparison_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Width optimization across control authority loss values")
    parser.add_argument("--plot", action='store_true', help="Generate plot from existing results without running optimization")
    parser.add_argument("--run_one", nargs=4, metavar=('CONFIG', 'LOSS', 'DESIGN_ID', 'CSV_PATH'),
                        help="Run single configuration: CONFIG_NAME LOSS_VALUE DESIGN_ID CSV_PATH")
    parser.add_argument("--solve", action='store_true', help="Run optimization sweep")
    parser.add_argument("--popsize", type=int, help="Override CMA population size")
    parser.add_argument("--max_generations", type=int, help="Override max generations")
    parser.add_argument("--multistart", type=int, help="Number of independent restarts (best is returned)")
    args = parser.parse_args()

    cfg = Config()
    if args.popsize:
        cfg.width_cma_population_size = args.popsize
    if args.max_generations:
        cfg.width_cma_max_generations = args.max_generations
    if args.multistart:
        cfg.multistart = args.multistart

    if args.plot:
        for csv_path in sorted(cfg.results_dir.glob("width_loss_authority_*.csv")):
            pdf_path = csv_path.with_name(csv_path.stem + "_comparison.pdf")
            cfg.results_csv = csv_path
            cfg.comparison_pdf = pdf_path
            generate_comparison_plot(cfg, None)

    elif args.run_one:
        config_name, loss_str, design_id_str, csv_path = args.run_one
        if config_name not in CONTROL_AUTHORITY_CONSTRAINT_CONFIGURATIONS:
            print(f"Error: Unknown configuration '{config_name}'")
            print(f"Available configurations: {list(CONTROL_AUTHORITY_CONSTRAINT_CONFIGURATIONS.keys())}")
            sys.exit(1)

        loss_value = float(loss_str)
        design_id = int(design_id_str)
        configuration = CONTROL_AUTHORITY_CONSTRAINT_CONFIGURATIONS[config_name]
        seed = cfg.seed + hash((config_name, loss_value, design_id)) % 100000

        design_info = optimize_width_for_design(cfg, config_name, configuration, loss_value, seed, design_id)

        result = {
            'config_name': config_name,
            'loss_value': loss_value,
            'design_id': design_id,
            'folded_width': design_info['folded_width'],
            'unfolded_width': design_info['unfolded_width'],
            'width_reduction_percent': design_info['width_reduction_percent'],
            'torque_fn_left': design_info['aabb_info']['torque_fn_left'],
            'torque_fn_right': design_info['aabb_info']['torque_fn_right'],
            **{f'x_{i}': design_info['x_0_1'][i] for i in range(len(design_info['x_0_1']))}
        }

        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(out, index=False)

        print(f"Folded width: {design_info['folded_width']*100:.1f}cm, "
              f"Reduction: {design_info['width_reduction_percent']:.1f}%")

    elif args.solve:
        local_results_dir = cfg.results_dir / cfg.experiment_name
        queue = JobQueue(
            local_results_dir=local_results_dir,
            combined_csv=cfg.results_csv,
        )

        for design_id in range(cfg.n_airframes_per_config):
            for loss_value in cfg.target_control_authority_loss:
                for config_name, configuration in CONTROL_AUTHORITY_CONSTRAINT_CONFIGURATIONS.items():
                    seed = cfg.seed + hash((config_name, loss_value, design_id)) % 100000
                    task_id = f"{config_name}_{loss_value}_{design_id}"
                    queue.add_task(
                        task_id=task_id,
                        local_fn_args=(cfg, config_name, configuration, loss_value, design_id, seed),
                    )

        queue.run()

    else:
        parser.print_help()
