"""
Behaviour-based airframe optimization with independent NMPC evaluations.

This script runs Sobol initialization, Bayesian optimization, and CMA-ES over a
normalized vector containing airframe geometry parameters and four shared torque
response parameters. With --rigid, the side arm joints are fixed and only the
geometry parameters are optimized. With --planar_fold_axis, the geometry is
fixed to the reference planar quadrotor and only the left/right fold axis tilts
(side_base_rotation_x, compensated by side_motor_rotation_x so motor positions
and thrust directions stay planar) and the torque response parameters are
optimized. Every candidate is evaluated by independent NMPC objective jobs
queued across workers.
"""
# EXP=behaviour_erc_nmpc$(date +%Y%m%d_%H%M%S)
# python scripts/optimize_design_behaviour_based_nmpc_cma_es.py solve --worker_type_list local --experiment_name "$EXP"
# python scripts/optimize_design_behaviour_based_nmpc_cma_es.py solve --rigid --worker_type_list local --experiment_name "$EXP"
# python scripts/optimize_design_behaviour_based_nmpc_cma_es.py solve --planar_fold_axis --worker_type_list local --experiment_name "$EXP"

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airframe_encoding import Airframe, PARAM_BOUNDS
from airframe_validations import build_collision_torque_response, repair_solution_acceleration
from arm_detachment_torque import one_sided_detachment_envelope_rad_nm
from erc_torque_profiles import (
    DEFAULT_SPRING_CATALOG,
    DEFAULT_SPRING_ID,
    DETACHMENT_MIN_ANGLE_DEG,
    DETACHMENT_N_SAMPLES,
    DETACHMENT_REQUEST_MARGIN_NM,
    DETACHMENT_SAFETY_FACTOR,
    FORMULA_TORQUE_A_BOUNDS_NM,
    FORMULA_TORQUE_B_BOUNDS,
    FORMULA_TORQUE_C_BOUNDS_NM,
    FORMULA_TORQUE_D_BOUNDS_NM,
    FormulaTorqueParameters,
    build_formula_erc_torque_function,
)
from optimization_algorithms import (
    ALGORITHM_SOBOL_BO_CMAES,
    CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA,
    CMA_LIMIT_CHANGE_TYPE_CLIP_SOLUTION,
    CMA_VERBOSE,
    OBJECTIVE_SENSE_MAXIMIZE,
    OptimizationAlgorithm,
    SOBOL_BO_CMAES_STAGE_BO,
    SOBOL_BO_CMAES_STAGE_CMAES,
    SOBOL_BO_CMAES_STAGE_SOBOL,
    SobolBOCMAES,
)
from remote import RemoteEvaluator


PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
RESULTS_ROOT = PROJECT_ROOT / "results" / "behaviour_based_nmpc_cma_es"
WORKSPACE_ROOT = PROJECT_ROOT / "cache" / "workspace_local"
GEOMETRY_PARAM_NAMES = list(PARAM_BOUNDS.keys())
N_GEOMETRY_PARAMS = len(GEOMETRY_PARAM_NAMES)
TORQUE_RESPONSE_PARAM_BOUNDS = {
    "torque_response_a": FORMULA_TORQUE_A_BOUNDS_NM,
    "torque_response_b": FORMULA_TORQUE_B_BOUNDS,
    "torque_response_c": FORMULA_TORQUE_C_BOUNDS_NM,
    "torque_response_d": FORMULA_TORQUE_D_BOUNDS_NM,
}
TORQUE_RESPONSE_COLLISION_DELTA_DEG = 0.1
TORQUE_RESPONSE_ERC_N_GRID = 1000
TORQUE_RESPONSE_ERC_SAFETY_FACTOR = 1.1
TORQUE_RESPONSE_ERC_MAX_TORQUE_RMSE_NM = 0.6
PARAM_NAMES = GEOMETRY_PARAM_NAMES + list(TORQUE_RESPONSE_PARAM_BOUNDS.keys())
VAE_RESULTS_CHECKPOINT_FILENAME = "airframe_vae_model.pth"
VAE_LATENT_PARAM_PREFIX = "airframe_latent_z"
VAE_LATENT_PARAM_BOUNDS = (-3.0, 3.0)
PLANAR_ARM_LENGTHS = {
    "front_arm_length": 0.088,
    "back_arm_length": 0.088,
    "side_arm_seg2_length_left": 0.078,
    "side_arm_seg2_length_right": 0.078,
}
PLANAR_FOLD_AXIS_PARAM_NAMES = ["side_base_rotation_x_left", "side_base_rotation_x_right"]
PLANAR_FOLD_AXIS_REPAIR_TOLERANCE = 1e-6
PLANAR_FOLD_AXIS_COMPENSATION_PARAM_NAMES = {
    "side_base_rotation_x_left": "side_motor_rotation_x_left",
    "side_base_rotation_x_right": "side_motor_rotation_x_right",
}
CONTROL_AUTHORITY_REPAIR_LOSS = 0.05
CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS = {
    "linear": CONTROL_AUTHORITY_REPAIR_LOSS,
    "angular": CONTROL_AUTHORITY_REPAIR_LOSS,
}
CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE = 0.0
CONTROL_AUTHORITY_REPAIR_CONFIGURATION = [
    [0.0, 0.0, "+lin_z", 0.0, 0.0, 0.0],
    [0.0, "free", 9.81, "+ang_x", 0.0, 0.0],
    [0.0, "free", 9.81, "-ang_x", 0.0, 0.0],
    ["free", 0.0, 9.81, 0.0, "+ang_y", 0.0],
    ["free", 0.0, 9.81, 0.0, "-ang_y", 0.0],
    [0.0, 0.0, "free", 0.0, 0.0, "+ang_z"],
    [0.0, 0.0, "free", 0.0, 0.0, "-ang_z"],
]
DEFAULT_EXPERIMENT_NAME = "behaviour_based_nmpc_cma_es_default"
DEFAULT_POPULATION_SIZE = 20
DEFAULT_MAX_GENERATIONS = 100
DEFAULT_INITIAL_SIGMA = 0.05
DEFAULT_MAX_SIGMA = 0.05
DEFAULT_LIMIT_CHANGE_TYPE = CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA
DEFAULT_SEED = 20260603
DEFAULT_EVAL_SEED = 43
DEFAULT_NUM_ENVS_EVAL = 100
DEFAULT_NUM_EPISODES = 1
DEFAULT_DEVICE = "cuda:0"
DEFAULT_WALL_GAP_MIN = 0.15
DEFAULT_WALL_GAP_MAX = 0.25
DEFAULT_RESULTS_DIR = RESULTS_ROOT / DEFAULT_EXPERIMENT_NAME
DEFAULT_RESULTS_CSV = DEFAULT_RESULTS_DIR / "trials.csv"
DEFAULT_PLOT_PATH = DEFAULT_RESULTS_DIR / "progress.pdf"
BO_BATCH_SIZE = 8
N_EVALS_SOBOL = 50
N_EVALS_BO = 350
N_EVALS_CMAES = 400
NMPC_JOB_TIMEOUT_SECONDS = 45 * 60
REQUIRED_METRIC_KEYS = [
    "success_percentage",
    "crash_percentage",
    "timeout_percentage",
    "mean_position_error",
    "mean_velocity_error",
    "mean_attitude_error",
    "mean_control_effort",
]
PLOT_PHASE_ORDER = [
    SOBOL_BO_CMAES_STAGE_SOBOL,
    SOBOL_BO_CMAES_STAGE_BO,
    SOBOL_BO_CMAES_STAGE_CMAES,
]
PLOT_PHASE_COLORS = {
    SOBOL_BO_CMAES_STAGE_SOBOL: "#4c78a8",
    SOBOL_BO_CMAES_STAGE_BO: "#f58518",
    SOBOL_BO_CMAES_STAGE_CMAES: "#54a24b",
}
PLOT_AXIS_COLORS = {
    "x": "#d62728",
    "y": "#008000",
    "z": "#1f77b4",
}
PLOT_NON_AXIS_COLOR = "#6f6f6f"
PLOT_POSITION_LINESTYLES = {
    "front": "-",
    "back": (0, (1, 1)),
    "left": (0, (5, 2)),
    "right": (0, (2, 2)),
}
PLOT_NO_POSITION_LINESTYLE = "-"
PLOT_PARAMETER_GROUPS = [
    (
        "Main Arm Lengths",
        "Length (m)",
        ["front_arm_length", "back_arm_length", "side_arm_seg2_length_left", "side_arm_seg2_length_right"],
    ),
    (
        "Front And Back Prop Rotations",
        "Rotation (deg)",
        [
            "front_prop_rotation_x",
            "front_prop_rotation_y",
            "front_prop_rotation_z",
            "back_prop_rotation_x",
            "back_prop_rotation_y",
            "back_prop_rotation_z",
        ],
    ),
    (
        "Side Base Rotations",
        "Rotation (deg)",
        [
            "side_base_rotation_x_left",
            "side_base_rotation_y_left",
            "side_base_rotation_z_left",
            "side_base_rotation_x_right",
            "side_base_rotation_y_right",
            "side_base_rotation_z_right",
        ],
    ),
    (
        "Side Motor Rotations",
        "Rotation (deg)",
        [
            "side_motor_rotation_x_left",
            "side_motor_rotation_y_left",
            "side_motor_rotation_z_left",
            "side_motor_rotation_x_right",
            "side_motor_rotation_y_right",
            "side_motor_rotation_z_right",
        ],
    ),
    (
        "Torque Response",
        "Actual value",
        [
            "torque_response_a",
            "torque_response_b",
            "torque_response_c",
            "torque_response_d",
        ],
    ),
]
PLOT_REQUIRED_COLUMNS = [
    "algorithm",
    "phase",
    "generation",
    "individual_id",
    "success_rate",
    "crash_percentage",
    "timeout_percentage",
    "mean_position_error",
    "mean_velocity_error",
    "mean_attitude_error",
    "mean_control_effort",
    "eval_time_seconds",
    "generation_time_seconds",
    "sigma",
]


@dataclass
class RunConfig:
    experiment_name: str
    rigid: bool
    planar_fold_axis: bool
    results_dir: Path
    results_csv: Path
    plot_path: Path
    cma_state_path: Path
    run_config_path: Path
    best_solution_path: Path
    vae_source_checkpoint_path: Path | None
    vae_checkpoint_path: Path | None
    vae_latent_dim: int | None
    vae_input_dim: int | None
    population_size: int
    max_generations: int
    initial_sigma: float
    max_sigma: float
    limit_change_type: str
    seed: int
    eval_seed: int
    num_envs_eval: int
    num_episodes: int
    device: str
    wall_gap_size_tuple: tuple[float, float]
    n_evals_sobol: int
    n_evals_bo: int
    n_evals_cmaes: int


def add_common_run_arguments(parser):
    parser.add_argument(
        "--worker_type_list",
        nargs="+",
        required=True,
        choices=["local"],
        help="RemoteEvaluator worker services to register",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="Experiment name used for the results directory",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="CMA-ES population size; must divide the CMA-ES evaluation budget",
    )
    parser.add_argument(
        "--tiny-test",
        dest="tiny_test",
        action="store_const",
        const=True,
        help="Run a tiny local test with reduced optimization budgets and CMA-ES popsize",
    )
    parser.add_argument(
        "--rigid",
        dest="rigid",
        action="store_const",
        const=True,
        help="Optimize only the geometry of a rigid-arm airframe (no torque response parameters)",
    )
    parser.add_argument(
        "--planar_fold_axis",
        dest="planar_fold_axis",
        action="store_const",
        const=True,
        help="Fix the reference planar quadrotor geometry and optimize only the fold axis tilts and torque response",
    )
    parser.add_argument(
        "--max_generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help="Legacy CMA-ES option stored in run_config; SOBOL_BO_CMAES stops after its evaluation budget",
    )
    parser.add_argument(
        "--initial_sigma",
        type=float,
        default=DEFAULT_INITIAL_SIGMA,
        help="Initial CMA-ES sigma",
    )
    parser.add_argument(
        "--max_sigma",
        type=float,
        default=DEFAULT_MAX_SIGMA,
        help="Maximum allowed CMA-ES sigma",
    )
    parser.add_argument(
        "--limit_change_type",
        choices=["clip_sigma", "clip_solution"],
        default=DEFAULT_LIMIT_CHANGE_TYPE,
        help="How CMA-ES candidate updates are clipped",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="CMA-ES random seed")
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Evaluation seed passed to NMPC rollouts",
    )
    parser.add_argument(
        "--num_envs_eval",
        type=int,
        default=DEFAULT_NUM_ENVS_EVAL,
        help="Number of parallel environments for evaluation",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes per environment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device string forwarded to the evaluation script",
    )
    parser.add_argument(
        "--wall_gap_min",
        type=float,
        default=DEFAULT_WALL_GAP_MIN,
        help="Minimum wall-gap width",
    )
    parser.add_argument(
        "--wall_gap_max",
        type=float,
        default=DEFAULT_WALL_GAP_MAX,
        help="Maximum wall-gap width",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SOBOL_BO_CMAES airframe optimization using direct NMPC objective evaluations",
        formatter_class=argparse.HelpFormatter,
    )
    parser.add_argument(
        "--f-over-time",
        dest="f_over_time_results_dir",
        type=str,
        help="Generate an MP4 animation from an existing results directory",
    )
    subparsers = parser.add_subparsers(dest="command")

    solve_parser = subparsers.add_parser(
        "solve",
        help="Start SOBOL_BO_CMAES optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_run_arguments(solve_parser)
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume from a saved results directory and optimizer state",
        formatter_class=argparse.HelpFormatter,
    )
    resume_parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results directory containing run_config.json and cma_state.pkl",
    )
    resume_parser.add_argument(
        "--max_generations",
        type=int,
        help="Legacy CMA-ES option stored in run_config; SOBOL_BO_CMAES stops after its evaluation budget",
    )
    resume_parser.add_argument(
        "--worker_type_list",
        nargs="+",
        required=True,
        choices=["local"],
        help="RemoteEvaluator worker services to register",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot an existing optimization CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_parser.add_argument(
        "--results_csv",
        type=str,
        default=str(DEFAULT_RESULTS_CSV),
        help="Optimization CSV to plot",
    )
    plot_parser.add_argument(
        "--plot_path",
        type=str,
        default=str(DEFAULT_PLOT_PATH),
        help="Output path for the generated PDF",
    )

    return parser.parse_args()


def validate_positive_int(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_positive_float(value, name):
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_wall_gap_tuple(wall_gap_size_tuple):
    if wall_gap_size_tuple[0] > wall_gap_size_tuple[1]:
        raise ValueError(
            f"wall_gap_min must be <= wall_gap_max, got {wall_gap_size_tuple}"
        )


def load_dim_reduction_mask_api():
    if str(SCRIPT_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPT_ROOT))
    import dim_reduction_mask

    return dim_reduction_mask


def load_vae_decoder_from_checkpoint(checkpoint_path):
    dim_reduction_mask = load_dim_reduction_mask_api()
    decoder = dim_reduction_mask.decode_from_file(checkpoint_path)
    if decoder.input_dim != N_GEOMETRY_PARAMS:
        raise ValueError(
            f"VAE decoder input_dim must match geometry parameter count "
            f"{N_GEOMETRY_PARAMS}, got {decoder.input_dim}"
        )
    if list(decoder.param_names) != GEOMETRY_PARAM_NAMES:
        raise ValueError("VAE decoder param_names do not match current geometry parameter names")
    return decoder


def trained_vae_checkpoint_path():
    dim_reduction_mask = load_dim_reduction_mask_api()
    checkpoint_path = Path(dim_reduction_mask.CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained VAE checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_trained_vae_metadata(checkpoint_path):
    decoder = load_vae_decoder_from_checkpoint(checkpoint_path)
    return int(decoder.latent_dim), int(decoder.input_dim)


def run_param_names(rigid):
    if rigid:
        return list(GEOMETRY_PARAM_NAMES)
    return list(PARAM_NAMES)


def n_torque_response_params(rigid):
    if rigid:
        return 0
    return len(TORQUE_RESPONSE_PARAM_BOUNDS)


def validate_design_space(latent_dim, rigid, planar_fold_axis):
    if planar_fold_axis:
        if rigid:
            raise ValueError("planar_fold_axis and rigid cannot both be set")
        if latent_dim is not None:
            raise ValueError(f"planar_fold_axis runs do not use a VAE latent, got latent_dim={latent_dim}")
    else:
        validate_positive_int(int(latent_dim), "vae_latent_dim")


def n_algorithm_geometry_params(latent_dim, rigid, planar_fold_axis):
    validate_design_space(latent_dim, rigid, planar_fold_axis)
    if planar_fold_axis:
        return len(PLANAR_FOLD_AXIS_PARAM_NAMES)
    return int(latent_dim)


def algorithm_param_names(latent_dim, rigid, planar_fold_axis):
    validate_design_space(latent_dim, rigid, planar_fold_axis)
    if planar_fold_axis:
        return list(PLANAR_FOLD_AXIS_PARAM_NAMES) + list(TORQUE_RESPONSE_PARAM_BOUNDS.keys())
    latent_param_names = [
        f"{VAE_LATENT_PARAM_PREFIX}_{i}"
        for i in range(int(latent_dim))
    ]
    if rigid:
        return latent_param_names
    return latent_param_names + list(TORQUE_RESPONSE_PARAM_BOUNDS.keys())


def algorithm_param_bounds(latent_dim, rigid, planar_fold_axis):
    n_geometry = n_algorithm_geometry_params(latent_dim, rigid, planar_fold_axis)
    geometry_bounds = [(0.0, 1.0) for _ in range(n_geometry)]
    torque_bounds = [(0.0, 1.0) for _ in range(n_torque_response_params(rigid))]
    return geometry_bounds + torque_bounds


def validate_algorithm_solution(solution, latent_dim, rigid, planar_fold_axis):
    n_geometry = n_algorithm_geometry_params(latent_dim, rigid, planar_fold_axis)
    expected_shape = (n_geometry + n_torque_response_params(rigid),)
    x_array = np.asarray(solution, dtype=float)
    if x_array.shape != expected_shape:
        raise ValueError(f"Expected algorithm solution shape {expected_shape}, got {x_array.shape}")
    if not np.all(np.isfinite(x_array)):
        raise ValueError(f"Algorithm solution contains non-finite values: {x_array}")
    if np.any(x_array < 0.0) or np.any(x_array > 1.0):
        raise ValueError(f"Algorithm solution must stay within [0, 1], got {x_array}")
    return x_array


def vae_latent_z_from_algorithm_solution(algorithm_solution, latent_dim, rigid):
    x_algorithm = validate_algorithm_solution(algorithm_solution, latent_dim, rigid, False)
    latent_lo, latent_hi = VAE_LATENT_PARAM_BOUNDS
    return latent_lo + x_algorithm[:int(latent_dim)] * (latent_hi - latent_lo)


def planar_geometry_x_0_1():
    geometry = np.full(N_GEOMETRY_PARAMS, 0.5)
    for param_name, length in PLANAR_ARM_LENGTHS.items():
        lo, hi = PARAM_BOUNDS[param_name]
        geometry[GEOMETRY_PARAM_NAMES.index(param_name)] = (length - lo) / (hi - lo)
    return validate_decoded_geometry_x_0_1(geometry)


def planar_fold_axis_geometry_x_0_1(fold_axis_x_0_1):
    fold_axis_array = np.asarray(fold_axis_x_0_1, dtype=float)
    if fold_axis_array.shape != (len(PLANAR_FOLD_AXIS_PARAM_NAMES),):
        raise ValueError(
            f"Expected fold axis shape {(len(PLANAR_FOLD_AXIS_PARAM_NAMES),)}, got {fold_axis_array.shape}"
        )
    geometry = planar_geometry_x_0_1()
    for value, base_param_name in zip(fold_axis_array, PLANAR_FOLD_AXIS_PARAM_NAMES, strict=True):
        base_angle_deg = linear_scaled_value_from_x_0_1(value, PARAM_BOUNDS[base_param_name])
        motor_param_name = PLANAR_FOLD_AXIS_COMPENSATION_PARAM_NAMES[base_param_name]
        motor_lo, motor_hi = PARAM_BOUNDS[motor_param_name]
        geometry[GEOMETRY_PARAM_NAMES.index(base_param_name)] = value
        geometry[GEOMETRY_PARAM_NAMES.index(motor_param_name)] = (-base_angle_deg - motor_lo) / (motor_hi - motor_lo)
    return validate_decoded_geometry_x_0_1(geometry)


def optimizer_mean_algorithm_space(optimizer, cfg):
    raw_mean = np.asarray(optimizer.mean, dtype=float)
    if not np.all(np.isfinite(raw_mean)):
        return raw_mean
    return validate_algorithm_solution(raw_mean, cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)


def validate_decoded_geometry_x_0_1(geometry_x_0_1):
    x_array = np.asarray(geometry_x_0_1, dtype=float)
    if x_array.shape != (N_GEOMETRY_PARAMS,):
        raise ValueError(f"Expected decoded geometry shape {(N_GEOMETRY_PARAMS,)}, got {x_array.shape}")
    if not np.all(np.isfinite(x_array)):
        raise ValueError(f"Decoded geometry contains non-finite values: {x_array}")
    if np.any(x_array < 0.0) or np.any(x_array > 1.0):
        raise ValueError(f"Decoded geometry must stay within [0, 1], got {x_array}")
    return x_array


def validate_x_0_1(x_0_1, rigid):
    n_params = len(run_param_names(rigid))
    x_array = np.asarray(x_0_1, dtype=float)
    if x_array.shape != (n_params,):
        raise ValueError(f"Expected {n_params} normalized parameters, got shape {x_array.shape}")
    if not np.all(np.isfinite(x_array)):
        raise ValueError(f"x_0_1 contains non-finite values: {x_array}")
    if np.any(x_array < 0.0) or np.any(x_array > 1.0):
        raise ValueError(f"x_0_1 must stay within [0, 1], got {x_array}")
    return x_array


def geometry_x_0_1_from_solution(x_0_1, rigid):
    x_array = validate_x_0_1(x_0_1, rigid)
    return x_array[:N_GEOMETRY_PARAMS]


def linear_scaled_value_from_x_0_1(value, bounds):
    lo, hi = bounds
    return float(lo + float(value) * (hi - lo))


def torque_response_values_from_solution(x_0_1):
    x_array = validate_x_0_1(x_0_1, False)
    values = []
    for offset, param_name in enumerate(TORQUE_RESPONSE_PARAM_BOUNDS):
        values.append(
            linear_scaled_value_from_x_0_1(
                x_array[N_GEOMETRY_PARAMS + offset],
                TORQUE_RESPONSE_PARAM_BOUNDS[param_name],
            )
        )
    return tuple(values)


def collision_end_angles_rad_from_airframe(airframe):
    torque_response = build_collision_torque_response(
        airframe,
        TORQUE_RESPONSE_COLLISION_DELTA_DEG,
        False,
    )
    return (
        float(np.deg2rad(torque_response["end_angle_b_left"])),
        float(np.deg2rad(torque_response["end_angle_b_right"])),
    )


def side_torque_response_metadata(side_name, torque_build):
    return {
        f"{side_name}_torque_response_angle_offset_rad": float(torque_build.angle_offset_rad),
        f"{side_name}_torque_response_angle_offset_deg": float(np.rad2deg(torque_build.angle_offset_rad)),
        f"{side_name}_torque_response_erc_rmse_nm": float(torque_build.design_result.torque_rmse_nm),
        f"{side_name}_torque_response_erc_was_repaired": bool(torque_build.design_result.was_repaired),
        f"{side_name}_torque_response_erc_was_scaled": bool(torque_build.design_result.was_scaled),
    }


@lru_cache(maxsize=4096)
def build_nmpc_torque_response_payload_cached(x_0_1_tuple):
    x_array = validate_x_0_1(x_0_1_tuple, False)
    airframe = Airframe.from_x_0_1(geometry_x_0_1_from_solution(x_array, False))
    airframe_data = airframe.to_json_dict()
    end_angle_left_rad, end_angle_right_rad = collision_end_angles_rad_from_airframe(airframe)
    torque_response_a, torque_response_b, torque_response_c, torque_response_d = torque_response_values_from_solution(x_array)
    shared_parameters = {
        "a": torque_response_a,
        "b": torque_response_b,
        "c": torque_response_c,
        "d": torque_response_d,
    }
    min_detachment_angle_rad = math.radians(DETACHMENT_MIN_ANGLE_DEG)
    left_detachment_envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        1,
        end_angle_left_rad,
        DETACHMENT_N_SAMPLES,
        DETACHMENT_SAFETY_FACTOR,
        1,
        min_detachment_angle_rad,
    )
    right_detachment_envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        3,
        end_angle_right_rad,
        DETACHMENT_N_SAMPLES,
        DETACHMENT_SAFETY_FACTOR,
        -1,
        min_detachment_angle_rad,
    )
    left_build = build_formula_erc_torque_function(
        FormulaTorqueParameters(
            shared_parameters["a"],
            shared_parameters["b"],
            shared_parameters["c"],
            shared_parameters["d"],
            end_angle_left_rad,
        ),
        "torque_fn_left",
        False,
        left_detachment_envelope,
        DETACHMENT_REQUEST_MARGIN_NM,
        DEFAULT_SPRING_CATALOG,
        DEFAULT_SPRING_ID,
        TORQUE_RESPONSE_ERC_N_GRID,
        TORQUE_RESPONSE_ERC_SAFETY_FACTOR,
        TORQUE_RESPONSE_ERC_MAX_TORQUE_RMSE_NM,
    )
    right_build = build_formula_erc_torque_function(
        FormulaTorqueParameters(
            shared_parameters["a"],
            shared_parameters["b"],
            shared_parameters["c"],
            shared_parameters["d"],
            end_angle_right_rad,
        ),
        "torque_fn_right",
        True,
        right_detachment_envelope,
        DETACHMENT_REQUEST_MARGIN_NM,
        DEFAULT_SPRING_CATALOG,
        DEFAULT_SPRING_ID,
        TORQUE_RESPONSE_ERC_N_GRID,
        TORQUE_RESPONSE_ERC_SAFETY_FACTOR,
        TORQUE_RESPONSE_ERC_MAX_TORQUE_RMSE_NM,
    )
    metadata = {
        "torque_response_a": torque_response_a,
        "torque_response_b": torque_response_b,
        "torque_response_c": torque_response_c,
        "torque_response_d": torque_response_d,
        "torque_response_requested_a": torque_response_a,
        "torque_response_requested_b": torque_response_b,
        "torque_response_requested_c": torque_response_c,
        "torque_response_requested_d": torque_response_d,
        "torque_response_end_angle_left_rad": end_angle_left_rad,
        "torque_response_end_angle_right_rad": end_angle_right_rad,
        "torque_response_end_angle_left_deg": float(np.rad2deg(end_angle_left_rad)),
        "torque_response_end_angle_right_deg": float(np.rad2deg(end_angle_right_rad)),
    }
    metadata.update(side_torque_response_metadata("left", left_build))
    metadata.update(side_torque_response_metadata("right", right_build))
    return metadata, left_build.function_string, right_build.function_string


def build_nmpc_torque_response_payload(x_0_1):
    x_array = validate_x_0_1(x_0_1, False)
    metadata, torque_fn_left, torque_fn_right = build_nmpc_torque_response_payload_cached(
        tuple(float(value) for value in x_array)
    )
    return dict(metadata), torque_fn_left, torque_fn_right


def build_nmpc_torque_response_functions(x_0_1):
    _, torque_fn_left, torque_fn_right = build_nmpc_torque_response_payload(x_0_1)
    return torque_fn_left, torque_fn_right


def torque_response_metadata_from_solution(x_0_1):
    metadata, _, _ = build_nmpc_torque_response_payload(x_0_1)
    return metadata


def actual_param_value_from_x_0_1(param_name, value):
    if param_name in TORQUE_RESPONSE_PARAM_BOUNDS:
        return linear_scaled_value_from_x_0_1(value, TORQUE_RESPONSE_PARAM_BOUNDS[param_name])
    lo, hi = PARAM_BOUNDS[param_name]
    return float(value) * (hi - lo) + lo


def validate_airframe_generation_constraints(x_0_1, rigid):
    x_array = validate_x_0_1(x_0_1, rigid)
    if not rigid:
        build_nmpc_torque_response_functions(x_array)
    return x_array


def repair_control_authority(x_0_1, rigid, planar_fold_axis):
    x_array = validate_x_0_1(x_0_1, rigid)
    airframe = Airframe.from_x_0_1(geometry_x_0_1_from_solution(x_array, rigid))
    repaired_airframe = repair_solution_acceleration(
        airframe,
        target_accel_ratios=CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS,
        verbose=False,
        minimum_arm_torque=CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
        configuration=CONTROL_AUTHORITY_REPAIR_CONFIGURATION,
    )
    repaired_geometry_x_0_1 = np.asarray(repaired_airframe.x_0_1, dtype=float)
    if repaired_geometry_x_0_1.shape != (N_GEOMETRY_PARAMS,):
        raise ValueError(
            f"Expected repaired geometry shape {(N_GEOMETRY_PARAMS,)}, got {repaired_geometry_x_0_1.shape}"
        )
    if planar_fold_axis:
        repair_change = np.max(np.abs(repaired_geometry_x_0_1 - x_array[:N_GEOMETRY_PARAMS]))
        if repair_change > PLANAR_FOLD_AXIS_REPAIR_TOLERANCE:
            raise ValueError(
                "Control authority repair changed a planar fold axis geometry by "
                f"{repair_change}; the fixed planar geometry must satisfy the control authority targets. "
                f"Before: {x_array[:N_GEOMETRY_PARAMS]}, after: {repaired_geometry_x_0_1}"
            )
        return validate_airframe_generation_constraints(x_array, rigid)
    repaired_x_0_1 = np.concatenate(
        [repaired_geometry_x_0_1, x_array[N_GEOMETRY_PARAMS:]]
    )
    return validate_airframe_generation_constraints(repaired_x_0_1, rigid)


def make_config_from_args(args):
    results_dir = RESULTS_ROOT / args.experiment_name
    planar_fold_axis = args.planar_fold_axis is True
    if planar_fold_axis:
        vae_source_checkpoint_path = None
        vae_checkpoint_path = None
        vae_latent_dim, vae_input_dim = None, None
    else:
        vae_source_checkpoint_path = trained_vae_checkpoint_path()
        vae_checkpoint_path = results_dir / VAE_RESULTS_CHECKPOINT_FILENAME
        vae_latent_dim, vae_input_dim = load_trained_vae_metadata(vae_source_checkpoint_path)
    wall_gap_size_tuple = (args.wall_gap_min, args.wall_gap_max)
    population_size = args.population_size
    n_evals_sobol = N_EVALS_SOBOL
    n_evals_bo = N_EVALS_BO
    n_evals_cmaes = N_EVALS_CMAES
    if args.tiny_test is True:
        population_size = 2
        n_evals_sobol = 1
        n_evals_bo = 2
        n_evals_cmaes = 4
        args.num_envs_eval = 16
    cfg = RunConfig(
        experiment_name=args.experiment_name,
        rigid=args.rigid is True,
        planar_fold_axis=planar_fold_axis,
        results_dir=results_dir,
        results_csv=results_dir / "trials.csv",
        plot_path=results_dir / "progress.pdf",
        cma_state_path=results_dir / "cma_state.pkl",
        run_config_path=results_dir / "run_config.json",
        best_solution_path=results_dir / "best_solution.json",
        vae_source_checkpoint_path=vae_source_checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        vae_latent_dim=vae_latent_dim,
        vae_input_dim=vae_input_dim,
        population_size=population_size,
        max_generations=args.max_generations,
        initial_sigma=args.initial_sigma,
        max_sigma=args.max_sigma,
        limit_change_type=args.limit_change_type,
        seed=args.seed,
        eval_seed=args.eval_seed,
        num_envs_eval=args.num_envs_eval,
        num_episodes=args.num_episodes,
        device=args.device,
        wall_gap_size_tuple=wall_gap_size_tuple,
        n_evals_sobol=n_evals_sobol,
        n_evals_bo=n_evals_bo,
        n_evals_cmaes=n_evals_cmaes,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    validate_design_space(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)
    if cfg.planar_fold_axis:
        if cfg.vae_input_dim is not None or cfg.vae_checkpoint_path is not None or cfg.vae_source_checkpoint_path is not None:
            raise ValueError("planar_fold_axis runs must not configure a VAE")
    elif cfg.vae_input_dim != N_GEOMETRY_PARAMS:
        raise ValueError(f"vae_input_dim must be {N_GEOMETRY_PARAMS}, got {cfg.vae_input_dim}")
    validate_positive_int(cfg.population_size, "population_size")
    validate_positive_int(cfg.max_generations, "max_generations")
    validate_positive_float(cfg.initial_sigma, "initial_sigma")
    validate_positive_float(cfg.max_sigma, "max_sigma")
    validate_positive_int(cfg.num_envs_eval, "num_envs_eval")
    validate_positive_int(cfg.num_episodes, "num_episodes")
    validate_positive_int(cfg.n_evals_sobol, "n_evals_sobol")
    validate_positive_int(cfg.n_evals_bo, "n_evals_bo")
    validate_positive_int(cfg.n_evals_cmaes, "n_evals_cmaes")
    validate_wall_gap_tuple(cfg.wall_gap_size_tuple)
    if cfg.limit_change_type not in {CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA, CMA_LIMIT_CHANGE_TYPE_CLIP_SOLUTION}:
        raise ValueError(f"Unknown limit_change_type: {cfg.limit_change_type}")
    if cfg.n_evals_cmaes % cfg.population_size != 0:
        raise ValueError(
            f"N_EVALS_CMAES must be divisible by population_size, got "
            f"{cfg.n_evals_cmaes} and {cfg.population_size}"
        )


def config_to_json_dict(cfg):
    data = {
        "algorithm": ALGORITHM_SOBOL_BO_CMAES,
        "experiment_name": cfg.experiment_name,
        "rigid": cfg.rigid,
        "planar_fold_axis": cfg.planar_fold_axis,
        "results_dir": str(cfg.results_dir),
        "population_size": cfg.population_size,
        "max_generations": cfg.max_generations,
        "initial_sigma": cfg.initial_sigma,
        "max_sigma": cfg.max_sigma,
        "limit_change_type": cfg.limit_change_type,
        "seed": cfg.seed,
        "eval_seed": cfg.eval_seed,
        "num_envs_eval": cfg.num_envs_eval,
        "num_episodes": cfg.num_episodes,
        "device": cfg.device,
        "wall_gap_size_tuple": list(cfg.wall_gap_size_tuple),
        "param_names": run_param_names(cfg.rigid),
        "algorithm_param_names": algorithm_param_names(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis),
        "algorithm_param_bounds": [
            list(bounds)
            for bounds in algorithm_param_bounds(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)
        ],
        "geometry_param_names": GEOMETRY_PARAM_NAMES,
        "control_authority_repair_loss": CONTROL_AUTHORITY_REPAIR_LOSS,
        "control_authority_repair_target_accel_ratios": CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS,
        "control_authority_repair_minimum_arm_torque": CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
        "control_authority_repair_configuration": CONTROL_AUTHORITY_REPAIR_CONFIGURATION,
        "n_evals_sobol": cfg.n_evals_sobol,
        "n_evals_bo": cfg.n_evals_bo,
        "n_evals_cmaes": cfg.n_evals_cmaes,
    }
    if cfg.planar_fold_axis:
        data.update(
            {
                "planar_arm_lengths": PLANAR_ARM_LENGTHS,
                "planar_fold_axis_param_names": PLANAR_FOLD_AXIS_PARAM_NAMES,
                "planar_fold_axis_compensation_param_names": PLANAR_FOLD_AXIS_COMPENSATION_PARAM_NAMES,
                "planar_fold_axis_repair_tolerance": PLANAR_FOLD_AXIS_REPAIR_TOLERANCE,
                "planar_geometry_x_0_1": planar_geometry_x_0_1().tolist(),
            }
        )
    else:
        data.update(
            {
                "vae_source_checkpoint_path": str(cfg.vae_source_checkpoint_path),
                "vae_checkpoint_path": str(cfg.vae_checkpoint_path),
                "vae_latent_dim": cfg.vae_latent_dim,
                "vae_input_dim": cfg.vae_input_dim,
                "vae_decode_latent_bounds": list(VAE_LATENT_PARAM_BOUNDS),
            }
        )
    if not cfg.rigid:
        data.update(
            {
                "torque_response_param_bounds": TORQUE_RESPONSE_PARAM_BOUNDS,
                "torque_response_collision_delta_deg": TORQUE_RESPONSE_COLLISION_DELTA_DEG,
                "torque_response_erc_n_grid": TORQUE_RESPONSE_ERC_N_GRID,
                "torque_response_erc_safety_factor": TORQUE_RESPONSE_ERC_SAFETY_FACTOR,
                "torque_response_erc_max_torque_rmse_nm": TORQUE_RESPONSE_ERC_MAX_TORQUE_RMSE_NM,
            }
        )
    return data


def config_from_json_dict(data, results_dir, requested_max_generations):
    if data["algorithm"] != ALGORITHM_SOBOL_BO_CMAES:
        raise ValueError(f"Saved algorithm does not match {ALGORITHM_SOBOL_BO_CMAES}")
    rigid = data["rigid"]
    if not isinstance(rigid, bool):
        raise TypeError(f"Saved rigid must be a bool, got {rigid!r}")
    planar_fold_axis = data["planar_fold_axis"]
    if not isinstance(planar_fold_axis, bool):
        raise TypeError(f"Saved planar_fold_axis must be a bool, got {planar_fold_axis!r}")
    if data["param_names"] != run_param_names(rigid):
        raise ValueError("Saved param_names do not match current optimization parameter names")
    if planar_fold_axis:
        vae_source_checkpoint_path = None
        vae_checkpoint_path = None
        vae_latent_dim = None
        vae_input_dim = None
        if data["planar_arm_lengths"] != PLANAR_ARM_LENGTHS:
            raise ValueError("Saved planar_arm_lengths do not match current values")
        if data["planar_fold_axis_param_names"] != PLANAR_FOLD_AXIS_PARAM_NAMES:
            raise ValueError("Saved planar_fold_axis_param_names do not match current values")
        if data["planar_fold_axis_compensation_param_names"] != PLANAR_FOLD_AXIS_COMPENSATION_PARAM_NAMES:
            raise ValueError("Saved planar_fold_axis_compensation_param_names do not match current values")
        if float(data["planar_fold_axis_repair_tolerance"]) != PLANAR_FOLD_AXIS_REPAIR_TOLERANCE:
            raise ValueError("Saved planar_fold_axis_repair_tolerance does not match current value")
        if data["planar_geometry_x_0_1"] != planar_geometry_x_0_1().tolist():
            raise ValueError("Saved planar_geometry_x_0_1 does not match current planar geometry")
    else:
        vae_source_checkpoint_path = Path(data["vae_source_checkpoint_path"])
        vae_checkpoint_path = Path(data["vae_checkpoint_path"])
        vae_latent_dim = int(data["vae_latent_dim"])
        vae_input_dim = int(data["vae_input_dim"])
        saved_vae_decode_latent_bounds = tuple(float(value) for value in data["vae_decode_latent_bounds"])
        if saved_vae_decode_latent_bounds != VAE_LATENT_PARAM_BOUNDS:
            raise ValueError("Saved vae_decode_latent_bounds do not match current VAE decode latent bounds")
    if data["algorithm_param_names"] != algorithm_param_names(vae_latent_dim, rigid, planar_fold_axis):
        raise ValueError("Saved algorithm_param_names do not match current algorithm parameter names")
    saved_algorithm_param_bounds = [tuple(bounds) for bounds in data["algorithm_param_bounds"]]
    if saved_algorithm_param_bounds != algorithm_param_bounds(vae_latent_dim, rigid, planar_fold_axis):
        raise ValueError("Saved algorithm_param_bounds do not match current algorithm bounds")
    if data["geometry_param_names"] != GEOMETRY_PARAM_NAMES:
        raise ValueError("Saved geometry_param_names do not match current PARAM_BOUNDS keys")
    if not rigid:
        saved_torque_bounds = {
            name: tuple(bounds)
            for name, bounds in data["torque_response_param_bounds"].items()
        }
        if saved_torque_bounds != TORQUE_RESPONSE_PARAM_BOUNDS:
            raise ValueError("Saved torque_response_param_bounds do not match current bounds")
        if float(data["torque_response_collision_delta_deg"]) != TORQUE_RESPONSE_COLLISION_DELTA_DEG:
            raise ValueError("Saved torque_response_collision_delta_deg does not match current value")
        if int(data["torque_response_erc_n_grid"]) != TORQUE_RESPONSE_ERC_N_GRID:
            raise ValueError("Saved torque_response_erc_n_grid does not match current value")
        if float(data["torque_response_erc_safety_factor"]) != TORQUE_RESPONSE_ERC_SAFETY_FACTOR:
            raise ValueError("Saved torque_response_erc_safety_factor does not match current value")
        if float(data["torque_response_erc_max_torque_rmse_nm"]) != TORQUE_RESPONSE_ERC_MAX_TORQUE_RMSE_NM:
            raise ValueError("Saved torque_response_erc_max_torque_rmse_nm does not match current value")
    if float(data["control_authority_repair_loss"]) != CONTROL_AUTHORITY_REPAIR_LOSS:
        raise ValueError("Saved control_authority_repair_loss does not match current value")
    saved_target_accel_ratios = {
        name: float(value)
        for name, value in data["control_authority_repair_target_accel_ratios"].items()
    }
    if saved_target_accel_ratios != CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS:
        raise ValueError("Saved control_authority_repair_target_accel_ratios do not match current values")
    if float(data["control_authority_repair_minimum_arm_torque"]) != CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE:
        raise ValueError("Saved control_authority_repair_minimum_arm_torque does not match current value")
    if data["control_authority_repair_configuration"] != CONTROL_AUTHORITY_REPAIR_CONFIGURATION:
        raise ValueError("Saved control_authority_repair_configuration does not match current configuration")
    if requested_max_generations is None:
        max_generations = int(data["max_generations"])
    else:
        max_generations = int(requested_max_generations)

    cfg = RunConfig(
        experiment_name=data["experiment_name"],
        rigid=rigid,
        planar_fold_axis=planar_fold_axis,
        results_dir=results_dir,
        results_csv=results_dir / "trials.csv",
        plot_path=results_dir / "progress.pdf",
        cma_state_path=results_dir / "cma_state.pkl",
        run_config_path=results_dir / "run_config.json",
        best_solution_path=results_dir / "best_solution.json",
        vae_source_checkpoint_path=vae_source_checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        vae_latent_dim=vae_latent_dim,
        vae_input_dim=vae_input_dim,
        population_size=int(data["population_size"]),
        max_generations=max_generations,
        initial_sigma=float(data["initial_sigma"]),
        max_sigma=float(data["max_sigma"]),
        limit_change_type=data["limit_change_type"],
        seed=int(data["seed"]),
        eval_seed=int(data["eval_seed"]),
        num_envs_eval=int(data["num_envs_eval"]),
        num_episodes=int(data["num_episodes"]),
        device=data["device"],
        wall_gap_size_tuple=(float(data["wall_gap_size_tuple"][0]), float(data["wall_gap_size_tuple"][1])),
        n_evals_sobol=int(data["n_evals_sobol"]),
        n_evals_bo=int(data["n_evals_bo"]),
        n_evals_cmaes=int(data["n_evals_cmaes"]),
    )
    validate_config(cfg)
    return cfg


def prepare_new_results_dir(cfg):
    if cfg.results_dir.exists():
        raise FileExistsError(f"Results directory already exists: {cfg.results_dir}")
    cfg.results_dir.mkdir(parents=True)
    if not cfg.planar_fold_axis:
        shutil.copy2(cfg.vae_source_checkpoint_path, cfg.vae_checkpoint_path)
    with open(cfg.run_config_path, "w") as f:
        json.dump(config_to_json_dict(cfg), f, indent=2)


def load_resume_config(results_dir, requested_max_generations):
    run_config_path = results_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    with open(run_config_path, "r") as f:
        data = json.load(f)
    saved_results_dir = Path(data["results_dir"]).resolve()
    requested_results_dir = results_dir.resolve()
    if saved_results_dir != requested_results_dir:
        raise ValueError(
            f"Saved results_dir {saved_results_dir} does not match requested {requested_results_dir}"
        )
    return config_from_json_dict(data, results_dir, requested_max_generations)


def read_recorded_generations(results_csv):
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv, usecols=["generation"])
    return {int(generation) for generation in df["generation"].unique()}


def parse_workspace_generation(workspace_name, experiment_name):
    prefix = f"{experiment_name}_gen"
    if not workspace_name.startswith(prefix):
        raise ValueError(f"Workspace name does not start with expected prefix {prefix}: {workspace_name}")
    suffix = workspace_name[len(prefix):]
    generation_text, separator, individual_text = suffix.partition("_ind")
    if separator != "_ind" or not generation_text.isdecimal() or not individual_text.isdecimal():
        raise ValueError(f"Workspace name does not match expected generation/individual pattern: {workspace_name}")
    return int(generation_text)


def remove_unrecorded_resume_workspaces(cfg, start_generation):
    recorded_generations = read_recorded_generations(cfg.results_csv)
    recorded_resume_generations = [
        generation
        for generation in recorded_generations
        if generation >= start_generation
    ]
    if recorded_resume_generations:
        raise RuntimeError(
            f"{cfg.results_csv} already contains generations >= resume generation {start_generation}: "
            f"{sorted(recorded_resume_generations)}"
        )

    workspace_prefix = f"{cfg.experiment_name}_gen"
    stale_workspace_dirs = []
    for workspace_dir in WORKSPACE_ROOT.glob(f"{workspace_prefix}*_ind*"):
        if not workspace_dir.is_dir():
            raise NotADirectoryError(f"Expected workspace path to be a directory: {workspace_dir}")
        generation = parse_workspace_generation(workspace_dir.name, cfg.experiment_name)
        if generation >= start_generation:
            stale_workspace_dirs.append(workspace_dir)

    for workspace_dir in sorted(stale_workspace_dirs):
        print(f"[Resume] Removing unrecorded workspace: {workspace_dir}")
        shutil.rmtree(workspace_dir)


def bo_max_parallelism_from_evaluator(evaluator):
    n_workers = len(evaluator.workers)
    validate_positive_int(n_workers, "n_workers")
    return min(n_workers, BO_BATCH_SIZE)


def create_sobol_bo_cmaes_optimizer(cfg, bo_max_parallelism):
    return SobolBOCMAES(
        cma_sigma0=cfg.initial_sigma,
        cma_max_sigma=cfg.max_sigma,
        cma_opts={
            "popsize": cfg.population_size,
            "seed": cfg.seed,
            "verbose": CMA_VERBOSE,
        },
        param_names=algorithm_param_names(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis),
        param_bounds=algorithm_param_bounds(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis),
        fixed_params={},
        log_params=[],
        limit_change_type=cfg.limit_change_type,
        objective_sense=OBJECTIVE_SENSE_MAXIMIZE,
        cma_sigma_min=None,
        use_generation_only_stop_options=True,
        seed=cfg.seed,
        bo_max_parallelism=bo_max_parallelism,
        n_evals_sobol=cfg.n_evals_sobol,
        n_evals_bo=cfg.n_evals_bo,
        n_evals_cmaes=cfg.n_evals_cmaes,
    )


def optimization_algorithm_show(optimizer, cfg):
    return [
        validate_algorithm_solution(solution, cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)
        for solution in optimizer.ask()
    ]


def tell_solutions(optimizer, algorithm_solutions, fitness_values):
    optimizer.tell([solution.tolist() for solution in algorithm_solutions], fitness_values)


def expand_algorithm_solution(decoder, cfg, algorithm_solution):
    x_algorithm = validate_algorithm_solution(algorithm_solution, cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)
    n_geometry = n_algorithm_geometry_params(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)
    torque_x_0_1 = x_algorithm[n_geometry:]
    if cfg.planar_fold_axis:
        if decoder is not None:
            raise ValueError("planar_fold_axis runs must not use a VAE decoder")
        geometry_x_0_1 = planar_fold_axis_geometry_x_0_1(x_algorithm[:n_geometry])
    else:
        latent_z = vae_latent_z_from_algorithm_solution(x_algorithm, cfg.vae_latent_dim, cfg.rigid)
        geometry_x_0_1 = validate_decoded_geometry_x_0_1(decoder(latent_z))
    expanded_x_0_1 = np.concatenate([geometry_x_0_1, torque_x_0_1])
    return validate_x_0_1(expanded_x_0_1, cfg.rigid)


def expand_algorithm_solutions(decoder, cfg, algorithm_solutions):
    return [
        expand_algorithm_solution(decoder, cfg, solution)
        for solution in algorithm_solutions
    ]


def validate_workspace_available(cfg, generation, individual_id):
    workspace_id = f"{cfg.experiment_name}_gen{generation}_ind{individual_id}"
    workspace_dir = WORKSPACE_ROOT / workspace_id
    if workspace_dir.exists():
        raise FileExistsError(f"Workspace already exists: {workspace_dir}")
    return workspace_id


def submit_solution(evaluator, cfg, x_0_1, generation, individual_id):
    evaluated_x_0_1 = validate_airframe_generation_constraints(x_0_1, cfg.rigid)
    workspace_id = validate_workspace_available(cfg, generation, individual_id)
    geometry_x_0_1 = geometry_x_0_1_from_solution(evaluated_x_0_1, cfg.rigid)
    cmd_args = {
        "wall_gap_size_tuple": list(cfg.wall_gap_size_tuple),
        "num_episodes": cfg.num_episodes,
        "rigid": cfg.rigid,
    }
    if not cfg.rigid:
        _, torque_fn_left, torque_fn_right = build_nmpc_torque_response_payload(evaluated_x_0_1)
        cmd_args["torque_fn_left"] = torque_fn_left
        cmd_args["torque_fn_right"] = torque_fn_right
    evaluator.submit_job(
        x=geometry_x_0_1.tolist(),
        workspace_id=workspace_id,
        parent_checkpoint_path=None,
        gen=generation,
        ind_id=individual_id,
        seed=cfg.eval_seed,
        num_envs_train=0,
        max_epochs=0,
        num_envs_eval=cfg.num_envs_eval,
        cmd_args=cmd_args,
        until_no_improvement=False,
        training_method="nmpc",
        n_repeat_train=None,
    )


def wait_for_population_results(evaluator, cfg, repaired_solutions, generation):
    while evaluator.check_jobs_still_running():
        time.sleep(10)

    partial_rows = []
    fitness_values = []

    for individual_id, x_0_1 in enumerate(repaired_solutions):
        key = (generation, individual_id)
        if key not in evaluator.completed_results:
            raise RuntimeError(f"Missing completed result for generation {generation}, individual {individual_id}")

        result = evaluator.completed_results.pop(key)
        metrics = result["metrics"]
        eval_time = result["eval_time"]
        workspace_id = result["workspace_id"]

        for metric_key in REQUIRED_METRIC_KEYS:
            metrics[metric_key]

        fitness_values.append(float(metrics["success_percentage"]))
        partial_rows.append((individual_id, x_0_1, metrics, eval_time, workspace_id))

    return np.asarray(fitness_values, dtype=float), partial_rows


def make_result_row(
    cfg,
    generation,
    phase,
    individual_id,
    algorithm_solution,
    expanded_x_0_1,
    repaired_x_0_1,
    metrics,
    eval_time,
    generation_time,
    workspace_id,
    sigma,
):
    validated_algorithm_solution = validate_algorithm_solution(
        algorithm_solution, cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis
    )
    expanded_x_0_1 = validate_x_0_1(expanded_x_0_1, cfg.rigid)
    repaired_x_0_1 = validate_airframe_generation_constraints(repaired_x_0_1, cfg.rigid)
    row = {
        "algorithm": ALGORITHM_SOBOL_BO_CMAES,
        "rigid": cfg.rigid,
        "planar_fold_axis": cfg.planar_fold_axis,
        "phase": phase,
        "generation": generation,
        "individual_id": individual_id,
        "success_rate": float(metrics["success_percentage"]),
        "crash_percentage": float(metrics["crash_percentage"]),
        "timeout_percentage": float(metrics["timeout_percentage"]),
        "mean_position_error": float(metrics["mean_position_error"]),
        "mean_velocity_error": float(metrics["mean_velocity_error"]),
        "mean_attitude_error": float(metrics["mean_attitude_error"]),
        "mean_control_effort": float(metrics["mean_control_effort"]),
        "eval_time_seconds": eval_time,
        "generation_time_seconds": generation_time,
        "workspace_id": str(Path("cache") / "workspace_local" / workspace_id),
        "sigma": sigma,
        "eval_seed": cfg.eval_seed,
        "num_envs_eval": cfg.num_envs_eval,
        "num_episodes": cfg.num_episodes,
        "wall_gap_min": cfg.wall_gap_size_tuple[0],
        "wall_gap_max": cfg.wall_gap_size_tuple[1],
        "control_authority_repair_loss": CONTROL_AUTHORITY_REPAIR_LOSS,
        "control_authority_repair_minimum_arm_torque": CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
    }
    if not cfg.rigid:
        torque_response_metadata, torque_fn_left, torque_fn_right = build_nmpc_torque_response_payload(repaired_x_0_1)
        row.update(torque_response_metadata)
        row["torque_fn_left"] = torque_fn_left
        row["torque_fn_right"] = torque_fn_right
    for i, value in enumerate(repaired_x_0_1):
        row[f"x_{i}"] = float(value)
        row[f"repaired_x_{i}"] = float(value)
    for i, value in enumerate(expanded_x_0_1):
        row[f"expanded_x_{i}"] = float(value)
    for i, value in enumerate(validated_algorithm_solution):
        row[f"algorithm_x_{i}"] = float(value)
    return row


def evaluate_population(evaluator, cfg, decoder, algorithm_solutions, generation, phase, sigma):
    generation_start_time = time.time()
    expanded_solutions = expand_algorithm_solutions(decoder, cfg, algorithm_solutions)
    repaired_solutions = [
        repair_control_authority(solution, cfg.rigid, cfg.planar_fold_axis)
        for solution in expanded_solutions
    ]

    for individual_id, x_0_1 in enumerate(repaired_solutions):
        print(f"\n[Generation {generation}] Queueing individual {individual_id}/{len(algorithm_solutions) - 1}")
        submit_solution(evaluator, cfg, x_0_1, generation, individual_id)

    fitness_values, partial_rows = wait_for_population_results(evaluator, cfg, repaired_solutions, generation)

    generation_time = time.time() - generation_start_time
    rows = [
        make_result_row(
            cfg=cfg,
            generation=generation,
            phase=phase,
            individual_id=individual_id,
            algorithm_solution=algorithm_solutions[individual_id],
            expanded_x_0_1=expanded_solutions[individual_id],
            repaired_x_0_1=x_0_1,
            metrics=metrics,
            eval_time=eval_time,
            generation_time=generation_time,
            workspace_id=workspace_id,
            sigma=sigma,
        )
        for individual_id, x_0_1, metrics, eval_time, workspace_id in partial_rows
    ]

    return np.asarray(fitness_values, dtype=float), rows, expanded_solutions, repaired_solutions


def append_rows(cfg, rows):
    df_new = pd.DataFrame(rows)
    if cfg.results_csv.exists():
        df_existing = pd.read_csv(cfg.results_csv)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(cfg.results_csv, index=False)
    print(f"[Results] Saved {len(rows)} rows to {cfg.results_csv}")


def write_best_solution(cfg):
    df = pd.read_csv(cfg.results_csv)
    sort_columns = ["success_rate", "mean_position_error"]
    ascending = [False, True]
    best_row = df.sort_values(sort_columns, ascending=ascending).iloc[0]
    n_params = len(run_param_names(cfg.rigid))
    x_0_1 = [float(best_row[f"x_{i}"]) for i in range(n_params)]
    repaired_x_0_1 = [float(best_row[f"repaired_x_{i}"]) for i in range(n_params)]
    expanded_x_0_1 = [float(best_row[f"expanded_x_{i}"]) for i in range(n_params)]
    algorithm_solution = [
        float(best_row[f"algorithm_x_{i}"])
        for i in range(
            n_algorithm_geometry_params(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis)
            + n_torque_response_params(cfg.rigid)
        )
    ]
    payload = {
        "rigid": cfg.rigid,
        "planar_fold_axis": cfg.planar_fold_axis,
        "generation": int(best_row["generation"]),
        "individual_id": int(best_row["individual_id"]),
        "success_rate": float(best_row["success_rate"]),
        "mean_position_error": float(best_row["mean_position_error"]),
        "workspace_id": best_row["workspace_id"],
        "x_0_1": x_0_1,
        "repaired_x_0_1": repaired_x_0_1,
        "expanded_x_0_1": expanded_x_0_1,
        "algorithm_solution": algorithm_solution,
        "algorithm_param_names": algorithm_param_names(cfg.vae_latent_dim, cfg.rigid, cfg.planar_fold_axis),
        "geometry_x_0_1": geometry_x_0_1_from_solution(repaired_x_0_1, cfg.rigid).tolist(),
        "expanded_geometry_x_0_1": geometry_x_0_1_from_solution(expanded_x_0_1, cfg.rigid).tolist(),
        "control_authority_repair_loss": CONTROL_AUTHORITY_REPAIR_LOSS,
        "control_authority_repair_minimum_arm_torque": CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
    }
    if not cfg.rigid:
        torque_response_metadata, torque_fn_left, torque_fn_right = build_nmpc_torque_response_payload(repaired_x_0_1)
        payload.update(torque_response_metadata)
        payload["torque_fn_left"] = torque_fn_left
        payload["torque_fn_right"] = torque_fn_right
    with open(cfg.best_solution_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Best] success={payload['success_rate']:.2f}% workspace={payload['workspace_id']}")


def load_plot_config(results_csv):
    run_config_path = results_csv.parent / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found next to results CSV: {run_config_path}")
    with open(run_config_path, "r") as f:
        data = json.load(f)
    if "algorithm" not in data:
        raise KeyError(f"run_config.json must contain algorithm for progress plotting: {run_config_path}")
    if data["algorithm"] != ALGORITHM_SOBOL_BO_CMAES:
        raise ValueError(f"Expected algorithm {ALGORITHM_SOBOL_BO_CMAES}, got {data['algorithm']}")
    if "param_names" not in data:
        raise KeyError(f"run_config.json must contain param_names for progress plotting: {run_config_path}")
    param_names = list(data["param_names"])
    if param_names != run_param_names(data["rigid"]):
        raise ValueError("Saved param_names do not match current optimization parameter names")
    return data


def validate_plot_dataframe(df, param_names, results_csv):
    if len(df) == 0:
        raise ValueError(f"No rows in results CSV: {results_csv}")

    x_columns = [f"x_{i}" for i in range(len(param_names))]
    missing_columns = [
        column
        for column in PLOT_REQUIRED_COLUMNS + x_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Results CSV is missing required plotting columns: {missing_columns}")

    unknown_phases = sorted(set(df["phase"]) - set(PLOT_PHASE_ORDER))
    if unknown_phases:
        raise ValueError(f"Unknown optimization phases in results CSV: {unknown_phases}")
    unknown_algorithms = sorted(set(df["algorithm"]) - {ALGORITHM_SOBOL_BO_CMAES})
    if unknown_algorithms:
        raise ValueError(f"Unknown algorithms in results CSV: {unknown_algorithms}")

    numeric_columns = [
        "generation",
        "individual_id",
        "success_rate",
        "crash_percentage",
        "timeout_percentage",
        "mean_position_error",
        "mean_velocity_error",
        "mean_attitude_error",
        "mean_control_effort",
        "eval_time_seconds",
        "generation_time_seconds",
        "sigma",
    ] + x_columns
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    finite_columns = [column for column in numeric_columns if column != "sigma"]
    for column in finite_columns:
        non_finite_mask = ~np.isfinite(df[column].to_numpy(dtype=float))
        if np.any(non_finite_mask):
            bad_indices = df.index[non_finite_mask].tolist()
            raise ValueError(f"Column {column} contains non-finite values at rows {bad_indices}")

    for group_title, _, group_param_names in plot_parameter_groups(param_names):
        missing_params = sorted(set(group_param_names) - set(param_names))
        if missing_params:
            raise ValueError(f"Plot group {group_title} references unknown parameters: {missing_params}")


def plot_parameter_groups(param_names):
    return [
        group
        for group in PLOT_PARAMETER_GROUPS
        if not set(group[2]).issubset(TORQUE_RESPONSE_PARAM_BOUNDS) or set(group[2]).issubset(param_names)
    ]


def ordered_plot_dataframe(df):
    ordered_df = df.sort_values(["generation", "individual_id"], kind="mergesort").reset_index(drop=True)
    ordered_df["evaluation"] = np.arange(1, len(ordered_df) + 1)
    return ordered_df


def plot_row_rank_key(row):
    return (
        float(row["success_rate"]),
        -float(row["mean_position_error"]),
        -float(row["mean_velocity_error"]),
        -float(row["mean_attitude_error"]),
        -float(row["mean_control_effort"]),
    )


def plot_row_is_better(row, best_row):
    return plot_row_rank_key(row) > plot_row_rank_key(best_row)


def select_best_plot_row(df):
    if len(df) == 0:
        raise ValueError("Cannot select best row from an empty dataframe")
    best_row = None
    for _, row in df.iterrows():
        if best_row is None or plot_row_is_better(row, best_row):
            best_row = row.copy()
    return best_row


def cumulative_best_dataframe(ordered_df):
    rows = []
    best_row = None
    for _, row in ordered_df.iterrows():
        if best_row is None or plot_row_is_better(row, best_row):
            best_row = row.copy()
        cumulative_row = best_row.copy()
        cumulative_row["evaluation"] = row["evaluation"]
        rows.append(cumulative_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def generation_best_dataframe(ordered_df):
    rows = []
    for _, generation_df in ordered_df.groupby("generation", sort=True):
        rows.append(select_best_plot_row(generation_df))
    return pd.DataFrame(rows).reset_index(drop=True)


def phase_spans(ordered_df):
    spans = []
    current_phase = str(ordered_df.iloc[0]["phase"])
    start_evaluation = int(ordered_df.iloc[0]["evaluation"])
    previous_evaluation = start_evaluation
    for _, row in ordered_df.iloc[1:].iterrows():
        row_phase = str(row["phase"])
        row_evaluation = int(row["evaluation"])
        if row_phase != current_phase:
            spans.append((current_phase, start_evaluation, previous_evaluation))
            current_phase = row_phase
            start_evaluation = row_evaluation
        previous_evaluation = row_evaluation
    spans.append((current_phase, start_evaluation, previous_evaluation))
    return spans


def add_phase_background(ax, spans):
    for phase, start_evaluation, end_evaluation in spans:
        color = PLOT_PHASE_COLORS[phase]
        ax.axvspan(
            start_evaluation - 0.5,
            end_evaluation + 0.5,
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        ax.text(
            (start_evaluation + end_evaluation) / 2.0,
            0.98,
            phase,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )


def format_param_label(param_name):
    return param_name.replace("_", " ")


def parameter_axis_token(param_name):
    axis_tokens = [token for token in param_name.split("_") if token in PLOT_AXIS_COLORS]
    if len(axis_tokens) > 1:
        raise ValueError(f"Parameter name contains more than one axis token: {param_name}")
    if len(axis_tokens) == 1:
        return axis_tokens[0]
    return None


def parameter_position_token(param_name):
    position_tokens = [token for token in param_name.split("_") if token in PLOT_POSITION_LINESTYLES]
    if len(position_tokens) > 1:
        raise ValueError(f"Parameter name contains more than one position token: {param_name}")
    if len(position_tokens) == 1:
        return position_tokens[0]
    return None


def parameter_plot_style(param_name):
    axis_token = parameter_axis_token(param_name)
    position_token = parameter_position_token(param_name)
    if axis_token is None:
        color = PLOT_NON_AXIS_COLOR
    else:
        color = PLOT_AXIS_COLORS[axis_token]
    if position_token is None:
        linestyle = PLOT_NO_POSITION_LINESTYLE
    else:
        linestyle = PLOT_POSITION_LINESTYLES[position_token]
    return color, linestyle


def actual_param_series(df, param_names, param_name):
    param_idx = param_names.index(param_name)
    return [
        actual_param_value_from_x_0_1(param_name, value)
        for value in df[f"x_{param_idx}"]
    ]


def phase_summary_rows(ordered_df):
    rows = []
    for phase in PLOT_PHASE_ORDER:
        phase_df = ordered_df[ordered_df["phase"] == phase]
        if len(phase_df) == 0:
            continue
        best_row = select_best_plot_row(phase_df)
        generation_times = phase_df.drop_duplicates("generation")["generation_time_seconds"]
        rows.append(
            [
                phase,
                f"{len(phase_df)}",
                f"{phase_df['generation'].nunique()}",
                f"{float(best_row['success_rate']):.2f}",
                f"{int(best_row['generation'])}",
                f"{int(best_row['individual_id'])}",
                f"{phase_df['eval_time_seconds'].mean() / 60.0:.1f}",
                f"{generation_times.sum() / 3600.0:.2f}",
            ]
        )
    return rows


def save_overview_page(pdf, ordered_df, best_per_eval, generation_best, spans, experiment_name):
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(f"{experiment_name} Progress", fontsize=15)

    for ax in axes:
        add_phase_background(ax, spans)
        ax.grid(True, alpha=0.3)

    for phase in PLOT_PHASE_ORDER:
        phase_df = ordered_df[ordered_df["phase"] == phase]
        if len(phase_df) == 0:
            continue
        axes[0].scatter(
            phase_df["evaluation"],
            phase_df["success_rate"],
            s=24,
            color=PLOT_PHASE_COLORS[phase],
            alpha=0.55,
            label=f"{phase} candidates",
        )
    axes[0].plot(
        best_per_eval["evaluation"],
        best_per_eval["success_rate"],
        color="black",
        linewidth=2.0,
        label="Best so far",
    )
    cmaes_generation_best = generation_best[generation_best["phase"] == SOBOL_BO_CMAES_STAGE_CMAES]
    axes[0].scatter(
        cmaes_generation_best["evaluation"],
        cmaes_generation_best["success_rate"],
        marker="x",
        s=42,
        color="black",
        label="Best in generation",
    )
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(ordered_df["evaluation"], ordered_df["crash_percentage"], color="#d62728", alpha=0.65, label="Crash")
    axes[1].plot(ordered_df["evaluation"], ordered_df["timeout_percentage"], color="#9467bd", alpha=0.65, label="Timeout")
    axes[1].set_ylabel("Failure rate (%)")
    axes[1].legend(fontsize=8)

    axes[2].plot(best_per_eval["evaluation"], best_per_eval["mean_position_error"], label="Position")
    axes[2].plot(best_per_eval["evaluation"], best_per_eval["mean_velocity_error"], label="Velocity")
    axes[2].plot(best_per_eval["evaluation"], best_per_eval["mean_attitude_error"], label="Attitude")
    axes[2].plot(best_per_eval["evaluation"], best_per_eval["mean_control_effort"], label="Control effort")
    axes[2].set_ylabel("Best-so-far metric")
    axes[2].legend(ncol=2, fontsize=8)

    generation_summary = ordered_df.drop_duplicates("generation").sort_values("generation")
    generation_end_eval = ordered_df.groupby("generation")["evaluation"].max().reset_index()
    generation_summary = generation_summary.merge(generation_end_eval, on="generation", suffixes=("", "_end"))
    cumulative_time_hours = generation_summary["generation_time_seconds"].cumsum() / 3600.0
    axes[3].plot(
        generation_summary["evaluation_end"],
        cumulative_time_hours,
        marker="o",
        color="#1f77b4",
        label="Cumulative batch time",
    )
    axes[3].set_ylabel("Time (h)")
    sigma_mask = np.isfinite(generation_summary["sigma"].to_numpy(dtype=float))
    sigma_axis = axes[3].twinx()
    if np.any(sigma_mask):
        sigma_axis.plot(
            generation_summary.loc[sigma_mask, "evaluation_end"],
            generation_summary.loc[sigma_mask, "sigma"],
            marker="s",
            color="#2ca02c",
            label="CMA-ES sigma",
        )
    sigma_axis.set_ylabel("Sigma")
    axes[3].set_xlabel("Evaluation")
    axes[3].legend(loc="upper left", fontsize=8)
    if np.any(sigma_mask):
        sigma_axis.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_phase_summary_page(pdf, ordered_df, experiment_name):
    summary_rows = phase_summary_rows(ordered_df)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f"{experiment_name} Algorithm Summary", fontsize=15)

    phases = [row[0] for row in summary_rows]
    eval_counts = [int(row[1]) for row in summary_rows]
    best_success = [float(row[3]) for row in summary_rows]
    colors = [PLOT_PHASE_COLORS[phase] for phase in phases]
    axes[0].bar(phases, eval_counts, color=colors, alpha=0.75, label="Evaluations")
    axes[0].set_ylabel("Evaluations")
    axes[0].grid(True, axis="y", alpha=0.3)
    best_axis = axes[0].twinx()
    best_axis.plot(phases, best_success, color="black", marker="o", label="Best success")
    best_axis.set_ylabel("Best success rate (%)")

    axes[1].axis("off")
    table = axes[1].table(
        cellText=summary_rows,
        colLabels=[
            "Algorithm",
            "Evaluations",
            "Batches",
            "Best success (%)",
            "Best batch",
            "Best individual",
            "Mean eval (min)",
            "Batch time (h)",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_parameter_page(pdf, best_per_eval, param_names, spans, experiment_name):
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    fig.suptitle(f"{experiment_name} Best-So-Far Design Parameters", fontsize=15)
    flat_axes = axes.ravel()

    parameter_groups = plot_parameter_groups(param_names)
    for ax_idx, (group_title, ylabel, group_param_names) in enumerate(parameter_groups):
        ax = flat_axes[ax_idx]
        add_phase_background(ax, spans)
        for param_name in group_param_names:
            color, linestyle = parameter_plot_style(param_name)
            ax.plot(
                best_per_eval["evaluation"],
                actual_param_series(best_per_eval, param_names, param_name),
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                label=format_param_label(param_name),
            )
        ax.set_title(group_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    for ax in flat_axes[len(parameter_groups):]:
        ax.axis("off")
    flat_axes[len(parameter_groups) - 1].set_xlabel("Evaluation")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_parameter_heatmap_page(pdf, best_per_eval, param_names, experiment_name):
    x_columns = [f"x_{i}" for i in range(len(param_names))]
    matrix = best_per_eval[x_columns].to_numpy(dtype=float).T
    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"{experiment_name} Normalized Best-So-Far Parameter Values")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Parameter")
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_yticklabels([format_param_label(name) for name in param_names], fontsize=8)
    tick_count = min(10, len(best_per_eval))
    tick_positions = np.unique(np.linspace(0, len(best_per_eval) - 1, tick_count, dtype=int))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(best_per_eval.iloc[idx]["evaluation"])) for idx in tick_positions])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Normalized value")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_results(results_csv, plot_path):
    plot_config = load_plot_config(results_csv)
    param_names = list(plot_config["param_names"])
    df = pd.read_csv(results_csv)
    validate_plot_dataframe(df, param_names, results_csv)

    ordered_df = ordered_plot_dataframe(df)
    best_per_eval = cumulative_best_dataframe(ordered_df)
    generation_best = generation_best_dataframe(ordered_df)
    spans = phase_spans(ordered_df)
    experiment_name = str(plot_config["experiment_name"])

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(plot_path) as pdf:
        save_overview_page(pdf, ordered_df, best_per_eval, generation_best, spans, experiment_name)
        save_phase_summary_page(pdf, ordered_df, experiment_name)
        save_parameter_page(pdf, best_per_eval, param_names, spans, experiment_name)
        save_parameter_heatmap_page(pdf, best_per_eval, param_names, experiment_name)

    print(f"[Plot] Saved to {plot_path}")


def save_cma_state(cfg, optimizer, next_generation, best_fitness, best_solution):
    serialized_best_solution = None
    if best_solution is not None:
        serialized_best_solution = validate_algorithm_solution(
            best_solution,
            cfg.vae_latent_dim,
            cfg.rigid,
            cfg.planar_fold_axis,
        ).tolist()
    optimizer.save_state(
        cfg.cma_state_path,
        next_generation,
        best_fitness,
        serialized_best_solution,
    )


def load_cma_state(cfg):
    if not cfg.cma_state_path.exists():
        raise FileNotFoundError(f"CMA state not found: {cfg.cma_state_path}")
    state = OptimizationAlgorithm.load_state(cfg.cma_state_path)
    if state["best_solution"] is not None:
        state["best_solution"] = validate_algorithm_solution(
            state["best_solution"],
            cfg.vae_latent_dim,
            cfg.rigid,
            cfg.planar_fold_axis,
        )
    return state


def initialize_sobol_bo_cmaes(cfg, evaluator):
    print(f"\n[Initialization] Starting {ALGORITHM_SOBOL_BO_CMAES}")
    bo_max_parallelism = bo_max_parallelism_from_evaluator(evaluator)
    print(f"[Initialization] BO max parallelism: {bo_max_parallelism}")
    optimizer = create_sobol_bo_cmaes_optimizer(cfg, bo_max_parallelism)
    return optimizer, 0, -np.inf, None


def run_optimization_loop(evaluator, cfg, optimizer, decoder, start_generation, best_fitness, best_solution):
    generation = start_generation
    while not optimizer.is_finished:
        phase = optimizer.stage
        print("\n" + "=" * 80)
        print(f"BATCH {generation} | PHASE {phase}")
        optimizer_mean = optimizer_mean_algorithm_space(optimizer, cfg)
        print(f"Mean algorithm solution first 8 values: {[round(v, 4) for v in optimizer_mean[:8]]}")
        print(f"Sigma: {optimizer.sigma:.6f}")
        print(f"Best fitness so far: {best_fitness:.2f}%")
        print("=" * 80)

        algorithm_solutions = optimization_algorithm_show(optimizer, cfg)
        fitness_values, rows, expanded_solutions, repaired_solutions = evaluate_population(
            evaluator,
            cfg,
            decoder,
            algorithm_solutions,
            generation,
            phase,
            optimizer.sigma,
        )
        tell_solutions(optimizer, algorithm_solutions, fitness_values)
        for row in rows:
            row["sigma"] = float(optimizer.sigma)

        generation_best_idx = int(np.argmax(fitness_values))
        generation_best_fitness = float(fitness_values[generation_best_idx])
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_solution = algorithm_solutions[generation_best_idx]
            print(f"[Best] New best fitness: {best_fitness:.2f}%")
        else:
            print(f"[Generation] Best fitness: {generation_best_fitness:.2f}%")

        append_rows(cfg, rows)
        save_cma_state(cfg, optimizer, generation + 1, best_fitness, best_solution)
        write_best_solution(cfg)
        plot_results(cfg.results_csv, cfg.plot_path)

        generation += 1

    print("\n" + "=" * 80)
    print(f"Optimization finished. Best fitness: {best_fitness:.2f}%")
    print(f"Results CSV: {cfg.results_csv}")
    print(f"Best solution JSON: {cfg.best_solution_path}")
    print("=" * 80)


def load_run_decoder(cfg):
    if cfg.planar_fold_axis:
        return None
    return load_vae_decoder_from_checkpoint(cfg.vae_checkpoint_path)


def solve_command(args):
    cfg = make_config_from_args(args)
    prepare_new_results_dir(cfg)
    decoder = load_run_decoder(cfg)
    evaluator = RemoteEvaluator(worker_type_list=args.worker_type_list, job_timeout=NMPC_JOB_TIMEOUT_SECONDS)
    optimizer, start_generation, best_fitness, best_solution = initialize_sobol_bo_cmaes(cfg, evaluator)
    save_cma_state(cfg, optimizer, start_generation, best_fitness, best_solution)
    run_optimization_loop(evaluator, cfg, optimizer, decoder, start_generation, best_fitness, best_solution)


def resume_command(args):
    results_dir = Path(args.results_dir)
    cfg = load_resume_config(results_dir, args.max_generations)
    decoder = load_run_decoder(cfg)
    evaluator = RemoteEvaluator(worker_type_list=args.worker_type_list, job_timeout=NMPC_JOB_TIMEOUT_SECONDS)
    if cfg.cma_state_path.exists():
        state = load_cma_state(cfg)
    else:
        if cfg.results_csv.exists():
            raise FileNotFoundError(
                f"Optimization results exist without an optimizer checkpoint: {cfg.cma_state_path}"
            )
        optimizer, start_generation, best_fitness, best_solution = initialize_sobol_bo_cmaes(cfg, evaluator)
        save_cma_state(cfg, optimizer, start_generation, best_fitness, best_solution)
        state = {
            "optimizer": optimizer,
            "next_generation": start_generation,
            "best_fitness": best_fitness,
            "best_solution": best_solution,
        }
        print(f"[Resume] Initialized prepared run: {cfg.results_dir}")
    start_generation = int(state["next_generation"])
    remove_unrecorded_resume_workspaces(cfg, start_generation)
    state["optimizer"].bo_max_parallelism = bo_max_parallelism_from_evaluator(evaluator)
    print(f"[Resume] BO max parallelism: {state['optimizer'].bo_max_parallelism}")
    run_optimization_loop(
        evaluator,
        cfg,
        state["optimizer"],
        decoder,
        start_generation,
        float(state["best_fitness"]),
        state["best_solution"],
    )


def main():
    args = parse_args()
    if args.f_over_time_results_dir is not None:
        from scripts.video_f_over_time_optimize_design_behaviour_based_nmpc_cma_es import generate_f_over_time_animation

        mp4_path = generate_f_over_time_animation(args.f_over_time_results_dir)
        print(f"[Animation] Saved to {mp4_path}")
    elif args.command == "solve":
        solve_command(args)
    elif args.command == "resume":
        resume_command(args)
    elif args.command == "plot":
        plot_results(Path(args.results_csv), Path(args.plot_path))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
