from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from airframe_encoding import Airframe, PARAM_BOUNDS, PARAM_NAMES
from airframe_validations import repair_solution_acceleration
from optimize_design_behaviour_based_nmpc_cma_es import (
    CONTROL_AUTHORITY_REPAIR_CONFIGURATION,
    CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
    CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS,
)


RESULTS_DIR = PROJECT_ROOT / "results"
DIM_REDUCTION_MASK_RESULTS_DIR = RESULTS_DIR / "dim_reduction_mask"
DATA_NPZ = DIM_REDUCTION_MASK_RESULTS_DIR / "vae_samples.npz"
CHECKPOINT_PATH = DIM_REDUCTION_MASK_RESULTS_DIR / "best_model_weights.pth"
TRAINING_CONFIG_JSON = DIM_REDUCTION_MASK_RESULTS_DIR / "training_config.json"
TENSORBOARD_DIR = DIM_REDUCTION_MASK_RESULTS_DIR / "tensorboard"
REPORT_PDF = DIM_REDUCTION_MASK_RESULTS_DIR / "vae_reconstruction_and_coverage_report.pdf"
ANALYSIS_NPZ = DIM_REDUCTION_MASK_RESULTS_DIR / "vae_analysis_arrays.npz"
LATENT_DIMENSION_SWEEP_DIR = DIM_REDUCTION_MASK_RESULTS_DIR / "latent_dimension_sweep"
LATENT_DIMENSION_SWEEP_METRICS_JSON = LATENT_DIMENSION_SWEEP_DIR / "latent_dimension_sweep_metrics.json"
LATENT_DIMENSION_SWEEP_METRICS_NPZ = LATENT_DIMENSION_SWEEP_DIR / "latent_dimension_sweep_metrics.npz"
LATENT_DIMENSION_SWEEP_PDF = LATENT_DIMENSION_SWEEP_DIR / "latent_dimension_sweep_metrics.pdf"

GET_SAMPLES_N_SAMPLES = 20_000
GET_SAMPLES_SEED = 42
WORKER_CPU_IDS = (0, 1, 2, 3, 4, 5)

DEVICE = "cpu"
LATENT_DIM = 9
LATENT_DIMENSION_SWEEP_MIN = 2
LATENT_DIMENSION_SWEEP_MAX = 16
HIDDEN_DIMENSIONS = (256, 128, 64)
TRAIN_FRACTION = 0.8
BATCH_SIZE = 128
MAX_EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 1e-3
BETA = 1e-4
MIN_IMPROVEMENT = 1e-8
TRAIN_SEED = 42
NUM_DATALOADER_WORKERS = 0

ANALYZE_GENERATED_N_SAMPLES = 20_000
ANALYZE_GENERATED_SEED = 123
NEAREST_DISTANCE_QUERY_CHUNK_SIZE = 512
SURFACE_COVERAGE_GRID_SIZE = 40

NORMALIZED_BOUNDS_TOLERANCE = 1e-9
VALID_ACTIONS = {"--get_samples", "--train_vae", "--analyze_vae", "--every_latent_dimension", "--decode"}
CANONICALIZATION_NUMERICAL_TOLERANCE = 1e-9
MOTOR_SELF_ROTATION_PARAM_GROUPS = (
    ("front_prop_rotation_x", "front_prop_rotation_y", "front_prop_rotation_z"),
    ("back_prop_rotation_x", "back_prop_rotation_y", "back_prop_rotation_z"),
    ("side_motor_rotation_x_left", "side_motor_rotation_y_left", "side_motor_rotation_z_left"),
    ("side_motor_rotation_x_right", "side_motor_rotation_y_right", "side_motor_rotation_z_right"),
)


class AirframeVAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dimensions):
        super().__init__()
        encoder_layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dimensions:
            encoder_layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(torch.nn.ELU())
            previous_dim = hidden_dim

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.mu_layer = torch.nn.Linear(previous_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(previous_dim, latent_dim)

        decoder_layers = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dimensions):
            decoder_layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            decoder_layers.append(torch.nn.ELU())
            previous_dim = hidden_dim
        decoder_layers.append(torch.nn.Linear(previous_dim, input_dim))
        decoder_layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def encode(self, x):
        encoded = self.encoder(x)
        return self.mu_layer(encoded), self.logvar_layer(encoded)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class AirframeDecoderFromFile:
    def __init__(self, checkpoint_path, checkpoint, decoder_module):
        self.checkpoint_path = Path(checkpoint_path)
        self.latent_dim = int(checkpoint["latent_dim"])
        self.input_dim = int(checkpoint["input_dim"])
        self.hidden_dimensions = tuple(int(value) for value in checkpoint["hidden_dimensions"])
        self.param_names = tuple(str(value) for value in checkpoint["param_names"])
        self.decoder_module = decoder_module
        self.decoder_module.eval()

    def __call__(self, latent_z):
        z_array = np.asarray(latent_z, dtype=np.float32)
        single_sample = False
        if z_array.ndim == 1:
            single_sample = True
            z_array = z_array.reshape(1, z_array.shape[0])
        elif z_array.ndim != 2:
            raise ValueError(f"latent_z must be 1D or 2D, got shape {z_array.shape}")
        if z_array.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent_z width must match checkpoint latent_dim={self.latent_dim}, got shape {z_array.shape}"
            )
        if not np.all(np.isfinite(z_array)):
            raise ValueError(f"latent_z contains non-finite values: {z_array}")

        z_tensor = torch.as_tensor(z_array, dtype=torch.float32)
        with torch.no_grad():
            decoded = self.decoder_module(z_tensor).detach().cpu().numpy()
        if decoded.shape != (z_array.shape[0], self.input_dim):
            raise RuntimeError(
                f"decoded output shape must be {(z_array.shape[0], self.input_dim)}, got {decoded.shape}"
            )
        if not np.all(np.isfinite(decoded)):
            raise RuntimeError(f"decoded output contains non-finite values: {decoded}")
        if np.min(decoded) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(decoded) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
            raise RuntimeError(f"decoded output contains values outside [0, 1]: {decoded}")
        decoded = np.clip(decoded, 0.0, 1.0)
        if single_sample:
            return decoded[0]
        return decoded


def beta_vae_loss(x_hat, x, mu, logvar, beta):
    reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = reconstruction_loss + beta * kld_loss
    return total_loss, reconstruction_loss, kld_loss


def validate_hidden_dimensions(hidden_dimensions):
    dimensions = tuple(int(value) for value in hidden_dimensions)
    if len(dimensions) < 1:
        raise ValueError("hidden_dimensions must contain at least one layer width")
    invalid_dimensions = [value for value in dimensions if value <= 0]
    if invalid_dimensions:
        raise ValueError(f"hidden_dimensions must be positive, got {invalid_dimensions}")
    return dimensions


def build_airframe_vae(input_dim, latent_dim, hidden_dimensions, device):
    if int(input_dim) <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if int(latent_dim) <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    model = AirframeVAE(
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        hidden_dimensions=validate_hidden_dimensions(hidden_dimensions),
    )
    return model.to(torch.device(device))


def serialize_decoder_module(model, latent_dim):
    was_training = model.training
    model.eval()
    model_device = next(model.parameters()).device
    example_z = torch.zeros((1, int(latent_dim)), dtype=torch.float32, device=model_device)
    traced_decoder = torch.jit.trace(model.decoder, example_z)
    buffer = io.BytesIO()
    torch.jit.save(traced_decoder, buffer)
    serialized_decoder = buffer.getvalue()
    if was_training:
        model.train()
    return serialized_decoder


def save_airframe_vae_checkpoint(
    checkpoint_path,
    model,
    input_dim,
    latent_dim,
    hidden_dimensions,
    param_names,
    training_config,
    training_history,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "torchscript_decoder_bytes": serialize_decoder_module(model, latent_dim),
        "input_dim": int(input_dim),
        "latent_dim": int(latent_dim),
        "hidden_dimensions": list(validate_hidden_dimensions(hidden_dimensions)),
        "param_names": list(param_names),
        "canonicalization_motor_self_rotation_param_groups": [
            list(group) for group in MOTOR_SELF_ROTATION_PARAM_GROUPS
        ],
        "training_config": dict(training_config),
        "training_history": list(training_history),
    }
    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_airframe_vae_checkpoint(checkpoint_path, device):
    input_path = Path(checkpoint_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {input_path}")
    checkpoint = torch.load(input_path, map_location=torch.device(device), weights_only=False)
    required_keys = {
        "model_state_dict",
        "input_dim",
        "latent_dim",
        "hidden_dimensions",
        "param_names",
        "training_config",
        "training_history",
    }
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"{input_path} is missing checkpoint keys: {sorted(missing_keys)}")
    model = build_airframe_vae(
        checkpoint["input_dim"],
        checkpoint["latent_dim"],
        checkpoint["hidden_dimensions"],
        device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_self_contained_decoder_checkpoint(checkpoint_path):
    input_path = Path(checkpoint_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {input_path}")
    checkpoint = torch.load(input_path, map_location=torch.device("cpu"), weights_only=False)
    required_keys = {
        "torchscript_decoder_bytes",
        "input_dim",
        "latent_dim",
        "hidden_dimensions",
        "param_names",
    }
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"{input_path} is missing self-contained decoder keys: {sorted(missing_keys)}")
    decoder_module = torch.jit.load(io.BytesIO(checkpoint["torchscript_decoder_bytes"]), map_location=torch.device("cpu"))
    return checkpoint, decoder_module


def decode_from_file(vae_filepath):
    checkpoint, decoder_module = load_self_contained_decoder_checkpoint(vae_filepath)
    return AirframeDecoderFromFile(vae_filepath, checkpoint, decoder_module)


def load_decode_latent_input(latent_input_npy):
    input_path = Path(latent_input_npy)
    if not input_path.is_file():
        raise FileNotFoundError(f"Latent input file not found: {input_path}")
    latent_z = np.load(input_path, allow_pickle=False)
    if isinstance(latent_z, np.lib.npyio.NpzFile):
        latent_z.close()
        raise ValueError(f"Latent input must be a .npy array, got npz archive: {input_path}")
    return np.asarray(latent_z, dtype=np.float32)


def decode_vae_file(vae_filepath, latent_input_npy, decoded_output_npz):
    decoder = decode_from_file(vae_filepath)
    latent_z = load_decode_latent_input(latent_input_npy)
    decoded_x = decoder(latent_z)
    output_path = Path(decoded_output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        latent_z=latent_z,
        decoded_x=decoded_x,
        param_names=np.asarray(decoder.param_names),
        checkpoint_path=np.asarray(str(decoder.checkpoint_path)),
        latent_dim=np.asarray(decoder.latent_dim),
        input_dim=np.asarray(decoder.input_dim),
    )
    print(f"saved decoded output: {output_path}", flush=True)


def parse_action(argv):
    if len(argv) < 2:
        raise ValueError(f"Expected an action argument. Use one of: {sorted(VALID_ACTIONS)}")
    action = argv[1]
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action {action!r}. Use one of: {sorted(VALID_ACTIONS)}")
    if action == "--decode":
        if len(argv) != 5:
            raise ValueError("Expected: --decode <vae_filepath> <latent_input_npy> <decoded_output_npz>")
        return action, tuple(argv[2:5])
    if len(argv) != 2:
        raise ValueError(f"Action {action} does not accept extra arguments")
    return action, ()


def configuration_to_json(configuration):
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def parameter_bounds_degrees(parameter_name):
    if parameter_name not in PARAM_BOUNDS:
        raise ValueError(f"Unknown parameter name: {parameter_name}")
    lower_degrees, upper_degrees = PARAM_BOUNDS[parameter_name]
    if lower_degrees >= upper_degrees:
        raise ValueError(
            f"Parameter {parameter_name} must have lower bound smaller than upper bound, "
            f"got {PARAM_BOUNDS[parameter_name]}"
        )
    return float(lower_degrees), float(upper_degrees)


def parameter_bounds_radians(parameter_name):
    lower_degrees, upper_degrees = parameter_bounds_degrees(parameter_name)
    return np.deg2rad(lower_degrees), np.deg2rad(upper_degrees)


def parameter_value_to_normalized(parameter_name, value):
    lower, upper = PARAM_BOUNDS[parameter_name]
    normalized = (float(value) - float(lower)) / (float(upper) - float(lower))
    if normalized < -CANONICALIZATION_NUMERICAL_TOLERANCE or normalized > 1.0 + CANONICALIZATION_NUMERICAL_TOLERANCE:
        raise ValueError(
            f"Canonical value for {parameter_name} is outside bounds: "
            f"value={value}, bounds={PARAM_BOUNDS[parameter_name]}, normalized={normalized}"
        )
    return float(np.clip(normalized, 0.0, 1.0))


def validate_rotation_bounds_for_canonicalization(roll_name, pitch_name, yaw_name):
    roll_lower, roll_upper = parameter_bounds_radians(roll_name)
    pitch_lower, pitch_upper = parameter_bounds_radians(pitch_name)
    yaw_lower, yaw_upper = parameter_bounds_radians(yaw_name)
    half_pi = 0.5 * np.pi
    if roll_lower <= -half_pi or roll_upper >= half_pi:
        raise ValueError(
            f"{roll_name} bounds must stay strictly inside (-90, 90) degrees for canonicalization, "
            f"got {PARAM_BOUNDS[roll_name]}"
        )
    if pitch_lower <= -half_pi or pitch_upper >= half_pi:
        raise ValueError(
            f"{pitch_name} bounds must stay strictly inside (-90, 90) degrees for canonicalization, "
            f"got {PARAM_BOUNDS[pitch_name]}"
        )
    if yaw_upper - yaw_lower >= 2.0 * np.pi:
        raise ValueError(
            f"{yaw_name} bounds must span less than 360 degrees for canonicalization, "
            f"got {PARAM_BOUNDS[yaw_name]}"
        )


def rpy_to_rotation_matrix(roll, pitch, yaw):
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def motor_thrust_direction_from_rpy(roll, pitch, yaw):
    return rpy_to_rotation_matrix(roll, pitch, yaw) @ np.array([0.0, 0.0, 1.0], dtype=float)


def motor_rpy_from_thrust_direction_and_yaw(thrust_direction, yaw):
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local_x = cos_yaw * thrust_direction[0] + sin_yaw * thrust_direction[1]
    local_y = -sin_yaw * thrust_direction[0] + cos_yaw * thrust_direction[1]
    local_z = thrust_direction[2]
    roll = -np.arcsin(np.clip(local_y, -1.0, 1.0))
    pitch = np.arctan2(local_x, local_z)
    return np.array([roll, pitch, yaw], dtype=float)


def motor_yaw_is_feasible(thrust_direction, yaw, roll_bounds, pitch_bounds, yaw_bounds):
    roll, pitch, yaw_value = motor_rpy_from_thrust_direction_and_yaw(thrust_direction, yaw)
    return (
        roll_bounds[0] - CANONICALIZATION_NUMERICAL_TOLERANCE <= roll <= roll_bounds[1] + CANONICALIZATION_NUMERICAL_TOLERANCE
        and pitch_bounds[0] - CANONICALIZATION_NUMERICAL_TOLERANCE <= pitch <= pitch_bounds[1] + CANONICALIZATION_NUMERICAL_TOLERANCE
        and yaw_bounds[0] - CANONICALIZATION_NUMERICAL_TOLERANCE <= yaw_value <= yaw_bounds[1] + CANONICALIZATION_NUMERICAL_TOLERANCE
    )


def add_periodic_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, base_yaw):
    period = 2.0 * np.pi
    first_period_index = int(np.floor((yaw_lower - base_yaw) / period)) - 1
    last_period_index = int(np.ceil((yaw_upper - base_yaw) / period)) + 1
    for period_index in range(first_period_index, last_period_index + 1):
        yaw = base_yaw + period * period_index
        if yaw_lower - CANONICALIZATION_NUMERICAL_TOLERANCE <= yaw <= yaw_upper + CANONICALIZATION_NUMERICAL_TOLERANCE:
            boundary_yaws.append(float(np.clip(yaw, yaw_lower, yaw_upper)))


def add_sine_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, alpha, scaled_value):
    if abs(scaled_value) > 1.0 + CANONICALIZATION_NUMERICAL_TOLERANCE:
        return
    clipped_value = float(np.clip(scaled_value, -1.0, 1.0))
    angle = np.arcsin(clipped_value)
    for phase in (angle, np.pi - angle):
        add_periodic_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, alpha - phase)


def add_cosine_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, alpha, scaled_value):
    if abs(scaled_value) > 1.0 + CANONICALIZATION_NUMERICAL_TOLERANCE:
        return
    clipped_value = float(np.clip(scaled_value, -1.0, 1.0))
    angle = np.arccos(clipped_value)
    for phase in (angle, -angle):
        add_periodic_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, alpha - phase)


def canonicalize_motor_rpy_degrees(roll_degrees, pitch_degrees, yaw_degrees, roll_name, pitch_name, yaw_name):
    validate_rotation_bounds_for_canonicalization(roll_name, pitch_name, yaw_name)
    roll_bounds = parameter_bounds_radians(roll_name)
    pitch_bounds = parameter_bounds_radians(pitch_name)
    yaw_bounds = parameter_bounds_radians(yaw_name)
    original_rpy = np.deg2rad([roll_degrees, pitch_degrees, yaw_degrees])
    thrust_direction = motor_thrust_direction_from_rpy(original_rpy[0], original_rpy[1], original_rpy[2])
    horizontal_norm = float(np.hypot(thrust_direction[0], thrust_direction[1]))
    if horizontal_norm <= CANONICALIZATION_NUMERICAL_TOLERANCE:
        canonical_yaw = float(np.clip(0.0, yaw_bounds[0], yaw_bounds[1]))
        if not motor_yaw_is_feasible(thrust_direction, canonical_yaw, roll_bounds, pitch_bounds, yaw_bounds):
            raise ValueError(
                f"Cannot canonicalize vertical motor direction for {roll_name}, {pitch_name}, {yaw_name} "
                f"because the yaw closest to zero is outside the feasible bounds"
            )
        return np.rad2deg(motor_rpy_from_thrust_direction_and_yaw(thrust_direction, canonical_yaw))

    alpha = float(np.arctan2(thrust_direction[1], thrust_direction[0]))
    yaw_lower, yaw_upper = yaw_bounds
    boundary_yaws = [yaw_lower, yaw_upper]
    roll_sine_lower = np.sin(-roll_bounds[1])
    roll_sine_upper = np.sin(-roll_bounds[0])
    pitch_cosine_lower = thrust_direction[2] * np.tan(pitch_bounds[0])
    pitch_cosine_upper = thrust_direction[2] * np.tan(pitch_bounds[1])
    for boundary_value in (roll_sine_lower, roll_sine_upper):
        add_sine_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, alpha, boundary_value / horizontal_norm)
    for boundary_value in (pitch_cosine_lower, pitch_cosine_upper):
        add_cosine_boundary_yaws(boundary_yaws, yaw_lower, yaw_upper, alpha, boundary_value / horizontal_norm)

    boundary_yaws = sorted(set(round(float(value), 15) for value in boundary_yaws))
    candidate_yaws = []
    for yaw in boundary_yaws:
        if motor_yaw_is_feasible(thrust_direction, yaw, roll_bounds, pitch_bounds, yaw_bounds):
            candidate_yaws.append(yaw)

    for left_yaw, right_yaw in zip(boundary_yaws[:-1], boundary_yaws[1:]):
        midpoint_yaw = 0.5 * (left_yaw + right_yaw)
        if motor_yaw_is_feasible(thrust_direction, midpoint_yaw, roll_bounds, pitch_bounds, yaw_bounds):
            yaw_closest_to_zero = float(np.clip(0.0, left_yaw, right_yaw))
            if motor_yaw_is_feasible(thrust_direction, yaw_closest_to_zero, roll_bounds, pitch_bounds, yaw_bounds):
                candidate_yaws.append(yaw_closest_to_zero)
            if motor_yaw_is_feasible(thrust_direction, left_yaw, roll_bounds, pitch_bounds, yaw_bounds):
                candidate_yaws.append(left_yaw)
            if motor_yaw_is_feasible(thrust_direction, right_yaw, roll_bounds, pitch_bounds, yaw_bounds):
                candidate_yaws.append(right_yaw)

    if not candidate_yaws:
        raise ValueError(
            f"No feasible canonical yaw found for {roll_name}, {pitch_name}, {yaw_name}; "
            f"original degrees={[roll_degrees, pitch_degrees, yaw_degrees]}, "
            f"thrust_direction={thrust_direction.tolist()}"
        )

    canonical_yaw = min(candidate_yaws, key=lambda value: (abs(value), value))
    canonical_rpy = motor_rpy_from_thrust_direction_and_yaw(thrust_direction, canonical_yaw)
    canonical_thrust_direction = motor_thrust_direction_from_rpy(canonical_rpy[0], canonical_rpy[1], canonical_rpy[2])
    thrust_error = np.linalg.norm(canonical_thrust_direction - thrust_direction)
    if thrust_error > 1e-8:
        raise RuntimeError(
            f"Canonicalized motor RPY changed thrust direction for {roll_name}, {pitch_name}, {yaw_name}: "
            f"error={thrust_error}"
        )
    if not motor_yaw_is_feasible(thrust_direction, canonical_yaw, roll_bounds, pitch_bounds, yaw_bounds):
        raise RuntimeError(
            f"Canonicalized motor RPY is outside bounds for {roll_name}, {pitch_name}, {yaw_name}: "
            f"{np.rad2deg(canonical_rpy).tolist()}"
        )
    return np.rad2deg(canonical_rpy)


def canonicalize_airframe_x(x_0_1):
    x_array = np.asarray(x_0_1, dtype=float)
    if x_array.ndim != 1:
        raise ValueError(f"x_0_1 must be 1D, got shape {x_array.shape}")
    if x_array.shape[0] != len(PARAM_NAMES):
        raise ValueError(f"Expected {len(PARAM_NAMES)} parameters, got {x_array.shape[0]}")
    if not np.all(np.isfinite(x_array)):
        raise ValueError(f"x_0_1 contains non-finite values: {x_array}")
    if np.min(x_array) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(x_array) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
        raise ValueError(f"x_0_1 contains values outside [0, 1]: {x_array}")

    canonical_x = np.clip(x_array.copy(), 0.0, 1.0)
    params = Airframe.from_x_0_1(canonical_x).params
    for roll_name, pitch_name, yaw_name in MOTOR_SELF_ROTATION_PARAM_GROUPS:
        canonical_rpy_degrees = canonicalize_motor_rpy_degrees(
            params[roll_name],
            params[pitch_name],
            params[yaw_name],
            roll_name,
            pitch_name,
            yaw_name,
        )
        for parameter_name, canonical_value in zip((roll_name, pitch_name, yaw_name), canonical_rpy_degrees):
            canonical_x[PARAM_NAMES.index(parameter_name)] = parameter_value_to_normalized(parameter_name, canonical_value)

    canonical_x = np.asarray(canonical_x, dtype=float)
    if np.min(canonical_x) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(canonical_x) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
        raise RuntimeError(f"Canonicalized x_0_1 contains values outside [0, 1]: {canonical_x}")
    return np.clip(canonical_x, 0.0, 1.0)


def canonicalize_airframe_x_array(x_array):
    array = np.asarray(x_array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"x_array must be 2D, got shape {array.shape}")
    return np.asarray([canonicalize_airframe_x(row) for row in array], dtype=array.dtype)


def validate_worker_cpu_ids(cpu_ids):
    if len(cpu_ids) < 1:
        raise ValueError("WORKER_CPU_IDS must contain at least one CPU ID")
    if len(set(cpu_ids)) != len(cpu_ids):
        raise ValueError(f"WORKER_CPU_IDS contains duplicate CPU IDs: {cpu_ids}")
    if min(cpu_ids) < 0:
        raise ValueError(f"WORKER_CPU_IDS must contain non-negative CPU IDs, got {cpu_ids}")

    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise RuntimeError("os.cpu_count() returned None; cannot validate WORKER_CPU_IDS")
    too_large_cpu_ids = [cpu_id for cpu_id in cpu_ids if cpu_id >= cpu_count]
    if too_large_cpu_ids:
        raise ValueError(f"CPU IDs exceed os.cpu_count()={cpu_count}: {too_large_cpu_ids}")


def validate_worker_cpu_ids_usable(cpu_ids):
    original_affinity = os.sched_getaffinity(0)
    try:
        for cpu_id in cpu_ids:
            os.sched_setaffinity(0, {cpu_id})
            current_affinity = os.sched_getaffinity(0)
            if current_affinity != {cpu_id}:
                raise RuntimeError(
                    f"could not pin current process to CPU {cpu_id}; observed affinity {sorted(current_affinity)}"
                )
    finally:
        os.sched_setaffinity(0, original_affinity)


def configure_worker_cpu(worker_cpu_ids, worker_index, worker_index_lock):
    with worker_index_lock:
        cpu_index = worker_index.value
        worker_index.value += 1

    if cpu_index >= len(worker_cpu_ids):
        raise RuntimeError(
            f"ProcessPoolExecutor started worker index {cpu_index}, "
            f"but only {len(worker_cpu_ids)} CPU IDs were provided"
        )

    cpu_id = worker_cpu_ids[cpu_index]
    os.sched_setaffinity(0, {cpu_id})
    current_affinity = os.sched_getaffinity(0)
    if current_affinity != {cpu_id}:
        raise RuntimeError(f"worker expected CPU affinity [{cpu_id}], got {sorted(current_affinity)}")


def repair_one_sample(sample_index, original_row, target_accel_ratios, minimum_arm_torque, configuration):
    try:
        airframe = Airframe.from_x_0_1(original_row)
        repaired_airframe = repair_solution_acceleration(
            airframe,
            target_accel_ratios=target_accel_ratios,
            verbose=False,
            minimum_arm_torque=minimum_arm_torque,
            configuration=configuration,
        )
        repaired_row = canonicalize_airframe_x(repaired_airframe.x_0_1)
        expected_shape = np.asarray(original_row, dtype=float).shape
        if repaired_row.shape != expected_shape:
            raise ValueError(f"Expected repaired shape {expected_shape}, got {repaired_row.shape}")
        if not np.all(np.isfinite(repaired_row)):
            raise ValueError(f"Repair produced non-finite values: {repaired_row}")
        if np.min(repaired_row) < 0.0 or np.max(repaired_row) > 1.0:
            raise ValueError(f"Repair produced values outside [0, 1]: {repaired_row}")
        return sample_index, True, repaired_row, "", sorted(os.sched_getaffinity(0))
    except Exception:
        repaired_row = np.full_like(np.asarray(original_row, dtype=float), np.nan)
        return sample_index, False, repaired_row, traceback.format_exc(), sorted(os.sched_getaffinity(0))


def sample_and_repair(n_samples, seed, target_accel_ratios, minimum_arm_torque, configuration, worker_cpu_ids):
    validate_worker_cpu_ids(worker_cpu_ids)
    validate_worker_cpu_ids_usable(worker_cpu_ids)

    rng = np.random.default_rng(seed)
    original_x = rng.uniform(0.0, 1.0, size=(n_samples, len(PARAM_NAMES)))
    repaired_x = np.full_like(original_x, np.nan)
    is_valid = np.zeros(n_samples, dtype=bool)
    repair_error = np.full(n_samples, "", dtype=object)

    context = mp.get_context("spawn")
    worker_index = context.Value("i", 0)
    worker_index_lock = context.Lock()
    start_time = time.monotonic()
    completed_samples = 0

    with ProcessPoolExecutor(
        max_workers=len(worker_cpu_ids),
        mp_context=context,
        initializer=configure_worker_cpu,
        initargs=(worker_cpu_ids, worker_index, worker_index_lock),
    ) as executor:
        futures = [
            executor.submit(
                repair_one_sample,
                sample_index,
                original_x[sample_index],
                target_accel_ratios,
                minimum_arm_torque,
                configuration,
            )
            for sample_index in range(n_samples)
        ]

        for future in as_completed(futures):
            sample_index, valid, repaired_row, error_text, cpu_affinity = future.result()
            repaired_x[sample_index] = repaired_row
            is_valid[sample_index] = bool(valid)
            repair_error[sample_index] = error_text
            completed_samples += 1
            elapsed_time = time.monotonic() - start_time
            print(
                f"repair sample {completed_samples}/{n_samples} original_index={sample_index} "
                f"valid={bool(valid)} cpu_affinity={cpu_affinity} elapsed_s={elapsed_time:.2f}",
                flush=True,
            )

    return original_x, repaired_x, is_valid, repair_error


def validate_dataset_arrays(original_x, repaired_x, is_valid):
    if original_x.shape != repaired_x.shape:
        raise ValueError(f"original_x shape {original_x.shape} does not match repaired_x shape {repaired_x.shape}")
    if original_x.ndim != 2:
        raise ValueError(f"original_x must be 2D, got ndim={original_x.ndim}")
    if original_x.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"Expected {len(PARAM_NAMES)} columns, got {original_x.shape[1]}")
    if is_valid.shape != (original_x.shape[0],):
        raise ValueError(f"is_valid shape must be {(original_x.shape[0],)}, got {is_valid.shape}")
    if not np.all(np.isfinite(original_x)):
        raise ValueError("original_x contains non-finite values")
    if np.min(original_x) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(original_x) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
        raise ValueError("original_x contains values outside [0, 1]")

    valid_repaired_x = repaired_x[is_valid]
    if len(valid_repaired_x) == 0:
        raise ValueError("No valid repaired samples were produced")
    if not np.all(np.isfinite(valid_repaired_x)):
        raise ValueError("valid repaired_x rows contain non-finite values")
    if np.min(valid_repaired_x) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(valid_repaired_x) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
        raise ValueError("valid repaired_x rows contain values outside [0, 1]")


def save_dataset(
    data_npz,
    original_x,
    repaired_x,
    is_valid,
    repair_error,
    seed,
    target_accel_ratios,
    minimum_arm_torque,
    configuration,
):
    validate_dataset_arrays(original_x, repaired_x, is_valid)
    repair_delta = repaired_x - original_x
    repair_l2_distance = np.full(original_x.shape[0], np.nan, dtype=float)
    repair_linf_distance = np.full(original_x.shape[0], np.nan, dtype=float)
    valid_delta = repair_delta[is_valid]
    repair_l2_distance[is_valid] = np.linalg.norm(valid_delta, axis=1)
    repair_linf_distance[is_valid] = np.max(np.abs(valid_delta), axis=1)

    output_path = Path(data_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        original_x=original_x,
        repaired_x=repaired_x,
        repair_delta=repair_delta,
        repair_l2_distance=repair_l2_distance,
        repair_linf_distance=repair_linf_distance,
        is_valid=is_valid,
        repair_error=np.asarray(repair_error, dtype=str),
        param_names=np.asarray(PARAM_NAMES),
        seed=np.asarray(seed),
        n_samples=np.asarray(original_x.shape[0]),
        target_accel_linear=np.asarray(float(target_accel_ratios["linear"])),
        target_accel_angular=np.asarray(float(target_accel_ratios["angular"])),
        minimum_arm_torque=np.asarray(float(minimum_arm_torque)),
        configuration_json=np.asarray(configuration_to_json(configuration)),
        canonicalized=np.asarray(True),
        canonicalization_motor_self_rotation_param_groups=np.asarray(
            MOTOR_SELF_ROTATION_PARAM_GROUPS,
            dtype=str,
        ),
    )


def get_samples():
    original_x, repaired_x, is_valid, repair_error = sample_and_repair(
        GET_SAMPLES_N_SAMPLES,
        GET_SAMPLES_SEED,
        CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS,
        CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
        CONTROL_AUTHORITY_REPAIR_CONFIGURATION,
        WORKER_CPU_IDS,
    )
    save_dataset(
        DATA_NPZ,
        original_x,
        repaired_x,
        is_valid,
        repair_error,
        GET_SAMPLES_SEED,
        CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS,
        CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
        CONTROL_AUTHORITY_REPAIR_CONFIGURATION,
    )
    print(f"saved dataset: {DATA_NPZ} samples={original_x.shape[0]} valid={int(np.sum(is_valid))}", flush=True)


def load_valid_repaired_dataset(data_npz):
    input_path = Path(data_npz)
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    with np.load(input_path, allow_pickle=False) as data:
        required_keys = {"repaired_x", "is_valid", "param_names"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise ValueError(f"{input_path} is missing required arrays: {sorted(missing_keys)}")
        repaired_x = np.asarray(data["repaired_x"], dtype=np.float32)
        is_valid = np.asarray(data["is_valid"], dtype=bool)
        param_names = [str(value) for value in data["param_names"]]

    if param_names != list(PARAM_NAMES):
        raise ValueError(f"Dataset param_names do not match current PARAM_NAMES: {param_names}")
    if repaired_x.ndim != 2:
        raise ValueError(f"repaired_x must be 2D, got ndim={repaired_x.ndim}")
    if repaired_x.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"Expected {len(PARAM_NAMES)} columns, got {repaired_x.shape[1]}")
    if is_valid.shape != (repaired_x.shape[0],):
        raise ValueError(f"is_valid shape must be {(repaired_x.shape[0],)}, got {is_valid.shape}")

    valid_x = canonicalize_airframe_x_array(repaired_x[is_valid])
    if valid_x.shape[0] < 2:
        raise ValueError(f"Need at least 2 valid repaired samples, got {valid_x.shape[0]}")
    if not np.all(np.isfinite(valid_x)):
        raise ValueError("Valid repaired samples contain non-finite values")
    if np.min(valid_x) < 0.0 or np.max(valid_x) > 1.0:
        raise ValueError("Valid repaired samples contain values outside [0, 1]")
    return valid_x, param_names, int(np.sum(is_valid)), int(is_valid.shape[0])


def split_dataset(valid_x, train_fraction, seed):
    n_samples = valid_x.shape[0]
    n_train = int(np.floor(float(train_fraction) * float(n_samples)))
    n_val = n_samples - n_train
    if n_train < 1 or n_val < 1:
        raise ValueError(
            f"TRAIN_FRACTION={train_fraction} yields n_train={n_train}, n_val={n_val} for n_samples={n_samples}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    return valid_x[train_indices], valid_x[val_indices]


def make_loader(x_array, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.as_tensor(x_array, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )


def run_epoch(model, loader, optimizer, beta, device, train):
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kld_loss_sum = 0.0
    n_samples = 0

    for (batch,) in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(batch)
            total_loss, reconstruction_loss, kld_loss = beta_vae_loss(x_hat, batch, mu, logvar, beta)
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                x_hat, mu, logvar = model(batch)
                total_loss, reconstruction_loss, kld_loss = beta_vae_loss(x_hat, batch, mu, logvar, beta)

        current_batch_size = batch.shape[0]
        total_loss_sum += float(total_loss.detach().cpu()) * current_batch_size
        reconstruction_loss_sum += float(reconstruction_loss.detach().cpu()) * current_batch_size
        kld_loss_sum += float(kld_loss.detach().cpu()) * current_batch_size
        n_samples += current_batch_size

    if n_samples == 0:
        raise ValueError("DataLoader produced zero samples")
    return {
        "total": total_loss_sum / n_samples,
        "mse": reconstruction_loss_sum / n_samples,
        "kld": kld_loss_sum / n_samples,
    }


def make_training_config(
    data_npz,
    checkpoint_path,
    latent_dim,
    n_total,
    n_valid,
    n_train,
    n_val,
):
    return {
        "data_npz": str(data_npz),
        "checkpoint_path": str(checkpoint_path),
        "latent_dim": int(latent_dim),
        "hidden_dimensions": list(HIDDEN_DIMENSIONS),
        "train_fraction": TRAIN_FRACTION,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "min_improvement": MIN_IMPROVEMENT,
        "seed": TRAIN_SEED,
        "device": DEVICE,
        "num_dataloader_workers": NUM_DATALOADER_WORKERS,
        "n_total_samples": n_total,
        "n_valid_samples": n_valid,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "canonicalization_motor_self_rotation_param_groups": [
            list(group) for group in MOTOR_SELF_ROTATION_PARAM_GROUPS
        ],
    }


def write_training_config(training_config, training_config_json):
    output_path = Path(training_config_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_config, f, indent=2)


def train_airframe_vae(
    data_npz,
    checkpoint_path,
    training_config_json,
    tensorboard_dir,
    latent_dim,
):
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    valid_x, param_names, n_valid, n_total = load_valid_repaired_dataset(data_npz)
    train_x, val_x = split_dataset(valid_x, TRAIN_FRACTION, TRAIN_SEED)
    train_loader = make_loader(train_x, BATCH_SIZE, True, TRAIN_SEED, NUM_DATALOADER_WORKERS)
    val_loader = make_loader(val_x, BATCH_SIZE, False, TRAIN_SEED, NUM_DATALOADER_WORKERS)

    torch_device = torch.device(DEVICE)
    model = build_airframe_vae(valid_x.shape[1], latent_dim, HIDDEN_DIMENSIONS, torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    training_config = make_training_config(
        data_npz,
        checkpoint_path,
        latent_dim,
        n_total,
        n_valid,
        int(train_x.shape[0]),
        int(val_x.shape[0]),
    )
    write_training_config(training_config, training_config_json)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    start_time = time.monotonic()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, BETA, torch_device, True)
        val_metrics = run_epoch(model, val_loader, optimizer, BETA, torch_device, False)
        scheduler.step(val_metrics["total"])
        learning_rate = float(optimizer.param_groups[0]["lr"])

        writer.add_scalar("train/total", train_metrics["total"], epoch)
        writer.add_scalar("train/mse", train_metrics["mse"], epoch)
        writer.add_scalar("train/kld", train_metrics["kld"], epoch)
        writer.add_scalar("validation/total", val_metrics["total"], epoch)
        writer.add_scalar("validation/mse", val_metrics["mse"], epoch)
        writer.add_scalar("validation/kld", val_metrics["kld"], epoch)
        writer.add_scalar("optimizer/learning_rate", learning_rate, epoch)

        record = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_mse": train_metrics["mse"],
            "train_kld": train_metrics["kld"],
            "validation_total": val_metrics["total"],
            "validation_mse": val_metrics["mse"],
            "validation_kld": val_metrics["kld"],
            "learning_rate": learning_rate,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.8f} "
            f"val_total={val_metrics['total']:.8f} val_mse={val_metrics['mse']:.8f} "
            f"val_kld={val_metrics['kld']:.8f} lr={learning_rate:.8g}",
            flush=True,
        )

        improved = val_metrics["total"] < best_val_loss - MIN_IMPROVEMENT
        if improved:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            save_airframe_vae_checkpoint(
                checkpoint_path,
                model,
                valid_x.shape[1],
                latent_dim,
                HIDDEN_DIMENSIONS,
                param_names,
                training_config,
                history,
            )
            print(f"saved best checkpoint: {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stopping at epoch={epoch}", flush=True)
                break

    writer.flush()
    writer.close()
    checkpoint_output_path = Path(checkpoint_path)
    if not checkpoint_output_path.is_file():
        raise FileNotFoundError(f"Training did not write checkpoint: {checkpoint_output_path}")
    print(f"best validation total loss={best_val_loss:.8f}", flush=True)
    return best_val_loss


def train_vae():
    train_airframe_vae(
        DATA_NPZ,
        CHECKPOINT_PATH,
        TRAINING_CONFIG_JSON,
        TENSORBOARD_DIR,
        LATENT_DIM,
    )
    analyze_vae()


def validation_split_from_training_config(valid_x, training_config):
    train_fraction = float(training_config["train_fraction"])
    seed = int(training_config["seed"])
    n_samples = valid_x.shape[0]
    n_train = int(np.floor(train_fraction * float(n_samples)))
    n_val = n_samples - n_train
    if n_train < 1 or n_val < 1:
        raise ValueError(f"Stored train_fraction yields n_train={n_train}, n_val={n_val}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    val_indices = indices[n_train:]
    return valid_x[val_indices]


def encode_decode_validation(model, validation_x, device):
    x_tensor = torch.as_tensor(validation_x, dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        mu, logvar = model.encode(x_tensor)
        reconstruction = model.decode(mu)
    return (
        mu.detach().cpu().numpy(),
        logvar.detach().cpu().numpy(),
        canonicalize_airframe_x_array(reconstruction.detach().cpu().numpy()),
    )


def decode_standard_normal_samples(model, n_generated, latent_dim, seed, device):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_generated, latent_dim)).astype(np.float32)
    z_tensor = torch.as_tensor(z, dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        decoded = model.decode(z_tensor).detach().cpu().numpy()
    if np.min(decoded) < 0.0 or np.max(decoded) > 1.0:
        raise ValueError("Decoder produced generated samples outside [0, 1]")
    return z, canonicalize_airframe_x_array(decoded)


def repair_generated_samples(generated_x, worker_cpu_ids):
    validate_worker_cpu_ids(worker_cpu_ids)
    validate_worker_cpu_ids_usable(worker_cpu_ids)
    repaired_x = np.full_like(generated_x, np.nan)
    is_valid = np.zeros(generated_x.shape[0], dtype=bool)
    repair_error = np.full(generated_x.shape[0], "", dtype=object)

    context = mp.get_context("spawn")
    worker_index = context.Value("i", 0)
    worker_index_lock = context.Lock()
    start_time = time.monotonic()
    completed_samples = 0

    with ProcessPoolExecutor(
        max_workers=len(worker_cpu_ids),
        mp_context=context,
        initializer=configure_worker_cpu,
        initargs=(worker_cpu_ids, worker_index, worker_index_lock),
    ) as executor:
        futures = [
            executor.submit(
                repair_one_sample,
                sample_index,
                generated_x[sample_index],
                CONTROL_AUTHORITY_REPAIR_TARGET_ACCEL_RATIOS,
                CONTROL_AUTHORITY_REPAIR_MINIMUM_ARM_TORQUE,
                CONTROL_AUTHORITY_REPAIR_CONFIGURATION,
            )
            for sample_index in range(generated_x.shape[0])
        ]
        for future in as_completed(futures):
            sample_index, valid, repaired_row, error_text, cpu_affinity = future.result()
            repaired_x[sample_index] = repaired_row
            is_valid[sample_index] = bool(valid)
            repair_error[sample_index] = error_text
            completed_samples += 1
            elapsed_time = time.monotonic() - start_time
            print(
                f"generated repair {completed_samples}/{generated_x.shape[0]} original_index={sample_index} "
                f"valid={bool(valid)} cpu_affinity={cpu_affinity} elapsed_s={elapsed_time:.2f}",
                flush=True,
            )

    return repaired_x, is_valid, repair_error


def nearest_distances(query_x, reference_x):
    if query_x.ndim != 2:
        raise ValueError(f"query_x must be 2D, got ndim={query_x.ndim}")
    if reference_x.ndim != 2:
        raise ValueError(f"reference_x must be 2D, got ndim={reference_x.ndim}")
    if query_x.shape[1] != reference_x.shape[1]:
        raise ValueError(f"query_x columns {query_x.shape[1]} do not match reference_x columns {reference_x.shape[1]}")
    if reference_x.shape[0] < 1:
        raise ValueError("reference_x must contain at least one sample")

    query_squared_norm = np.sum(query_x * query_x, axis=1)
    reference_squared_norm = np.sum(reference_x * reference_x, axis=1)
    distances = np.empty(query_x.shape[0], dtype=float)
    for start_index in range(0, query_x.shape[0], NEAREST_DISTANCE_QUERY_CHUNK_SIZE):
        stop_index = min(start_index + NEAREST_DISTANCE_QUERY_CHUNK_SIZE, query_x.shape[0])
        squared_distance = (
            query_squared_norm[start_index:stop_index, None]
            + reference_squared_norm[None, :]
            - 2.0 * (query_x[start_index:stop_index] @ reference_x.T)
        )
        np.maximum(squared_distance, 0.0, out=squared_distance)
        distances[start_index:stop_index] = np.sqrt(np.min(squared_distance, axis=1))
    return distances


def nearest_other_distances(x_array):
    if x_array.ndim != 2:
        raise ValueError(f"x_array must be 2D, got ndim={x_array.ndim}")
    if x_array.shape[0] < 2:
        raise ValueError(f"Need at least two samples for nearest-other distances, got {x_array.shape[0]}")
    squared_norm = np.sum(x_array * x_array, axis=1)
    distances = np.empty(x_array.shape[0], dtype=float)
    for start_index in range(0, x_array.shape[0], NEAREST_DISTANCE_QUERY_CHUNK_SIZE):
        stop_index = min(start_index + NEAREST_DISTANCE_QUERY_CHUNK_SIZE, x_array.shape[0])
        squared_distance = (
            squared_norm[start_index:stop_index, None]
            + squared_norm[None, :]
            - 2.0 * (x_array[start_index:stop_index] @ x_array.T)
        )
        np.maximum(squared_distance, 0.0, out=squared_distance)
        squared_distance[np.arange(stop_index - start_index), np.arange(start_index, stop_index)] = np.inf
        distances[start_index:stop_index] = np.sqrt(np.min(squared_distance, axis=1))
    return distances


def pca_project(reference_x, arrays):
    if reference_x.ndim != 2:
        raise ValueError(f"reference_x must be 2D, got ndim={reference_x.ndim}")
    if reference_x.shape[0] < 2:
        raise ValueError(f"Need at least two reference samples for PCA, got {reference_x.shape[0]}")
    centered_reference = reference_x - np.mean(reference_x, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered_reference, full_matrices=False)
    if vh.shape[0] < 2:
        raise ValueError(f"Need at least two PCA components, got {vh.shape[0]}")
    components = vh[:2].T
    variance = singular_values * singular_values / float(reference_x.shape[0] - 1)
    explained_variance_ratio = variance[:2] / np.sum(variance)

    projected_arrays = []
    reference_mean = np.mean(reference_x, axis=0, keepdims=True)
    for x_array in arrays:
        if x_array.ndim != 2:
            raise ValueError(f"PCA input array must be 2D, got ndim={x_array.ndim}")
        if x_array.shape[1] != reference_x.shape[1]:
            raise ValueError(f"PCA input columns {x_array.shape[1]} do not match reference columns {reference_x.shape[1]}")
        projected_arrays.append((x_array - reference_mean) @ components)
    return projected_arrays, explained_variance_ratio


def safe_metric_ratio(numerator, denominator, label):
    if not np.isfinite(numerator):
        raise ValueError(f"{label} numerator is not finite: {numerator}")
    if not np.isfinite(denominator):
        raise ValueError(f"{label} denominator is not finite: {denominator}")
    if denominator <= 0.0:
        raise ValueError(f"{label} denominator must be positive, got {denominator}")
    return numerator / denominator


def projection_grid_edges(reference_projection, candidate_projection, grid_size, label):
    if reference_projection.ndim != 2 or reference_projection.shape[1] != 2:
        raise ValueError(f"{label} reference_projection must have shape (n, 2), got {reference_projection.shape}")
    if candidate_projection.ndim != 2 or candidate_projection.shape[1] != 2:
        raise ValueError(f"{label} candidate_projection must have shape (n, 2), got {candidate_projection.shape}")
    if reference_projection.shape[0] < 1:
        raise ValueError(f"{label} reference_projection must contain at least one sample")
    if candidate_projection.shape[0] < 1:
        raise ValueError(f"{label} candidate_projection must contain at least one sample")
    if int(grid_size) < 2:
        raise ValueError(f"{label} grid_size must be at least 2, got {grid_size}")
    if not np.all(np.isfinite(reference_projection)):
        raise ValueError(f"{label} reference_projection contains non-finite values")
    if not np.all(np.isfinite(candidate_projection)):
        raise ValueError(f"{label} candidate_projection contains non-finite values")

    combined_projection = np.vstack((reference_projection, candidate_projection))
    lower = np.min(combined_projection, axis=0)
    upper = np.max(combined_projection, axis=0)
    span = upper - lower
    invalid_span = span <= 0.0
    if np.any(invalid_span):
        raise ValueError(f"{label} projection span must be positive in both axes, got span={span}")
    padding = 0.01 * span
    edges_x = np.linspace(lower[0] - padding[0], upper[0] + padding[0], int(grid_size) + 1)
    edges_y = np.linspace(lower[1] - padding[1], upper[1] + padding[1], int(grid_size) + 1)
    return edges_x, edges_y


def normalized_solution_grid_edges(grid_size):
    if int(grid_size) < 2:
        raise ValueError(f"grid_size must be at least 2, got {grid_size}")
    return np.linspace(0.0, 1.0, int(grid_size) + 1), np.linspace(0.0, 1.0, int(grid_size) + 1)


def occupied_projection_cells(projection, edges_x, edges_y, label):
    if projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError(f"{label} projection must have shape (n, 2), got {projection.shape}")
    if projection.shape[0] < 1:
        raise ValueError(f"{label} projection must contain at least one sample")
    if not np.all(np.isfinite(projection)):
        raise ValueError(f"{label} projection contains non-finite values")
    if np.any(projection[:, 0] < edges_x[0]) or np.any(projection[:, 0] > edges_x[-1]):
        raise ValueError(f"{label} x projection lies outside the supplied grid")
    if np.any(projection[:, 1] < edges_y[0]) or np.any(projection[:, 1] > edges_y[-1]):
        raise ValueError(f"{label} y projection lies outside the supplied grid")

    grid_size_x = len(edges_x) - 1
    grid_size_y = len(edges_y) - 1
    x_indices = np.searchsorted(edges_x, projection[:, 0], side="right") - 1
    y_indices = np.searchsorted(edges_y, projection[:, 1], side="right") - 1
    x_indices = np.clip(x_indices, 0, grid_size_x - 1)
    y_indices = np.clip(y_indices, 0, grid_size_y - 1)
    return np.unique(y_indices * grid_size_x + x_indices)


def surface_coverage_metrics(reference_projection, candidate_projection, edges_x, edges_y, label):
    reference_cells = occupied_projection_cells(reference_projection, edges_x, edges_y, f"{label} reference")
    candidate_cells = occupied_projection_cells(candidate_projection, edges_x, edges_y, f"{label} candidate")
    shared_count = len(np.intersect1d(reference_cells, candidate_cells, assume_unique=True))
    union_count = len(np.union1d(reference_cells, candidate_cells))
    total_cells = (len(edges_x) - 1) * (len(edges_y) - 1)
    if total_cells < 1:
        raise ValueError(f"{label} total cell count must be positive, got {total_cells}")
    if len(reference_cells) < 1:
        raise ValueError(f"{label} reference occupied zero cells")
    if len(candidate_cells) < 1:
        raise ValueError(f"{label} candidate occupied zero cells")
    if union_count < 1:
        raise ValueError(f"{label} union occupied zero cells")

    return {
        "reference_occupied_count": int(len(reference_cells)),
        "candidate_occupied_count": int(len(candidate_cells)),
        "shared_occupied_count": int(shared_count),
        "union_occupied_count": int(union_count),
        "total_cell_count": int(total_cells),
        "reference_occupied_percent": 100.0 * float(len(reference_cells)) / float(total_cells),
        "candidate_occupied_percent": 100.0 * float(len(candidate_cells)) / float(total_cells),
        "candidate_covers_reference_percent": 100.0 * float(shared_count) / float(len(reference_cells)),
        "reference_covers_candidate_percent": 100.0 * float(shared_count) / float(len(candidate_cells)),
        "jaccard_percent": 100.0 * float(shared_count) / float(union_count),
    }


def compute_pca_surface_coverage(valid_x, generated_x, grid_size):
    projections, explained_variance_ratio = pca_project(valid_x, [valid_x, generated_x])
    valid_projection, generated_projection = projections
    edges_x, edges_y = projection_grid_edges(
        valid_projection,
        generated_projection,
        grid_size,
        "pca surface coverage",
    )
    metrics = surface_coverage_metrics(
        valid_projection,
        generated_projection,
        edges_x,
        edges_y,
        "pca surface coverage",
    )
    metrics["pc1_explained_variance_ratio"] = float(explained_variance_ratio[0])
    metrics["pc2_explained_variance_ratio"] = float(explained_variance_ratio[1])
    return metrics


def compute_pairwise_surface_coverage(valid_x, generated_x, grid_size):
    if valid_x.ndim != 2:
        raise ValueError(f"valid_x must be 2D, got ndim={valid_x.ndim}")
    if generated_x.ndim != 2:
        raise ValueError(f"generated_x must be 2D, got ndim={generated_x.ndim}")
    if valid_x.shape[1] != generated_x.shape[1]:
        raise ValueError(f"valid_x columns {valid_x.shape[1]} do not match generated_x columns {generated_x.shape[1]}")
    if valid_x.shape[1] < 2:
        raise ValueError(f"Need at least two parameters for pairwise coverage, got {valid_x.shape[1]}")
    if np.min(valid_x) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(valid_x) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
        raise ValueError("valid_x contains values outside [0, 1]")
    if np.min(generated_x) < -NORMALIZED_BOUNDS_TOLERANCE or np.max(generated_x) > 1.0 + NORMALIZED_BOUNDS_TOLERANCE:
        raise ValueError("generated_x contains values outside [0, 1]")

    edges_x, edges_y = normalized_solution_grid_edges(grid_size)
    pair_i = []
    pair_j = []
    reference_occupied_percent = []
    candidate_occupied_percent = []
    candidate_covers_reference_percent = []
    reference_covers_candidate_percent = []
    jaccard_percent = []
    reference_occupied_count = []
    candidate_occupied_count = []
    shared_occupied_count = []

    for first_index in range(valid_x.shape[1] - 1):
        for second_index in range(first_index + 1, valid_x.shape[1]):
            label = f"pairwise surface coverage {first_index}-{second_index}"
            metrics = surface_coverage_metrics(
                valid_x[:, [first_index, second_index]],
                generated_x[:, [first_index, second_index]],
                edges_x,
                edges_y,
                label,
            )
            pair_i.append(first_index)
            pair_j.append(second_index)
            reference_occupied_percent.append(metrics["reference_occupied_percent"])
            candidate_occupied_percent.append(metrics["candidate_occupied_percent"])
            candidate_covers_reference_percent.append(metrics["candidate_covers_reference_percent"])
            reference_covers_candidate_percent.append(metrics["reference_covers_candidate_percent"])
            jaccard_percent.append(metrics["jaccard_percent"])
            reference_occupied_count.append(metrics["reference_occupied_count"])
            candidate_occupied_count.append(metrics["candidate_occupied_count"])
            shared_occupied_count.append(metrics["shared_occupied_count"])

    return {
        "pair_i": np.asarray(pair_i, dtype=int),
        "pair_j": np.asarray(pair_j, dtype=int),
        "reference_occupied_percent": np.asarray(reference_occupied_percent, dtype=float),
        "candidate_occupied_percent": np.asarray(candidate_occupied_percent, dtype=float),
        "candidate_covers_reference_percent": np.asarray(candidate_covers_reference_percent, dtype=float),
        "reference_covers_candidate_percent": np.asarray(reference_covers_candidate_percent, dtype=float),
        "jaccard_percent": np.asarray(jaccard_percent, dtype=float),
        "reference_occupied_count": np.asarray(reference_occupied_count, dtype=int),
        "candidate_occupied_count": np.asarray(candidate_occupied_count, dtype=int),
        "shared_occupied_count": np.asarray(shared_occupied_count, dtype=int),
        "grid_size": np.asarray(int(grid_size), dtype=int),
        "total_cell_count": np.asarray(int(grid_size) * int(grid_size), dtype=int),
    }


def save_analysis_arrays(
    output_npz,
    valid_x,
    validation_x,
    latent_mu,
    latent_logvar,
    reconstruction_x,
    generated_z,
    generated_x,
    reconstruction_l2,
    reconstruction_linf,
    repaired_nearest_other_l2,
    repaired_to_generated_l2,
    generated_to_repaired_l2,
    pca_surface_coverage,
    pairwise_surface_coverage,
):
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        valid_repaired_x=valid_x,
        validation_x=validation_x,
        latent_mu=latent_mu,
        latent_logvar=latent_logvar,
        reconstruction_x=reconstruction_x,
        generated_z=generated_z,
        generated_x=generated_x,
        reconstruction_l2=reconstruction_l2,
        reconstruction_linf=reconstruction_linf,
        repaired_nearest_other_l2=repaired_nearest_other_l2,
        repaired_to_generated_l2=repaired_to_generated_l2,
        generated_to_repaired_l2=generated_to_repaired_l2,
        surface_coverage_grid_size=np.asarray(SURFACE_COVERAGE_GRID_SIZE),
        pca_surface_reference_occupied_percent=np.asarray(pca_surface_coverage["reference_occupied_percent"]),
        pca_surface_candidate_occupied_percent=np.asarray(pca_surface_coverage["candidate_occupied_percent"]),
        pca_surface_candidate_covers_reference_percent=np.asarray(
            pca_surface_coverage["candidate_covers_reference_percent"],
        ),
        pca_surface_reference_covers_candidate_percent=np.asarray(
            pca_surface_coverage["reference_covers_candidate_percent"],
        ),
        pca_surface_jaccard_percent=np.asarray(pca_surface_coverage["jaccard_percent"]),
        pca_surface_reference_occupied_count=np.asarray(pca_surface_coverage["reference_occupied_count"]),
        pca_surface_candidate_occupied_count=np.asarray(pca_surface_coverage["candidate_occupied_count"]),
        pca_surface_shared_occupied_count=np.asarray(pca_surface_coverage["shared_occupied_count"]),
        pca_surface_union_occupied_count=np.asarray(pca_surface_coverage["union_occupied_count"]),
        pca_surface_total_cell_count=np.asarray(pca_surface_coverage["total_cell_count"]),
        pca_surface_pc1_explained_variance_ratio=np.asarray(
            pca_surface_coverage["pc1_explained_variance_ratio"],
        ),
        pca_surface_pc2_explained_variance_ratio=np.asarray(
            pca_surface_coverage["pc2_explained_variance_ratio"],
        ),
        pairwise_surface_pair_i=pairwise_surface_coverage["pair_i"],
        pairwise_surface_pair_j=pairwise_surface_coverage["pair_j"],
        pairwise_surface_reference_occupied_percent=pairwise_surface_coverage["reference_occupied_percent"],
        pairwise_surface_candidate_occupied_percent=pairwise_surface_coverage["candidate_occupied_percent"],
        pairwise_surface_candidate_covers_reference_percent=(
            pairwise_surface_coverage["candidate_covers_reference_percent"]
        ),
        pairwise_surface_reference_covers_candidate_percent=(
            pairwise_surface_coverage["reference_covers_candidate_percent"]
        ),
        pairwise_surface_jaccard_percent=pairwise_surface_coverage["jaccard_percent"],
        pairwise_surface_reference_occupied_count=pairwise_surface_coverage["reference_occupied_count"],
        pairwise_surface_candidate_occupied_count=pairwise_surface_coverage["candidate_occupied_count"],
        pairwise_surface_shared_occupied_count=pairwise_surface_coverage["shared_occupied_count"],
        pairwise_surface_total_cell_count=pairwise_surface_coverage["total_cell_count"],
        param_names=np.asarray(PARAM_NAMES),
    )


def add_summary_page(pdf, summary_lines):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, "Airframe VAE Validation Summary", fontsize=18, weight="bold")
    fig.text(0.06, 0.88, "\n".join(summary_lines), fontsize=9, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_latent_page(pdf, latent_mu, latent_logvar):
    latent_dim = latent_mu.shape[1]
    if latent_dim >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(latent_mu[:, 0], latent_mu[:, 1], s=8, alpha=0.5)
        axes[0].set_xlabel("mu[0]")
        axes[0].set_ylabel("mu[1]")
        axes[0].set_title("Validation Latent Means")
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(latent_mu.reshape(-1), bins=50, alpha=0.8, label="mu")
        axes[1].hist(np.exp(latent_logvar.reshape(-1)), bins=50, alpha=0.5, label="variance")
        axes[1].set_title("Latent Mean and Variance Distribution")
        axes[1].legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(latent_mu[:, 0], bins=50)
        axes[0].set_title("Validation Latent Mean")
        axes[1].hist(np.exp(latent_logvar[:, 0]), bins=50)
        axes[1].set_title("Validation Latent Variance")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_training_history_page(pdf, training_history):
    if len(training_history) < 1:
        raise ValueError("training_history is empty")
    epochs = np.asarray([record["epoch"] for record in training_history], dtype=int)
    train_total = np.asarray([record["train_total"] for record in training_history], dtype=float)
    validation_total = np.asarray([record["validation_total"] for record in training_history], dtype=float)
    train_mse = np.asarray([record["train_mse"] for record in training_history], dtype=float)
    validation_mse = np.asarray([record["validation_mse"] for record in training_history], dtype=float)
    train_kld = np.asarray([record["train_kld"] for record in training_history], dtype=float)
    validation_kld = np.asarray([record["validation_kld"] for record in training_history], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, train_total, label="train")
    axes[0].plot(epochs, validation_total, label="validation")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, train_mse, label="train")
    axes[1].plot(epochs, validation_mse, label="validation")
    axes[1].set_title("MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(epochs, train_kld, label="train")
    axes[2].plot(epochs, validation_kld, label="validation")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_reconstruction_page(pdf, validation_x, reconstruction_x):
    abs_error = np.abs(reconstruction_x - validation_x)
    l2_error = np.linalg.norm(reconstruction_x - validation_x, axis=1)
    max_abs_error = np.max(abs_error, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(l2_error, bins=50)
    axes[0, 0].set_title("Validation Reconstruction L2 Error")
    axes[0, 1].hist(max_abs_error, bins=50)
    axes[0, 1].set_title("Validation Reconstruction Max Abs Error")
    axes[1, 0].bar(np.arange(abs_error.shape[1]), np.mean(abs_error, axis=0))
    axes[1, 0].set_title("Mean Abs Error Per Parameter")
    axes[1, 0].set_xlabel("Parameter Index")
    axes[1, 1].bar(np.arange(abs_error.shape[1]), np.percentile(abs_error, 95, axis=0))
    axes[1, 1].set_title("95th Percentile Abs Error Per Parameter")
    axes[1, 1].set_xlabel("Parameter Index")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_reconstruction_examples_page(pdf, validation_x, reconstruction_x, param_names):
    reconstruction_l2 = np.linalg.norm(reconstruction_x - validation_x, axis=1)
    sorted_indices = np.argsort(reconstruction_l2)
    selected_indices = np.asarray(
        [
            sorted_indices[0],
            sorted_indices[len(sorted_indices) // 2],
            sorted_indices[int(np.floor(0.95 * float(len(sorted_indices) - 1)))],
            sorted_indices[-1],
        ],
        dtype=int,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    x_positions = np.arange(len(param_names))
    for axis, sample_index, title in zip(
        axes.reshape(-1),
        selected_indices,
        ("best", "median", "95th percentile", "worst"),
    ):
        axis.plot(x_positions, validation_x[sample_index], marker="o", linewidth=1.2, label="validation")
        axis.plot(x_positions, reconstruction_x[sample_index], marker="x", linewidth=1.2, label="reconstruction")
        axis.set_title(f"{title} reconstruction, L2={reconstruction_l2[sample_index]:.4f}")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=0.25)
    axes[1, 0].set_xticks(x_positions)
    axes[1, 0].set_xticklabels(np.arange(len(param_names)), rotation=90)
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(np.arange(len(param_names)), rotation=90)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Representative Validation Reconstructions")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_parameter_distribution_page(pdf, valid_x, validation_x, reconstruction_x, generated_x):
    validation_mean = np.mean(validation_x, axis=0)
    reconstruction_mean = np.mean(reconstruction_x, axis=0)
    generated_mean = np.mean(generated_x, axis=0)
    valid_std = np.std(valid_x, axis=0)
    reconstruction_std = np.std(reconstruction_x, axis=0)
    generated_std = np.std(generated_x, axis=0)

    x_positions = np.arange(valid_x.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(x_positions, validation_mean, marker="o", label="validation mean")
    axes[0].plot(x_positions, reconstruction_mean, marker="x", label="reconstruction mean")
    axes[0].plot(x_positions, generated_mean, marker="s", label="generated mean")
    axes[0].set_title("Per-Parameter Mean")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_positions, valid_std, marker="o", label="valid repaired std")
    axes[1].plot(x_positions, reconstruction_std, marker="x", label="reconstruction std")
    axes[1].plot(x_positions, generated_std, marker="s", label="generated std")
    axes[1].set_title("Per-Parameter Standard Deviation")
    axes[1].set_xlabel("Parameter Index")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pca_coverage_page(pdf, valid_x, validation_x, reconstruction_x, generated_x):
    projections, explained_variance_ratio = pca_project(
        valid_x,
        [valid_x, validation_x, reconstruction_x, generated_x],
    )
    valid_projection, validation_projection, reconstruction_projection, generated_projection = projections

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    axes[0].scatter(valid_projection[:, 0], valid_projection[:, 1], s=10, alpha=0.25, label="all repaired data")
    axes[0].scatter(validation_projection[:, 0], validation_projection[:, 1], s=16, alpha=0.65, label="validation")
    axes[0].scatter(reconstruction_projection[:, 0], reconstruction_projection[:, 1], s=16, alpha=0.65, label="reconstruction")
    axes[0].set_title("Validation Reconstructions in Repaired-Data PCA Space")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(valid_projection[:, 0], valid_projection[:, 1], s=10, alpha=0.25, label="all repaired data")
    axes[1].scatter(generated_projection[:, 0], generated_projection[:, 1], s=16, alpha=0.55, label="decoded N(0, I)")
    axes[1].set_title("Generated Samples in Repaired-Data PCA Space")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(
        f"PCA fitted on repaired airframes, explained variance: "
        f"PC1={explained_variance_ratio[0]:.3f}, PC2={explained_variance_ratio[1]:.3f}"
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_coverage_distance_page(
    pdf,
    repaired_nearest_other_l2,
    repaired_to_generated_l2,
    generated_to_repaired_l2,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(repaired_nearest_other_l2, bins=50, alpha=0.65, label="repaired-to-repaired nearest other")
    axes[0].hist(repaired_to_generated_l2, bins=50, alpha=0.55, label="repaired-to-generated nearest")
    axes[0].set_title("Coverage of Repaired Data by Generated Samples")
    axes[0].set_xlabel("Nearest-neighbor L2 distance")
    axes[0].legend()

    axes[1].hist(generated_to_repaired_l2, bins=50, alpha=0.65, label="generated-to-repaired nearest")
    axes[1].set_title("Distance from Generated Samples to Repaired Data")
    axes[1].set_xlabel("Nearest-neighbor L2 distance")
    axes[1].legend()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_surface_coverage_page(pdf, pca_surface_coverage, pairwise_surface_coverage, param_names):
    n_params = len(param_names)
    coverage_matrix = np.full((n_params, n_params), np.nan, dtype=float)
    for first_index, second_index, coverage_percent in zip(
        pairwise_surface_coverage["pair_i"],
        pairwise_surface_coverage["pair_j"],
        pairwise_surface_coverage["candidate_covers_reference_percent"],
    ):
        coverage_matrix[first_index, second_index] = coverage_percent
        coverage_matrix[second_index, first_index] = coverage_percent

    pairwise_candidate_covers_reference = pairwise_surface_coverage["candidate_covers_reference_percent"]
    pairwise_reference_covers_candidate = pairwise_surface_coverage["reference_covers_candidate_percent"]
    pairwise_jaccard = pairwise_surface_coverage["jaccard_percent"]
    pairwise_reference_occupied = pairwise_surface_coverage["reference_occupied_percent"]
    pairwise_candidate_occupied = pairwise_surface_coverage["candidate_occupied_percent"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [
        "PCA decoded covers repaired",
        "PCA repaired covers decoded",
        "Pairwise median decoded covers repaired",
        "Pairwise median repaired covers decoded",
        "Pairwise median overlap",
    ]
    values = [
        pca_surface_coverage["candidate_covers_reference_percent"],
        pca_surface_coverage["reference_covers_candidate_percent"],
        float(np.median(pairwise_candidate_covers_reference)),
        float(np.median(pairwise_reference_covers_candidate)),
        float(np.median(pairwise_jaccard)),
    ]
    axes[0, 0].bar(np.arange(len(values)), values)
    axes[0, 0].set_ylim(0.0, 100.0)
    axes[0, 0].set_xticks(np.arange(len(values)))
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 0].set_ylabel("Occupied-cell coverage [%]")
    axes[0, 0].set_title("Surface Coverage Summary")
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    axes[0, 1].hist(pairwise_candidate_covers_reference, bins=30, alpha=0.70, label="decoded covers repaired")
    axes[0, 1].hist(pairwise_reference_covers_candidate, bins=30, alpha=0.55, label="repaired covers decoded")
    axes[0, 1].set_xlim(0.0, 100.0)
    axes[0, 1].set_xlabel("Coverage [%]")
    axes[0, 1].set_title("Coverage Across Parameter-Pair Surfaces")
    axes[0, 1].legend()

    image = axes[1, 0].imshow(coverage_matrix, vmin=0.0, vmax=100.0, cmap="viridis")
    axes[1, 0].set_title("Decoded Covers Repaired Surface by Parameter Pair")
    axes[1, 0].set_xlabel("Parameter index")
    axes[1, 0].set_ylabel("Parameter index")
    axes[1, 0].set_xticks(np.arange(n_params))
    axes[1, 0].set_yticks(np.arange(n_params))
    fig.colorbar(image, ax=axes[1, 0], label="Coverage [%]")

    axes[1, 1].scatter(pairwise_reference_occupied, pairwise_candidate_occupied, s=18, alpha=0.7)
    axes[1, 1].set_xlabel("Repaired occupied cells [% of pair grid]")
    axes[1, 1].set_ylabel("Decoded occupied cells [% of pair grid]")
    axes[1, 1].set_title("Visited Surface Size per Parameter Pair")
    axes[1, 1].grid(True, alpha=0.25)

    grid_size = int(pairwise_surface_coverage["grid_size"].item())
    fig.suptitle(
        f"Surface coverage grid: {grid_size}x"
        f"{grid_size} occupied cells"
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def analyze_vae():
    model, checkpoint = load_airframe_vae_checkpoint(CHECKPOINT_PATH, DEVICE)
    if checkpoint["param_names"] != list(PARAM_NAMES):
        raise ValueError("Checkpoint param_names do not match current PARAM_NAMES")

    valid_x, _, _, _ = load_valid_repaired_dataset(DATA_NPZ)
    validation_x = validation_split_from_training_config(valid_x, checkpoint["training_config"])
    latent_mu, latent_logvar, reconstruction_x = encode_decode_validation(model, validation_x, DEVICE)
    generated_z, generated_x = decode_standard_normal_samples(
        model,
        ANALYZE_GENERATED_N_SAMPLES,
        int(checkpoint["latent_dim"]),
        ANALYZE_GENERATED_SEED,
        DEVICE,
    )

    reconstruction_l2 = np.linalg.norm(reconstruction_x - validation_x, axis=1)
    reconstruction_linf = np.max(np.abs(reconstruction_x - validation_x), axis=1)
    reconstruction_abs = np.abs(reconstruction_x - validation_x)

    repaired_nearest_other_l2 = nearest_other_distances(valid_x)
    repaired_to_generated_l2 = nearest_distances(valid_x, generated_x)
    generated_to_repaired_l2 = nearest_distances(generated_x, valid_x)
    pca_surface_coverage = compute_pca_surface_coverage(valid_x, generated_x, SURFACE_COVERAGE_GRID_SIZE)
    pairwise_surface_coverage = compute_pairwise_surface_coverage(valid_x, generated_x, SURFACE_COVERAGE_GRID_SIZE)

    validation_to_reconstruction_l2_median = float(np.median(reconstruction_l2))
    repaired_spacing_l2_median = float(np.median(repaired_nearest_other_l2))
    repaired_to_generated_l2_median = float(np.median(repaired_to_generated_l2))
    generated_to_repaired_l2_median = float(np.median(generated_to_repaired_l2))

    reconstruction_to_spacing_ratio = safe_metric_ratio(
        validation_to_reconstruction_l2_median,
        repaired_spacing_l2_median,
        "reconstruction_to_spacing_ratio",
    )
    generated_coverage_ratio = safe_metric_ratio(
        repaired_to_generated_l2_median,
        repaired_spacing_l2_median,
        "generated_coverage_ratio",
    )

    latent_mu_std = np.std(latent_mu, axis=0)
    latent_variance_mean = np.mean(np.exp(latent_logvar), axis=0)
    valid_std_mean = float(np.mean(np.std(valid_x, axis=0)))
    generated_std_ratio = safe_metric_ratio(
        float(np.mean(np.std(generated_x, axis=0))),
        valid_std_mean,
        "generated_std_ratio",
    )
    latest_training_record = checkpoint["training_history"][-1]
    coverage_success = (
        reconstruction_to_spacing_ratio <= 2.0
        and generated_coverage_ratio <= 2.0
        and float(np.mean(latent_mu_std)) >= 0.1
    )
    if coverage_success:
        coverage_statement = "coverage assessment: PASS by the diagnostics in this report"
    else:
        coverage_statement = "coverage assessment: FAIL by the diagnostics in this report"

    save_analysis_arrays(
        ANALYSIS_NPZ,
        valid_x,
        validation_x,
        latent_mu,
        latent_logvar,
        reconstruction_x,
        generated_z,
        generated_x,
        reconstruction_l2,
        reconstruction_linf,
        repaired_nearest_other_l2,
        repaired_to_generated_l2,
        generated_to_repaired_l2,
        pca_surface_coverage,
        pairwise_surface_coverage,
    )

    summary_lines = [
        f"checkpoint: {CHECKPOINT_PATH}",
        f"data_npz: {DATA_NPZ}",
        f"analysis_npz: {ANALYSIS_NPZ}",
        f"validation samples: {validation_x.shape[0]}",
        f"valid repaired samples: {valid_x.shape[0]}",
        f"latent_dim: {checkpoint['latent_dim']}",
        f"hidden_dimensions: {checkpoint['hidden_dimensions']}",
        f"best checkpoint epoch: {latest_training_record['epoch']}",
        f"best checkpoint validation total/mse/kld: "
        f"{latest_training_record['validation_total']:.8f} / "
        f"{latest_training_record['validation_mse']:.8f} / "
        f"{latest_training_record['validation_kld']:.8f}",
        f"latent_mu std per dimension: {np.array2string(latent_mu_std, precision=6)}",
        f"latent variance mean per dimension: {np.array2string(latent_variance_mean, precision=6)}",
        f"reconstruction_l2 mean/median/p95: "
        f"{np.mean(reconstruction_l2):.8f} / {np.median(reconstruction_l2):.8f} / {np.percentile(reconstruction_l2, 95):.8f}",
        f"reconstruction_linf mean/median/p95: "
        f"{np.mean(reconstruction_linf):.8f} / {np.median(reconstruction_linf):.8f} / {np.percentile(reconstruction_linf, 95):.8f}",
        f"max parameter mean abs error: {np.max(np.mean(reconstruction_abs, axis=0)):.8f}",
        f"repaired empirical nearest-other L2 median/p95: "
        f"{repaired_spacing_l2_median:.8f} / {np.percentile(repaired_nearest_other_l2, 95):.8f}",
        f"repaired-to-generated nearest L2 median/p95: "
        f"{repaired_to_generated_l2_median:.8f} / {np.percentile(repaired_to_generated_l2, 95):.8f}",
        f"generated-to-repaired nearest L2 median/p95: "
        f"{generated_to_repaired_l2_median:.8f} / {np.percentile(generated_to_repaired_l2, 95):.8f}",
        f"reconstruction median / empirical spacing median: {reconstruction_to_spacing_ratio:.3f}",
        f"repaired-to-generated median / empirical spacing median: {generated_coverage_ratio:.3f}",
        f"generated std / repaired-data std: {generated_std_ratio:.3f}",
        f"surface coverage grid: {SURFACE_COVERAGE_GRID_SIZE}x{SURFACE_COVERAGE_GRID_SIZE}",
        f"PCA surface repaired occupied / decoded occupied / shared cells: "
        f"{pca_surface_coverage['reference_occupied_count']} / "
        f"{pca_surface_coverage['candidate_occupied_count']} / "
        f"{pca_surface_coverage['shared_occupied_count']} "
        f"of {pca_surface_coverage['total_cell_count']}",
        f"PCA surface decoded covers repaired / repaired covers decoded / overlap: "
        f"{pca_surface_coverage['candidate_covers_reference_percent']:.2f}% / "
        f"{pca_surface_coverage['reference_covers_candidate_percent']:.2f}% / "
        f"{pca_surface_coverage['jaccard_percent']:.2f}%",
        f"pairwise surface repaired occupied median/p95: "
        f"{np.median(pairwise_surface_coverage['reference_occupied_percent']):.2f}% / "
        f"{np.percentile(pairwise_surface_coverage['reference_occupied_percent'], 95):.2f}%",
        f"pairwise surface decoded occupied median/p95: "
        f"{np.median(pairwise_surface_coverage['candidate_occupied_percent']):.2f}% / "
        f"{np.percentile(pairwise_surface_coverage['candidate_occupied_percent'], 95):.2f}%",
        f"pairwise surface decoded covers repaired min/median/p95: "
        f"{np.min(pairwise_surface_coverage['candidate_covers_reference_percent']):.2f}% / "
        f"{np.median(pairwise_surface_coverage['candidate_covers_reference_percent']):.2f}% / "
        f"{np.percentile(pairwise_surface_coverage['candidate_covers_reference_percent'], 95):.2f}%",
        f"pairwise surface repaired covers decoded min/median/p95: "
        f"{np.min(pairwise_surface_coverage['reference_covers_candidate_percent']):.2f}% / "
        f"{np.median(pairwise_surface_coverage['reference_covers_candidate_percent']):.2f}% / "
        f"{np.percentile(pairwise_surface_coverage['reference_covers_candidate_percent'], 95):.2f}%",
        f"pairwise surface overlap min/median/p95: "
        f"{np.min(pairwise_surface_coverage['jaccard_percent']):.2f}% / "
        f"{np.median(pairwise_surface_coverage['jaccard_percent']):.2f}% / "
        f"{np.percentile(pairwise_surface_coverage['jaccard_percent'], 95):.2f}%",
        coverage_statement,
    ]

    REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT_PDF) as pdf:
        add_summary_page(pdf, summary_lines)
        add_training_history_page(pdf, checkpoint["training_history"])
        add_latent_page(pdf, latent_mu, latent_logvar)
        add_reconstruction_page(pdf, validation_x, reconstruction_x)
        add_reconstruction_examples_page(pdf, validation_x, reconstruction_x, PARAM_NAMES)
        add_parameter_distribution_page(pdf, valid_x, validation_x, reconstruction_x, generated_x)
        add_pca_coverage_page(pdf, valid_x, validation_x, reconstruction_x, generated_x)
        add_coverage_distance_page(
            pdf,
            repaired_nearest_other_l2,
            repaired_to_generated_l2,
            generated_to_repaired_l2,
        )
        add_surface_coverage_page(pdf, pca_surface_coverage, pairwise_surface_coverage, PARAM_NAMES)

    print(f"saved report: {REPORT_PDF}", flush=True)
    print(f"saved analysis arrays: {ANALYSIS_NPZ}", flush=True)
    print("\n".join(summary_lines), flush=True)


def latent_dimension_sweep_values():
    if LATENT_DIMENSION_SWEEP_MIN < 1:
        raise ValueError(f"LATENT_DIMENSION_SWEEP_MIN must be positive, got {LATENT_DIMENSION_SWEEP_MIN}")
    if LATENT_DIMENSION_SWEEP_MAX < LATENT_DIMENSION_SWEEP_MIN:
        raise ValueError(
            f"LATENT_DIMENSION_SWEEP_MAX={LATENT_DIMENSION_SWEEP_MAX} is smaller than "
            f"LATENT_DIMENSION_SWEEP_MIN={LATENT_DIMENSION_SWEEP_MIN}"
        )
    return list(range(LATENT_DIMENSION_SWEEP_MIN, LATENT_DIMENSION_SWEEP_MAX + 1))


def latent_dimension_sweep_paths(latent_dim):
    dimension_dir = LATENT_DIMENSION_SWEEP_DIR / f"latent_dim_{int(latent_dim):02d}"
    checkpoint_path = dimension_dir / "best_model_weights.pth"
    training_config_json = dimension_dir / "training_config.json"
    tensorboard_dir = dimension_dir / "tensorboard"
    return dimension_dir, checkpoint_path, training_config_json, tensorboard_dir


def compute_latent_dimension_sweep_metrics(checkpoint_path, data_npz, device):
    model, checkpoint = load_airframe_vae_checkpoint(checkpoint_path, device)
    if checkpoint["param_names"] != list(PARAM_NAMES):
        raise ValueError(f"Checkpoint param_names do not match current PARAM_NAMES: {checkpoint_path}")

    valid_x, _, _, _ = load_valid_repaired_dataset(data_npz)
    validation_x = validation_split_from_training_config(valid_x, checkpoint["training_config"])
    _, _, reconstruction_x = encode_decode_validation(model, validation_x, device)
    _, generated_x = decode_standard_normal_samples(
        model,
        ANALYZE_GENERATED_N_SAMPLES,
        int(checkpoint["latent_dim"]),
        ANALYZE_GENERATED_SEED,
        device,
    )
    reconstruction_error = reconstruction_x - validation_x
    reconstruction_abs = np.abs(reconstruction_error)
    reconstruction_l2 = np.linalg.norm(reconstruction_error, axis=1)
    reconstruction_linf = np.max(reconstruction_abs, axis=1)
    repaired_nearest_other_l2 = nearest_other_distances(valid_x)
    repaired_to_generated_l2 = nearest_distances(valid_x, generated_x)
    generated_to_repaired_l2 = nearest_distances(generated_x, valid_x)
    repaired_spacing_l2_median = float(np.median(repaired_nearest_other_l2))
    repaired_to_generated_l2_median = float(np.median(repaired_to_generated_l2))
    generated_to_repaired_l2_median = float(np.median(generated_to_repaired_l2))
    pca_surface_coverage = compute_pca_surface_coverage(valid_x, generated_x, SURFACE_COVERAGE_GRID_SIZE)
    pairwise_surface_coverage = compute_pairwise_surface_coverage(valid_x, generated_x, SURFACE_COVERAGE_GRID_SIZE)
    latest_training_record = checkpoint["training_history"][-1]

    return {
        "latent_dim": int(checkpoint["latent_dim"]),
        "checkpoint_path": str(checkpoint_path),
        "training_config": checkpoint["training_config"],
        "best_checkpoint_epoch": int(latest_training_record["epoch"]),
        "best_validation_total": float(latest_training_record["validation_total"]),
        "best_validation_mse": float(latest_training_record["validation_mse"]),
        "best_validation_kld": float(latest_training_record["validation_kld"]),
        "deterministic_reconstruction_mse": float(np.mean(reconstruction_error * reconstruction_error)),
        "deterministic_reconstruction_mse_p95": float(np.percentile(reconstruction_error * reconstruction_error, 95)),
        "reconstruction_l2_mean": float(np.mean(reconstruction_l2)),
        "reconstruction_l2_median": float(np.median(reconstruction_l2)),
        "reconstruction_l2_p95": float(np.percentile(reconstruction_l2, 95)),
        "repaired_spacing_l2_median": repaired_spacing_l2_median,
        "repaired_to_generated_l2_median": repaired_to_generated_l2_median,
        "repaired_to_generated_l2_p95": float(np.percentile(repaired_to_generated_l2, 95)),
        "generated_to_repaired_l2_median": generated_to_repaired_l2_median,
        "generated_to_repaired_l2_p95": float(np.percentile(generated_to_repaired_l2, 95)),
        "generated_coverage_ratio": safe_metric_ratio(
            repaired_to_generated_l2_median,
            repaired_spacing_l2_median,
            "generated_coverage_ratio",
        ),
        "generated_to_repaired_spacing_ratio": safe_metric_ratio(
            generated_to_repaired_l2_median,
            repaired_spacing_l2_median,
            "generated_to_repaired_spacing_ratio",
        ),
        "pca_surface_candidate_covers_reference_percent": float(
            pca_surface_coverage["candidate_covers_reference_percent"],
        ),
        "pca_surface_reference_covers_candidate_percent": float(
            pca_surface_coverage["reference_covers_candidate_percent"],
        ),
        "pca_surface_jaccard_percent": float(pca_surface_coverage["jaccard_percent"]),
        "pairwise_surface_candidate_covers_reference_min": float(
            np.min(pairwise_surface_coverage["candidate_covers_reference_percent"]),
        ),
        "pairwise_surface_candidate_covers_reference_median": float(
            np.median(pairwise_surface_coverage["candidate_covers_reference_percent"]),
        ),
        "pairwise_surface_candidate_covers_reference_p95": float(
            np.percentile(pairwise_surface_coverage["candidate_covers_reference_percent"], 95),
        ),
        "pairwise_surface_reference_covers_candidate_min": float(
            np.min(pairwise_surface_coverage["reference_covers_candidate_percent"]),
        ),
        "pairwise_surface_reference_covers_candidate_median": float(
            np.median(pairwise_surface_coverage["reference_covers_candidate_percent"]),
        ),
        "pairwise_surface_reference_covers_candidate_p95": float(
            np.percentile(pairwise_surface_coverage["reference_covers_candidate_percent"], 95),
        ),
        "pairwise_surface_jaccard_min": float(np.min(pairwise_surface_coverage["jaccard_percent"])),
        "pairwise_surface_jaccard_median": float(np.median(pairwise_surface_coverage["jaccard_percent"])),
        "pairwise_surface_jaccard_p95": float(np.percentile(pairwise_surface_coverage["jaccard_percent"], 95)),
        "n_generated_samples": int(generated_x.shape[0]),
        "surface_coverage_grid_size": int(SURFACE_COVERAGE_GRID_SIZE),
        "n_validation_samples": int(validation_x.shape[0]),
    }


def validate_sweep_metrics(metrics):
    if len(metrics) < 1:
        raise ValueError("No latent-dimension sweep metrics were produced")
    latent_dimensions = [int(record["latent_dim"]) for record in metrics]
    if len(set(latent_dimensions)) != len(latent_dimensions):
        raise ValueError(f"Duplicate latent dimensions in sweep metrics: {latent_dimensions}")

    required_metric_keys = {
        "latent_dim",
        "best_validation_mse",
        "deterministic_reconstruction_mse",
        "deterministic_reconstruction_mse_p95",
        "reconstruction_l2_median",
        "reconstruction_l2_p95",
        "repaired_spacing_l2_median",
        "repaired_to_generated_l2_median",
        "repaired_to_generated_l2_p95",
        "generated_to_repaired_l2_median",
        "generated_to_repaired_l2_p95",
        "generated_coverage_ratio",
        "generated_to_repaired_spacing_ratio",
        "pca_surface_candidate_covers_reference_percent",
        "pca_surface_reference_covers_candidate_percent",
        "pca_surface_jaccard_percent",
        "pairwise_surface_candidate_covers_reference_min",
        "pairwise_surface_candidate_covers_reference_median",
        "pairwise_surface_candidate_covers_reference_p95",
        "pairwise_surface_reference_covers_candidate_min",
        "pairwise_surface_reference_covers_candidate_median",
        "pairwise_surface_reference_covers_candidate_p95",
        "pairwise_surface_jaccard_min",
        "pairwise_surface_jaccard_median",
        "pairwise_surface_jaccard_p95",
        "n_generated_samples",
        "surface_coverage_grid_size",
    }
    for record in metrics:
        missing_keys = required_metric_keys.difference(record.keys())
        if missing_keys:
            raise ValueError(f"Sweep metric record is missing keys {sorted(missing_keys)}: {record}")
        for key in required_metric_keys:
            value = record[key]
            if key == "latent_dim":
                if int(value) < 1:
                    raise ValueError(f"Invalid latent_dim in sweep metrics: {value}")
            elif not np.isfinite(float(value)):
                raise ValueError(f"Sweep metric {key} is not finite for latent_dim={record['latent_dim']}: {value}")


def save_latent_dimension_sweep_metrics(metrics):
    validate_sweep_metrics(metrics)
    LATENT_DIMENSION_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATENT_DIMENSION_SWEEP_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    latent_dimensions = np.asarray([record["latent_dim"] for record in metrics], dtype=int)
    np.savez_compressed(
        LATENT_DIMENSION_SWEEP_METRICS_NPZ,
        latent_dim=latent_dimensions,
        best_validation_mse=np.asarray([record["best_validation_mse"] for record in metrics], dtype=float),
        deterministic_reconstruction_mse=np.asarray(
            [record["deterministic_reconstruction_mse"] for record in metrics],
            dtype=float,
        ),
        deterministic_reconstruction_mse_p95=np.asarray(
            [record["deterministic_reconstruction_mse_p95"] for record in metrics],
            dtype=float,
        ),
        reconstruction_l2_median=np.asarray([record["reconstruction_l2_median"] for record in metrics], dtype=float),
        reconstruction_l2_p95=np.asarray([record["reconstruction_l2_p95"] for record in metrics], dtype=float),
        repaired_spacing_l2_median=np.asarray(
            [record["repaired_spacing_l2_median"] for record in metrics],
            dtype=float,
        ),
        repaired_to_generated_l2_median=np.asarray(
            [record["repaired_to_generated_l2_median"] for record in metrics],
            dtype=float,
        ),
        repaired_to_generated_l2_p95=np.asarray(
            [record["repaired_to_generated_l2_p95"] for record in metrics],
            dtype=float,
        ),
        generated_to_repaired_l2_median=np.asarray(
            [record["generated_to_repaired_l2_median"] for record in metrics],
            dtype=float,
        ),
        generated_to_repaired_l2_p95=np.asarray(
            [record["generated_to_repaired_l2_p95"] for record in metrics],
            dtype=float,
        ),
        generated_coverage_ratio=np.asarray([record["generated_coverage_ratio"] for record in metrics], dtype=float),
        generated_to_repaired_spacing_ratio=np.asarray(
            [record["generated_to_repaired_spacing_ratio"] for record in metrics],
            dtype=float,
        ),
        pca_surface_candidate_covers_reference_percent=np.asarray(
            [record["pca_surface_candidate_covers_reference_percent"] for record in metrics],
            dtype=float,
        ),
        pca_surface_reference_covers_candidate_percent=np.asarray(
            [record["pca_surface_reference_covers_candidate_percent"] for record in metrics],
            dtype=float,
        ),
        pca_surface_jaccard_percent=np.asarray(
            [record["pca_surface_jaccard_percent"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_candidate_covers_reference_min=np.asarray(
            [record["pairwise_surface_candidate_covers_reference_min"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_candidate_covers_reference_median=np.asarray(
            [record["pairwise_surface_candidate_covers_reference_median"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_candidate_covers_reference_p95=np.asarray(
            [record["pairwise_surface_candidate_covers_reference_p95"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_reference_covers_candidate_min=np.asarray(
            [record["pairwise_surface_reference_covers_candidate_min"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_reference_covers_candidate_median=np.asarray(
            [record["pairwise_surface_reference_covers_candidate_median"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_reference_covers_candidate_p95=np.asarray(
            [record["pairwise_surface_reference_covers_candidate_p95"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_jaccard_min=np.asarray(
            [record["pairwise_surface_jaccard_min"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_jaccard_median=np.asarray(
            [record["pairwise_surface_jaccard_median"] for record in metrics],
            dtype=float,
        ),
        pairwise_surface_jaccard_p95=np.asarray(
            [record["pairwise_surface_jaccard_p95"] for record in metrics],
            dtype=float,
        ),
        n_generated_samples=np.asarray([record["n_generated_samples"] for record in metrics], dtype=int),
        surface_coverage_grid_size=np.asarray(
            [record["surface_coverage_grid_size"] for record in metrics],
            dtype=int,
        ),
    )


def plot_latent_dimension_sweep_pdf(metrics):
    validate_sweep_metrics(metrics)
    sorted_metrics = sorted(metrics, key=lambda record: int(record["latent_dim"]))
    latent_dimensions = np.asarray([record["latent_dim"] for record in sorted_metrics], dtype=int)
    best_validation_mse = np.asarray([record["best_validation_mse"] for record in sorted_metrics], dtype=float)
    deterministic_mse = np.asarray(
        [record["deterministic_reconstruction_mse"] for record in sorted_metrics],
        dtype=float,
    )
    deterministic_mse_p95 = np.asarray(
        [record["deterministic_reconstruction_mse_p95"] for record in sorted_metrics],
        dtype=float,
    )
    reconstruction_l2_median = np.asarray(
        [record["reconstruction_l2_median"] for record in sorted_metrics],
        dtype=float,
    )
    reconstruction_l2_p95 = np.asarray([record["reconstruction_l2_p95"] for record in sorted_metrics], dtype=float)
    generated_coverage_ratio = np.asarray(
        [record["generated_coverage_ratio"] for record in sorted_metrics],
        dtype=float,
    )
    generated_to_repaired_spacing_ratio = np.asarray(
        [record["generated_to_repaired_spacing_ratio"] for record in sorted_metrics],
        dtype=float,
    )
    pca_candidate_covers_reference = np.asarray(
        [record["pca_surface_candidate_covers_reference_percent"] for record in sorted_metrics],
        dtype=float,
    )
    pca_reference_covers_candidate = np.asarray(
        [record["pca_surface_reference_covers_candidate_percent"] for record in sorted_metrics],
        dtype=float,
    )
    pca_jaccard = np.asarray([record["pca_surface_jaccard_percent"] for record in sorted_metrics], dtype=float)
    pairwise_candidate_covers_reference_min = np.asarray(
        [record["pairwise_surface_candidate_covers_reference_min"] for record in sorted_metrics],
        dtype=float,
    )
    pairwise_candidate_covers_reference_median = np.asarray(
        [record["pairwise_surface_candidate_covers_reference_median"] for record in sorted_metrics],
        dtype=float,
    )
    pairwise_candidate_covers_reference_p95 = np.asarray(
        [record["pairwise_surface_candidate_covers_reference_p95"] for record in sorted_metrics],
        dtype=float,
    )
    pairwise_reference_covers_candidate_median = np.asarray(
        [record["pairwise_surface_reference_covers_candidate_median"] for record in sorted_metrics],
        dtype=float,
    )
    pairwise_jaccard_median = np.asarray(
        [record["pairwise_surface_jaccard_median"] for record in sorted_metrics],
        dtype=float,
    )

    LATENT_DIMENSION_SWEEP_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(LATENT_DIMENSION_SWEEP_PDF) as pdf:
        fig, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.plot(latent_dimensions, best_validation_mse, marker="o", label="best validation MSE")
        axis.plot(latent_dimensions, deterministic_mse, marker="s", label="decode(mu) MSE")
        axis.plot(latent_dimensions, deterministic_mse_p95, marker="^", label="decode(mu) MSE p95")
        axis.set_title("Per-Parameter Reconstruction Error")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Error")
        axis.set_xticks(latent_dimensions)
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.plot(latent_dimensions, generated_coverage_ratio, marker="o", label="repaired-to-decoded / spacing")
        axis.plot(
            latent_dimensions,
            generated_to_repaired_spacing_ratio,
            marker="s",
            label="decoded-to-repaired / spacing",
        )
        axis.axhline(1.0, color="black", linewidth=1.0, alpha=0.45)
        axis.set_title("Nearest-Neighbor Coverage Ratios")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Ratio to repaired-data spacing")
        axis.set_xticks(latent_dimensions)
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.plot(latent_dimensions, pca_candidate_covers_reference, marker="o", label="decoded covers repaired")
        axis.plot(latent_dimensions, pca_reference_covers_candidate, marker="s", label="repaired covers decoded")
        axis.plot(latent_dimensions, pca_jaccard, marker="^", label="overlap")
        axis.set_title("PCA Surface Coverage")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Occupied-cell coverage [%]")
        axis.set_ylim(0.0, 100.0)
        axis.set_xticks(latent_dimensions)
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.plot(
            latent_dimensions,
            pairwise_candidate_covers_reference_min,
            marker="o",
            label="decoded covers repaired min",
        )
        axis.plot(
            latent_dimensions,
            pairwise_candidate_covers_reference_median,
            marker="s",
            label="decoded covers repaired median",
        )
        axis.plot(
            latent_dimensions,
            pairwise_candidate_covers_reference_p95,
            marker="^",
            label="decoded covers repaired p95",
        )
        axis.plot(
            latent_dimensions,
            pairwise_reference_covers_candidate_median,
            marker="x",
            label="repaired covers decoded median",
        )
        axis.plot(latent_dimensions, pairwise_jaccard_median, marker="d", label="overlap median")
        axis.set_title("Pairwise Parameter-Surface Coverage")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Occupied-cell coverage [%]")
        axis.set_ylim(0.0, 100.0)
        axis.set_xticks(latent_dimensions)
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.plot(latent_dimensions, reconstruction_l2_median, marker="o", label="L2 median")
        axis.plot(latent_dimensions, reconstruction_l2_p95, marker="o", linestyle="--", label="L2 p95")
        axis.set_title("Sample Reconstruction Error")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Error")
        axis.set_xticks(latent_dimensions)
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def train_every_latent_dimension():
    metrics = []
    for latent_dim in latent_dimension_sweep_values():
        dimension_dir, checkpoint_path, training_config_json, tensorboard_dir = latent_dimension_sweep_paths(latent_dim)
        dimension_dir.mkdir(parents=True, exist_ok=True)
        print(f"training latent_dim={latent_dim}", flush=True)
        train_airframe_vae(
            DATA_NPZ,
            checkpoint_path,
            training_config_json,
            tensorboard_dir,
            latent_dim,
        )
        metric_record = compute_latent_dimension_sweep_metrics(checkpoint_path, DATA_NPZ, DEVICE)
        metrics.append(metric_record)
        save_latent_dimension_sweep_metrics(metrics)
        plot_latent_dimension_sweep_pdf(metrics)
        print(
            f"latent_dim={latent_dim} deterministic_mse={metric_record['deterministic_reconstruction_mse']:.8f} "
            f"mse_p95={metric_record['deterministic_reconstruction_mse_p95']:.8f} "
            f"l2_median={metric_record['reconstruction_l2_median']:.8f} "
            f"generated_coverage_ratio={metric_record['generated_coverage_ratio']:.3f} "
            f"pairwise_decoded_covers_repaired_median="
            f"{metric_record['pairwise_surface_candidate_covers_reference_median']:.2f}%",
            flush=True,
        )

    print(f"saved latent-dimension sweep PDF: {LATENT_DIMENSION_SWEEP_PDF}", flush=True)
    print(f"saved latent-dimension sweep metrics: {LATENT_DIMENSION_SWEEP_METRICS_JSON}", flush=True)
    print(f"saved latent-dimension sweep arrays: {LATENT_DIMENSION_SWEEP_METRICS_NPZ}", flush=True)


def main():
    action, action_args = parse_action(sys.argv)
    if action == "--get_samples":
        get_samples()
    elif action == "--train_vae":
        train_vae()
    elif action == "--analyze_vae":
        analyze_vae()
    elif action == "--every_latent_dimension":
        train_every_latent_dimension()
    elif action == "--decode":
        decode_vae_file(*action_args)
    else:
        raise ValueError(f"Unhandled action: {action}")


if __name__ == "__main__":
    main()
