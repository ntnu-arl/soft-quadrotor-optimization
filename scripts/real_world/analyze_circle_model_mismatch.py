import argparse
from pathlib import Path

import numpy as np
import pandas as pd


START_TIME_S = 10.0
END_TIME_S = 24.0
MOVE_SPEED_THRESHOLD_MPS = 0.05
NUM_MOTORS = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def quat_to_rotmat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    qw, qx, qy, qz = quat
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def compute_body_thrust_from_rps(params, motor_rps):
    actual_forces = np.zeros(NUM_MOTORS, dtype=np.float64)
    for motor_index in range(NUM_MOTORS):
        thrust_coeff = params[47 + motor_index]
        if not np.isfinite(thrust_coeff) or not thrust_coeff > 0.0:
            raise RuntimeError(f"invalid thrust coefficient p{47 + motor_index:02d}={thrust_coeff}")
        actual_forces[motor_index] = thrust_coeff * motor_rps[motor_index] * motor_rps[motor_index]

    force_body = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        for motor_index in range(NUM_MOTORS):
            force_body[axis] += params[23 + 6 * motor_index + axis] * actual_forces[motor_index]
    return force_body


def fit_multiaxis_model(name, features, targets):
    coefficients, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
    predictions = features @ coefficients
    ss_res = np.sum((targets - predictions) ** 2, axis=0)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((targets - predictions) ** 2, axis=0))
    print(name)
    print(f"  R2   = {np.round(r2, 4)}")
    print(f"  RMSE = {np.round(rmse, 5)}")
    print(f"  coefficients =\n{np.round(coefficients, 5)}")
    return coefficients, predictions


def fit_single_axis_model(name, features, target):
    coefficients, _, _, _ = np.linalg.lstsq(features, target, rcond=None)
    prediction = features @ coefficients
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((target - prediction) ** 2))
    print(f"{name}: coefficients={np.round(coefficients, 5)} R2={r2:.4f} RMSE={rmse:.5f}")
    return coefficients, prediction


def compute_world_force_metrics(predicted_force_world, target_force_world, tangent_world, radial_world):
    tangential_prediction = np.sum(predicted_force_world * tangent_world, axis=1)
    tangential_target = np.sum(target_force_world * tangent_world, axis=1)
    radial_prediction = np.sum(predicted_force_world * radial_world, axis=1)
    radial_target = np.sum(target_force_world * radial_world, axis=1)
    world_z_prediction = predicted_force_world[:, 2]
    world_z_target = target_force_world[:, 2]
    return {
        "force_rmse_norm": float(np.sqrt(np.mean(np.sum((target_force_world - predicted_force_world) ** 2, axis=1)))),
        "tangent_rmse": float(np.sqrt(np.mean((tangential_target - tangential_prediction) ** 2))),
        "tangent_r2": float(1.0 - np.sum((tangential_target - tangential_prediction) ** 2) / np.sum((tangential_target - np.mean(tangential_target)) ** 2)),
        "radial_rmse": float(np.sqrt(np.mean((radial_target - radial_prediction) ** 2))),
        "world_z_rmse": float(np.sqrt(np.mean((world_z_target - world_z_prediction) ** 2))),
        "world_z_r2": float(1.0 - np.sum((world_z_target - world_z_prediction) ** 2) / np.sum((world_z_target - np.mean(world_z_target)) ** 2)),
        "world_z_bias_prediction": float(np.mean(world_z_prediction)),
    }


def fit_shared_thrust_efficiency_model(name, scalar_features, thrust_body, targets):
    sample_count = targets.shape[0]
    feature_count = scalar_features.shape[1]
    design_matrix = np.zeros((sample_count * 3, 3 + feature_count), dtype=np.float64)
    stacked_targets = targets.reshape(sample_count * 3)

    for sample_index in range(sample_count):
        row_base = 3 * sample_index
        for axis in range(3):
            design_matrix[row_base + axis, axis] = 1.0
            design_matrix[row_base + axis, 3:] = thrust_body[sample_index, axis] * scalar_features[sample_index]

    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, stacked_targets, rcond=None)
    bias = coefficients[:3]
    shared_coefficients = coefficients[3:]
    scalar_loss = scalar_features @ shared_coefficients
    prediction = bias[None, :] + thrust_body * scalar_loss[:, None]

    ss_res = np.sum((targets - prediction) ** 2, axis=0)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((targets - prediction) ** 2, axis=0))
    print(name)
    print(f"  R2   = {np.round(r2, 4)}")
    print(f"  RMSE = {np.round(rmse, 5)}")
    print(f"  bias = {np.round(bias, 5)}")
    print(f"  shared_coefficients = {np.round(shared_coefficients, 5)}")
    return bias, shared_coefficients, prediction, scalar_loss


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    solver_path = run_dir / "nmpc_solver_data_combined.csv"
    offline_path = run_dir / "nmpc_solver_data_combined_offline_disturbance.csv"
    if not solver_path.is_file():
        raise FileNotFoundError(f"missing solver log: {solver_path}")
    if not offline_path.is_file():
        raise FileNotFoundError(f"missing offline disturbance log: {offline_path}")

    solver_df = pd.read_csv(solver_path)
    offline_df = pd.read_csv(offline_path)
    merged_df = solver_df.merge(
        offline_df[
            [
                "seq",
                "estimated_force_bx",
                "estimated_force_by",
                "estimated_force_bz",
                "estimated_torque_bx",
                "estimated_torque_by",
                "estimated_torque_bz",
            ]
        ],
        on="seq",
        how="inner",
        validate="one_to_one",
    )

    time_s = (merged_df["t_us"].to_numpy(dtype=np.float64) - merged_df["t_us"].iloc[0]) * 1.0e-6
    window_mask = (time_s >= START_TIME_S) & (time_s <= END_TIME_S)
    df = merged_df.loc[window_mask].copy()
    time_s = time_s[window_mask]

    reference_velocity_world = df[["sp_vx", "sp_vy", "sp_vz"]].to_numpy(dtype=np.float64)
    reference_speed = np.linalg.norm(reference_velocity_world[:, 0:2], axis=1)
    move_mask = reference_speed > MOVE_SPEED_THRESHOLD_MPS
    if not np.any(move_mask):
        raise RuntimeError("no moving samples found in the analysis window")

    phase = np.arctan2(reference_velocity_world[:, 0], reference_velocity_world[:, 1])
    state_quat = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    state_velocity_world = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    state_velocity_body = np.empty_like(state_velocity_world)
    estimated_force_body = df[["estimated_force_bx", "estimated_force_by", "estimated_force_bz"]].to_numpy(dtype=np.float64)
    estimated_torque_body = df[["estimated_torque_bx", "estimated_torque_by", "estimated_torque_bz"]].to_numpy(dtype=np.float64)
    estimated_force_world = np.empty_like(estimated_force_body)
    thrust_force_body = np.empty_like(estimated_force_body)
    tilt_metric = np.empty(len(df), dtype=np.float64)
    params = np.array([float(df[f"p{param_index:02d}"].iloc[0]) for param_index in range(68)], dtype=np.float64)
    corrected_rps = df[[f"corrected_rps{motor_index}" for motor_index in range(NUM_MOTORS)]].to_numpy(dtype=np.float64)

    for sample_index in range(len(df)):
        rotation_world_from_body = quat_to_rotmat(state_quat[sample_index])
        state_velocity_body[sample_index] = rotation_world_from_body.T @ state_velocity_world[sample_index]
        estimated_force_world[sample_index] = rotation_world_from_body @ estimated_force_body[sample_index]
        thrust_force_body[sample_index] = compute_body_thrust_from_rps(params, corrected_rps[sample_index])
        tilt_metric[sample_index] = 1.0 - rotation_world_from_body[2, 2]

    move_force_body = estimated_force_body[move_mask]
    move_torque_body = estimated_torque_body[move_mask]
    move_velocity_body = state_velocity_body[move_mask]
    move_force_world = estimated_force_world[move_mask]
    move_thrust_force_body = thrust_force_body[move_mask]
    move_tilt_metric = tilt_metric[move_mask]
    move_phase = phase[move_mask]
    move_reference_speed = reference_speed[move_mask]
    move_lateral_speed_body = np.linalg.norm(move_velocity_body[:, 0:2], axis=1)

    print(f"run_dir = {run_dir}")
    print(f"window = [{START_TIME_S}, {END_TIME_S}] s")
    print(f"samples = {len(df)}, moving_samples = {int(np.sum(move_mask))}")
    print(f"mass_kg = {df['p00'].iloc[0]:.9f}")
    print(f"force_mean_N = {np.round(np.mean(move_force_body, axis=0), 5)}")
    print(f"force_std_N = {np.round(np.std(move_force_body, axis=0), 5)}")
    print(f"torque_mean_Nm = {np.round(np.mean(move_torque_body, axis=0), 6)}")
    print(f"torque_std_Nm = {np.round(np.std(move_torque_body, axis=0), 6)}")

    phase_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            np.sin(move_phase),
            np.cos(move_phase),
        ]
    )
    velocity_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_velocity_body[:, 0],
            move_velocity_body[:, 1],
            move_velocity_body[:, 2],
        ]
    )
    velocity_quadratic_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_velocity_body[:, 0],
            move_velocity_body[:, 1],
            move_velocity_body[:, 2],
            np.abs(move_velocity_body[:, 0]) * move_velocity_body[:, 0],
            np.abs(move_velocity_body[:, 1]) * move_velocity_body[:, 1],
            np.abs(move_velocity_body[:, 2]) * move_velocity_body[:, 2],
        ]
    )

    fit_multiaxis_model("force_from_phase", phase_features, move_force_body)
    linear_force_coefficients, linear_force_prediction = fit_multiaxis_model(
        "force_from_linear_body_velocity",
        velocity_features,
        move_force_body,
    )
    fit_multiaxis_model("force_from_linear_plus_quadratic_body_velocity", velocity_quadratic_features, move_force_body)
    fit_multiaxis_model("torque_from_phase", phase_features, move_torque_body)
    fit_multiaxis_model("torque_from_linear_body_velocity", velocity_features, move_torque_body)

    fit_single_axis_model(
        "bx_from_body_vx",
        np.column_stack([np.ones(int(np.sum(move_mask)), dtype=np.float64), move_velocity_body[:, 0]]),
        move_force_body[:, 0],
    )
    fit_single_axis_model(
        "by_from_body_vy",
        np.column_stack([np.ones(int(np.sum(move_mask)), dtype=np.float64), move_velocity_body[:, 1]]),
        move_force_body[:, 1],
    )
    fit_single_axis_model(
        "bz_from_body_vz",
        np.column_stack([np.ones(int(np.sum(move_mask)), dtype=np.float64), move_velocity_body[:, 2]]),
        move_force_body[:, 2],
    )

    tangent_world = np.zeros_like(reference_velocity_world)
    tangent_world[move_mask, 0:2] = reference_velocity_world[move_mask, 0:2] / move_reference_speed[:, None]
    radial_world = np.column_stack(
        [
            tangent_world[:, 1],
            -tangent_world[:, 0],
            np.zeros(len(df), dtype=np.float64),
        ]
    )
    tangential_force_world = np.sum(estimated_force_world * tangent_world, axis=1)
    radial_force_world = np.sum(estimated_force_world * radial_world, axis=1)
    print(
        "world_force_tangential_N mean/std/maxabs = "
        f"{np.mean(tangential_force_world[move_mask]):.5f} / "
        f"{np.std(tangential_force_world[move_mask]):.5f} / "
        f"{np.max(np.abs(tangential_force_world[move_mask])):.5f}"
    )
    print(
        "world_force_radial_N mean/std/maxabs = "
        f"{np.mean(radial_force_world[move_mask]):.5f} / "
        f"{np.std(radial_force_world[move_mask]):.5f} / "
        f"{np.max(np.abs(radial_force_world[move_mask])):.5f}"
    )

    estimated_rps = df[[f"estimated_rps{motor_index}" for motor_index in range(4)]].to_numpy(dtype=np.float64)
    corrected_rps = df[[f"corrected_rps{motor_index}" for motor_index in range(4)]].to_numpy(dtype=np.float64)
    corrected_minus_estimated_rps = corrected_rps - estimated_rps
    print(f"mean_abs_corrected_minus_estimated_rps = {np.round(np.mean(np.abs(corrected_minus_estimated_rps[move_mask]), axis=0), 5)}")
    print(f"max_abs_corrected_minus_estimated_rps = {np.round(np.max(np.abs(corrected_minus_estimated_rps[move_mask]), axis=0), 5)}")

    planar_drag_force_prediction = np.column_stack(
        [
            linear_force_coefficients[0, 0] + linear_force_coefficients[1, 0] * move_velocity_body[:, 0],
            linear_force_coefficients[0, 1] + linear_force_coefficients[2, 1] * move_velocity_body[:, 1],
            np.full(int(np.sum(move_mask)), np.mean(move_force_body[:, 2]), dtype=np.float64),
        ]
    )
    planar_drag_residual = move_force_body - planar_drag_force_prediction
    full_linear_residual = move_force_body - linear_force_prediction
    print(
        "baseline_force_rmse_norm_N = "
        f"{np.sqrt(np.mean(np.sum(move_force_body ** 2, axis=1))):.5f}"
    )
    print(
        "after_planar_drag_force_rmse_norm_N = "
        f"{np.sqrt(np.mean(np.sum(planar_drag_residual ** 2, axis=1))):.5f}"
    )
    print(
        "after_full_linear_force_rmse_norm_N = "
        f"{np.sqrt(np.mean(np.sum(full_linear_residual ** 2, axis=1))):.5f}"
    )

    move_tangent_world = tangent_world[move_mask]
    move_radial_world = radial_world[move_mask]
    move_rotation_world_from_body = np.array([quat_to_rotmat(quat) for quat in state_quat[move_mask]], dtype=np.float64)

    additive_planar_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_velocity_body[:, 0],
            move_velocity_body[:, 1],
        ]
    )
    additive_full_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_velocity_body[:, 0],
            move_velocity_body[:, 1],
            move_velocity_body[:, 2],
        ]
    )
    thrust_constant_features = np.ones((int(np.sum(move_mask)), 1), dtype=np.float64)
    thrust_speed_features = np.column_stack([np.ones(int(np.sum(move_mask)), dtype=np.float64), move_lateral_speed_body])
    thrust_tilt_features = np.column_stack([np.ones(int(np.sum(move_mask)), dtype=np.float64), move_tilt_metric])
    thrust_speed_tilt_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_lateral_speed_body,
            move_tilt_metric,
        ]
    )
    thrust_interaction_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_lateral_speed_body,
            move_tilt_metric,
            move_lateral_speed_body * move_tilt_metric,
        ]
    )
    combined_features = np.column_stack(
        [
            np.ones(int(np.sum(move_mask)), dtype=np.float64),
            move_velocity_body[:, 0],
            move_velocity_body[:, 1],
            move_thrust_force_body[:, 0],
            move_thrust_force_body[:, 0] * move_lateral_speed_body,
            move_thrust_force_body[:, 0] * move_tilt_metric,
            move_thrust_force_body[:, 1],
            move_thrust_force_body[:, 1] * move_lateral_speed_body,
            move_thrust_force_body[:, 1] * move_tilt_metric,
            move_thrust_force_body[:, 2],
            move_thrust_force_body[:, 2] * move_lateral_speed_body,
            move_thrust_force_body[:, 2] * move_tilt_metric,
        ]
    )

    model_predictions = {}
    _, additive_planar_prediction = fit_multiaxis_model("force_from_planar_body_velocity", additive_planar_features, move_force_body)
    model_predictions["additive_planar_drag"] = additive_planar_prediction
    _, additive_full_prediction = fit_multiaxis_model("force_from_full_body_velocity", additive_full_features, move_force_body)
    model_predictions["additive_full_velocity"] = additive_full_prediction
    _, _, thrust_constant_prediction, thrust_constant_loss = fit_shared_thrust_efficiency_model(
        "force_from_constant_thrust_efficiency_loss",
        thrust_constant_features,
        move_thrust_force_body,
        move_force_body,
    )
    model_predictions["thrust_constant"] = thrust_constant_prediction
    _, _, thrust_speed_prediction, thrust_speed_loss = fit_shared_thrust_efficiency_model(
        "force_from_speed_dependent_thrust_efficiency_loss",
        thrust_speed_features,
        move_thrust_force_body,
        move_force_body,
    )
    model_predictions["thrust_speed"] = thrust_speed_prediction
    _, _, thrust_tilt_prediction, thrust_tilt_loss = fit_shared_thrust_efficiency_model(
        "force_from_tilt_dependent_thrust_efficiency_loss",
        thrust_tilt_features,
        move_thrust_force_body,
        move_force_body,
    )
    model_predictions["thrust_tilt"] = thrust_tilt_prediction
    _, _, thrust_speed_tilt_prediction, thrust_speed_tilt_loss = fit_shared_thrust_efficiency_model(
        "force_from_speed_plus_tilt_thrust_efficiency_loss",
        thrust_speed_tilt_features,
        move_thrust_force_body,
        move_force_body,
    )
    model_predictions["thrust_speed_tilt"] = thrust_speed_tilt_prediction
    _, _, thrust_interaction_prediction, thrust_interaction_loss = fit_shared_thrust_efficiency_model(
        "force_from_speed_tilt_interaction_thrust_efficiency_loss",
        thrust_interaction_features,
        move_thrust_force_body,
        move_force_body,
    )
    model_predictions["thrust_speed_tilt_interaction"] = thrust_interaction_prediction
    _, combined_prediction = fit_multiaxis_model("force_from_planar_drag_plus_axiswise_thrust_terms", combined_features, move_force_body)
    model_predictions["combined_planar_plus_thrust_terms"] = combined_prediction

    print("model_comparison_world_metrics")
    for model_name, body_prediction in model_predictions.items():
        predicted_force_world = np.einsum("nij,nj->ni", move_rotation_world_from_body, body_prediction)
        metrics = compute_world_force_metrics(predicted_force_world, move_force_world, move_tangent_world, move_radial_world)
        print(
            f"  {model_name}: "
            f"force_rmse_norm={metrics['force_rmse_norm']:.5f} "
            f"tangent_rmse={metrics['tangent_rmse']:.5f} "
            f"tangent_r2={metrics['tangent_r2']:.4f} "
            f"world_z_rmse={metrics['world_z_rmse']:.5f} "
            f"world_z_r2={metrics['world_z_r2']:.4f} "
            f"world_z_bias_prediction={metrics['world_z_bias_prediction']:.5f}"
        )

    thrust_norm = np.linalg.norm(move_thrust_force_body, axis=1)
    valid_thrust_mask = thrust_norm > 1.0e-9
    unit_thrust = np.zeros_like(move_thrust_force_body)
    unit_thrust[valid_thrust_mask] = move_thrust_force_body[valid_thrust_mask] / thrust_norm[valid_thrust_mask, None]
    residual_parallel = np.sum(move_force_body * unit_thrust, axis=1)[:, None] * unit_thrust
    residual_orthogonal = move_force_body - residual_parallel
    print(
        "parallel_to_thrust_energy_fraction = "
        f"{np.sum(residual_parallel ** 2) / np.sum(move_force_body ** 2):.4f}"
    )
    print(
        "scalar_loss_statistics"
        f" constant_mean={np.mean(thrust_constant_loss):.5f}"
        f" speed_mean={np.mean(thrust_speed_loss):.5f}"
        f" tilt_mean={np.mean(thrust_tilt_loss):.5f}"
        f" speed_tilt_mean={np.mean(thrust_speed_tilt_loss):.5f}"
        f" interaction_mean={np.mean(thrust_interaction_loss):.5f}"
    )
    print(
        "position_error_circle_mean_m = "
        f"{np.round(np.mean(df.loc[move_mask, ['px', 'py', 'pz']].to_numpy(dtype=np.float64) - df.loc[move_mask, ['sp_px', 'sp_py', 'sp_pz']].to_numpy(dtype=np.float64), axis=0), 5)}"
    )
    print(
        "position_error_circle_std_m = "
        f"{np.round(np.std(df.loc[move_mask, ['px', 'py', 'pz']].to_numpy(dtype=np.float64) - df.loc[move_mask, ['sp_px', 'sp_py', 'sp_pz']].to_numpy(dtype=np.float64), axis=0), 5)}"
    )


if __name__ == "__main__":
    main()
