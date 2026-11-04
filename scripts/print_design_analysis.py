#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airframe_encoding import PARAM_BOUNDS, Airframe
from airframe_validations import build_collision_torque_response
from arm_detachment_torque import one_sided_detachment_envelope_rad_nm
from erc_torque_profiles import (
    DEFAULT_SPRING_CATALOG,
    DEFAULT_SPRING_ID,
    DETACHMENT_MIN_ANGLE_DEG,
    DETACHMENT_N_SAMPLES,
    DETACHMENT_REQUEST_MARGIN_NM,
    DETACHMENT_SAFETY_FACTOR,
    FormulaTorqueParameters,
    build_formula_erc_torque_function,
)

TORQUE_RESPONSE_ERC_N_GRID = 1000
TORQUE_RESPONSE_ERC_SAFETY_FACTOR = 1.1
TORQUE_RESPONSE_ERC_MAX_TORQUE_RMSE_NM = 0.6


def build_airframe_from_args(args_cli):
    if args_cli.params_list is not None:
        if len(args_cli.params_list) != len(PARAM_BOUNDS):
            raise ValueError(
                f"--params_list expects {len(PARAM_BOUNDS)} values, got {len(args_cli.params_list)}"
            )
        airframe = Airframe.from_x_0_1(args_cli.params_list)
    else:
        missing = [name for name in PARAM_BOUNDS if getattr(args_cli, name) is None]
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. Use --params_list or provide all individual parameters."
            )
        airframe = Airframe.from_params_dict({name: getattr(args_cli, name) for name in PARAM_BOUNDS})
    return airframe


def build_design_bundle_from_args(args_cli):
    if args_cli.best_solution_json is not None:
        with open(args_cli.best_solution_json, "r", encoding="utf-8") as handle:
            best_solution = json.load(handle)
        geometry_x_0_1 = best_solution["geometry_x_0_1"]
        if len(geometry_x_0_1) != len(PARAM_BOUNDS):
            raise ValueError(
                f"geometry_x_0_1 must have {len(PARAM_BOUNDS)} entries, got {len(geometry_x_0_1)}"
            )
        airframe = Airframe.from_x_0_1(geometry_x_0_1)
        torque_response = {
            "source": os.path.abspath(args_cli.best_solution_json),
            "max_left_deg": float(best_solution["torque_response_end_angle_left_deg"]),
            "max_right_deg": -float(best_solution["torque_response_end_angle_right_deg"]),
            "end_angle_b_left": float(best_solution["torque_response_end_angle_left_deg"]),
            "end_angle_b_right": float(best_solution["torque_response_end_angle_right_deg"]),
            "torque_response_a": float(best_solution["torque_response_a"]),
            "torque_response_b": float(best_solution["torque_response_b"]),
            "torque_response_c": float(best_solution["torque_response_c"]),
            "torque_response_d": float(best_solution["torque_response_d"]),
            "torque_response_end_angle_left_rad": float(best_solution["torque_response_end_angle_left_rad"]),
            "torque_response_end_angle_right_rad": float(best_solution["torque_response_end_angle_right_rad"]),
            "torque_fn_left": best_solution["torque_fn_left"],
            "torque_fn_right": best_solution["torque_fn_right"],
        }
        return airframe, torque_response

    airframe = build_airframe_from_args(args_cli)
    torque_response = build_collision_torque_response(airframe, delta_deg=0.01, verbose=False)
    torque_response["source"] = "collision_limited_geometry"
    return airframe, torque_response


def save_torque_response_plot(
    torque_response,
    airframe_data,
    torque_fn_left_str,
    torque_fn_right_str,
    output_path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace_left = {"torch": torch}
    namespace_right = {"torch": torch}
    exec(torque_fn_left_str, namespace_left)
    exec(torque_fn_right_str, namespace_right)

    torque_fn_left = namespace_left["torque_fn_left"]
    torque_fn_right = namespace_right["torque_fn_right"]

    required_torque_response_keys = (
        "torque_response_a",
        "torque_response_b",
        "torque_response_c",
        "torque_response_d",
        "torque_response_end_angle_left_rad",
        "torque_response_end_angle_right_rad",
    )
    missing_torque_response_keys = [
        key for key in required_torque_response_keys if key not in torque_response
    ]
    if missing_torque_response_keys:
        raise ValueError(
            "Torque response geometry plots require ERC formula metadata. "
            f"Missing keys: {missing_torque_response_keys}"
        )

    min_detachment_angle_rad = math.radians(DETACHMENT_MIN_ANGLE_DEG)
    left_detachment_envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        1,
        float(torque_response["torque_response_end_angle_left_rad"]),
        DETACHMENT_N_SAMPLES,
        DETACHMENT_SAFETY_FACTOR,
        1,
        min_detachment_angle_rad,
    )
    right_detachment_envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        3,
        float(torque_response["torque_response_end_angle_right_rad"]),
        DETACHMENT_N_SAMPLES,
        DETACHMENT_SAFETY_FACTOR,
        -1,
        min_detachment_angle_rad,
    )
    left_build = build_formula_erc_torque_function(
        FormulaTorqueParameters(
            float(torque_response["torque_response_a"]),
            float(torque_response["torque_response_b"]),
            float(torque_response["torque_response_c"]),
            float(torque_response["torque_response_d"]),
            float(torque_response["torque_response_end_angle_left_rad"]),
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
            float(torque_response["torque_response_a"]),
            float(torque_response["torque_response_b"]),
            float(torque_response["torque_response_c"]),
            float(torque_response["torque_response_d"]),
            float(torque_response["torque_response_end_angle_right_rad"]),
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

    if "max_angle_a_left" in namespace_left and "end_angle_b_left" in namespace_left:
        angles_left = torch.linspace(
            -float(namespace_left["max_angle_a_left"]),
            float(namespace_left["end_angle_b_left"]),
            1000,
        )
    elif hasattr(torque_fn_left, "erc_angle_min_rad") and hasattr(torque_fn_left, "erc_angle_max_rad"):
        angles_left = torch.linspace(
            float(torque_fn_left.erc_angle_min_rad),
            float(torque_fn_left.erc_angle_max_rad),
            1000,
        )
    else:
        raise RuntimeError("Unable to infer left torque plot domain from torque function string.")

    if "max_angle_a_right" in namespace_right and "end_angle_b_right" in namespace_right:
        angles_right = torch.linspace(
            -float(namespace_right["end_angle_b_right"]),
            float(namespace_right["max_angle_a_right"]),
            1000,
        )
    elif hasattr(torque_fn_right, "erc_angle_min_rad") and hasattr(torque_fn_right, "erc_angle_max_rad"):
        angles_right = torch.linspace(
            float(torque_fn_right.erc_angle_min_rad),
            float(torque_fn_right.erc_angle_max_rad),
            1000,
        )
    else:
        raise RuntimeError("Unable to infer right torque plot domain from torque function string.")

    with torch.no_grad():
        torques_left = torque_fn_left(angles_left)
        torques_right = torque_fn_right(angles_right)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axis_left = axes[0, 0]
    axis_right = axes[0, 1]
    axis_erc_left = axes[1, 0]
    axis_erc_right = axes[1, 1]

    angles_left_deg = angles_left.numpy() * 180.0 / np.pi
    torques_left_np = torques_left.numpy()
    angles_right_deg = angles_right.numpy() * 180.0 / np.pi
    torques_right_np = torques_right.numpy()
    cam_left_mm = left_build.design_result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0
    cam_right_mm = right_build.design_result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0

    axis_left.plot(angles_left_deg, torques_left_np, "b-", linewidth=2)
    axis_left.axvline(x=0.0, color="k", linestyle="--", alpha=0.3)
    axis_left.axhline(y=0.0, color="k", linestyle="--", alpha=0.3)
    axis_left.set_xlabel("Joint Angle (degrees)")
    axis_left.set_ylabel("Torque (N*m)")
    axis_left.set_title("Left Arm Torque Response")
    axis_left.grid(True, alpha=0.3)

    axis_right.plot(angles_right_deg, torques_right_np, "r-", linewidth=2)
    axis_right.axvline(x=0.0, color="k", linestyle="--", alpha=0.3)
    axis_right.axhline(y=0.0, color="k", linestyle="--", alpha=0.3)
    axis_right.set_xlabel("Joint Angle (degrees)")
    axis_right.set_ylabel("Torque (N*m)")
    axis_right.set_title("Right Arm Torque Response")
    axis_right.grid(True, alpha=0.3)

    axis_erc_left.plot(cam_left_mm[:, 0], cam_left_mm[:, 1], "b-", linewidth=2)
    axis_erc_left.scatter([0.0], [0.0], color="k", s=18)
    axis_erc_left.set_xlabel("x [mm]")
    axis_erc_left.set_ylabel("y [mm]")
    axis_erc_left.set_title("Left Arm Joint Geometry")
    axis_erc_left.axis("equal")
    axis_erc_left.grid(True, alpha=0.3)

    axis_erc_right.plot(cam_right_mm[:, 0], cam_right_mm[:, 1], "r-", linewidth=2)
    axis_erc_right.scatter([0.0], [0.0], color="k", s=18)
    axis_erc_right.set_xlabel("x [mm]")
    axis_erc_right.set_ylabel("y [mm]")
    axis_erc_right.set_title("Right Arm Joint Geometry")
    axis_erc_right.axis("equal")
    axis_erc_right.grid(True, alpha=0.3)

    annotation_step_deg = 0.25

    def sampled_polyline_points(torque_function, angle_min_rad, angle_max_rad):
        angle_min_deg = float(np.rad2deg(angle_min_rad))
        angle_max_deg = float(np.rad2deg(angle_max_rad))
        if angle_max_deg <= angle_min_deg:
            raise ValueError("Annotation sampling domain must have positive width.")

        first_grid_index = int(math.ceil(angle_min_deg / annotation_step_deg))
        last_grid_index = int(math.floor(angle_max_deg / annotation_step_deg))
        sampled_angles_deg = [angle_min_deg]
        sampled_angles_deg.extend(
            grid_index * annotation_step_deg
            for grid_index in range(first_grid_index, last_grid_index + 1)
        )
        sampled_angles_deg.append(angle_max_deg)

        deduplicated_angles_deg = []
        for angle_deg in sorted(sampled_angles_deg):
            if deduplicated_angles_deg and abs(angle_deg - deduplicated_angles_deg[-1]) <= 1e-12:
                continue
            deduplicated_angles_deg.append(float(angle_deg))

        sampled_angles_rad = [float(np.deg2rad(angle_deg)) for angle_deg in deduplicated_angles_deg]
        query_angles = torch.tensor(sampled_angles_rad, dtype=torch.float64)
        with torch.no_grad():
            sampled_torques = torque_function(query_angles).detach().cpu().numpy()
        return tuple(
            (float(angle_rad), float(torque_nm))
            for angle_rad, torque_nm in zip(sampled_angles_rad, sampled_torques)
        )

    def visually_distinct_points(axis, annotation_points_rad_nm):
        if len(annotation_points_rad_nm) < 3:
            return tuple(annotation_points_rad_nm)

        angles_deg = np.array([np.rad2deg(point[0]) for point in annotation_points_rad_nm], dtype=float)
        torques_nm = np.array([point[1] for point in annotation_points_rad_nm], dtype=float)
        display_points = axis.transData.transform(np.column_stack([angles_deg, torques_nm]))

        for point_index in range(len(annotation_points_rad_nm) - 1):
            if annotation_points_rad_nm[point_index + 1][0] <= annotation_points_rad_nm[point_index][0]:
                raise ValueError("Annotation points must have strictly increasing angles.")

        pixel_tolerance = 8.0

        def point_segment_distance(point_xy, segment_start_xy, segment_end_xy):
            segment_vector = segment_end_xy - segment_start_xy
            segment_length_squared = float(np.dot(segment_vector, segment_vector))
            if segment_length_squared <= 0.0:
                return float(np.linalg.norm(point_xy - segment_start_xy))
            projection = float(np.dot(point_xy - segment_start_xy, segment_vector)) / segment_length_squared
            projection = min(1.0, max(0.0, projection))
            closest_point = segment_start_xy + projection * segment_vector
            return float(np.linalg.norm(point_xy - closest_point))

        selected_point_indices = {
            int(np.argmin(np.abs(angles_deg))),
            int(np.argmin(torques_nm)),
            int(np.argmax(torques_nm)),
        }

        def simplify_segment(first_index, last_index):
            if last_index <= first_index + 1:
                return

            start_xy = display_points[first_index]
            end_xy = display_points[last_index]
            max_distance = -1.0
            max_distance_index = None
            for point_index in range(first_index + 1, last_index):
                distance = point_segment_distance(display_points[point_index], start_xy, end_xy)
                if distance > max_distance:
                    max_distance = distance
                    max_distance_index = point_index

            if max_distance_index is None:
                return
            if max_distance > pixel_tolerance:
                selected_point_indices.add(max_distance_index)
                simplify_segment(first_index, max_distance_index)
                simplify_segment(max_distance_index, last_index)

        simplify_segment(0, len(annotation_points_rad_nm) - 1)
        return tuple(annotation_points_rad_nm[point_index] for point_index in sorted(selected_point_indices))

    def annotate_curve(axis, annotation_points_rad_nm):
        for angle_rad, torque_nm in visually_distinct_points(axis, annotation_points_rad_nm):
            x_value = float(np.rad2deg(angle_rad))
            y_value = float(torque_nm)
            axis.plot(x_value, y_value, "ko", markersize=3)
            axis.annotate(
                f"({x_value:.2f}, {y_value:.3f})",
                xy=(x_value, y_value),
                xytext=(4, 6 if y_value <= 0.0 else -10),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.7", "alpha": 0.85},
            )

    annotate_curve(
        axis_left,
        sampled_polyline_points(
            torque_fn_left,
            float(angles_left[0].item()),
            float(angles_left[-1].item()),
        ),
    )
    annotate_curve(
        axis_right,
        sampled_polyline_points(
            torque_fn_right,
            float(angles_right[0].item()),
            float(angles_right[-1].item()),
        ),
    )

    plt.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_airframe_data_with_usd(x_0_1, output_dir, torque_fn_left, torque_fn_right, rigid):
    temp_workspace = tempfile.mkdtemp(prefix="print_design_analysis_")
    script_path = Path(__file__).with_name("convert_morphy_urdf.py")
    output_json_path = Path(output_dir) / "airframe_data.json"

    try:
        command = [
            sys.executable,
            str(script_path),
            "--headless",
            "--output_dir",
            temp_workspace,
            "--params_list",
            *[str(value) for value in x_0_1],
        ]
        if rigid:
            command.append("--rigid")
        else:
            command.extend(["--torque_fn_left", torque_fn_left, "--torque_fn_right", torque_fn_right])

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "USD generation failed.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        temp_json_path = Path(temp_workspace) / "airframe_data.json"
        if not temp_json_path.exists():
            raise RuntimeError(
                "USD generation did not create airframe_data.json.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(temp_json_path, output_json_path)

        with open(output_json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        shutil.rmtree(temp_workspace)


def print_basic_design_info(airframe, torque_response):
    print("Design parameters:")
    for name in PARAM_BOUNDS:
        print(f"  {name}: {airframe.params[name]}")

    print("\nNormalized design vector x_0_1:")
    print(np.array2string(np.asarray(airframe.x_0_1), precision=6, separator=", "))

    print("\nTorque response limits:")
    print(f"  source: {torque_response['source']}")
    print(
        f"  Left : max_left_deg={torque_response['max_left_deg']:.6f}, "
        f"end_angle_b_left={torque_response['end_angle_b_left']:.6f}"
    )
    print(
        f"  Right: max_right_deg={torque_response['max_right_deg']:.6f}, "
        f"end_angle_b_right={torque_response['end_angle_b_right']:.6f}"
    )
    print(f"  torque_fn_left : {torque_response['torque_fn_left']}")
    print(f"  torque_fn_right: {torque_response['torque_fn_right']}")


def print_allocation_data(allocation_data):
    print("\nAllocation data:")
    print(f"  mass: {allocation_data['mass']:.6f} kg")
    print(f"  motor_directions: {allocation_data['motor_directions'].tolist()}")
    print("  motor_positions_body_frame [m]:")
    for motor_index, motor_position in enumerate(allocation_data["motor_positions"]):
        print(f"    Motor {motor_index}: {np.array2string(motor_position, precision=6, separator=', ')}")
    print("  motor_thrust_directions_body_frame:")
    for motor_index, thrust_direction in enumerate(allocation_data["motor_thrust_directions"]):
        print(f"    Motor {motor_index}: {np.array2string(thrust_direction, precision=6, separator=', ')}")
    print(f"  min_forces [N]: {np.array2string(allocation_data['min_forces'], precision=6, separator=', ')}")
    print(f"  max_forces [N]: {np.array2string(allocation_data['max_forces'], precision=6, separator=', ')}")
    print(
        "  thrust_to_torque_ratio [m]: "
        f"{np.array2string(allocation_data['thrust_to_torque_ratio'], precision=6, separator=', ')}"
    )
    print("  allocation_matrix B:")
    print(np.array2string(allocation_data["B"], precision=6, separator=", "))


def print_world_pose_data(airframe_data):
    print("\nAirframe data extracted from USD:")
    print(f"  Total mass: {airframe_data['total_mass']:.6f} kg")
    print(f"  Number of motors: {airframe_data['num_motors']}")
    print("  Motor positions (body frame):")
    for motor_index, motor_position in enumerate(airframe_data["motor_positions_body_frame"]):
        print(
            f"    Motor {motor_index}: "
            f"[{motor_position[0]:.6f}, {motor_position[1]:.6f}, {motor_position[2]:.6f}]"
        )

    print("  Motor poses (world frame):")
    for motor_index in range(airframe_data["num_motors"]):
        world_position = airframe_data["motor_world_positions"][motor_index]
        rotation_matrix = np.asarray(airframe_data["motor_world_rotations"][motor_index], dtype=float)
        print(f"    Motor {motor_index}:")
        print(
            f"      Position: "
            f"[{world_position[0]:.6f}, {world_position[1]:.6f}, {world_position[2]:.6f}]"
        )
        print("      Rotation matrix:")
        print(
            f"        [{rotation_matrix[0][0]:.6f}, {rotation_matrix[0][1]:.6f}, {rotation_matrix[0][2]:.6f}]"
        )
        print(
            f"        [{rotation_matrix[1][0]:.6f}, {rotation_matrix[1][1]:.6f}, {rotation_matrix[1][2]:.6f}]"
        )
        print(
            f"        [{rotation_matrix[2][0]:.6f}, {rotation_matrix[2][1]:.6f}, {rotation_matrix[2][2]:.6f}]"
        )

        cy = math.sqrt(rotation_matrix[0][0] ** 2 + rotation_matrix[0][1] ** 2)
        if cy > 1e-6:
            roll = math.atan2(-rotation_matrix[1][2], rotation_matrix[2][2])
            pitch = math.atan2(rotation_matrix[0][2], cy)
            yaw = math.atan2(-rotation_matrix[0][1], rotation_matrix[0][0])
        else:
            roll = math.atan2(rotation_matrix[2][1], rotation_matrix[1][1])
            pitch = math.atan2(rotation_matrix[0][2], cy)
            yaw = 0.0

        print(
            "      Euler angles (roll, pitch, yaw) [deg]: "
            f"[{math.degrees(roll):.6f}, {math.degrees(pitch):.6f}, {math.degrees(yaw):.6f}]"
        )

        thrust_direction = rotation_matrix[:, 2]
        elevation = math.degrees(math.asin(float(np.clip(thrust_direction[2], -1.0, 1.0))))
        azimuth = math.degrees(math.atan2(float(thrust_direction[1]), float(thrust_direction[0])))
        print(
            "      Thrust direction (elevation, azimuth) [deg]: "
            f"elevation={elevation:.6f}, azimuth={azimuth:.6f}"
        )


def print_cad_frame_analysis(airframe_data):
    arm_attachments = [
        np.array([0.036, 0.0, 0.0]),
        np.array([0.0, 0.0195, 0.0]),
        np.array([-0.036, 0.0, 0.0]),
        np.array([0.0, -0.0195, 0.0]),
    ]
    cad_offset_m = 0.0155

    motor_positions = [np.asarray(position, dtype=float) for position in airframe_data["motor_positions_body_frame"]]
    thrust_directions = [np.asarray(direction, dtype=float) for direction in airframe_data["motor_thrust_directions"]]
    rigid_arms = airframe_data["rigid_arms"]

    flexible_joints_by_motor = {}
    if not rigid_arms:
        flexible_joints_by_motor = {
            1: (
                "joint_1",
                np.asarray(airframe_data["joint_1_position"], dtype=float),
                np.asarray(airframe_data["joint_1_axis"], dtype=float),
            ),
            3: (
                "joint_3",
                np.asarray(airframe_data["joint_3_position"], dtype=float),
                np.asarray(airframe_data["joint_3_axis"], dtype=float),
            ),
        }

    def rotation_z(theta):
        cosine = np.cos(theta)
        sine = np.sin(theta)
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])

    def unit_vector(vector, label):
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError(f"{label} has near-zero length")
        return vector / norm

    def compute_flexible_joint_pose_in_cad(motor_index, cad_origin, rotation_body_from_cad, arm_attach_body, rotation_body_from_attach):
        joint_name, joint_position_body, joint_axis_body = flexible_joints_by_motor[motor_index]
        joint_axis_body = unit_vector(joint_axis_body, f"{joint_name} axis")
        joint_position_cad = rotation_body_from_cad.T @ (joint_position_body - cad_origin)
        joint_axis_cad = unit_vector(
            rotation_body_from_cad.T @ joint_axis_body,
            f"{joint_name} axis in CAD frame",
        )

        joint_x_body = motor_positions[motor_index] - joint_position_body
        joint_x_body = joint_x_body - np.dot(joint_x_body, joint_axis_body) * joint_axis_body
        joint_x_body = unit_vector(joint_x_body, f"{joint_name} motor-side direction")
        joint_y_body = unit_vector(np.cross(joint_axis_body, joint_x_body), f"{joint_name} y axis")
        joint_x_body = unit_vector(np.cross(joint_y_body, joint_axis_body), f"{joint_name} x axis")

        rotation_body_from_joint = np.column_stack([joint_x_body, joint_y_body, joint_axis_body])
        rotation_cad_from_joint = rotation_body_from_cad.T @ rotation_body_from_joint

        joint_position_attach = rotation_body_from_attach.T @ (joint_position_body - arm_attach_body)
        rotation_attach_from_joint = rotation_body_from_attach.T @ rotation_body_from_joint

        return {
            "name": joint_name,
            "position_cad": joint_position_cad,
            "axis_cad": joint_axis_cad,
            "rotation_cad": rotation_cad_from_joint,
            "position_attach": joint_position_attach,
            "rotation_attach": rotation_attach_from_joint,
        }

    arm_orientations_body = [
        rotation_z(0.0),
        rotation_z(np.pi / 2.0),
        rotation_z(np.pi),
        rotation_z(-np.pi / 2.0),
    ]

    print("\nArm attachment pose in CAD motor frames:")
    if rigid_arms:
        print("  Rigid airframe: no flexible joints to express in CAD motor frames.")

    for motor_index in range(4):
        z_cad = thrust_directions[motor_index] / np.linalg.norm(thrust_directions[motor_index])
        cad_origin = motor_positions[motor_index] - cad_offset_m * z_cad
        arm_attach = arm_attachments[motor_index]
        delta_x = cad_origin - arm_attach
        y_cad = np.cross(z_cad, delta_x)
        y_cad = y_cad / np.linalg.norm(y_cad)
        x_cad = np.cross(y_cad, z_cad)
        x_cad = x_cad / np.linalg.norm(x_cad)

        rotation_body_from_cad = np.column_stack([x_cad, y_cad, z_cad])
        attach_cad = rotation_body_from_cad.T @ (arm_attach - cad_origin)
        rotation_attach_cad = rotation_body_from_cad.T @ arm_orientations_body[motor_index]

        direction = rotation_attach_cad[:, 0]
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        elevation = np.degrees(np.arctan2(direction[2], np.sqrt(direction[0] ** 2 + direction[1] ** 2)))
        cosine_azimuth = np.cos(np.radians(azimuth))
        sine_azimuth = np.sin(np.radians(azimuth))
        cosine_elevation = np.cos(np.radians(elevation))
        sine_elevation = np.sin(np.radians(elevation))
        rotation_0 = np.array(
            [
                [cosine_azimuth * cosine_elevation, -sine_azimuth, -cosine_azimuth * sine_elevation],
                [sine_azimuth * cosine_elevation, cosine_azimuth, -sine_azimuth * sine_elevation],
                [sine_elevation, 0.0, cosine_elevation],
            ]
        )
        rotation_relative = rotation_0.T @ rotation_attach_cad
        twist = np.degrees(np.arctan2(rotation_relative[2, 1], rotation_relative[1, 1]))

        print(f"  Motor {motor_index}:")
        print(
            f"    Position (CAD frame) [m]: "
            f"[{attach_cad[0]:.6f}, {attach_cad[1]:.6f}, {attach_cad[2]:.6f}]"
        )
        print("    Orientation (CAD frame) rotation matrix:")
        print(
            f"      [{rotation_attach_cad[0, 0]:.6f}, {rotation_attach_cad[0, 1]:.6f}, {rotation_attach_cad[0, 2]:.6f}]"
        )
        print(
            f"      [{rotation_attach_cad[1, 0]:.6f}, {rotation_attach_cad[1, 1]:.6f}, {rotation_attach_cad[1, 2]:.6f}]"
        )
        print(
            f"      [{rotation_attach_cad[2, 0]:.6f}, {rotation_attach_cad[2, 1]:.6f}, {rotation_attach_cad[2, 2]:.6f}]"
        )
        print(
            "    Orientation (CAD frame): "
            f"azimuth={azimuth:.6f} deg, elevation={elevation:.6f} deg, twist={twist:.6f} deg"
        )

        if motor_index in flexible_joints_by_motor:
            flexible_joint = compute_flexible_joint_pose_in_cad(
                motor_index,
                cad_origin,
                rotation_body_from_cad,
                arm_attach,
                arm_orientations_body[motor_index],
            )
            joint_position_cad = flexible_joint["position_cad"]
            joint_axis_cad = flexible_joint["axis_cad"]
            joint_rotation_cad = flexible_joint["rotation_cad"]
            joint_position_attach = flexible_joint["position_attach"]
            joint_rotation_attach = flexible_joint["rotation_attach"]

            print(f"    Flexible joint {flexible_joint['name']}:")
            print(
                f"      Position (CAD frame) [m]: "
                f"[{joint_position_cad[0]:.6f}, {joint_position_cad[1]:.6f}, {joint_position_cad[2]:.6f}]"
            )
            print(
                f"      Axis (CAD frame): "
                f"[{joint_axis_cad[0]:.6f}, {joint_axis_cad[1]:.6f}, {joint_axis_cad[2]:.6f}]"
            )
            print("      Pose (CAD frame) rotation matrix:")
            print(
                f"        [{joint_rotation_cad[0, 0]:.6f}, {joint_rotation_cad[0, 1]:.6f}, {joint_rotation_cad[0, 2]:.6f}]"
            )
            print(
                f"        [{joint_rotation_cad[1, 0]:.6f}, {joint_rotation_cad[1, 1]:.6f}, {joint_rotation_cad[1, 2]:.6f}]"
            )
            print(
                f"        [{joint_rotation_cad[2, 0]:.6f}, {joint_rotation_cad[2, 1]:.6f}, {joint_rotation_cad[2, 2]:.6f}]"
            )
            print(
                f"      Position (attach frame) [m]: "
                f"[{joint_position_attach[0]:.6f}, {joint_position_attach[1]:.6f}, {joint_position_attach[2]:.6f}]"
            )
            print("      Pose (attach frame) rotation matrix:")
            print(
                f"        [{joint_rotation_attach[0, 0]:.6f}, {joint_rotation_attach[0, 1]:.6f}, {joint_rotation_attach[0, 2]:.6f}]"
            )
            print(
                f"        [{joint_rotation_attach[1, 0]:.6f}, {joint_rotation_attach[1, 1]:.6f}, {joint_rotation_attach[1, 2]:.6f}]"
            )
            print(
                f"        [{joint_rotation_attach[2, 0]:.6f}, {joint_rotation_attach[2, 1]:.6f}, {joint_rotation_attach[2, 2]:.6f}]"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Print Morphy motor/design analysis and save torque_response.jpeg."
    )
    parser.add_argument(
        "--best_solution_json",
        type=str,
        help="Path to best_solution.json containing geometry_x_0_1 and torque_fn_left/right",
    )
    parser.add_argument(
        "--params_list",
        type=float,
        nargs="*",
        help=f"List of {len(PARAM_BOUNDS)} normalized parameters in [0, 1]",
    )
    for parameter_name in PARAM_BOUNDS:
        parser.add_argument(f"--{parameter_name}", type=float, help=parameter_name)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where airframe_data.json and torque_response.jpeg are written",
    )
    parser.add_argument(
        "--rigid",
        type=str,
        choices=["true", "false"],
        required=True,
        help="Whether to generate a rigid airframe: true or false",
    )
    args_cli = parser.parse_args()

    if args_cli.best_solution_json is not None and args_cli.params_list is not None:
        raise ValueError("Provide either --best_solution_json or --params_list, not both.")
    if args_cli.best_solution_json is not None:
        for parameter_name in PARAM_BOUNDS:
            if getattr(args_cli, parameter_name) is not None:
                raise ValueError("Do not mix --best_solution_json with individual geometry parameters.")

    airframe, torque_response = build_design_bundle_from_args(args_cli)
    allocation_data = airframe.compute_allocation_matrix()
    rigid = args_cli.rigid == "true"

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    airframe_data = generate_airframe_data_with_usd(
        airframe.x_0_1.tolist(),
        output_dir,
        torque_response["torque_fn_left"],
        torque_response["torque_fn_right"],
        rigid,
    )

    torque_plot_path = os.path.join(output_dir, "torque_response.jpeg")
    save_torque_response_plot(
        torque_response,
        airframe_data,
        torque_response["torque_fn_left"],
        torque_response["torque_fn_right"],
        torque_plot_path,
    )

    print_basic_design_info(airframe, torque_response)
    print_allocation_data(allocation_data)
    print_world_pose_data(airframe_data)
    print_cad_frame_analysis(airframe_data)
    print(f"\nSaved airframe data to {os.path.join(output_dir, 'airframe_data.json')}")
    print(f"Saved torque response plot to {torque_plot_path}")


if __name__ == "__main__":
    main()
