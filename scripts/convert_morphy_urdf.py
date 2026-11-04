#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert Morphy URDF to USD format with parametric geometry.

Usage - Parameter list plus torque response functions:
python scripts/convert_morphy_urdf.py --headless --output_dir cache/workspace_local/my_airframe --torque_fn_left "amplitude_left=1.0;torque_fn_left=lambda x: -amplitude_left*torch.sin(x)" --torque_fn_right "amplitude_right=1.0;torque_fn_right=lambda x: -amplitude_right*torch.sin(x)" --params_list 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5

Optional motor spin top view:
python scripts/convert_morphy_urdf.py --headless --output_dir cache/workspace_local/my_airframe --top_view_motor_spins --torque_fn_left "amplitude_left=1.0;torque_fn_left=lambda x: -amplitude_left*torch.sin(x)" --torque_fn_right "amplitude_right=1.0;torque_fn_right=lambda x: -amplitude_right*torch.sin(x)" --params_list 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airframe_encoding import PARAM_BOUNDS, AIRFRAME_CONSTANTS, Airframe, ADD_ROD



def generate_airframe_json(x: list, output_json_path: str, torque_fn_left: str, torque_fn_right: str):
    """Generate airframe properties JSON from parameter vector by creating USD and extracting data.

    Args:
        x: Full parameter vector (normalized 0-1 or actual values)
        output_json_path: Path where to save the JSON file
    """
    import json
    import os
    import tempfile
    import shutil
    import subprocess
    import sys

    if all(0 <= val <= 1 for val in x):
        x_0_1 = list(x)
    else:
        x_0_1 = Airframe.from_params_dict(dict(zip(PARAM_BOUNDS.keys(), x))).x_0_1.tolist()

    temp_workspace = tempfile.mkdtemp(prefix="airframe_validation_")

    try:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "convert_morphy_urdf.py"),
            "--headless",
            "--output_dir", temp_workspace,
            "--torque_fn_left", torque_fn_left,
            "--torque_fn_right", torque_fn_right,
            "--params_list"
        ] + [str(val) for val in x_0_1]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"USD generation failed: {result.stderr}")

        temp_json = os.path.join(temp_workspace, "airframe_data.json")
        if not os.path.exists(temp_json):
            raise RuntimeError(f"airframe_data.json not found in {temp_workspace}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        shutil.copy(temp_json, output_json_path)

        with open(output_json_path, 'r') as f:
            airframe_data = json.load(f)

        return airframe_data
    finally:
        shutil.rmtree(temp_workspace, ignore_errors=True)


def save_top_view_motor_spins_plot(airframe_data: dict, output_path: str):
    """Save a top-view plot of generated motor positions and simulation spin directions."""
    import sys
    import numpy as np
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motor_translations = np.asarray(airframe_data["motor_positions_body_frame"], dtype=float)
    motor_directions = np.asarray(AIRFRAME_CONSTANTS["motor_directions"], dtype=int)

    if motor_translations.ndim != 2 or motor_translations.shape[1] != 3:
        raise ValueError(f"Expected motor positions with shape (N, 3), got {motor_translations.shape}.")
    if motor_translations.shape[0] != 4:
        raise ValueError(f"Morphy simulation expects 4 motors, got {motor_translations.shape[0]}.")
    if motor_translations.shape[0] != motor_directions.shape[0]:
        raise ValueError(
            f"Expected the same number of motor positions and directions, got "
            f"{motor_translations.shape[0]} and {motor_directions.shape[0]}."
        )
    if not np.all((motor_directions == 1) | (motor_directions == -1)):
        raise ValueError(f"Simulation motor directions must be either +1 or -1, got {motor_directions.tolist()}.")

    motor_x = motor_translations[:, 0]
    motor_y = motor_translations[:, 1]

    horizontal_axis = motor_y
    vertical_axis = motor_x

    positive_mask = motor_directions == 1
    negative_mask = motor_directions == -1

    spin_labels = {
        1: "↻ CW",
        -1: "↺ CCW",
    }

    figure, axis = plt.subplots(figsize=(7, 7))

    for horizontal_value, vertical_value in zip(horizontal_axis, vertical_axis):
        axis.plot([0.0, horizontal_value], [0.0, vertical_value], color="0.80", linewidth=1.2, zorder=1)

    closed_order = [0, 1, 2, 3, 0]
    axis.plot(
        horizontal_axis[closed_order],
        vertical_axis[closed_order],
        color="0.75",
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )

    axis.scatter(
        horizontal_axis[positive_mask],
        vertical_axis[positive_mask],
        s=500,
        c="#1f77b4",
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )
    axis.scatter(
        horizontal_axis[negative_mask],
        vertical_axis[negative_mask],
        s=500,
        c="#ff7f0e",
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )

    for motor_index, (horizontal_value, vertical_value, direction_value) in enumerate(
        zip(horizontal_axis, vertical_axis, motor_directions)
    ):
        horizontal_offset = -30 if horizontal_value > 0.0 else 12
        vertical_offset = 18 if vertical_value > 0.0 else -24
        axis.annotate(
            f"{spin_labels[direction_value]} {motor_index}",
            xy=(horizontal_value, vertical_value),
            xytext=(horizontal_offset, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "linewidth": 1.0,
                "shrinkA": 4,
                "shrinkB": 8,
            },
            zorder=4,
        )

    axis.scatter([0.0], [0.0], s=80, c="black", marker="+", linewidths=2.0, zorder=4)
    axis.text(0.0, 0.0, " center", ha="left", va="bottom", fontsize=10, color="black")

    axis.set_title("Generated Morphy: motor spin directions")
    axis.set_xlabel("y axis, horizontal, positive to the left")
    axis.set_ylabel("x axis, vertical, positive upward")
    axis.grid(True, linestyle=":", linewidth=0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.invert_xaxis()

    max_extent = float(np.max(np.abs(np.concatenate([horizontal_axis, vertical_axis]))))
    padding = 0.06
    axis.set_xlim(max_extent + padding, -max_extent - padding)
    axis.set_ylim(-max_extent - padding, max_extent + padding)

    plt.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved top-view motor spin plot to {Path(output_path).resolve()}")


if __name__ == "__main__":
    # Create argument parser
    import argparse
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Convert Morphy URDF to USD with parametric geometry.")
    parser.add_argument("--params_list", type=float, nargs='*', help=f"List of {len(PARAM_BOUNDS)} normalized parameters [0-1] mapped to parameter ranges")
    parser.add_argument("--front_arm_length", type=float, help="Front arm length")
    parser.add_argument("--front_prop_rotation_x", type=float, help="Front prop rotation around x-axis")
    parser.add_argument("--front_prop_rotation_y", type=float, help="Front prop rotation around y-axis")
    parser.add_argument("--front_prop_rotation_z", type=float, help="Front prop rotation around z-axis")
    parser.add_argument("--back_arm_length", type=float, help="Back arm length")
    parser.add_argument("--back_prop_rotation_x", type=float, help="Back prop rotation around x-axis")
    parser.add_argument("--back_prop_rotation_y", type=float, help="Back prop rotation around y-axis")
    parser.add_argument("--back_prop_rotation_z", type=float, help="Back prop rotation around z-axis")
    parser.add_argument("--side_arm_seg2_length_left", type=float, help="Left side arm second segment length")
    parser.add_argument("--side_arm_seg2_length_right", type=float, help="Right side arm second segment length")
    parser.add_argument("--side_base_rotation_x_left", type=float, help="Left side arm base rotation around x-axis")
    parser.add_argument("--side_base_rotation_y_left", type=float, help="Left side arm base rotation around y-axis")
    parser.add_argument("--side_base_rotation_z_left", type=float, help="Left side arm base rotation around z-axis")
    parser.add_argument("--side_base_rotation_x_right", type=float, help="Right side arm base rotation around x-axis")
    parser.add_argument("--side_base_rotation_y_right", type=float, help="Right side arm base rotation around y-axis")
    parser.add_argument("--side_base_rotation_z_right", type=float, help="Right side arm base rotation around z-axis")
    parser.add_argument("--side_motor_rotation_x_left", type=float, help="Left side motor rotation around x-axis")
    parser.add_argument("--side_motor_rotation_y_left", type=float, help="Left side motor rotation around y-axis")
    parser.add_argument("--side_motor_rotation_z_left", type=float, help="Left side motor rotation around z-axis")
    parser.add_argument("--side_motor_rotation_x_right", type=float, help="Right side motor rotation around x-axis")
    parser.add_argument("--side_motor_rotation_y_right", type=float, help="Right side motor rotation around y-axis")
    parser.add_argument("--side_motor_rotation_z_right", type=float, help="Right side motor rotation around z-axis")
    parser.add_argument("--torque_fn_left", type=str, help="Left arm torque response code string evaluated at runtime; it must define torque_fn_left. Required for soft airframes, forbidden with --rigid")
    parser.add_argument("--torque_fn_right", type=str, help="Right arm torque response code string evaluated at runtime; it must define torque_fn_right. Required for soft airframes, forbidden with --rigid")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for USD files")
    parser.add_argument("--rigid", action="store_true", help="Make side arm joints fixed instead of revolute (rigid airframe)")
    parser.add_argument("--cad_poses", action="store_true", help="Print CAD pose diagnostics and open CAD-frame plots")
    parser.add_argument("--top_view_motor_spins", action="store_const", const=True, help="Save a top-view PNG of motor spin directions using generated geometry")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.rigid:
        if args_cli.torque_fn_left is not None or args_cli.torque_fn_right is not None:
            parser.error("--torque_fn_left/--torque_fn_right must not be given with --rigid")
    elif args_cli.torque_fn_left is None or args_cli.torque_fn_right is None:
        parser.error("--torque_fn_left and --torque_fn_right are required for soft airframes")

    if args_cli.params_list is not None:
        if len(args_cli.params_list) != len(PARAM_BOUNDS):
            parser.error(f"argument --params_list: expected {len(PARAM_BOUNDS)} arguments, got {len(args_cli.params_list)}")
        airframe = Airframe.from_x_0_1(args_cli.params_list)
        for name, val in airframe.params.items():
            setattr(args_cli, name, val)
    else:
        missing = [p for p in PARAM_BOUNDS if getattr(args_cli, p) is None]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}. Use --params_list or provide all individual parameters.")

    x_0_1 = Airframe.from_params_dict({p: getattr(args_cli, p) for p in PARAM_BOUNDS}).x_0_1.tolist()
    print(f"[Convert URDF to USD] x_0_1 = {x_0_1}")

    # Launch the app
    app_launcher = AppLauncher(args_cli, multi_gpu=False)
    simulation_app = app_launcher.app

    """Rest everything follows."""

    import os
    import math
    from pxr import UsdPhysics
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg, spawn_rigid_body_material
    import isaacsim.core.utils.prims as prim_utils

    def calculate_box_inertia(mass, length, width, height, com_x, com_y, com_z):
        """Calculate inertia tensor for a box with COM offset."""
        Ixx_cm = mass * (width**2 + height**2) / 12
        Iyy_cm = mass * (length**2 + height**2) / 12
        Izz_cm = mass * (length**2 + width**2) / 12
        Ixx = Ixx_cm + mass * (com_y**2 + com_z**2)
        Iyy = Iyy_cm + mass * (com_x**2 + com_z**2)
        Izz = Izz_cm + mass * (com_x**2 + com_y**2)
        return Ixx, Iyy, Izz

    def generate_parametric_urdf(output_path, params, rigid):
        """Generate parametric URDF based on input parameters."""
        import numpy as np

        front_arm_len = params['front_arm_length']
        front_prop_rot_x = math.radians(params['front_prop_rotation_x'])
        front_prop_rot_y = math.radians(params['front_prop_rotation_y'])
        front_prop_rot_z = math.radians(params['front_prop_rotation_z'])
        back_arm_len = params['back_arm_length']
        back_prop_rot_x = math.radians(params['back_prop_rotation_x'])
        back_prop_rot_y = math.radians(params['back_prop_rotation_y'])
        back_prop_rot_z = math.radians(params['back_prop_rotation_z'])
        side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
        side_seg2_len_left = params['side_arm_seg2_length_left']
        side_seg2_len_right = params['side_arm_seg2_length_right']
        side_base_rot_x_left = math.radians(params['side_base_rotation_x_left'])
        side_base_rot_y_left = math.radians(params['side_base_rotation_y_left'])
        side_base_rot_z_left = math.radians(params['side_base_rotation_z_left'])
        side_base_rot_x_right = math.radians(params['side_base_rotation_x_right'])
        side_base_rot_y_right = math.radians(params['side_base_rotation_y_right'])
        side_base_rot_z_right = math.radians(params['side_base_rotation_z_right'])
        side_motor_rot_x_left = math.radians(params['side_motor_rotation_x_left'])
        side_motor_rot_y_left = math.radians(params['side_motor_rotation_y_left'])
        side_motor_rot_z_left = math.radians(params['side_motor_rotation_z_left'])
        side_motor_rot_x_right = math.radians(params['side_motor_rotation_x_right'])
        side_motor_rot_y_right = math.radians(params['side_motor_rotation_y_right'])
        side_motor_rot_z_right = math.radians(params['side_motor_rotation_z_right'])

        arm_width = AIRFRAME_CONSTANTS['arm_width']
        arm_height = AIRFRAME_CONSTANTS['arm_height']
        arm_density = AIRFRAME_CONSTANTS['arm_density']
        motor_mass = AIRFRAME_CONSTANTS['motor_mass']
        motor_radius = AIRFRAME_CONSTANTS['motor_radius']
        motor_height = AIRFRAME_CONSTANTS['motor_height']
        base_mass = AIRFRAME_CONSTANTS['base_mass']
        base_box_dimensions = AIRFRAME_CONSTANTS['base_box_dimensions']

        front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
        back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
        side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
        side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
        side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density

        front_arm_com = front_arm_len / 2
        back_arm_com = back_arm_len / 2
        side_seg1_com = side_seg1_len / 2
        side_seg2_com_left = side_seg2_len_left / 2
        side_seg2_com_right = side_seg2_len_right / 2

        def build_rotation_matrix(roll, pitch, yaw):
            """Build rotation matrix from RPY angles."""
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            return Rz @ Ry @ Rx

        def compute_rod_from_arm_geometry(arm_base_rot_x, arm_base_rot_y, arm_base_rot_z,
                                          motor_rot_x, motor_rot_y, motor_rot_z, seg2_len):
            """Compute rod parameters connecting motor edge (highest world-x) to the revolute joint.

            The rod is in motor frame and connects:
            - Motor edge: the point on motor cylinder with highest world-x
            - Joint: arm_segment_2 origin (revolute joint location)
            """
            R_arm = build_rotation_matrix(arm_base_rot_x, arm_base_rot_y, arm_base_rot_z)
            R_motor = build_rotation_matrix(motor_rot_x, motor_rot_y, motor_rot_z)
            R_combined = R_arm @ R_motor

            world_x_in_motor = R_combined.T @ np.array([1, 0, 0])
            xy_norm = np.linalg.norm(world_x_in_motor[:2])
            edge_dir = np.array([world_x_in_motor[0] / xy_norm, world_x_in_motor[1] / xy_norm, 0]) if xy_norm > 1e-6 else np.array([1, 0, 0])

            edge_point_motor = motor_radius * edge_dir
            joint_in_motor = R_motor.T @ np.array([-seg2_len, 0, 0])

            rod_vector = edge_point_motor - joint_in_motor
            rod_length = float(np.linalg.norm(rod_vector))
            rod_center = (edge_point_motor + joint_in_motor) / 2

            rod_direction = rod_vector / rod_length
            rod_xy_len = math.sqrt(rod_direction[0]**2 + rod_direction[1]**2)
            rod_pitch = 1.5707963268 - math.atan2(rod_direction[2], rod_xy_len)
            rod_yaw = math.atan2(rod_direction[1], rod_direction[0])

            return rod_length, rod_center, rod_pitch, rod_yaw

        rod_length_left, rod_com_left, rod_pitch_left, rod_yaw_left = compute_rod_from_arm_geometry(
            side_base_rot_x_left, side_base_rot_y_left, side_base_rot_z_left + 1.5707963268,
            side_motor_rot_x_left, side_motor_rot_y_left, side_motor_rot_z_left, side_seg2_len_left)
        rod_length_right, rod_com_right, rod_pitch_right, rod_yaw_right = compute_rod_from_arm_geometry(
            side_base_rot_x_right, side_base_rot_y_right, side_base_rot_z_right - 1.5707963268,
            side_motor_rot_x_right, side_motor_rot_y_right, side_motor_rot_z_right, side_seg2_len_right)

        front_arm_ixx, front_arm_iyy, front_arm_izz = calculate_box_inertia(
            front_arm_mass, front_arm_len, arm_width, arm_height, front_arm_com, 0, 0)
        back_arm_ixx, back_arm_iyy, back_arm_izz = calculate_box_inertia(
            back_arm_mass, back_arm_len, arm_width, arm_height, back_arm_com, 0, 0)
        side_seg1_ixx, side_seg1_iyy, side_seg1_izz = calculate_box_inertia(
            side_seg1_mass, side_seg1_len, arm_width, arm_height, side_seg1_com, 0, 0)
        side_seg2_ixx_left, side_seg2_iyy_left, side_seg2_izz_left = calculate_box_inertia(
            side_seg2_mass_left, side_seg2_len_left, arm_width, arm_height, side_seg2_com_left, 0, 0)
        side_seg2_ixx_right, side_seg2_iyy_right, side_seg2_izz_right = calculate_box_inertia(
            side_seg2_mass_right, side_seg2_len_right, arm_width, arm_height, side_seg2_com_right, 0, 0)

        urdf_content = f"""<?xml version="1.0"?>
    <robot name="quadrotor">
      <link name="base_link">
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{base_box_dimensions[0]} {base_box_dimensions[1]} {base_box_dimensions[2]}"/>
          </geometry>
          <material name="White">
            <color rgba="1 1 1 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{base_box_dimensions[0]} {base_box_dimensions[1]} {base_box_dimensions[2]}"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="{base_mass}"/>
          <inertia ixx="0.00042249999999999997" ixy="0.0" ixz="0.0" iyx="0.0" iyy="0.00042249999999999997" iyz="0.0" izx="0.0" izy="0.0" izz="0.0008449999999999999"/>
        </inertial>
      </link>

      <!-- Front Arm (idx 0) - Fixed -->
      <link name="arm_segment_2_0">
        <visual>
          <origin xyz="{front_arm_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{front_arm_len} {arm_width} {arm_height}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="{front_arm_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{front_arm_len} {arm_width} {arm_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="{front_arm_com} 0 0" rpy="0 0 0"/>
          <mass value="{front_arm_mass}"/>
          <inertia ixx="{front_arm_ixx}" ixy="0.0" ixz="0.0" iyy="{front_arm_iyy}" iyz="0.0" izz="{front_arm_izz}"/>
        </inertial>
      </link>
      <joint name="arm_segment_1_to_arm_segment_2_0" type="fixed" dont_collapse="true">
        <parent link="base_link"/>
        <child link="arm_segment_2_0"/>
        <origin xyz="0.035 0 0" rpy="0.0 0.0 0.0"/>
      </joint>
      <link name="motor_0">
        <visual>
          <origin xyz="0 0 {0}" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
          <material name="PastelBlue">
            <color rgba="0.68 0.85 0.9 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="0 0 {0}" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="{motor_mass}"/>
          <inertia ixx="1.0e-08" ixy="0.0" ixz="0.0" iyy="1.0e-08" iyz="0.0" izz="1.0e-08"/>
        </inertial>
      </link>
      <joint name="arm_segment_2_to_motor_0" type="fixed" dont_collapse="true">
        <parent link="arm_segment_2_0"/>
        <child link="motor_0"/>
        <origin xyz="{front_arm_len} 0 0" rpy="{front_prop_rot_x} {front_prop_rot_y} {front_prop_rot_z}"/>
      </joint>

      <!-- Side Arm 1 (idx 1) - Revolute -->
      <link name="arm_segment_1_1">
        <visual>
          <origin xyz="{side_seg1_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg1_len} {arm_width} {arm_height}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="{side_seg1_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg1_len} {arm_width} {arm_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="{side_seg1_com} 0 0" rpy="0 0 0"/>
          <mass value="{side_seg1_mass}"/>
          <inertia ixx="{side_seg1_ixx}" ixy="0.0" ixz="0.0" iyy="{side_seg1_iyy}" iyz="0.0" izz="{side_seg1_izz}"/>
        </inertial>
      </link>
      <joint name="base_link_to_arm_segment_1_1" type="fixed" dont_collapse="true">
        <parent link="base_link"/>
        <child link="arm_segment_1_1"/>
        <origin xyz="0 0.025 0" rpy="{side_base_rot_x_left} {side_base_rot_y_left} {side_base_rot_z_left + 1.5707963268}"/>
      </joint>
      <link name="arm_segment_2_1">
        <visual>
          <origin xyz="{side_seg2_com_left} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg2_len_left} {arm_width} {arm_height}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="{side_seg2_com_left} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg2_len_left} {arm_width} {arm_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="{side_seg2_com_left} 0 0" rpy="0 0 0"/>
          <mass value="{side_seg2_mass_left}"/>
          <inertia ixx="{side_seg2_ixx_left}" ixy="0.0" ixz="0.0" iyy="{side_seg2_iyy_left}" iyz="0.0" izz="{side_seg2_izz_left}"/>
        </inertial>
      </link>
      <joint name="arm_segment_1_to_arm_segment_2_1" type="{'fixed' if rigid else 'revolute'}" dont_collapse="{'true' if rigid else 'false'}">
        <parent link="arm_segment_1_1"/>
        <child link="arm_segment_2_1"/>
        <origin xyz="{side_seg1_len} 0 0" rpy="0.0 0.0 0.0"/>
        {'<axis xyz="0 0 1"/>' if not rigid else ''}
      </joint>
      <link name="motor_1">
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
          <material name="PastelYellow">
            <color rgba="1.0 1.0 0.7 1.0"/>
          </material>
        </visual>
        {f'''<visual>
          <origin xyz="{rod_com_left[0]} {rod_com_left[1]} {rod_com_left[2]}" rpy="0 {rod_pitch_left} {rod_yaw_left}"/>
          <geometry>
            <capsule radius="{AIRFRAME_CONSTANTS["rod_radius"]}" length="{rod_length_left}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>''' if ADD_ROD else ''}
        <collision>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
        </collision>
        {f'''<collision>
          <origin xyz="{rod_com_left[0]} {rod_com_left[1]} {rod_com_left[2]}" rpy="0 {rod_pitch_left} {rod_yaw_left}"/>
          <geometry>
            <capsule radius="{AIRFRAME_CONSTANTS["rod_radius"]}" length="{rod_length_left}"/>
          </geometry>
        </collision>''' if ADD_ROD else ''}
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="{motor_mass}"/>
          <inertia ixx="1.0e-08" ixy="0.0" ixz="0.0" iyy="1.0e-08" iyz="0.0" izz="1.0e-08"/>
        </inertial>
      </link>
      <joint name="arm_segment_2_to_motor_1" type="fixed" dont_collapse="true">
        <parent link="arm_segment_2_1"/>
        <child link="motor_1"/>
        <origin xyz="{side_seg2_len_left} 0 0" rpy="{side_motor_rot_x_left} {side_motor_rot_y_left} {side_motor_rot_z_left}"/>
      </joint>

      <!-- Back Arm (idx 2) - Fixed -->
      <link name="arm_segment_2_2">
        <visual>
          <origin xyz="{back_arm_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{back_arm_len} {arm_width} {arm_height}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="{back_arm_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{back_arm_len} {arm_width} {arm_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="{back_arm_com} 0 0" rpy="0 0 0"/>
          <mass value="{back_arm_mass}"/>
          <inertia ixx="{back_arm_ixx}" ixy="0.0" ixz="0.0" iyy="{back_arm_iyy}" iyz="0.0" izz="{back_arm_izz}"/>
        </inertial>
      </link>
      <joint name="arm_segment_1_to_arm_segment_2_2" type="fixed" dont_collapse="true">
        <parent link="base_link"/>
        <child link="arm_segment_2_2"/>
        <origin xyz="-0.035 0 0" rpy="0.0 0.0 3.14159265359"/>
      </joint>
      <link name="motor_2">
        <visual>
          <origin xyz="0 0 {0}" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
          <material name="PastelBlue">
            <color rgba="0.68 0.85 0.9 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="0 0 {0}" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="{motor_mass}"/>
          <inertia ixx="1.0e-08" ixy="0.0" ixz="0.0" iyy="1.0e-08" iyz="0.0" izz="1.0e-08"/>
        </inertial>
      </link>
      <joint name="arm_segment_2_to_motor_2" type="fixed" dont_collapse="true">
        <parent link="arm_segment_2_2"/>
        <child link="motor_2"/>
        <origin xyz="{back_arm_len} 0 0" rpy="{back_prop_rot_x} {back_prop_rot_y} {back_prop_rot_z}"/>
      </joint>

      <!-- Side Arm 3 (idx 3) - Revolute -->
      <link name="arm_segment_1_3">
        <visual>
          <origin xyz="{side_seg1_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg1_len} {arm_width} {arm_height}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="{side_seg1_com} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg1_len} {arm_width} {arm_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="{side_seg1_com} 0 0" rpy="0 0 0"/>
          <mass value="{side_seg1_mass}"/>
          <inertia ixx="{side_seg1_ixx}" ixy="0.0" ixz="0.0" iyy="{side_seg1_iyy}" iyz="0.0" izz="{side_seg1_izz}"/>
        </inertial>
      </link>
      <joint name="base_link_to_arm_segment_1_3" type="fixed" dont_collapse="true">
        <parent link="base_link"/>
        <child link="arm_segment_1_3"/>
        <origin xyz="0 -0.025 0" rpy="{side_base_rot_x_right} {side_base_rot_y_right} {side_base_rot_z_right - 1.5707963268}"/>
      </joint>
      <link name="arm_segment_2_3">
        <visual>
          <origin xyz="{side_seg2_com_right} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg2_len_right} {arm_width} {arm_height}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>
        <collision>
          <origin xyz="{side_seg2_com_right} 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="{side_seg2_len_right} {arm_width} {arm_height}"/>
          </geometry>
        </collision>
        <inertial>
          <origin xyz="{side_seg2_com_right} 0 0" rpy="0 0 0"/>
          <mass value="{side_seg2_mass_right}"/>
          <inertia ixx="{side_seg2_ixx_right}" ixy="0.0" ixz="0.0" iyy="{side_seg2_iyy_right}" iyz="0.0" izz="{side_seg2_izz_right}"/>
        </inertial>
      </link>
      <joint name="arm_segment_1_to_arm_segment_2_3" type="{'fixed' if rigid else 'revolute'}" dont_collapse="{'true' if rigid else 'false'}">
        <parent link="arm_segment_1_3"/>
        <child link="arm_segment_2_3"/>
        <origin xyz="{side_seg1_len} 0 0" rpy="0.0 0.0 0.0"/>
        {'<axis xyz="0 0 1"/>' if not rigid else ''}
      </joint>
      <link name="motor_3">
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
          <material name="PastelYellow">
            <color rgba="1.0 1.0 0.7 1.0"/>
          </material>
        </visual>
        {f'''<visual>
          <origin xyz="{rod_com_right[0]} {rod_com_right[1]} {rod_com_right[2]}" rpy="0 {rod_pitch_right} {rod_yaw_right}"/>
          <geometry>
            <capsule radius="{AIRFRAME_CONSTANTS["rod_radius"]}" length="{rod_length_right}"/>
          </geometry>
          <material name="Orange">
            <color rgba="1 0.5 0 1.0"/>
          </material>
        </visual>''' if ADD_ROD else ''}
        <collision>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="{motor_radius}" length="{motor_height}"/>
          </geometry>
        </collision>
        {f'''<collision>
          <origin xyz="{rod_com_right[0]} {rod_com_right[1]} {rod_com_right[2]}" rpy="0 {rod_pitch_right} {rod_yaw_right}"/>
          <geometry>
            <capsule radius="{AIRFRAME_CONSTANTS["rod_radius"]}" length="{rod_length_right}"/>
          </geometry>
        </collision>''' if ADD_ROD else ''}
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="{motor_mass}"/>
          <inertia ixx="1.0e-08" ixy="0.0" ixz="0.0" iyy="1.0e-08" iyz="0.0" izz="1.0e-08"/>
        </inertial>
      </link>
      <joint name="arm_segment_2_to_motor_3" type="fixed" dont_collapse="true">
        <parent link="arm_segment_2_3"/>
        <child link="motor_3"/>
        <origin xyz="{side_seg2_len_right} 0 0" rpy="{side_motor_rot_x_right} {side_motor_rot_y_right} {side_motor_rot_z_right}"/>
      </joint>
    </robot>
    """

        with open(output_path, 'w') as f:
            f.write(urdf_content)

        print(f"Generated parametric URDF with:")
        print(f"  Front arm: {front_arm_len}m, rotation X: {params['front_prop_rotation_x']}°, Y: {params['front_prop_rotation_y']}°, Z: {params['front_prop_rotation_z']}°")
        print(f"  Back arm: {back_arm_len}m, rotation X: {params['back_prop_rotation_x']}°, Y: {params['back_prop_rotation_y']}°, Z: {params['back_prop_rotation_z']}°")
        print(f"  Side arm seg1: {side_seg1_len}m (hardcoded), seg2 left: {side_seg2_len_left}m, seg2 right: {side_seg2_len_right}m")
        print(f"  Side base rotation left X: {params['side_base_rotation_x_left']}°, Y: {params['side_base_rotation_y_left']}°, Z: {params['side_base_rotation_z_left']}°")
        print(f"  Side base rotation right X: {params['side_base_rotation_x_right']}°, Y: {params['side_base_rotation_y_right']}°, Z: {params['side_base_rotation_z_right']}°")
        print(f"  Side motor rotation left X: {params['side_motor_rotation_x_left']}°, Y: {params['side_motor_rotation_y_left']}°, Z: {params['side_motor_rotation_z_left']}°")
        print(f"  Side motor rotation right X: {params['side_motor_rotation_x_right']}°, Y: {params['side_motor_rotation_y_right']}°, Z: {params['side_motor_rotation_z_right']}°")

    def apply_visual_materials(usd_path: str):
        """Apply render materials to generated visual geometry."""
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        usd_file_path = Path(usd_path)
        base_usd_path = usd_file_path.parent / "configuration" / f"{usd_file_path.stem}_base.usd"
        if not base_usd_path.is_file():
            raise FileNotFoundError(f"Base visual USD not found: {base_usd_path}")

        stage = Usd.Stage.Open(str(base_usd_path))
        root_prim = stage.GetPrimAtPath("/quadrotor")
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim '/quadrotor' not found in {base_usd_path}")

        material_specs = {
            "base": {
                "path": "/quadrotor/Looks/base_graphite",
                "color": (0.08, 0.09, 0.10),
                "roughness": 0.55,
                "metallic": 0.0,
            },
            "arm": {
                "path": "/quadrotor/Looks/arm_orange",
                "color": (0.52, 0.18, 0.02),
                "roughness": 0.50,
                "metallic": 0.0,
            },
            "motor_blue": {
                "path": "/quadrotor/Looks/motor_blue",
                "color": (0.05, 0.22, 0.48),
                "roughness": 0.45,
                "metallic": 0.0,
            },
            "motor_yellow": {
                "path": "/quadrotor/Looks/motor_yellow",
                "color": (0.58, 0.42, 0.04),
                "roughness": 0.45,
                "metallic": 0.0,
            },
        }

        materials = {}
        for name, spec in material_specs.items():
            material_path = Sdf.Path(spec["path"])
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(spec["color"])
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(spec["roughness"])
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(spec["metallic"])
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials[name] = material

        link_material_names = {
            "base_link": "base",
            "arm_segment_2_0": "arm",
            "arm_segment_1_1": "arm",
            "arm_segment_2_1": "arm",
            "arm_segment_2_2": "arm",
            "arm_segment_1_3": "arm",
            "arm_segment_2_3": "arm",
            "motor_0": "motor_blue",
            "motor_1": "motor_yellow",
            "motor_2": "motor_blue",
            "motor_3": "motor_yellow",
        }

        total_bound_count = 0
        for link_name, material_name in link_material_names.items():
            visual_source_prim = stage.GetPrimAtPath(f"/visuals/{link_name}")
            if not visual_source_prim.IsValid():
                raise RuntimeError(f"Visual source prim '/visuals/{link_name}' not found in {base_usd_path}")

            link_bound_count = 0
            for prim in Usd.PrimRange(visual_source_prim):
                if prim.IsA(UsdGeom.Gprim):
                    prim_material_name = material_name
                    if prim.GetTypeName() == "Capsule":
                        prim_material_name = "arm"
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding_api.Bind(materials[prim_material_name])
                    gprim = UsdGeom.Gprim(prim)
                    gprim.CreateDisplayColorAttr([material_specs[prim_material_name]["color"]])
                    gprim.CreateDisplayOpacityAttr([1.0])
                    link_bound_count += 1

            if link_bound_count == 0:
                raise RuntimeError(f"No visual geometry found under '/visuals/{link_name}' in {base_usd_path}")

            total_bound_count += link_bound_count

        stage.GetRootLayer().Save()
        print(f"Applied render materials to {total_bound_count} visual prims in {base_usd_path}")

    def apply_low_friction_material(usd_path: str):
        """Apply low friction material to motor collision spheres and base collision."""
        from isaacsim.core.utils.stage import get_current_stage, open_stage
        from pxr import UsdShade

        open_stage(usd_path)
        stage = get_current_stage()

        material_cfg = RigidBodyMaterialCfg(
            static_friction=0.00,
            dynamic_friction=0.00,
            restitution=1.0,
        )

        material_prim_path = "/quadrotor/motorPhysicsMaterial"
        material_prim = spawn_rigid_body_material(material_prim_path, material_cfg)

        links_to_apply = ["base_link", "motor_0", "motor_1", "motor_2", "motor_3"]
        bound_count = 0

        for link_name in links_to_apply:
            link_prim_path = f"/quadrotor/{link_name}"
            link_prim = prim_utils.get_prim_at_path(link_prim_path)

            if not link_prim.IsValid():
                continue

            collision_prim = None
            for child in link_prim.GetAllChildren():
                if "collisions" in str(child.GetPath()) and child.HasAPI(UsdPhysics.CollisionAPI):
                    collision_prim = child
                    break

            if collision_prim:
                binding_api = UsdShade.MaterialBindingAPI.Apply(collision_prim)
                binding_api.Bind(UsdShade.Material(material_prim), UsdShade.Tokens.physics)
                bound_count += 1
                print(f"  Bound material to {collision_prim.GetPath()}")

        stage.Save()
        print(f"Applied friction material to {bound_count}/{len(links_to_apply)} collision bodies in {usd_path}")

    def store_drone_parameters(usd_path: str, torque_fn_left: str, torque_fn_right: str, rigid_arms: bool):
        """Store drone-specific parameters as custom attributes in USD.

        Args:
            usd_path: Path to the USD file
            torque_fn_left: Left arm torque response code string
            torque_fn_right: Right arm torque response code string
            rigid_arms: Whether the airframe has rigid (fixed) arm joints
        """
        from isaacsim.core.utils.stage import get_current_stage, open_stage
        from pxr import Sdf

        open_stage(usd_path)
        stage = get_current_stage()

        root_prim = stage.GetPrimAtPath("/quadrotor")
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim '/quadrotor' not found in {usd_path}")

        root_prim.CreateAttribute("morphy:rigid_arms", Sdf.ValueTypeNames.Bool).Set(rigid_arms)
        if not rigid_arms:
            root_prim.CreateAttribute("morphy:torque_fn_left", Sdf.ValueTypeNames.String).Set(torque_fn_left)
            root_prim.CreateAttribute("morphy:torque_fn_right", Sdf.ValueTypeNames.String).Set(torque_fn_right)

        stage.Save()
        print(f"Stored drone parameters in {usd_path}:")
        print(f"  rigid_arms: {rigid_arms}")
        if not rigid_arms:
            print(f"  torque_fn_left: {torque_fn_left}")
            print(f"  torque_fn_right: {torque_fn_right}")

    def extract_airframe_data_from_usd(usd_path: str) -> dict:
        """Extract airframe physical properties directly from USD file.

        Args:
            usd_path: Path to the USD file

        Returns:
            Dictionary containing total_mass, num_motors, motor_positions_body_frame, motor_directions
        """
        import numpy as np
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from pxr import UsdPhysics, UsdGeom
        import traceback

        try:
            open_stage(usd_path)
            stage = get_current_stage()

            root_prim = stage.GetPrimAtPath("/quadrotor")
            if not root_prim.IsValid():
                raise RuntimeError(f"Root prim '/quadrotor' not found in {usd_path}")

            rigid_arms_attr = root_prim.GetAttribute("morphy:rigid_arms")
            if not rigid_arms_attr.IsValid() or rigid_arms_attr.Get() is None:
                raise RuntimeError("Failed to load morphy:rigid_arms from USD.")
            torque_param_names = [] if rigid_arms_attr.Get() is True else ['torque_fn_left', 'torque_fn_right']
            torque_params = {}
            for name in torque_param_names:
                attr = root_prim.GetAttribute(f"morphy:{name}")
                if not attr.IsValid() or attr.Get() is None:
                    raise RuntimeError(f"Failed to load morphy:{name} from USD.")
                torque_params[name] = str(attr.Get())
        except Exception as e:
            print(f"[ERROR] Failed to extract USD data: {e}")
            traceback.print_exc()
            raise

        total_mass = 0.0
        motor_info = []
        joint_info = {}

        base_link_prim = stage.GetPrimAtPath("/quadrotor/base_link")
        if not base_link_prim.IsValid():
            raise RuntimeError(f"base_link not found in {usd_path}")

        base_link_xform = UsdGeom.Xformable(base_link_prim)
        base_link_world_transform = base_link_xform.ComputeLocalToWorldTransform(0)
        base_link_world_transform_inv = base_link_world_transform.GetInverse()

        def traverse_prims(prim):
            nonlocal total_mass, motor_info, joint_info

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                if mass_api:
                    mass = mass_api.GetMassAttr().Get()
                    if mass is not None:
                        total_mass += mass

            prim_name = prim.GetName()
            if prim_name.startswith("motor_") and prim_name.split("_", 1)[1].isdigit():
                motor_idx = int(prim_name.split("_", 1)[1])
                xform = UsdGeom.Xformable(prim)
                motor_world_transform = xform.ComputeLocalToWorldTransform(0)
                motor_body_transform = base_link_world_transform_inv * motor_world_transform
                translation = motor_body_transform.ExtractTranslation()
                rotation_usd = motor_body_transform.ExtractRotationMatrix()
                rotation = np.array([[float(rotation_usd[j][k]) for k in range(3)] for j in range(3)]).T
                thrust_direction = [float(rotation[0, 2]), float(rotation[1, 2]), float(rotation[2, 2])]
                world_translation = motor_world_transform.ExtractTranslation()
                world_rotation_usd = motor_world_transform.ExtractRotationMatrix()
                world_rotation = np.array([[float(world_rotation_usd[j][k]) for k in range(3)] for j in range(3)]).T
                motor_info.append((motor_idx, [float(translation[0]), float(translation[1]), float(translation[2])], thrust_direction,
                                   [float(world_translation[0]), float(world_translation[1]), float(world_translation[2])], world_rotation))

            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint = UsdPhysics.RevoluteJoint(prim)
                axis_attr = joint.GetAxisAttr().Get()
                if axis_attr is None:
                    axis_attr = "Z"

                row_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis_attr, 2)

                child_link_rel = prim.GetRelationship("physics:body1")
                child_link_targets = child_link_rel.GetTargets()
                if child_link_targets:
                    child_link_prim = stage.GetPrimAtPath(child_link_targets[0])
                    child_xform = UsdGeom.Xformable(child_link_prim)
                    child_world_transform = child_xform.ComputeLocalToWorldTransform(0)
                    child_body_transform = base_link_world_transform_inv * child_world_transform
                    child_translation = child_body_transform.ExtractTranslation()
                    child_rotation = child_body_transform.ExtractRotationMatrix()

                    joint_position_body = [float(child_translation[0]), float(child_translation[1]), float(child_translation[2])]

                    axis_body = child_rotation.GetRow(row_idx)
                    axis_body_normalized = np.array([float(axis_body[0]), float(axis_body[1]), float(axis_body[2])])
                    axis_body_normalized = axis_body_normalized / np.linalg.norm(axis_body_normalized)

                    if prim_name == "arm_segment_1_to_arm_segment_2_1":
                        joint_info["joint_1"] = {
                            "position": joint_position_body,
                            "axis": [float(axis_body_normalized[0]), float(axis_body_normalized[1]), float(axis_body_normalized[2])]
                        }
                    elif prim_name == "arm_segment_1_to_arm_segment_2_3":
                        joint_info["joint_3"] = {
                            "position": joint_position_body,
                            "axis": [float(axis_body_normalized[0]), float(axis_body_normalized[1]), float(axis_body_normalized[2])]
                        }

            for child in prim.GetAllChildren():
                traverse_prims(child)

        traverse_prims(root_prim)

        motor_info.sort(key=lambda x: x[0])
        motor_positions = [pos for _, pos, _, _, _ in motor_info]
        motor_thrust_directions = [thrust_dir for _, _, thrust_dir, _, _ in motor_info]
        motor_world_positions = [wpos for _, _, _, wpos, _ in motor_info]
        motor_world_rotations = [wrot for _, _, _, _, wrot in motor_info]
        motor_directions = AIRFRAME_CONSTANTS['motor_directions']

        rigid_attr = root_prim.GetAttribute("morphy:rigid_arms")
        is_rigid = rigid_attr.IsValid() and rigid_attr.Get() is True

        if not is_rigid and ("joint_1" not in joint_info or "joint_3" not in joint_info):
            raise RuntimeError(f"Failed to find revolute joints. Found joints: {list(joint_info.keys())}")

        arm_segment_1_1_prim = stage.GetPrimAtPath("/quadrotor/arm_segment_1_1")
        arm_segment_2_1_prim = stage.GetPrimAtPath("/quadrotor/arm_segment_2_1")
        if arm_segment_1_1_prim.IsValid() and arm_segment_2_1_prim.IsValid():
            seg1_xform = UsdGeom.Xformable(arm_segment_1_1_prim)
            seg2_xform = UsdGeom.Xformable(arm_segment_2_1_prim)
            seg1_world = seg1_xform.ComputeLocalToWorldTransform(0)
            seg2_world = seg2_xform.ComputeLocalToWorldTransform(0)
            seg1_body = base_link_world_transform_inv * seg1_world
            seg2_body = base_link_world_transform_inv * seg2_world
            seg1_trans = seg1_body.ExtractTranslation()
            seg2_trans = seg2_body.ExtractTranslation()

            for child in arm_segment_1_1_prim.GetAllChildren():
                if child.GetTypeName() == 'Mesh':
                    mesh = UsdGeom.Mesh(child)
                    extent = mesh.GetExtentAttr().Get()
                    if extent:
                        side_seg1_length = float(extent[1][0] - extent[0][0])
                        break
            else:
                side_seg1_length = AIRFRAME_CONSTANTS['side_seg1_len']

            for child in arm_segment_2_1_prim.GetAllChildren():
                if child.GetTypeName() == 'Mesh':
                    mesh = UsdGeom.Mesh(child)
                    extent = mesh.GetExtentAttr().Get()
                    if extent:
                        side_seg2_length_left = float(extent[1][0] - extent[0][0])
                        break
            else:
                if "joint_1" in joint_info:
                    joint_1_pos = np.array(joint_info["joint_1"]["position"])
                    motor_1_pos = np.array(motor_positions[1])
                    side_seg2_length_left = float(np.linalg.norm(motor_1_pos - joint_1_pos))
                else:
                    seg2_xform_left = UsdGeom.Xformable(arm_segment_2_1_prim)
                    seg2_world_left = seg2_xform_left.ComputeLocalToWorldTransform(0)
                    seg2_body_left = base_link_world_transform_inv * seg2_world_left
                    seg2_pos = seg2_body_left.ExtractTranslation()
                    motor_1_pos = np.array(motor_positions[1])
                    side_seg2_length_left = float(np.linalg.norm(motor_1_pos - np.array([float(seg2_pos[0]), float(seg2_pos[1]), float(seg2_pos[2])])))

            arm_segment_2_3_prim = stage.GetPrimAtPath("/quadrotor/arm_segment_2_3")
            for child in arm_segment_2_3_prim.GetAllChildren():
                if child.GetTypeName() == 'Mesh':
                    mesh = UsdGeom.Mesh(child)
                    extent = mesh.GetExtentAttr().Get()
                    if extent:
                        side_seg2_length_right = float(extent[1][0] - extent[0][0])
                        break
            else:
                if "joint_3" in joint_info:
                    joint_3_pos = np.array(joint_info["joint_3"]["position"])
                    motor_3_pos = np.array(motor_positions[3])
                    side_seg2_length_right = float(np.linalg.norm(motor_3_pos - joint_3_pos))
                else:
                    seg2_xform_right = UsdGeom.Xformable(arm_segment_2_3_prim)
                    seg2_world_right = seg2_xform_right.ComputeLocalToWorldTransform(0)
                    seg2_body_right = base_link_world_transform_inv * seg2_world_right
                    seg2_pos_r = seg2_body_right.ExtractTranslation()
                    motor_3_pos = np.array(motor_positions[3])
                    side_seg2_length_right = float(np.linalg.norm(motor_3_pos - np.array([float(seg2_pos_r[0]), float(seg2_pos_r[1]), float(seg2_pos_r[2])])))
        else:
            side_seg1_length = AIRFRAME_CONSTANTS['side_seg1_len']
            if "joint_1" in joint_info and "joint_3" in joint_info:
                joint_1_pos = np.array(joint_info["joint_1"]["position"])
                motor_1_pos = np.array(motor_positions[1])
                side_seg2_length_left = float(np.linalg.norm(motor_1_pos - joint_1_pos))
                joint_3_pos = np.array(joint_info["joint_3"]["position"])
                motor_3_pos = np.array(motor_positions[3])
                side_seg2_length_right = float(np.linalg.norm(motor_3_pos - joint_3_pos))
            else:
                side_seg2_length_left = 0.0
                side_seg2_length_right = 0.0

        result = {
            "total_mass": total_mass,
            "num_motors": len(motor_positions),
            "motor_positions_body_frame": motor_positions,
            "motor_thrust_directions": motor_thrust_directions,
            "motor_directions": motor_directions,
            "motor_world_positions": motor_world_positions,
            "motor_world_rotations": [[[float(motor_world_rotations[i][j, k]) for k in range(3)] for j in range(3)] for i in range(len(motor_world_rotations))],
            "rigid_arms": is_rigid,
            "side_seg1_length": side_seg1_length,
            "side_seg2_length_left": side_seg2_length_left,
            "side_seg2_length_right": side_seg2_length_right
        }
        if not is_rigid:
            result["joint_1_position"] = joint_info["joint_1"]["position"]
            result["joint_3_position"] = joint_info["joint_3"]["position"]
            result["joint_1_axis"] = joint_info["joint_1"]["axis"]
            result["joint_3_axis"] = joint_info["joint_3"]["axis"]
        result.update(torque_params)
        return result

    def save_airframe_data(output_dir: str, usd_path: str, x_0_1: list, params: dict, cad_poses: bool):
        """Save airframe physical properties to JSON by reading from USD."""
        import json

        airframe_data = extract_airframe_data_from_usd(usd_path)
        airframe_data["motor_idx_list"] = AIRFRAME_CONSTANTS["motor_idx_list"]
        airframe_data["x_0_1"] = x_0_1
        airframe_data["parameters"] = params

        json_path = os.path.join(output_dir, "airframe_data.json")
        with open(json_path, 'w') as f:
            json.dump(airframe_data, f, indent=2)

        print(f"Saved airframe data to {json_path}")

        if cad_poses:
            import numpy as np

            print(f"Airframe data extracted from USD:")
            print(f"  Total mass: {airframe_data['total_mass']:.6f} kg")
            print(f"  Number of motors: {airframe_data['num_motors']}")
            print(f"  Motor positions (body frame):")
            for i, pos in enumerate(airframe_data['motor_positions_body_frame']):
                print(f"    Motor {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")

            print(f"  Motor poses (world frame):")
            for i in range(airframe_data['num_motors']):
                wpos = airframe_data['motor_world_positions'][i]
                R = airframe_data['motor_world_rotations'][i]
                r = np.array(R)
                print(f"    Motor {i}:")
                print(f"      Position: [{wpos[0]:.6f}, {wpos[1]:.6f}, {wpos[2]:.6f}]")
                print(f"      Rotation matrix:")
                print(f"        [{r[0][0]:.6f}, {r[0][1]:.6f}, {r[0][2]:.6f}]")
                print(f"        [{r[1][0]:.6f}, {r[1][1]:.6f}, {r[1][2]:.6f}]")
                print(f"        [{r[2][0]:.6f}, {r[2][1]:.6f}, {r[2][2]:.6f}]")
                cy = math.sqrt(r[0][0]**2 + r[0][1]**2)
                if cy > 1e-6:
                    roll = math.atan2(-r[1][2], r[2][2])
                    pitch = math.atan2(r[0][2], cy)
                    yaw = math.atan2(-r[0][1], r[0][0])
                else:
                    roll = math.atan2(r[2][1], r[1][1])
                    pitch = math.atan2(r[0][2], cy)
                    yaw = 0.0
                print(f"      Euler angles (roll, pitch, yaw) [deg]: [{math.degrees(roll):.6f}, {math.degrees(pitch):.6f}, {math.degrees(yaw):.6f}]")
                thrust_dir = r[:, 2]
                elevation = math.degrees(math.asin(float(np.clip(thrust_dir[2], -1, 1))))
                azimuth = math.degrees(math.atan2(float(thrust_dir[1]), float(thrust_dir[0])))
                print(f"      Thrust direction (elevation, azimuth) [deg]: elevation={elevation:.6f}, azimuth={azimuth:.6f}")

            compute_and_plot_motor_cad_frames(airframe_data)

        return airframe_data

    def compute_and_plot_motor_cad_frames(airframe_data: dict):
        # Goal: express the arm attachment point (position + orientation) in a per-motor
        # CAD reference frame, then visualise it in 3D (one window per motor).
        #
        # Frames involved
        # ---------------
        # Body frame   : the drone's root frame. All motor positions and thrust directions
        #                from the USD are given in this frame.
        #
        # CAD frame    : one per motor, defined as follows.
        #   Origin     : 15.5 mm below the motor centre along the thrust direction
        #                  cad_origin = motor_pos - 0.0155 * z_cad
        #   z_cad      : same as the motor thrust direction (local motor z in sim).
        #   y_cad      : cross(z_cad, delta_x)  where delta_x = cad_origin - arm_attachment
        #                (vector from the arm attachment point to the CAD origin).
        #   x_cad      : cross(y_cad, z_cad)   (completes the right-handed frame).
        #
        # Arm attachment points (body frame, hardcoded from mechanical design):
        #   motor 0 → ( 0.036,  0,      0 )   faces +x  → orientation = Rz(0)
        #   motor 1 → ( 0,      0.0195, 0 )   faces +y  → orientation = Rz(+90°)
        #   motor 2 → (-0.036,  0,      0 )   faces −x  → orientation = Rz(180°)
        #   motor 3 → ( 0,     -0.0195, 0 )   faces −y  → orientation = Rz(−90°)
        #
        # The attachment orientation in the body frame is a pure Rz rotation that aligns
        # the attachment's approach axis (local x) with the arm direction.
        # We then express both the attachment position and orientation in the CAD frame.
        import numpy as np
        import matplotlib.pyplot as plt

        ARM_ATTACHMENTS = [
            np.array([0.036, 0.0, 0.0]),
            np.array([0.0, 0.0195, 0.0]),
            np.array([-0.036, 0.0, 0.0]),
            np.array([0.0, -0.0195, 0.0]),
        ]
        CAD_OFFSET_M = 0.0155  # distance from motor centre to CAD frame origin along thrust

        motor_positions = [np.array(p) for p in airframe_data['motor_positions_body_frame']]
        thrust_dirs = [np.array(t) for t in airframe_data['motor_thrust_directions']]
        rigid_arms = airframe_data["rigid_arms"]

        flexible_joints_by_motor = {}
        if not rigid_arms:
            flexible_joints_by_motor = {
                1: ("joint_1", np.array(airframe_data["joint_1_position"]), np.array(airframe_data["joint_1_axis"])),
                3: ("joint_3", np.array(airframe_data["joint_3_position"]), np.array(airframe_data["joint_3_axis"])),
            }

        def Rz(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        def unit_vector(vector, label):
            norm = np.linalg.norm(vector)
            if norm <= 1e-12:
                raise ValueError(f"{label} has near-zero length")
            return vector / norm

        # Flexible-joint transform math
        # -----------------------------
        # Let B be the body frame and C be the motor CAD frame. The CAD frame is
        # represented in body coordinates by:
        #     R_body_from_cad = [x_cad_B, y_cad_B, z_cad_B]
        #     cad_origin_B = motor_pos_B - CAD_OFFSET_M * z_cad_B
        #
        # Any body-frame point p_B is expressed in CAD coordinates by:
        #     p_C = R_body_from_cad.T @ (p_B - cad_origin_B)
        #
        # Any body-frame direction v_B is expressed in CAD coordinates by:
        #     v_C = R_body_from_cad.T @ v_B
        #
        # For the joint pose, the revolute axis defines z_joint_B. The x axis is
        # the motor-side segment direction, projected onto the plane normal to
        # z_joint_B so the pose stays orthonormal:
        #     z_joint_B = normalize(joint_axis_B)
        #     x_raw_B = motor_pos_B - joint_pos_B
        #     x_joint_B = normalize(x_raw_B - dot(x_raw_B, z_joint_B) * z_joint_B)
        #     y_joint_B = normalize(cross(z_joint_B, x_joint_B))
        #     x_joint_B = normalize(cross(y_joint_B, z_joint_B))
        #     R_body_from_joint = column_stack([x_joint_B, y_joint_B, z_joint_B])
        #
        # The complete joint pose in CAD coordinates is therefore:
        #     joint_pos_C = R_body_from_cad.T @ (joint_pos_B - cad_origin_B)
        #     joint_axis_C = R_body_from_cad.T @ z_joint_B
        #     R_cad_from_joint = R_body_from_cad.T @ R_body_from_joint
        def compute_flexible_joint_pose_in_cad(motor_idx, cad_origin, R_body_from_cad, arm_attach_body, R_body_from_attach):
            joint_name, joint_pos_body, joint_axis_body = flexible_joints_by_motor[motor_idx]
            joint_axis_body = unit_vector(joint_axis_body, f"{joint_name} axis")
            joint_pos_cad = R_body_from_cad.T @ (joint_pos_body - cad_origin)
            joint_axis_cad = unit_vector(R_body_from_cad.T @ joint_axis_body, f"{joint_name} axis in CAD frame")

            joint_x_body = motor_positions[motor_idx] - joint_pos_body
            joint_x_body = joint_x_body - np.dot(joint_x_body, joint_axis_body) * joint_axis_body
            joint_x_body = unit_vector(joint_x_body, f"{joint_name} motor-side direction")
            joint_y_body = unit_vector(np.cross(joint_axis_body, joint_x_body), f"{joint_name} y axis")
            joint_x_body = unit_vector(np.cross(joint_y_body, joint_axis_body), f"{joint_name} x axis")
            R_body_from_joint = np.column_stack([joint_x_body, joint_y_body, joint_axis_body])
            R_cad_from_joint = R_body_from_cad.T @ R_body_from_joint

            joint_pos_attach = R_body_from_attach.T @ (joint_pos_body - arm_attach_body)
            R_attach_from_joint = R_body_from_attach.T @ R_body_from_joint

            return {
                "name": joint_name,
                "position_body": joint_pos_body,
                "axis_body": joint_axis_body,
                "position_cad": joint_pos_cad,
                "axis_cad": joint_axis_cad,
                "rotation_cad": R_cad_from_joint,
                "position_attach": joint_pos_attach,
                "rotation_attach": R_attach_from_joint,
            }

        # Attachment orientations in the body frame: approach axis aligned with arm direction.
        ARM_ORIENTATIONS_BODY = [Rz(0), Rz(np.pi/2), Rz(np.pi), Rz(-np.pi/2)]

        print("Arm attachment pose in CAD motor frames:")
        if rigid_arms:
            print("Rigid airframe: no flexible joints to express in CAD motor frames.")
        results = []
        for i in range(4):
            z_cad = thrust_dirs[i] / np.linalg.norm(thrust_dirs[i])
            cad_origin = motor_positions[i] - CAD_OFFSET_M * z_cad
            arm_attach = ARM_ATTACHMENTS[i]
            # delta_x points from arm attachment to CAD origin, used to define y_cad
            delta_x = cad_origin - arm_attach
            y_cad = np.cross(z_cad, delta_x)
            y_cad /= np.linalg.norm(y_cad)
            x_cad = np.cross(y_cad, z_cad)
            x_cad /= np.linalg.norm(x_cad)
            p_in_body = arm_attach - cad_origin
            R_body_from_cad = np.column_stack([x_cad, y_cad, z_cad])
            attach_cad = R_body_from_cad.T @ p_in_body
            R_attach_cad = R_body_from_cad.T @ ARM_ORIENTATIONS_BODY[i]

            d = R_attach_cad[:, 0]
            azimuth = np.degrees(np.arctan2(d[1], d[0]))
            elevation = np.degrees(np.arctan2(d[2], np.sqrt(d[0]**2 + d[1]**2)))
            ca, sa = np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))
            ce, se = np.cos(np.radians(elevation)), np.sin(np.radians(elevation))
            R0 = np.array([[ca*ce, -sa, -ca*se], [sa*ce, ca, -sa*se], [se, 0, ce]])
            R_rel = R0.T @ R_attach_cad
            twist = np.degrees(np.arctan2(R_rel[2, 1], R_rel[1, 1]))

            flexible_joint = None
            if i in flexible_joints_by_motor:
                flexible_joint = compute_flexible_joint_pose_in_cad(i, cad_origin, R_body_from_cad, arm_attach, ARM_ORIENTATIONS_BODY[i])

            results.append((motor_positions[i], cad_origin, x_cad, y_cad, z_cad, arm_attach, attach_cad, R_attach_cad, flexible_joint))
            r = R_attach_cad
            print(f"  Motor {i}:")
            print(f"    Position (CAD frame) [m]: [{attach_cad[0]:.6f}, {attach_cad[1]:.6f}, {attach_cad[2]:.6f}]")
            print(f"    Orientation (CAD frame) rotation matrix:")
            print(f"      [{r[0,0]:.6f}, {r[0,1]:.6f}, {r[0,2]:.6f}]")
            print(f"      [{r[1,0]:.6f}, {r[1,1]:.6f}, {r[1,2]:.6f}]")
            print(f"      [{r[2,0]:.6f}, {r[2,1]:.6f}, {r[2,2]:.6f}]")
            print(f"    Orientation (CAD frame): azimuth={azimuth:.6f} deg, elevation={elevation:.6f} deg, twist={twist:.6f} deg")
            if flexible_joint is not None:
                jp = flexible_joint["position_cad"]
                ja = flexible_joint["axis_cad"]
                jr = flexible_joint["rotation_cad"]
                print(f"    Flexible joint {flexible_joint['name']}:")
                print(f"      Position (CAD frame) [m]: [{jp[0]:.6f}, {jp[1]:.6f}, {jp[2]:.6f}]")
                print(f"      Axis (CAD frame): [{ja[0]:.6f}, {ja[1]:.6f}, {ja[2]:.6f}]")
                print(f"      Pose (CAD frame) rotation matrix:")
                print(f"        [{jr[0,0]:.6f}, {jr[0,1]:.6f}, {jr[0,2]:.6f}]")
                print(f"        [{jr[1,0]:.6f}, {jr[1,1]:.6f}, {jr[1,2]:.6f}]")
                print(f"        [{jr[2,0]:.6f}, {jr[2,1]:.6f}, {jr[2,2]:.6f}]")
                jpa = flexible_joint["position_attach"]
                jra = flexible_joint["rotation_attach"]
                print(f"      Position (attach frame) [m]: [{jpa[0]:.6f}, {jpa[1]:.6f}, {jpa[2]:.6f}]")
                print(f"      Pose (attach frame) rotation matrix:")
                print(f"        [{jra[0,0]:.6f}, {jra[0,1]:.6f}, {jra[0,2]:.6f}]")
                print(f"        [{jra[1,0]:.6f}, {jra[1,1]:.6f}, {jra[1,2]:.6f}]")
                print(f"        [{jra[2,0]:.6f}, {jra[2,1]:.6f}, {jra[2,2]:.6f}]")

        S = 0.015

        def axis3d(ax, origin, direction, color, label):
            tip = origin + direction * S
            ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]], color=color, lw=2)
            ax.scatter(*tip, color=color, s=30, zorder=5)
            if label:
                ax.text(tip[0], tip[1], tip[2], label, color=color, fontsize=8)

        BODY_AXES = [(np.array([1,0,0]), '#dd4444', 'Xb'), (np.array([0,1,0]), '#44aa44', 'Yb'), (np.array([0,0,1]), '#4444dd', 'Zb')]
        CAD_COLORS = [('red', 'x_cad'), ('green', 'y_cad'), ('blue', 'z_cad')]

        for i, (motor_pos, cad_origin, x_cad, y_cad, z_cad, arm_attach, attach_cad, R_attach_cad, flexible_joint) in enumerate(results):
            joint_title = ""
            if flexible_joint is not None:
                jp = flexible_joint["position_cad"]
                ja = flexible_joint["axis_cad"]
                joint_title = (
                    f"\nFlexible joint {flexible_joint['name']} CAD [mm]: [{jp[0]*1000:.2f}, {jp[1]*1000:.2f}, {jp[2]*1000:.2f}]    "
                    f"Axis CAD: [{ja[0]:.3f}, {ja[1]:.3f}, {ja[2]:.3f}]"
                )

            fig = plt.figure(figsize=(10, 9))
            fig.suptitle(
                f"Motor {i} — CAD Frame Analysis\n"
                f"Motor pos (body) [mm]: [{motor_pos[0]*1000:.2f}, {motor_pos[1]*1000:.2f}, {motor_pos[2]*1000:.2f}]    "
                f"CAD origin (body) [mm]: [{cad_origin[0]*1000:.2f}, {cad_origin[1]*1000:.2f}, {cad_origin[2]*1000:.2f}]\n"
                f"Arm attachment body [mm]: [{ARM_ATTACHMENTS[i][0]*1000:.2f}, {ARM_ATTACHMENTS[i][1]*1000:.2f}, {ARM_ATTACHMENTS[i][2]*1000:.2f}]    "
                f"Arm attachment CAD [mm]: [{attach_cad[0]*1000:.2f}, {attach_cad[1]*1000:.2f}, {attach_cad[2]*1000:.2f}]"
                f"{joint_title}",
                fontsize=10
            )

            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.set_xlabel("X (m)")
            ax3d.set_ylabel("Y (m)")
            ax3d.set_zlabel("Z (m)")

            for bvec, bc, bl in BODY_AXES:
                axis3d(ax3d, np.zeros(3), bvec, bc, bl)

            ax3d.scatter(*motor_pos, c='black', s=60, zorder=5, label='Motor center')
            ax3d.plot([motor_pos[0], cad_origin[0]], [motor_pos[1], cad_origin[1]], [motor_pos[2], cad_origin[2]], 'k--', lw=1, alpha=0.5)
            ax3d.scatter(*cad_origin, c='black', s=80, marker='^', zorder=5, label='CAD origin')

            for (cc, cl), cvec in zip(CAD_COLORS, [x_cad, y_cad, z_cad]):
                axis3d(ax3d, cad_origin, cvec, cc, cl)

            ax3d.scatter(*arm_attach, c='magenta', s=80, marker='s', zorder=5, label='Arm attach')
            ax3d.plot([cad_origin[0], arm_attach[0]], [cad_origin[1], arm_attach[1]], [cad_origin[2], arm_attach[2]], 'm-', lw=1.5, alpha=0.8)
            R_attach = ARM_ORIENTATIONS_BODY[i]
            for col_idx, (ac, al) in enumerate([('orange', 'Xa'), ('limegreen', 'Ya'), ('purple', 'Za')]):
                axis3d(ax3d, arm_attach, R_attach[:, col_idx], ac, al)
            if flexible_joint is not None:
                joint_pos_body = flexible_joint["position_body"]
                joint_axis_body = flexible_joint["axis_body"]
                ax3d.scatter(*joint_pos_body, c='cyan', s=80, marker='D', zorder=5, label=f"{flexible_joint['name']} flexible joint")
                ax3d.plot([cad_origin[0], joint_pos_body[0]], [cad_origin[1], joint_pos_body[1]], [cad_origin[2], joint_pos_body[2]], color='cyan', lw=1.5, alpha=0.8)
                axis3d(ax3d, joint_pos_body, joint_axis_body, 'cyan', f"{flexible_joint['name']} axis")
            ax3d.legend(fontsize=8)

            all_pts_list = [np.zeros(3), motor_pos, cad_origin, arm_attach,
                            cad_origin + x_cad * S, cad_origin + y_cad * S, cad_origin + z_cad * S]
            if flexible_joint is not None:
                joint_pos_body = flexible_joint["position_body"]
                joint_axis_body = flexible_joint["axis_body"]
                all_pts_list.extend([joint_pos_body, joint_pos_body + joint_axis_body * S, joint_pos_body - joint_axis_body * S])
            all_pts = np.array(all_pts_list)
            mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
            half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
            ax3d.set_xlim(mid[0] - half, mid[0] + half)
            ax3d.set_ylim(mid[1] - half, mid[1] + half)
            ax3d.set_zlim(mid[2] - half, mid[2] + half)
            ax3d.set_box_aspect([1, 1, 1])

            plt.tight_layout()
            plt.show()

    def main():
        """Convert the Morphy URDF to USD."""

        # Get absolute paths
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(script_dir, "src/morphy_prog_parametric.urdf")

        if args_cli.output_dir:
            output_dir = os.path.abspath(args_cli.output_dir)
        else:
            output_dir = os.path.join(script_dir, "cache/workspace_local/usd")

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Converting Parametric Morphy")
        print(f"  - Base: Box (7x5x8 cm, WITH collision)")
        print(f"  - Front/Back arms: Parametric length, prop pitch")
        print(f"  - Side arms: Parametric 2-segment arms with revolute joints")
        print(f"  - Motors: Spheres (WITH collision, radius=4.5cm)")
        print(f"{'='*80}\n")

        # Create parameters dictionary from CLI args
        params = {key: getattr(args_cli, key) for key in PARAM_BOUNDS.keys()}

        generate_parametric_urdf(urdf_path, params, rigid=args_cli.rigid)

        urdf_converter_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=output_dir,
            usd_file_name="morphy_prog.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
        )

        # Create converter and convert
        urdf_converter = UrdfConverter(urdf_converter_cfg)

        output_path = urdf_converter.usd_path
        print(f"Conversion complete!")
        print(f"USD file saved to: {output_path}")

        store_drone_parameters(
            output_path,
            torque_fn_left=args_cli.torque_fn_left,
            torque_fn_right=args_cli.torque_fn_right,
            rigid_arms=args_cli.rigid,
        )

        apply_visual_materials(output_path)

        apply_low_friction_material(output_path)

        airframe_data = save_airframe_data(output_dir, output_path, x_0_1, params, args_cli.cad_poses)

        if args_cli.top_view_motor_spins is True:
            top_view_path = os.path.join(output_dir, "top_view_motor_spins.png")
            save_top_view_motor_spins_plot(airframe_data, top_view_path)

        print(f"")
        print(f"{'='*80}\n")

    main()
    simulation_app.close()
