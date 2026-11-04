"""Implementation of airframe visualization video generation."""

import argparse
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate airframe visualization video")
parser.add_argument("--workspace_dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli, multi_gpu=False)
simulation_app = app_launcher.app

import torch
import numpy as np
import math
import sys
import json
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from pxr import Usd, Sdf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, quat_apply
from airframe_encoding import Airframe
from airframe_validations import build_collision_torque_response
from utils import Recorder

# Hardcoded parameters
ANIMATION_DURATION = 30.0  # Duration for the animation rotations
NUM_ROTATIONS = 3  # Number of full camera rotations
CAMERA_DISTANCE = 0.6  # Distance from drone center
CAMERA_HEIGHT = 1.5  # Height of camera
VIDEO_FPS = 60  # Video frames per second
PHYSICS_HZ = 240  # Physics simulation frequency
AXES_LENGTH = 0.3  # Length of drone body frame axes
MOTOR_AXIS_LENGTH = 0.05  # Length of motor thrust vectors
MOTOR_AXIS_ADDITIONAL_MARGIN = 0.00  # Additional Z offset for motor arrows in body frame
RENDER_RESOLUTION = (1920, 1080)
TEXT_PANEL_WIDTH = 420


def load_joint_limits_from_workspace(workspace_dir: Path) -> tuple[float, float]:
    """Load collision-derived side-arm joint limits from airframe_data.json."""
    json_path = workspace_dir / "airframe_data.json"
    with open(json_path, "r") as f:
        airframe_data = json.load(f)
    angle_b = Airframe.from_x_0_1(airframe_data["x_0_1"]).compute_end_angle_b()
    return math.radians(angle_b["end_angle_b_left"]), math.radians(angle_b["end_angle_b_right"])


def load_width_metrics_from_workspace(workspace_dir: Path) -> dict[str, float]:
    """Load width metrics derived from the current airframe geometry."""
    json_path = workspace_dir / "airframe_data.json"
    with open(json_path, "r") as f:
        airframe_data = json.load(f)
    airframe = Airframe.from_x_0_1(airframe_data["x_0_1"])
    torque_response = build_collision_torque_response(airframe, delta_deg=0.01, verbose=False)
    aabb_info = torque_response["aabb_info"]
    extended_width = float(aabb_info["unfolded_width"])
    folded_width = float(aabb_info["max_width"])
    width_reduction_percent = (1.0 - folded_width / extended_width) * 100.0
    return {
        "extended_width": extended_width,
        "folded_width": folded_width,
        "width_reduction_percent": width_reduction_percent,
    }


def build_width_overlay_postprocessor(width_metrics: dict[str, float], render_resolution: tuple[int, int], panel_width: int):
    """Create a frame postprocessor that appends a metrics panel on the right."""
    output_width = render_resolution[0] + panel_width
    output_height = render_resolution[1]

    lines = [
        ("Extended width", f"{width_metrics['extended_width'] * 100:.1f} cm"),
        ("Folded width", f"{width_metrics['folded_width'] * 100:.1f} cm"),
        ("Reduction in width", f"{width_metrics['width_reduction_percent']:.1f} %"),
    ]

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    body_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.1
    body_scale = 1.0
    title_thickness = 3
    body_thickness = 2
    text_color = (0, 0, 0)
    line_gap = 120
    value_gap = 48

    def postprocess(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[1] != render_resolution[0] or frame_bgr.shape[0] != render_resolution[1]:
            raise ValueError(
                f"Unexpected captured frame shape {frame_bgr.shape[:2]}, expected {(render_resolution[1], render_resolution[0])}"
            )

        canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
        canvas[:, :render_resolution[0]] = frame_bgr

        panel_x0 = render_resolution[0]
        cv2.rectangle(canvas, (panel_x0, 0), (panel_x0 + panel_width - 1, output_height - 1), (220, 220, 220), 2)

        text_x = panel_x0 + 28
        title_y = 90
        cv2.putText(canvas, "Airframe width", (text_x, title_y), title_font, title_scale, text_color, title_thickness, cv2.LINE_AA)

        for idx, (label, value) in enumerate(lines):
            block_top = title_y + 90 + idx * line_gap
            cv2.putText(canvas, label, (text_x, block_top), body_font, body_scale, text_color, body_thickness, cv2.LINE_AA)
            cv2.putText(canvas, value, (text_x, block_top + value_gap), body_font, body_scale, text_color, body_thickness, cv2.LINE_AA)

        return canvas

    return postprocess


@configclass
class VisualizationSceneCfg(InteractiveSceneCfg):
    """Configuration for visualization scene."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path="", activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        actuators={
            "soft_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_segment_1_to_arm_segment_2_1",
                    "arm_segment_1_to_arm_segment_2_3",
                ],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=20000.0,
                velocity_limit_sim=10000.0,
            )
        },
    )


def main():
    workspace_dir = Path(args_cli.workspace_dir)
    usd_path = workspace_dir / "morphy_prog.usd"

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(dt=1.0/PHYSICS_HZ, device="cuda:0", gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)

    scene_cfg = VisualizationSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    scene_cfg.robot.spawn.usd_path = str(usd_path)
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]

    sim.reset()

    soft_joint_names = [
        "arm_segment_1_to_arm_segment_2_1",
        "arm_segment_1_to_arm_segment_2_3",
    ]
    soft_joint_indices = [robot.joint_names.index(name) for name in soft_joint_names]

    max_angle_left, max_angle_right = load_joint_limits_from_workspace(workspace_dir)
    width_metrics = load_width_metrics_from_workspace(workspace_dir)

    print(f"[INFO] Loaded robot from {usd_path}")
    print(f"[INFO] Soft joint indices: {soft_joint_indices}")
    print(f"[INFO] Joint limits from USD: left={math.degrees(max_angle_left):.1f}°, right={math.degrees(max_angle_right):.1f}°")
    print(
        f"[INFO] Width metrics: extended={width_metrics['extended_width']:.4f}m, "
        f"folded={width_metrics['folded_width']:.4f}m, reduction={width_metrics['width_reduction_percent']:.1f}%"
    )

    # Setup drone body frame axes markers (RGB = XYZ)
    drone_axes_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/DroneAxes",
        markers={
            "x_axis": sim_utils.CylinderCfg(
                radius=0.003,
                height=AXES_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "y_axis": sim_utils.CylinderCfg(
                radius=0.003,
                height=AXES_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "z_axis": sim_utils.CylinderCfg(
                radius=0.003,
                height=AXES_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )
    drone_axes_marker = VisualizationMarkers(drone_axes_cfg)

    # Setup motor thrust vector markers (arrow = shaft + cone tip)
    motor_names = [name for name in robot.body_names if name.startswith("motor")]
    motor_indices = [robot.body_names.index(name) for name in motor_names]

    thrust_shaft_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/MotorThrustShaft",
        markers={
            "shaft": sim_utils.CylinderCfg(
                radius=0.003,
                height=MOTOR_AXIS_LENGTH * 0.7,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            )
        }
    )
    thrust_shaft_marker = VisualizationMarkers(thrust_shaft_cfg)

    thrust_cone_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/MotorThrustCone",
        markers={
            "cone": sim_utils.ConeCfg(
                radius=0.008,
                height=MOTOR_AXIS_LENGTH * 0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            )
        }
    )
    thrust_cone_marker = VisualizationMarkers(thrust_cone_cfg)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    video_path = results_dir / f"{timestamp}_airframe_visualization.mp4"

    print(f"[INFO] Recording video: {video_path}")
    print(f"[INFO] Duration: {ANIMATION_DURATION}s, Rotations: {NUM_ROTATIONS}, Video FPS: {VIDEO_FPS}, Physics: {PHYSICS_HZ}Hz")

    rotation_duration = ANIMATION_DURATION / NUM_ROTATIONS
    total_physics_steps = int(ANIMATION_DURATION * PHYSICS_HZ)

    # Rotation 1: 0-10s (neutral)
    rotation_1_end_step = int(rotation_duration * PHYSICS_HZ)
    # Rotation 2: 10-20s (increase to max, hold at max)
    rotation_2_end_step = int(2 * rotation_duration * PHYSICS_HZ)
    # Rotation 3: 20-30s (return in first half, hold at neutral in second half)
    rotation_3_end_step = total_physics_steps

    recorder = Recorder(
        playback_speed=1.0,
        mp4_path=str(video_path),
        dt=1.0/VIDEO_FPS,
        sim=sim,
        resolution=(RENDER_RESOLUTION[0] + TEXT_PANEL_WIDTH, RENDER_RESOLUTION[1]),
        camera_eye=[CAMERA_DISTANCE, 0.0, CAMERA_HEIGHT],
        camera_target=[0.0, 0.0, 0.5]
    )
    recorder.capture_resolution = RENDER_RESOLUTION
    recorder.frame_postprocessor = build_width_overlay_postprocessor(width_metrics, RENDER_RESOLUTION, TEXT_PANEL_WIDTH)

    total_video_frames = int(ANIMATION_DURATION * VIDEO_FPS)
    physics_steps_per_video_frame = PHYSICS_HZ // VIDEO_FPS

    from utils import quat_from_euler_xyz_tensor

    for physics_step in range(total_physics_steps):
        should_record = (physics_step % physics_steps_per_video_frame == 0)
        video_frame = physics_step // physics_steps_per_video_frame

        t = physics_step / PHYSICS_HZ
        rotation_t = (t % rotation_duration) / rotation_duration
        yaw_angle = rotation_t * 2 * np.pi

        # Rotation 1: Neutral position
        if physics_step < rotation_1_end_step:
            joint_angle_1 = 0.0
            joint_angle_2 = 0.0

        # Rotation 2: First half increase to max, second half hold at max
        elif physics_step < rotation_2_end_step:
            rotation_2_progress = (physics_step - rotation_1_end_step) / (rotation_2_end_step - rotation_1_end_step)
            if rotation_2_progress < 0.5:
                ramp = rotation_2_progress * 2
                joint_angle_1 = ramp * max_angle_left
                joint_angle_2 = ramp * (-max_angle_right)
            else:
                joint_angle_1 = max_angle_left
                joint_angle_2 = -max_angle_right

        # Rotation 3: First half return to neutral, second half hold neutral
        else:
            rotation_3_progress = (physics_step - rotation_2_end_step) / (rotation_3_end_step - rotation_2_end_step)
            if rotation_3_progress < 0.5:
                ramp = 1.0 - (rotation_3_progress * 2)
                joint_angle_1 = max_angle_left * ramp
                joint_angle_2 = -max_angle_right * ramp
            else:
                joint_angle_1 = 0.0
                joint_angle_2 = 0.0

        joint_positions = robot.data.default_joint_pos.clone()
        joint_positions[0, soft_joint_indices[0]] = joint_angle_1
        joint_positions[0, soft_joint_indices[1]] = joint_angle_2

        quat = quat_from_euler_xyz_tensor(torch.tensor([[0.0, 0.0, yaw_angle]], device=robot.device))
        root_state = robot.data.default_root_state.clone()
        root_state[0, 3:7] = quat[0]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_positions, robot.data.default_joint_vel)

        sim.step()
        robot.update(dt=sim.get_physics_dt())

        # Update drone body frame axes (cylinders default along Z, rotate to X and Y for those axes)
        drone_pos = robot.data.root_pos_w[0:1]
        drone_quat = robot.data.root_quat_w[0:1]

        x_rot = quat_from_angle_axis(torch.tensor([torch.pi/2], device=robot.device), torch.tensor([[0.0, 1.0, 0.0]], device=robot.device))
        y_rot = quat_from_angle_axis(torch.tensor([-torch.pi/2], device=robot.device), torch.tensor([[1.0, 0.0, 0.0]], device=robot.device))

        drone_x_quat = quat_mul(drone_quat, x_rot)
        drone_y_quat = quat_mul(drone_quat, y_rot)
        drone_z_quat = drone_quat

        axis_offset = AXES_LENGTH / 2
        drone_x_offset = quat_apply(drone_x_quat, torch.tensor([[0.0, 0.0, axis_offset]], device=robot.device))
        drone_y_offset = quat_apply(drone_y_quat, torch.tensor([[0.0, 0.0, axis_offset]], device=robot.device))
        drone_z_offset = quat_apply(drone_z_quat, torch.tensor([[0.0, 0.0, axis_offset]], device=robot.device))

        drone_axes_marker.visualize(
            translations=torch.cat([drone_pos + drone_x_offset, drone_pos + drone_y_offset, drone_pos + drone_z_offset], dim=0),
            orientations=torch.cat([drone_x_quat, drone_y_quat, drone_z_quat], dim=0),
            marker_indices=[0, 1, 2]
        )

        # Update motor thrust vector visualizations (arrow = shaft + cone)
        motor_positions = robot.data.body_pos_w[0, motor_indices]
        motor_quats = robot.data.body_quat_w[0, motor_indices]

        shaft_height = MOTOR_AXIS_LENGTH * 0.7
        cone_height = MOTOR_AXIS_LENGTH * 0.3

        shaft_offset = MOTOR_AXIS_ADDITIONAL_MARGIN + shaft_height / 2
        shaft_offsets = quat_apply(motor_quats, torch.tensor([[0.0, 0.0, shaft_offset]], device=robot.device).repeat(len(motor_indices), 1))
        thrust_shaft_marker.visualize(translations=motor_positions + shaft_offsets, orientations=motor_quats)

        cone_offset = MOTOR_AXIS_ADDITIONAL_MARGIN + shaft_height + cone_height / 2
        cone_offsets = quat_apply(motor_quats, torch.tensor([[0.0, 0.0, cone_offset]], device=robot.device).repeat(len(motor_indices), 1))
        thrust_cone_marker.visualize(translations=motor_positions + cone_offsets, orientations=motor_quats)

        if should_record:
            recorder.record_frame()
            if video_frame % 60 == 0:
                print(f"[INFO] Video frame {video_frame}/{total_video_frames} ({100*video_frame/total_video_frames:.1f}%)")
                sys.stdout.flush()

    print("[DEBUG] All frames complete, about to save video...")
    sys.stdout.flush()
    recorder.save()
    print(f"[INFO] Video generation complete: {video_path}")
    sys.stdout.flush()

    print("[DEBUG] About to close simulation app...")
    sys.stdout.flush()
    simulation_app.close()
    print("[DEBUG] Simulation app closed")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
