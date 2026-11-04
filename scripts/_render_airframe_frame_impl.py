"""Implementation of single-frame airframe rendering for PDF reports.

This script receives a workspace_dir with a pre-generated USD file (same pattern as video generation).
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render single frame of airframe")
parser.add_argument("--workspace_dir", type=str, required=True, help="Directory containing morphy_prog.usd")
parser.add_argument("--output_path", type=str, required=True, help="Output PNG path")
parser.add_argument("--camera_distance", type=float, required=True, help="Camera distance from drone")
parser.add_argument("--camera_height", type=float, required=True, help="Camera height")
parser.add_argument("--camera_azimuth", type=float, required=True, help="Camera azimuth angle in degrees")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli, multi_gpu=False)
simulation_app = app_launcher.app
print("[INFO] Isaac Sim app launched")
sys_stdout_flush = __import__("sys").stdout.flush
sys_stdout_flush()

import torch
import numpy as np
import sys
import cv2
from pxr import Gf, UsdGeom

sys.path.append(str(Path(__file__).parent.parent / "src"))

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, quat_apply
from isaacsim.core.utils.stage import get_current_stage
print("[INFO] Render script imports complete")
sys.stdout.flush()

AXES_LENGTH = 0.25
MOTOR_AXIS_LENGTH = 0.04
MOTOR_AXIS_ADDITIONAL_MARGIN = 0.00
PHYSICS_HZ = 240
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CAMERA_PRIM_PATH = "/World/RenderCamera"


def camera_quaternion_from_view(camera_eye, camera_target):
    eye = np.asarray(camera_eye, dtype=float)
    target = np.asarray(camera_target, dtype=float)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 0.0:
        raise ValueError(f"Invalid camera view with identical eye and target: {camera_eye}")
    forward = forward / forward_norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-8:
        raise ValueError(f"Invalid camera view with forward parallel to world up: eye={camera_eye}, target={camera_target}")
    right = right / right_norm
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    rotation = np.column_stack([right, up, -forward])
    return quaternion_from_rotation_matrix(rotation)


def quaternion_from_rotation_matrix(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.asarray([w, x, y, z], dtype=float)
    return quat / np.linalg.norm(quat)


def set_camera_prim_pose(camera_prim_path, camera_eye, camera_target):
    prim = UsdGeom.Camera.Get(get_current_stage(), camera_prim_path).GetPrim()
    if not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_prim_path}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    quat = camera_quaternion_from_view(camera_eye, camera_target)
    translate_op.Set(Gf.Vec3d(float(camera_eye[0]), float(camera_eye[1]), float(camera_eye[2])))
    orient_op.Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))


@configclass
class RenderSceneCfg(InteractiveSceneCfg):
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
                joint_names_expr=["arm_segment_1_to_arm_segment_2_1", "arm_segment_1_to_arm_segment_2_3"],
                stiffness=0.0, damping=0.0, effort_limit_sim=20000.0, velocity_limit_sim=10000.0,
            )
        },
    )


def main():
    workspace_dir = Path(args_cli.workspace_dir)
    usd_path = workspace_dir / "morphy_prog.usd"

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    print("[INFO] Creating simulation context")
    sys.stdout.flush()
    sim_cfg = sim_utils.SimulationCfg(dt=1.0/PHYSICS_HZ, device="cuda:0", gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)

    print("[INFO] Creating scene")
    sys.stdout.flush()
    scene_cfg = RenderSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    scene_cfg.robot.spawn.usd_path = str(usd_path)
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]

    azimuth_rad = np.radians(args_cli.camera_azimuth)
    camera_x = args_cli.camera_distance * np.cos(azimuth_rad)
    camera_y = args_cli.camera_distance * np.sin(azimuth_rad)
    camera_eye = [camera_x, camera_y, args_cli.camera_height]
    camera_target = [0.0, 0.0, 0.5]

    print("[INFO] Creating render camera")
    sys.stdout.flush()
    camera_cfg = CameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        update_period=0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
    )
    camera = Camera(camera_cfg)

    print("[INFO] Setting render camera pose")
    sys.stdout.flush()
    set_camera_prim_pose(CAMERA_PRIM_PATH, camera_eye, camera_target)

    print("[INFO] Resetting simulation")
    sys.stdout.flush()
    sim.reset()

    print(f"[INFO] Loaded robot from {usd_path}")
    sys.stdout.flush()

    drone_axes_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/DroneAxes",
        markers={
            "x_axis": sim_utils.CylinderCfg(radius=0.003, height=AXES_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
            "y_axis": sim_utils.CylinderCfg(radius=0.003, height=AXES_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
            "z_axis": sim_utils.CylinderCfg(radius=0.003, height=AXES_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))),
        }
    )
    drone_axes_marker = VisualizationMarkers(drone_axes_cfg)

    motor_names = [name for name in robot.body_names if name.startswith("motor")]
    motor_indices = [robot.body_names.index(name) for name in motor_names]

    thrust_shaft_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/MotorThrustShaft",
        markers={"shaft": sim_utils.CylinderCfg(radius=0.003, height=MOTOR_AXIS_LENGTH * 0.7,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)))}
    )
    thrust_shaft_marker = VisualizationMarkers(thrust_shaft_cfg)

    thrust_cone_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/MotorThrustCone",
        markers={"cone": sim_utils.ConeCfg(radius=0.008, height=MOTOR_AXIS_LENGTH * 0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)))}
    )
    thrust_cone_marker = VisualizationMarkers(thrust_cone_cfg)

    output_path = Path(args_cli.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Starting simulation loop...")
    sys.stdout.flush()

    num_steps = 120
    for step in range(num_steps):
        sim.step()
        robot.update(dt=sim.get_physics_dt())

        drone_pos = robot.data.root_pos_w[0:1]
        drone_quat = robot.data.root_quat_w[0:1]

        x_rot = quat_from_angle_axis(torch.tensor([torch.pi/2], device=robot.device),
                                     torch.tensor([[0.0, 1.0, 0.0]], device=robot.device))
        y_rot = quat_from_angle_axis(torch.tensor([-torch.pi/2], device=robot.device),
                                     torch.tensor([[1.0, 0.0, 0.0]], device=robot.device))

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

        if step % 30 == 0:
            print(f"[INFO] Step {step}/{num_steps}")
            sys.stdout.flush()

    print(f"[INFO] Simulation loop complete. Capturing frame...")
    sys.stdout.flush()

    camera.update(dt=sim.get_physics_dt())
    if "rgb" not in camera.data.output:
        raise RuntimeError("Camera did not produce rgb output")
    rgb_tensor = camera.data.output["rgb"]
    if rgb_tensor.shape != (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise RuntimeError(f"Unexpected camera rgb shape: {rgb_tensor.shape}")
    rgb_frame = rgb_tensor[0].detach().cpu().numpy()
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), bgr_frame):
        raise RuntimeError(f"Failed to write frame: {output_path}")
    print(f"[INFO] Frame saved: {output_path}")
    sys.stdout.flush()

    print("[INFO] Closing simulation app...")
    sys.stdout.flush()
    simulation_app.close()


if __name__ == "__main__":
    main()
