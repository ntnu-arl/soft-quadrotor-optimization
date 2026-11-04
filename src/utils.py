from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import yaml



def quat_mul(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q):
    """Compute quaternion conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_inverse(q):
    """Compute quaternion inverse (assumes unit quaternions)."""
    return quat_conjugate(q)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q. q: [w,x,y,z], v: [x,y,z]."""
    q_w, q_xyz = q[..., 0:1], q[..., 1:4]
    t = 2 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


@torch.jit.script
def quat_axis(q, axis: int = 0):
    """Rotate a basis vector by quaternion to get the corresponding body frame axis in world frame.

    Args:
        q: Quaternion [w,x,y,z]
        axis: Which basis vector to rotate (0=x, 1=y, 2=z)

    Returns:
        Rotated basis vector
    """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def quat_to_rotation_matrix(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)

    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - w*z)
    R[..., 0, 2] = 2*(x*z + w*y)

    R[..., 1, 0] = 2*(x*y + w*z)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - w*x)

    R[..., 2, 0] = 2*(x*z - w*y)
    R[..., 2, 1] = 2*(y*z + w*x)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)

    return R


def quat_from_rotation_matrix(R):
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

    mask_trace_positive = trace > 0

    s = torch.sqrt(trace[mask_trace_positive] + 1.0) * 2
    q[mask_trace_positive, 0] = 0.25 * s
    q[mask_trace_positive, 1] = (R[mask_trace_positive, 2, 1] - R[mask_trace_positive, 1, 2]) / s
    q[mask_trace_positive, 2] = (R[mask_trace_positive, 0, 2] - R[mask_trace_positive, 2, 0]) / s
    q[mask_trace_positive, 3] = (R[mask_trace_positive, 1, 0] - R[mask_trace_positive, 0, 1]) / s

    mask_x_largest = (~mask_trace_positive) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask_x_largest, 0, 0] - R[mask_x_largest, 1, 1] - R[mask_x_largest, 2, 2]) * 2
    q[mask_x_largest, 0] = (R[mask_x_largest, 2, 1] - R[mask_x_largest, 1, 2]) / s
    q[mask_x_largest, 1] = 0.25 * s
    q[mask_x_largest, 2] = (R[mask_x_largest, 0, 1] + R[mask_x_largest, 1, 0]) / s
    q[mask_x_largest, 3] = (R[mask_x_largest, 0, 2] + R[mask_x_largest, 2, 0]) / s

    mask_y_largest = (~mask_trace_positive) & (~mask_x_largest) & (R[..., 1, 1] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask_y_largest, 1, 1] - R[mask_y_largest, 0, 0] - R[mask_y_largest, 2, 2]) * 2
    q[mask_y_largest, 0] = (R[mask_y_largest, 0, 2] - R[mask_y_largest, 2, 0]) / s
    q[mask_y_largest, 1] = (R[mask_y_largest, 0, 1] + R[mask_y_largest, 1, 0]) / s
    q[mask_y_largest, 2] = 0.25 * s
    q[mask_y_largest, 3] = (R[mask_y_largest, 1, 2] + R[mask_y_largest, 2, 1]) / s

    mask_z_largest = (~mask_trace_positive) & (~mask_x_largest) & (~mask_y_largest)
    s = torch.sqrt(1.0 + R[mask_z_largest, 2, 2] - R[mask_z_largest, 0, 0] - R[mask_z_largest, 1, 1]) * 2
    q[mask_z_largest, 0] = (R[mask_z_largest, 1, 0] - R[mask_z_largest, 0, 1]) / s
    q[mask_z_largest, 1] = (R[mask_z_largest, 0, 2] + R[mask_z_largest, 2, 0]) / s
    q[mask_z_largest, 2] = (R[mask_z_largest, 1, 2] + R[mask_z_largest, 2, 1]) / s
    q[mask_z_largest, 3] = 0.25 * s

    return normalize_quat(q)


def euler_xyz_from_quat(q):
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw]."""
    R = quat_to_rotation_matrix(q)
    return matrix_to_euler_xyz(R)


def matrix_to_euler_xyz(R):
    """Extract roll, pitch, yaw from rotation matrix."""
    sy = torch.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
    singular = sy < 1e-6

    roll = torch.where(singular,
                       torch.zeros_like(sy),
                       torch.atan2(R[..., 2, 1], R[..., 2, 2]))
    pitch = torch.where(singular,
                        torch.atan2(-R[..., 2, 0], sy),
                        torch.atan2(-R[..., 2, 0], sy))
    yaw = torch.where(singular,
                      torch.atan2(-R[..., 1, 2], R[..., 1, 1]),
                      torch.atan2(R[..., 1, 0], R[..., 0, 0]))

    return torch.stack([roll, pitch, yaw], dim=-1)


def vee_map(skew_matrix):
    """Extract vector from skew-symmetric matrix."""
    return torch.stack([
        skew_matrix[..., 2, 1],
        skew_matrix[..., 0, 2],
        skew_matrix[..., 1, 0]
    ], dim=-1)


def normalize_quat(q):
    """Normalize quaternion."""
    return q / torch.norm(q, dim=-1, keepdim=True)

def quat_from_euler_xyz_tensor(euler_xyz_tensor: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles [roll, pitch, yaw] to quaternion [w, x, y, z]."""
    roll = euler_xyz_tensor[..., 0]
    pitch = euler_xyz_tensor[..., 1]
    yaw = euler_xyz_tensor[..., 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    result = torch.stack([qw, qx, qy, qz], dim=-1)
    return result


class Recorder:
    """Records viewport frames to MP4 video at specified playback speed and 60fps."""

    def __init__(
        self,
        playback_speed: float,
        mp4_path: str,
        dt: float,
        sim,
        resolution: tuple[int, int] = (1280, 720),
        camera_eye: list[float] = None,
        camera_target: list[float] = None
    ):
        """Initialize the recorder.

        Args:
            playback_speed: Playback speed multiplier (e.g., 0.25 for 4x slower)
            mp4_path: Output path for MP4 file
            dt: Simulation timestep in seconds
            sim: Simulation context for camera control
            resolution: (width, height) for rendering. Defaults to (1280, 720)
            camera_eye: Camera position [x, y, z]. Defaults to [3.5, 3.5, 3.5]
            camera_target: Camera look-at point [x, y, z]. Defaults to [0.0, 0.0, 0.0]
        """
        self.playback_speed = playback_speed
        self.mp4_path = Path(mp4_path)
        self.dt = dt
        self.sim = sim
        self.resolution = resolution
        self.camera_eye = camera_eye if camera_eye is not None else [3.5, 3.5, 3.5]
        self.camera_target = camera_target if camera_target is not None else [0.0, 0.0, 0.0]
        self.target_fps = 60
        self.sim_time_per_video_frame = (1.0 / self.target_fps) * playback_speed
        self.current_sim_time = 0.0
        self.video_writer = None
        self.initialized = False
        self.frame_count = 0
        self._rgb_annotator = None
        self.frame_postprocessor = None

    def record_frame(self):
        """Record a frame if needed based on timing. Call every simulation step."""
        expected_frame_count = int(self.current_sim_time / self.sim_time_per_video_frame)
        frames_to_write = expected_frame_count - self.frame_count

        if frames_to_write > 0:
            if not self.initialized:
                self._initialize()

            frame = self._capture_frame()
            if frame is not None and frame.size > 0:
                for _ in range(frames_to_write):
                    self.video_writer.write(frame)
                    self.frame_count += 1

        self.current_sim_time += self.dt

    def _initialize(self):
        """Initialize viewport capture."""
        import cv2
        try:
            import omni.replicator.core as rep
            import carb
            self.sim.set_camera_view(self.camera_eye, self.camera_target)

            capture_resolution = self.capture_resolution if hasattr(self, "capture_resolution") else self.resolution
            viewport_api = rep.create.render_product("/OmniverseKit_Persp", capture_resolution)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([viewport_api])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.mp4_path), fourcc, self.target_fps, self.resolution)

            self.initialized = True
            print(f"[Recorder]: Recording viewport at {capture_resolution[0]}x{capture_resolution[1]} -> {self.resolution[0]}x{self.resolution[1]} @ {self.target_fps}fps")
            print(f"[Recorder]: Camera eye={self.camera_eye}, target={self.camera_target}")
            print(f"[Recorder]: Playback speed: {self.playback_speed}x")
        except Exception as e:
            import carb
            carb.log_error(f"Failed to initialize recorder: {e}")
            self.initialized = False

    def _capture_frame(self) -> np.ndarray:
        import cv2
        """Capture frame from viewport.

        Returns:
            BGR frame as numpy array, or None if capture failed
        """
        try:
            rgb_data = self._rgb_annotator.get_data()
            if rgb_data is None or rgb_data.size == 0:
                capture_resolution = self.capture_resolution if hasattr(self, "capture_resolution") else self.resolution
                return np.zeros((capture_resolution[1], capture_resolution[0], 3), dtype=np.uint8)

            frame_rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            frame_bgr = cv2.cvtColor(frame_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
            if self.frame_postprocessor is not None:
                frame_bgr = self.frame_postprocessor(frame_bgr)
            return frame_bgr
        except Exception as e:
            import carb
            carb.log_error(f"Frame capture error: {e}")
            return None

    def save(self):
        """Release video writer and finalize the video file."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[Recorder]: Video saved to {self.mp4_path} ({self.frame_count} frames)")
            self.video_writer = None
        else:
            import carb
            carb.log_warn("No video to save - recorder was not initialized")




class RewardVisualizer:
    """Visualizes reward component contributions per episode."""
    def __init__(self, component_names, max_episodes_display=10000, update_every=1):
        matplotlib.use('TkAgg')
        self.component_names = component_names
        self.chart_data = []
        self.episode_counter = 0
        self.max_episodes_display = max_episodes_display
        self.update_every = update_every

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.ion()
        self.fig.show()

    def add_episode(self, components_dict):
        """Add episode reward components and update visualization."""
        self.episode_counter += 1

        if self.episode_counter % self.update_every == 0:
            positive = {k: v for k, v in components_dict.items() if v > 0}
            negative = {k: v for k, v in components_dict.items() if v < 0}
            self.chart_data.append({'positive': positive, 'negative': negative})
            self._update_plot()

    def _update_plot(self):
        """Update the stacked bar chart with positive/negative split."""
        self.ax.clear()

        display_data = self.chart_data[-self.max_episodes_display:]
        n_episodes = len(display_data)

        if n_episodes == 0:
            return

        colors = plt.cm.tab20(np.linspace(0, 1, len(self.component_names)))
        color_map = {name: colors[i] for i, name in enumerate(self.component_names)}
        bar_width = 0.4
        pair_spacing = 0.3
        labeled = set()

        for ep_idx, ep_data in enumerate(display_data):
            x_center = ep_idx * (2 * bar_width + pair_spacing)
            x_pos = x_center - bar_width / 2
            x_neg = x_center + bar_width / 2

            pos_bottom = 0
            for comp_name in self.component_names:
                if comp_name in ep_data['positive']:
                    height = ep_data['positive'][comp_name]
                    label = comp_name if comp_name not in labeled else ""
                    self.ax.bar(x_pos, height, bottom=pos_bottom, color=color_map[comp_name],
                              width=bar_width, label=label)
                    labeled.add(comp_name)
                    pos_bottom += height

            neg_top = pos_bottom
            for comp_name in self.component_names:
                if comp_name in ep_data['negative']:
                    height = abs(ep_data['negative'][comp_name])
                    label = comp_name if comp_name not in labeled else ""
                    self.ax.bar(x_neg, height, bottom=neg_top - height, color=color_map[comp_name],
                              width=bar_width, label=label)
                    labeled.add(comp_name)
                    neg_top -= height

        self.ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        self.ax.set_xlabel('Sample (averaged over episode length)')
        self.ax.set_ylabel('Reward Value')
        self.ax.set_title(f'Reward Components (Last {n_episodes} Samples)')
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        start_sample = max(0, len(self.chart_data) - self.max_episodes_display)
        x_positions = [i * (2 * bar_width + pair_spacing) for i in range(n_episodes)]
        tick_indices = list(range(0, n_episodes, max(1, n_episodes // 10)))
        self.ax.set_xticks([x_positions[i] for i in tick_indices])
        self.ax.set_xticklabels([str(start_sample + i) for i in tick_indices])

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        import os
        os.makedirs("results", exist_ok=True)
        self.fig.savefig("results/reward_components.png", dpi=300, bbox_inches='tight')

    def close(self):
        """Close the visualization."""
        plt.close(self.fig)


def extract_best_checkpoint(runs_dir: Path, checkpoints_dir: Path):
    """Extract best and last policies from rl-games training run.

    Args:
        runs_dir: Directory containing rl-games runs
        checkpoints_dir: Directory to save extracted policies
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = []
    for nn_dir in runs_dir.glob("morphy_position_control_*/nn"):
        if nn_dir.exists():
            checkpoint_files.extend(nn_dir.glob("*.pth"))

    if not checkpoint_files:
        print("[WARNING] No checkpoints found")
        return

    best_reward = float('-inf')
    best_checkpoint = None

    for checkpoint in checkpoint_files:
        if "_rew_" in checkpoint.name:
            try:
                reward_str = checkpoint.name.split("_rew_")[1].replace(".pth", "").strip("_")
                reward = float(reward_str)
                if reward > best_reward:
                    best_reward = reward
                    best_checkpoint = checkpoint
            except (ValueError, IndexError):
                pass

    checkpoint_files_sorted = sorted(checkpoint_files, key=lambda p: p.stat().st_mtime)
    last_checkpoint = checkpoint_files_sorted[-1]

    if best_checkpoint is None:
        best_checkpoint = last_checkpoint
        print(f"[INFO] No reward checkpoints found, using latest as best: {best_checkpoint.name}")
    else:
        print(f"[INFO] Best checkpoint: {best_checkpoint.name} (reward: {best_reward:.2f})")

    print(f"[INFO] Last checkpoint: {last_checkpoint.name}")

    best_policy_path = checkpoints_dir / "best_policy.pth"
    last_policy_path = checkpoints_dir / "last_policy.pth"
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pth"
    last_checkpoint_path = checkpoints_dir / "last_checkpoint.pth"

    extract_clean_policy(best_checkpoint, None, best_policy_path)
    extract_clean_policy(last_checkpoint, None, last_policy_path)

    import shutil
    shutil.copy2(best_checkpoint, best_checkpoint_path)
    shutil.copy2(last_checkpoint, last_checkpoint_path)

    print(f"[INFO] Extracted best policy to {best_policy_path}")
    print(f"[INFO] Extracted last policy to {last_policy_path}")
    print(f"[INFO] Saved best checkpoint to {best_checkpoint_path}")
    print(f"[INFO] Saved last checkpoint to {last_checkpoint_path}")


class CleanPolicy(torch.nn.Module):
    """Clean policy containing only essentials for inference."""

    def __init__(self, running_mean, running_var, actor_mlp, mu_layer):
        super().__init__()
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.actor_mlp = actor_mlp
        self.mu_layer = mu_layer

    def normalize_obs(self, obs):
        """Normalize observation using running statistics."""
        std = torch.sqrt(self.running_var + 1e-8)
        return (obs - self.running_mean.unsqueeze(0)) / std.unsqueeze(0)

    def forward(self, obs):
        """Forward pass through policy network."""
        normalized_obs = self.normalize_obs(obs)
        x = self.actor_mlp(normalized_obs)
        mu = self.mu_layer(x)
        return torch.clamp(mu, min=-1.0, max=1.0)


def extract_clean_policy(checkpoint_path: Path, config_path: Path, output_path: Path):
    """Extract clean policy from rl-games checkpoint.

    Args:
        checkpoint_path: Path to full rl-games checkpoint
        config_path: Path to rl-games config file
        output_path: Path to save clean policy
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['model']

    clean_state_dict = {}
    for key, value in state_dict.items():
        if 'running_mean_std.running_mean' in key:
            clean_state_dict['running_mean'] = value.float()
        elif 'running_mean_std.running_var' in key:
            clean_state_dict['running_var'] = value.float()
        elif 'a2c_network.actor_mlp' in key:
            new_key = key.replace('a2c_network.actor_mlp.', 'actor_mlp.')
            clean_state_dict[new_key] = value
        elif 'a2c_network.mu.' in key:
            new_key = key.replace('a2c_network.mu.', 'mu_layer.')
            clean_state_dict[new_key] = value

    torch.save(clean_state_dict, output_path)
    print(f"[INFO] Saved clean policy to {output_path}")


def world_to_body_frame(vec_world: torch.Tensor, quat_world: torch.Tensor) -> torch.Tensor:
    """Convert world frame vector to body frame.

    Args:
        vec_world: (n_envs, 3) vectors in world frame
        quat_world: (n_envs, 4) quaternions [w,x,y,z]

    Returns:
        vec_body: (n_envs, 3) vectors in body frame
    """
    quat_inv = quat_inverse(quat_world)
    return quat_rotate(quat_inv, vec_world)


def point_to_segment_distance(point: torch.Tensor, segment_start: torch.Tensor, segment_end: torch.Tensor) -> torch.Tensor:
    """Compute distance from point to line segment.

    Args:
        point: (n_envs, 3) current positions
        segment_start: (n_envs, 3) start of segment
        segment_end: (n_envs, 3) end of segment

    Returns:
        distances: (n_envs,) distances to segment
    """
    segment = segment_end - segment_start
    segment_length_sq = torch.sum(segment ** 2, dim=-1, keepdim=True)
    point_vec = point - segment_start
    t = torch.sum(point_vec * segment, dim=-1, keepdim=True) / torch.clamp(segment_length_sq, min=1e-8)
    t = torch.clamp(t, 0.0, 1.0)
    closest_point = segment_start + t * segment
    return torch.norm(point - closest_point, dim=-1)


def rotation_matrix_x(angles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create rotation matrices around X axis.

    Args:
        angles: (n,) rotation angles in radians
        device: torch device

    Returns:
        R: (n, 3, 3) rotation matrices
    """
    n = len(angles)
    R = torch.zeros((n, 3, 3), device=device)
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = torch.cos(angles)
    R[:, 1, 2] = -torch.sin(angles)
    R[:, 2, 1] = torch.sin(angles)
    R[:, 2, 2] = torch.cos(angles)
    return R


def rotation_matrix_y(angles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create rotation matrices around Y axis.

    Args:
        angles: (n,) rotation angles in radians
        device: torch device

    Returns:
        R: (n, 3, 3) rotation matrices
    """
    n = len(angles)
    R = torch.zeros((n, 3, 3), device=device)
    R[:, 0, 0] = torch.cos(angles)
    R[:, 0, 2] = torch.sin(angles)
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -torch.sin(angles)
    R[:, 2, 2] = torch.cos(angles)
    return R


def rotation_matrix_z(angles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create rotation matrices around Z axis.

    Args:
        angles: (n,) rotation angles in radians
        device: torch device

    Returns:
        R: (n, 3, 3) rotation matrices
    """
    n = len(angles)
    R = torch.zeros((n, 3, 3), device=device)
    R[:, 0, 0] = torch.cos(angles)
    R[:, 0, 1] = -torch.sin(angles)
    R[:, 1, 0] = torch.sin(angles)
    R[:, 1, 1] = torch.cos(angles)
    R[:, 2, 2] = 1.0
    return R



def visualize_drone_comprehensive(motor_positions, motor_thrust_directions, motor_directions, optimal_forces, thrust_to_torque_ratio=0.01, desired_wrench=None, achieved_wrench=None, mass=None, force_limits=None):
    """Comprehensive 4-panel visualization of drone configuration and torque generation.

    Args:
        motor_positions: (n_motors, 3) array of motor positions
        motor_thrust_directions: (n_motors, 3) array of thrust directions
        motor_directions: (n_motors, 1) or (n_motors,) array of motor spin directions (-1 or +1)
        optimal_forces: (n_motors,) array of motor thrust forces
        thrust_to_torque_ratio: scalar or (n_motors,) array of thrust-to-torque ratios
        desired_wrench: (6,) array of desired [fx, fy, fz, tx, ty, tz] (optional)
        achieved_wrench: (6,) array of achieved wrench (optional, computed if not provided)
        mass: total mass in kg (optional)
        force_limits: tuple of (min_force, max_force) or (min_forces_array, max_forces_array) (optional)
    """
    if motor_directions.ndim == 2:
        motor_dirs = motor_directions[:, 0]
    else:
        motor_dirs = motor_directions

    if np.isscalar(thrust_to_torque_ratio):
        thrust_to_torque_ratio = np.full(len(motor_positions), thrust_to_torque_ratio)

    thrust_norms = np.linalg.norm(motor_thrust_directions, axis=1, keepdims=True)
    thrust_dirs_normalized = motor_thrust_directions / thrust_norms

    moment_arms = np.cross(motor_positions, motor_thrust_directions)
    reactive_torques = motor_dirs[:, np.newaxis] * thrust_to_torque_ratio[:, np.newaxis] * motor_thrust_directions
    total_torques = moment_arms + reactive_torques

    if achieved_wrench is None:
        n_motors = len(motor_positions)
        B = np.zeros((6, n_motors))
        for i in range(n_motors):
            r = motor_positions[i].reshape(3, 1)
            thrust_vec = motor_thrust_directions[i].reshape(3, 1)
            force_contribution = thrust_vec
            torque_contribution = np.cross(r.flatten(), thrust_vec.flatten()).reshape(3, 1)
            yaw_torque = motor_dirs[i] * thrust_to_torque_ratio[i] * thrust_vec
            B[0:3, i] = force_contribution.flatten()
            B[3:6, i] = (torque_contribution + yaw_torque).flatten()
        achieved_wrench = B @ optimal_forces

    fig = plt.figure(figsize=(16, 18))

    colors = ['red', 'blue', 'red', 'blue']
    labels = [f'M{i} ({"CW" if motor_dirs[i] > 0 else "CCW"})' for i in range(len(motor_positions))]

    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax1.set_title('3D View - Thrust Directions', fontsize=12, fontweight='bold')

    axis_length = 0.15
    ax1.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=2, label='X (forward)')
    ax1.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.2, linewidth=2, label='Y (left)')
    ax1.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.2, linewidth=2, label='Z (up)')

    for i in range(len(motor_positions)):
        ax1.scatter(*motor_positions[i], color=colors[i], s=200, marker='o', edgecolors='black', linewidth=2)
        ax1.text(motor_positions[i, 0], motor_positions[i, 1], motor_positions[i, 2] + 0.02, labels[i], fontsize=9)

        scale = 0.15
        ax1.quiver(motor_positions[i, 0], motor_positions[i, 1], motor_positions[i, 2],
                   thrust_dirs_normalized[i, 0], thrust_dirs_normalized[i, 1], thrust_dirs_normalized[i, 2],
                   color=colors[i], arrow_length_ratio=0.3, linewidth=2, alpha=0.7, length=scale)

    ax1.set_xlabel('X (forward)')
    ax1.set_ylabel('Y (left)')
    ax1.set_zlabel('Z (up)')
    ax1.set_xlim([-0.2, 0.2])
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_zlim([-0.05, 0.2])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.view_init(elev=20, azim=45)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title('Top View - Moment Arms & Yaw Torque', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.plot(0, 0, 'ko', markersize=10, label='CoM')

    yaw_contribs = total_torques[:, 2]
    for i in range(len(motor_positions)):
        ax2.plot(motor_positions[i, 1], motor_positions[i, 0], 'o', color=colors[i], markersize=15,
                 markeredgecolor='black', markeredgewidth=2)
        ax2.text(motor_positions[i, 1] + 0.02, motor_positions[i, 0] + 0.02, labels[i], fontsize=9)

        thrust_xy = motor_thrust_directions[i, :2]
        thrust_xy_norm = thrust_xy / (np.linalg.norm(thrust_xy) + 1e-6)
        ax2.arrow(motor_positions[i, 1], motor_positions[i, 0],
                  thrust_xy_norm[1] * 0.08, thrust_xy_norm[0] * 0.08,
                  head_width=0.015, head_length=0.01, fc=colors[i], ec=colors[i], linewidth=2, alpha=0.7)

        ax2.plot([0, motor_positions[i, 1]], [0, motor_positions[i, 0]], 'k--', linewidth=1, alpha=0.3)

        sign = '+' if yaw_contribs[i] > 0 else ''
        ax2.text(motor_positions[i, 1] - 0.02, motor_positions[i, 0] - 0.05,
                 f'Tz={sign}{yaw_contribs[i]:.4f}', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax2.set_xlabel('Y (left)')
    ax2.set_ylabel('X (forward)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_xlim([-0.22, 0.22])
    ax2.set_ylim([-0.22, 0.22])

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title('Side View (X-Z) - Thrust Tilt', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.plot(0, 0, 'ko', markersize=10, label='CoM')

    for i in range(len(motor_positions)):
        ax3.plot(motor_positions[i, 0], motor_positions[i, 2], 'o', color=colors[i], markersize=15,
                 markeredgecolor='black', markeredgewidth=2)
        ax3.text(motor_positions[i, 0] + 0.02, motor_positions[i, 2] + 0.02, f'M{i}', fontsize=9)

        scale = 0.12
        ax3.arrow(motor_positions[i, 0], motor_positions[i, 2],
                  thrust_dirs_normalized[i, 0] * scale, thrust_dirs_normalized[i, 2] * scale,
                  head_width=0.015, head_length=0.01, fc=colors[i], ec=colors[i], linewidth=2, alpha=0.7)

    ax3.set_xlabel('X (forward)')
    ax3.set_ylabel('Z (up)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.set_xlim([-0.22, 0.22])
    ax3.set_ylim([-0.05, 0.2])

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title('Yaw Torque Breakdown (per 1N thrust)', fontsize=12, fontweight='bold')

    x = np.arange(len(motor_positions))
    width = 0.25

    moment_arm_yaw = moment_arms[:, 2]
    reactive_yaw = reactive_torques[:, 2]
    total_yaw = total_torques[:, 2]

    ax4.bar(x - width, moment_arm_yaw, width, label='Moment Arm', color='skyblue', edgecolor='black')
    ax4.bar(x, reactive_yaw, width, label='Reactive (motor spin)', color='orange', edgecolor='black')
    ax4.bar(x + width, total_yaw, width, label='Total', color='red', edgecolor='black', alpha=0.7)

    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Motor')
    ax4.set_ylabel('Yaw Torque (Nm/N)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'M{i} ({"CW" if motor_dirs[i] > 0 else "CCW"})' for i in range(len(motor_positions))])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set_title('Wrench Components: Desired vs Achieved', fontsize=12, fontweight='bold')

    wrench_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    x_wrench = np.arange(len(wrench_labels))
    width_wrench = 0.35

    if desired_wrench is not None and achieved_wrench is not None:
        ax5.bar(x_wrench - width_wrench/2, desired_wrench, width_wrench,
                label='Desired', color='skyblue', edgecolor='black')
        ax5.bar(x_wrench + width_wrench/2, achieved_wrench, width_wrench,
                label='Achieved', color='coral', edgecolor='black')

        wrench_error = achieved_wrench - desired_wrench
        wrench_error_norm = np.linalg.norm(wrench_error)
        max_component_error = np.abs(wrench_error).max()

        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        ax5.set_xlabel('Wrench Component')
        ax5.set_ylabel('Value (N or N⋅m)')
        ax5.set_xticks(x_wrench)
        ax5.set_xticklabels(wrench_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        error_text = f'||Error||: {wrench_error_norm:.6f}\nMax|Δ|: {max_component_error:.6f}'
        ax5.text(0.98, 0.97, error_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    elif achieved_wrench is not None:
        ax5.bar(x_wrench, achieved_wrench, width_wrench*2,
                label='Achieved', color='coral', edgecolor='black')
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        ax5.set_xlabel('Wrench Component')
        ax5.set_ylabel('Value (N or N⋅m)')
        ax5.set_xticks(x_wrench)
        ax5.set_xticklabels(wrench_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No wrench data available',
                transform=ax5.transAxes, ha='center', va='center', fontsize=12)
        ax5.set_xticks([])
        ax5.set_yticks([])

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.set_title('Motor Force Distribution', fontsize=12, fontweight='bold')

    motor_x = np.arange(len(motor_positions))
    ax6.bar(motor_x, optimal_forces, color=colors[:len(motor_positions)],
            edgecolor='black', linewidth=1.5)

    ax6.axhline(y=optimal_forces.mean(), color='green', linestyle='--',
                linewidth=2, label=f'Mean: {optimal_forces.mean():.3f}N')

    if force_limits is not None:
        min_forces, max_forces = force_limits
        if np.isscalar(min_forces):
            ax6.axhline(y=min_forces, color='red', linestyle=':', linewidth=2,
                       label=f'Min Limit: {min_forces:.3f}N')
            ax6.axhline(y=max_forces, color='red', linestyle=':', linewidth=2,
                       label=f'Max Limit: {max_forces:.3f}N')
        else:
            min_force_val = min_forces.min() if hasattr(min_forces, 'min') else min(min_forces)
            max_force_val = max_forces.max() if hasattr(max_forces, 'max') else max(max_forces)
            ax6.axhline(y=min_force_val, color='red', linestyle=':', linewidth=2,
                       label=f'Min Limit: {min_force_val:.3f}N')
            ax6.axhline(y=max_force_val, color='red', linestyle=':', linewidth=2,
                       label=f'Max Limit: {max_force_val:.3f}N')

    ax6.set_xlabel('Motor')
    ax6.set_ylabel('Force (N)')
    ax6.set_xticks(motor_x)
    ax6.set_xticklabels([f'M{i}' for i in range(len(motor_positions))])
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.legend(fontsize=8)

    plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.95)
    plt.show()


def _visualize_airframe(motor_positions, motor_thrust_directions, motor_directions, desired_thrusts, thrust_to_torque_ratio=0.01, desired_wrench=None, achieved_wrench=None, mass=None, force_limits=None):
    """Visualize airframe motors and thrust vectors using comprehensive 6-panel view."""
    visualize_drone_comprehensive(motor_positions, motor_thrust_directions, motor_directions, desired_thrusts, thrust_to_torque_ratio, desired_wrench, achieved_wrench, mass, force_limits)


def compute_misalignment_angles(quat: torch.Tensor) -> torch.Tensor:
    """Compute misalignment angles between body axes and world axes.

    Args:
        quat: (n, 4) quaternions [w,x,y,z]

    Returns:
        angles: (n, 3) tensor of [angle_x, angle_y, angle_z] in radians
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    body_x_in_world = torch.stack([
        1 - 2 * (y**2 + z**2),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=-1)

    body_y_in_world = torch.stack([
        2 * (x * y - w * z),
        1 - 2 * (x**2 + z**2),
        2 * (y * z + w * x)
    ], dim=-1)

    body_z_in_world = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)
    ], dim=-1)

    angle_x = torch.acos(torch.clamp(body_x_in_world[:, 0], -1.0, 1.0))
    angle_y = torch.acos(torch.clamp(body_y_in_world[:, 1], -1.0, 1.0))
    angle_z = torch.acos(torch.clamp(body_z_in_world[:, 2], -1.0, 1.0))

    return torch.stack([angle_x, angle_y, angle_z], dim=-1)


def render_airframe_top_view(ax, airframe, alpha=0.6):
    """Draw top-view (X-Y projection) of folded airframe geometry.

    Args:
        ax: matplotlib axis to draw on
        airframe: Airframe object
        alpha: alpha value for component fill
    """
    import math
    from airframe_encoding import AIRFRAME_CONSTANTS, ADD_ROD
    from airframe_validations import get_max_arm_angles_no_collision
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    COMPONENT_COLORS = {
        'base': '#888888', 'front_arm': '#FF8C00', 'back_arm': '#FF8C00',
        'left_seg1': '#FF8C00', 'left_seg2': '#FFD700', 'right_seg1': '#FF8C00', 'right_seg2': '#FFD700',
        'front_motor': '#00CED1', 'back_motor': '#00CED1', 'left_motor': '#32CD32', 'right_motor': '#32CD32',
        'left_rod': '#8B4513', 'right_rod': '#8B4513',
    }

    def rpy_to_rotation_matrix(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def build_geometry(left_angle_rad, right_angle_rad):
        p = airframe._params
        c = AIRFRAME_CONSTANTS
        arm_w, arm_h = c['arm_width'], c['arm_height']
        motor_r, motor_h = c['motor_radius'], c['motor_height']
        base_dims = c['base_box_dimensions']
        seg1_len = c['side_seg1_len']
        rod_radius = c['rod_radius']

        front_len = p['front_arm_length']
        back_len = p['back_arm_length']
        seg2_len_l = p['side_arm_seg2_length_left']
        seg2_len_r = p['side_arm_seg2_length_right']

        front_prop_rot = np.radians([p['front_prop_rotation_x'], p['front_prop_rotation_y'], p['front_prop_rotation_z']])
        back_prop_rot = np.radians([p['back_prop_rotation_x'], p['back_prop_rotation_y'], p['back_prop_rotation_z']])
        side_base_rot_l = np.array([math.radians(p['side_base_rotation_x_left']),
                                     math.radians(p['side_base_rotation_y_left']),
                                     math.radians(p['side_base_rotation_z_left']) + math.pi/2])
        side_base_rot_r = np.array([math.radians(p['side_base_rotation_x_right']),
                                     math.radians(p['side_base_rotation_y_right']),
                                     math.radians(p['side_base_rotation_z_right']) - math.pi/2])
        side_motor_rot_l = np.radians([p['side_motor_rotation_x_left'], p['side_motor_rotation_y_left'], p['side_motor_rotation_z_left']])
        side_motor_rot_r = np.radians([p['side_motor_rotation_x_right'], p['side_motor_rotation_y_right'], p['side_motor_rotation_z_right']])

        def compute_rod_geometry(arm_base_rot, motor_rot, seg2_len):
            R_arm = rpy_to_rotation_matrix(*arm_base_rot)
            R_motor = rpy_to_rotation_matrix(*motor_rot)
            R_combined = R_arm @ R_motor
            world_x_in_motor = R_combined.T @ np.array([1, 0, 0])
            xy_norm = np.linalg.norm(world_x_in_motor[:2])
            edge_dir = np.array([world_x_in_motor[0]/xy_norm, world_x_in_motor[1]/xy_norm, 0]) if xy_norm > 1e-6 else np.array([1, 0, 0])
            edge_point = motor_r * edge_dir
            joint_in_motor = R_motor.T @ np.array([-seg2_len, 0, 0])
            rod_vec = edge_point - joint_in_motor
            rod_len = float(np.linalg.norm(rod_vec))
            rod_center = (edge_point + joint_in_motor) / 2
            rod_dir = rod_vec / rod_len
            rod_xy = math.sqrt(rod_dir[0]**2 + rod_dir[1]**2)
            return rod_len, rod_center, math.pi/2 - math.atan2(rod_dir[2], rod_xy), math.atan2(rod_dir[1], rod_dir[0])

        rod_l = compute_rod_geometry(side_base_rot_l, side_motor_rot_l, seg2_len_l)
        rod_r = compute_rod_geometry(side_base_rot_r, side_motor_rot_r, seg2_len_r)

        geom = {}
        geom['base'] = ('box', base_dims, np.zeros(3), np.eye(3))
        geom['front_arm'] = ('box', [front_len, arm_w, arm_h], np.array([0.035 + front_len/2, 0, 0]), np.eye(3))
        geom['back_arm'] = ('box', [back_len, arm_w, arm_h], np.array([-0.035 - back_len/2, 0, 0]), np.eye(3))

        R_front = rpy_to_rotation_matrix(*front_prop_rot)
        geom['front_motor'] = ('cyl', motor_r, motor_h, np.array([0.035 + front_len, 0, 0]), R_front)

        R_back = rpy_to_rotation_matrix(0, 0, math.pi) @ rpy_to_rotation_matrix(*back_prop_rot)
        geom['back_motor'] = ('cyl', motor_r, motor_h, np.array([-0.035 - back_len, 0, 0]), R_back)

        R_l1 = rpy_to_rotation_matrix(*side_base_rot_l)
        l1_base = np.array([0, 0.025, 0])
        l1_center = l1_base + R_l1 @ np.array([seg1_len/2, 0, 0])
        geom['left_seg1'] = ('box', [seg1_len, arm_w, arm_h], l1_center, R_l1)

        l_joint = l1_base + R_l1 @ np.array([seg1_len, 0, 0])
        R_l2 = R_l1 @ rpy_to_rotation_matrix(0, 0, left_angle_rad)
        l2_center = l_joint + R_l2 @ np.array([seg2_len_l/2, 0, 0])
        geom['left_seg2'] = ('box', [seg2_len_l, arm_w, arm_h], l2_center, R_l2)

        l_motor_pos = l_joint + R_l2 @ np.array([seg2_len_l, 0, 0])
        R_l_motor = R_l2 @ rpy_to_rotation_matrix(*side_motor_rot_l)
        geom['left_motor'] = ('cyl', motor_r, motor_h, l_motor_pos, R_l_motor)

        if ADD_ROD:
            rod_l_len, rod_l_center_motor, rod_l_pitch, rod_l_yaw = rod_l
            R_rod_l = R_l_motor @ rpy_to_rotation_matrix(0, rod_l_pitch, rod_l_yaw)
            rod_l_center_world = l_motor_pos + R_l_motor @ rod_l_center_motor
            geom['left_rod'] = ('cyl', rod_radius, rod_l_len, rod_l_center_world, R_rod_l)

        R_r1 = rpy_to_rotation_matrix(*side_base_rot_r)
        r1_base = np.array([0, -0.025, 0])
        r1_center = r1_base + R_r1 @ np.array([seg1_len/2, 0, 0])
        geom['right_seg1'] = ('box', [seg1_len, arm_w, arm_h], r1_center, R_r1)

        r_joint = r1_base + R_r1 @ np.array([seg1_len, 0, 0])
        R_r2 = R_r1 @ rpy_to_rotation_matrix(0, 0, right_angle_rad)
        r2_center = r_joint + R_r2 @ np.array([seg2_len_r/2, 0, 0])
        geom['right_seg2'] = ('box', [seg2_len_r, arm_w, arm_h], r2_center, R_r2)

        r_motor_pos = r_joint + R_r2 @ np.array([seg2_len_r, 0, 0])
        R_r_motor = R_r2 @ rpy_to_rotation_matrix(*side_motor_rot_r)
        geom['right_motor'] = ('cyl', motor_r, motor_h, r_motor_pos, R_r_motor)

        if ADD_ROD:
            rod_r_len, rod_r_center_motor, rod_r_pitch, rod_r_yaw = rod_r
            R_rod_r = R_r_motor @ rpy_to_rotation_matrix(0, rod_r_pitch, rod_r_yaw)
            rod_r_center_world = r_motor_pos + R_r_motor @ rod_r_center_motor
            geom['right_rod'] = ('cyl', rod_radius, rod_r_len, rod_r_center_world, R_rod_r)

        return geom

    def project_cylinder(radius, height, center, R, plane='xy', n_circle=32):
        plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        idx1, idx2 = plane_map[plane]
        angles = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
        circle_local = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros(n_circle)])
        axis_half = np.array([0, 0, height/2])
        top_3d = np.array([center + R @ (c + axis_half) for c in circle_local])
        bot_3d = np.array([center + R @ (c - axis_half) for c in circle_local])
        all_pts = np.vstack([top_3d[:, [idx1, idx2]], bot_3d[:, [idx1, idx2]]])
        hull = ConvexHull(all_pts)
        return all_pts[hull.vertices]

    def draw_projection(geom, plane='xy'):
        plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        idx1, idx2 = plane_map[plane]

        for name, info in geom.items():
            color = COMPONENT_COLORS.get(name, '#AAAAAA')
            if info[0] == 'box':
                _, dims, center, R = info
                dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
                corners_3d = np.array([[-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
                                        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]])
                world = np.array([center + R @ c for c in corners_3d])
                proj = world[:, [idx1, idx2]] * 100
                hull = ConvexHull(proj)
                poly = Polygon(proj[hull.vertices], closed=True, facecolor=color, edgecolor='black',
                              alpha=alpha, linewidth=0.5)
                ax.add_patch(poly)
            else:
                _, radius, height, center, R = info
                hull_pts = project_cylinder(radius, height, center, R, plane=plane) * 100
                poly = Polygon(hull_pts, closed=True, facecolor=color, edgecolor='black',
                              alpha=alpha, linewidth=0.5)
                ax.add_patch(poly)
                if 'motor' in name:
                    n_c = 64
                    angles = np.linspace(0, 2*np.pi, n_c, endpoint=True)
                    circle_local = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros(n_c)])
                    axis_half = np.array([0, 0, height/2])
                    for sign in [1, -1]:
                        pts_3d = np.array([center + R @ (c + sign * axis_half) for c in circle_local])
                        pts_2d = pts_3d[:, [idx1, idx2]] * 100
                        ax.plot(pts_2d[:, 0], pts_2d[:, 1], color='black', linewidth=0.4, linestyle='-')

    max_left, max_right, aabb_info = get_max_arm_angles_no_collision(airframe, delta_deg=0.01, verbose=False)
    geom = build_geometry(math.radians(max_left), math.radians(max_right))
    draw_projection(geom, plane='xy')

    y_min, y_max = aabb_info['aabb_min'][1] * 100, aabb_info['aabb_max'][1] * 100
    ax.axhline(y=y_min, color='black', linewidth=0.25, linestyle='--', zorder=10)
    ax.axhline(y=y_max, color='black', linewidth=0.25, linestyle='--', zorder=10)
    width_cm = y_max - y_min
    ax.text(0.98, 0.02, f"{width_cm:.1f} cm", transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', color='black', zorder=100,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=1.0))

    ax.set_xlim(-18, 18)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (cm)', fontsize=10)
    ax.set_ylabel('Y (cm)', fontsize=10)


def generate_width_loss_plot_with_top_views(df, planar_csv, selected_designs, output_path):
    from airframe_encoding import Airframe

    planar_df = pd.read_csv(planar_csv)
    best_planar_width = planar_df['folded_width'].min() * 100

    config_order = []
    designs_by_config = {}
    for d in selected_designs:
        cfg_name = d['config_name']
        if cfg_name not in designs_by_config:
            config_order.append(cfg_name)
            designs_by_config[cfg_name] = []
        designs_by_config[cfg_name].append(d)

    n_rows = len(config_order)
    n_cols = max(len(v) for v in designs_by_config.values())
    config_colors = {cfg_name: plt.cm.tab10(i) for i, cfg_name in enumerate(config_order)}

    fig_height = 4 + 2.5 * n_rows
    fig = plt.figure(figsize=(12, fig_height), dpi=100)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.5, n_rows], hspace=0.4)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_views = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs_outer[1], hspace=0, wspace=0)

    ax_main = fig.add_subplot(gs_outer[0])

    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']

    for idx, (config_name, config_data) in enumerate(df.groupby('config_name')):
        loss_values = sorted(config_data['loss_value'].unique())
        medians, q25s, q75s = [], [], []
        for loss_value in loss_values:
            widths = config_data[config_data['loss_value'] == loss_value]['folded_width'].values * 100
            medians.append(np.median(widths))
            q25s.append(np.percentile(widths, 25))
            q75s.append(np.percentile(widths, 75))
        label = config_name.replace('config1_', '').replace('config2_', '').replace('config3_', '').title()
        color = plt.cm.tab10(idx)
        ax_main.fill_between(loss_values, q25s, q75s, alpha=0.2, color=color)
        ax_main.plot(loss_values, medians,
                     marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)],
                     color=color, linewidth=2.0, markersize=5, label=label, zorder=3)

    ax_main.plot(0.0, best_planar_width, marker='x', color='black', markersize=10,
                 markeredgewidth=2, label='Planar', zorder=5)
    ax_main.set_xlabel('Control Authority Loss', fontsize=12)
    ax_main.set_ylabel('Folded Width (cm)', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(fontsize=10, loc='upper right', framealpha=1.0, facecolor='white').set_zorder(100)

    from airframe_encoding import PARAM_NAMES
    x_cols = [f'x_{i}' for i in range(len(PARAM_NAMES))]
    ax_ref = None
    top_row_axes = {}  # col_idx -> (ax_view, loss_value)
    for row_idx, cfg_name in enumerate(config_order):
        designs = designs_by_config[cfg_name]
        config_label = cfg_name.replace('config1_', '').replace('config2_', '').replace('config3_', '').title()
        config_color = config_colors[cfg_name]

        for col_idx in range(n_cols):
            ax_view = fig.add_subplot(gs_views[row_idx, col_idx], sharex=ax_ref, sharey=ax_ref)
            if ax_ref is None:
                ax_ref = ax_view

            if col_idx < len(designs):
                design = designs[col_idx]
                subset = df[(df['config_name'] == design['config_name']) & (df['loss_value'] == design['loss_value'])]
                if len(subset) > 0:
                    median_width = subset['folded_width'].median()
                    best_row = subset.iloc[(subset['folded_width'] - median_width).abs().argsort().iloc[0]]
                    render_airframe_top_view(ax_view, Airframe.from_x_0_1(best_row[x_cols].values.astype(float)))
                    if row_idx == 0:
                        ax_view.set_title(design.get('label', f"Loss={design['loss_value']}          "),
                                          fontsize=9, color='black', fontweight='bold', loc="center", pad=2)
                        top_row_axes[col_idx] = (ax_view, design['loss_value'])
            else:
                ax_view.set_visible(False)
                continue

            ax_view.set_xlabel('')
            ax_view.set_ylabel('')
            is_bottom = row_idx == n_rows - 1
            ax_view.tick_params(bottom=is_bottom, labelbottom=is_bottom, top=False, labeltop=False)
            plt.setp(ax_view.get_yticklabels(), visible=(col_idx == 0))
            plt.setp(ax_view.get_yticklines(), visible=(col_idx == 0))

            if row_idx == n_rows - 1:
                ax_view.set_xlabel('X (cm)', fontsize=8)
            if col_idx == 0:
                ax_view.set_ylabel(config_label, fontsize=9, color=config_color, fontweight='bold')

    from matplotlib.patches import ConnectionPatch
    for col_idx, (ax_top, loss_value) in top_row_axes.items():
        ax_main.axvline(x=loss_value, color='black', linestyle='--', linewidth=0.8, zorder=2)
        con = ConnectionPatch(
            xyA=(loss_value, ax_main.get_ylim()[0]),
            xyB=(0.25, 1.0),
            coordsA="data", coordsB="axes fraction",
            axesA=ax_main, axesB=ax_top,
            color="black", linestyle="--", linewidth=0.8, zorder=0
        )
        fig.add_artist(con)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
