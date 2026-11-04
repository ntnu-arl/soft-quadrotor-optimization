"""
MorphyWallGapEnv: RL Environment for Morphy Wall Gap Navigation

Implements a Gym-compatible environment for training RL agents to navigate
the Morphy soft drone through a narrow gap in a wall by squeezing its soft arms.
"""

import torch
import numpy as np
import gym.spaces
import gymnasium
from typing import Optional, Dict, Any, Tuple
from rl_games.common.vecenv import IVecEnv
from morphy_simulator import MorphySimulator
from utils import euler_xyz_from_quat, quat_axis, quat_from_euler_xyz_tensor, world_to_body_frame, point_to_segment_distance, compute_misalignment_angles, normalize_quat
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from airframe_encoding import AIRFRAME_CONSTANTS

EXPECTED_REWARD_WEIGHT_KEYS = {
    'position_scale', 'angular_velocity_scale', 'orientation_scale', 'crash_scale',
    'success_scale', 'exponential_position_scale', 'exponential_position_rate',
    'segment_distance_scale', 'passage_scale', 'passage_sigma_x', 'passage_sigma_y', 'passage_max_vx'
}

OBSERVATION_NOISE_KEYS = [
    'noise_current_position_episode',
    'noise_current_position_timestep',
    'noise_velocity_episode',
    'noise_velocity_timestep',
    'noise_orientation_euler_deg_episode',
    'noise_orientation_euler_deg_timestep',
    'noise_angvel_rad_episode',
    'noise_angvel_rad_timestep',
]


def validate_reward_weights(reward_weights: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Validate reward weights.

    Args:
        reward_weights: Reward weights dict (can be None for evaluation)

    Returns:
        Validated reward weights dictionary or None

    Raises:
        ValueError: If reward_weights has missing/extra keys
    """
    if reward_weights is None:
        return None

    missing_keys = EXPECTED_REWARD_WEIGHT_KEYS - set(reward_weights.keys())
    extra_keys = set(reward_weights.keys()) - EXPECTED_REWARD_WEIGHT_KEYS

    if missing_keys:
        raise ValueError(f"Incomplete reward_weights. Missing keys: {missing_keys}")

    if extra_keys:
        raise ValueError(f"Unknown reward weight keys: {extra_keys}. Valid keys: {EXPECTED_REWARD_WEIGHT_KEYS}")

    return reward_weights


class MorphyWallGapEnv:
    """RL environment for Morphy wall gap navigation task."""

    def __init__(
        self,
        simulator: MorphySimulator,
        config: Dict[str, Any],
        reward_weights: Optional[Dict[str, float]] = None,
        track_reward_components: bool = False,
    ):
        """Initialize the wall gap RL environment.

        Args:
            simulator: MorphySimulator instance
            config: Task configuration dict (from YAML). Must contain 'environment' and 'initial_state' keys.
                   Can also contain 'reward_weights' if not specified separately.
            reward_weights: Override reward weights (if None, use config['reward_weights'])
            track_reward_components: Enable tracking of individual reward components per episode
        """
        self.sim = simulator
        self.n_envs = simulator.n_envs
        self.device = simulator.device
        self.all_env_ids = torch.arange(self.n_envs, device=self.device)

        env_cfg = config['environment']
        initial_state_cfg = config['initial_state']

        self.action_decimation = env_cfg['action_decimation']
        self.dt_rl = simulator.dt * self.action_decimation
        self.episode_length_s = env_cfg['episode_length_s']
        self.max_steps = int(self.episode_length_s / self.dt_rl)

        self.reward_weights = validate_reward_weights(reward_weights if reward_weights is not None else config.get('reward_weights'))

        self.num_obs = env_cfg['num_obs']
        self.num_actions = env_cfg['num_actions']
        self.termination_max_misalignment_x = env_cfg['termination_max_misalignment_x'] * np.pi / 180.0
        self.termination_max_misalignment_y = env_cfg['termination_max_misalignment_y'] * np.pi / 180.0
        self.termination_max_misalignment_z = env_cfg['termination_max_misalignment_z'] * np.pi / 180.0
        assert env_cfg['termination_max_misalignment_x'] < 179.999, "Termination is checked using axes_missalignment x angle, which is always <= 180; hence termination_max_misalignment_x must be < 180 degrees."
        assert env_cfg['termination_max_misalignment_y'] < 179.999, "Termination is checked using axes_misalignment y angle, which is always <= 180; hence termination_max_misalignment_y must be < 180 degrees."
        assert env_cfg['termination_max_misalignment_z'] < 179.999, "Termination is checked using axes_misalignment z angle, which is always <= 180; hence termination_max_misalignment_z must be < 180 degrees."
        self.x_close_to_zero_zone_termination_max_misalignment_x = env_cfg['x_close_to_zero_zone_termination_max_misalignment_x'] * np.pi / 180.0
        self.x_close_to_zero_zone_termination_max_misalignment_y = env_cfg['x_close_to_zero_zone_termination_max_misalignment_y'] * np.pi / 180.0
        self.x_close_to_zero_zone_termination_max_misalignment_z = env_cfg['x_close_to_zero_zone_termination_max_misalignment_z'] * np.pi / 180.0
        assert env_cfg['x_close_to_zero_zone_termination_max_misalignment_x'] < 179.999, "Termination is checked using axes_missalignment x angle, which is always <= 180; hence x_close_to_zero_zone_termination_max_misalignment_x must be < 180 degrees."
        assert env_cfg['x_close_to_zero_zone_termination_max_misalignment_y'] < 179.999, "Termination is checked using axes_misalignment y angle, which is always <= 180; hence x_close_to_zero_zone_termination_max_misalignment_y must be < 180 degrees."
        assert env_cfg['x_close_to_zero_zone_termination_max_misalignment_z'] < 179.999, "Termination is checked using axes_misalignment z angle, which is always <= 180; hence x_close_to_zero_zone_termination_max_misalignment_z must be < 180 degrees."
        self.success_target_distance = env_cfg['success_target_distance']
        self.success_target_velocity = env_cfg['success_target_velocity']
        self.max_arm_angle_success = env_cfg['max_arm_angle_success'] * np.pi / 180.0
        self.success_max_velocity_to_get_potential_reward = env_cfg['success_max_velocity_to_get_potential_reward']
        self.min_gap_width = env_cfg['min_gap_width']
        self.max_gap_width = env_cfg['max_gap_width']
        self.crash_lower_bound = torch.tensor(env_cfg['crash_lower_bound'], device=self.device)
        self.crash_upper_bound = torch.tensor(env_cfg['crash_upper_bound'], device=self.device)
        self.x_close_to_zero_zone_min = env_cfg['x_close_to_zero_zone_min']
        self.x_close_to_zero_zone_max = env_cfg['x_close_to_zero_zone_max']
        self.x_close_to_zero_zone_min_forward_velocity = env_cfg['x_close_to_zero_zone_min_forward_velocity']

        noise_cfg = config['observation_noise']
        self.observation_noise_enabled = noise_cfg['observation_noise']
        if not isinstance(self.observation_noise_enabled, bool):
            raise ValueError(f"observation_noise.observation_noise must be bool, got {self.observation_noise_enabled}")
        self.observation_noise_std = {}
        for key in OBSERVATION_NOISE_KEYS:
            values = noise_cfg[key]
            if len(values) != 3:
                raise ValueError(f"observation_noise.{key} must contain exactly 3 values, got {values}")
            self.observation_noise_std[key] = torch.tensor(values, dtype=torch.float32, device=self.device)

        self.initial_position_min = torch.tensor(initial_state_cfg['initial_position_min'], device=self.device)
        self.initial_position_max = torch.tensor(initial_state_cfg['initial_position_max'], device=self.device)
        self.initial_orientation_euler_min = torch.tensor(initial_state_cfg['initial_orientation_euler_min'], device=self.device)
        self.initial_orientation_euler_max = torch.tensor(initial_state_cfg['initial_orientation_euler_max'], device=self.device)
        self.initial_linvel_min = torch.tensor(initial_state_cfg['initial_linvel_min'], device=self.device)
        self.initial_linvel_max = torch.tensor(initial_state_cfg['initial_linvel_max'], device=self.device)
        self.initial_angvel_min = torch.tensor(initial_state_cfg['initial_angvel_min'], device=self.device)
        self.initial_angvel_max = torch.tensor(initial_state_cfg['initial_angvel_max'], device=self.device)
        self.initial_motor_action_min = torch.tensor(initial_state_cfg['initial_motor_action_min'], device=self.device)
        self.initial_motor_action_max = torch.tensor(initial_state_cfg['initial_motor_action_max'], device=self.device)
        self.goal_position_min = torch.tensor(initial_state_cfg['goal_position_min'], device=self.device)
        self.goal_position_max = torch.tensor(initial_state_cfg['goal_position_max'], device=self.device)

        self.current_step = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.target_positions = torch.zeros((self.n_envs, 3), device=self.device)
        self.spawn_positions = torch.zeros((self.n_envs, 3), device=self.device)

        self.current_distance = torch.zeros(self.n_envs, device=self.device)
        self.current_segment_distance = torch.zeros(self.n_envs, device=self.device)
        self.current_velocity_magnitude = torch.zeros(self.n_envs, device=self.device)
        self.current_axes_misalignment = torch.zeros((self.n_envs, 3), device=self.device)
        self.current_orientation_error = torch.zeros(self.n_envs, device=self.device)
        self.current_ang_vel_body = torch.zeros((self.n_envs, 3), device=self.device)
        self.current_potential = torch.zeros(self.n_envs, device=self.device)
        self.current_position = torch.zeros((self.n_envs, 3), device=self.device)

        self.previous_distance = torch.zeros(self.n_envs, device=self.device)
        self.previous_segment_distance = torch.zeros(self.n_envs, device=self.device)
        self.previous_velocity_magnitude = torch.zeros(self.n_envs, device=self.device)
        self.previous_axes_misalignment = torch.zeros((self.n_envs, 3), device=self.device)
        self.previous_orientation_error = torch.zeros(self.n_envs, device=self.device)
        self.previous_ang_vel_body = torch.zeros((self.n_envs, 3), device=self.device)
        self.previous_potential = torch.zeros(self.n_envs, device=self.device)
        self.previous_position = torch.zeros((self.n_envs, 3), device=self.device)

        self.observation_noise_episode_current_position = torch.zeros((self.n_envs, 3), device=self.device)
        self.observation_noise_episode_velocity = torch.zeros((self.n_envs, 3), device=self.device)
        self.observation_noise_episode_orientation_euler_deg = torch.zeros((self.n_envs, 3), device=self.device)
        self.observation_noise_episode_angvel_rad = torch.zeros((self.n_envs, 3), device=self.device)

        self.max_x_reached = torch.full((self.n_envs,), -float('inf'), device=self.device)

        self.track_reward_components = track_reward_components
        self.track_reward_components = True
        self.episode_reward_components = {
            'position_reward': torch.zeros(self.n_envs, device=self.device),
            'exponential_position_reward': torch.zeros(self.n_envs, device=self.device),
            'angular_velocity_penalty': torch.zeros(self.n_envs, device=self.device),
            'orientation_reward': torch.zeros(self.n_envs, device=self.device),
            'crash_penalty': torch.zeros(self.n_envs, device=self.device),
            'success_reward': torch.zeros(self.n_envs, device=self.device),
            'success_bonus': torch.zeros(self.n_envs, device=self.device),
            'segment_distance_penalty': torch.zeros(self.n_envs, device=self.device),
            'passage_reward': torch.zeros(self.n_envs, device=self.device),
            'position_completion_bonus': torch.zeros(self.n_envs, device=self.device),
            'orientation_completion_bonus': torch.zeros(self.n_envs, device=self.device),
        }
        self.reward_callback = None
        self.callback_step_counter = 0

        self.hover_action = self._compute_hover_action()

        self.gap_widths = torch.linspace(self.min_gap_width, self.max_gap_width, self.n_envs, device=self.device)

        self.wall_x_position = 0.0
        self.wall_size_x = 0.03
        self.wall_size_y = 4.0
        self.wall_size_z = 4.0

        self.spawn_walls()

        self.success_count = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.episode_count = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.auto_reset = True

        print(f"[MorphyWallGapEnv] Initialized with:")
        print(f"  - {self.n_envs} environments")
        print(f"  - Observation dim: {self.num_obs}")
        print(f"  - Action dim: {self.num_actions}")
        print(f"  - Episode length: {self.episode_length_s}s ({self.max_steps} steps)")
        print(f"  - dt_rl: {self.dt_rl}s, dt_sim: {self.sim.dt}s (decimation: {self.action_decimation})")
        print(f"  - Gap widths: {self.min_gap_width}m to {self.max_gap_width}m ({self.n_envs} unique gap widths)")
        print(f"  - Reward weights: {self.reward_weights}")
        print(f"  - Hover action: {self.hover_action}")

    def spawn_walls(self):
        """Spawn wall structures for each environment with varying gap sizes."""
        low_friction_material = RigidBodyMaterialCfg(
            static_friction=0.00,
            dynamic_friction=0.00,
            restitution=1.0,
            compliant_contact_stiffness=0.0,
            compliant_contact_damping=0.0,
        )

        box_cfg = sim_utils.CuboidCfg(
            size=(self.wall_size_x, self.wall_size_y, self.wall_size_z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.8)),
            physics_material=low_friction_material,
        )

        for env_idx in range(self.n_envs):
            gap_width = self.gap_widths[env_idx].item()
            half_gap_width = gap_width / 2.0
            env_origin = self.sim.scene.env_origins[env_idx].cpu().numpy()

            wall_center_y = 0.0
            wall_center_z = 0.5

            left_y = wall_center_y - half_gap_width - self.wall_size_y / 2.0
            right_y = wall_center_y + half_gap_width + self.wall_size_y / 2.0

            wall_x_global = env_origin[0] + self.wall_x_position

            left_translation = (wall_x_global, env_origin[1] + left_y, env_origin[2] + wall_center_z)
            right_translation = (wall_x_global, env_origin[1] + right_y, env_origin[2] + wall_center_z)

            box_cfg.func(f"/World/env_{env_idx}/gap_left", box_cfg, translation=left_translation)
            box_cfg.func(f"/World/env_{env_idx}/gap_right", box_cfg, translation=right_translation)

        print(f"[MorphyWallGapEnv] Spawned walls for {self.n_envs} environments")
        print(f"[MorphyWallGapEnv] Gap widths (first 10 envs): {self.gap_widths[:10].cpu().numpy()}")

    def get_observations(self) -> torch.Tensor:
        """Compute observations for all environments.

        Returns:
            observations: (n_envs, 16) tensor with:
                - rel_pos (3): relative position to target
                - quat (4): orientation quaternion [w,x,y,z]
                - lin_vel (3): linear velocity in world frame
                - ang_vel (3): angular velocity in body frame
                - joint_pos (2): revolute joint positions
                - joint_vel (1): joint velocities (averaged)
        """
        pos, quat, lin_vel, ang_vel_world = self.get_observed_base_state()
        joint_pos, joint_vel = self.sim.get_joint_state()

        rel_pos = pos - self.target_positions

        ang_vel_body = world_to_body_frame(ang_vel_world, quat)

        joint_vel_avg = joint_vel.mean(dim=-1, keepdim=True)

        obs = torch.cat([
            rel_pos,
            quat,
            lin_vel,
            ang_vel_body,
            joint_pos,
            joint_vel_avg,
        ], dim=-1)

        return obs

    def _sample_observation_noise(self, key: str, n_samples: int) -> torch.Tensor:
        std = self.observation_noise_std[key]
        return torch.randn((n_samples, 3), dtype=torch.float32, device=self.device) * std

    def _reset_observation_noise_episode(self, env_ids: torch.Tensor) -> None:
        if not self.observation_noise_enabled:
            self.observation_noise_episode_current_position[env_ids] = 0.0
            self.observation_noise_episode_velocity[env_ids] = 0.0
            self.observation_noise_episode_orientation_euler_deg[env_ids] = 0.0
            self.observation_noise_episode_angvel_rad[env_ids] = 0.0
            return

        n_reset = len(env_ids)
        self.observation_noise_episode_current_position[env_ids] = self._sample_observation_noise('noise_current_position_episode', n_reset)
        self.observation_noise_episode_velocity[env_ids] = self._sample_observation_noise('noise_velocity_episode', n_reset)
        self.observation_noise_episode_orientation_euler_deg[env_ids] = self._sample_observation_noise('noise_orientation_euler_deg_episode', n_reset)
        self.observation_noise_episode_angvel_rad[env_ids] = self._sample_observation_noise('noise_angvel_rad_episode', n_reset)

    def get_observed_base_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos, quat, lin_vel, ang_vel_world = self.sim.get_base_state()
        if not self.observation_noise_enabled:
            return pos, quat, lin_vel, ang_vel_world

        pos_noise = self.observation_noise_episode_current_position + self._sample_observation_noise('noise_current_position_timestep', self.n_envs)
        vel_noise = self.observation_noise_episode_velocity + self._sample_observation_noise('noise_velocity_timestep', self.n_envs)
        euler_noise_deg = self.observation_noise_episode_orientation_euler_deg + self._sample_observation_noise('noise_orientation_euler_deg_timestep', self.n_envs)
        angvel_noise = self.observation_noise_episode_angvel_rad + self._sample_observation_noise('noise_angvel_rad_timestep', self.n_envs)

        observed_pos = pos + pos_noise
        observed_lin_vel = lin_vel + vel_noise
        observed_euler = euler_xyz_from_quat(quat) + euler_noise_deg * (torch.pi / 180.0)
        observed_quat = normalize_quat(quat_from_euler_xyz_tensor(observed_euler))
        observed_ang_vel_world = ang_vel_world + angvel_noise
        return observed_pos, observed_quat, observed_lin_vel, observed_ang_vel_world

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one RL step (multiple sim steps).

        Args:
            actions: (n_envs, 4) motor commands in [-1, 1]

        Returns:
            observations: (n_envs, num_obs)
            rewards: (n_envs,)
            dones: (n_envs,)
            infos: Dictionary with auxiliary information
        """
        actions = torch.clamp(actions, -1.0, 1.0)
        actions = (actions + 1.0) / 2.0
        for _ in range(self.action_decimation):
            self.sim.set_actions(actions)
            self.sim.step()

        obs = self.get_observations()
        pos, quat, lin_vel, ang_vel_world = self.sim.get_base_state()
        self._copy_state_to_prev(self.all_env_ids)
        self._update_state(self.all_env_ids, pos, quat, lin_vel, ang_vel_world)
        dones, crashed, succeeded = self._check_termination(pos, quat, lin_vel, ang_vel_world)
        time_limit = (self.current_step + 1) >= self.max_steps

        rewards = self._compute_rewards(crashed, succeeded)

        self.current_step += 1

        self.success_count += succeeded.long()

        if self.track_reward_components:
            self.callback_step_counter += 1
            if self.callback_step_counter >= self.max_steps and self.reward_callback is not None:
                avg_components = {k: (v.sum() / self.n_envs).item() for k, v in self.episode_reward_components.items()}
                self.reward_callback(avg_components)
                for k in self.episode_reward_components.keys():
                    self.episode_reward_components[k][:] = 0.0
                self.callback_step_counter = 0

        infos = {
            'in_traversal_zone': (
                (pos[:, 0] - self.sim.scene.env_origins[:, 0] > self.x_close_to_zero_zone_min)
                & (pos[:, 0] - self.sim.scene.env_origins[:, 0] < self.x_close_to_zero_zone_max)
            ),
            'traversal_tilt_deg': torch.rad2deg(self.current_axes_misalignment[:, 2]).clone(),
            'time_outs': time_limit,
            'crashed': crashed,
            'succeeded': succeeded,
        }

        if dones.any() and self.auto_reset:
            self._handle_resets(dones)

        return obs, rewards, dones, infos

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.

        Returns:
            observations: (len(env_ids), num_obs)
        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        n_reset = len(env_ids)

        initial_positions = torch.zeros((n_reset, 7), device=self.device)
        random_pos = torch.rand((n_reset, 3), device=self.device)
        initial_positions[:, :3] = self.sim.scene.env_origins[env_ids] + self.initial_position_min + random_pos * (self.initial_position_max - self.initial_position_min)

        random_euler = torch.rand((n_reset, 3), device=self.device)
        euler_angles = self.initial_orientation_euler_min + random_euler * (self.initial_orientation_euler_max - self.initial_orientation_euler_min)
        initial_positions[:, 3:7] = quat_from_euler_xyz_tensor(euler_angles)
        self.sim.robot.write_root_pose_to_sim(initial_positions, env_ids)

        initial_velocities = torch.zeros((n_reset, 6), device=self.device)
        random_linvel = torch.rand((n_reset, 3), device=self.device)
        initial_velocities[:, :3] = self.initial_linvel_min + random_linvel * (self.initial_linvel_max - self.initial_linvel_min)
        random_angvel = torch.rand((n_reset, 3), device=self.device)
        initial_velocities[:, 3:] = self.initial_angvel_min + random_angvel * (self.initial_angvel_max - self.initial_angvel_min)
        self.sim.robot.write_root_velocity_to_sim(initial_velocities, env_ids)

        initial_joint_pos = torch.zeros((n_reset, len(self.sim.soft_joint_indices)), device=self.device)
        initial_joint_vel = torch.zeros((n_reset, len(self.sim.soft_joint_indices)), device=self.device)
        self.sim.robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel, joint_ids=self.sim.soft_joint_indices, env_ids=env_ids)

        if self.sim.motor_model != "motors_disabled":
            random_motor_action = torch.rand((n_reset, 4), device=self.device)
            initial_motor_action = self.initial_motor_action_min + random_motor_action * (
                self.initial_motor_action_max - self.initial_motor_action_min
            )
            self.sim.rotor_dynamics.reset_idx(env_ids, initial_motor_action)
            self.sim._last_motor_actions[env_ids] = initial_motor_action

        self.spawn_positions[env_ids] = 0.0
        self.spawn_positions[env_ids, 0] = initial_positions[:, 0]

        random_goal = torch.rand((n_reset, 3), device=self.device)
        goal_offset = self.goal_position_min + random_goal * (self.goal_position_max - self.goal_position_min)
        self.target_positions[env_ids] = self.sim.scene.env_origins[env_ids] + goal_offset

        self._update_state(env_ids, initial_positions[:, :3], initial_positions[:, 3:7], initial_velocities[:, :3], initial_velocities[:, 3:])
        self._copy_state_to_prev(env_ids)

        self.current_step[env_ids] = 0
        self.max_x_reached[env_ids] = -float('inf')
        self._reset_observation_noise_episode(env_ids)

        return self.get_observations()

    def _compute_rewards(self, crashed: torch.Tensor, succeeded: torch.Tensor) -> torch.Tensor:
        """Compute rewards from pre-computed state attributes.

        Args:
            crashed: (n_envs,) crash flags (excessive tilt or out of bounds)
            succeeded: (n_envs,) success flags (reached target with low velocity)

        Returns:
            rewards: (n_envs,)
        """
        if self.reward_weights is None:
            return torch.zeros(self.n_envs, device=self.device)
        position_reward = self.reward_weights['position_scale'] * (self.previous_distance - self.current_distance)
        exponential_position_reward = self.reward_weights['exponential_position_scale'] * torch.exp(-self.reward_weights['exponential_position_rate'] * self.current_distance)
        segment_distance_penalty = self.reward_weights['segment_distance_scale'] * (self.previous_segment_distance - self.current_segment_distance)
        ang_vel_penalty = -self.reward_weights['angular_velocity_scale'] * torch.sum(torch.abs(self.current_ang_vel_body), dim=-1)
        orientation_reward = self.reward_weights['orientation_scale'] * (self.previous_orientation_error - self.current_orientation_error)
        crash_penalty = torch.where(crashed, -self.reward_weights['crash_scale'], torch.zeros_like(crashed, dtype=torch.float32))
        success_reward_value = self.reward_weights['success_scale'] * (self.current_potential - self.previous_potential)
        success_bonus = torch.where(succeeded, self.reward_weights['success_scale'] * 3.0, torch.zeros_like(succeeded, dtype=torch.float32))
        position_completion_bonus = torch.where(succeeded, self.reward_weights['position_scale'] * self.current_distance, torch.zeros_like(succeeded, dtype=torch.float32))
        orientation_completion_bonus = torch.where(succeeded, self.reward_weights['orientation_scale'] * self.current_orientation_error, torch.zeros_like(succeeded, dtype=torch.float32))
        passage_reward = self._compute_passage_reward()

        if self.track_reward_components:
            self.episode_reward_components['position_reward'] += position_reward
            self.episode_reward_components['exponential_position_reward'] += exponential_position_reward
            self.episode_reward_components['segment_distance_penalty'] += segment_distance_penalty
            self.episode_reward_components['angular_velocity_penalty'] += ang_vel_penalty
            self.episode_reward_components['orientation_reward'] += orientation_reward
            self.episode_reward_components['crash_penalty'] += crash_penalty
            self.episode_reward_components['success_reward'] += success_reward_value
            self.episode_reward_components['success_bonus'] += success_bonus
            self.episode_reward_components['passage_reward'] += passage_reward
            self.episode_reward_components['position_completion_bonus'] += position_completion_bonus
            self.episode_reward_components['orientation_completion_bonus'] += orientation_completion_bonus
        rewards = position_reward + exponential_position_reward + segment_distance_penalty + ang_vel_penalty + orientation_reward + crash_penalty + success_reward_value + success_bonus + passage_reward + position_completion_bonus + orientation_completion_bonus

        return rewards

    def _compute_passage_reward(self) -> torch.Tensor:
        """Compute passage reward for forward velocity near wall gap center.

        Encourages the drone to pass through the wall gap center with forward velocity.
        Reward is depleted (only given at new maximum x positions) to prevent exploitation.

        Returns:
            passage_reward: (n_envs,) passage reward values
        """
        pos_rel_to_env = self.current_position - self.sim.scene.env_origins
        prev_pos_rel_to_env = self.previous_position - self.sim.scene.env_origins


        wall_center_x = self.wall_x_position
        wall_center_y = 0.0

        distance_x = torch.abs(pos_rel_to_env[:, 0] - wall_center_x)
        distance_y = torch.abs(pos_rel_to_env[:, 1] - wall_center_y)

        sigma_x = self.reward_weights['passage_sigma_x']
        sigma_y = self.reward_weights['passage_sigma_y']
        proximity_weight = torch.where(
            pos_rel_to_env[:, 0] > 0,
            torch.zeros_like(distance_x),
            torch.exp(
            -distance_x**2 / (2 * sigma_x**2)
            - distance_y**2 / (2 * sigma_y**2)
            )
        )
        # plot of proximity weight -> https://chatgpt.com/s/t_691dabd0aab88191b5fcc9567863b116

        v_x = self.sim.robot.data.root_lin_vel_w[:, 0]
        v_x_forward = torch.clamp(v_x, min=0.0, max=self.reward_weights['passage_max_vx'])

        delta_x = pos_rel_to_env[:, 0] - prev_pos_rel_to_env[:, 0]
        delta_x_forward = torch.clamp(delta_x, min=0.0)

        passage_reward_increment = self.reward_weights['passage_scale'] * proximity_weight * v_x_forward * delta_x_forward

        at_new_max_x = pos_rel_to_env[:, 0] > self.max_x_reached
        passage_reward = torch.where(
            at_new_max_x,
            passage_reward_increment,
            torch.zeros_like(passage_reward_increment)
        )

        self.max_x_reached = torch.maximum(self.max_x_reached, pos_rel_to_env[:, 0])

        return passage_reward

    def _copy_state_to_prev(self, env_ids: torch.Tensor):
        """Copy current state to previous state for specified environments.

        Args:
            env_ids: Indices of environments to copy.
        """
        self.previous_distance[env_ids] = self.current_distance[env_ids].clone()
        self.previous_segment_distance[env_ids] = self.current_segment_distance[env_ids].clone()
        self.previous_velocity_magnitude[env_ids] = self.current_velocity_magnitude[env_ids].clone()
        self.previous_axes_misalignment[env_ids] = self.current_axes_misalignment[env_ids].clone()
        self.previous_orientation_error[env_ids] = self.current_orientation_error[env_ids].clone()
        self.previous_ang_vel_body[env_ids] = self.current_ang_vel_body[env_ids].clone()
        self.previous_potential[env_ids] = self.current_potential[env_ids].clone()
        self.previous_position[env_ids] = self.current_position[env_ids].clone()

    def _update_state(self,  env_ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, lin_vel: torch.Tensor, ang_vel_world: torch.Tensor,):
        """Compute and update current state for specified environments.

        Args:
            pos: (n_envs, 3) or (n_reset, 3) current positions
            quat: (n_envs, 4) or (n_reset, 4) current orientations
            lin_vel: (n_envs, 3) or (n_reset, 3) current linear velocities
            ang_vel_world: (n_envs, 3) or (n_reset, 3) current angular velocities in world frame
            env_ids: Indices of environments to update.
        """
        self.current_position[env_ids] = pos
        self.current_distance[env_ids] = torch.norm(pos - self.target_positions[env_ids], dim=-1)
        self.current_segment_distance[env_ids] = point_to_segment_distance(pos, self.spawn_positions[env_ids], self.target_positions[env_ids])
        self.current_velocity_magnitude[env_ids] = torch.norm(lin_vel, dim=-1)
        self.compute_misalignment_axes_and_orientation_error(env_ids, quat)
        self.current_ang_vel_body[env_ids] = world_to_body_frame(ang_vel_world, quat)
        pos_rel_to_env = pos - self.sim.scene.env_origins[env_ids]
        self.current_potential[env_ids] = self._compute_success_potential(self.current_velocity_magnitude[env_ids], pos_rel_to_env[:, 0])


    def _compute_success_potential(self, velocity_magnitude: torch.Tensor, x_position_relative: torch.Tensor) -> torch.Tensor:
        """Compute success potential: rewards for achieving low velocity when past the wall.

        Uses quadratic shaping for steeper gradient near target velocity.

        Args:
            velocity_magnitude: (n,) velocity magnitudes
            x_position_relative: (n,) x positions relative to environment origin

        Returns:
            potential: (n,) potential values (only non-zero when past success threshold)
        """
        in_success_zone = x_position_relative > self.success_target_distance
        v_range = self.success_max_velocity_to_get_potential_reward - self.success_target_velocity
        normalized_excess = torch.clamp((velocity_magnitude - self.success_target_velocity) / v_range, 0.0, 1.0)
        velocity_factor = (1.0 - normalized_excess) ** 2
        return torch.where(in_success_zone, velocity_factor, torch.zeros_like(velocity_factor))

    def _check_termination(self, pos: torch.Tensor, quat: torch.Tensor, lin_vel: torch.Tensor, ang_vel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check termination conditions for all environments.

        Returns:
            dones: (n_envs,) boolean tensor indicating any termination
            crashed: (n_envs,) boolean tensor indicating crash-specific termination
            succeeded: (n_envs,) boolean tensor indicating success
        """

        pos_rel_to_env = pos - self.sim.scene.env_origins
        x_rel = pos_rel_to_env[:, 0]
        y_rel = pos_rel_to_env[:, 1]
        z_rel = pos_rel_to_env[:, 2]
        unhealthy_orientation = (self.current_axes_misalignment[:, 0] > self.termination_max_misalignment_x) | ((self.current_axes_misalignment[:, 1] > self.termination_max_misalignment_y) & (self.current_axes_misalignment[:, 2] > self.termination_max_misalignment_z))
        in_x_close_to_zero_zone = (x_rel > self.x_close_to_zero_zone_min) & (x_rel < self.x_close_to_zero_zone_max)
        unhealthy_orientation_in_x_close_to_zero_zone = in_x_close_to_zero_zone & (
            (self.current_axes_misalignment[:, 0] > self.x_close_to_zero_zone_termination_max_misalignment_x)
            | (
                (self.current_axes_misalignment[:, 1] > self.x_close_to_zero_zone_termination_max_misalignment_y)
                & (self.current_axes_misalignment[:, 2] > self.x_close_to_zero_zone_termination_max_misalignment_z)
            )
        )

        wall_x_min = self.wall_x_position - self.wall_size_x / 2.0
        wall_x_max = self.wall_x_position + self.wall_size_x / 2.0
        near_wall = (x_rel >= wall_x_min) & (x_rel <= wall_x_max)

        y_min = -self.wall_size_y / 2.0
        y_max = self.wall_size_y / 2.0
        z_min = -self.wall_size_z / 2.0
        z_max = self.wall_size_z / 2.0

        outside_gap_y = (y_rel < y_min) | (y_rel > y_max)
        outside_gap_z = (z_rel < z_min) | (z_rel > z_max)
        blocked_by_wall = near_wall & (outside_gap_y | outside_gap_z)
        outside_crash_bounds = ((pos_rel_to_env < self.crash_lower_bound) | (pos_rel_to_env > self.crash_upper_bound)).any(dim=-1)
        below_min_forward_velocity = lin_vel[:, 0] < self.x_close_to_zero_zone_min_forward_velocity
        low_forward_velocity_in_zone = in_x_close_to_zero_zone & below_min_forward_velocity

        velocity_magnitude = torch.norm(lin_vel, dim=-1)
        joint_pos, _ = self.sim.get_joint_state()
        arm_angles_within_success = (torch.abs(joint_pos) < self.max_arm_angle_success).all(dim=-1)
        succeeded = (
            (x_rel > self.success_target_distance)
            & (velocity_magnitude < self.success_target_velocity)
            & arm_angles_within_success
        )

        time_limit = (self.current_step + 1) >= self.max_steps

        crashed = unhealthy_orientation | unhealthy_orientation_in_x_close_to_zero_zone | blocked_by_wall | outside_crash_bounds | low_forward_velocity_in_zone
        dones = crashed | time_limit | succeeded

        return dones, crashed, succeeded

    def _handle_resets(self, dones: torch.Tensor):
        env_ids = torch.where(dones)[0]
        if len(env_ids) > 0:
            self.episode_count[env_ids] += 1
            self.reset(env_ids)

    def _compute_hover_action(self) -> torch.Tensor:
        """Compute the action that generates hover (0,0,g) acceleration and (0,0,0) angular acceleration.

        Returns:
            hover_action: (4,) normalized actions [0-1] for hovering
        """
        from control_allocator import ControlAllocator

        mass = self.sim.robot_mass
        gravity_magnitude = abs(self.sim.gravity[2])

        motor_idx_list = AIRFRAME_CONSTANTS['motor_idx_list']
        allocator = ControlAllocator(
            num_envs=1,
            dt=self.sim.dt,
            motor_idx_list=motor_idx_list,
            device=self.device,
            motor_directions=self.sim.motor_directions_list,
            motor_model="instantaneous_forces"
        )

        motor_positions = self.sim.get_motor_positions_body_frame_for_allocation_matrix()[0:1]
        motor_thrust_directions = self.sim.get_motor_thrust_directions_body_frame_for_allocation_matrix()[0:1]
        allocator.update_allocation_matrix(motor_positions, motor_thrust_directions)

        hover_wrench = torch.zeros((1, 6), device=self.device)
        hover_wrench[0, 2] = mass * gravity_magnitude

        motor_forces = allocator.allocate(hover_wrench)
        hover_action = allocator.motor_dynamics.force_to_action(motor_forces)[0]
        return hover_action
    
    def _compute_lee_controller_action(self) -> torch.Tensor:
        pass

    def compute_misalignment_axes_and_orientation_error(self, env_ids: torch.Tensor, quat: torch.Tensor) -> None:
        """Compute orientation error metric from quaternion.

        Measures deviation of body frame from world frame by computing angles between
        each body axis and its corresponding world axis, then returns the maximum
        normalized squared angle.

        Args:
            quat: (n, 4) quaternions [w,x,y,z]

        Returns:
            orientation_error: (n,) scalar error metric (max squared normalized angle)
        """
        self.current_axes_misalignment[env_ids] = compute_misalignment_angles(quat)
        normalized_x = (self.current_axes_misalignment[env_ids, 0] / self.termination_max_misalignment_x)**2
        normalized_y = (self.current_axes_misalignment[env_ids, 1] / self.termination_max_misalignment_y)**2
        normalized_z = (self.current_axes_misalignment[env_ids, 2] / self.termination_max_misalignment_z)**2
        self.current_orientation_error[env_ids] = torch.stack([normalized_x, normalized_y, normalized_z], dim=1).max(dim=1).values

    def get_gap_statistics(self, episode_counts: Optional[torch.Tensor] = None, success_counts: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Get success statistics per gap width.

        Args:
            episode_counts: Optional (n_envs,) tensor of episode counts per environment. If None, use internal tracking.
            success_counts: Optional (n_envs,) tensor of success counts per environment. If None, use internal tracking.

        Returns:
            Dictionary with gap width statistics
        """
        ep_counts = episode_counts if episode_counts is not None else self.episode_count
        succ_counts = success_counts if success_counts is not None else self.success_count

        unique_gaps = torch.unique(self.gap_widths)
        stats = []

        for gap in unique_gaps:
            mask = self.gap_widths == gap
            total_episodes = ep_counts[mask].sum().item()
            total_successes = succ_counts[mask].sum().item()
            success_rate = total_successes / max(1, total_episodes)

            stats.append({
                'gap_width': gap.item(),
                'num_episodes': total_episodes,
                'num_successes': total_successes,
                'success_rate': success_rate,
            })

        stats.sort(key=lambda x: x['gap_width'])

        smallest_100_percent = None
        for stat in stats:
            if stat['success_rate'] >= 1.0 and stat['num_episodes'] > 0:
                smallest_100_percent = stat['gap_width']
                break

        largest_not_100_percent = 0.0
        for stat in reversed(stats):
            if stat['success_rate'] < 1.0 and stat['num_episodes'] > 0:
                largest_not_100_percent = stat['gap_width']
                break

        return {
            'per_gap_stats': stats,
            'smallest_100_percent_gap': smallest_100_percent,
            'largest_not_100_percent_gap': largest_not_100_percent,
        }


class VecEnvRLGamesWrapper(IVecEnv):
    """Wrapper to make MorphyWallGapEnv compatible with rl-games VecEnv interface."""

    def __init__(self, env: MorphyWallGapEnv, clip_obs: float = 100.0, clip_actions: float = 1.0):
        self.env = env
        self.num_envs = env.n_envs
        self.device = env.device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions

        self._observation_space = gym.spaces.Box(
            low=-clip_obs, high=clip_obs, shape=(env.num_obs,), dtype=np.float32
        )
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.num_actions,), dtype=np.float32
        )

    @property
    def observation_space(self) -> gymnasium.Space:
        """Return observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Return action space."""
        return self._action_space

    @property
    def state_space(self) -> gymnasium.Space:
        """Return state space (empty for symmetric actor-critic)."""
        return gym.spaces.Box(low=-self._clip_obs, high=self._clip_obs, shape=(0,), dtype=np.float32)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step all environments."""
        obs, rewards, dones, infos = self.env.step(actions)
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        return {"obs": obs}, rewards, dones, infos

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments."""
        obs = self.env.reset()
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        return {"obs": obs}

    def get_number_of_agents(self) -> int:
        """Return number of agents (single-agent, returns 1)."""
        return 1

    def get_env_info(self) -> Dict[str, Any]:
        """Return environment information required by rl-games."""
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'state_space': self.state_space,
        }
