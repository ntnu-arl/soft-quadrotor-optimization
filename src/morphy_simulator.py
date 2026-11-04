"""
MorphySimulator: Clean interface for Morphy soft drone simulation.

This class provides direct control over the Morphy drone simulation without
high-level configuration abstractions. It allows direct application of:
- Forces and torques to motor bodies
- Torques to soft joints
"""

from typing import Optional
import torch
import matplotlib.pyplot as plt
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.utils import configclass


from airframe_encoding import AIRFRAME_CONSTANTS

@configclass
class MorphySceneCfg(InteractiveSceneCfg):
    """Configuration for Morphy scene."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=None
            )
        ),
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
                effort_limit=20000.0,
                velocity_limit=10000.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                armature=0.0,
            )
        },
    )


class MorphySimulator:
    _shared_sim_context = None

    def __init__(
        self,
        usd_path: str,
        n_envs: int = 1,
        dt: float = 0.001,
        device: str = "cuda:0",
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        motor_model: str = "motors_disabled",
        motor_directions: list[int] = None,
        revolute_joint_damping_coeff: float = None,
        env_spacing: float = 2.0,
        fix_base: bool = False,
    ):
        self.usd_path = usd_path
        self.n_envs = n_envs
        self.dt = dt
        self.device = device
        self.gravity = gravity
        self.motor_model = motor_model
        self.motor_directions_list = AIRFRAME_CONSTANTS['motor_directions']
        self.damping = revolute_joint_damping_coeff
        self.env_spacing = env_spacing
        self.fix_base = fix_base
        self.track_joint_dynamics = True
        self._joint_track = None

        self._setup_simulation()
        self._detect_rigid_arms()
        if self.has_soft_joints:
            assert revolute_joint_damping_coeff is not None, "revolute_joint_damping_coeff must be provided for soft airframes"
        self._load_robot()
        self._load_drone_parameters()
        if self.has_soft_joints:
            self._build_joint_torque_functions()
        self._initialize_rotor_dynamics()
        self._compute_robot_mass()
        self._cache_zero_arm_allocation_geometry()
        print(f"[MorphySimulator] Initialized with {n_envs} environments, dt={dt}s, gravity={gravity}, motor_model={motor_model}, has_soft_joints={self.has_soft_joints}")

    def _setup_simulation(self):
        if MorphySimulator._shared_sim_context is None:
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.dt,
                device=self.device,
                gravity=self.gravity,
                physx=PhysxCfg(enable_stabilization=False,
                # min_position_iteration_count=8,
                # max_position_iteration_count=8,
                # min_velocity_iteration_count=4,
                # max_velocity_iteration_count=4,
                ),
            )
            MorphySimulator._shared_sim_context = SimulationContext(sim_cfg)
            MorphySimulator._shared_sim_context.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
            print("[MorphySimulator] Created new shared SimulationContext")
        else:
            print("[MorphySimulator] Reusing existing shared SimulationContext")

        self.sim = MorphySimulator._shared_sim_context

    def _detect_rigid_arms(self):
        """Detect whether the USD has rigid arms by reading the morphy:rigid_arms attribute."""
        from pxr import Usd
        stage = Usd.Stage.Open(self.usd_path)
        root_prim = stage.GetPrimAtPath("/quadrotor")
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim '/quadrotor' not found in {self.usd_path}")
        rigid_attr = root_prim.GetAttribute("morphy:rigid_arms")
        if not rigid_attr.IsValid() or rigid_attr.Get() is None:
            raise RuntimeError(f"Failed to load morphy:rigid_arms from {self.usd_path}.")
        is_rigid = rigid_attr.Get() is True
        self.has_soft_joints = not is_rigid
        print(f"[MorphySimulator] Detected rigid_arms={is_rigid}, has_soft_joints={self.has_soft_joints}")

    def _load_robot(self):
        print(f"[MorphySimulator] num_envs: {self.n_envs}, env_spacing: {self.env_spacing}")
        scene_cfg = MorphySceneCfg(num_envs=self.n_envs, env_spacing=self.env_spacing, replicate_physics=False)
        scene_cfg.robot.spawn.usd_path = self.usd_path
        if not self.has_soft_joints:
            scene_cfg.robot.actuators = {}
            scene_cfg.robot.init_state.joint_pos = {}
            scene_cfg.robot.init_state.joint_vel = {}
        if self.fix_base:
            scene_cfg.robot.spawn.articulation_props.fix_root_link = True
            print("[MorphySimulator] Fixing base using ArticulationRootPropertiesCfg.fix_root_link")
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["robot"]
        self.sim.reset()
        if self.has_soft_joints:
            soft_joint_names = [
                "arm_segment_1_to_arm_segment_2_1",
                "arm_segment_1_to_arm_segment_2_3",
            ]
            self.soft_joint_indices = [self.robot.joint_names.index(name) for name in soft_joint_names]
            print(f"[MorphySimulator] Soft joint indices: {self.soft_joint_indices}")
        else:
            self.soft_joint_indices = []
            print("[MorphySimulator] No soft joints (rigid airframe)")
        motor_body_names = ["motor_0", "motor_1", "motor_2", "motor_3"]
        self.motor_body_indices = [self.robot.body_names.index(name) for name in motor_body_names]
        print(f"[MorphySimulator] Loaded {self.n_envs} robot(s) from {self.usd_path}")
        if self.fix_base:
            print(f"[MorphySimulator] Base is fixed: {self.robot.is_fixed_base}")

    def _load_drone_parameters(self):
        """Load drone-specific torque parameters from USD custom attributes."""
        if not self.has_soft_joints:
            self.torque_params = {}
            print("[MorphySimulator] Rigid airframe - skipping torque parameter loading")
            return

        from pxr import Usd

        stage = Usd.Stage.Open(self.usd_path)
        root_prim = stage.GetPrimAtPath("/quadrotor")
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim '/quadrotor' not found in {self.usd_path}")

        param_names = ['torque_fn_left', 'torque_fn_right']
        self.torque_params = {}
        for name in param_names:
            attr = root_prim.GetAttribute(f"morphy:{name}")
            if not attr.IsValid() or attr.Get() is None:
                raise RuntimeError(f"Failed to load morphy:{name} from {self.usd_path}.")
            self.torque_params[name] = str(attr.Get())

        print(f"[MorphySimulator] Loaded drone torque parameters from {self.usd_path}:")
        for name, value in self.torque_params.items():
            print(f"  {name}: {value}")

    def _build_joint_torque_functions(self):
        """Build separate torque functions for left and right arms from USD strings."""
        p = self.torque_params
        fn_str_left = p['torque_fn_left']
        fn_str_right = p['torque_fn_right']

        print("[MorphySimulator] Evaluating torque functions:")
        print(f"  Left:  {fn_str_left}")
        print(f"  Right: {fn_str_right}")

        self.torque_fn_left = self._execute_torque_function_string(fn_str_left, "left")
        self.torque_fn_right = self._execute_torque_function_string(fn_str_right, "right")

        angles_left = self._diagnostic_torque_plot_angles(self.torque_fn_left, "left")
        angles_right = self._diagnostic_torque_plot_angles(self.torque_fn_right, "right")
        with torch.no_grad():
            torques_left = self.torque_fn_left(angles_left)
            torques_right = self.torque_fn_right(angles_right)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(angles_left.cpu().numpy() * 180 / torch.pi, torques_left.cpu().numpy(), 'b-', linewidth=2)
        ax1.axvline(x=0.0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Joint Angle (degrees)')
        ax1.set_ylabel('Torque (N*m)')
        ax1.set_title('Left Arm Torque Response')
        ax1.grid(True, alpha=0.3)

        ax2.plot(angles_right.cpu().numpy() * 180 / torch.pi, torques_right.cpu().numpy(), 'r-', linewidth=2)
        ax2.axvline(x=0.0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Joint Angle (degrees)')
        ax2.set_ylabel('Torque (N*m)')
        ax2.set_title('Right Arm Torque Response')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        import os
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/torque_response.jpeg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[MorphySimulator] Built separate left/right torque functions, damping={self.damping}")

    def _diagnostic_torque_plot_angles(self, torque_fn, side: str) -> torch.Tensor:
        if hasattr(torque_fn, "erc_angle_min_rad") and hasattr(torque_fn, "erc_angle_max_rad"):
            angle_min = float(torque_fn.erc_angle_min_rad)
            angle_max = float(torque_fn.erc_angle_max_rad)
        elif side == "left":
            angle_min = -20.0 * float(torch.pi) / 180.0
            angle_max = 120.0 * float(torch.pi) / 180.0
        elif side == "right":
            angle_min = -120.0 * float(torch.pi) / 180.0
            angle_max = 20.0 * float(torch.pi) / 180.0
        else:
            raise ValueError(f"Unexpected soft-joint side: {side}")
        if angle_min >= angle_max:
            raise ValueError(f"Invalid diagnostic torque plot domain for {side}: {angle_min} >= {angle_max}")
        return torch.linspace(angle_min, angle_max, 1000, device=self.device)

    def _execute_torque_function_string(self, fn_str: str, side: str):
        namespace = {"torch": torch}
        exec(fn_str, namespace)
        fn_name = f"torque_fn_{side}"
        if fn_name not in namespace:
            raise RuntimeError(f"Torque function string did not define {fn_name}")
        torque_fn = namespace[fn_name]
        if not callable(torque_fn):
            raise RuntimeError(f"{fn_name} is not callable")
        return torque_fn

    def _initialize_rotor_dynamics(self):
        if self.motor_model == "motors_disabled":
            self.rotor_dynamics = None
            self.motor_directions = None
            print("[MorphySimulator] Motors disabled")
        else:
            from rotor_dynamics import RotorDynamics
            motor_idx_list = AIRFRAME_CONSTANTS['motor_idx_list']
            self.rotor_dynamics = RotorDynamics(
                n_envs=self.n_envs,
                motor_idx_list=motor_idx_list,
                dt=self.dt,
                device=self.device,
                motor_model=self.motor_model
            )
            if self.motor_directions_list is None:
                raise ValueError("motor_directions must be provided when motor_model is not 'motors_disabled'")
            self.motor_directions = torch.tensor(self.motor_directions_list, device=self.device)
            self._last_motor_actions = torch.zeros(self.n_envs, 4, device=self.device)
            print(f"[MorphySimulator] Initialized rotor dynamics with motor_model={self.motor_model}, motor_bodies={self.motor_body_indices}")

    def step(self, extra_joint_torques: Optional[torch.Tensor] = None):
        """Advance one simulation step.

        Args:
            extra_joint_torques: Optional ``(n_envs, n_soft_joints)`` tensor of
                additional torques to add on top of the ERC profile torques.
        """
        if self.has_soft_joints:
            joint_pos, joint_vel = self.get_joint_state()
            interp, damp = self.compute_joint_torques_decomposed(joint_pos, joint_vel)
            if self.track_joint_dynamics:
                self._record_joint_data(joint_pos, joint_vel, interp, damp)
            torques = interp + damp
            if extra_joint_torques is not None:
                torques = torques + extra_joint_torques
            env_indices = torch.arange(self.n_envs, dtype=torch.long, device=self.device)
            self.robot.set_joint_effort_target(torques, joint_ids=self.soft_joint_indices, env_ids=env_indices)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        self.robot.reset(env_ids)
        print(f"[MorphySimulator] Reset {len(env_ids)} environment(s)")

    def get_base_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get base (root) state.

        Returns:
            position: (n_envs, 3) in world frame
            orientation: (n_envs, 4) quaternion (w, x, y, z) in world frame
            linear_vel: (n_envs, 3) in world frame
            angular_vel: (n_envs, 3) in world frame
        """
        return (
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w
        )

    def get_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get soft joint state (2 joints only).

        Returns:
            joint_pos: (n_envs, 2) soft joint positions in radians, or (n_envs, 0) if rigid
            joint_vel: (n_envs, 2) soft joint velocities in rad/s, or (n_envs, 0) if rigid
        """
        if not self.has_soft_joints:
            return torch.zeros(self.n_envs, 0, device=self.device), torch.zeros(self.n_envs, 0, device=self.device)
        return self.robot.data.joint_pos[:, self.soft_joint_indices], self.robot.data.joint_vel[:, self.soft_joint_indices]

    def get_motor_forces(self) -> torch.Tensor:
        """Get actual motor forces in Newtons (accounting for motor dynamics).

        Returns:
            forces: (n_envs, 4) actual motor thrust forces in Newtons
        """
        forces, _ = self.rotor_dynamics._get_current_forces()
        return forces

    def get_body_state(self, body_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get state of specific bodies.

        Args:
            body_names: List of body names to query

        Returns:
            positions: (n_envs, n_bodies, 3) positions in world frame
            orientations: (n_envs, n_bodies, 4) quaternions (w, x, y, z) in world frame
        """
        body_indices = [self.robot.body_names.index(name) for name in body_names]
        positions = self.robot.data.body_pos_w[:, body_indices, :]
        orientations = self.robot.data.body_quat_w[:, body_indices, :]
        return positions, orientations

    def compute_joint_torques(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Compute soft joint torques using separate left/right functions and damping.

        Args:
            joint_pos: (n_envs, 2) soft joint positions in radians [left, right]
            joint_vel: (n_envs, 2) soft joint velocities in rad/s

        Returns:
            torques: (n_envs, 2) computed joint torques
        """
        interpolated, damping = self.compute_joint_torques_decomposed(joint_pos, joint_vel)
        return interpolated + damping

    def compute_joint_torques_decomposed(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft joint torques, returning interpolated and damping components separately.

        Args:
            joint_pos: (n_envs, 2) soft joint positions in radians [left, right]
            joint_vel: (n_envs, 2) soft joint velocities in rad/s

        Returns:
            interpolated_torque: (n_envs, 2) torque from runtime function (position-based)
            damping_torque: (n_envs, 2) torque from damping (velocity-based)
        """
        left_torque = self.torque_fn_left(joint_pos[:, 0:1])
        right_torque = self.torque_fn_right(joint_pos[:, 1:2])
        interpolated_torque = torch.cat([left_torque, right_torque], dim=1)
        damping_torque = -self.damping * joint_vel
        return interpolated_torque, damping_torque

    def set_motor_rps(self, desired_rps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set desired motor RPS and get current state.

        Args:
            desired_rps: (n_envs, 4) desired rotor speeds in revolutions per second

        Returns:
            current_rps: (n_envs, 4) current motor RPS
            forces: (n_envs, 4) thrust forces in Newtons
            torques: (n_envs, 4) reactive torques in N*m
        """
        if self.motor_model == "motors_disabled":
            return torch.zeros_like(desired_rps), torch.zeros_like(desired_rps), torch.zeros_like(desired_rps)

        self.rotor_dynamics.update_current_rps(desired_rps)
        forces, reactive_torques = self.rotor_dynamics._get_current_forces()
        return self.rotor_dynamics.current_rps, forces, reactive_torques

    def apply_motor_forces(self, forces: torch.Tensor, torques: torch.Tensor, debug=False):
        """Apply motor thrust and reactive torques in body frame.

        Args:
            forces: (n_envs, 4) thrust forces in Newtons (applied in +Z body frame)
            torques: (n_envs, 4) reactive torques in N*m (applied around Z axis with rotor direction)
            debug: if True, print detailed debug information
        """
        if self.motor_model == "motors_disabled":
            return

        forces_and_torques = torch.zeros(self.n_envs, self.robot.num_bodies, 6, device=self.device)
        for i in range(4):
            motor_idx = self.motor_body_indices[i]
            forces_and_torques[:, motor_idx, 2] = forces[:, i]
            forces_and_torques[:, motor_idx, 5] = self.motor_directions[i] * torques[:, i]

        env_indices = torch.arange(self.n_envs, dtype=torch.int32, device=self.device)
        self.robot.root_physx_view.apply_forces_and_torques_at_position(
            forces_and_torques[:, :, :3],
            forces_and_torques[:, :, 3:],
            None,
            env_indices,
            is_global=False
        )

    def _compute_robot_mass(self):
        """Compute total robot mass from URDF."""
        body_masses = self.robot.root_physx_view.get_masses()
        self.robot_mass = body_masses[0].sum().item()
        print(f"[MorphySimulator] Robot mass: {self.robot_mass:.3f} kg")

    def get_inertia_tensor(self) -> torch.Tensor:
        """Get composite inertia tensor about base link origin in body frame.

        Uses parallel axis theorem: I_total = sum(I_body + m*(||r||^2*I - r*r^T))
        where r is the vector from base link origin to each body's COM.

        Returns:
            inertia: (n_envs, 3, 3) composite inertia tensor in body frame
        """
        from utils import quat_rotate, quat_inverse

        body_inertias = self.robot.root_physx_view.get_inertias().to(self.device)
        body_masses = self.robot.root_physx_view.get_masses().to(self.device)

        body_com_pos_w = self.robot.data.body_com_pos_w
        base_pos_world = self.robot.data.root_pos_w
        base_quat_world = self.robot.data.root_quat_w

        body_pos_relative_world = body_com_pos_w - base_pos_world.unsqueeze(1)
        body_pos_relative_body = quat_rotate(
            quat_inverse(base_quat_world).unsqueeze(1).expand(-1, body_com_pos_w.shape[1], -1),
            body_pos_relative_world
        )

        composite_inertia = torch.zeros((self.n_envs, 3, 3), device=self.device)

        for body_idx in range(self.robot.num_bodies):
            m = body_masses[:, body_idx]
            r = body_pos_relative_body[:, body_idx, :]
            I_body = body_inertias[:, body_idx, :].reshape(self.n_envs, 3, 3)

            r_norm_sq = torch.sum(r * r, dim=-1, keepdim=True).unsqueeze(-1)
            identity = torch.eye(3, device=self.device).unsqueeze(0).expand(self.n_envs, -1, -1)
            r_outer = torch.bmm(r.unsqueeze(-1), r.unsqueeze(-2))

            parallel_axis_term = m.unsqueeze(-1).unsqueeze(-1) * (r_norm_sq * identity - r_outer)
            composite_inertia += I_body + parallel_axis_term

        return composite_inertia

    def get_com_offset(self) -> torch.Tensor:
        """Get system COM offset from base link origin in body frame.

        Returns:
            com_offset: (n_envs, 3) COM position relative to base link origin
        """
        from utils import quat_rotate, quat_inverse

        body_masses = self.robot.root_physx_view.get_masses().to(self.device)
        body_com_pos_w = self.robot.data.body_com_pos_w
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        total_mass = body_masses.sum(dim=1, keepdim=True)
        system_com_w = (body_com_pos_w * body_masses.unsqueeze(-1)).sum(dim=1) / total_mass
        com_offset_w = system_com_w - base_pos_w
        com_offset_body = quat_rotate(quat_inverse(base_quat_w), com_offset_w)
        return com_offset_body

    def get_motor_positions_body_frame(self) -> torch.Tensor:
        """Get motor positions in body frame directly from simulation.

        Returns:
            motor_positions: (n_envs, 4, 3) motor positions in body frame
        """
        from utils import quat_rotate, quat_inverse

        motor_body_names = ["motor_0", "motor_1", "motor_2", "motor_3"]
        motor_body_indices = [self.robot.body_names.index(name) for name in motor_body_names]

        motor_pos_world = self.robot.data.body_pos_w[:, motor_body_indices, :]
        base_pos_world = self.robot.data.root_pos_w
        base_quat_world = self.robot.data.root_quat_w

        motor_pos_relative_world = motor_pos_world - base_pos_world.unsqueeze(1)
        motor_pos_body = quat_rotate(
            quat_inverse(base_quat_world).unsqueeze(1).expand(-1, 4, -1),
            motor_pos_relative_world
        )

        return motor_pos_body

    def get_motor_positions_body_frame_for_allocation_matrix(self) -> torch.Tensor:
        """Get motor positions in body frame for allocation matrix updates."""
        if self.has_soft_joints:
            return self._zero_arm_motor_positions_body
        return self.get_motor_positions_body_frame()

    def get_motor_thrust_directions_body_frame(self) -> torch.Tensor:
        """Get motor thrust directions in body frame from current simulation state.

        Returns:
            thrust_directions: (n_envs, 4, 3) motor thrust directions in body frame
        """
        from utils import quat_rotate, quat_inverse

        thrust_dirs_body = torch.zeros((self.n_envs, 4, 3), device=self.device, dtype=torch.float32)
        base_quat_world = self.robot.data.root_quat_w
        local_z = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        for motor_idx in range(4):
            motor_body_idx = self.motor_body_indices[motor_idx]
            motor_quat_world = self.robot.data.body_quat_w[:, motor_body_idx, :]
            thrust_dir_world = quat_rotate(motor_quat_world, local_z.unsqueeze(0).expand(self.n_envs, -1))
            thrust_dir_body = quat_rotate(quat_inverse(base_quat_world), thrust_dir_world)
            thrust_dirs_body[:, motor_idx, :] = thrust_dir_body

        return thrust_dirs_body

    def get_motor_thrust_directions_body_frame_for_allocation_matrix(self) -> torch.Tensor:
        """Get motor thrust directions in body frame for allocation matrix updates."""
        if self.has_soft_joints:
            return self._zero_arm_motor_thrust_directions_body
        return self.get_motor_thrust_directions_body_frame()

    def _cache_zero_arm_allocation_geometry(self):
        self._zero_arm_motor_positions_body = self.get_motor_positions_body_frame().clone()
        self._zero_arm_motor_thrust_directions_body = self.get_motor_thrust_directions_body_frame().clone()

    def set_actions(self, motor_actions: torch.Tensor):
        """Apply motor actions through rotor dynamics.

        Args:
            motor_actions: (n_envs, 4) motor commands in [0, 1] range
        """
        if self.motor_model == "motors_disabled":
            return

        self._last_motor_actions = motor_actions
        forces, torques = self.rotor_dynamics.update_state_and_get_current_forces(motor_actions)
        self.apply_motor_forces(forces, torques)

    def get_motor_torques_on_joints(self) -> torch.Tensor:
        """Compute torque that motors 1 and 3 produce on their respective joints.

        Uses current motor positions, thrust directions, and forces to compute
        the torque about each joint axis. This accounts for the current joint angle.

        Returns:
            torques: (n_envs, 2) torques on [joint_1, joint_3] in Nm
        """
        if self.motor_model == "motors_disabled":
            return torch.zeros(self.n_envs, 2, device=self.device)

        from utils import quat_rotate, quat_inverse

        motor_pos_body = self.get_motor_positions_body_frame()
        thrust_dirs_body = self.get_motor_thrust_directions_body_frame()
        forces = self.get_motor_forces()
        _, reaction_torques = self.rotor_dynamics._get_current_forces()

        joint_positions = torch.zeros(self.n_envs, 2, 3, device=self.device)
        joint_axes = torch.zeros(self.n_envs, 2, 3, device=self.device)

        joint_pos, _ = self.get_joint_state()
        base_quat = self.robot.data.root_quat_w

        for idx, motor_idx in enumerate([1, 3]):
            seg1_body_name = f"arm_segment_1_{motor_idx}"
            seg1_body_idx = self.robot.body_names.index(seg1_body_name)
            seg1_pos_world = self.robot.data.body_pos_w[:, seg1_body_idx, :]
            seg1_quat_world = self.robot.data.body_quat_w[:, seg1_body_idx, :]

            base_pos_world = self.robot.data.root_pos_w
            seg1_pos_rel_world = seg1_pos_world - base_pos_world
            joint_positions[:, idx, :] = quat_rotate(quat_inverse(base_quat), seg1_pos_rel_world)

            joint_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            joint_axes[:, idx, :] = quat_rotate(
                quat_inverse(base_quat).unsqueeze(1).expand(-1, 1, -1).squeeze(1),
                quat_rotate(seg1_quat_world, joint_axis_local.unsqueeze(0).expand(self.n_envs, -1))
            )

        result = torch.zeros(self.n_envs, 2, device=self.device)
        for idx, motor_idx in enumerate([1, 3]):
            motor_pos = motor_pos_body[:, motor_idx, :]
            thrust_dir = thrust_dirs_body[:, motor_idx, :]
            thrust_force = forces[:, motor_idx:motor_idx+1]
            reaction_torque = reaction_torques[:, motor_idx]
            motor_direction = self.motor_directions[motor_idx]

            joint_pos_vec = joint_positions[:, idx, :]
            joint_axis = joint_axes[:, idx, :]

            r_vec = motor_pos - joint_pos_vec
            thrust_vec = thrust_dir * thrust_force

            torque_from_thrust = torch.sum(torch.cross(r_vec, thrust_vec, dim=-1) * joint_axis, dim=-1)
            reaction_torque_vec = motor_direction * reaction_torque.unsqueeze(-1) * thrust_dir
            torque_from_reaction = torch.sum(reaction_torque_vec * joint_axis, dim=-1)

            result[:, idx] = torque_from_thrust + torque_from_reaction

        return result

    def debug_motor_torque_on_joint(self, motor_idx: int, env_idx: int = 0):
        """Print detailed debug info about motor torque on joint.

        Args:
            motor_idx: Motor index (1 for left, 3 for right)
            env_idx: Environment index
        """
        from utils import quat_rotate, quat_inverse
        import numpy as np

        joint_idx = 0 if motor_idx == 1 else 1
        joint_name = "left" if motor_idx == 1 else "right"

        joint_pos, joint_vel = self.get_joint_state()
        joint_angle_rad = joint_pos[env_idx, joint_idx].item()
        joint_angle_deg = np.degrees(joint_angle_rad)

        motor_pos_body = self.get_motor_positions_body_frame()
        thrust_dirs_body = self.get_motor_thrust_directions_body_frame()
        forces = self.get_motor_forces()
        _, reaction_torques = self.rotor_dynamics._get_current_forces()

        motor_pos = motor_pos_body[env_idx, motor_idx, :].cpu().numpy()
        thrust_dir = thrust_dirs_body[env_idx, motor_idx, :].cpu().numpy()
        thrust_force = forces[env_idx, motor_idx].item()
        reaction_torque = reaction_torques[env_idx, motor_idx].item()
        motor_direction = self.motor_directions[motor_idx].item()

        seg1_body_name = f"arm_segment_1_{motor_idx}"
        seg1_body_idx = self.robot.body_names.index(seg1_body_name)
        seg1_pos_world = self.robot.data.body_pos_w[env_idx, seg1_body_idx, :]
        seg1_quat_world = self.robot.data.body_quat_w[env_idx, seg1_body_idx, :]
        base_pos_world = self.robot.data.root_pos_w[env_idx, :]
        base_quat_world = self.robot.data.root_quat_w[env_idx, :]

        seg1_pos_rel_world = seg1_pos_world - base_pos_world
        joint_pos_body = quat_rotate(quat_inverse(base_quat_world.unsqueeze(0)), seg1_pos_rel_world.unsqueeze(0))[0].cpu().numpy()

        joint_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        joint_axis_world = quat_rotate(seg1_quat_world.unsqueeze(0), joint_axis_local.unsqueeze(0))[0]
        joint_axis_body = quat_rotate(quat_inverse(base_quat_world.unsqueeze(0)), joint_axis_world.unsqueeze(0))[0].cpu().numpy()

        r_vec = motor_pos - joint_pos_body
        thrust_vec = thrust_dir * thrust_force

        torque_from_thrust_vec = np.cross(r_vec, thrust_vec)
        torque_from_thrust = np.dot(torque_from_thrust_vec, joint_axis_body)

        reaction_torque_vec = motor_direction * reaction_torque * thrust_dir
        torque_from_reaction = np.dot(reaction_torque_vec, joint_axis_body)

        total_torque = torque_from_thrust + torque_from_reaction

        print(f"Motor {motor_idx} ({joint_name} arm) Debug Info:")
        print(f"  Joint angle: {joint_angle_deg:.2f}° ({joint_angle_rad:.4f} rad)")
        print(f"  Joint velocity: {joint_vel[env_idx, joint_idx].item():.4f} rad/s")
        print()
        print(f"  Motor position (body frame): [{motor_pos[0]:.4f}, {motor_pos[1]:.4f}, {motor_pos[2]:.4f}]")
        print(f"  Joint position (body frame): [{joint_pos_body[0]:.4f}, {joint_pos_body[1]:.4f}, {joint_pos_body[2]:.4f}]")
        print(f"  Lever arm r_vec:             [{r_vec[0]:.4f}, {r_vec[1]:.4f}, {r_vec[2]:.4f}]")
        print(f"  |r_vec|: {np.linalg.norm(r_vec):.4f} m")
        print()
        print(f"  Thrust direction (body frame): [{thrust_dir[0]:.4f}, {thrust_dir[1]:.4f}, {thrust_dir[2]:.4f}]")
        print(f"  Thrust force: {thrust_force:.4f} N")
        print(f"  Thrust vector: [{thrust_vec[0]:.4f}, {thrust_vec[1]:.4f}, {thrust_vec[2]:.4f}]")
        print()
        print(f"  Joint axis (body frame): [{joint_axis_body[0]:.4f}, {joint_axis_body[1]:.4f}, {joint_axis_body[2]:.4f}]")
        print(f"  |joint_axis|: {np.linalg.norm(joint_axis_body):.4f}")
        print()
        print(f"  r × F (torque from thrust): [{torque_from_thrust_vec[0]:.6f}, {torque_from_thrust_vec[1]:.6f}, {torque_from_thrust_vec[2]:.6f}]")
        print(f"  (r × F) · joint_axis = {torque_from_thrust:.6f} Nm")
        print()
        print(f"  Motor direction: {motor_direction}")
        print(f"  Motor reaction torque: {reaction_torque:.6f} Nm")
        print(f"  Reaction torque vector: [{reaction_torque_vec[0]:.6f}, {reaction_torque_vec[1]:.6f}, {reaction_torque_vec[2]:.6f}]")
        print(f"  reaction_torque_vec · joint_axis = {torque_from_reaction:.6f} Nm")
        print()
        print(f"  TOTAL motor torque on joint: {total_torque:.6f} Nm")
        print(f"  Expected (from static analysis at 100% thrust): ~-0.0535 Nm")
        print(f"  At current thrust ({thrust_force:.2f}N / 2.06N = {thrust_force/2.06*100:.1f}%): expected ~{-0.0535 * thrust_force/2.06:.6f} Nm")

    def _record_joint_data(self, joint_pos, joint_vel, interp, damp):
        """Record one timestep of joint data for env[0]."""
        if self._joint_track is None:
            self._joint_track = {'time': [], 'pos': [], 'vel': [], 'interp': [], 'damp': []}
            self._joint_track_step = 0
        self._joint_track['time'].append(self._joint_track_step * self.dt)
        self._joint_track['pos'].append(joint_pos[0].cpu().numpy().copy())
        self._joint_track['vel'].append(joint_vel[0].cpu().numpy().copy())
        self._joint_track['interp'].append(interp[0].cpu().numpy().copy())
        self._joint_track['damp'].append(damp[0].cpu().numpy().copy())
        self._joint_track_step += 1

    def plot_joint_dynamics(self):
        """Plot recorded joint dynamics to PDF (one page per arm)."""
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        def fmt(v): return str(v).replace('.', '_').replace('-', 'neg')
        output_path = f"results/joint_dynamics_dt_{fmt(self.dt)}_damping_{fmt(self.damping)}.pdf"

        self.track_joint_dynamics = False
        data = self._joint_track
        self._joint_track = None
        if data is None or len(data['time']) == 0:
            print("[MorphySimulator] No joint data recorded, skipping plot")
            return
        t = np.array(data['time'])
        pos = np.degrees(np.array(data['pos']))
        vel = np.degrees(np.array(data['vel']))
        interp = np.array(data['interp'])
        damp = np.array(data['damp'])

        with PdfPages(output_path) as pdf:
            for arm_idx, arm_name in enumerate(['Left', 'Right']):
                fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
                axes[0].plot(t, pos[:, arm_idx], 'b-', lw=1.5)
                axes[0].set_ylabel('Position (deg)')
                axes[0].axhline(0, color='k', ls='--', alpha=0.3)
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(t, pos[:, arm_idx], 'b-', lw=1.5)
                axes[1].set_ylabel('Position (deg)\n[zoomed]')
                axes[1].set_ylim(-5, 5)
                axes[1].axhline(0, color='k', ls='--', alpha=0.3)
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(t, vel[:, arm_idx], 'g-', lw=1.5)
                axes[2].set_ylabel('Velocity (deg/s)')
                axes[2].axhline(0, color='k', ls='--', alpha=0.3)
                axes[2].grid(True, alpha=0.3)

                axes[3].plot(t, interp[:, arm_idx], 'r-', lw=1.5)
                axes[3].set_ylabel('Interpolated (Nm)')
                axes[3].axhline(0, color='k', ls='--', alpha=0.3)
                axes[3].grid(True, alpha=0.3)

                axes[4].plot(t, damp[:, arm_idx], 'm-', lw=1.5)
                axes[4].set_ylabel('Damping (Nm)')
                axes[4].set_xlabel('Time (s)')
                axes[4].axhline(0, color='k', ls='--', alpha=0.3)
                axes[4].grid(True, alpha=0.3)

                fig.suptitle(f'{arm_name} Arm Joint Dynamics', fontsize=14)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"[MorphySimulator] Saved joint dynamics to {output_path}")

    def cleanup(self):
        """Clean up simulation resources."""
        self.scene.reset()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
