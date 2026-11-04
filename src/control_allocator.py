import torch
from rotor_dynamics import RotorDynamics


class ControlAllocator:
    """
    Control allocator for Morphy drone with dynamic allocation matrix.
    Updates allocation matrix based on joint positions at each timestep.
    """

    def __init__(self, num_envs, dt, motor_idx_list, device, motor_directions, motor_model="asymmetric_second_order"):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.num_motors = len(motor_idx_list)

        self.motor_dynamics = RotorDynamics(
            n_envs=num_envs,
            motor_idx_list=motor_idx_list,
            dt=dt,
            device=device,
            motor_model=motor_model
        )

        max_force, max_torque, _ = RotorDynamics.get_motor_max_force_torque_and_rps(motor_idx_list)
        self.thrust_to_torque_ratio = torch.tensor(
            max_torque / max_force, device=device, dtype=torch.float32
        )

        self.motor_directions = torch.tensor(motor_directions, device=device, dtype=torch.float32)

        self.allocation_matrix = torch.zeros((num_envs, 6, self.num_motors), device=device)
        self.allocation_matrix_pinv = torch.zeros((num_envs, self.num_motors, 6), device=device)

        self.debug_data = {}
        self.step_count = 0

    def update_allocation_matrix(self, motor_positions_body, motor_thrust_directions_body):
        """
        Update allocation matrix based on actual motor positions and thrust directions from simulator.

        Args:
            motor_positions_body: (num_envs, 4, 3) motor positions in body frame
            motor_thrust_directions_body: (num_envs, 4, 3) motor thrust directions in body frame
        """
        for i in range(self.num_motors):
            motor_pos = motor_positions_body[:, i, :]
            thrust_direction = motor_thrust_directions_body[:, i, :]

            self.allocation_matrix[:, 0:3, i] = thrust_direction

            moment_arm = torch.cross(motor_pos, thrust_direction, dim=-1)
            torque_dir = self.motor_directions[i] * self.thrust_to_torque_ratio[i] * thrust_direction
            total_torque = moment_arm + torque_dir

            self.allocation_matrix[:, 3:6, i] = total_torque

        self.allocation_matrix_pinv = torch.linalg.pinv(self.allocation_matrix)

    def allocate(self, wrench_command):
        """
        Convert wrench command to motor thrust forces.

        Args:
            wrench_command: (num_envs, 6) desired wrench [Fx, Fy, Fz, Tx, Ty, Tz]

        Returns:
            motor_forces: (num_envs, num_motors) motor forces in Newtons
        """
        min_force = self.motor_dynamics.component_data.min_force
        max_force = self.motor_dynamics.component_data.max_force

        desired_thrusts = torch.bmm(
            self.allocation_matrix_pinv,
            wrench_command.unsqueeze(-1)
        ).squeeze(-1)

        motor_forces = torch.clamp(desired_thrusts, min=min_force, max=max_force)

        self.debug_data = {
            'desired_thrusts': desired_thrusts,
            'motor_forces': motor_forces,
            'clamped': (torch.abs(desired_thrusts - motor_forces).sum() > 1e-6).item(),
        }

        self.step_count += 1
        return motor_forces

    def reset_idx(self, env_ids, initial_action=None):
        """Reset motor dynamics for specific environments."""
        self.motor_dynamics.reset_idx(env_ids, initial_action)
