import torch
from controllers import BaseController
from utils import (
    quat_mul, quat_inverse, quat_rotate, quat_to_rotation_matrix,
    quat_from_rotation_matrix, vee_map, normalize_quat
)


class FranchiPositionController(BaseController):
    """Franchi position controller (TRO 2018) for laterally bounded force platforms.

    Outputs body-frame wrench [Fx, Fy, Fz, Tx, Ty, Tz].
    """

    def __init__(self, num_envs, device, K_pos, K_vel, K_rot, K_angvel, dt, bisection_iter,
                 recovery_angvel_threshold, recovery_orientation_threshold):
        super().__init__(num_envs, device)
        self.K_pos = torch.tensor(K_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
        self.K_vel = torch.tensor(K_vel, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
        self.K_rot = K_rot
        self.K_angvel = K_angvel
        self.dt = dt
        self.bisection_iter = bisection_iter
        self.recovery_angvel_threshold = recovery_angvel_threshold
        self.recovery_orientation_threshold = recovery_orientation_threshold

        self.wrench_command = torch.zeros((num_envs, 6), device=device)
        self.desired_quat = torch.zeros((num_envs, 4), device=device)
        self.desired_quat[:, 0] = 1.0

        self.prev_R_d = torch.eye(3, device=device).unsqueeze(0).expand(num_envs, -1, -1).clone()
        self.prev_omega_d = torch.zeros((num_envs, 3), device=device)

        self.r_xy = torch.zeros(num_envs, device=device)
        self.alloc_pinv = None
        self.motor_max_force = None
        self.motor_min_force = None
        self._first_step = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.debug_data = {}

    def set_r_xy(self, allocator):
        """Compute lateral force bound and cache allocator data for motor-aware scaling."""
        alloc = allocator.allocation_matrix
        max_force = allocator.motor_dynamics.component_data.max_force
        min_force = allocator.motor_dynamics.component_data.min_force
        max_force_tensor = torch.tensor(max_force, device=self.device, dtype=torch.float32)
        fx_row = alloc[:, 0, :]
        fy_row = alloc[:, 1, :]
        r_x = (fx_row.abs() * max_force_tensor.unsqueeze(0)).sum(dim=-1)
        r_y = (fy_row.abs() * max_force_tensor.unsqueeze(0)).sum(dim=-1)
        self.r_xy = torch.min(r_x, r_y)
        self.alloc_pinv = allocator.allocation_matrix_pinv.clone()
        self.motor_max_force = max_force_tensor
        self.motor_min_force = torch.tensor(min_force, device=self.device, dtype=torch.float32)

    def reset(self, env_ids):
        self.prev_R_d[env_ids] = torch.eye(3, device=self.device)
        self.prev_omega_d[env_ids] = 0.0
        self._first_step[env_ids] = True

    def update(self, setpoint_position, setpoint_velocity, setpoint_yaw, setpoint_yaw_rate):
        self.wrench_command.zero_()

        R = quat_to_rotation_matrix(self.robot_orientation)

        e_p = self.robot_position - setpoint_position
        e_v = self.robot_linvel - setpoint_velocity
        f_r = -self.mass * self.gravity - self.K_pos * e_p - self.K_vel * e_v

        cos_yaw = torch.cos(setpoint_yaw)
        sin_yaw = torch.sin(setpoint_yaw)
        zeros = torch.zeros_like(setpoint_yaw)
        ones = torch.ones_like(setpoint_yaw)
        R_r = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=-2)

        b3_r = R_r[:, :, 2]
        b1_r = R_r[:, :, 0]

        b3_d = self._calc_b3d(b3_r, f_r, self.r_xy)

        b2_d_raw = torch.cross(b3_d, b1_r, dim=-1)
        b2_d = b2_d_raw / torch.norm(b2_d_raw, dim=-1, keepdim=True).clamp(min=1e-6)
        b1_d = torch.cross(b2_d, b3_d, dim=-1)
        b1_d = b1_d / torch.norm(b1_d, dim=-1, keepdim=True).clamp(min=1e-6)

        R_d = torch.stack([b1_d, b2_d, b3_d], dim=-1)

        if self._first_step.any():
            self.prev_R_d[self._first_step] = R_d[self._first_step]
            self._first_step[:] = False

        omega_d_hat = (torch.bmm(self.prev_R_d.transpose(-2, -1), R_d) - torch.eye(3, device=self.device).unsqueeze(0)) / self.dt
        omega_d = vee_map(omega_d_hat - omega_d_hat.transpose(-2, -1)) * 0.5
        omega_d_norm = torch.norm(omega_d, dim=-1, keepdim=True).clamp(min=1e-8)
        omega_d = torch.where(omega_d_norm > 10.0, omega_d * 10.0 / omega_d_norm, omega_d)
        omega_dot_d = (omega_d - self.prev_omega_d) / self.dt

        self.prev_R_d = R_d.clone()
        self.prev_omega_d = omega_d.clone()

        self.desired_quat[:] = normalize_quat(quat_from_rotation_matrix(R_d))

        f_body = torch.bmm(R.transpose(-2, -1), f_r.unsqueeze(-1)).squeeze(-1)
        f_xy = f_body[:, :2]
        f_xy_norm = torch.norm(f_xy, dim=-1, keepdim=True)
        scale = torch.where(f_xy_norm.squeeze(-1) > self.r_xy, self.r_xy / f_xy_norm.squeeze(-1), torch.ones_like(self.r_xy))
        self.wrench_command[:, 0] = f_body[:, 0] * scale
        self.wrench_command[:, 1] = f_body[:, 1] * scale
        self.wrench_command[:, 2] = f_body[:, 2]

        e_R_mat = torch.bmm(R_d.transpose(-2, -1), R) - torch.bmm(R.transpose(-2, -1), R_d)
        e_R = 0.5 * vee_map(e_R_mat)

        omega_body = self.robot_body_angvel
        R_d_T_R = torch.bmm(R.transpose(-2, -1), R_d)
        e_omega = omega_body - torch.bmm(R_d_T_R, omega_d.unsqueeze(-1)).squeeze(-1)

        inertia_omega = torch.bmm(self.inertia, omega_body.unsqueeze(-1)).squeeze(-1)
        gyroscopic = torch.cross(omega_body, inertia_omega, dim=-1)

        omega_cross = torch.zeros((self.num_envs, 3, 3), device=self.device)
        omega_cross[:, 0, 1] = -omega_body[:, 2]
        omega_cross[:, 0, 2] = omega_body[:, 1]
        omega_cross[:, 1, 0] = omega_body[:, 2]
        omega_cross[:, 1, 2] = -omega_body[:, 0]
        omega_cross[:, 2, 0] = -omega_body[:, 1]
        omega_cross[:, 2, 1] = omega_body[:, 0]

        feedfwd_term1 = torch.bmm(omega_cross, torch.bmm(R_d_T_R, omega_d.unsqueeze(-1))).squeeze(-1)
        feedfwd_term2 = torch.bmm(R_d_T_R, omega_dot_d.unsqueeze(-1)).squeeze(-1)
        feedfwd = feedfwd_term1 - feedfwd_term2

        self.wrench_command[:, 3:6] = gyroscopic - self.K_rot * e_R - self.K_angvel * e_omega + torch.bmm(self.inertia, feedfwd.unsqueeze(-1)).squeeze(-1)

        omega_norm = torch.norm(omega_body, dim=-1)
        e_R_norm = torch.norm(e_R, dim=-1)
        recovery_angvel = omega_norm > self.recovery_angvel_threshold
        recovery_orient = (~recovery_angvel) & (e_R_norm > self.recovery_orientation_threshold)

        if recovery_angvel.any():
            self.wrench_command[recovery_angvel, :3] = 0.0
            self.wrench_command[recovery_angvel, 3:6] = gyroscopic[recovery_angvel] - self.K_angvel * e_omega[recovery_angvel]

        if recovery_orient.any():
            force_scale = (self.recovery_orientation_threshold / e_R_norm[recovery_orient].clamp(min=1e-6)).unsqueeze(-1)
            self.wrench_command[recovery_orient, :3] *= force_scale

        if self.alloc_pinv is not None:
            self._scale_wrench_to_motor_limits()

        self.debug_data = {
            'e_p': e_p, 'e_v': e_v, 'f_r': f_r,
            'b3_d': b3_d, 'e_R': e_R, 'e_omega': e_omega,
            'R_diag': torch.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], dim=-1),
            'gyroscopic': gyroscopic, 'omega_d': omega_d,
            'recovery_angvel': recovery_angvel, 'recovery_orient': recovery_orient,
        }

        return self.wrench_command

    def _scale_wrench_to_motor_limits(self):
        """Scale wrench so desired motor thrusts stay within [min_force, max_force].

        Prioritizes torque over force: if the full wrench exceeds motor limits,
        forces are scaled down first. Only if torque alone exceeds limits is
        torque scaled as well.
        """
        torque_only = torch.zeros_like(self.wrench_command)
        torque_only[:, 3:6] = self.wrench_command[:, 3:6]
        torque_thrusts = torch.bmm(self.alloc_pinv, torque_only.unsqueeze(-1)).squeeze(-1)

        torque_over = (torque_thrusts / self.motor_max_force.unsqueeze(0)).max(dim=-1).values
        torque_ratio = torque_over.clamp(min=1.0)
        needs_torque_scale = torque_ratio > 1.0
        if needs_torque_scale.any():
            self.wrench_command[needs_torque_scale, 3:6] /= torque_ratio[needs_torque_scale].unsqueeze(-1)

        full_thrusts = torch.bmm(self.alloc_pinv, self.wrench_command.unsqueeze(-1)).squeeze(-1)
        full_over = (full_thrusts / self.motor_max_force.unsqueeze(0)).max(dim=-1).values
        full_under = torch.where(
            full_thrusts > 1e-8,
            self.motor_min_force.unsqueeze(0) / full_thrusts,
            torch.zeros_like(full_thrusts)
        ).max(dim=-1).values
        full_ratio = torch.max(full_over, full_under).clamp(min=1.0)
        needs_force_scale = full_ratio > 1.0
        if needs_force_scale.any():
            self.wrench_command[needs_force_scale, :3] /= full_ratio[needs_force_scale].unsqueeze(-1)

    def _calc_b3d(self, b3_r, f_r, r_xy):
        """Algorithm 1: compute feasible b3_d closest to b3_r."""
        f_norm = torch.norm(f_r, dim=-1)
        f_dot_b3r = (f_r * b3_r).sum(dim=-1)
        feasibility_rhs = torch.sqrt(torch.clamp(f_norm ** 2 - r_xy ** 2, min=0.0))
        feasible = f_dot_b3r >= feasibility_rhs

        b3_d = b3_r.clone()

        if feasible.all():
            return b3_d

        infeasible = ~feasible
        f_inf = f_r[infeasible]
        b3_r_inf = b3_r[infeasible]
        r_xy_inf = r_xy[infeasible]
        f_norm_inf = f_norm[infeasible]

        f_hat = f_inf / f_norm_inf.unsqueeze(-1).clamp(min=1e-8)

        rot_axis = torch.cross(f_hat, b3_r_inf, dim=-1)
        rot_axis_norm = torch.norm(rot_axis, dim=-1, keepdim=True).clamp(min=1e-8)
        rot_axis = rot_axis / rot_axis_norm

        angle_max = torch.acos(torch.clamp((f_inf * b3_r_inf).sum(dim=-1) / f_norm_inf.clamp(min=1e-8), -1.0, 1.0))
        angle_min = torch.zeros_like(angle_max)

        target_cos = feasibility_rhs[infeasible] / f_norm_inf.clamp(min=1e-8)

        for _ in range(self.bisection_iter):
            angle_mid = (angle_min + angle_max) / 2.0
            b3_candidate = self._rodrigues(f_hat, rot_axis, angle_mid)
            cos_val = (f_inf * b3_candidate).sum(dim=-1) / f_norm_inf.clamp(min=1e-8)
            too_close = cos_val > target_cos
            angle_min = torch.where(too_close, angle_mid, angle_min)
            angle_max = torch.where(too_close, angle_max, angle_mid)

        b3_result = self._rodrigues(f_hat, rot_axis, (angle_min + angle_max) / 2.0)
        b3_result = b3_result / torch.norm(b3_result, dim=-1, keepdim=True).clamp(min=1e-8)
        b3_d[infeasible] = b3_result

        return b3_d

    def _rodrigues(self, v, k, theta):
        """Rotate v around axis k by angle theta using Rodrigues' formula."""
        cos_t = torch.cos(theta).unsqueeze(-1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        k_dot_v = (k * v).sum(dim=-1, keepdim=True)
        return v * cos_t + torch.cross(k, v, dim=-1) * sin_t + k * k_dot_v * (1.0 - cos_t)


class ReferenceTrajectory:
    """Multi-setpoint reference trajectory.

    Each setpoint has position, velocity, and threshold.
    Position [null, null, null] means use current robot position (velocity-only setpoint).
    Threshold null means final setpoint (no transition).
    Threshold float means transition to next setpoint when robot is within that distance of this setpoint's position.
    For velocity-only setpoints, threshold checks distance to the next setpoint's position.
    setpoint_time float means transition after that many seconds in this setpoint.
    Each setpoint must have exactly one of threshold or setpoint_time set (not both, not neither),
    except the final setpoint which must have both null.
    """

    def __init__(self, env_origins, setpoints, device):
        self.device = device
        self.n_envs = env_origins.shape[0]
        self.env_origins = env_origins
        self.n_setpoints = len(setpoints)

        self.sp_positions = []
        self.sp_velocities = []
        self.sp_thresholds = []
        self.sp_times = []
        self.sp_has_position = []
        for i, sp in enumerate(setpoints):
            pos = sp['position']
            has_pos = pos[0] is not None
            threshold = sp['threshold']
            setpoint_time = sp['setpoint_time']
            is_last = (i == len(setpoints) - 1)
            if is_last:
                assert threshold is None and setpoint_time is None, \
                    f"Final setpoint must have both threshold and setpoint_time as null, got threshold={threshold}, setpoint_time={setpoint_time}"
            else:
                assert (threshold is None) != (setpoint_time is None), \
                    f"Setpoint {i}: exactly one of threshold or setpoint_time must be set, got threshold={threshold}, setpoint_time={setpoint_time}"
            self.sp_has_position.append(has_pos)
            self.sp_positions.append(
                torch.tensor(pos if has_pos else [0.0, 0.0, 0.0], device=device, dtype=torch.float32)
            )
            self.sp_velocities.append(
                torch.tensor(sp['velocity'], device=device, dtype=torch.float32)
            )
            self.sp_thresholds.append(threshold)
            self.sp_times.append(setpoint_time)

        self.phase = torch.zeros(self.n_envs, dtype=torch.long, device=device)
        self.phase_start_time = torch.zeros(self.n_envs, device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0
        self.phase_start_time[env_ids] = 0.0

    def get_reference(self, pos_world, current_time):
        """Returns (ref_pos, ref_vel, ref_yaw, ref_yaw_rate) for controller."""
        pos_rel = pos_world - self.env_origins
        ref_pos = torch.zeros((self.n_envs, 3), device=self.device)
        ref_vel = torch.zeros((self.n_envs, 3), device=self.device)
        ref_yaw = torch.zeros(self.n_envs, device=self.device)
        ref_yaw_rate = torch.zeros(self.n_envs, device=self.device)

        for i in range(self.n_setpoints - 1):
            in_phase = self.phase == i
            if not in_phase.any():
                continue
            threshold = self.sp_thresholds[i]
            setpoint_time = self.sp_times[i]
            if threshold is not None:
                if self.sp_has_position[i]:
                    dist = torch.norm(pos_rel[in_phase] - self.sp_positions[i], dim=-1)
                else:
                    dist = torch.norm(pos_rel[in_phase] - self.sp_positions[i + 1], dim=-1)
                transition = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                transition[in_phase] = dist < threshold
            else:
                elapsed = current_time[in_phase] - self.phase_start_time[in_phase]
                transition = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                transition[in_phase] = elapsed >= setpoint_time
            if transition.any():
                self.phase[transition] = i + 1
                self.phase_start_time[transition] = current_time[transition]

        for i in range(self.n_setpoints):
            mask = self.phase == i
            if not mask.any():
                continue
            if self.sp_has_position[i]:
                ref_pos[mask] = self.env_origins[mask] + self.sp_positions[i]
            else:
                ref_pos[mask] = pos_world[mask]
            ref_vel[mask] = self.sp_velocities[i]

        return ref_pos, ref_vel, ref_yaw, ref_yaw_rate
