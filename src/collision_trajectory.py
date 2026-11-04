import torch


NMPC_DIRECT_ACTUATOR = "NMPC_DIRECT_ACTUATOR"
PX4_POSITION_CONTROL = "PX4_POSITION_CONTROL"
NMPC_FLIGHT_MODE = "NMPC_FLIGHT_MODE"
NMPC_RECOVERY_MODE = "NMPC_RECOVERY_MODE"
NMPC_COAST_MODE = "NMPC_COAST_MODE"

SUPPORTED_CONTROL_MODES = {
    NMPC_DIRECT_ACTUATOR,
    PX4_POSITION_CONTROL,
}
SUPPORTED_NMPC_MODES = {
    NMPC_FLIGHT_MODE,
    NMPC_RECOVERY_MODE,
    NMPC_COAST_MODE,
}


def _require_vector3(name, value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list")
    vector = torch.tensor(value, dtype=torch.float32)
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


class CollisionReferenceTrajectory:
    def __init__(self, env_origins, wall_x_position, collision_config, device):
        self.device = device
        self.n_envs = env_origins.shape[0]
        self.env_origins = env_origins
        self.wall_x_position = float(wall_x_position)

        if "setpoints" not in collision_config:
            raise KeyError("trajectory.collision.setpoints is required")
        setpoints = collision_config["setpoints"]
        if not isinstance(setpoints, list) or len(setpoints) == 0:
            raise ValueError("trajectory.collision.setpoints must not be empty")

        self.n_setpoints = len(setpoints)
        self.gap_frame_origins = self.env_origins + torch.tensor(
            [self.wall_x_position, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        self.sp_positions_rel_gap = []
        self.sp_positions_world = []
        self.sp_velocities = []
        self.sp_setpoint_times = []
        self.sp_waypoint_x_limits = []
        self.sp_control_modes = []
        self.sp_nmpc_modes = []

        for index, setpoint in enumerate(setpoints):
            position_rel_gap = _require_vector3(
                f"trajectory.collision.setpoints[{index}].position_rel_gap",
                setpoint["position_rel_gap"],
            )
            velocity = _require_vector3(
                f"trajectory.collision.setpoints[{index}].velocity",
                setpoint["velocity"],
            )
            control_mode = setpoint["control_mode"]
            nmpc_mode = setpoint["nmpc_mode"]
            setpoint_time = setpoint["setpoint_time"]
            waypoint_x_limit = setpoint["waypoint_x_limit_rel_gap"]

            if control_mode not in SUPPORTED_CONTROL_MODES:
                raise ValueError(f"invalid control_mode at setpoint {index}: {control_mode}")
            if nmpc_mode not in SUPPORTED_NMPC_MODES:
                raise ValueError(f"invalid nmpc_mode at setpoint {index}: {nmpc_mode}")

            is_last = index == self.n_setpoints - 1
            if not is_last:
                has_time = setpoint_time is not None
                has_x_limit = waypoint_x_limit is not None
                if not has_time and not has_x_limit:
                    raise ValueError(
                        "each non-final trajectory.collision setpoint must define at least one of "
                        "setpoint_time or waypoint_x_limit_rel_gap"
                    )

            if setpoint_time is not None:
                setpoint_time = float(setpoint_time)
                if not torch.isfinite(torch.tensor(setpoint_time)):
                    raise ValueError(f"trajectory.collision.setpoints[{index}].setpoint_time must be finite")
            if waypoint_x_limit is not None:
                waypoint_x_limit = float(waypoint_x_limit)
                if not torch.isfinite(torch.tensor(waypoint_x_limit)):
                    raise ValueError(
                        f"trajectory.collision.setpoints[{index}].waypoint_x_limit_rel_gap must be finite"
                    )

            self.sp_positions_rel_gap.append(position_rel_gap.to(device=self.device))
            self.sp_positions_world.append(position_rel_gap.to(device=self.device) + self.gap_frame_origins)
            self.sp_velocities.append(velocity.to(device=self.device))
            self.sp_setpoint_times.append(setpoint_time)
            self.sp_waypoint_x_limits.append(waypoint_x_limit)
            self.sp_control_modes.append(control_mode)
            self.sp_nmpc_modes.append(nmpc_mode)

        self.phase = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.phase_start_time = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0
        self.phase_start_time[env_ids] = 0.0

    def require_direct_actuator_control(self):
        invalid_indices = [
            index
            for index, control_mode in enumerate(self.sp_control_modes)
            if control_mode != NMPC_DIRECT_ACTUATOR
        ]
        if len(invalid_indices) > 0:
            raise ValueError(
                "NMPC simulation only supports trajectory.collision setpoints with "
                f"control_mode={NMPC_DIRECT_ACTUATOR}, invalid setpoints={invalid_indices}"
            )

    def _advance_phase(self, pos_world, current_time):
        pos_rel_gap = pos_world - self.gap_frame_origins
        for phase_index in range(self.n_setpoints - 1):
            in_phase = self.phase == phase_index
            if not in_phase.any():
                continue

            setpoint_time = self.sp_setpoint_times[phase_index]
            waypoint_x_limit = self.sp_waypoint_x_limits[phase_index]

            if setpoint_time is None and waypoint_x_limit is None:
                raise RuntimeError(
                    "trajectory.collision phase has neither setpoint_time nor waypoint_x_limit_rel_gap"
                )

            # Matches the PX4 setpoint sequencer: advance when the setpoint time
            # elapses or the x limit is crossed, whichever happens first.
            transition = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            if setpoint_time is not None:
                elapsed = current_time[in_phase] - self.phase_start_time[in_phase]
                transition[in_phase] = elapsed >= setpoint_time
            if waypoint_x_limit is not None:
                transition[in_phase] |= pos_rel_gap[in_phase, 0] >= waypoint_x_limit

            if transition.any():
                self.phase[transition] = phase_index + 1
                self.phase_start_time[transition] = current_time[transition]

    def get_reference(self, pos_world, current_time):
        self._advance_phase(pos_world, current_time)

        ref_pos = torch.zeros((self.n_envs, 3), device=self.device, dtype=torch.float32)
        ref_vel = torch.zeros((self.n_envs, 3), device=self.device, dtype=torch.float32)
        ref_yaw = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        ref_yaw_rate = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        nmpc_mode = []
        control_mode = []

        for phase_index in range(self.n_setpoints):
            mask = self.phase == phase_index
            if not mask.any():
                continue
            ref_pos[mask] = self.sp_positions_world[phase_index][mask]
            ref_vel[mask] = self.sp_velocities[phase_index]

        for env_index in range(self.n_envs):
            phase_index = int(self.phase[env_index].item())
            nmpc_mode.append(self.sp_nmpc_modes[phase_index])
            control_mode.append(self.sp_control_modes[phase_index])

        return ref_pos, ref_vel, ref_yaw, ref_yaw_rate, control_mode, nmpc_mode
