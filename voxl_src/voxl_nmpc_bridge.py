import struct
import socket
import time
import numpy as np
import torch
from controllers import BaseController
from utils import normalize_quat

NUM_MOTORS = 4
NX = 17
MEASURED_STATE_DIM = 13
PARAM_U_PREV_START = 64
NP = 68

STATE_FMT = "<IB3xQQQQ13d68d4d4QBB6f"
CTRL_FMT = "<Ii4dd4d"
STATE_SIZE = struct.calcsize(STATE_FMT)
CTRL_SIZE = struct.calcsize(CTRL_FMT)

FLAG_REINIT = 0x01
FLAG_RECOVERY = 0x02


class VoxlNMPCController(BaseController):

    def __init__(self, num_envs, device, voxl_ip, voxl_port, timeout_ms, controller_dt,
                 thrust_coefficients, motor_time_constants):
        assert num_envs == 1, (
            f"VoxlNMPCController only supports num_envs=1, got {num_envs}. "
            f"The TCP solver has a single motor-state estimator per connection, so with "
            f"num_envs>1 the commanded RPS from env N-1 would contaminate env N's estimate."
        )
        super().__init__(num_envs, device)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_ms / 1000.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.voxl_addr = (voxl_ip, voxl_port)
        self.sock.connect(self.voxl_addr)
        self.seq = 0
        self.controller_dt = controller_dt
        self.controller_dt_us = int(round(controller_dt * 1e6))

        self.thrust_coefficients = np.array(thrust_coefficients, dtype=np.float64)
        self.motor_time_constants = np.array(motor_time_constants, dtype=np.float64)

        self.motor_forces = torch.zeros((num_envs, NUM_MOTORS), device=device)
        self.desired_quat = torch.zeros((num_envs, 4), device=device)
        self.desired_quat[:, 0] = 1.0

        self._alloc_matrix = None
        self._com_offset = None
        self._p_buf = np.zeros(NP, dtype=np.float64)
        self._measured_state = np.zeros(MEASURED_STATE_DIM, dtype=np.float64)
        self._sample_timestamp_us = np.zeros(num_envs, dtype=np.uint64)
        self._commanded_force_prev = np.zeros((num_envs, NUM_MOTORS), dtype=np.float64)
        self._prev_setpoint_pos = np.full((num_envs, 3), np.nan)
        self._prev_setpoint_vel = np.full((num_envs, 3), np.nan)

        self.timing_log = []

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("VOXL solver disconnected")
            buf.extend(chunk)
        return bytes(buf)

    def _pack_state_packet(self, seq, flags, sample_timestamp_us, measured_state, params):
        motor_rps_meas = (0.0, 0.0, 0.0, 0.0)
        motor_rps_timestamps_us = (0, 0, 0, 0)
        motor_rps_valid_mask = 0
        reserved0 = 0
        nominal_position_enu = (0.0, 0.0, 0.0)
        nominal_velocity_enu = (0.0, 0.0, 0.0)
        return struct.pack(
            STATE_FMT,
            seq,
            flags,
            sample_timestamp_us,
            sample_timestamp_us,
            sample_timestamp_us,
            sample_timestamp_us,
            *measured_state.tolist(),
            *params.tolist(),
            *motor_rps_meas,
            *motor_rps_timestamps_us,
            motor_rps_valid_mask,
            reserved0,
            *nominal_position_enu,
            *nominal_velocity_enu,
        )

    def ping(self):
        measured_state = np.zeros(MEASURED_STATE_DIM, dtype=np.float64)
        measured_state[2] = 1.0
        measured_state[6] = 1.0
        p = np.zeros(NP, dtype=np.float64)
        p[0] = 1.0
        p[1] = 0.01
        p[4] = 0.01
        p[6] = 0.02
        p[9] = -9.81
        p[12] = 1.0
        p[19] = 1.0
        p[25] = 1.0
        p[31] = 1.0
        p[37] = 1.0
        p[43] = 1.0
        p[47:51] = self.thrust_coefficients
        p[51:55] = self.motor_time_constants
        p[PARAM_U_PREV_START:PARAM_U_PREV_START + NUM_MOTORS] = 0.0
        pkt = self._pack_state_packet(0, 0, 0, measured_state, p)
        print(f"[PING] Sending {len(pkt)}-byte test packet to {self.voxl_addr}...")
        self.sock.sendall(pkt)
        try:
            data = self._recv_exact(CTRL_SIZE)
            vals = struct.unpack(CTRL_FMT, data)
            print(f"[PING] Got response: status={vals[1]}, solve_time={vals[6]:.0f}us. VOXL solver is alive.")
        except socket.timeout:
            raise ConnectionError(
                f"No response from VOXL solver at {self.voxl_addr} within {self.sock.gettimeout()*1000:.0f}ms. "
                f"Check that adb forward is active and nmpc_udp_solver is running on VOXL."
            )

    def set_com_offset(self, com_offset_np):
        self._com_offset = com_offset_np

    def set_allocation_matrix(self, allocation_matrix_np):
        self._alloc_matrix = allocation_matrix_np

    def reset(self, env_ids):
        ids = env_ids.cpu().numpy()
        self._prev_setpoint_pos[ids] = np.nan
        self._prev_setpoint_vel[ids] = np.nan
        self._sample_timestamp_us[ids] = 0
        self._commanded_force_prev[ids] = 0.0

    def update(self, setpoint_position, setpoint_velocity, setpoint_yaw, setpoint_yaw_rate, nmpc_mode):
        self.motor_forces.zero_()

        pos_np = self.robot_position.cpu().numpy()
        vel_np = self.robot_linvel.cpu().numpy()
        quat_np = self.robot_orientation.cpu().numpy()
        omega_body_np = self.robot_body_angvel.cpu().numpy()

        sp_pos_np = setpoint_position.cpu().numpy()
        sp_vel_np = setpoint_velocity.cpu().numpy()
        sp_yaw_np = setpoint_yaw.cpu().numpy()
        nmpc_mode_np = np.asarray(nmpc_mode, dtype=object)

        mass_np = self.mass.cpu().numpy()
        inertia_np = self.inertia.cpu().numpy()
        gravity_np = self.gravity.cpu().numpy()

        for env_i in range(self.num_envs):
            m = mass_np[env_i, 0]
            I = inertia_np[env_i]
            g = gravity_np[env_i]

            self._measured_state[0:3] = pos_np[env_i]
            self._measured_state[3:6] = vel_np[env_i]
            self._measured_state[6:10] = quat_np[env_i]
            self._measured_state[10:13] = omega_body_np[env_i]

            self._p_buf[0] = m
            self._p_buf[1] = I[0, 0]
            self._p_buf[2] = I[0, 1]
            self._p_buf[3] = I[0, 2]
            self._p_buf[4] = I[1, 1]
            self._p_buf[5] = I[1, 2]
            self._p_buf[6] = I[2, 2]
            self._p_buf[7:10] = g

            yaw = sp_yaw_np[env_i]
            self._p_buf[10:13] = sp_pos_np[env_i]
            self._p_buf[13:16] = sp_vel_np[env_i]
            self._p_buf[16:19] = 0.0
            self._p_buf[19:23] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

            flags = FLAG_REINIT if np.any(np.isnan(self._prev_setpoint_pos[env_i])) else 0
            if nmpc_mode_np[env_i] == "NMPC_RECOVERY_MODE":
                flags |= FLAG_RECOVERY
            elif nmpc_mode_np[env_i] != "NMPC_FLIGHT_MODE":
                raise ValueError(f"unsupported NMPC mode for env {env_i}: {nmpc_mode_np[env_i]}")
            if flags & FLAG_REINIT:
                self._commanded_force_prev[env_i] = m * np.linalg.norm(g) / NUM_MOTORS

            B = self._alloc_matrix[env_i]
            self._p_buf[23:23 + 6 * NUM_MOTORS] = B.flatten(order='F')
            kf_offset = 23 + 6 * NUM_MOTORS
            self._p_buf[kf_offset:kf_offset + NUM_MOTORS] = self.thrust_coefficients
            self._p_buf[kf_offset + NUM_MOTORS:kf_offset + 2 * NUM_MOTORS] = self.motor_time_constants
            com_offset = kf_offset + 2 * NUM_MOTORS
            self._p_buf[com_offset:com_offset + 3] = self._com_offset[env_i]
            self._p_buf[PARAM_U_PREV_START:PARAM_U_PREV_START + NUM_MOTORS] = self._commanded_force_prev[env_i]

            self._prev_setpoint_pos[env_i] = sp_pos_np[env_i]
            self._prev_setpoint_vel[env_i] = sp_vel_np[env_i]
            self._sample_timestamp_us[env_i] += self.controller_dt_us

            self.seq += 1
            pkt = self._pack_state_packet(
                self.seq,
                flags,
                int(self._sample_timestamp_us[env_i]),
                self._measured_state,
                self._p_buf,
            )

            t_send = time.monotonic()
            self.sock.sendall(pkt)

            try:
                data = self._recv_exact(CTRL_SIZE)
                t_recv = time.monotonic()

                vals = struct.unpack(CTRL_FMT, data)
                status = vals[1]
                u_opt = np.array(vals[2:6])
                solve_time_us = vals[6]
                quat_next = np.array(vals[7:11])

                self.motor_forces[env_i] = torch.tensor(u_opt, device=self.device, dtype=torch.float32)
                self.desired_quat[env_i] = torch.tensor(quat_next, device=self.device, dtype=torch.float32)
                self._commanded_force_prev[env_i] = u_opt

                rtt_ms = (t_recv - t_send) * 1000.0
                self.timing_log.append({
                    'seq': self.seq, 'env': env_i,
                    'rtt_ms': rtt_ms, 'solve_us': solve_time_us,
                    'status': status, 'flags': flags,
                })
            except socket.timeout:
                self.timing_log.append({
                    'seq': self.seq, 'env': env_i,
                    'rtt_ms': -1, 'solve_us': -1, 'status': -999,
                })

        self.desired_quat[:] = normalize_quat(self.desired_quat)
        return self.motor_forces
