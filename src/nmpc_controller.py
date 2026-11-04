import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time

import torch
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
from controllers import BaseController
from utils import normalize_quat


from nmpc_solver_worker import (
    NUM_MOTORS,
    PARAM_MASS,
    PARAM_INERTIA_START,
    PARAM_GRAVITY_START,
    PARAM_POS_REF_START,
    PARAM_VEL_REF_START,
    PARAM_ACC_REF_START,
    PARAM_QUAT_REF_START,
    PARAM_ALLOC_START,
    PARAM_ALLOC_SIZE,
    PARAM_KF_START,
    PARAM_TAU_START,
    PARAM_COM_START,
    PARAM_DIST_FORCE_START,
    PARAM_DIST_TORQUE_START,
    PARAM_U_PREV_START,
    PARAM_COUNT,
    _acados_solver_worker,
    _solve_acados_problem,
)
REPO_ROOT = Path(__file__).resolve().parent.parent
ACADOS_JSON_FILE = str(REPO_ROOT / "acados_ocp_quadrotor.json")
ACADOS_RECOVERY_JSON_FILE = str(REPO_ROOT / "acados_ocp_quadrotor_recovery.json")
ACADOS_BUILD_LOCK_FILE = REPO_ROOT / "acados_ocp_quadrotor.lock"
ACADOS_WORKER_START_TIMEOUT_S = 300.0
ACADOS_WORKER_SOLVE_TIMEOUT_S = 600.0
ACADOS_WORKER_CLOSE_TIMEOUT_S = 30.0
VOXL_NMPC_PIPE_SOLVER_PATH = REPO_ROOT / "voxl_src" / "nmpc_pipe_solver.c"
_VOXL_NMPC_CONSTANT_NAMES = (
    "BODY_DRAG_COEFF_BX",
    "BODY_DRAG_COEFF_BY",
    "BODY_DRAG_COEFF_BZ",
)

ESTIMATOR_FLAG_RESET = 1 << 0
ESTIMATOR_FLAG_WARMUP = 1 << 1
ESTIMATOR_FLAG_NOT_READY = 1 << 2
ESTIMATOR_FLAG_FORCE_GATED = 1 << 3
ESTIMATOR_FLAG_TORQUE_GATED = 1 << 4
ESTIMATOR_FLAG_FORCE_SATURATED = 1 << 5
ESTIMATOR_FLAG_TORQUE_SATURATED = 1 << 6
ESTIMATOR_FLAG_VALID = 1 << 7
ESTIMATOR_FLAG_MEASUREMENT_READY = 1 << 8


def _load_voxl_nmpc_constants():
    source = VOXL_NMPC_PIPE_SOLVER_PATH.read_text(encoding="utf-8")
    constants = {}
    for name in _VOXL_NMPC_CONSTANT_NAMES:
        pattern = rf"^\s*#define\s+{name}\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
        match = re.search(pattern, source, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing required define {name} in {VOXL_NMPC_PIPE_SOLVER_PATH}")
        constants[name] = float(match.group(1))
    return constants


_VOXL_NMPC_CONSTANTS = _load_voxl_nmpc_constants()
BODY_DRAG_COEFF_BX = _VOXL_NMPC_CONSTANTS["BODY_DRAG_COEFF_BX"]
BODY_DRAG_COEFF_BY = _VOXL_NMPC_CONSTANTS["BODY_DRAG_COEFF_BY"]
BODY_DRAG_COEFF_BZ = _VOXL_NMPC_CONSTANTS["BODY_DRAG_COEFF_BZ"]


def _quat_to_rotmat_np(quat):
    qw, qx, qy, qz = quat
    return np.array(
        [
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def _build_inertia_matrix_np(params):
    return np.array(
        [
            [params[1], params[2], params[3]],
            [params[2], params[4], params[5]],
            [params[3], params[5], params[6]],
        ],
        dtype=np.float64,
    )


def _body_velocity_drag_force_np(velocity_body):
    return np.array(
        [
            -BODY_DRAG_COEFF_BX * velocity_body[0],
            -BODY_DRAG_COEFF_BY * velocity_body[1],
            -BODY_DRAG_COEFF_BZ * velocity_body[2],
        ],
        dtype=np.float64,
    )


class AcadosArtifactLock:
    def __init__(self, lock_path):
        self.lock_path = Path(lock_path)
        self._lock_file = None

    def __enter__(self):
        self._lock_file = open(self.lock_path, "w")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        self._lock_file.close()


class PersistentDisturbanceEstimator:
    WARMUP_TIME_S = 2.0
    WARMUP_MIN_SAMPLES = 40
    FORCE_BOUND = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    TORQUE_BOUND = np.array([0.1, 0.1, 0.02], dtype=np.float64)
    INITIAL_SIGMA_RATIO = 0.50
    READY_SIGMA_RATIO = 0.15
    FORCE_VARIANCE_FLOOR_SIGMA_RATIO = 0.005
    TORQUE_VARIANCE_FLOOR_SIGMA_RATIO = 0.05
    FORCE_PROCESS_SIGMA_RATIO = 0.03
    TORQUE_PROCESS_SIGMA_RATIO = 0.03
    FORCE_MEAS_SIGMA_RATIO = 0.03
    TORQUE_MEAS_SIGMA_RATIO = 0.20
    FORCE_ABS_GATE_RATIO = 2.0
    TORQUE_ABS_GATE_RATIO = 2.0
    FORCE_INNOVATION_GATE_NIS = 1000.0
    TORQUE_INNOVATION_GATE_NIS = 25.0
    SLEW_RATE_RATIO_PER_S = 0.75

    def __init__(self, num_envs, dt):
        self.num_envs = num_envs
        self.dt = dt

        self.estimated_force = np.zeros((num_envs, 3), dtype=np.float64)
        self.estimated_torque = np.zeros((num_envs, 3), dtype=np.float64)
        self.applied_force = np.zeros((num_envs, 3), dtype=np.float64)
        self.applied_torque = np.zeros((num_envs, 3), dtype=np.float64)
        self.force_variance = np.zeros((num_envs, 3), dtype=np.float64)
        self.torque_variance = np.zeros((num_envs, 3), dtype=np.float64)
        self.prev_velocity = np.zeros((num_envs, 3), dtype=np.float64)
        self.prev_omega = np.zeros((num_envs, 3), dtype=np.float64)
        self.initialized = np.zeros(num_envs, dtype=bool)
        self.reset_pending = np.ones(num_envs, dtype=bool)
        self.valid = np.zeros(num_envs, dtype=bool)
        self.warmup_elapsed = np.zeros(num_envs, dtype=np.float64)
        self.measurement_count = np.zeros(num_envs, dtype=np.int64)
        self.flags = np.zeros(num_envs, dtype=np.int32)
        self.force_innovation_norm = np.full(num_envs, np.nan, dtype=np.float64)
        self.torque_innovation_norm = np.full(num_envs, np.nan, dtype=np.float64)
        self.force_measurement = np.zeros((num_envs, 3), dtype=np.float64)
        self.torque_measurement = np.zeros((num_envs, 3), dtype=np.float64)
        self.force_bound = np.zeros((num_envs, 3), dtype=np.float64)
        self.torque_bound = np.zeros((num_envs, 3), dtype=np.float64)
        self.force_confidence = np.zeros(num_envs, dtype=np.float64)
        self.torque_confidence = np.zeros(num_envs, dtype=np.float64)

    def reset(self, env_ids):
        self.estimated_force[env_ids] = 0.0
        self.estimated_torque[env_ids] = 0.0
        self.applied_force[env_ids] = 0.0
        self.applied_torque[env_ids] = 0.0
        self.force_variance[env_ids] = 0.0
        self.torque_variance[env_ids] = 0.0
        self.prev_velocity[env_ids] = 0.0
        self.prev_omega[env_ids] = 0.0
        self.initialized[env_ids] = False
        self.reset_pending[env_ids] = True
        self.valid[env_ids] = False
        self.warmup_elapsed[env_ids] = 0.0
        self.measurement_count[env_ids] = 0
        self.flags[env_ids] = 0
        self.force_innovation_norm[env_ids] = np.nan
        self.torque_innovation_norm[env_ids] = np.nan
        self.force_measurement[env_ids] = 0.0
        self.torque_measurement[env_ids] = 0.0
        self.force_bound[env_ids] = 0.0
        self.torque_bound[env_ids] = 0.0
        self.force_confidence[env_ids] = 0.0
        self.torque_confidence[env_ids] = 0.0

    def _compute_bounds(self, params):
        del params
        return self.FORCE_BOUND.copy(), self.TORQUE_BOUND.copy()

    def _measurement_wrench(self, env_i, x0, params):
        quat = x0[6:10]
        quat_norm = np.linalg.norm(quat)
        if not np.isfinite(quat_norm) or quat_norm <= 1e-9:
            raise ValueError(f"invalid quaternion norm={quat_norm}")

        rotation = _quat_to_rotmat_np(quat / quat_norm)
        translational_rotation = rotation
        mass = params[PARAM_MASS]
        gravity = params[PARAM_GRAVITY_START:PARAM_GRAVITY_START + 3]
        inertia = _build_inertia_matrix_np(params)
        alloc_matrix = params[PARAM_ALLOC_START:PARAM_ALLOC_START + PARAM_ALLOC_SIZE].reshape(6, NUM_MOTORS, order="F")
        kf = params[PARAM_KF_START:PARAM_KF_START + NUM_MOTORS]
        r_com = params[PARAM_COM_START:PARAM_COM_START + 3]

        if np.any(kf <= 0.0):
            raise ValueError(f"invalid thrust coefficients={kf}")

        actual_forces = kf * x0[13:13 + NUM_MOTORS] ** 2
        wrench = alloc_matrix @ actual_forces
        force_body = wrench[0:3]
        torque_body = wrench[3:6]
        velocity_body = translational_rotation.T @ x0[3:6]
        drag_force_body = _body_velocity_drag_force_np(velocity_body)

        measured_world_acc = (x0[3:6] - self.prev_velocity[env_i]) / self.dt
        measured_body_force = mass * (translational_rotation.T @ (measured_world_acc - gravity))

        omega = x0[10:13]
        omega_dot = (omega - self.prev_omega[env_i]) / self.dt
        gravity_body = rotation.T @ (mass * gravity)
        torque_gravity = np.cross(r_com, gravity_body)
        measured_torque = inertia @ omega_dot - torque_body - torque_gravity + np.cross(omega, inertia @ omega)

        disturbance_force = measured_body_force - force_body - drag_force_body
        disturbance_torque = measured_torque
        return disturbance_force, disturbance_torque

    def update(self, env_i, x0, params):
        if not np.all(np.isfinite(x0)):
            raise ValueError("state x0 contains non-finite values")
        if not np.all(np.isfinite(params)):
            raise ValueError("parameter vector contains non-finite values")

        force_bound, torque_bound = self._compute_bounds(params)
        self.force_bound[env_i] = force_bound
        self.torque_bound[env_i] = torque_bound

        force_initial_sigma = self.INITIAL_SIGMA_RATIO * force_bound
        torque_initial_sigma = self.INITIAL_SIGMA_RATIO * torque_bound

        if not self.initialized[env_i]:
            self.initialized[env_i] = True
            self.valid[env_i] = False
            self.prev_velocity[env_i] = x0[3:6]
            self.prev_omega[env_i] = x0[10:13]
            self.force_variance[env_i] = force_initial_sigma ** 2
            self.torque_variance[env_i] = torque_initial_sigma ** 2
            self.flags[env_i] = ESTIMATOR_FLAG_RESET | ESTIMATOR_FLAG_WARMUP | ESTIMATOR_FLAG_NOT_READY
            return

        self.measurement_count[env_i] += 1
        self.warmup_elapsed[env_i] += self.dt

        force_meas, torque_meas = self._measurement_wrench(env_i, x0, params)
        if not np.all(np.isfinite(force_meas)) or not np.all(np.isfinite(torque_meas)):
            raise ValueError("disturbance measurement contains non-finite values")

        self.force_measurement[env_i] = force_meas
        self.torque_measurement[env_i] = torque_meas

        force_process_var = (self.FORCE_PROCESS_SIGMA_RATIO * force_bound) ** 2 * self.dt
        torque_process_var = (self.TORQUE_PROCESS_SIGMA_RATIO * torque_bound) ** 2 * self.dt
        force_meas_var = (self.FORCE_MEAS_SIGMA_RATIO * force_bound) ** 2
        torque_meas_var = (self.TORQUE_MEAS_SIGMA_RATIO * torque_bound) ** 2
        force_min_var = (self.FORCE_VARIANCE_FLOOR_SIGMA_RATIO * force_bound) ** 2
        torque_min_var = (self.TORQUE_VARIANCE_FLOOR_SIGMA_RATIO * torque_bound) ** 2
        force_max_var = force_bound ** 2
        torque_max_var = torque_bound ** 2

        self.force_variance[env_i] = np.clip(self.force_variance[env_i] + force_process_var, force_min_var, force_max_var)
        self.torque_variance[env_i] = np.clip(self.torque_variance[env_i] + torque_process_var, torque_min_var, torque_max_var)

        force_innovation = force_meas - self.estimated_force[env_i]
        torque_innovation = torque_meas - self.estimated_torque[env_i]
        force_innovation_cov = self.force_variance[env_i] + force_meas_var
        torque_innovation_cov = self.torque_variance[env_i] + torque_meas_var
        force_nis = np.sum(force_innovation ** 2 / force_innovation_cov)
        torque_nis = np.sum(torque_innovation ** 2 / torque_innovation_cov)
        self.force_innovation_norm[env_i] = force_nis
        self.torque_innovation_norm[env_i] = torque_nis

        force_gated = np.any(np.abs(force_meas) > self.FORCE_ABS_GATE_RATIO * force_bound) or force_nis > self.FORCE_INNOVATION_GATE_NIS
        torque_gated = np.any(np.abs(torque_meas) > self.TORQUE_ABS_GATE_RATIO * torque_bound) or torque_nis > self.TORQUE_INNOVATION_GATE_NIS

        if not force_gated:
            force_gain = self.force_variance[env_i] / force_innovation_cov
            self.estimated_force[env_i] += force_gain * force_innovation
            self.force_variance[env_i] = np.clip(
                (1.0 - force_gain) * self.force_variance[env_i],
                force_min_var,
                force_max_var,
            )

        if not torque_gated:
            torque_gain = self.torque_variance[env_i] / torque_innovation_cov
            self.estimated_torque[env_i] += torque_gain * torque_innovation
            self.torque_variance[env_i] = np.clip(
                (1.0 - torque_gain) * self.torque_variance[env_i],
                torque_min_var,
                torque_max_var,
            )

        force_before_clip = self.estimated_force[env_i].copy()
        torque_before_clip = self.estimated_torque[env_i].copy()
        self.estimated_force[env_i] = np.clip(self.estimated_force[env_i], -force_bound, force_bound)
        self.estimated_torque[env_i] = np.clip(self.estimated_torque[env_i], -torque_bound, torque_bound)
        force_saturated = not np.allclose(force_before_clip, self.estimated_force[env_i])
        torque_saturated = not np.allclose(torque_before_clip, self.estimated_torque[env_i])

        ready_force = np.all(np.sqrt(self.force_variance[env_i]) <= self.READY_SIGMA_RATIO * force_bound)
        ready_torque = np.all(np.sqrt(self.torque_variance[env_i]) <= self.READY_SIGMA_RATIO * torque_bound)
        ready = (
            self.warmup_elapsed[env_i] >= self.WARMUP_TIME_S
            and self.measurement_count[env_i] >= self.WARMUP_MIN_SAMPLES
            and ready_force
            and ready_torque
        )
        self.valid[env_i] = ready

        if ready:
            force_step = self.SLEW_RATE_RATIO_PER_S * force_bound * self.dt
            torque_step = self.SLEW_RATE_RATIO_PER_S * torque_bound * self.dt
            force_delta = np.clip(self.estimated_force[env_i] - self.applied_force[env_i], -force_step, force_step)
            torque_delta = np.clip(self.estimated_torque[env_i] - self.applied_torque[env_i], -torque_step, torque_step)
            self.applied_force[env_i] += force_delta
            self.applied_torque[env_i] += torque_delta
            self.applied_force[env_i] = np.clip(self.applied_force[env_i], -force_bound, force_bound)
            self.applied_torque[env_i] = np.clip(self.applied_torque[env_i], -torque_bound, torque_bound)
        else:
            self.applied_force[env_i] = 0.0
            self.applied_torque[env_i] = 0.0

        self.force_confidence[env_i] = 1.0 - float(np.mean(np.sqrt(self.force_variance[env_i]) / force_bound))
        self.torque_confidence[env_i] = 1.0 - float(np.mean(np.sqrt(self.torque_variance[env_i]) / torque_bound))

        flags = ESTIMATOR_FLAG_MEASUREMENT_READY
        if self.reset_pending[env_i]:
            flags |= ESTIMATOR_FLAG_RESET
        if not ready:
            flags |= ESTIMATOR_FLAG_WARMUP | ESTIMATOR_FLAG_NOT_READY
        if force_gated:
            flags |= ESTIMATOR_FLAG_FORCE_GATED
        if torque_gated:
            flags |= ESTIMATOR_FLAG_TORQUE_GATED
        if force_saturated:
            flags |= ESTIMATOR_FLAG_FORCE_SATURATED
        if torque_saturated:
            flags |= ESTIMATOR_FLAG_TORQUE_SATURATED
        if ready:
            flags |= ESTIMATOR_FLAG_VALID
        self.flags[env_i] = flags

        self.prev_velocity[env_i] = x0[3:6]
        self.prev_omega[env_i] = x0[10:13]
        self.reset_pending[env_i] = False

        if not np.all(np.isfinite(self.estimated_force[env_i])) or not np.all(np.isfinite(self.estimated_torque[env_i])):
            raise ValueError("estimated disturbance became non-finite")
        if not np.all(np.isfinite(self.applied_force[env_i])) or not np.all(np.isfinite(self.applied_torque[env_i])):
            raise ValueError("applied disturbance became non-finite")
        if not np.all(np.isfinite(self.force_variance[env_i])) or not np.all(np.isfinite(self.torque_variance[env_i])):
            raise ValueError("disturbance variance became non-finite")


def _quat_mult_casadi(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def _quaternion_error_casadi(q, q_ref):
    q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
    q_err = _quat_mult_casadi(q_ref_inv, q)
    return ca.vertcat(q_err[0] - 1, q_err[1], q_err[2], q_err[3])


def _rotation_vector_error_casadi(q, q_ref):
    q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
    q_err = _quat_mult_casadi(q_ref_inv, q)
    shortest_arc_sign = ca.if_else(q_err[0] >= 0.0, 1.0, -1.0)
    q_err_short = shortest_arc_sign * q_err
    q_err_vec = q_err_short[1:4]
    q_err_vec_norm = ca.norm_2(q_err_vec)
    rotation_angle = 2.0 * ca.atan2(q_err_vec_norm, q_err_short[0])
    rotation_scale = ca.if_else(q_err_vec_norm > 1e-9, rotation_angle / q_err_vec_norm, 2.0)
    return rotation_scale * q_err_vec


def _body_velocity_drag_force_casadi(velocity_body):
    return ca.vertcat(
        -BODY_DRAG_COEFF_BX * velocity_body[0],
        -BODY_DRAG_COEFF_BY * velocity_body[1],
        -BODY_DRAG_COEFF_BZ * velocity_body[2],
    )


def _validate_stage_weight_vector(name, weights, expected_length):
    weights_array = np.array(weights, dtype=np.float64)
    if weights_array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence, got shape {weights_array.shape}")
    if len(weights_array) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} entries, got {len(weights_array)}")
    if np.any(~np.isfinite(weights_array)):
        raise ValueError(f"{name} contains non-finite entries: {weights_array}")
    if np.any(weights_array < 0.0):
        raise ValueError(f"{name} must be non-negative, got {weights_array}")
    return weights_array


def _build_quadrotor_model(model_name):
    """Build CasADi rigid-body quadrotor model with first-order motor dynamics.

    State x = [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, rps0..rps3] (17)
      px,py,pz = root link position
      vx,vy,vz = root link velocity
    Control u = [f1_cmd,f2_cmd,f3_cmd,f4_cmd] commanded motor forces (NUM_MOTORS)

    Parameters p = [m, Ixx,Ixy,Ixz,Iyy,Iyz,Izz, gx,gy,gz,
                    px_ref,py_ref,pz_ref, vx_ref,vy_ref,vz_ref,
                    qw_ref,qx_ref,qy_ref,qz_ref,
                    B_00..B_53,
                    kf0..kf3, tau0..tau3,
                    r_com_x,r_com_y,r_com_z,
                    dFbx,dFby,dFbz,
                    dTbx,dTby,dTbz] (61)
    where B is the 6xNUM_MOTORS allocation matrix,
    kf are thrust coefficients (force = kf * rps^2),
    tau are motor time constants,
    r_com is the system COM offset from the base link origin (body frame).
    """
    n_alloc = PARAM_ALLOC_SIZE
    n_p = PARAM_COUNT

    model = AcadosModel()
    model.name = model_name

    nx = 13 + NUM_MOTORS
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", NUM_MOTORS)
    p = ca.SX.sym("p", n_p)

    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]
    motor_rps = x[13:13 + NUM_MOTORS]

    m = p[0]
    Ixx, Ixy, Ixz, Iyy, Iyz, Izz = p[1], p[2], p[3], p[4], p[5], p[6]
    g = p[7:10]

    I_mat = ca.vertcat(
        ca.horzcat(Ixx, Ixy, Ixz),
        ca.horzcat(Ixy, Iyy, Iyz),
        ca.horzcat(Ixz, Iyz, Izz),
    )

    B = ca.reshape(p[PARAM_ALLOC_START:PARAM_ALLOC_START + n_alloc], 6, NUM_MOTORS)
    kf = p[PARAM_KF_START:PARAM_KF_START + NUM_MOTORS]
    tau = p[PARAM_TAU_START:PARAM_TAU_START + NUM_MOTORS]
    r_com = p[PARAM_COM_START:PARAM_COM_START + 3]
    disturbance_force_body = p[PARAM_DIST_FORCE_START:PARAM_DIST_FORCE_START + 3]
    disturbance_torque_body = p[PARAM_DIST_TORQUE_START:PARAM_DIST_TORQUE_START + 3]

    actual_forces = kf * motor_rps**2
    wrench = B @ actual_forces
    F_body = wrench[0:3]
    tau_body = wrench[3:6]

    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    R = ca.vertcat(
        ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
        ca.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
        ca.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)),
    )

    vel_body = R.T @ vel
    drag_force_body = _body_velocity_drag_force_casadi(vel_body)
    vel_dot = g + R @ (F_body + drag_force_body + disturbance_force_body) / m
    omega_quat = ca.vertcat(0, omega[0], omega[1], omega[2])
    quat_dot = 0.5 * _quat_mult_casadi(quat, omega_quat)
    g_body = R.T @ (m * g)
    tau_grav = ca.cross(r_com, g_body)
    I_omega = I_mat @ omega
    omega_dot = ca.solve(I_mat, tau_body + tau_grav + disturbance_torque_body - ca.cross(omega, I_omega))

    pos_dot = vel

    desired_rps = ca.sqrt(u / kf)
    rps_dot = (desired_rps - motor_rps) / tau

    x_dot = ca.vertcat(pos_dot, vel_dot, quat_dot, omega_dot, rps_dot)

    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = x_dot
    model.xdot = ca.SX.sym("xdot", nx)
    model.f_impl_expr = model.xdot - x_dot

    return model, n_p


def build_nmpc_ocp(
    dt,
    N,
    Q_pos,
    Q_vel,
    Q_acc,
    Q_att,
    Q_omega,
    R,
    R_delta_u,
    terminal_factor,
    motor_force_min,
    motor_force_max,
    model_name,
):
    model, n_p = _build_quadrotor_model(model_name)

    ocp = AcadosOcp()
    ocp.model = model

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    ocp.dims.np = n_p
    ocp.parameter_values = np.zeros(n_p)

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p

    pos_ref = p[PARAM_POS_REF_START:PARAM_POS_REF_START + 3]
    vel_ref = p[PARAM_VEL_REF_START:PARAM_VEL_REF_START + 3]
    acc_ref = p[PARAM_ACC_REF_START:PARAM_ACC_REF_START + 3]
    quat_ref = p[PARAM_QUAT_REF_START:PARAM_QUAT_REF_START + 4]
    u_prev_ref = p[PARAM_U_PREV_START:PARAM_U_PREV_START + NUM_MOTORS]

    e_pos = x[0:3] - pos_ref
    e_vel = x[3:6] - vel_ref
    e_acc = model.f_expl_expr[3:6] - acc_ref
    e_ori = _rotation_vector_error_casadi(x[6:10], quat_ref)
    e_omega = x[10:13]
    e_delta_u = u - u_prev_ref

    Q_rps = 0.0
    Q_pos_vec = _validate_stage_weight_vector("Q_pos", Q_pos, 3)
    Q_vel_vec = _validate_stage_weight_vector("Q_vel", Q_vel, 3)
    Q_acc_vec = _validate_stage_weight_vector("Q_acc", Q_acc, 3)
    Q_att_vec = _validate_stage_weight_vector("Q_att", Q_att, 3)
    Q_omega_vec = _validate_stage_weight_vector("Q_omega", Q_omega, 3)
    Q_state_stage = np.concatenate([Q_pos_vec, Q_vel_vec, Q_acc_vec, Q_att_vec, Q_omega_vec])
    Q_state_terminal = np.concatenate([Q_pos_vec, Q_vel_vec, Q_att_vec, Q_omega_vec]) * terminal_factor
    Q_stage = np.concatenate([Q_state_stage, np.full(NUM_MOTORS, Q_rps, dtype=np.float64)])
    R_diag = np.full(NUM_MOTORS, R, dtype=np.float64)
    R_delta_u_diag = np.full(NUM_MOTORS, R_delta_u, dtype=np.float64)

    ny = 15 + 2 * NUM_MOTORS
    ny_e = 12

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.model.cost_y_expr = ca.vertcat(e_pos, e_vel, e_acc, e_ori, e_omega, u, e_delta_u)
    ocp.model.cost_y_expr_e = ca.vertcat(e_pos, e_vel, e_ori, e_omega)

    W_stage = np.diag(np.concatenate([Q_stage[:15], R_diag, R_delta_u_diag]))
    W_terminal = np.diag(Q_state_terminal)
    ocp.cost.W = W_stage
    ocp.cost.W_e = W_terminal
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    ocp.constraints.lbu = np.full(NUM_MOTORS, motor_force_min)
    ocp.constraints.ubu = np.full(NUM_MOTORS, motor_force_max)
    ocp.constraints.idxbu = np.arange(NUM_MOTORS)

    ocp.constraints.x0 = np.zeros(model.x.shape[0])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    return ocp, n_p


def _select_mode_results(nominal_results, recovery_results, active_nmpc_mode):
    selected_results = {
        "solver_statuses": np.zeros_like(nominal_results["solver_statuses"]),
        "solver_costs": np.zeros_like(nominal_results["solver_costs"]),
        "u_opt": np.zeros_like(nominal_results["u_opt"]),
        "x1": np.zeros_like(nominal_results["x1"]),
        "x_end": np.zeros_like(nominal_results["x_end"]),
        "horizon_controls": np.zeros_like(nominal_results["horizon_controls"]),
    }
    recovery_mask = active_nmpc_mode == "NMPC_RECOVERY_MODE"
    flight_mask = active_nmpc_mode == "NMPC_FLIGHT_MODE"
    coast_mask = active_nmpc_mode == "NMPC_COAST_MODE"
    if np.any(~(recovery_mask | flight_mask | coast_mask)):
        invalid_modes = active_nmpc_mode[~(recovery_mask | flight_mask | coast_mask)].tolist()
        raise ValueError(f"unsupported NMPC mode(s): {invalid_modes}")
    # Coasting environments keep zeroed entries; the caller overwrites them with
    # the fixed coast motor commands.
    for result_key in selected_results:
        selected_results[result_key][flight_mask] = nominal_results[result_key][flight_mask]
        selected_results[result_key][recovery_mask] = recovery_results[result_key][recovery_mask]
    return selected_results


class ParallelAcadosOcpSolvers:
    """Owns one persistent acados solver process per simulation environment."""

    def __init__(self, num_envs, json_file, horizon):
        if num_envs <= 1:
            raise ValueError("ParallelAcadosOcpSolvers requires num_envs > 1")

        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"acados json file not found: {json_path}")

        self.num_envs = num_envs
        self.horizon = horizon
        self.json_file = str(json_path)
        self._context = mp.get_context("forkserver")
        self._context.set_forkserver_preload(["numpy", "acados_template"])
        self._connections = []
        self._processes = []
        self._closed = False

        try:
            self._start_workers()
            self._wait_for_ready()
        except BaseException:
            self._terminate_workers()
            raise

    def _start_worker(self, env_i):
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_acados_solver_worker,
            args=(env_i, child_conn, self.json_file, self.horizon, os.getpid()),
            name=f"nmpc-acados-env-{env_i}",
        )
        process.daemon = True
        try:
            process.start()
        except BaseException:
            parent_conn.close()
            raise
        finally:
            child_conn.close()

        if env_i == len(self._connections):
            self._connections.append(parent_conn)
            self._processes.append(process)
            return
        if env_i > len(self._connections):
            raise IndexError(f"cannot start acados worker for env {env_i}: invalid worker index")

        self._connections[env_i] = parent_conn
        self._processes[env_i] = process

    def _start_workers(self):
        for env_i in range(self.num_envs):
            self._start_worker(env_i)

    def _worker_failure_message(self, env_i):
        process = self._processes[env_i]
        return (
            f"acados worker for env {env_i} exited before responding "
            f"(pid={process.pid}, exitcode={process.exitcode})"
        )

    def _receive_worker_messages(self, pending_envs, timeout_s):
        deadline = time.monotonic() + timeout_s
        messages = {}

        while pending_envs:
            for env_i in tuple(pending_envs):
                connection = self._connections[env_i]
                process = self._processes[env_i]

                if connection.poll(0.0):
                    try:
                        messages[env_i] = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(self._worker_failure_message(env_i)) from exc
                    if messages[env_i][0] == "error":
                        self._raise_worker_error(env_i, messages[env_i][1])
                    pending_envs.remove(env_i)
                elif process.exitcode is not None:
                    raise RuntimeError(self._worker_failure_message(env_i))

            if pending_envs:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"timed out waiting for acados workers: {sorted(pending_envs)}")
                time.sleep(min(0.01, remaining))

        return messages

    def _raise_worker_error(self, env_i, payload):
        raise RuntimeError(
            f"acados worker for env {env_i} failed with {payload['type']}: {payload['message']}\n"
            f"{payload['traceback']}"
        )

    def _wait_for_ready(self):
        self._wait_for_ready_subset(set(range(self.num_envs)))

    def _wait_for_ready_subset(self, pending_envs):
        messages = self._receive_worker_messages(pending_envs, ACADOS_WORKER_START_TIMEOUT_S)
        for env_i, message in messages.items():
            tag, payload = message
            if tag == "error":
                self._raise_worker_error(env_i, payload)
            if tag != "ready":
                raise RuntimeError(f"unexpected acados worker startup message for env {env_i}: {tag}")

    def solve_all(self, x0_all, params_all, hover_force_all, needs_reinit_all):
        if self._closed:
            raise RuntimeError("parallel acados solvers are closed")
        if x0_all.shape[0] != self.num_envs:
            raise ValueError(f"x0_all has {x0_all.shape[0]} environments, expected {self.num_envs}")
        if params_all.shape[0] != self.num_envs:
            raise ValueError(f"params_all has {params_all.shape[0]} environments, expected {self.num_envs}")
        if params_all.shape[1:] != (self.horizon + 1, PARAM_COUNT):
            raise ValueError(
                f"params_all has stage shape {params_all.shape[1:]}, expected {(self.horizon + 1, PARAM_COUNT)}"
            )
        if hover_force_all.shape[0] != self.num_envs:
            raise ValueError(f"hover_force_all has {hover_force_all.shape[0]} environments, expected {self.num_envs}")
        if needs_reinit_all.shape[0] != self.num_envs:
            raise ValueError(f"needs_reinit_all has {needs_reinit_all.shape[0]} environments, expected {self.num_envs}")

        try:
            for env_i in range(self.num_envs):
                process = self._processes[env_i]
                if process.exitcode is not None:
                    raise RuntimeError(self._worker_failure_message(env_i))
                self._connections[env_i].send(
                    (
                        "solve",
                        np.array(x0_all[env_i], dtype=np.float64, copy=True),
                        np.array(params_all[env_i], dtype=np.float64, copy=True),
                        float(hover_force_all[env_i]),
                        bool(needs_reinit_all[env_i]),
                    )
                )

            pending_envs = set(range(self.num_envs))
            messages = self._receive_worker_messages(pending_envs, ACADOS_WORKER_SOLVE_TIMEOUT_S)
        except BaseException:
            self._terminate_workers()
            raise

        solver_statuses = np.zeros(self.num_envs, dtype=np.int32)
        solver_costs = np.zeros(self.num_envs, dtype=np.float64)
        u_opt = np.zeros((self.num_envs, NUM_MOTORS), dtype=np.float64)
        x1 = np.zeros((self.num_envs, 13 + NUM_MOTORS), dtype=np.float64)
        x_end = np.zeros((self.num_envs, 13 + NUM_MOTORS), dtype=np.float64)
        horizon_controls = np.zeros((self.num_envs, self.horizon, NUM_MOTORS), dtype=np.float64)

        try:
            for env_i, message in messages.items():
                tag, payload = message
                if tag == "error":
                    self._raise_worker_error(env_i, payload)
                if tag != "result":
                    raise RuntimeError(f"unexpected acados worker solve message for env {env_i}: {tag}")

                solver_statuses[env_i] = payload["status"]
                solver_costs[env_i] = payload["cost"]
                u_opt[env_i] = payload["u_opt"]
                x1[env_i] = payload["x1"]
                x_end[env_i] = payload["x_end"]
                horizon_controls[env_i] = payload["horizon_controls"]
        except BaseException:
            self._terminate_workers()
            raise

        return {
            "solver_statuses": solver_statuses,
            "solver_costs": solver_costs,
            "u_opt": u_opt,
            "x1": x1,
            "x_end": x_end,
            "horizon_controls": horizon_controls,
        }

    def _terminate_workers(self):
        for process in self._processes:
            if process is None:
                continue
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            if process is None:
                continue
            process.join(ACADOS_WORKER_CLOSE_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(ACADOS_WORKER_CLOSE_TIMEOUT_S)
        for connection in self._connections:
            if connection is None:
                continue
            connection.close()
        self._closed = True

    def _close_worker(self, env_i):
        if self._closed:
            raise RuntimeError("parallel acados solvers are closed")

        process = self._processes[env_i]
        connection = self._connections[env_i]
        if process is None or connection is None:
            raise RuntimeError(f"acados worker for env {env_i} is not running")
        if process.exitcode is not None:
            raise RuntimeError(self._worker_failure_message(env_i))

        connection.send(("close",))
        messages = self._receive_worker_messages({env_i}, ACADOS_WORKER_CLOSE_TIMEOUT_S)
        tag, payload = messages[env_i]
        if tag == "error":
            self._raise_worker_error(env_i, payload)
        if tag != "closed":
            raise RuntimeError(f"unexpected acados worker close message for env {env_i}: {tag}")

        process.join(ACADOS_WORKER_CLOSE_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(ACADOS_WORKER_CLOSE_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(ACADOS_WORKER_CLOSE_TIMEOUT_S)
        if process.is_alive():
            raise RuntimeError(f"acados worker did not exit after close: pid={process.pid}")

        connection.close()
        self._connections[env_i] = None
        self._processes[env_i] = None

    def restart_workers(self, env_ids):
        if self._closed:
            raise RuntimeError("parallel acados solvers are closed")

        env_ids_array = np.array(env_ids, dtype=np.int64, copy=False).reshape(-1)
        if env_ids_array.size == 0:
            raise ValueError("env_ids must contain at least one environment index")

        unique_env_ids = np.unique(env_ids_array)
        if np.any(unique_env_ids < 0) or np.any(unique_env_ids >= self.num_envs):
            raise IndexError(
                f"env_ids must be in [0, {self.num_envs}), got {unique_env_ids.tolist()}"
            )

        try:
            for env_i in unique_env_ids.tolist():
                self._close_worker(env_i)
            for env_i in unique_env_ids.tolist():
                self._start_worker(env_i)
            self._wait_for_ready_subset(set(unique_env_ids.tolist()))
        except BaseException:
            self._terminate_workers()
            raise

    def close(self):
        if self._closed:
            return

        try:
            for env_i in range(self.num_envs):
                process = self._processes[env_i]
                if process.exitcode is not None:
                    raise RuntimeError(self._worker_failure_message(env_i))
                self._connections[env_i].send(("close",))

            pending_envs = set(range(self.num_envs))
            messages = self._receive_worker_messages(pending_envs, ACADOS_WORKER_CLOSE_TIMEOUT_S)
            for env_i, message in messages.items():
                tag, payload = message
                if tag == "error":
                    self._raise_worker_error(env_i, payload)
                if tag != "closed":
                    raise RuntimeError(f"unexpected acados worker close message for env {env_i}: {tag}")
        except BaseException:
            self._terminate_workers()
            raise

        for process in self._processes:
            process.join(ACADOS_WORKER_CLOSE_TIMEOUT_S)
            if process.is_alive():
                self._terminate_workers()
                raise RuntimeError(f"acados worker did not exit after close: pid={process.pid}")
        for connection in self._connections:
            connection.close()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except BaseException:
            try:
                self._terminate_workers()
            except BaseException:
                pass


class NMPCController(BaseController):
    """Nonlinear MPC controller outputting motor forces directly.

    Uses one in-process solver for a single environment and independent worker
    solvers for multi-environment evaluation.
    """

    def __init__(self, num_envs, device, dt, nmpc_dt, N, Q_pos, Q_vel, Q_acc, Q_att, Q_omega,
                 R, R_delta_u, terminal_factor, recovery_cost_config, motor_rps_min, motor_force_min,
                 motor_force_max, thrust_coefficients, motor_time_constants):
        super().__init__(num_envs, device)

        self.dt = dt
        self.nmpc_dt = nmpc_dt
        self.N = N
        self.nx = 13 + NUM_MOTORS
        self.nu = NUM_MOTORS
        self.model, self.np = _build_quadrotor_model("quadrotor_nmpc_model")
        self.n_p = self.np

        self.thrust_coefficients = np.array(thrust_coefficients, dtype=np.float64)
        self.motor_time_constants = np.array(motor_time_constants, dtype=np.float64)

        motor_rps_min = float(motor_rps_min)
        if not np.isfinite(motor_rps_min) or motor_rps_min <= 0.0:
            raise ValueError(f"motor_rps_min must be a positive finite value, got {motor_rps_min}")
        # Fixed per-motor forces commanded while coasting (NMPC_COAST_MODE),
        # mirroring the pipe solver which commands motor_rps_min on all motors.
        self._coast_motor_forces = self.thrust_coefficients * motor_rps_min ** 2
        self._solver_build_config = {
            "nmpc_dt": nmpc_dt,
            "N": N,
            "nominal_cost_config": {
                "Q_pos": Q_pos,
                "Q_vel": Q_vel,
                "Q_acc": Q_acc,
                "Q_att": Q_att,
                "Q_omega": Q_omega,
                "R": R,
                "R_delta_u": R_delta_u,
                "terminal_factor": terminal_factor,
            },
            "recovery_cost_config": recovery_cost_config,
            "motor_force_min": motor_force_min,
            "motor_force_max": motor_force_max,
        }

        self.motor_forces = torch.zeros((num_envs, NUM_MOTORS), device=device)
        self.desired_quat = torch.zeros((num_envs, 4), device=device)
        self.desired_quat[:, 0] = 1.0

        self._parallel_solvers = None
        self._parallel_recovery_solvers = None
        self.solver = None
        self.recovery_solver = None
        if num_envs == 1:
            self.solver = self._build_solver(
                nmpc_dt, N, self._solver_build_config["nominal_cost_config"], motor_force_min, motor_force_max,
                ACADOS_JSON_FILE, "quadrotor_nmpc",
            )
            self.recovery_solver = self._build_solver(
                nmpc_dt, N, self._solver_build_config["recovery_cost_config"], motor_force_min, motor_force_max,
                ACADOS_RECOVERY_JSON_FILE, "quadrotor_nmpc_recovery",
            )
            self._parallel_solvers = None
            self._parallel_recovery_solvers = None
        else:
            self.solver = None
            self.recovery_solver = None
            self._ensure_solver_artifacts(
                nmpc_dt, N, self._solver_build_config["nominal_cost_config"], motor_force_min, motor_force_max,
                ACADOS_JSON_FILE, "quadrotor_nmpc",
            )
            self._ensure_solver_artifacts(
                nmpc_dt, N, self._solver_build_config["recovery_cost_config"], motor_force_min, motor_force_max,
                ACADOS_RECOVERY_JSON_FILE, "quadrotor_nmpc_recovery",
            )
            self._parallel_solvers = ParallelAcadosOcpSolvers(num_envs, ACADOS_JSON_FILE, N)
            try:
                self._parallel_recovery_solvers = ParallelAcadosOcpSolvers(num_envs, ACADOS_RECOVERY_JSON_FILE, N)
            except BaseException:
                self._parallel_solvers.close()
                raise

        self.motor_force_min = motor_force_min
        self.motor_force_max = motor_force_max

        self._alloc_matrix = None
        self._com_offset = None
        self._motor_rps_hat = np.zeros((num_envs, NUM_MOTORS), dtype=np.float64)
        self._commanded_rps_prev = np.zeros((num_envs, NUM_MOTORS), dtype=np.float64)
        self._commanded_force_prev = np.zeros((num_envs, NUM_MOTORS), dtype=np.float64)
        self._previous_horizon_controls = np.zeros((num_envs, self.N, NUM_MOTORS), dtype=np.float64)
        self._motor_estimator_initialized = np.zeros(num_envs, dtype=bool)
        self._measured_motor_rps = np.full((num_envs, NUM_MOTORS), np.nan, dtype=np.float64)
        self._disturbance_estimator = PersistentDisturbanceEstimator(num_envs, dt)
        self.debug_data = {}
        self._prev_setpoint_pos = np.full((num_envs, 3), np.nan)
        self._prev_setpoint_vel = np.full((num_envs, 3), np.nan)
        self._prev_nmpc_mode = np.full(num_envs, "", dtype=object)

    def set_com_offset(self, com_offset_np):
        self._com_offset = com_offset_np

    def set_allocation_matrix(self, allocation_matrix_np):
        self._alloc_matrix = allocation_matrix_np

    def set_measured_motor_rps(self, motor_rps_np):
        motor_rps_np = np.ascontiguousarray(motor_rps_np, dtype=np.float64)
        expected_shape = (self.num_envs, NUM_MOTORS)
        if motor_rps_np.shape != expected_shape:
            raise ValueError(f"motor_rps_np must have shape {expected_shape}, got {motor_rps_np.shape}")
        if not np.all(np.isfinite(motor_rps_np)):
            raise ValueError("motor_rps_np contains non-finite values")
        self._measured_motor_rps[:] = motor_rps_np

    def _build_solver(self, dt, N, cost_config, f_min, f_max, json_file, model_name):
        ocp, _ = build_nmpc_ocp(
            dt, N,
            cost_config["Q_pos"],
            cost_config["Q_vel"],
            cost_config["Q_acc"],
            cost_config["Q_att"],
            cost_config["Q_omega"],
            cost_config["R"],
            cost_config["R_delta_u"],
            cost_config["terminal_factor"],
            f_min,
            f_max,
            model_name,
        )
        ocp.code_gen_opts.code_export_directory = str(self._solver_code_export_directory(json_file))
        with AcadosArtifactLock(ACADOS_BUILD_LOCK_FILE):
            return AcadosOcpSolver(ocp, json_file=json_file)

    def _solver_code_export_directory(self, json_file):
        json_path = Path(json_file).resolve()
        if json_path == Path(ACADOS_JSON_FILE).resolve():
            return REPO_ROOT / "c_generated_code"
        if json_path == Path(ACADOS_RECOVERY_JSON_FILE).resolve():
            return REPO_ROOT / "c_generated_code_recovery"
        raise ValueError(f"unknown acados solver JSON path: {json_path}")

    def _solver_shared_lib_path(self, json_file):
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"acados json file not found: {json_path}")

        with open(json_path, "r") as f:
            solver_json = json.load(f)

        code_export_directory = Path(solver_json["code_gen_opts"]["code_export_directory"])
        if not code_export_directory.is_absolute():
            code_export_directory = json_path.parent / code_export_directory
        solver_name = solver_json["name"]
        return code_export_directory / f"{get_shared_lib_prefix()}acados_ocp_solver_{solver_name}{get_shared_lib_ext()}"

    def _solver_shared_lib_exists(self, json_file):
        return self._solver_shared_lib_path(json_file).exists()

    def _remove_compiled_solver_artifacts(self, code_export_directory):
        code_export_path = Path(code_export_directory)
        if not code_export_path.exists():
            raise FileNotFoundError(f"acados code export directory not found: {code_export_path}")
        for pattern in ("*.o", f"*{get_shared_lib_ext()}"):
            for artifact_path in code_export_path.rglob(pattern):
                if artifact_path.is_dir():
                    raise IsADirectoryError(f"expected compiled artifact file, got directory: {artifact_path}")
                artifact_path.unlink()

    def _build_solver_artifacts(self, code_export_directory, json_file):
        self._remove_compiled_solver_artifacts(code_export_directory)
        AcadosOcpSolver.build(
            code_export_directory,
            with_cython=False,
            cmake_builder=None,
            verbose=True,
        )
        shared_lib_path = self._solver_shared_lib_path(json_file)
        if not shared_lib_path.exists():
            raise RuntimeError(f"acados build did not create expected shared library: {shared_lib_path}")

    def _validate_solver_artifacts_loadable(self, json_file):
        solver = AcadosOcpSolver(None, json_file=json_file, build=False, generate=False, verbose=False)
        del solver

    def _ensure_solver_artifacts(self, dt, N, cost_config, f_min, f_max, json_file, model_name):
        with AcadosArtifactLock(ACADOS_BUILD_LOCK_FILE):
            ocp, _ = build_nmpc_ocp(
                dt, N,
                cost_config["Q_pos"],
                cost_config["Q_vel"],
                cost_config["Q_acc"],
                cost_config["Q_att"],
                cost_config["Q_omega"],
                cost_config["R"],
                cost_config["R_delta_u"],
                cost_config["terminal_factor"],
                f_min,
                f_max,
                model_name,
            )
            ocp.code_gen_opts.code_export_directory = str(self._solver_code_export_directory(json_file))
            ocp.code_gen_opts.json_file = json_file
            ocp.make_consistent(verbose=True)

            reuse_possible = AcadosOcpSolver.is_code_reuse_possible(None, ocp, json_file, verbose=True)
            if not reuse_possible:
                AcadosOcpSolver.generate(
                    ocp,
                    json_file=json_file,
                    simulink_opts=None,
                    cmake_builder=None,
                    verbose=True,
                )
                self._build_solver_artifacts(ocp.code_gen_opts.code_export_directory, json_file)
                self._validate_solver_artifacts_loadable(json_file)
                return

            if not self._solver_shared_lib_exists(json_file):
                self._build_solver_artifacts(ocp.code_gen_opts.code_export_directory, json_file)
            self._validate_solver_artifacts_loadable(json_file)

    def _get_hover_motor_rps(self, hover_force):
        if np.any(self.thrust_coefficients <= 0.0):
            raise ValueError("thrust_coefficients must be strictly positive")
        return np.sqrt(hover_force / self.thrust_coefficients)

    def _convert_motor_forces_to_rps(self, motor_forces):
        if np.any(self.thrust_coefficients <= 0.0):
            raise ValueError("thrust_coefficients must be strictly positive")
        return np.sqrt(np.maximum(motor_forces, 0.0) / self.thrust_coefficients)

    def _initialize_motor_estimator(self, env_i):
        measured_rps = self._measured_motor_rps[env_i]
        if not np.all(np.isfinite(measured_rps)):
            raise RuntimeError(
                f"measured motor rps for env {env_i} must be set before NMPC update, got {measured_rps.tolist()}"
            )
        measured_force = self.thrust_coefficients * measured_rps ** 2
        self._motor_rps_hat[env_i] = measured_rps
        self._commanded_rps_prev[env_i] = measured_rps
        self._commanded_force_prev[env_i] = measured_force
        self._previous_horizon_controls[env_i] = measured_force
        self._motor_estimator_initialized[env_i] = True

    def _propagate_motor_estimator(self, env_i):
        if np.any(self.motor_time_constants <= 0.0):
            raise ValueError("motor_time_constants must be strictly positive")

        previous_command = self._commanded_rps_prev[env_i]
        current_estimate = self._motor_rps_hat[env_i]
        decay = np.exp(-self.dt / self.motor_time_constants)
        self._motor_rps_hat[env_i] = previous_command + (current_estimate - previous_command) * decay

    def reset(self, env_ids):
        ids = env_ids.cpu().numpy()
        self._prev_setpoint_pos[ids] = np.nan
        self._prev_setpoint_vel[ids] = np.nan
        self._motor_rps_hat[ids] = 0.0
        self._commanded_rps_prev[ids] = 0.0
        self._commanded_force_prev[ids] = 0.0
        self._previous_horizon_controls[ids] = 0.0
        self._motor_estimator_initialized[ids] = False
        self._measured_motor_rps[ids] = np.nan
        self._disturbance_estimator.reset(ids)
        self._prev_nmpc_mode[ids] = ""

    def restart_acados(self, env_ids):
        ids = env_ids.cpu().numpy()
        if ids.size == 0:
            raise ValueError("env_ids must contain at least one environment index")
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise IndexError(f"env_ids must be in [0, {self.num_envs}), got {ids.tolist()}")

        self._prev_setpoint_pos[ids] = np.nan
        self._prev_setpoint_vel[ids] = np.nan
        self._previous_horizon_controls[ids] = 0.0
        self._prev_nmpc_mode[ids] = ""

        if self.num_envs == 1:
            if ids.size != 1 or int(ids[0]) != 0:
                raise ValueError(f"single-environment controller can only restart env 0, got {ids.tolist()}")
            if self.solver is None:
                raise RuntimeError("single-environment acados solver is not initialized")
            del self.solver
            del self.recovery_solver
            self.solver = self._build_solver(
                self._solver_build_config["nmpc_dt"],
                self._solver_build_config["N"],
                self._solver_build_config["nominal_cost_config"],
                self._solver_build_config["motor_force_min"],
                self._solver_build_config["motor_force_max"],
                ACADOS_JSON_FILE,
                "quadrotor_nmpc",
            )
            self.recovery_solver = self._build_solver(
                self._solver_build_config["nmpc_dt"],
                self._solver_build_config["N"],
                self._solver_build_config["recovery_cost_config"],
                self._solver_build_config["motor_force_min"],
                self._solver_build_config["motor_force_max"],
                ACADOS_RECOVERY_JSON_FILE,
                "quadrotor_nmpc_recovery",
            )
            return

        if self._parallel_solvers is None:
            raise RuntimeError("parallel acados solvers are not initialized")
        self._parallel_solvers.restart_workers(ids)
        self._parallel_recovery_solvers.restart_workers(ids)

    def close(self):
        try:
            if self._parallel_solvers is not None:
                self._parallel_solvers.close()
        finally:
            self._parallel_solvers = None
            try:
                if self._parallel_recovery_solvers is not None:
                    self._parallel_recovery_solvers.close()
            finally:
                self._parallel_recovery_solvers = None
                self.solver = None
                self.recovery_solver = None

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass

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

        solver_statuses = np.zeros(self.num_envs, dtype=np.int32)
        solver_costs = np.zeros(self.num_envs, dtype=np.float64)
        predicted_pos_end = np.zeros((self.num_envs, 3), dtype=np.float64)
        predicted_vel_end = np.zeros((self.num_envs, 3), dtype=np.float64)
        predicted_quat_end = np.zeros((self.num_envs, 4), dtype=np.float64)
        horizon_controls = np.zeros((self.num_envs, self.N, NUM_MOTORS), dtype=np.float64)
        estimated_motor_rps = np.zeros((self.num_envs, NUM_MOTORS), dtype=np.float64)
        previous_commanded_motor_rps = np.zeros((self.num_envs, NUM_MOTORS), dtype=np.float64)
        commanded_motor_rps = np.zeros((self.num_envs, NUM_MOTORS), dtype=np.float64)
        estimated_disturbance_force = np.zeros((self.num_envs, 3), dtype=np.float64)
        estimated_disturbance_torque = np.zeros((self.num_envs, 3), dtype=np.float64)
        applied_disturbance_force = np.zeros((self.num_envs, 3), dtype=np.float64)
        applied_disturbance_torque = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_force_variance = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_torque_variance = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_force_measurement = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_torque_measurement = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_force_bound = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_torque_bound = np.zeros((self.num_envs, 3), dtype=np.float64)
        disturbance_flags = np.zeros(self.num_envs, dtype=np.int32)
        disturbance_force_confidence = np.zeros(self.num_envs, dtype=np.float64)
        disturbance_torque_confidence = np.zeros(self.num_envs, dtype=np.float64)
        disturbance_force_innovation = np.zeros(self.num_envs, dtype=np.float64)
        disturbance_torque_innovation = np.zeros(self.num_envs, dtype=np.float64)
        active_nmpc_mode = np.empty(self.num_envs, dtype=object)
        x0_all = np.zeros((self.num_envs, self.nx), dtype=np.float64)
        params_all = np.zeros((self.num_envs, self.N + 1, self.n_p), dtype=np.float64)
        hover_forces = np.zeros(self.num_envs, dtype=np.float64)
        needs_reinit_all = np.zeros(self.num_envs, dtype=bool)

        for env_i in range(self.num_envs):
            m = mass_np[env_i, 0]
            I = inertia_np[env_i]
            g = gravity_np[env_i]
            hover_force = m * np.linalg.norm(g) / NUM_MOTORS
            current_mode = nmpc_mode_np[env_i]
            if current_mode not in ("NMPC_FLIGHT_MODE", "NMPC_RECOVERY_MODE", "NMPC_COAST_MODE"):
                raise ValueError(f"unsupported NMPC mode for env {env_i}: {current_mode}")
            needs_reinit = np.any(np.isnan(self._prev_setpoint_pos[env_i])) or self._prev_nmpc_mode[env_i] != current_mode
            hover_forces[env_i] = hover_force
            needs_reinit_all[env_i] = needs_reinit

            if needs_reinit or not self._motor_estimator_initialized[env_i]:
                self._initialize_motor_estimator(env_i)
            else:
                self._propagate_motor_estimator(env_i)

            x0 = x0_all[env_i]
            params = params_all[env_i]

            x0[0:3] = pos_np[env_i]
            x0[3:6] = vel_np[env_i]
            x0[6:10] = quat_np[env_i]
            x0[10:13] = omega_body_np[env_i]
            x0[13:13 + NUM_MOTORS] = self._motor_rps_hat[env_i]

            params[:, PARAM_MASS] = m
            params[:, PARAM_INERTIA_START] = I[0, 0]
            params[:, PARAM_INERTIA_START + 1] = I[0, 1]
            params[:, PARAM_INERTIA_START + 2] = I[0, 2]
            params[:, PARAM_INERTIA_START + 3] = I[1, 1]
            params[:, PARAM_INERTIA_START + 4] = I[1, 2]
            params[:, PARAM_INERTIA_START + 5] = I[2, 2]
            params[:, PARAM_GRAVITY_START:PARAM_GRAVITY_START + 3] = g

            yaw = sp_yaw_np[env_i]
            params[:, PARAM_POS_REF_START:PARAM_POS_REF_START + 3] = sp_pos_np[env_i]
            params[:, PARAM_VEL_REF_START:PARAM_VEL_REF_START + 3] = sp_vel_np[env_i]
            params[:, PARAM_ACC_REF_START:PARAM_ACC_REF_START + 3] = 0.0
            quat_ref = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
            params[:, PARAM_QUAT_REF_START:PARAM_QUAT_REF_START + 4] = quat_ref

            B = self._alloc_matrix[env_i]
            params[:, PARAM_ALLOC_START:PARAM_ALLOC_START + PARAM_ALLOC_SIZE] = B.flatten(order='F')
            params[:, PARAM_KF_START:PARAM_KF_START + NUM_MOTORS] = self.thrust_coefficients
            params[:, PARAM_TAU_START:PARAM_TAU_START + NUM_MOTORS] = self.motor_time_constants
            params[:, PARAM_COM_START:PARAM_COM_START + 3] = self._com_offset[env_i]

            self._disturbance_estimator.update(env_i, x0, params[0])
            params[:, PARAM_DIST_FORCE_START:PARAM_DIST_FORCE_START + 3] = self._disturbance_estimator.applied_force[env_i]
            params[:, PARAM_DIST_TORQUE_START:PARAM_DIST_TORQUE_START + 3] = self._disturbance_estimator.applied_torque[env_i]
            params[0, PARAM_U_PREV_START:PARAM_U_PREV_START + NUM_MOTORS] = self._commanded_force_prev[env_i]
            params[1:, PARAM_U_PREV_START:PARAM_U_PREV_START + NUM_MOTORS] = self._previous_horizon_controls[env_i]

            self._prev_setpoint_pos[env_i] = sp_pos_np[env_i]
            self._prev_setpoint_vel[env_i] = sp_vel_np[env_i]
            self._prev_nmpc_mode[env_i] = current_mode
            active_nmpc_mode[env_i] = current_mode

            estimated_motor_rps[env_i] = self._motor_rps_hat[env_i]
            previous_commanded_motor_rps[env_i] = self._commanded_rps_prev[env_i]
            estimated_disturbance_force[env_i] = self._disturbance_estimator.estimated_force[env_i]
            estimated_disturbance_torque[env_i] = self._disturbance_estimator.estimated_torque[env_i]
            applied_disturbance_force[env_i] = self._disturbance_estimator.applied_force[env_i]
            applied_disturbance_torque[env_i] = self._disturbance_estimator.applied_torque[env_i]
            disturbance_force_variance[env_i] = self._disturbance_estimator.force_variance[env_i]
            disturbance_torque_variance[env_i] = self._disturbance_estimator.torque_variance[env_i]
            disturbance_force_measurement[env_i] = self._disturbance_estimator.force_measurement[env_i]
            disturbance_torque_measurement[env_i] = self._disturbance_estimator.torque_measurement[env_i]
            disturbance_force_bound[env_i] = self._disturbance_estimator.force_bound[env_i]
            disturbance_torque_bound[env_i] = self._disturbance_estimator.torque_bound[env_i]
            disturbance_flags[env_i] = self._disturbance_estimator.flags[env_i]
            disturbance_force_confidence[env_i] = self._disturbance_estimator.force_confidence[env_i]
            disturbance_torque_confidence[env_i] = self._disturbance_estimator.torque_confidence[env_i]
            disturbance_force_innovation[env_i] = self._disturbance_estimator.force_innovation_norm[env_i]
            disturbance_torque_innovation[env_i] = self._disturbance_estimator.torque_innovation_norm[env_i]

        recovery_mask = active_nmpc_mode == "NMPC_RECOVERY_MODE"
        coast_mask = active_nmpc_mode == "NMPC_COAST_MODE"
        if self.num_envs == 1:
            solve_results = {
                "solver_statuses": np.zeros(self.num_envs, dtype=np.int32),
                "solver_costs": np.zeros(self.num_envs, dtype=np.float64),
                "u_opt": np.zeros((self.num_envs, NUM_MOTORS), dtype=np.float64),
                "x1": np.zeros((self.num_envs, 13 + NUM_MOTORS), dtype=np.float64),
                "x_end": np.zeros((self.num_envs, 13 + NUM_MOTORS), dtype=np.float64),
                "horizon_controls": np.zeros((self.num_envs, self.N, NUM_MOTORS), dtype=np.float64),
            }
            if not coast_mask[0]:
                solver = self.recovery_solver if recovery_mask[0] else self.solver
                single_result = _solve_acados_problem(
                    solver,
                    self.N,
                    x0_all[0],
                    params_all[0],
                    hover_forces[0],
                    needs_reinit_all[0],
                )
                solve_results["solver_statuses"][0] = single_result["status"]
                solve_results["solver_costs"][0] = single_result["cost"]
                solve_results["u_opt"][0] = single_result["u_opt"]
                solve_results["x1"][0] = single_result["x1"]
                solve_results["x_end"][0] = single_result["x_end"]
                solve_results["horizon_controls"][0] = single_result["horizon_controls"]
        else:
            nominal_results = self._parallel_solvers.solve_all(x0_all, params_all, hover_forces, needs_reinit_all)
            recovery_results = self._parallel_recovery_solvers.solve_all(x0_all, params_all, hover_forces, needs_reinit_all)
            solve_results = _select_mode_results(nominal_results, recovery_results, active_nmpc_mode)

        if np.any(coast_mask):
            # Ballistic pass-through matching the pipe solver: no solve, all
            # motors held at motor_rps_min. Horizon controls are set to hover so
            # the first post-coast solve sees the same u_prev references as the
            # reinitialized real solver.
            solve_results["solver_statuses"][coast_mask] = 0
            solve_results["solver_costs"][coast_mask] = 0.0
            solve_results["u_opt"][coast_mask] = self._coast_motor_forces
            solve_results["x1"][coast_mask] = x0_all[coast_mask]
            solve_results["x_end"][coast_mask] = x0_all[coast_mask]
            solve_results["horizon_controls"][coast_mask] = hover_forces[coast_mask, None, None]

        commanded_motor_rps[:] = self._convert_motor_forces_to_rps(solve_results["u_opt"])
        self._commanded_rps_prev[:] = commanded_motor_rps
        self._commanded_force_prev[:] = solve_results["u_opt"]
        self._previous_horizon_controls[:] = solve_results["horizon_controls"]
        self.motor_forces[:] = torch.tensor(solve_results["u_opt"], device=self.device, dtype=torch.float32)
        self.desired_quat[:] = torch.tensor(solve_results["x1"][:, 6:10], device=self.device, dtype=torch.float32)

        solver_statuses[:] = solve_results["solver_statuses"]
        solver_costs[:] = solve_results["solver_costs"]
        x_end_all = solve_results["x_end"]
        predicted_pos_end[:] = x_end_all[:, 0:3]
        predicted_vel_end[:] = x_end_all[:, 3:6]
        predicted_quat_end[:] = x_end_all[:, 6:10]
        horizon_controls[:] = solve_results["horizon_controls"]

        self.debug_data = {
            'solver_status': solver_statuses,
            'solver_cost': solver_costs,
            'predicted_pos_end': predicted_pos_end,
            'predicted_vel_end': predicted_vel_end,
            'predicted_quat_end': predicted_quat_end,
            'horizon_controls': horizon_controls,
            'estimated_motor_rps': estimated_motor_rps,
            'previous_commanded_motor_rps': previous_commanded_motor_rps,
            'commanded_motor_rps': commanded_motor_rps,
            'estimated_disturbance_force_body': estimated_disturbance_force,
            'estimated_disturbance_torque_body': estimated_disturbance_torque,
            'applied_disturbance_force_body': applied_disturbance_force,
            'applied_disturbance_torque_body': applied_disturbance_torque,
            'disturbance_force_variance': disturbance_force_variance,
            'disturbance_torque_variance': disturbance_torque_variance,
            'disturbance_force_measurement_body': disturbance_force_measurement,
            'disturbance_torque_measurement_body': disturbance_torque_measurement,
            'disturbance_force_bound_body': disturbance_force_bound,
            'disturbance_torque_bound_body': disturbance_torque_bound,
            'disturbance_estimator_flags': disturbance_flags,
            'disturbance_force_confidence': disturbance_force_confidence,
            'disturbance_torque_confidence': disturbance_torque_confidence,
            'disturbance_force_innovation_nis': disturbance_force_innovation,
            'disturbance_torque_innovation_nis': disturbance_torque_innovation,
            'active_nmpc_mode': active_nmpc_mode,
        }

        self.desired_quat[:] = normalize_quat(self.desired_quat)
        return self.motor_forces
