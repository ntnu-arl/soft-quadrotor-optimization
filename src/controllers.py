import torch
import json
from pathlib import Path
from utils import (
    quat_mul, quat_inverse, quat_rotate, quat_to_rotation_matrix,
    quat_from_rotation_matrix, euler_xyz_from_quat, vee_map, normalize_quat
)


class BaseController:
    """Base controller class that stores robot state and implements common functionality."""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.robot_position = None
        self.robot_orientation = None
        self.robot_linvel = None
        self.robot_angvel = None
        self.robot_body_linvel = None
        self.robot_body_angvel = None
        self.robot_euler_angles = None
        self.mass = None
        self.inertia = None
        self.gravity = None

    def set_state(self, position, orientation, linvel, angvel, mass, inertia, gravity):
        """Set current robot state. All quaternions in [w,x,y,z] format."""
        self.robot_position = position
        self.robot_orientation = normalize_quat(orientation)
        self.robot_linvel = linvel
        self.robot_angvel = angvel

        self.robot_body_linvel = quat_rotate(quat_inverse(self.robot_orientation), self.robot_linvel)
        self.robot_body_angvel = quat_rotate(quat_inverse(self.robot_orientation), self.robot_angvel)
        self.robot_euler_angles = euler_xyz_from_quat(self.robot_orientation)

        self.mass = mass
        self.inertia = inertia
        self.gravity = gravity

    def update(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement update()")

