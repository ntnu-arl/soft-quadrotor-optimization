"""Airframe class for unified representation and manipulation of drone airframe parameters.

This module provides a single Airframe class that encapsulates all airframe-related
operations, including:
- Parameter storage in real units (meters, degrees)
- Conversion between different representations (normalized [0,1], actual values, dict)
- Generation of airframe properties JSON
- Computation of allocation matrices, torque parameters, max arm angles
- Validation of hover capability and arm deflection constraints
- Repair functions to fix invalid designs

Internal representation uses actual parameter values. All conversions are handled
through properties and factory methods.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from typing import Optional
from functools import cached_property
from collections.abc import Mapping

PARAM_BOUNDS = {
    'front_arm_length': (0.05, 0.15),
    'front_prop_rotation_x': (-45, 45),
    'front_prop_rotation_y': (-45, 45),
    'front_prop_rotation_z': (-45, 45),
    'back_arm_length': (0.05, 0.15),
    'back_prop_rotation_x': (-45, 45),
    'back_prop_rotation_y': (-45, 45),
    'back_prop_rotation_z': (-45, 45),
    'side_arm_seg2_length_left': (0.05, 0.15),
    'side_arm_seg2_length_right': (0.05, 0.15),
    'side_base_rotation_x_left': (-30, 30),
    'side_base_rotation_y_left': (-45, 45),
    'side_base_rotation_z_left': (-5, 5),
    'side_base_rotation_x_right': (-30, 30),
    'side_base_rotation_y_right': (-45, 45),
    'side_base_rotation_z_right': (-5, 5),
    'side_motor_rotation_x_left': (-45, 45),
    'side_motor_rotation_y_left': (-45, 45),
    'side_motor_rotation_z_left': (-45, 45),
    'side_motor_rotation_x_right': (-45, 45),
    'side_motor_rotation_y_right': (-45, 45),
    'side_motor_rotation_z_right': (-45, 45),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

TORQUE_RESPONSE_MAX_ANGLE_DEG = 180.0
TORQUE_RESPONSE_RAMP_ANGLE_DEG = 2.0
TORQUE_RESPONSE_AMPLITUDE = 1.0
TORQUE_RESPONSE_BETWEEN_TORQUE = 0.0

ADD_ROD = True


def build_piecewise_torque_response_function(
    max_angle_a_deg: float,
    torque_between_a_b: float,
    start_angle_b_deg: float,
    end_angle_b_deg: float,
    side: str,
) -> str:
    if side not in {"left", "right"}:
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}")
    if max_angle_a_deg <= 0.0:
        raise ValueError(f"max_angle_a_deg must be positive, got {max_angle_a_deg}")
    if start_angle_b_deg < max_angle_a_deg:
        raise ValueError(
            f"start_angle_b_deg must be >= max_angle_a_deg, got {start_angle_b_deg} < {max_angle_a_deg}"
        )
    if end_angle_b_deg <= start_angle_b_deg:
        raise ValueError(
            f"end_angle_b_deg must be > start_angle_b_deg, got {end_angle_b_deg} <= {start_angle_b_deg}"
        )

    max_angle_a_rad = np.deg2rad(max_angle_a_deg)
    start_angle_b_rad = np.deg2rad(start_angle_b_deg)
    end_angle_b_rad = np.deg2rad(end_angle_b_deg)
    amplitude = TORQUE_RESPONSE_AMPLITUDE
    torque_between = torque_between_a_b
    assignments = ";".join(
        [
            f"max_angle_a_{side}={max_angle_a_rad:.17g}",
            f"torque_between_a_b_{side}={torque_between:.17g}",
            f"start_angle_b_{side}={start_angle_b_rad:.17g}",
            f"end_angle_b_{side}={end_angle_b_rad:.17g}",
            f"amplitude_{side}={amplitude:.17g}",
        ]
    )

    if side == "left":
        return (
            f"{assignments};"
            "torque_fn_left=lambda x: torch.where(x <= 0.0, "
            "torch.where(torch.abs(x) <= max_angle_a_left, "
            "(x * 0.0) + amplitude_left * (torch.abs(x) / max_angle_a_left), "
            "(x * 0.0) + amplitude_left), "
            "torch.where(torch.abs(x) <= max_angle_a_left, "
            "(x * 0.0) - amplitude_left * (torch.abs(x) / max_angle_a_left), "
            "torch.where(torch.abs(x) <= start_angle_b_left, "
            "(x * 0.0) - torque_between_a_b_left, "
            "torch.where(torch.abs(x) < end_angle_b_left, "
            "(x * 0.0) - (torque_between_a_b_left + ((torch.abs(x) - start_angle_b_left) / "
            "(end_angle_b_left - start_angle_b_left)) * (amplitude_left - torque_between_a_b_left)), "
            "(x * 0.0) - amplitude_left))))"
        )

    return (
        f"{assignments};"
        "torque_fn_right=lambda x: torch.where(x >= 0.0, "
        "torch.where(torch.abs(x) <= max_angle_a_right, "
        "(x * 0.0) - amplitude_right * (torch.abs(x) / max_angle_a_right), "
        "(x * 0.0) - amplitude_right), "
        "torch.where(torch.abs(x) <= max_angle_a_right, "
        "(x * 0.0) + amplitude_right * (torch.abs(x) / max_angle_a_right), "
        "torch.where(torch.abs(x) <= start_angle_b_right, "
        "(x * 0.0) + torque_between_a_b_right, "
        "torch.where(torch.abs(x) < end_angle_b_right, "
        "(x * 0.0) + (torque_between_a_b_right + ((torch.abs(x) - start_angle_b_right) / "
        "(end_angle_b_right - start_angle_b_right)) * (amplitude_right - torque_between_a_b_right)), "
        "(x * 0.0) + amplitude_right))))"
    )


def build_torque_response_functions(end_angle_b_left: float, end_angle_b_right: float) -> tuple[str, str]:
    start_angle_b_left = end_angle_b_left - TORQUE_RESPONSE_RAMP_ANGLE_DEG
    start_angle_b_right = end_angle_b_right - TORQUE_RESPONSE_RAMP_ANGLE_DEG
    return (
        build_piecewise_torque_response_function(
            TORQUE_RESPONSE_RAMP_ANGLE_DEG,
            TORQUE_RESPONSE_BETWEEN_TORQUE,
            start_angle_b_left,
            end_angle_b_left,
            "left",
        ),
        build_piecewise_torque_response_function(
            TORQUE_RESPONSE_RAMP_ANGLE_DEG,
            TORQUE_RESPONSE_BETWEEN_TORQUE,
            start_angle_b_right,
            end_angle_b_right,
            "right",
        ),
    )

class AirframeConstants(Mapping):
    arm_width = 0.01
    arm_height = 0.01
    arm_density = 200.0
    motor_mass = 0.01625
    motor_radius = 0.045
    motor_height = 0.035
    base_mass = 0.225
    base_box_dimensions = (0.07, 0.05, 0.08)
    side_seg1_len = 0.02
    rod_radius = 0.003

    motor_idx_list = [5, 5, 5, 5]
    motor_directions = [-1, 1, -1, 1]

    # ---- Lazy load of rotor dynamics. necessary because torch cannot be imported before isaacsim. ----
    @cached_property
    def _motor_data(self):
        from rotor_dynamics import manufacturerComponentData
        return manufacturerComponentData("cpu", self.motor_idx_list)

    @cached_property
    def thrust_coefficient(self):
        return self._motor_data.thrust_coefficient[0].item()

    @cached_property
    def cq(self):
        return self._motor_data.cq[0].item()

    @cached_property
    def max_rps(self):
        return self._motor_data.max_rps[0].item()

    @cached_property
    def min_rps(self):
        return self._motor_data.min_rps[0].item()

    _keys = {
        'arm_width',
        'arm_height',
        'arm_density',
        'motor_mass',
        'motor_radius',
        'motor_height',
        'base_mass',
        'base_box_dimensions',
        'side_seg1_len',
        'rod_radius',
        'motor_idx_list',
        'motor_directions',
        'thrust_coefficient',
        'cq',
        'max_rps',
        'min_rps',
    }

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

AIRFRAME_CONSTANTS = AirframeConstants()



class Airframe:
    """Unified airframe representation with conversion and computation methods.

    The internal representation stores parameters in their actual units (meters for lengths,
    degrees for angles). Factory methods and properties provide access to different
    representations as needed.

    Attributes:
        _params: Dictionary mapping parameter names to actual values (meters, degrees)
    """

    def __init__(self, params: dict[str, float], skipp_validation: bool = False) -> None:
        """Initialize Airframe with actual parameter values.

        Args:
            params: Dictionary mapping parameter names to actual values

        Raises:
            ValueError: If required parameters are missing or out of bounds
        """
        if not skipp_validation:
            self._validate_params(params)
        self._params = params.copy()
        self._cached_json_data = None

    def _validate_params(self, params: dict[str, float]) -> None:
        """Validate that all required parameters are present and within bounds."""
        missing = [name for name in PARAM_NAMES if name not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        for name, value in params.items():
            if name not in PARAM_BOUNDS:
                continue
            min_val, max_val = PARAM_BOUNDS[name]
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter '{name}' = {value} out of bounds [{min_val}, {max_val}]")

    @classmethod
    def from_x_0_1(cls, x_0_1: list | np.ndarray, skip_validation: bool = False) -> Airframe:
        """Create Airframe from normalized [0,1] parameter vector.

        Args:
            x_0_1: List or array of 31 normalized parameters in [0,1] range
            skip_validation: If True, skip parameter bounds validation

        Returns:
            Airframe instance
        """
        if len(x_0_1) != len(PARAM_NAMES):
            raise ValueError(f"Expected {len(PARAM_NAMES)} parameters, got {len(x_0_1)}")
        actual_params = cls._map_params_to_ranges(x_0_1)
        params = dict(zip(PARAM_NAMES, actual_params))
        return cls(params, skipp_validation=skip_validation)

    @classmethod
    def from_params_dict(cls, params: dict[str, float]) -> Airframe:
        """Create Airframe from parameter dictionary with actual values.

        Args:
            params: Dictionary mapping parameter names to actual values

        Returns:
            Airframe instance
        """
        return cls(params)

    @classmethod
    def from_json(cls, json_path: str | Path) -> Airframe:
        """Create Airframe from JSON file.

        Args:
            json_path: Path to airframe_data.json file

        Returns:
            Airframe instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'parameters' in data:
            params = data['parameters']
        elif 'x_0_1' in data:
            return cls.from_x_0_1(data['x_0_1'])
        else:
            raise ValueError("JSON must contain 'parameters' or 'x_0_1' field")

        return cls(params)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> Airframe:
        """Generate random airframe with parameters uniformly sampled from bounds.

        Args:
            rng: Optional numpy random generator for reproducibility

        Returns:
            Airframe instance
        """
        if rng is None:
            rng = np.random.default_rng()
        x_0_1 = rng.uniform(0, 1, len(PARAM_NAMES))
        return cls.from_x_0_1(x_0_1)

    @staticmethod
    def _map_params_to_ranges(x_0_1: list | np.ndarray) -> list[float]:
        """Convert normalized [0,1] parameters to actual values."""
        return [
            min_val + val * (max_val - min_val)
            for val, (min_val, max_val) in zip(x_0_1, PARAM_BOUNDS.values())
        ]

    @staticmethod
    def _normalize_params(actual_values: list | np.ndarray) -> list[float]:
        """Convert actual parameter values to normalized [0,1] range."""
        return [
            (val - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            for val, (min_val, max_val) in zip(actual_values, PARAM_BOUNDS.values())
        ]

    @property
    def params(self) -> dict[str, float]:
        """Get parameter dictionary with actual values (meters, degrees)."""
        return self._params.copy()

    @property
    def x_0_1(self) -> np.ndarray:
        """Get normalized [0,1] parameter vector."""
        actual_values = [self._params[name] for name in PARAM_NAMES]
        return np.array(self._normalize_params(actual_values))

    def with_updated_params(self, updates: dict[str, float]) -> Airframe:
        """Return new Airframe with updated parameters.

        Args:
            updates: Dict of parameter names and new values

        Returns:
            New Airframe instance with updated parameters
        """
        new_params = self._params.copy()
        new_params.update(updates)
        return Airframe(new_params)

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._cached_json_data = None

    def compute_total_mass(self) -> float:
        """Compute total mass of the airframe."""
        arm_width = AIRFRAME_CONSTANTS['arm_width']
        arm_height = AIRFRAME_CONSTANTS['arm_height']
        arm_density = AIRFRAME_CONSTANTS['arm_density']
        motor_mass = AIRFRAME_CONSTANTS['motor_mass']
        base_mass = AIRFRAME_CONSTANTS['base_mass']
        side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']

        front_arm_len = self._params['front_arm_length']
        back_arm_len = self._params['back_arm_length']
        side_seg2_len_left = self._params['side_arm_seg2_length_left']
        side_seg2_len_right = self._params['side_arm_seg2_length_right']

        front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
        back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
        side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
        side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
        side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density

        return base_mass + 4 * motor_mass + front_arm_mass + back_arm_mass + 2 * side_seg1_mass + side_seg2_mass_left + side_seg2_mass_right

    def compute_motor_positions_and_thrusts(self) -> dict:
        """Compute motor positions and thrust directions in body frame.

        Returns:
            Dictionary with:
                - motor_positions: (4, 3) array of motor positions
                - motor_thrust_directions: (4, 3) array of thrust directions
                - joint_1_position, joint_3_position: joint positions
                - joint_1_axis, joint_3_axis: joint rotation axes
        """
        import math

        def rpy_to_rotation_matrix(roll, pitch, yaw):
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            return R_z @ R_y @ R_x

        params = self._params
        side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
        side_seg2_len_left = params['side_arm_seg2_length_left']
        side_seg2_len_right = params['side_arm_seg2_length_right']

        motor_positions = []
        motor_thrust_directions = []

        front_prop_rot_x = math.radians(params['front_prop_rotation_x'])
        front_prop_rot_y = math.radians(params['front_prop_rotation_y'])
        front_prop_rot_z = math.radians(params['front_prop_rotation_z'])
        motor_0_pos = np.array([0.035 + params['front_arm_length'], 0.0, 0.0])
        motor_0_thrust = rpy_to_rotation_matrix(front_prop_rot_x, front_prop_rot_y, front_prop_rot_z) @ np.array([0, 0, 1])
        motor_positions.append(motor_0_pos.tolist())
        motor_thrust_directions.append(motor_0_thrust.tolist())

        side_base_rot_x_left = math.radians(params['side_base_rotation_x_left'])
        side_base_rot_y_left = math.radians(params['side_base_rotation_y_left'])
        side_base_rot_z_left = math.radians(params['side_base_rotation_z_left'])
        side_motor_rot_x_left = math.radians(params['side_motor_rotation_x_left'])
        side_motor_rot_y_left = math.radians(params['side_motor_rotation_y_left'])
        side_motor_rot_z_left = math.radians(params['side_motor_rotation_z_left'])
        R_base_1 = rpy_to_rotation_matrix(side_base_rot_x_left, side_base_rot_y_left, side_base_rot_z_left + np.pi/2)
        base_attachment_1 = np.array([0, 0.025, 0])
        joint_1_pos = base_attachment_1 + R_base_1 @ np.array([side_seg1_len, 0, 0])
        joint_1_axis = (R_base_1 @ np.array([0, 0, 1])).tolist()
        motor_1_pos = base_attachment_1 + R_base_1 @ np.array([side_seg1_len + side_seg2_len_left, 0, 0])
        R_motor_1 = R_base_1 @ rpy_to_rotation_matrix(side_motor_rot_x_left, side_motor_rot_y_left, side_motor_rot_z_left)
        motor_1_thrust = R_motor_1 @ np.array([0, 0, 1])
        motor_positions.append(motor_1_pos.tolist())
        motor_thrust_directions.append(motor_1_thrust.tolist())

        back_prop_rot_x = math.radians(params['back_prop_rotation_x'])
        back_prop_rot_y = math.radians(params['back_prop_rotation_y'])
        back_prop_rot_z = math.radians(params['back_prop_rotation_z'])
        R_back = rpy_to_rotation_matrix(0, 0, np.pi)
        pos_back = np.array([-0.035, 0, 0]) + R_back @ np.array([params['back_arm_length'], 0, 0])
        R_motor_2 = R_back @ rpy_to_rotation_matrix(back_prop_rot_x, back_prop_rot_y, back_prop_rot_z)
        motor_2_thrust = R_motor_2 @ np.array([0, 0, 1])
        motor_positions.append(pos_back.tolist())
        motor_thrust_directions.append(motor_2_thrust.tolist())

        side_base_rot_x_right = math.radians(params['side_base_rotation_x_right'])
        side_base_rot_y_right = math.radians(params['side_base_rotation_y_right'])
        side_base_rot_z_right = math.radians(params['side_base_rotation_z_right'])
        side_motor_rot_x_right = math.radians(params['side_motor_rotation_x_right'])
        side_motor_rot_y_right = math.radians(params['side_motor_rotation_y_right'])
        side_motor_rot_z_right = math.radians(params['side_motor_rotation_z_right'])
        R_base_3 = rpy_to_rotation_matrix(side_base_rot_x_right, side_base_rot_y_right, side_base_rot_z_right - np.pi/2)
        base_attachment_3 = np.array([0, -0.025, 0])
        joint_3_pos = base_attachment_3 + R_base_3 @ np.array([side_seg1_len, 0, 0])
        joint_3_axis = (R_base_3 @ np.array([0, 0, 1])).tolist()
        motor_3_pos = base_attachment_3 + R_base_3 @ np.array([side_seg1_len + side_seg2_len_right, 0, 0])
        R_motor_3 = R_base_3 @ rpy_to_rotation_matrix(side_motor_rot_x_right, side_motor_rot_y_right, side_motor_rot_z_right)
        motor_3_thrust = R_motor_3 @ np.array([0, 0, 1])
        motor_positions.append(motor_3_pos.tolist())
        motor_thrust_directions.append(motor_3_thrust.tolist())

        return {
            'motor_positions': motor_positions,
            'motor_thrust_directions': motor_thrust_directions,
            'joint_1_position': joint_1_pos.tolist(),
            'joint_3_position': joint_3_pos.tolist(),
            'joint_1_axis': joint_1_axis,
            'joint_3_axis': joint_3_axis,
        }

    def to_json_dict(self) -> dict:
        """Generate airframe properties dictionary.

        Returns:
            Dictionary containing all airframe properties
        """
        if self._cached_json_data is not None:
            return self._cached_json_data.copy()

        motor_data = self.compute_motor_positions_and_thrusts()
        params = self._params

        airframe_data = {
            "total_mass": self.compute_total_mass(),
            "num_motors": 4,
            "motor_positions_body_frame": motor_data['motor_positions'],
            "motor_thrust_directions": motor_data['motor_thrust_directions'],
            "motor_directions": AIRFRAME_CONSTANTS['motor_directions'],
            "motor_idx_list": AIRFRAME_CONSTANTS['motor_idx_list'],
            "x_0_1": self.x_0_1.tolist(),
            "parameters": params,
            "joint_1_position": motor_data['joint_1_position'],
            "joint_3_position": motor_data['joint_3_position'],
            "joint_1_axis": motor_data['joint_1_axis'],
            "joint_3_axis": motor_data['joint_3_axis'],
            "side_seg1_length": AIRFRAME_CONSTANTS['side_seg1_len'],
            "side_seg2_length_left": params['side_arm_seg2_length_left'],
            "side_seg2_length_right": params['side_arm_seg2_length_right']
        }

        self._cached_json_data = airframe_data
        return airframe_data.copy()

    def save_json(self, path: str | Path) -> None:
        """Save airframe properties to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_json_dict(), f, indent=2)

    def compute_allocation_matrix(self) -> dict:
        """Compute the 6xN allocation matrix and motor data.

        The allocation matrix B maps motor forces to body wrench: wrench = B @ forces

        Returns:
            Dictionary with:
                - B: (6, n_motors) allocation matrix
                - mass: total mass
                - motor_positions: (n_motors, 3) array
                - motor_thrust_directions: (n_motors, 3) array
                - motor_directions: (n_motors,) array
                - min_forces: (n_motors,) array
                - max_forces: (n_motors,) array
                - thrust_to_torque_ratio: (n_motors,) array
        """
        from rotor_dynamics import RotorDynamics, manufacturerComponentData

        json_data = self.to_json_dict()

        motor_positions = np.array(json_data["motor_positions_body_frame"])
        motor_thrust_directions = np.array(json_data["motor_thrust_directions"])
        motor_directions = np.array(json_data["motor_directions"])
        motor_idx_list = json_data["motor_idx_list"]

        max_forces, torque_coefficients, _ = RotorDynamics.get_motor_max_force_torque_and_rps(motor_idx_list)
        thrust_to_torque_ratio = torque_coefficients / max_forces
        component_data = manufacturerComponentData("cpu", motor_idx_list)
        min_forces = component_data.min_force.numpy()

        n_motors = len(motor_positions)
        B = np.zeros((6, n_motors))

        for i in range(n_motors):
            r = motor_positions[i].reshape(3, 1)
            direction = motor_directions[i]
            thrust_vec = motor_thrust_directions[i].reshape(3, 1)
            force_contribution = thrust_vec
            torque_contribution = np.cross(r.flatten(), thrust_vec.flatten()).reshape(3, 1)
            yaw_torque = direction * thrust_to_torque_ratio[i] * thrust_vec
            B[0:3, i] = force_contribution.flatten()
            B[3:6, i] = (torque_contribution + yaw_torque).flatten()

        return {
            'B': B,
            'mass': json_data["total_mass"],
            'motor_positions': motor_positions,
            'motor_thrust_directions': motor_thrust_directions,
            'motor_directions': motor_directions,
            'min_forces': min_forces,
            'max_forces': max_forces,
            'thrust_to_torque_ratio': thrust_to_torque_ratio
        }

    def compute_end_angle_b(self) -> dict:
        """Compute end_angle_b parameters for arm joints based on collision detection.

        Returns:
            Dictionary with end_angle_b_left/right and start_angle_b_left/right
        """
        max_left_deg, max_right_deg, _ = self.get_max_arm_angles(delta_deg=0.1, verbose=False)
        assert abs(max_left_deg) > 0.001 and abs(max_right_deg) > 0.001, "Arm self-collision at very small angles! ( collision at < 0.001 deg ). Design was x_0_1 = " + str(self.x_0_1.tolist())
        end_angle_b_left = min(max_left_deg, TORQUE_RESPONSE_MAX_ANGLE_DEG)
        end_angle_b_right = min(abs(max_right_deg), TORQUE_RESPONSE_MAX_ANGLE_DEG)

        return {
            'end_angle_b_left': end_angle_b_left,
            'end_angle_b_right': end_angle_b_right,
            'start_angle_b_left': end_angle_b_left - TORQUE_RESPONSE_RAMP_ANGLE_DEG,
            'start_angle_b_right': end_angle_b_right - TORQUE_RESPONSE_RAMP_ANGLE_DEG,
        }

    def get_max_arm_angles(self, delta_deg: float, verbose: bool) -> tuple[float, float, dict]:
        """Find maximum arm bend angles before self-collision.

        Args:
            delta_deg: Angle increment in degrees for collision search
            verbose: If True, display 3D visualization

        Returns:
            Tuple of (max_left_angle_deg, max_right_angle_deg, aabb_info)
        """
        import fcl
        import math

        params = self._params
        arm_width = AIRFRAME_CONSTANTS['arm_width']
        arm_height = AIRFRAME_CONSTANTS['arm_height']
        motor_radius = AIRFRAME_CONSTANTS['motor_radius']
        motor_height = AIRFRAME_CONSTANTS['motor_height']
        base_box_dims = AIRFRAME_CONSTANTS['base_box_dimensions']
        side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
        rod_radius = AIRFRAME_CONSTANTS['rod_radius']

        front_arm_len = params['front_arm_length']
        back_arm_len = params['back_arm_length']
        side_seg2_len_left = params['side_arm_seg2_length_left']
        side_seg2_len_right = params['side_arm_seg2_length_right']

        front_prop_rot = np.array([math.radians(params['front_prop_rotation_x']),
                                   math.radians(params['front_prop_rotation_y']),
                                   math.radians(params['front_prop_rotation_z'])])
        back_prop_rot = np.array([math.radians(params['back_prop_rotation_x']),
                                  math.radians(params['back_prop_rotation_y']),
                                  math.radians(params['back_prop_rotation_z'])])
        side_base_rot_left = np.array([math.radians(params['side_base_rotation_x_left']),
                                       math.radians(params['side_base_rotation_y_left']),
                                       math.radians(params['side_base_rotation_z_left']) + math.pi/2])
        side_base_rot_right = np.array([math.radians(params['side_base_rotation_x_right']),
                                        math.radians(params['side_base_rotation_y_right']),
                                        math.radians(params['side_base_rotation_z_right']) - math.pi/2])
        side_motor_rot_left = np.array([math.radians(params['side_motor_rotation_x_left']),
                                        math.radians(params['side_motor_rotation_y_left']),
                                        math.radians(params['side_motor_rotation_z_left'])])
        side_motor_rot_right = np.array([math.radians(params['side_motor_rotation_x_right']),
                                         math.radians(params['side_motor_rotation_y_right']),
                                         math.radians(params['side_motor_rotation_z_right'])])

        def rpy_to_rotation_matrix(roll, pitch, yaw):
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            return Rz @ Ry @ Rx

        def compute_rod_geometry(arm_base_rot, motor_rot, seg2_len):
            """Compute rod parameters connecting motor edge (highest world-x) to the revolute joint."""
            R_arm = rpy_to_rotation_matrix(*arm_base_rot)
            R_motor = rpy_to_rotation_matrix(*motor_rot)
            R_combined = R_arm @ R_motor
            world_x_in_motor = R_combined.T @ np.array([1, 0, 0])
            xy_norm = np.linalg.norm(world_x_in_motor[:2])
            edge_dir = np.array([world_x_in_motor[0] / xy_norm, world_x_in_motor[1] / xy_norm, 0]) if xy_norm > 1e-6 else np.array([1, 0, 0])
            edge_point_motor = motor_radius * edge_dir
            joint_in_motor = R_motor.T @ np.array([-seg2_len, 0, 0])
            rod_vector = edge_point_motor - joint_in_motor
            rod_length = float(np.linalg.norm(rod_vector))
            rod_center_motor = (edge_point_motor + joint_in_motor) / 2
            rod_direction = rod_vector / rod_length
            rod_xy_len = math.sqrt(rod_direction[0]**2 + rod_direction[1]**2)
            rod_pitch = math.pi/2 - math.atan2(rod_direction[2], rod_xy_len)
            rod_yaw = math.atan2(rod_direction[1], rod_direction[0])
            return rod_length, rod_center_motor, rod_pitch, rod_yaw

        rod_left_len, rod_left_center_motor, rod_left_pitch, rod_left_yaw = compute_rod_geometry(
            side_base_rot_left, side_motor_rot_left, side_seg2_len_left)
        rod_right_len, rod_right_center_motor, rod_right_pitch, rod_right_yaw = compute_rod_geometry(
            side_base_rot_right, side_motor_rot_right, side_seg2_len_right)

        def create_box_collision_object(dims, position, rotation_matrix):
            shape = fcl.Box(dims[0], dims[1], dims[2])
            transform = fcl.Transform(rotation_matrix, position)
            return fcl.CollisionObject(shape, transform), ('box', dims, position, rotation_matrix)

        def create_cylinder_collision_object(radius, height, position, rotation_matrix):
            shape = fcl.Cylinder(radius, height)
            transform = fcl.Transform(rotation_matrix, position)
            return fcl.CollisionObject(shape, transform), ('cylinder', radius, height, position, rotation_matrix)

        def create_capped_cylinder_objects(radius, height, position, rotation_matrix):
            """Create cylinder + top/bottom disk caps to ensure solid collision detection."""
            cap_thickness = 0.001
            main_cyl = create_cylinder_collision_object(radius, height, position, rotation_matrix)
            top_offset = rotation_matrix @ np.array([0, 0, height/2])
            top_cap = create_cylinder_collision_object(radius, cap_thickness, position + top_offset, rotation_matrix)
            bot_offset = rotation_matrix @ np.array([0, 0, -height/2])
            bot_cap = create_cylinder_collision_object(radius, cap_thickness, position + bot_offset, rotation_matrix)
            return main_cyl, top_cap, bot_cap

        def compute_aabb_corners(geom_info):
            if geom_info[0] == 'box':
                _, dims, center, R = geom_info
                dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
                local_corners = np.array([[-dx,-dy,-dz], [dx,-dy,-dz], [dx,dy,-dz], [-dx,dy,-dz],
                                          [-dx,-dy,dz], [dx,-dy,dz], [dx,dy,dz], [-dx,dy,dz]])
            else:
                _, radius, height, center, R = geom_info
                h = height/2
                angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
                circle = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
                local_corners = np.vstack([np.column_stack([circle, np.full(len(angles), -h)]),
                                           np.column_stack([circle, np.full(len(angles), h)])])
            return np.array([center + R @ c for c in local_corners])

        def build_airframe_collision_objects(left_joint_angle_rad, right_joint_angle_rad):
            objects = {}
            objects['base'] = create_box_collision_object(base_box_dims, np.array([0, 0, 0]), np.eye(3))

            front_arm_center = np.array([0.035 + front_arm_len/2, 0, 0])
            objects['front_arm'] = create_box_collision_object([front_arm_len, arm_width, arm_height], front_arm_center, np.eye(3))

            R_front_motor = rpy_to_rotation_matrix(*front_prop_rot)
            front_motor_center = np.array([0.035 + front_arm_len, 0, 0])
            fm_main, fm_top, fm_bot = create_capped_cylinder_objects(motor_radius, motor_height, front_motor_center, R_front_motor)
            objects['front_motor'] = fm_main
            objects['front_motor_top'] = fm_top
            objects['front_motor_bot'] = fm_bot

            back_arm_center = np.array([-0.035 - back_arm_len/2, 0, 0])
            objects['back_arm'] = create_box_collision_object([back_arm_len, arm_width, arm_height], back_arm_center, np.eye(3))

            R_back_base = rpy_to_rotation_matrix(0, 0, math.pi)
            R_back_motor = R_back_base @ rpy_to_rotation_matrix(*back_prop_rot)
            back_motor_center = np.array([-0.035 - back_arm_len, 0, 0])
            bm_main, bm_top, bm_bot = create_capped_cylinder_objects(motor_radius, motor_height, back_motor_center, R_back_motor)
            objects['back_motor'] = bm_main
            objects['back_motor_top'] = bm_top
            objects['back_motor_bot'] = bm_bot

            left_seg1_base = np.array([0, 0.025, 0])
            R_left_seg1 = rpy_to_rotation_matrix(*side_base_rot_left)
            left_seg1_center = left_seg1_base + R_left_seg1 @ np.array([side_seg1_len/2, 0, 0])
            objects['left_seg1'] = create_box_collision_object([side_seg1_len, arm_width, arm_height], left_seg1_center, R_left_seg1)

            left_joint_pos = left_seg1_base + R_left_seg1 @ np.array([side_seg1_len, 0, 0])
            R_left_joint = rpy_to_rotation_matrix(0, 0, left_joint_angle_rad)
            R_left_seg2 = R_left_seg1 @ R_left_joint
            left_seg2_center = left_joint_pos + R_left_seg2 @ np.array([side_seg2_len_left/2, 0, 0])
            objects['left_seg2'] = create_box_collision_object([side_seg2_len_left, arm_width, arm_height], left_seg2_center, R_left_seg2)

            left_motor_pos = left_joint_pos + R_left_seg2 @ np.array([side_seg2_len_left, 0, 0])
            R_left_motor = R_left_seg2 @ rpy_to_rotation_matrix(*side_motor_rot_left)
            lm_main, lm_top, lm_bot = create_capped_cylinder_objects(motor_radius, motor_height, left_motor_pos, R_left_motor)
            objects['left_motor'] = lm_main
            objects['left_motor_top'] = lm_top
            objects['left_motor_bot'] = lm_bot

            if ADD_ROD:
                R_rod_left = R_left_motor @ rpy_to_rotation_matrix(0, rod_left_pitch, rod_left_yaw)
                rod_left_center_world = left_motor_pos + R_left_motor @ rod_left_center_motor
                objects['left_rod'] = create_cylinder_collision_object(rod_radius, rod_left_len, rod_left_center_world, R_rod_left)

            right_seg1_base = np.array([0, -0.025, 0])
            R_right_seg1 = rpy_to_rotation_matrix(*side_base_rot_right)
            right_seg1_center = right_seg1_base + R_right_seg1 @ np.array([side_seg1_len/2, 0, 0])
            objects['right_seg1'] = create_box_collision_object([side_seg1_len, arm_width, arm_height], right_seg1_center, R_right_seg1)

            right_joint_pos = right_seg1_base + R_right_seg1 @ np.array([side_seg1_len, 0, 0])
            R_right_joint = rpy_to_rotation_matrix(0, 0, right_joint_angle_rad)
            R_right_seg2 = R_right_seg1 @ R_right_joint
            right_seg2_center = right_joint_pos + R_right_seg2 @ np.array([side_seg2_len_right/2, 0, 0])
            objects['right_seg2'] = create_box_collision_object([side_seg2_len_right, arm_width, arm_height], right_seg2_center, R_right_seg2)

            right_motor_pos = right_joint_pos + R_right_seg2 @ np.array([side_seg2_len_right, 0, 0])
            R_right_motor = R_right_seg2 @ rpy_to_rotation_matrix(*side_motor_rot_right)
            rm_main, rm_top, rm_bot = create_capped_cylinder_objects(motor_radius, motor_height, right_motor_pos, R_right_motor)
            objects['right_motor'] = rm_main
            objects['right_motor_top'] = rm_top
            objects['right_motor_bot'] = rm_bot

            if ADD_ROD:
                R_rod_right = R_right_motor @ rpy_to_rotation_matrix(0, rod_right_pitch, rod_right_yaw)
                rod_right_center_world = right_motor_pos + R_right_motor @ rod_right_center_motor
                objects['right_rod'] = create_cylinder_collision_object(rod_radius, rod_right_len, rod_right_center_world, R_rod_right)

            return objects

        def check_any_collision(objects):
            motor_parts = ['front_motor', 'back_motor', 'left_motor', 'right_motor']
            motor_all = motor_parts + [m + '_top' for m in motor_parts] + [m + '_bot' for m in motor_parts]
            collision_pairs = [
                ('base', 'front_motor'), ('base', 'back_motor'), ('base', 'left_motor'), ('base', 'right_motor'),
                ('base', 'left_seg2'), ('base', 'right_seg2'),
                ('front_arm', 'left_motor'), ('front_arm', 'right_motor'), ('front_arm', 'left_seg2'), ('front_arm', 'right_seg2'),
                ('back_arm', 'left_motor'), ('back_arm', 'right_motor'), ('back_arm', 'left_seg2'), ('back_arm', 'right_seg2'),
                ('front_motor', 'left_motor'), ('front_motor', 'right_motor'), ('front_motor', 'left_seg2'), ('front_motor', 'right_seg2'),
                ('back_motor', 'left_motor'), ('back_motor', 'right_motor'), ('back_motor', 'left_seg2'), ('back_motor', 'right_seg2'),
                ('left_seg1', 'right_seg1'), ('left_seg1', 'right_seg2'), ('left_seg1', 'right_motor'),
                ('right_seg1', 'left_seg2'), ('right_seg1', 'left_motor'),
                ('left_seg2', 'right_seg2'), ('left_seg2', 'right_motor'),
                ('right_seg2', 'left_motor'),
                ('left_motor', 'right_motor'),
            ]
            for m1 in motor_all:
                for m2 in motor_all:
                    base1, base2 = m1.split('_')[0] + '_motor', m2.split('_')[0] + '_motor'
                    if base1 != base2 and (m1, m2) not in collision_pairs and (m2, m1) not in collision_pairs:
                        collision_pairs.append((m1, m2))
            arm_parts = ['front_arm', 'back_arm', 'left_seg1', 'left_seg2', 'right_seg1', 'right_seg2', 'base']
            connected_pairs = {
                ('front_arm', 'front_motor'), ('back_arm', 'back_motor'),
                ('left_seg2', 'left_motor'), ('right_seg2', 'right_motor'),
            }
            for arm in arm_parts:
                for cap in [m + '_top' for m in motor_parts] + [m + '_bot' for m in motor_parts]:
                    motor_base = cap.rsplit('_', 1)[0]
                    if (arm, motor_base) in connected_pairs or (motor_base, arm) in connected_pairs:
                        continue
                    if (arm, cap) not in collision_pairs and (cap, arm) not in collision_pairs:
                        collision_pairs.append((arm, cap))

            if ADD_ROD:
                left_same_side = {'left_seg1', 'left_seg2', 'left_motor', 'left_motor_top', 'left_motor_bot'}
                right_same_side = {'right_seg1', 'right_seg2', 'right_motor', 'right_motor_top', 'right_motor_bot'}
                all_parts = set(objects.keys()) - {'left_rod', 'right_rod'}
                for part in all_parts:
                    if part not in left_same_side and ('left_rod', part) not in collision_pairs:
                        collision_pairs.append(('left_rod', part))
                    if part not in right_same_side and ('right_rod', part) not in collision_pairs:
                        collision_pairs.append(('right_rod', part))
                collision_pairs.append(('left_rod', 'right_rod'))

            request = fcl.CollisionRequest(enable_contact=False)
            all_collisions = []
            if verbose:
                print("--- Colliding object pairs ---")
            for obj1_name, obj2_name in collision_pairs:
                if obj1_name not in objects or obj2_name not in objects:
                    continue
                result = fcl.CollisionResult()
                fcl.collide(objects[obj1_name][0], objects[obj2_name][0], request, result)
                if result.is_collision:
                    all_collisions.append((obj1_name, obj2_name))
                    if verbose:
                        print(f"Collision detected: {obj1_name} <-> {obj2_name}")
            if all_collisions:
                return True, all_collisions[0], all_collisions
            return False, None, []

        max_angle_search_deg = 180.0
        delta_rad = math.radians(delta_deg)
        max_rad = math.radians(max_angle_search_deg)

        def check_collision(left_rad, right_rad):
            objects = build_airframe_collision_objects(left_rad, -right_rad)
            return check_any_collision(objects)[0]

        def get_collision_involvement(left_rad, right_rad):
            """Returns (has_collision, left_involved, right_involved)."""
            objects = build_airframe_collision_objects(left_rad, -right_rad)
            has_collision, _, all_collisions = check_any_collision(objects)
            if not has_collision:
                return False, False, False
            left_movable = {'left_seg2', 'left_motor', 'left_motor_top', 'left_motor_bot', 'left_rod'}
            right_movable = {'right_seg2', 'right_motor', 'right_motor_top', 'right_motor_bot', 'right_rod'}
            left_involved = any(p1 in left_movable or p2 in left_movable for p1, p2 in all_collisions)
            right_involved = any(p1 in right_movable or p2 in right_movable for p1, p2 in all_collisions)
            return True, left_involved, right_involved

        def bisect_single_arm(fixed_angle, is_left_fixed):
            """Bisect to find max angle for the non-fixed arm."""
            lo, hi = fixed_angle, max_rad
            check = (lambda a: check_collision(fixed_angle, a)) if is_left_fixed else (lambda a: check_collision(a, fixed_angle))
            if not check(max_rad):
                return max_rad
            while hi - lo > delta_rad:
                mid = (lo + hi) / 2
                if check(mid):
                    hi = mid
                else:
                    lo = mid
            return lo

        if not check_collision(max_rad, max_rad):
            max_left_angle, max_right_angle = max_rad, max_rad
        else:
            lo, hi = 0.0, max_rad
            while hi - lo > delta_rad:
                mid = (lo + hi) / 2
                if check_collision(mid, mid):
                    hi = mid
                else:
                    lo = mid
            _, left_involved, right_involved = get_collision_involvement(hi, hi)
            if left_involved and right_involved:
                max_left_angle, max_right_angle = lo, lo
            elif left_involved:
                max_left_angle = lo
                max_right_angle = bisect_single_arm(lo, is_left_fixed=True)
            else:
                max_right_angle = lo
                max_left_angle = bisect_single_arm(lo, is_left_fixed=False)

        max_left_angle_deg = math.degrees(max_left_angle)
        max_right_angle_deg = -math.degrees(max_right_angle)

        objects_at_max = build_airframe_collision_objects(max_left_angle, -max_right_angle)
        all_corners = np.vstack([compute_aabb_corners(obj_geom[1]) for obj_geom in objects_at_max.values()])
        aabb_min = all_corners.min(axis=0)
        aabb_max = all_corners.max(axis=0)

        objects_unfolded = build_airframe_collision_objects(0, 0)
        unfolded_corners = np.vstack([compute_aabb_corners(obj_geom[1]) for obj_geom in objects_unfolded.values()])
        unfolded_min = unfolded_corners.min(axis=0)
        unfolded_max = unfolded_corners.max(axis=0)

        aabb_info = {
            'aabb_min': aabb_min.tolist(),
            'aabb_max': aabb_max.tolist(),
            'max_width': aabb_max[1] - aabb_min[1],
            'max_height': aabb_max[2] - aabb_min[2],
            'max_length': aabb_max[0] - aabb_min[0],
            'unfolded_width': unfolded_max[1] - unfolded_min[1],
            'unfolded_height': unfolded_max[2] - unfolded_min[2],
            'unfolded_length': unfolded_max[0] - unfolded_min[0],
        }
        
        if verbose:
            def _visualize_from_fcl_objects(objects, delta_deg, colliding_parts):
                """Visualize using the actual FCL collision objects - guarantees match."""
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                def box_vertices(dims, center, R):
                    dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
                    corners = np.array([[-dx,-dy,-dz], [dx,-dy,-dz], [dx,dy,-dz], [-dx,dy,-dz],
                                        [-dx,-dy,dz], [dx,-dy,dz], [dx,dy,dz], [-dx,dy,dz]])
                    return np.array([center + R @ c for c in corners])

                def add_box(dims, center, R, color, alpha=0.5, label=None):
                    verts = box_vertices(dims, center, R)
                    faces = [[verts[j] for j in [0,1,2,3]], [verts[j] for j in [4,5,6,7]],
                            [verts[j] for j in [0,1,5,4]], [verts[j] for j in [2,3,7,6]],
                            [verts[j] for j in [0,3,7,4]], [verts[j] for j in [1,2,6,5]]]
                    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5))
                    if label:
                        ax.text(center[0], center[1], center[2], label, fontsize=8)

                def add_cylinder(radius, height, center, R, color, alpha=0.5, label=None):
                    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                    top = np.array([center + R @ np.array([radius*np.cos(a), radius*np.sin(a), height/2]) for a in angles])
                    bot = np.array([center + R @ np.array([radius*np.cos(a), radius*np.sin(a), -height/2]) for a in angles])
                    for i in range(len(top)):
                        face = [bot[i], bot[(i+1)%len(top)], top[(i+1)%len(top)], top[i]]
                        ax.add_collection3d(Poly3DCollection([face], alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.3))
                    ax.add_collection3d(Poly3DCollection([top], alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.3))
                    ax.add_collection3d(Poly3DCollection([bot], alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.3))
                    if label:
                        ax.text(center[0], center[1], center[2], label, fontsize=8)

                color_map = {
                    'base': 'gray', 'front_arm': 'orange', 'back_arm': 'orange',
                    'left_seg1': 'orange', 'left_seg2': 'yellow', 'right_seg1': 'orange', 'right_seg2': 'yellow',
                    'front_motor': 'cyan', 'back_motor': 'cyan', 'left_motor': 'lime', 'right_motor': 'lime',
                    'left_rod': 'brown', 'right_rod': 'brown',
                }
                label_map = {'front_motor': 'M0', 'left_motor': 'M1', 'back_motor': 'M2', 'right_motor': 'M3'}

                for name, (_, geom_info) in objects.items():
                    if '_top' in name or '_bot' in name:
                        base_name = name.rsplit('_', 1)[0]
                        color = 'blue' if base_name in ['front_motor', 'back_motor'] else 'darkgreen'
                        color = 'red' if name in colliding_parts else color
                    else:
                        color = 'red' if name in colliding_parts else color_map.get(name, 'gray')

                    if geom_info[0] == 'box':
                        _, dims, center, R = geom_info
                        add_box(dims, center, R, color, label=name if name in label_map else None)
                    else:
                        _, radius, height, center, R = geom_info
                        add_cylinder(radius, height, center, R, color, label=label_map.get(name))

                ax.scatter([0], [0], [0], color='red', s=100, marker='x')
                ax.set_xlim([-0.2, 0.2])
                ax.set_ylim([-0.2, 0.2])
                ax.set_zlim([-0.2, 0.2])
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                collision_str = f'\nColliding: {", ".join(sorted(colliding_parts))}' if colliding_parts else ''
                ax.set_title(f'FCL Collision Objects (delta={delta_deg}°){collision_str}')
                plt.tight_layout()
                import os
                os.makedirs("results", exist_ok=True)
                plt.savefig(f"results/debug_visualization_arm_max_angles.png")
                plt.show()

            objects_vis = build_airframe_collision_objects(max_left_angle, -max_right_angle)
            _visualize_from_fcl_objects(objects_vis, delta_deg, [])

        return max_left_angle_deg, max_right_angle_deg, aabb_info

    def validate_hover(self, max_min_motor_force_use: tuple[float, float], verbose: bool) -> tuple[bool, str]:
        """Validate that the airframe can achieve hover.

        Args:
            max_min_motor_force_use: (min, max) motor force usage range in [0,1]
            verbose: If True, print detailed validation info

        Returns:
            (is_valid, message): Tuple of validation result and message
        """
        from scipy.optimize import minimize

        alloc_data = self.compute_allocation_matrix()
        B = alloc_data['B']
        mass = alloc_data['mass']
        min_forces = alloc_data['min_forces']
        max_forces = alloc_data['max_forces']

        desired_wrench = np.array([0, 0, mass * 9.81, 0, 0, 0])

        adjusted_min_forces = max_min_motor_force_use[0] * (max_forces - min_forces) + min_forces
        adjusted_max_forces = max_min_motor_force_use[1] * (max_forces - min_forces) + min_forces
        bounds = [(adjusted_min_forces[i], adjusted_max_forces[i]) for i in range(len(min_forces))]

        def objective(f):
            wrench_error = B @ f - desired_wrench
            return 0.5 * np.sum(wrench_error ** 2)

        def jacobian(f):
            return B.T @ (B @ f - desired_wrench)

        rng = np.random.RandomState(42)
        best_result = None
        best_objective_value = np.inf

        for _ in range(10):
            f_init = rng.uniform(adjusted_min_forces, adjusted_max_forces)
            result = minimize(objective, f_init, method='L-BFGS-B', jac=jacobian, bounds=bounds,
                            options={'ftol': 1e-9, 'gtol': 1e-9})
            if result.success and result.fun < best_objective_value:
                best_result = result
                best_objective_value = result.fun

        result = best_result if best_result is not None else result

        if not result.success:
            msg = f"QP optimization failed: {result.message}"
            if verbose:
                print(f"[Validation] ✗ FAILED: {msg}")
            return False, msg

        optimal_forces = result.x
        achieved_wrench = B @ optimal_forces
        wrench_error = achieved_wrench - desired_wrench
        wrench_error_norm = np.linalg.norm(wrench_error)
        wrench_tolerance = 0.01

        if wrench_error_norm > wrench_tolerance * np.sqrt(6):
            msg = f"Cannot achieve hover wrench (error={wrench_error_norm:.3f}N)"
            if verbose:
                print(f"[Validation] ✗ INVALID: {msg}")
            return False, msg

        if verbose:
            print(f"[Validation] ✓ VALID: Can achieve hover (error={wrench_error_norm:.6f}N)")
        return True, "Valid airframe"

    def validate_arm_deflection(self, verbose: bool) -> bool:
        """Check that collision-derived torque response angles are usable.

        Args:
            verbose: If True, print detailed validation info

        Returns:
            is_valid: Boolean indicating if arm parameters are valid
        """
        angle_b = self.compute_end_angle_b()
        is_valid = (
            angle_b["start_angle_b_left"] >= TORQUE_RESPONSE_RAMP_ANGLE_DEG
            and angle_b["start_angle_b_right"] >= TORQUE_RESPONSE_RAMP_ANGLE_DEG
        )

        if verbose:
            print(f"[Arm Deflection Check] {'VALID' if is_valid else 'INVALID'}")
            print(f"  Left arm: start_angle_b={angle_b['start_angle_b_left']:.1f}°")
            print(f"  Right arm: start_angle_b={angle_b['start_angle_b_right']:.1f}°")

        return is_valid

    def generate_usd(self, output_dir: str | Path, torque_fn_left: str, torque_fn_right: str) -> Path:
        """Generate USD file for this airframe.

        Args:
            output_dir: Directory for output files

        Returns:
            Path to generated USD file
        """
        import subprocess
        import sys

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        script_path = Path(__file__).parent.parent / "scripts/convert_morphy_urdf.py"
        cmd = [sys.executable, str(script_path), "--headless", "--output_dir", str(output_dir),
               "--torque_fn_left", torque_fn_left, "--torque_fn_right", torque_fn_right,
               "--params_list"] + [str(v) for v in self.x_0_1]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"USD generation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        return output_dir / "morphy_prog.usd"

    def __repr__(self) -> str:
        return f"Airframe(mass={self.compute_total_mass():.4f}kg, x_0_1={self.x_0_1[:5].round(3)}...)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Airframe):
            return False
        return np.allclose(self.x_0_1, other.x_0_1, atol=1e-8)
