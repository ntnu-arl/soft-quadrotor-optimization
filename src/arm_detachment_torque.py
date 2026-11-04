from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
import torch

from rotor_dynamics import RotorDynamics


@dataclass(frozen=True)
class DetachmentEnvelopeViolation:
    side: str
    angle_rad: float
    actual_torque_nm: float
    required_torque_nm: float
    violation_nm: float


def rodrigues_rotation(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError(f"joint axis must have positive norm, got {axis_norm}")
    unit_axis = axis / axis_norm
    return (
        vector * math.cos(angle_rad)
        + np.cross(unit_axis, vector) * math.sin(angle_rad)
        + unit_axis * np.dot(unit_axis, vector) * (1.0 - math.cos(angle_rad))
    )


def motor_joint_torque_components_nm(
    airframe_data: dict,
    motor_idx: int,
    arm_angle_rad: float,
) -> tuple[float, float]:
    if motor_idx not in (1, 3):
        raise ValueError(f"motor_idx must be 1 or 3, got {motor_idx}")

    motor_idx_list = airframe_data["motor_idx_list"]
    motor_directions = airframe_data["motor_directions"]
    max_forces, max_reaction_torques, _ = RotorDynamics.get_motor_max_force_torque_and_rps(motor_idx_list)

    max_thrust = float(max_forces[motor_idx])
    max_reaction_torque = float(max_reaction_torques[motor_idx])
    motor_direction = float(motor_directions[motor_idx])

    motor_position = np.asarray(airframe_data["motor_positions_body_frame"][motor_idx], dtype=float)
    thrust_direction = np.asarray(airframe_data["motor_thrust_directions"][motor_idx], dtype=float)
    joint_position = np.asarray(airframe_data[f"joint_{motor_idx}_position"], dtype=float)
    joint_axis = np.asarray(airframe_data[f"joint_{motor_idx}_axis"], dtype=float)

    r_motor_at_zero = motor_position - joint_position
    r_motor = rodrigues_rotation(r_motor_at_zero, joint_axis, float(arm_angle_rad))
    thrust_direction = rodrigues_rotation(thrust_direction, joint_axis, float(arm_angle_rad))

    thrust_torque = float(np.dot(np.cross(r_motor, max_thrust * thrust_direction), joint_axis))
    reaction_torque = float(np.dot(motor_direction * max_reaction_torque * thrust_direction, joint_axis))
    return thrust_torque, reaction_torque


def motor_joint_torque_nm(
    airframe_data: dict,
    motor_idx: int,
    arm_angle_rad: float,
) -> float:
    thrust_torque, reaction_torque = motor_joint_torque_components_nm(
        airframe_data,
        motor_idx,
        arm_angle_rad,
    )
    return thrust_torque + reaction_torque


def one_sided_detachment_envelope_rad_nm(
    airframe_data: dict,
    motor_idx: int,
    end_angle_rad: float,
    n_samples: int,
    safety_factor: float,
    bend_sign: int,
    min_abs_angle_rad: float,
) -> tuple[tuple[float, float], ...]:
    if end_angle_rad <= 0.0:
        raise ValueError(f"end_angle_rad must be positive, got {end_angle_rad}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if safety_factor <= 0.0:
        raise ValueError(f"safety_factor must be positive, got {safety_factor}")
    if bend_sign not in (-1, 1):
        raise ValueError(f"bend_sign must be -1 or 1, got {bend_sign}")
    if min_abs_angle_rad < 0.0:
        raise ValueError(f"min_abs_angle_rad must be non-negative, got {min_abs_angle_rad}")
    if min_abs_angle_rad >= end_angle_rad:
        raise ValueError(
            f"min_abs_angle_rad must be smaller than end_angle_rad, got "
            f"{min_abs_angle_rad} >= {end_angle_rad}"
        )

    abs_angles = np.linspace(float(min_abs_angle_rad), float(end_angle_rad), int(n_samples))
    envelope = []
    for abs_angle in abs_angles:
        signed_angle = float(bend_sign) * float(abs_angle)
        motor_torque = motor_joint_torque_nm(airframe_data, motor_idx, signed_angle)
        required_torque = float(safety_factor * abs(motor_torque))
        envelope.append((signed_angle, required_torque))
    return tuple(envelope)


def validate_one_sided_detachment_envelope(
    torque_function: Callable[[torch.Tensor], torch.Tensor],
    envelope_rad_nm: Sequence[tuple[float, float]],
    tolerance_nm: float,
) -> None:
    if tolerance_nm < 0.0:
        raise ValueError(f"tolerance_nm must be non-negative, got {tolerance_nm}")
    violations = one_sided_detachment_envelope_violations(
        torque_function,
        envelope_rad_nm,
        tolerance_nm,
    )
    if len(violations) > 0:
        violation_parts = []
        positive_violations = [violation for violation in violations if violation.side == "positive"]
        negative_violations = [violation for violation in violations if violation.side == "negative"]
        if len(positive_violations) > 0:
            violation = positive_violations[0]
            violation_parts.append(
                "positive side "
                f"angle={violation.angle_rad:.6g} rad "
                f"torque={violation.actual_torque_nm:.6g} Nm "
                f"limit={-violation.required_torque_nm:.6g} Nm"
            )
        if len(negative_violations) > 0:
            violation = negative_violations[0]
            violation_parts.append(
                "negative side "
                f"angle={violation.angle_rad:.6g} rad "
                f"torque={violation.actual_torque_nm:.6g} Nm "
                f"limit={violation.required_torque_nm:.6g} Nm"
            )
        raise ValueError("ERC torque table violates motor-detachment envelope: " + "; ".join(violation_parts))


def one_sided_detachment_envelope_violations(
    torque_function: Callable[[torch.Tensor], torch.Tensor],
    envelope_rad_nm: Sequence[tuple[float, float]],
    tolerance_nm: float,
) -> tuple[DetachmentEnvelopeViolation, ...]:
    if tolerance_nm < 0.0:
        raise ValueError(f"tolerance_nm must be non-negative, got {tolerance_nm}")
    if len(envelope_rad_nm) == 0:
        return ()

    angles = torch.tensor(
        [float(angle) for angle, _ in envelope_rad_nm],
        dtype=torch.float64,
    )
    required_torques = torch.tensor(
        [float(required) for _, required in envelope_rad_nm],
        dtype=torch.float64,
    )
    torques = torque_function(angles)

    positive_side = angles > 0.0
    negative_side = angles < 0.0
    if torch.any(~(positive_side | negative_side)):
        raise ValueError("detachment envelope angles must be non-zero")

    violations = []
    for index in range(int(angles.numel())):
        angle = float(angles[index])
        required_torque = float(required_torques[index])
        actual_torque = float(torques[index])
        if angle > 0.0:
            violation_nm = actual_torque - (-required_torque)
            if violation_nm > tolerance_nm:
                violations.append(
                    DetachmentEnvelopeViolation(
                        "positive",
                        angle,
                        actual_torque,
                        required_torque,
                        violation_nm,
                    )
                )
        if angle < 0.0:
            violation_nm = required_torque - actual_torque
            if violation_nm > tolerance_nm:
                violations.append(
                    DetachmentEnvelopeViolation(
                        "negative",
                        angle,
                        actual_torque,
                        required_torque,
                        violation_nm,
                    )
                )
    return tuple(violations)
