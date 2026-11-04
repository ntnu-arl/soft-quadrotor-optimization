from __future__ import annotations

import math
from pathlib import Path
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airframe_encoding import Airframe, PARAM_NAMES
from arm_detachment_torque import (
    motor_joint_torque_components_nm,
    one_sided_detachment_envelope_rad_nm,
    rodrigues_rotation,
    validate_one_sided_detachment_envelope,
)
from erc_torque_profiles import (
    DEFAULT_SPRING_CATALOG,
    DEFAULT_SPRING_ID,
    FormulaTorqueParameters,
    build_formula_erc_torque_function,
    formula_torque_profile_knots_rad_nm,
    tighten_knots_for_one_sided_detachment_envelope,
)
from rotor_dynamics import RotorDynamics


def _best_solution_build(side: str):
    if side not in {"left", "right"}:
        raise ValueError(f"Unexpected side: {side}")
    best_solution_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "behaviour_based_nmpc_cma_es"
        / "behaviour_erc_nmpc20260604_004258"
        / "best_solution.json"
    )
    best_solution = json.loads(best_solution_path.read_text(encoding="utf-8"))
    airframe_data = Airframe.from_x_0_1(
        [float(value) for value in best_solution["geometry_x_0_1"]]
    ).to_json_dict()
    if side == "left":
        motor_idx = 1
        bend_direction = 1
        mirror = False
        fn_name = "torque_fn_left"
        end_angle_rad = float(best_solution["torque_response_end_angle_left_rad"])
    else:
        motor_idx = 3
        bend_direction = -1
        mirror = True
        fn_name = "torque_fn_right"
        end_angle_rad = float(best_solution["torque_response_end_angle_right_rad"])
    parameters = FormulaTorqueParameters(
        float(best_solution["torque_response_a"]),
        float(best_solution["torque_response_b"]),
        float(best_solution["torque_response_c"]),
        float(best_solution["torque_response_d"]),
        end_angle_rad,
    )
    envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        motor_idx,
        parameters.end_angle_rad,
        64,
        1.1,
        bend_direction,
        math.radians(1.0),
    )
    started_at = time.perf_counter()
    build = build_formula_erc_torque_function(
        parameters,
        fn_name,
        mirror,
        envelope,
        0.005,
        DEFAULT_SPRING_CATALOG,
        DEFAULT_SPRING_ID,
        151,
        1.1,
        0.6,
    )
    elapsed_s = time.perf_counter() - started_at
    return build, envelope, elapsed_s


def _threshold_crossing_abs_angle_rad(
    output_torque_table: torch.Tensor,
    envelope_rad_nm: tuple[tuple[float, float], ...],
    tolerance_nm: float,
) -> float:
    envelope_angles = [float(angle) for angle, _ in envelope_rad_nm]
    required_torque = max(float(required) for _, required in envelope_rad_nm)
    crossing_window_rad = min(abs(angle) for angle in envelope_angles)
    if any(angle == 0.0 for angle in envelope_angles):
        raise ValueError("detachment envelope angles must be non-zero")
    if any(angle > 0.0 for angle in envelope_angles) and any(angle < 0.0 for angle in envelope_angles):
        raise ValueError("detachment envelope must be one-sided")

    positive_side = all(angle > 0.0 for angle in envelope_angles)
    abs_angles = torch.linspace(
        0.0,
        crossing_window_rad,
        2001,
        dtype=torch.float64,
    )
    query_angles = abs_angles if positive_side else -abs_angles
    table_angles = output_torque_table[:, 0].contiguous()
    table_torques = output_torque_table[:, 1].contiguous()
    indices = torch.searchsorted(table_angles, query_angles, right=True) - 1
    indices = torch.clamp(indices, 0, table_angles.numel() - 2)
    x0 = table_angles[indices]
    x1 = table_angles[indices + 1]
    y0 = table_torques[indices]
    y1 = table_torques[indices + 1]
    torques = y0 + (query_angles - x0) / (x1 - x0) * (y1 - y0)
    if positive_side:
        satisfied = torques <= -required_torque + tolerance_nm
    else:
        satisfied = torques >= required_torque - tolerance_nm
    if not bool(torch.any(satisfied)):
        raise AssertionError("threshold was not reached inside the crossing window")
    return float(abs_angles[torch.where(satisfied)[0][0]])


def _manual_components(airframe_data: dict, motor_idx: int, angle_rad: float) -> tuple[float, float]:
    motor_idx_list = airframe_data["motor_idx_list"]
    motor_directions = airframe_data["motor_directions"]
    max_forces, max_motor_torques, _ = RotorDynamics.get_motor_max_force_torque_and_rps(motor_idx_list)

    motor_position = np.array(airframe_data["motor_positions_body_frame"][motor_idx])
    joint_position = np.array(airframe_data[f"joint_{motor_idx}_position"])
    joint_axis = np.array(airframe_data[f"joint_{motor_idx}_axis"])
    thrust_direction = np.array(airframe_data["motor_thrust_directions"][motor_idx])

    r_vec = rodrigues_rotation(motor_position - joint_position, joint_axis, angle_rad)
    thrust_direction = rodrigues_rotation(thrust_direction, joint_axis, angle_rad)

    thrust_torque = np.dot(np.cross(r_vec, max_forces[motor_idx] * thrust_direction), joint_axis)
    reaction_torque = np.dot(
        motor_directions[motor_idx] * max_motor_torques[motor_idx] * thrust_direction,
        joint_axis,
    )
    return float(thrust_torque), float(reaction_torque)


def test_motor_joint_torque_components_match_existing_formula() -> None:
    airframe_data = Airframe.from_x_0_1(np.full(len(PARAM_NAMES), 0.5)).to_json_dict()
    for motor_idx in (1, 3):
        for angle_deg in (-35.0, -5.0, 0.0, 22.0, 48.0):
            angle_rad = math.radians(angle_deg)
            expected = _manual_components(airframe_data, motor_idx, angle_rad)
            actual = motor_joint_torque_components_nm(airframe_data, motor_idx, angle_rad)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_one_sided_envelope_uses_configured_bend_direction() -> None:
    airframe_data = Airframe.from_x_0_1(np.full(len(PARAM_NAMES), 0.5)).to_json_dict()
    envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        1,
        math.radians(50.0),
        5,
        1.25,
        1,
        math.radians(10.0),
    )
    for angle, required_torque in envelope:
        assert angle > 0.0
        expected = 1.25 * abs(sum(_manual_components(airframe_data, 1, angle)))
        assert math.isclose(required_torque, expected, rel_tol=1e-12, abs_tol=1e-12)

    mirrored_envelope = one_sided_detachment_envelope_rad_nm(
        airframe_data,
        3,
        math.radians(50.0),
        5,
        1.25,
        -1,
        math.radians(10.0),
    )
    assert all(angle < 0.0 for angle, _ in mirrored_envelope)


def test_weak_requested_profile_is_tightened() -> None:
    parameters = FormulaTorqueParameters(
        a=0.001,
        b=0.5,
        c=0.0,
        d=0.0,
        end_angle_rad=math.radians(60.0),
    )
    knots = formula_torque_profile_knots_rad_nm(parameters, False)
    tightened = tighten_knots_for_one_sided_detachment_envelope(
        knots,
        ((math.radians(20.0), 0.2),),
        0.0,
    )
    tightened_by_angle = dict(tightened)
    assert math.isclose(tightened_by_angle[math.radians(20.0)], -0.2, rel_tol=0.0, abs_tol=1e-12)


def test_formula_erc_recovers_from_weak_initial_parameters_for_flat_threshold() -> None:
    end_angle_rad = 0.8881748761857863
    required_torque_nm = 0.10044689771612794
    envelope = tuple(
        (float(angle), required_torque_nm)
        for angle in np.linspace(math.radians(1.0), end_angle_rad, 64)
    )
    parameters = FormulaTorqueParameters(
        a=0.1880058493936743,
        b=0.023501114238595938,
        c=0.0,
        d=0.15088678510837414,
        end_angle_rad=end_angle_rad,
    )
    build = build_formula_erc_torque_function(
        parameters,
        "torque_fn_left",
        False,
        envelope,
        0.005,
        DEFAULT_SPRING_CATALOG,
        DEFAULT_SPRING_ID,
        151,
        1.1,
        0.6,
    )
    crossing_angle_rad = _threshold_crossing_abs_angle_rad(
        build.design_result.output_torque_table,
        envelope,
        1e-9,
    )
    assert crossing_angle_rad <= math.radians(1.0)
    assert len(build.requested_knots_deg_nm) == 9


def test_right_side_failing_case_reaches_detachment_threshold_inside_crossing_window() -> None:
    build, envelope, _ = _best_solution_build("right")
    crossing_angle_rad = _threshold_crossing_abs_angle_rad(
        build.design_result.output_torque_table,
        envelope,
        1e-9,
    )
    assert crossing_angle_rad <= math.radians(1.0)


def test_left_side_best_solution_reaches_detachment_threshold_inside_crossing_window() -> None:
    build, envelope, _ = _best_solution_build("left")
    crossing_angle_rad = _threshold_crossing_abs_angle_rad(
        build.design_result.output_torque_table,
        envelope,
        1e-9,
    )
    assert crossing_angle_rad <= math.radians(1.0)


def test_final_torque_functions_cross_zero_at_aligned_zero_angle() -> None:
    for side in ("left", "right"):
        build, _, _ = _best_solution_build(side)
        zero_torque = float(build.torque_function(torch.tensor([0.0], dtype=torch.float64))[0])
        assert math.isclose(zero_torque, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_formula_erc_reports_formula_and_secondary_requested_knots() -> None:
    for side in ("left", "right"):
        build, _, _ = _best_solution_build(side)
        assert len(build.requested_knots_deg_nm) == 9


def test_best_solution_erc_repairs_complete_under_ten_seconds_per_side() -> None:
    for side in ("left", "right"):
        _, _, elapsed_s = _best_solution_build(side)
        assert elapsed_s < 10.0


def test_final_serialized_torque_function_crosses_zero_at_aligned_zero_angle() -> None:
    build, _, _ = _best_solution_build("right")
    namespace = {"torch": torch}
    exec(build.function_string, namespace)
    zero_torque = float(
        namespace["torque_fn_right"](torch.tensor([0.0], dtype=torch.float64))[0]
    )
    assert math.isclose(zero_torque, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_infeasible_final_response_raises_clear_diagnostic() -> None:
    parameters = FormulaTorqueParameters(
        a=0.2,
        b=0.5,
        c=0.1,
        d=0.2,
        end_angle_rad=math.radians(78.0),
    )
    try:
        build_formula_erc_torque_function(
            parameters,
            "torque_fn_left",
            False,
            ((math.radians(10.0), 0.41),),
            0.0,
            DEFAULT_SPRING_CATALOG,
            DEFAULT_SPRING_ID,
            151,
            1.1,
            0.6,
        )
    except ValueError as exc:
        message = str(exc)
        assert "violation" in message
        assert "side=positive" in message
        assert "crossing window" in message
        assert "best actual torque" in message
        assert "required torque" in message
        assert "RMSE" in message
        assert "best a,b,c,d" in message
    else:
        raise AssertionError("expected motor-detachment envelope violation")


def test_validate_one_sided_detachment_envelope_accepts_opposing_profile() -> None:
    def torque_function(x: torch.Tensor) -> torch.Tensor:
        return -0.3 * torch.sign(x)

    validate_one_sided_detachment_envelope(
        torque_function,
        ((math.radians(5.0), 0.2), (-math.radians(30.0), 0.2)),
        1e-12,
    )


if __name__ == "__main__":
    test_motor_joint_torque_components_match_existing_formula()
    test_one_sided_envelope_uses_configured_bend_direction()
    test_weak_requested_profile_is_tightened()
    test_right_side_failing_case_reaches_detachment_threshold_inside_crossing_window()
    test_left_side_best_solution_reaches_detachment_threshold_inside_crossing_window()
    test_final_torque_functions_cross_zero_at_aligned_zero_angle()
    test_formula_erc_reports_formula_and_secondary_requested_knots()
    test_best_solution_erc_repairs_complete_under_ten_seconds_per_side()
    test_final_serialized_torque_function_crosses_zero_at_aligned_zero_angle()
    test_infeasible_final_response_raises_clear_diagnostic()
    test_validate_one_sided_detachment_envelope_accepts_opposing_profile()
