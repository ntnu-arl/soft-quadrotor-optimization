from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cma
import numpy as np
import torch

from arm_detachment_torque import DetachmentEnvelopeViolation
from erc_design import (
    ERCDesignConfig,
    ERCDesigner,
    PiecewiseLinearTorqueProfile,
    SpringCatalog,
    interpolate_output_torque_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPRING_CATALOG = PROJECT_ROOT / "configs" / "springs.yaml"
DEFAULT_SPRING_ID = "durovis_0.9x6.1x21"
FORMULA_TORQUE_A_BOUNDS_NM = (0.001, 0.4)
FORMULA_TORQUE_C_BOUNDS_NM = (0.0, 0.4)
FORMULA_TORQUE_D_BOUNDS_NM = (0.0, 0.399)
FORMULA_TORQUE_B_BOUNDS = (0.01, 0.99)
DETACHMENT_N_SAMPLES = 64
DETACHMENT_SAFETY_FACTOR = 1.1
DETACHMENT_MIN_ANGLE_DEG = 1.0
DETACHMENT_REQUEST_MARGIN_NM = 0.005
DETACHMENT_ENVELOPE_TOLERANCE_NM = 1e-9
CONSTRAINED_SECONDARY_HIGH_THETA_FRACTION = 0.85
CONSTRAINED_SECONDARY_STRIDE = 4
CONSTRAINED_NEAR_ZERO_GAIN = 1.4
CONSTRAINED_LEFT_NEAR_GAIN = 2.0
CONSTRAINED_LEFT_NEAR_WIDTH_DEG = 4.0
CONSTRAINED_CMA_VECTOR_LENGTH = 10
CONSTRAINED_CMA_SECONDARY_COUNT = 3
CONSTRAINED_CMA_BUDGET_S = 9.0
CONSTRAINED_CMA_SEED = 20260531
CONSTRAINED_CMA_POPSIZE = 12
CONSTRAINED_CMA_SIGMA0 = 0.22
CONSTRAINED_CMA_MAX_EVALUATIONS = 144
CONSTRAINED_CMA_SECONDARY_MIN_ANGLE_RAD = 1e-9
CONSTRAINED_CMA_INFEASIBLE_BASE_PENALTY = 1000.0
CONSTRAINED_CMA_BUILD_FAILURE_PENALTY = 1000000.0
CONSTRAINED_CMA_ZERO_TOLERANCE_NM = 1e-9


@dataclass(frozen=True)
class ERCTorqueFunctionBuild:
    torque_function: object
    function_string: str
    design_result: object
    requested_knots_deg_nm: tuple[tuple[float, float], ...]
    actual_parameters: tuple[float, ...]
    angle_offset_rad: float


@dataclass(frozen=True)
class FormulaTorqueParameters:
    a: float
    b: float
    c: float
    d: float
    end_angle_rad: float


@dataclass(frozen=True)
class FinalResponseEvaluation:
    violations: tuple[DetachmentEnvelopeViolation, ...]
    max_violation_nm: float
    max_violation: DetachmentEnvelopeViolation | None
    rmse_to_requested_nm: float
    domain_margin_rad: float


@dataclass(frozen=True)
class ConstrainedCandidateBuild:
    knots_rad_nm: tuple[tuple[float, float], ...]
    design_result: object
    angle_offset_rad: float
    evaluation: FinalResponseEvaluation
    actual_parameters: FormulaTorqueParameters


@dataclass(frozen=True)
class ConstrainedDecodedProfile:
    parameters: FormulaTorqueParameters
    formula_knots_rad_nm: tuple[tuple[float, float], ...]
    secondary_knots_rad_nm: tuple[tuple[float, float], ...]
    merged_knots_rad_nm: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ConstrainedCrossingWindow:
    side: str
    start_rad: float
    end_rad: float
    required_torque_nm: float


@dataclass(frozen=True)
class ConstrainedCandidateDiagnostic:
    score: float
    reason: str
    side_name: str
    crossing_window: ConstrainedCrossingWindow | None
    best_actual_torque_nm: float | None
    required_torque_nm: float | None
    rmse_to_requested_nm: float
    parameters: FormulaTorqueParameters | None


def build_erc_design(
    knots_deg_nm: Sequence[tuple[float, float]],
    *,
    spring_catalog_path: str | Path = DEFAULT_SPRING_CATALOG,
    spring_id: str = DEFAULT_SPRING_ID,
    n_grid: int = 1000,
    safety_factor: float = 1.1,
    angle_limits_deg: tuple[float, float] = (-90.0, 90.0),
):
    """Run full ERC design and return the ERCDesignResult."""
    knots = torch.tensor(knots_deg_nm, dtype=torch.float64)
    profile = PiecewiseLinearTorqueProfile.from_xy(
        knots[:, 0] * math.pi / 180.0,
        knots[:, 1],
    )
    spring = SpringCatalog.from_yaml(spring_catalog_path)[spring_id]
    designer = ERCDesigner(
        spring,
        ERCDesignConfig(
            n_grid=n_grid,
            safety_factor=safety_factor,
            angle_convention="joint",
            joint_angle_limits_rad=(
                math.radians(angle_limits_deg[0]),
                math.radians(angle_limits_deg[1]),
            ),
            invert_output_torque=True,
        ),
    )
    return designer.design(profile)


def build_erc_torque_table(
    knots_deg_nm: Sequence[tuple[float, float]],
    *,
    spring_catalog_path: str | Path = DEFAULT_SPRING_CATALOG,
    spring_id: str = DEFAULT_SPRING_ID,
    n_grid: int = 1000,
    safety_factor: float = 1.1,
    angle_limits_deg: tuple[float, float] = (-90.0, 90.0),
) -> torch.Tensor:
    """Build an Isaac-ready joint torque table from ERC profile knots.

    Returns:
        Torch tensor with columns ``[joint_angle_rad, torque_nm]``.
    """
    return build_erc_design(
        knots_deg_nm,
        spring_catalog_path=spring_catalog_path,
        spring_id=spring_id,
        n_grid=n_grid,
        safety_factor=safety_factor,
        angle_limits_deg=angle_limits_deg,
    ).output_torque_table


def torque_table_to_function_string(
    table: torch.Tensor,
    fn_name: str,
    extrapolation: str,
    flat_boundary_torques_nm: tuple[float, float] | None,
) -> str:
    """Serialize a torque table as a torch-callable function string."""

    if fn_name not in {"torque_fn_left", "torque_fn_right"}:
        raise ValueError(f"Unexpected torque function name: {fn_name}")
    if extrapolation not in {"raise", "flat"}:
        raise ValueError(f"Unexpected ERC torque extrapolation mode: {extrapolation}")

    table_cpu = table.detach().cpu()
    angles = [float(f"{float(v):.17g}") for v in table_cpu[:, 0]]
    torques = [float(f"{float(v):.17g}") for v in table_cpu[:, 1]]
    if extrapolation == "flat" and flat_boundary_torques_nm is None:
        raise ValueError("flat_boundary_torques_nm is required when extrapolation is 'flat'")
    if extrapolation == "raise" and flat_boundary_torques_nm is not None:
        raise ValueError("flat_boundary_torques_nm must be None when extrapolation is 'raise'")
    out_of_domain_logic = (
        f'raise ValueError("{fn_name} received an angle outside its ERC table domain")'
        if extrapolation == "raise"
        else "return torch.where(x < xp[0], _erc_flat_lower + x * 0.0, torch.where(x > xp[-1], _erc_flat_upper + x * 0.0, _erc_interp(x, xp, fp)))"
    )
    in_domain_logic = (
        "_erc_interp(x, xp, fp)"
        if extrapolation == "raise"
        else "torch.where(x < xp[0], _erc_flat_lower + x * 0.0, torch.where(x > xp[-1], _erc_flat_upper + x * 0.0, _erc_interp(x, xp, fp)))"
    )
    flat_lower = None if flat_boundary_torques_nm is None else float(flat_boundary_torques_nm[0])
    flat_upper = None if flat_boundary_torques_nm is None else float(flat_boundary_torques_nm[1])

    return f"""
_erc_angles = {angles!r}
_erc_torques = {torques!r}
_erc_flat_lower = {flat_lower!r}
_erc_flat_upper = {flat_upper!r}
_erc_cache = {{}}
def _erc_interp(x, xp, fp):
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = torch.clamp(idx, 0, xp.numel() - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
def {fn_name}(x):
    key = (str(x.device), x.dtype)
    if key not in _erc_cache:
        _erc_cache[key] = (
            torch.tensor(_erc_angles, device=x.device, dtype=x.dtype),
            torch.tensor(_erc_torques, device=x.device, dtype=x.dtype),
        )
    xp, fp = _erc_cache[key]
    if torch.any(x < xp[0]) or torch.any(x > xp[-1]):
        {out_of_domain_logic}
    return {in_domain_logic}
{fn_name}.erc_angle_min_rad = _erc_angles[0]
{fn_name}.erc_angle_max_rad = _erc_angles[-1]
"""


def formula_torque_profile_knots_rad_nm(
    parameters: FormulaTorqueParameters,
    mirror: bool,
) -> tuple[tuple[float, float], ...]:
    a = float(parameters.a)
    b = float(parameters.b)
    c = float(parameters.c)
    d = float(parameters.d)
    end_angle = float(parameters.end_angle_rad)
    if a < FORMULA_TORQUE_A_BOUNDS_NM[0] or a > FORMULA_TORQUE_A_BOUNDS_NM[1]:
        raise ValueError(f"a must be in {FORMULA_TORQUE_A_BOUNDS_NM}, got {a}")
    if b < FORMULA_TORQUE_B_BOUNDS[0] or b > FORMULA_TORQUE_B_BOUNDS[1]:
        raise ValueError(f"b must be in {FORMULA_TORQUE_B_BOUNDS}, got {b}")
    if c < FORMULA_TORQUE_C_BOUNDS_NM[0] or c > FORMULA_TORQUE_C_BOUNDS_NM[1]:
        raise ValueError(f"c must be in {FORMULA_TORQUE_C_BOUNDS_NM}, got {c}")
    if d < FORMULA_TORQUE_D_BOUNDS_NM[0] or d > FORMULA_TORQUE_D_BOUNDS_NM[1]:
        raise ValueError(f"d must be in {FORMULA_TORQUE_D_BOUNDS_NM}, got {d}")
    if end_angle <= 0.0:
        raise ValueError(f"end_angle_rad must be positive, got {end_angle}")

    point_1 = (math.radians(-1.0), 0.4)
    point_2 = (0.0, 0.0)
    point_3 = (math.radians(-a / -0.4), -a)
    point_5 = (end_angle - math.radians(1.0) - math.radians(-d / 0.4), -d)
    point_4 = (
        point_3[0] + b * (point_5[0] - point_3[0]),
        -c,
    )
    point_6 = (end_angle, -0.4)
    knots = (point_1, point_2, point_3, point_4, point_5, point_6)
    if any(right[0] <= left[0] for left, right in zip(knots, knots[1:])):
        raise ValueError(
            "formula torque-profile points must be strictly increasing in angle; "
            f"got {[point[0] for point in knots]} radians"
        )
    if mirror:
        knots = tuple(sorted((-angle, -torque) for angle, torque in knots))
    return tuple((float(angle), float(torque)) for angle, torque in knots)


def zero_crossing_angle_rad(table: torch.Tensor) -> float:
    angles = table[:, 0]
    torques = table[:, 1]
    exact_zero_indices = torch.where(torques == 0.0)[0]
    exact_zero_angles = angles[exact_zero_indices]

    crossing_indices = torch.where(torques[:-1] * torques[1:] < 0.0)[0]
    crossing_angles = []
    for index_tensor in crossing_indices:
        index = int(index_tensor)
        x0 = angles[index]
        x1 = angles[index + 1]
        y0 = torques[index]
        y1 = torques[index + 1]
        crossing_angles.append(x0 - y0 * (x1 - x0) / (y1 - y0))

    if exact_zero_angles.numel() == 0 and len(crossing_angles) == 0:
        raise ValueError("ERC repaired torque table has no zero crossing")

    all_crossing_angles = torch.cat(
        (exact_zero_angles, torch.stack(crossing_angles))
        if crossing_angles
        else (exact_zero_angles,)
    )
    closest_index = int(torch.argmin(torch.abs(all_crossing_angles)))
    return float(all_crossing_angles[closest_index])


def _interpolate_piecewise_linear_knots(
    knots_rad_nm: Sequence[tuple[float, float]],
    angle_rad: float,
) -> float:
    if angle_rad < knots_rad_nm[0][0] or angle_rad > knots_rad_nm[-1][0]:
        raise ValueError(
            f"angle_rad={angle_rad} is outside knot domain "
            f"[{knots_rad_nm[0][0]}, {knots_rad_nm[-1][0]}]"
        )
    for left, right in zip(knots_rad_nm, knots_rad_nm[1:]):
        left_angle, left_torque = left
        right_angle, right_torque = right
        if angle_rad <= right_angle:
            if right_angle == left_angle:
                raise ValueError(f"duplicate knot angle {right_angle}")
            fraction = (angle_rad - left_angle) / (right_angle - left_angle)
            return float(left_torque + fraction * (right_torque - left_torque))
    return float(knots_rad_nm[-1][1])


def tighten_knots_for_one_sided_detachment_envelope(
    knots_rad_nm: Sequence[tuple[float, float]],
    envelope_rad_nm: Sequence[tuple[float, float]],
    request_margin_nm: float,
) -> tuple[tuple[float, float], ...]:
    if request_margin_nm < 0.0:
        raise ValueError(f"request_margin_nm must be non-negative, got {request_margin_nm}")
    tightened_by_angle = {float(angle): float(torque) for angle, torque in knots_rad_nm}
    lower_angle = float(knots_rad_nm[0][0])
    upper_angle = float(knots_rad_nm[-1][0])

    for envelope_angle, required_torque in envelope_rad_nm:
        angle = float(envelope_angle)
        required = float(required_torque)
        if angle == 0.0:
            raise ValueError("detachment envelope angles must be non-zero")
        if required < 0.0:
            raise ValueError(f"detachment envelope torques must be non-negative, got {required}")
        required_with_margin = required + request_margin_nm

        if angle > 0.0 and lower_angle <= angle <= upper_angle:
            existing_torque = tightened_by_angle.get(
                angle,
                _interpolate_piecewise_linear_knots(knots_rad_nm, angle),
            )
            tightened_by_angle[angle] = min(existing_torque, -required_with_margin)

        if angle < 0.0 and lower_angle <= angle <= upper_angle:
            existing_torque = tightened_by_angle.get(
                angle,
                _interpolate_piecewise_linear_knots(knots_rad_nm, angle),
            )
            tightened_by_angle[angle] = max(existing_torque, required_with_margin)

    tightened_knots = tuple(sorted(tightened_by_angle.items()))
    if any(right[0] <= left[0] for left, right in zip(tightened_knots, tightened_knots[1:])):
        raise ValueError(
            "tightened torque-profile points must be strictly increasing in angle; "
            f"got {[point[0] for point in tightened_knots]} radians"
        )
    return tuple((float(angle), float(torque)) for angle, torque in tightened_knots)


def _target_formula_torque_at_angle_rad(
    original_knots_rad_nm: Sequence[tuple[float, float]],
    angle_rad: float,
) -> float:
    angle = float(angle_rad)
    if original_knots_rad_nm[0][0] <= angle <= original_knots_rad_nm[-1][0]:
        return _interpolate_piecewise_linear_knots(original_knots_rad_nm, angle)
    if angle < original_knots_rad_nm[0][0]:
        return float(original_knots_rad_nm[0][1])
    return float(original_knots_rad_nm[-1][1])


def _merge_formula_and_secondary_knots(
    formula_knots_rad_nm: Sequence[tuple[float, float]],
    secondary_knots_rad_nm: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged_by_angle = {}
    for angle, torque in formula_knots_rad_nm:
        angle = float(angle)
        if angle in merged_by_angle:
            raise ValueError(f"duplicate formula knot angle {angle}")
        merged_by_angle[angle] = float(torque)

    for angle, torque in secondary_knots_rad_nm:
        angle = float(angle)
        torque = float(torque)
        if angle not in merged_by_angle:
            merged_by_angle[angle] = torque
            continue
        if angle > 0.0:
            merged_by_angle[angle] = min(merged_by_angle[angle], torque)
        elif angle < 0.0:
            merged_by_angle[angle] = max(merged_by_angle[angle], torque)
        else:
            merged_by_angle[angle] = 0.0

    merged_knots = tuple(sorted(merged_by_angle.items()))
    if any(right[0] <= left[0] for left, right in zip(merged_knots, merged_knots[1:])):
        raise ValueError(
            "optimized formula-profile points must be strictly increasing in angle; "
            f"got {[point[0] for point in merged_knots]} radians"
        )
    return tuple((float(angle), float(torque)) for angle, torque in merged_knots)


def _constrained_formula_knot_angles_rad(
    original_knots_rad_nm: Sequence[tuple[float, float]],
    envelope_rad_nm: Sequence[tuple[float, float]],
    domain_extension_rad: float,
) -> tuple[float, ...]:
    if domain_extension_rad <= 0.0:
        raise ValueError(f"domain_extension_rad must be positive, got {domain_extension_rad}")
    angles = {float(angle) for angle, _ in original_knots_rad_nm}
    angles.add(0.0)
    if len(envelope_rad_nm) > 0:
        envelope_angles = [float(angle) for angle, _ in envelope_rad_nm]
        angles.update(envelope_angles)
        angles.add(min(envelope_angles) - domain_extension_rad)
        angles.add(max(envelope_angles) + domain_extension_rad)
    constrained_angles = tuple(sorted(angles))
    if any(right <= left for left, right in zip(constrained_angles, constrained_angles[1:])):
        raise ValueError(
            "constrained formula-profile points must be strictly increasing in angle; "
            f"got {constrained_angles} radians"
        )
    return constrained_angles


def _signed_torque_bounds_for_angle_rad(angle_rad: float) -> tuple[float, float]:
    angle = float(angle_rad)
    if angle < 0.0:
        return FORMULA_TORQUE_C_BOUNDS_NM[0], FORMULA_TORQUE_A_BOUNDS_NM[1]
    if angle > 0.0:
        return -FORMULA_TORQUE_A_BOUNDS_NM[1], -FORMULA_TORQUE_C_BOUNDS_NM[0]
    return 0.0, 0.0


def _build_shifted_formula_candidate(
    knots_rad_nm: Sequence[tuple[float, float]],
    original_knots_rad_nm: Sequence[tuple[float, float]],
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    actual_parameters: FormulaTorqueParameters,
    spring_catalog_path: str | Path,
    spring_id: str,
    n_grid: int,
    safety_factor: float,
) -> ConstrainedCandidateBuild:
    erc_knots_rad_nm = tuple((angle, -torque) for angle, torque in knots_rad_nm)
    knots = torch.tensor(erc_knots_rad_nm, dtype=torch.float64)
    profile = PiecewiseLinearTorqueProfile.from_xy(knots[:, 0], knots[:, 1])
    spring = SpringCatalog.from_yaml(spring_catalog_path)[spring_id]
    designer = ERCDesigner(
        spring,
        ERCDesignConfig(
            n_grid=n_grid,
            safety_factor=safety_factor,
            angle_convention="joint",
            joint_angle_limits_rad=(float(knots[0, 0]), float(knots[-1, 0])),
            energy_reference_angle_rad=0.0,
            invert_output_torque=True,
        ),
    )
    result = designer.design(profile)
    angle_offset_rad = zero_crossing_angle_rad(result.output_torque_table)
    shifted_output_torque_table = result.output_torque_table.clone()
    shifted_output_torque_table[:, 0] = shifted_output_torque_table[:, 0] - angle_offset_rad
    result = replace(result, output_torque_table=shifted_output_torque_table)
    evaluation = _evaluate_shifted_final_response(
        result.output_torque_table,
        original_knots_rad_nm,
        required_detachment_envelope_rad_nm,
        0.0,
    )
    return ConstrainedCandidateBuild(
        tuple((float(angle), float(torque)) for angle, torque in knots_rad_nm),
        result,
        angle_offset_rad,
        evaluation,
        actual_parameters,
    )


def _evaluate_shifted_final_response(
    shifted_output_torque_table: torch.Tensor,
    original_knots_rad_nm: Sequence[tuple[float, float]],
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    final_margin_nm: float,
) -> FinalResponseEvaluation:
    if final_margin_nm < 0.0:
        raise ValueError(f"final_margin_nm must be non-negative, got {final_margin_nm}")

    violations = []
    table_angles = shifted_output_torque_table[:, 0]
    if len(required_detachment_envelope_rad_nm) == 0:
        domain_margin = float("inf")
    else:
        envelope_angles_list = [float(angle) for angle, _ in required_detachment_envelope_rad_nm]
        envelope_required_list = [float(required) for _, required in required_detachment_envelope_rad_nm]
        if any(angle == 0.0 for angle in envelope_angles_list):
            raise ValueError("detachment envelope angles must be non-zero")
        if any(required < 0.0 for required in envelope_required_list):
            raise ValueError("detachment envelope torques must be non-negative")
        positive_side = [angle for angle in envelope_angles_list if angle > 0.0]
        negative_side = [angle for angle in envelope_angles_list if angle < 0.0]
        if len(positive_side) > 0 and len(negative_side) > 0:
            raise ValueError("detachment envelope must be one-sided")

        required_threshold = max(envelope_required_list) + final_margin_nm
        crossing_window_rad = min(abs(angle) for angle in envelope_angles_list)
        side = "positive" if len(positive_side) > 0 else "negative"
        if crossing_window_rad <= 0.0:
            raise ValueError("detachment crossing window must be positive")

        if side == "positive":
            window_start = 0.0
            window_end = crossing_window_rad
        else:
            window_start = -crossing_window_rad
            window_end = 0.0

        domain_margin = min(
            window_start - float(table_angles[0]),
            float(table_angles[-1]) - window_end,
        )
        if domain_margin < 0.0:
            raise ValueError(
                "detachment crossing window is outside the shifted final ERC table domain: "
                f"window=[{window_start:.6g}, {window_end:.6g}] rad "
                f"domain=[{float(table_angles[0]):.6g}, {float(table_angles[-1]):.6g}] rad"
            )

        interior_angles = table_angles[(table_angles > window_start) & (table_angles < window_end)]
        query_angles = torch.cat(
            (
                torch.tensor(
                    [window_start, window_end],
                    dtype=torch.float64,
                    device=shifted_output_torque_table.device,
                ),
                interior_angles,
            )
        )
        query_angles = torch.sort(query_angles).values
        query_torques = interpolate_output_torque_table(shifted_output_torque_table, query_angles)

        if side == "positive":
            best_index = int(torch.argmin(query_torques))
            best_actual = float(query_torques[best_index])
            violation_nm = best_actual - (-required_threshold)
        else:
            best_index = int(torch.argmax(query_torques))
            best_actual = float(query_torques[best_index])
            violation_nm = required_threshold - best_actual
        if violation_nm > 0.0:
            violations.append(
                DetachmentEnvelopeViolation(
                    side,
                    float(query_angles[best_index]),
                    best_actual,
                    required_threshold,
                    violation_nm,
                )
            )

    requested_torques = []
    comparison_torques = []
    for row in shifted_output_torque_table.detach().cpu():
        angle = float(row[0])
        if original_knots_rad_nm[0][0] <= angle <= original_knots_rad_nm[-1][0]:
            requested_torques.append(_interpolate_piecewise_linear_knots(original_knots_rad_nm, angle))
            comparison_torques.append(float(row[1]))
    if len(requested_torques) == 0:
        rmse_to_requested = float("inf")
    else:
        squared_errors = [
            (comparison - requested) ** 2
            for requested, comparison in zip(requested_torques, comparison_torques)
        ]
        rmse_to_requested = math.sqrt(sum(squared_errors) / len(squared_errors))

    max_violation = max(violations, key=lambda violation: violation.violation_nm) if violations else None
    max_violation_nm = 0.0 if max_violation is None else float(max_violation.violation_nm)
    return FinalResponseEvaluation(
        tuple(violations),
        max_violation_nm,
        max_violation,
        rmse_to_requested,
        domain_margin,
    )


def _format_constrained_failure(
    evaluation: FinalResponseEvaluation,
    side_name: str,
) -> str:
    violation = evaluation.max_violation
    if violation is None:
        return (
            "Constrained ERC torque projection failed without a remaining envelope violation "
            f"for {side_name}; current RMSE={evaluation.rmse_to_requested_nm:.6g} Nm"
        )
    required = (
        -violation.required_torque_nm
        if violation.side == "positive"
        else violation.required_torque_nm
    )
    return (
        "Constrained ERC torque projection could not satisfy the motor-detachment envelope "
        f"for {side_name}: largest_violation={violation.violation_nm:.6g} Nm "
        f"side={violation.side} angle={violation.angle_rad:.6g} rad "
        f"actual_torque={violation.actual_torque_nm:.6g} Nm "
        f"required_torque={required:.6g} Nm "
        f"current_RMSE={evaluation.rmse_to_requested_nm:.6g} Nm"
    )


def _candidate_with_projected_torques(
    knot_angles: Sequence[float],
    target_torques: Sequence[float],
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    detachment_request_margin_nm: float,
    projection_mode: str,
) -> tuple[tuple[float, float], ...]:
    if projection_mode not in {"target", "envelope_samples", "constant_required", "near_zero_required"}:
        raise ValueError(f"Unexpected constrained ERC projection mode: {projection_mode}")
    projected_torques = [float(torque) for torque in target_torques]
    if len(required_detachment_envelope_rad_nm) > 0:
        max_required_torque = max(float(required) for _, required in required_detachment_envelope_rad_nm)
    else:
        max_required_torque = 0.0
    envelope_by_angle = {
        float(angle): float(required) for angle, required in required_detachment_envelope_rad_nm
    }

    for index, angle in enumerate(knot_angles):
        angle = float(angle)
        lower_bound, upper_bound = _signed_torque_bounds_for_angle_rad(angle)
        target = projected_torques[index]
        if projection_mode == "target":
            projected = target
        elif projection_mode == "envelope_samples" and angle in envelope_by_angle:
            required = envelope_by_angle[angle] + detachment_request_margin_nm
            projected = min(target, -required) if angle > 0.0 else max(target, required)
        elif projection_mode == "constant_required" and angle != 0.0:
            required = max_required_torque + detachment_request_margin_nm
            projected = -required if angle > 0.0 else required
        elif projection_mode == "near_zero_required" and angle != 0.0:
            required = max_required_torque + detachment_request_margin_nm
            if abs(angle) <= max(math.radians(8.0), DETACHMENT_MIN_ANGLE_DEG * math.pi / 180.0):
                projected = -required if angle > 0.0 else required
            else:
                projected = target
        else:
            projected = target
        if angle == 0.0:
            projected = 0.0
        projected_torques[index] = min(upper_bound, max(lower_bound, projected))
    return tuple(
        (float(angle), float(torque)) for angle, torque in zip(knot_angles, projected_torques)
    )


def _secondary_crossing_knots_for_formula_profile(
    original_knots_rad_nm: Sequence[tuple[float, float]],
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    detachment_request_margin_nm: float,
    secondary_mode: str,
) -> tuple[tuple[float, float], ...]:
    if secondary_mode not in {"target", "near_threshold", "left_sparse_threshold"}:
        raise ValueError(f"Unexpected constrained ERC secondary mode: {secondary_mode}")
    if detachment_request_margin_nm < 0.0:
        raise ValueError(
            f"detachment_request_margin_nm must be non-negative, got {detachment_request_margin_nm}"
        )
    if len(required_detachment_envelope_rad_nm) == 0 or secondary_mode == "target":
        return tuple((float(angle), float(torque)) for angle, torque in original_knots_rad_nm)

    envelope_angles = [float(angle) for angle, _ in required_detachment_envelope_rad_nm]
    envelope_required = [float(required) for _, required in required_detachment_envelope_rad_nm]
    if any(angle == 0.0 for angle in envelope_angles):
        raise ValueError("detachment envelope angles must be non-zero")
    if any(required < 0.0 for required in envelope_required):
        raise ValueError("detachment envelope torques must be non-negative")
    positive_side = [angle for angle in envelope_angles if angle > 0.0]
    negative_side = [angle for angle in envelope_angles if angle < 0.0]
    if len(positive_side) > 0 and len(negative_side) > 0:
        raise ValueError("detachment envelope must be one-sided")

    bend_sign = 1.0 if len(positive_side) > 0 else -1.0
    max_abs_angle = max(abs(angle) for angle in envelope_angles)
    high_theta_cutoff = CONSTRAINED_SECONDARY_HIGH_THETA_FRACTION * max_abs_angle
    requested_threshold = max(envelope_required) + detachment_request_margin_nm
    knots_by_angle = {float(angle): float(torque) for angle, torque in original_knots_rad_nm}
    knots_by_angle[0.0] = 0.0

    for index, (angle, required_torque) in enumerate(required_detachment_envelope_rad_nm):
        angle = float(angle)
        if abs(angle) > high_theta_cutoff:
            continue
        required = float(required_torque) + detachment_request_margin_nm
        if secondary_mode == "near_threshold" and index == 0:
            requested_magnitude = min(
                FORMULA_TORQUE_A_BOUNDS_NM[1],
                CONSTRAINED_NEAR_ZERO_GAIN * requested_threshold,
            )
        elif secondary_mode == "left_sparse_threshold" and (
            index == 0 or index % CONSTRAINED_SECONDARY_STRIDE == 0
        ):
            abs_deg = abs(math.degrees(angle))
            local_gain = 1.0 + CONSTRAINED_LEFT_NEAR_GAIN * math.exp(
                -((abs_deg - DETACHMENT_MIN_ANGLE_DEG) / CONSTRAINED_LEFT_NEAR_WIDTH_DEG) ** 2
            )
            requested_magnitude = min(
                FORMULA_TORQUE_A_BOUNDS_NM[1],
                local_gain * required,
            )
        else:
            continue

        knots_by_angle[angle] = -requested_magnitude if bend_sign > 0.0 else requested_magnitude

    knots = tuple(sorted(knots_by_angle.items()))
    if any(right[0] <= left[0] for left, right in zip(knots, knots[1:])):
        raise ValueError(
            "secondary torque-profile points must be strictly increasing in angle; "
            f"got {[point[0] for point in knots]} radians"
        )
    return tuple((float(angle), float(torque)) for angle, torque in knots)


def _normalized_value_to_bounds(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(lower + float(value) * (upper - lower))


def _actual_value_to_normalized(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if upper <= lower:
        raise ValueError(f"bounds must be increasing, got {bounds}")
    return float((float(value) - lower) / (upper - lower))


def _crossing_window_from_envelope(
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    final_margin_nm: float,
) -> ConstrainedCrossingWindow | None:
    if final_margin_nm < 0.0:
        raise ValueError(f"final_margin_nm must be non-negative, got {final_margin_nm}")
    if len(required_detachment_envelope_rad_nm) == 0:
        return None

    envelope_angles = [float(angle) for angle, _ in required_detachment_envelope_rad_nm]
    envelope_required = [float(required) for _, required in required_detachment_envelope_rad_nm]
    if any(angle == 0.0 for angle in envelope_angles):
        raise ValueError("detachment envelope angles must be non-zero")
    if any(required < 0.0 for required in envelope_required):
        raise ValueError("detachment envelope torques must be non-negative")
    positive_side = [angle for angle in envelope_angles if angle > 0.0]
    negative_side = [angle for angle in envelope_angles if angle < 0.0]
    if len(positive_side) > 0 and len(negative_side) > 0:
        raise ValueError("detachment envelope must be one-sided")

    crossing_window_rad = min(abs(angle) for angle in envelope_angles)
    if crossing_window_rad <= 0.0:
        raise ValueError("detachment crossing window must be positive")
    side = "positive" if len(positive_side) > 0 else "negative"
    required_threshold = max(envelope_required) + final_margin_nm
    if side == "positive":
        return ConstrainedCrossingWindow(side, 0.0, crossing_window_rad, required_threshold)
    return ConstrainedCrossingWindow(side, -crossing_window_rad, 0.0, required_threshold)


def _bend_sign_from_envelope_or_mirror(
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    mirror: bool,
) -> float:
    crossing_window = _crossing_window_from_envelope(required_detachment_envelope_rad_nm, 0.0)
    if crossing_window is not None:
        return 1.0 if crossing_window.side == "positive" else -1.0
    return -1.0 if mirror else 1.0


def _decode_constrained_formula_cma_vector(
    normalized_vector: Sequence[float],
    original_parameters: FormulaTorqueParameters,
    mirror: bool,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
) -> ConstrainedDecodedProfile:
    values = [float(value) for value in normalized_vector]
    if len(values) != CONSTRAINED_CMA_VECTOR_LENGTH:
        raise ValueError(
            f"expected {CONSTRAINED_CMA_VECTOR_LENGTH} CMA-ES values, got {len(values)}"
        )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"CMA-ES normalized values must stay in [0, 1], got {values}")

    parameters = FormulaTorqueParameters(
        _normalized_value_to_bounds(values[0], FORMULA_TORQUE_A_BOUNDS_NM),
        _normalized_value_to_bounds(values[1], FORMULA_TORQUE_B_BOUNDS),
        _normalized_value_to_bounds(values[2], FORMULA_TORQUE_C_BOUNDS_NM),
        _normalized_value_to_bounds(values[3], FORMULA_TORQUE_D_BOUNDS_NM),
        float(original_parameters.end_angle_rad),
    )
    formula_knots = formula_torque_profile_knots_rad_nm(parameters, mirror)

    bend_sign = _bend_sign_from_envelope_or_mirror(required_detachment_envelope_rad_nm, mirror)
    secondary_angle_limit = (
        CONSTRAINED_SECONDARY_HIGH_THETA_FRACTION * float(original_parameters.end_angle_rad)
    )
    if secondary_angle_limit <= 0.0:
        raise ValueError(
            f"secondary angle limit must be positive, got {secondary_angle_limit}"
        )

    angle_values = sorted(values[4:7])
    torque_values = values[7:10]
    secondary_knots = []
    for angle_value, torque_value in zip(angle_values, torque_values):
        abs_angle = float(angle_value) * secondary_angle_limit
        if abs_angle <= CONSTRAINED_CMA_SECONDARY_MIN_ANGLE_RAD:
            raise ValueError(f"secondary knot angle must be positive, got {abs_angle}")
        if abs_angle > secondary_angle_limit:
            raise ValueError(
                f"secondary knot angle {abs_angle} exceeds limit {secondary_angle_limit}"
            )
        signed_angle = bend_sign * abs_angle
        magnitude = _normalized_value_to_bounds(
            torque_value,
            (0.0, FORMULA_TORQUE_A_BOUNDS_NM[1]),
        )
        signed_torque = -magnitude if bend_sign > 0.0 else magnitude
        secondary_knots.append((signed_angle, signed_torque))

    merged_knots = _merge_formula_and_secondary_knots(
        formula_knots,
        secondary_knots,
    )

    return ConstrainedDecodedProfile(
        parameters,
        tuple((float(angle), float(torque)) for angle, torque in formula_knots),
        tuple((float(angle), float(torque)) for angle, torque in sorted(secondary_knots)),
        tuple((float(angle), float(torque)) for angle, torque in merged_knots),
    )


def _seed_vector_for_constrained_formula_cma(
    parameters: FormulaTorqueParameters,
    mirror: bool,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    detachment_request_margin_nm: float,
    profile_mode: str,
) -> np.ndarray:
    if profile_mode not in {"formula", "near_crossing", "spread_crossing"}:
        raise ValueError(f"Unexpected constrained ERC CMA seed mode: {profile_mode}")
    if detachment_request_margin_nm < 0.0:
        raise ValueError(
            f"detachment_request_margin_nm must be non-negative, got {detachment_request_margin_nm}"
        )
    values = np.zeros(CONSTRAINED_CMA_VECTOR_LENGTH, dtype=np.float64)
    values[0] = _actual_value_to_normalized(parameters.a, FORMULA_TORQUE_A_BOUNDS_NM)
    values[1] = _actual_value_to_normalized(parameters.b, FORMULA_TORQUE_B_BOUNDS)
    values[2] = _actual_value_to_normalized(parameters.c, FORMULA_TORQUE_C_BOUNDS_NM)
    values[3] = _actual_value_to_normalized(parameters.d, FORMULA_TORQUE_D_BOUNDS_NM)

    secondary_angle_limit = CONSTRAINED_SECONDARY_HIGH_THETA_FRACTION * float(parameters.end_angle_rad)
    if secondary_angle_limit <= 0.0:
        raise ValueError(
            f"secondary angle limit must be positive, got {secondary_angle_limit}"
        )
    crossing_window = _crossing_window_from_envelope(required_detachment_envelope_rad_nm, 0.0)
    if crossing_window is None:
        near_angle = min(math.radians(DETACHMENT_MIN_ANGLE_DEG), 0.25 * secondary_angle_limit)
        required_torque = 0.0
    else:
        near_angle = min(abs(crossing_window.end_rad - crossing_window.start_rad), secondary_angle_limit)
        required_torque = crossing_window.required_torque_nm + detachment_request_margin_nm
    near_angle = max(CONSTRAINED_CMA_SECONDARY_MIN_ANGLE_RAD, near_angle)

    if profile_mode == "formula":
        angle_fractions = (0.20, 0.50, 0.80)
        torque_magnitudes = None
    elif profile_mode == "near_crossing":
        angle_fractions = (
            near_angle / secondary_angle_limit,
            min(0.50, max(near_angle / secondary_angle_limit, 0.25)),
            0.80,
        )
        near_magnitude = min(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            CONSTRAINED_NEAR_ZERO_GAIN * required_torque,
        )
        torque_magnitudes = (near_magnitude, None, None)
    else:
        angle_fractions = (
            near_angle / secondary_angle_limit,
            min(0.45, max(near_angle / secondary_angle_limit, 0.30)),
            0.80,
        )
        near_magnitude = min(FORMULA_TORQUE_A_BOUNDS_NM[1], 2.0 * required_torque)
        mid_magnitude = min(FORMULA_TORQUE_A_BOUNDS_NM[1], required_torque)
        torque_magnitudes = (near_magnitude, mid_magnitude, 0.5 * mid_magnitude)

    sorted_angle_fractions = sorted(
        min(1.0, max(CONSTRAINED_CMA_SECONDARY_MIN_ANGLE_RAD / secondary_angle_limit, value))
        for value in angle_fractions
    )
    values[4:7] = np.asarray(sorted_angle_fractions, dtype=np.float64)
    bend_sign = _bend_sign_from_envelope_or_mirror(required_detachment_envelope_rad_nm, mirror)
    formula_knots = formula_torque_profile_knots_rad_nm(parameters, mirror)
    inactive_magnitudes = []
    for angle_fraction in sorted_angle_fractions:
        signed_angle = bend_sign * angle_fraction * secondary_angle_limit
        formula_torque = _target_formula_torque_at_angle_rad(formula_knots, signed_angle)
        inactive_magnitudes.append(
            min(FORMULA_TORQUE_A_BOUNDS_NM[1], max(0.0, -bend_sign * formula_torque))
        )
    if torque_magnitudes is None:
        resolved_torque_magnitudes = inactive_magnitudes
    else:
        resolved_torque_magnitudes = [
            inactive_magnitude if magnitude is None else magnitude
            for magnitude, inactive_magnitude in zip(torque_magnitudes, inactive_magnitudes)
        ]
    values[7:10] = np.asarray(
        [magnitude / FORMULA_TORQUE_A_BOUNDS_NM[1] for magnitude in resolved_torque_magnitudes],
        dtype=np.float64,
    )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"constructed CMA-ES seed outside [0, 1]: {values.tolist()}")
    return values


def _clamped_formula_seed_parameters(
    a: float,
    b: float,
    c: float,
    d: float,
    end_angle_rad: float,
) -> FormulaTorqueParameters:
    return FormulaTorqueParameters(
        min(FORMULA_TORQUE_A_BOUNDS_NM[1], max(FORMULA_TORQUE_A_BOUNDS_NM[0], float(a))),
        min(FORMULA_TORQUE_B_BOUNDS[1], max(FORMULA_TORQUE_B_BOUNDS[0], float(b))),
        min(FORMULA_TORQUE_C_BOUNDS_NM[1], max(FORMULA_TORQUE_C_BOUNDS_NM[0], float(c))),
        min(FORMULA_TORQUE_D_BOUNDS_NM[1], max(FORMULA_TORQUE_D_BOUNDS_NM[0], float(d))),
        float(end_angle_rad),
    )


def _interpolated_formula_seed_parameters(
    start: FormulaTorqueParameters,
    end: FormulaTorqueParameters,
    alpha: float,
) -> FormulaTorqueParameters:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return _clamped_formula_seed_parameters(
        start.a + alpha * (end.a - start.a),
        start.b + alpha * (end.b - start.b),
        start.c + alpha * (end.c - start.c),
        start.d + alpha * (end.d - start.d),
        start.end_angle_rad,
    )


def _feasible_oriented_formula_seed_parameters(
    parameters: FormulaTorqueParameters,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    detachment_request_margin_nm: float,
) -> tuple[FormulaTorqueParameters, ...]:
    if detachment_request_margin_nm < 0.0:
        raise ValueError(
            f"detachment_request_margin_nm must be non-negative, got {detachment_request_margin_nm}"
        )
    if len(required_detachment_envelope_rad_nm) == 0:
        return (parameters,)

    required_torque = max(float(required) for _, required in required_detachment_envelope_rad_nm)
    required_threshold = required_torque + detachment_request_margin_nm

    prototypes = [
        _clamped_formula_seed_parameters(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            max(parameters.b, 0.2),
            max(parameters.c, required_threshold),
            max(parameters.d, required_threshold),
            parameters.end_angle_rad,
        ),
        _clamped_formula_seed_parameters(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            max(parameters.b, 0.5),
            max(parameters.c, required_threshold),
            max(parameters.d, 0.5 * required_threshold),
            parameters.end_angle_rad,
        ),
        _clamped_formula_seed_parameters(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            max(parameters.b, 0.5),
            max(parameters.c, 0.5 * required_threshold),
            max(parameters.d, required_threshold),
            parameters.end_angle_rad,
        ),
        _clamped_formula_seed_parameters(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            max(parameters.b, 0.8),
            max(parameters.c, required_threshold),
            max(parameters.d, required_threshold),
            parameters.end_angle_rad,
        ),
        _clamped_formula_seed_parameters(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            FORMULA_TORQUE_B_BOUNDS[1],
            FORMULA_TORQUE_C_BOUNDS_NM[1],
            FORMULA_TORQUE_D_BOUNDS_NM[1],
            parameters.end_angle_rad,
        ),
        _clamped_formula_seed_parameters(
            FORMULA_TORQUE_A_BOUNDS_NM[1],
            max(parameters.b, 0.5),
            FORMULA_TORQUE_C_BOUNDS_NM[1],
            FORMULA_TORQUE_D_BOUNDS_NM[1],
            parameters.end_angle_rad,
        ),
    ]

    seed_parameter_sets = [parameters]
    for prototype in prototypes:
        seed_parameter_sets.append(prototype)
        for alpha in (0.35, 0.7):
            seed_parameter_sets.append(
                _interpolated_formula_seed_parameters(
                    parameters,
                    prototype,
                    alpha,
                )
            )

    deduplicated = []
    seen = set()
    for candidate in seed_parameter_sets:
        key = (
            round(candidate.a, 12),
            round(candidate.b, 12),
            round(candidate.c, 12),
            round(candidate.d, 12),
            round(candidate.end_angle_rad, 12),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return tuple(deduplicated)


def _zero_torque_at_shifted_origin_nm(shifted_output_torque_table: torch.Tensor) -> float:
    query = torch.tensor([0.0], dtype=torch.float64, device=shifted_output_torque_table.device)
    return float(interpolate_output_torque_table(shifted_output_torque_table, query)[0])


def _diagnostic_from_evaluation(
    evaluation: FinalResponseEvaluation,
    parameters: FormulaTorqueParameters,
    side_name: str,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    reason: str,
    score: float,
) -> ConstrainedCandidateDiagnostic:
    crossing_window = _crossing_window_from_envelope(required_detachment_envelope_rad_nm, 0.0)
    if evaluation.max_violation is None:
        best_actual_torque = None
        required_torque = None
    else:
        best_actual_torque = float(evaluation.max_violation.actual_torque_nm)
        required_torque = (
            -float(evaluation.max_violation.required_torque_nm)
            if evaluation.max_violation.side == "positive"
            else float(evaluation.max_violation.required_torque_nm)
        )
    return ConstrainedCandidateDiagnostic(
        score,
        reason,
        side_name,
        crossing_window,
        best_actual_torque,
        required_torque,
        evaluation.rmse_to_requested_nm,
        parameters,
    )


def _diagnostic_from_failure(
    reason: str,
    side_name: str,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    parameters: FormulaTorqueParameters | None,
    score: float,
) -> ConstrainedCandidateDiagnostic:
    crossing_window = _crossing_window_from_envelope(required_detachment_envelope_rad_nm, 0.0)
    return ConstrainedCandidateDiagnostic(
        score,
        reason,
        side_name,
        crossing_window,
        None,
        None if crossing_window is None else crossing_window.required_torque_nm,
        float("inf"),
        parameters,
    )


def _score_constrained_formula_cma_vector(
    normalized_vector: Sequence[float],
    original_parameters: FormulaTorqueParameters,
    mirror: bool,
    original_knots_rad_nm: Sequence[tuple[float, float]],
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    spring_catalog_path: str | Path,
    spring_id: str,
    n_grid: int,
    safety_factor: float,
    side_name: str,
) -> tuple[float, ConstrainedCandidateBuild | None, ConstrainedCandidateDiagnostic]:
    decoded_parameters = None
    try:
        decoded = _decode_constrained_formula_cma_vector(
            normalized_vector,
            original_parameters,
            mirror,
            required_detachment_envelope_rad_nm,
        )
        decoded_parameters = decoded.parameters
        candidate = _build_shifted_formula_candidate(
            decoded.merged_knots_rad_nm,
            original_knots_rad_nm,
            required_detachment_envelope_rad_nm,
            decoded.parameters,
            spring_catalog_path,
            spring_id,
            n_grid,
            safety_factor,
        )
        zero_torque = _zero_torque_at_shifted_origin_nm(candidate.design_result.output_torque_table)
        if abs(zero_torque) > CONSTRAINED_CMA_ZERO_TOLERANCE_NM:
            score = (
                CONSTRAINED_CMA_INFEASIBLE_BASE_PENALTY
                + 1000.0 * abs(zero_torque)
                + candidate.evaluation.rmse_to_requested_nm
            )
            diagnostic = _diagnostic_from_evaluation(
                candidate.evaluation,
                decoded.parameters,
                side_name,
                required_detachment_envelope_rad_nm,
                f"zero_torque_at_origin={zero_torque:.6g} Nm",
                score,
            )
            return score, None, diagnostic
        if candidate.evaluation.max_violation_nm <= DETACHMENT_ENVELOPE_TOLERANCE_NM:
            score = candidate.evaluation.rmse_to_requested_nm
            diagnostic = _diagnostic_from_evaluation(
                candidate.evaluation,
                decoded.parameters,
                side_name,
                required_detachment_envelope_rad_nm,
                "feasible",
                score,
            )
            return score, candidate, diagnostic
        score = (
            CONSTRAINED_CMA_INFEASIBLE_BASE_PENALTY
            + 1000.0 * candidate.evaluation.max_violation_nm
            + candidate.evaluation.rmse_to_requested_nm
        )
        diagnostic = _diagnostic_from_evaluation(
            candidate.evaluation,
            decoded.parameters,
            side_name,
            required_detachment_envelope_rad_nm,
            "crossing_window_violation",
            score,
        )
        return score, None, diagnostic
    except ValueError as exc:
        diagnostic = _diagnostic_from_failure(
            str(exc),
            side_name,
            required_detachment_envelope_rad_nm,
            decoded_parameters,
            CONSTRAINED_CMA_BUILD_FAILURE_PENALTY,
        )
        return diagnostic.score, None, diagnostic


def _format_constrained_cma_failure(diagnostic: ConstrainedCandidateDiagnostic) -> str:
    if diagnostic.crossing_window is None:
        crossing_window = "none"
        crossing_side = "none"
    else:
        crossing_side = diagnostic.crossing_window.side
        crossing_window = (
            f"[{diagnostic.crossing_window.start_rad:.6g}, "
            f"{diagnostic.crossing_window.end_rad:.6g}] rad"
        )
    parameters = diagnostic.parameters
    if parameters is None:
        parameter_text = "best a,b,c,d=unavailable"
    else:
        parameter_text = (
            "best a,b,c,d="
            f"({parameters.a:.6g}, {parameters.b:.6g}, {parameters.c:.6g}, {parameters.d:.6g})"
        )
    best_actual = (
        "unavailable"
        if diagnostic.best_actual_torque_nm is None
        else f"{diagnostic.best_actual_torque_nm:.6g} Nm"
    )
    required = (
        "unavailable"
        if diagnostic.required_torque_nm is None
        else f"{diagnostic.required_torque_nm:.6g} Nm"
    )
    return (
        "Constrained ERC CMA-ES repair could not satisfy the motor-detachment envelope "
        f"for {diagnostic.side_name}: violation reason={diagnostic.reason} "
        f"side={crossing_side} crossing window={crossing_window} "
        f"best actual torque={best_actual} required torque={required} "
        f"RMSE={diagnostic.rmse_to_requested_nm:.6g} Nm {parameter_text}"
    )


def _constrained_formula_erc_build(
    parameters: FormulaTorqueParameters,
    mirror: bool,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    detachment_request_margin_nm: float,
    spring_catalog_path: str | Path,
    spring_id: str,
    n_grid: int,
    safety_factor: float,
    max_torque_rmse_nm: float,
    side_name: str,
) -> ConstrainedCandidateBuild:
    if detachment_request_margin_nm < 0.0:
        raise ValueError(
            f"detachment_request_margin_nm must be non-negative, got {detachment_request_margin_nm}"
        )
    original_knots_rad_nm = formula_torque_profile_knots_rad_nm(parameters, mirror)
    if max_torque_rmse_nm < 0.0:
        raise ValueError(f"max_torque_rmse_nm must be non-negative, got {max_torque_rmse_nm}")

    seed_parameter_sets = _feasible_oriented_formula_seed_parameters(
        parameters,
        required_detachment_envelope_rad_nm,
        detachment_request_margin_nm,
    )

    seed_vectors = []
    for seed_parameters in seed_parameter_sets:
        for seed_mode in ("formula", "near_crossing", "spread_crossing"):
            seed_vectors.append(
                _seed_vector_for_constrained_formula_cma(
                    seed_parameters,
                    mirror,
                    required_detachment_envelope_rad_nm,
                    detachment_request_margin_nm,
                    seed_mode,
                )
            )

    deadline = time.monotonic() + CONSTRAINED_CMA_BUDGET_S
    best_feasible_candidate = None
    best_diagnostic = None
    best_seed_vector = np.asarray(seed_vectors[0], dtype=np.float64)
    best_seed_score = float("inf")
    n_evaluations = 0

    for seed_vector in seed_vectors:
        if time.monotonic() >= deadline:
            break
        score, candidate, diagnostic = _score_constrained_formula_cma_vector(
            seed_vector,
            parameters,
            mirror,
            original_knots_rad_nm,
            required_detachment_envelope_rad_nm,
            spring_catalog_path,
            spring_id,
            n_grid,
            safety_factor,
            side_name,
        )
        n_evaluations += 1
        if score < best_seed_score:
            best_seed_score = score
            best_seed_vector = np.asarray(seed_vector, dtype=np.float64)
        if candidate is not None and (
            best_feasible_candidate is None
            or score < best_feasible_candidate.evaluation.rmse_to_requested_nm
        ):
            best_feasible_candidate = candidate
        if best_diagnostic is None or diagnostic.score < best_diagnostic.score:
            best_diagnostic = diagnostic

    optimizer = cma.CMAEvolutionStrategy(
        best_seed_vector,
        CONSTRAINED_CMA_SIGMA0,
        {
            "bounds": [0.0, 1.0],
            "popsize": CONSTRAINED_CMA_POPSIZE,
            "seed": CONSTRAINED_CMA_SEED + (1 if mirror else 0),
            "verbose": -1,
            "maxfevals": CONSTRAINED_CMA_MAX_EVALUATIONS,
            "tolfun": 1e-12,
            "tolx": 1e-10,
        },
    )

    while (
        time.monotonic() < deadline
        and n_evaluations < CONSTRAINED_CMA_MAX_EVALUATIONS
        and not optimizer.stop()
    ):
        solutions = optimizer.ask()
        scores = []
        for solution in solutions:
            if time.monotonic() >= deadline or n_evaluations >= CONSTRAINED_CMA_MAX_EVALUATIONS:
                diagnostic = _diagnostic_from_failure(
                    "CMA-ES wall-clock budget reached before candidate evaluation",
                    side_name,
                    required_detachment_envelope_rad_nm,
                    None,
                    CONSTRAINED_CMA_BUILD_FAILURE_PENALTY,
                )
                scores.append(diagnostic.score)
                if best_diagnostic is None or diagnostic.score < best_diagnostic.score:
                    best_diagnostic = diagnostic
                continue
            score, candidate, diagnostic = _score_constrained_formula_cma_vector(
                solution,
                parameters,
                mirror,
                original_knots_rad_nm,
                required_detachment_envelope_rad_nm,
                spring_catalog_path,
                spring_id,
                n_grid,
                safety_factor,
                side_name,
            )
            n_evaluations += 1
            scores.append(score)
            if candidate is not None and (
                best_feasible_candidate is None
                or score < best_feasible_candidate.evaluation.rmse_to_requested_nm
            ):
                best_feasible_candidate = candidate
            if best_diagnostic is None or diagnostic.score < best_diagnostic.score:
                best_diagnostic = diagnostic
        optimizer.tell(solutions, scores)

    if best_feasible_candidate is not None:
        return best_feasible_candidate
    if best_diagnostic is None:
        best_diagnostic = _diagnostic_from_failure(
            "CMA-ES produced no evaluable candidates",
            side_name,
            required_detachment_envelope_rad_nm,
            None,
            CONSTRAINED_CMA_BUILD_FAILURE_PENALTY,
        )
    raise ValueError(_format_constrained_cma_failure(best_diagnostic))


def build_formula_erc_torque_function(
    parameters: FormulaTorqueParameters,
    fn_name: str,
    mirror: bool,
    required_detachment_envelope_rad_nm: Sequence[tuple[float, float]],
    detachment_request_margin_nm: float,
    spring_catalog_path: str | Path,
    spring_id: str,
    n_grid: int,
    safety_factor: float,
    max_torque_rmse_nm: float,
) -> ERCTorqueFunctionBuild:
    """Build a repaired ERC torque response from the formula-defined knots.

    The formula angles are radians because the simulator calls torque
    functions with joint angles in radians.
    """

    candidate = _constrained_formula_erc_build(
        parameters,
        mirror,
        required_detachment_envelope_rad_nm,
        detachment_request_margin_nm,
        spring_catalog_path,
        spring_id,
        n_grid,
        safety_factor,
        max_torque_rmse_nm,
        fn_name,
    )
    result = candidate.design_result
    function_string = torque_table_to_function_string(
        result.output_torque_table,
        fn_name,
        "flat",
        (0.4, -0.4),
    )
    namespace = {"torch": torch}
    exec(function_string, namespace)
    final_evaluation = _evaluate_shifted_final_response(
        result.output_torque_table,
        formula_torque_profile_knots_rad_nm(parameters, mirror),
        required_detachment_envelope_rad_nm,
        0.0,
    )
    if final_evaluation.max_violation_nm > DETACHMENT_ENVELOPE_TOLERANCE_NM:
        raise ValueError(_format_constrained_failure(final_evaluation, fn_name))
    knots_deg_nm = tuple(
        (float(angle * 180.0 / math.pi), float(torque)) for angle, torque in candidate.knots_rad_nm
    )
    return ERCTorqueFunctionBuild(
        torque_function=namespace[fn_name],
        function_string=function_string,
        design_result=result,
        requested_knots_deg_nm=knots_deg_nm,
        actual_parameters=(
            float(candidate.actual_parameters.a),
            float(candidate.actual_parameters.b),
            float(candidate.actual_parameters.c),
            float(candidate.actual_parameters.d),
            float(candidate.actual_parameters.end_angle_rad),
        ),
        angle_offset_rad=candidate.angle_offset_rad,
    )


def build_example_erc_torque_functions(
    left_segments: Sequence[tuple[float, float]],
    right_segments: Sequence[tuple[float, float]],
    *,
    n_grid: int = 151,
    spring_id: str = DEFAULT_SPRING_ID,
):
    """Build ERC torque functions from user-supplied profile segments.

    Args:
        left_segments: ``(joint_angle_deg, torque_Nm)`` knots for the left joint.
        right_segments: ``(joint_angle_deg, torque_Nm)`` knots for the right joint.

    Returns:
        Tuple of ``(torque_fn_left_str, torque_fn_right_str, left_result, right_result)``.
    """
    left_result = build_erc_design(left_segments, n_grid=n_grid, spring_id=spring_id)
    right_result = build_erc_design(right_segments, n_grid=n_grid, spring_id=spring_id)
    return (
        torque_table_to_function_string(left_result.output_torque_table, "torque_fn_left", "raise", None),
        torque_table_to_function_string(right_result.output_torque_table, "torque_fn_right", "raise", None),
        left_result,
        right_result,
    )
