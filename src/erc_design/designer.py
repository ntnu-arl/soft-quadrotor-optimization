from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import torch

from .profiles import PiecewiseLinearTorqueProfile, interpolate_pwl
from .springs import Spring


AngleConvention = Literal["joint", "cam"]


@dataclass(frozen=True)
class ERCDesignConfig:
    """Numerical and convention settings for ERC profile synthesis."""

    n_grid: int = 1000
    safety_factor: float = 1.1
    convexity_eps_m: float = 1e-3
    energy_scale_eps: float = 1e-6
    angle_convention: AngleConvention = "joint"
    joint_angle_limits_rad: tuple[float, float] | None = (-math.pi / 2.0, math.pi / 2.0)
    angle_limit_tolerance_rad: float = 1e-7
    energy_reference_angle_rad: float = 0.0
    invert_output_torque: bool = True
    dtype: torch.dtype = torch.float64
    device: str | torch.device | None = None

    @property
    def external_to_support_angle_scale(self) -> float:
        if self.angle_convention == "joint":
            return 0.5
        if self.angle_convention == "cam":
            return 1.0
        raise ValueError("angle_convention must be 'joint' or 'cam'")

    @property
    def joint_to_profile_angle_scale(self) -> float:
        if self.angle_convention == "joint":
            return 1.0
        if self.angle_convention == "cam":
            return 0.5
        raise ValueError("angle_convention must be 'joint' or 'cam'")


@dataclass(frozen=True)
class ERCDesignResult:
    spring: Spring
    input_profile: PiecewiseLinearTorqueProfile
    support_angles_rad: torch.Tensor
    target_torque_nm: torch.Tensor
    scaled_torque_nm: torch.Tensor
    repaired_torque_nm: torch.Tensor
    output_torque_table: torch.Tensor
    support_radius_m: torch.Tensor
    repaired_support_radius_m: torch.Tensor
    curvature_radius_m: torch.Tensor
    repaired_curvature_radius_m: torch.Tensor
    cam_xy_m: torch.Tensor
    repaired_cam_xy_m: torch.Tensor
    alpha: float
    torque_rmse_nm: float
    was_scaled: bool
    was_repaired: bool

    def export_xyz(
        self,
        path: str | Path,
        *,
        repaired: bool = True,
        n_points: int | None = None,
    ) -> None:
        export_curve_xyz(
            path,
            self.repaired_cam_xy_m if repaired else self.cam_xy_m,
            n_points=n_points,
        )

    def export_sldcrv(
        self,
        path: str | Path,
        *,
        repaired: bool = True,
        n_points: int | None = None,
    ) -> None:
        self.export_xyz(path, repaired=repaired, n_points=n_points)


class ERCDesigner:
    """Support-function ERC designer ported from ``ERC_model.mlx``."""

    def __init__(self, spring: Spring, config: ERCDesignConfig | None = None):
        self.spring = spring
        self.config = config or ERCDesignConfig()

    def design(
        self,
        profile: PiecewiseLinearTorqueProfile,
        *,
        repair: bool = True,
    ) -> ERCDesignResult:
        cfg = self.config
        device = torch.device(cfg.device) if cfg.device is not None else profile.knots.device
        dtype = cfg.dtype
        knots = profile.knots.to(dtype=dtype, device=device)
        profile = PiecewiseLinearTorqueProfile(knots)
        angle_scale = cfg.external_to_support_angle_scale
        if cfg.joint_angle_limits_rad is not None:
            joint_lo, joint_hi = cfg.joint_angle_limits_rad
            limit_scale = cfg.joint_to_profile_angle_scale
            lo = joint_lo * limit_scale
            hi = joint_hi * limit_scale
            if lo >= hi:
                raise ValueError("joint_angle_limits_rad must be ordered as (min, max)")
            tol = cfg.angle_limit_tolerance_rad
            if torch.any(profile.angles < lo - tol) or torch.any(profile.angles > hi + tol):
                raise ValueError(
                    "profile angles must stay inside "
                    f"[{lo:.6g}, {hi:.6g}] rad for angle_convention={cfg.angle_convention!r}"
                )

        support_knots = profile.angles * angle_scale
        x = torch.linspace(
            support_knots[0],
            support_knots[-1],
            cfg.n_grid,
            dtype=dtype,
            device=device,
        )
        hx = x[1] - x[0]
        torque = interpolate_pwl(x, support_knots, profile.torques)

        energy = cumulative_trapezoid(torque, x)
        ref_support_angle = torch.as_tensor(
            cfg.energy_reference_angle_rad * angle_scale,
            dtype=dtype,
            device=device,
        )
        if ref_support_angle < x[0] or ref_support_angle > x[-1]:
            raise ValueError("energy_reference_angle_rad must lie inside the profile domain")
        energy = energy - interpolate_pwl(ref_support_angle.reshape(1), x, energy)[0]

        alpha, initial_energy = self._energy_scaling(energy)
        scaled_torque = torque * alpha
        scaled_energy = energy * alpha

        extension = self._extension_from_energy(initial_energy + scaled_energy)
        radius = 0.5 * (
            torch.as_tensor(self.spring.min_length_m, dtype=dtype, device=device) + extension
        )
        tension = (
            torch.as_tensor(self.spring.total_tension_at_rest_n, dtype=dtype, device=device)
            + torch.as_tensor(self.spring.total_stiffness_n_per_m, dtype=dtype, device=device)
            * extension
        )
        if torch.any(tension <= 0):
            raise ValueError("spring tension became non-positive; increase preload or energy margin")

        dtorque = finite_difference_first(scaled_torque, hx)
        dr = scaled_torque / (2.0 * tension)
        ddr = (
            dtorque / (2.0 * tension)
            - self.spring.total_stiffness_n_per_m
            * scaled_torque**2
            / (2.0 * tension**3)
        )
        curvature = radius + ddr
        cam_xy = support_to_xy(x, radius, dr)

        if repair and bool(torch.min(curvature) < 0.0):
            repaired_radius, repaired_dr, repaired_ddr = repair_support_function(
                x, radius, dr, ddr, cfg.convexity_eps_m
            )
            repaired_extension = 2.0 * repaired_radius - self.spring.min_length_m
            repaired_tension = (
                self.spring.total_tension_at_rest_n
                + self.spring.total_stiffness_n_per_m * repaired_extension
            )
            repaired_torque = 2.0 * repaired_tension * repaired_dr
            repaired_curvature = repaired_radius + repaired_ddr
            repaired_cam_xy = support_to_xy(x, repaired_radius, repaired_dr)
            was_repaired = True
        else:
            repaired_radius = radius
            repaired_torque = scaled_torque
            repaired_curvature = curvature
            repaired_cam_xy = cam_xy
            was_repaired = False

        external_angles = x / angle_scale
        output_torque = -repaired_torque if cfg.invert_output_torque else repaired_torque
        output_table = torch.stack((external_angles, output_torque), dim=-1)
        rmse = torch.sqrt(torch.mean((repaired_torque - torque) ** 2))

        return ERCDesignResult(
            spring=self.spring,
            input_profile=profile,
            support_angles_rad=x,
            target_torque_nm=torque,
            scaled_torque_nm=scaled_torque,
            repaired_torque_nm=repaired_torque,
            output_torque_table=output_table,
            support_radius_m=radius,
            repaired_support_radius_m=repaired_radius,
            curvature_radius_m=curvature,
            repaired_curvature_radius_m=repaired_curvature,
            cam_xy_m=cam_xy,
            repaired_cam_xy_m=repaired_cam_xy,
            alpha=float(alpha),
            torque_rmse_nm=float(rmse.detach().cpu()),
            was_scaled=bool(alpha < 1.0),
            was_repaired=was_repaired,
        )

    def _energy_scaling(self, energy: torch.Tensor) -> tuple[float, torch.Tensor]:
        cfg = self.config
        dtype = energy.dtype
        device = energy.device
        safe_extension = self.spring.safe_extension_m(cfg.safety_factor)
        if safe_extension <= 0.0:
            raise ValueError("spring has no usable extension after applying safety_factor")

        energy_limit = self.spring.energy_from_extension_j(safe_extension)
        energy_limit_t = torch.as_tensor(energy_limit, dtype=dtype, device=device)
        emax = torch.max(energy)
        emin = torch.min(energy)
        span = emax - emin

        limits = [torch.ones((), dtype=dtype, device=device)]
        if emax > 0:
            limits.append(energy_limit_t / emax)
        if span > 0:
            limits.append(energy_limit_t / span)
        alpha_t = torch.minimum(torch.stack(limits).min(), torch.ones((), dtype=dtype, device=device))
        if alpha_t < 1.0:
            alpha_t = torch.clamp(alpha_t - cfg.energy_scale_eps, min=0.0)

        if emax > 0:
            initial_energy = energy_limit_t - alpha_t * emax
        else:
            initial_energy = -alpha_t * emin

        return float(alpha_t.detach().cpu()), initial_energy

    def _extension_from_energy(self, energy_j: torch.Tensor) -> torch.Tensor:
        k = torch.as_tensor(
            self.spring.total_stiffness_n_per_m,
            dtype=energy_j.dtype,
            device=energy_j.device,
        )
        t0 = torch.as_tensor(
            self.spring.total_tension_at_rest_n,
            dtype=energy_j.dtype,
            device=energy_j.device,
        )
        discriminant = t0**2 + 2.0 * k * torch.clamp(energy_j, min=0.0)
        return (-t0 + torch.sqrt(discriminant)) / k


def cumulative_trapezoid(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dx = torch.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return torch.cat((torch.zeros_like(y[:1]), torch.cumsum(area, dim=0)))


def finite_difference_first(y: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    dy = torch.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    dy[0] = (y[1] - y[0]) / dx
    dy[-1] = (y[-1] - y[-2]) / dx
    return dy


def finite_difference_second(y: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    ddy = torch.empty_like(y)
    ddy[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / dx**2
    ddy[0] = ddy[1]
    ddy[-1] = ddy[-2]
    return ddy


def interpolate_output_torque_table(
    output_torque_table: torch.Tensor,
    joint_angles_rad: torch.Tensor,
) -> torch.Tensor:
    if output_torque_table.ndim != 2 or output_torque_table.shape[1] != 2:
        raise ValueError("output_torque_table must have shape (n_points, 2)")
    angles = output_torque_table[:, 0].contiguous()
    torques = output_torque_table[:, 1].contiguous()
    if torch.any(torch.diff(angles) <= 0.0):
        raise ValueError("output_torque_table angles must be strictly increasing")
    if torch.any(joint_angles_rad < angles[0]) or torch.any(joint_angles_rad > angles[-1]):
        raise ValueError(
            "joint angle query is outside the final ERC torque-table domain "
            f"[{float(angles[0]):.6g}, {float(angles[-1]):.6g}] rad"
        )
    index = torch.searchsorted(angles, joint_angles_rad, right=True) - 1
    index = torch.clamp(index, 0, angles.numel() - 2)
    x0 = angles[index]
    x1 = angles[index + 1]
    y0 = torques[index]
    y1 = torques[index + 1]
    return y0 + (joint_angles_rad - x0) / (x1 - x0) * (y1 - y0)


def repair_support_function(
    x: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor,
    ddr: torch.Tensor,
    eps_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finite-difference convexity repair from ``repair_support_function.m``."""

    n = x.numel()
    dx = x[1] - x[0]
    rho = r + ddr
    g = torch.clamp(rho, min=eps_m).clone()

    a = torch.zeros((n, n), dtype=x.dtype, device=x.device)
    idx = torch.arange(n, device=x.device)
    a[idx, idx] = -2.0 / dx**2 + 1.0
    a[idx[1:], idx[:-1]] = 1.0 / dx**2
    a[idx[:-1], idx[1:]] = 1.0 / dx**2

    a[0, :] = 0.0
    a[0, 0] = 1.0
    g[0] = r[0]

    a[1, :] = 0.0
    a[1, 0:3] = torch.as_tensor([-3.0, 4.0, -1.0], dtype=x.dtype, device=x.device) / (
        2.0 * dx
    )
    g[1] = dr[0]

    a[-2, :] = 0.0
    a[-2, -3:] = torch.as_tensor([1.0, -4.0, 3.0], dtype=x.dtype, device=x.device) / (
        2.0 * dx
    )
    g[-2] = dr[-1]

    a[-1, :] = 0.0
    a[-1, -1] = 1.0
    g[-1] = r[-1]

    r_new = torch.linalg.solve(a, g)
    dr_new = finite_difference_first(r_new, dx)
    ddr_new = finite_difference_second(r_new, dx)
    dr_new[0] = dr[0]
    dr_new[-1] = dr[-1]
    ddr_new[0] = ddr_new[1]
    ddr_new[-2] = ddr_new[-3]
    ddr_new[-1] = ddr_new[-3]
    return r_new, dr_new, ddr_new


def support_to_xy(
    angles_rad: torch.Tensor,
    support_radius_m: torch.Tensor,
    support_radius_derivative_m: torch.Tensor,
) -> torch.Tensor:
    x = support_radius_m * torch.cos(angles_rad) - support_radius_derivative_m * torch.sin(
        angles_rad
    )
    y = support_radius_m * torch.sin(angles_rad) + support_radius_derivative_m * torch.cos(
        angles_rad
    )
    return torch.stack((x, y), dim=-1)


def export_curve_xyz(
    path: str | Path,
    xy_m: torch.Tensor,
    *,
    n_points: int | None = None,
    z_m: float = 0.0,
) -> None:
    """Export a 2D cam curve as SolidWorks-friendly mm XYZ rows."""

    xy = xy_m.detach().cpu()
    if n_points is None:
        n_points = int(xy.shape[0] * 2)
    if n_points != xy.shape[0]:
        xy = resample_by_arclength(xy, n_points)

    xyz_mm = torch.cat(
        (
            xy * 1000.0,
            torch.full((xy.shape[0], 1), z_m * 1000.0, dtype=xy.dtype),
        ),
        dim=-1,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\r\n") as handle:
        for row in xyz_mm:
            handle.write(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f}\n")


def resample_by_arclength(xy: torch.Tensor, n_points: int) -> torch.Tensor:
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    delta = torch.diff(xy, dim=0)
    distances = torch.sqrt(torch.sum(delta**2, dim=-1))
    cumulative = torch.cat((torch.zeros(1, dtype=xy.dtype), torch.cumsum(distances, dim=0)))
    even = torch.linspace(0.0, float(cumulative[-1]), n_points, dtype=xy.dtype)
    x = interpolate_pwl(even, cumulative, xy[:, 0])
    y = interpolate_pwl(even, cumulative, xy[:, 1])
    return torch.stack((x, y), dim=-1)
