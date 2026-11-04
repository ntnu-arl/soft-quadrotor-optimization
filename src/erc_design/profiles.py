from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _as_tensor(
    value,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value
        if dtype is not None or device is not None:
            out = out.to(dtype=dtype or out.dtype, device=device or out.device)
        return out
    return torch.as_tensor(value, dtype=dtype or torch.float64, device=device)


@dataclass(frozen=True)
class PiecewiseLinearTorqueProfile:
    """Piecewise-linear torque profile.

    ``knots`` is shaped ``(n_points, 2)`` with columns
    ``[angle_rad, torque_nm]``. The angle convention is interpreted by
    ``ERCDesignConfig``; by default it is the external joint angle.
    """

    knots: torch.Tensor

    def __post_init__(self):
        if self.knots.ndim != 2 or self.knots.shape[1] != 2:
            raise ValueError("knots must have shape (n_points, 2)")
        if self.knots.shape[0] < 2:
            raise ValueError("at least two knots are required")
        if torch.any(torch.diff(self.angles) <= 0):
            raise ValueError("profile angles must be strictly increasing")

    @property
    def angles(self) -> torch.Tensor:
        return self.knots[:, 0]

    @property
    def torques(self) -> torch.Tensor:
        return self.knots[:, 1]

    @property
    def n_segments(self) -> int:
        return int(self.knots.shape[0] - 1)

    def to_tensor(self, *, invert_torque: bool = False) -> torch.Tensor:
        out = self.knots.clone()
        if invert_torque:
            out[:, 1] = -out[:, 1]
        return out

    def interpolate(self, angles_rad: torch.Tensor) -> torch.Tensor:
        return interpolate_pwl(angles_rad, self.angles, self.torques)

    @classmethod
    def from_xy(
        cls,
        angles_rad,
        torques_nm,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "PiecewiseLinearTorqueProfile":
        angles = _as_tensor(angles_rad, dtype=dtype, device=device)
        torques = _as_tensor(torques_nm, dtype=angles.dtype, device=angles.device)
        return cls(torch.stack((angles, torques), dim=-1))


@dataclass(frozen=True)
class ProfileParameterization:
    """Decode optimizer vectors into PWL torque profiles.

    Default settings model five segments, i.e. six profile knots. If
    ``angle_bounds_rad`` is provided, the two endpoint angles are fixed and
    the parameter vector only contains the interior angles. Torques are always
    free unless ``fixed_start_torque_nm`` or ``fixed_end_torque_nm`` is set.

    The vector layout is split: all angle parameters first, then all free
    torque parameters.
    """

    n_segments: int = 5
    angle_bounds_rad: tuple[float, float] | None = None
    fixed_start_torque_nm: float | None = None
    fixed_end_torque_nm: float | None = None
    sort_angles: bool = True

    @property
    def n_points(self) -> int:
        return self.n_segments + 1

    @property
    def n_angle_parameters(self) -> int:
        return self.n_points if self.angle_bounds_rad is None else self.n_points - 2

    @property
    def n_torque_parameters(self) -> int:
        fixed = int(self.fixed_start_torque_nm is not None) + int(
            self.fixed_end_torque_nm is not None
        )
        return self.n_points - fixed

    @property
    def n_parameters(self) -> int:
        return self.n_angle_parameters + self.n_torque_parameters

    def decode(
        self,
        parameters,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> PiecewiseLinearTorqueProfile:
        params = _as_tensor(parameters, dtype=dtype, device=device).flatten()
        if params.numel() != self.n_parameters:
            raise ValueError(
                f"expected {self.n_parameters} parameters, got {params.numel()}"
            )

        angle_params = params[: self.n_angle_parameters]
        torque_params = params[self.n_angle_parameters :]

        if self.angle_bounds_rad is None:
            angles = angle_params
            if self.sort_angles:
                angles = torch.sort(angles).values
        else:
            lo, hi = self.angle_bounds_rad
            interior = angle_params
            if self.sort_angles:
                interior = torch.sort(interior).values
            endpoints = torch.as_tensor([lo, hi], dtype=params.dtype, device=params.device)
            angles = torch.cat((endpoints[:1], interior, endpoints[1:]))

        torques = []
        cursor = 0
        if self.fixed_start_torque_nm is None:
            torques.append(torque_params[cursor])
            cursor += 1
        else:
            torques.append(
                torch.as_tensor(
                    self.fixed_start_torque_nm, dtype=params.dtype, device=params.device
                )
            )

        n_interior_torques = self.n_points - 2
        torques.extend(torque_params[cursor : cursor + n_interior_torques])
        cursor += n_interior_torques

        if self.fixed_end_torque_nm is None:
            torques.append(torque_params[cursor])
        else:
            torques.append(
                torch.as_tensor(
                    self.fixed_end_torque_nm, dtype=params.dtype, device=params.device
                )
            )

        return PiecewiseLinearTorqueProfile(torch.stack((angles, torch.stack(torques)), dim=-1))

    def sample(
        self,
        *,
        angle_bounds_rad: tuple[float, float] | None = None,
        torque_bounds_nm: tuple[float, float],
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> PiecewiseLinearTorqueProfile:
        bounds = self.angle_bounds_rad or angle_bounds_rad
        if bounds is None:
            raise ValueError("angle_bounds_rad is required when endpoints are not fixed")

        lo, hi = bounds
        interior = torch.rand(
            self.n_points - 2, generator=generator, dtype=dtype, device=device
        )
        interior = torch.sort(lo + (hi - lo) * interior).values
        angles = torch.cat(
            (
                torch.as_tensor([lo], dtype=dtype, device=device),
                interior,
                torch.as_tensor([hi], dtype=dtype, device=device),
            )
        )

        tlo, thi = torque_bounds_nm
        torques = tlo + (thi - tlo) * torch.rand(
            self.n_points, generator=generator, dtype=dtype, device=device
        )
        if self.fixed_start_torque_nm is not None:
            torques[0] = self.fixed_start_torque_nm
        if self.fixed_end_torque_nm is not None:
            torques[-1] = self.fixed_end_torque_nm

        return PiecewiseLinearTorqueProfile(torch.stack((angles, torques), dim=-1))


def interpolate_pwl(
    query: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation matching MATLAB ``interp1(..., 'linear')`` in range."""

    if torch.any(query < x[0]) or torch.any(query > x[-1]):
        raise ValueError("query contains values outside the interpolation domain")
    idx = torch.searchsorted(x, query, right=True) - 1
    idx = torch.clamp(idx, 0, x.numel() - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    w = (query - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)
