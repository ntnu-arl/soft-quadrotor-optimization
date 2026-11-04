from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent / "src"))

from collision_trajectory import CollisionReferenceTrajectory
from nmpc_controller import _select_mode_results


def _build_collision_config():
    return {
        "use_absolute_position": True,
        "gap_ref1_mm": [0.0, 0.0, 0.0],
        "gap_ref2_mm": [0.0, 0.0, 0.0],
        "setpoints": [
            {
                "position_rel_gap": [-2.0, 0.1, 1.05],
                "velocity": [0.0, 0.0, 0.0],
                "setpoint_time": 1.0,
                "waypoint_x_limit_rel_gap": None,
                "control_mode": "NMPC_DIRECT_ACTUATOR",
                "nmpc_mode": "NMPC_FLIGHT_MODE",
            },
            {
                "position_rel_gap": [0.25, 0.0, 1.05],
                "velocity": [2.1, 0.0, 0.0],
                "setpoint_time": None,
                "waypoint_x_limit_rel_gap": 0.0,
                "control_mode": "NMPC_DIRECT_ACTUATOR",
                "nmpc_mode": "NMPC_RECOVERY_MODE",
            },
            {
                "position_rel_gap": [1.0, 0.0, 1.05],
                "velocity": [0.0, 0.0, 0.0],
                "setpoint_time": None,
                "waypoint_x_limit_rel_gap": None,
                "control_mode": "NMPC_DIRECT_ACTUATOR",
                "nmpc_mode": "NMPC_FLIGHT_MODE",
            },
        ],
    }


def test_collision_reference_uses_gap_relative_world_position():
    env_origins = torch.tensor([[10.0, 5.0, 0.25]], dtype=torch.float32)
    traj = CollisionReferenceTrajectory(env_origins, 0.3, _build_collision_config(), "cpu")

    pos_world = env_origins.clone()
    current_time = torch.tensor([0.0], dtype=torch.float32)
    ref_pos, ref_vel, _, _, control_mode, nmpc_mode = traj.get_reference(pos_world, current_time)

    assert torch.allclose(ref_pos[0], torch.tensor([8.3, 5.1, 1.3], dtype=torch.float32))
    assert torch.allclose(ref_vel[0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    assert control_mode == ["NMPC_DIRECT_ACTUATOR"]
    assert nmpc_mode == ["NMPC_FLIGHT_MODE"]


def test_collision_reference_transitions_on_setpoint_time():
    env_origins = torch.zeros((1, 3), dtype=torch.float32)
    traj = CollisionReferenceTrajectory(env_origins, 0.0, _build_collision_config(), "cpu")

    pos_world = torch.tensor([[-2.0, 0.0, 1.05]], dtype=torch.float32)
    ref_pos, ref_vel, _, _, _, nmpc_mode = traj.get_reference(pos_world, torch.tensor([1.01], dtype=torch.float32))

    assert torch.allclose(ref_pos[0], torch.tensor([0.25, 0.0, 1.05], dtype=torch.float32))
    assert torch.allclose(ref_vel[0], torch.tensor([2.1, 0.0, 0.0], dtype=torch.float32))
    assert nmpc_mode == ["NMPC_RECOVERY_MODE"]


def test_collision_reference_transitions_on_x_limit():
    env_origins = torch.zeros((1, 3), dtype=torch.float32)
    traj = CollisionReferenceTrajectory(env_origins, 0.0, _build_collision_config(), "cpu")
    traj.phase[0] = 1

    pos_world = torch.tensor([[0.02, 0.0, 1.05]], dtype=torch.float32)
    ref_pos, ref_vel, _, _, _, nmpc_mode = traj.get_reference(pos_world, torch.tensor([1.5], dtype=torch.float32))

    assert torch.allclose(ref_pos[0], torch.tensor([1.0, 0.0, 1.05], dtype=torch.float32))
    assert torch.allclose(ref_vel[0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    assert nmpc_mode == ["NMPC_FLIGHT_MODE"]


def test_collision_reference_rejects_non_direct_actuator_for_simulation():
    collision_config = _build_collision_config()
    collision_config["setpoints"][1]["control_mode"] = "PX4_POSITION_CONTROL"
    traj = CollisionReferenceTrajectory(torch.zeros((1, 3), dtype=torch.float32), 0.0, collision_config, "cpu")

    with pytest.raises(ValueError, match="NMPC simulation only supports"):
        traj.require_direct_actuator_control()


def test_collision_reference_requires_non_empty_setpoints():
    with pytest.raises(KeyError, match="trajectory.collision.setpoints is required"):
        CollisionReferenceTrajectory(torch.zeros((1, 3), dtype=torch.float32), 0.0, {}, "cpu")

    with pytest.raises(ValueError, match="trajectory.collision.setpoints must not be empty"):
        CollisionReferenceTrajectory(
            torch.zeros((1, 3), dtype=torch.float32),
            0.0,
            {"setpoints": []},
            "cpu",
        )


def test_select_mode_results_uses_nominal_and_recovery_outputs():
    nominal_results = {
        "solver_statuses": np.array([10, 11], dtype=np.int32),
        "solver_costs": np.array([100.0, 101.0], dtype=np.float64),
        "u_opt": np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float64),
        "x1": np.array([[1.0] * 17, [2.0] * 17], dtype=np.float64),
        "x_end": np.array([[3.0] * 17, [4.0] * 17], dtype=np.float64),
        "horizon_controls": np.array([
            [[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0]],
            [[7.0, 7.0, 7.0, 7.0], [8.0, 8.0, 8.0, 8.0]],
        ], dtype=np.float64),
    }
    recovery_results = {
        "solver_statuses": np.array([20, 21], dtype=np.int32),
        "solver_costs": np.array([200.0, 201.0], dtype=np.float64),
        "u_opt": np.array([[9.0, 9.0, 9.0, 9.0], [10.0, 10.0, 10.0, 10.0]], dtype=np.float64),
        "x1": np.array([[11.0] * 17, [12.0] * 17], dtype=np.float64),
        "x_end": np.array([[13.0] * 17, [14.0] * 17], dtype=np.float64),
        "horizon_controls": np.array([
            [[15.0, 15.0, 15.0, 15.0], [16.0, 16.0, 16.0, 16.0]],
            [[17.0, 17.0, 17.0, 17.0], [18.0, 18.0, 18.0, 18.0]],
        ], dtype=np.float64),
    }

    selected = _select_mode_results(
        nominal_results,
        recovery_results,
        np.array(["NMPC_FLIGHT_MODE", "NMPC_RECOVERY_MODE"], dtype=object),
    )

    assert selected["solver_statuses"].tolist() == [10, 21]
    assert selected["solver_costs"].tolist() == [100.0, 201.0]
    assert selected["u_opt"][0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert selected["u_opt"][1].tolist() == [10.0, 10.0, 10.0, 10.0]


def test_select_mode_results_rejects_invalid_nmpc_mode():
    nominal_results = {
        "solver_statuses": np.array([1], dtype=np.int32),
        "solver_costs": np.array([1.0], dtype=np.float64),
        "u_opt": np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        "x1": np.array([[1.0] * 17], dtype=np.float64),
        "x_end": np.array([[1.0] * 17], dtype=np.float64),
        "horizon_controls": np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float64),
    }
    recovery_results = {
        "solver_statuses": np.array([2], dtype=np.int32),
        "solver_costs": np.array([2.0], dtype=np.float64),
        "u_opt": np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float64),
        "x1": np.array([[2.0] * 17], dtype=np.float64),
        "x_end": np.array([[2.0] * 17], dtype=np.float64),
        "horizon_controls": np.array([[[2.0, 2.0, 2.0, 2.0]]], dtype=np.float64),
    }

    with pytest.raises(ValueError, match="unsupported NMPC mode"):
        _select_mode_results(
            nominal_results,
            recovery_results,
            np.array(["BAD_MODE"], dtype=object),
        )
