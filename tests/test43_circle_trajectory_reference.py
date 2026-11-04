import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

sys.path.append(str(Path(__file__).parent.parent / "voxl_src"))
from trajectory_reference import (
    compute_attitude_quaternion_from_acceleration,
    compute_circle_reference,
    parse_circle_config,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "test43_trajectory_reference"
CONFIG_PATH = PROJECT_ROOT / "configs" / "nmpc_controller_cfg.yaml"
TEST_START_POSITION = [-0.75, 0.5, 1.05]
VISUALIZATION_START_POSITION = [0.0, 0.0, 0.0]


def assert_vector_close(actual, expected):
    assert actual == pytest.approx(expected, abs=1e-12)


def quat_to_rotmat(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def build_test_circle():
    return parse_circle_config(
        {
            "peak_velocity": 1.0,
            "max_acceleration": 1.0,
            "trajectory_diameter_x": 2.0,
            "hold_before_s": 2.0,
            "hold_after_s": 3.0,
            "repeat_count": 2,
        }
    )


def test_circle_reference_holds_before_start():
    circle = build_test_circle()

    position, velocity, acceleration = compute_circle_reference(circle, TEST_START_POSITION, 1.0)

    assert_vector_close(position, [-0.75, 0.5, 1.05])
    assert_vector_close(velocity, [0.0, 0.0, 0.0])
    assert_vector_close(acceleration, [0.0, 0.0, 0.0])


def test_circle_reference_starts_at_zero_speed():
    circle = build_test_circle()

    position, velocity, acceleration = compute_circle_reference(circle, TEST_START_POSITION, circle["hold_before_s"])

    assert_vector_close(position, [-0.75, 0.5, 1.05])
    assert_vector_close(velocity, [0.0, 0.0, 0.0])
    assert_vector_close(acceleration, [0.0, 0.0, 0.0])


def test_circle_reference_ramps_counterclockwise():
    circle = build_test_circle()

    position, velocity, acceleration = compute_circle_reference(
        circle,
        TEST_START_POSITION,
        circle["hold_before_s"] + 0.5 * circle["accel_time_s"],
    )

    assert position[0] > -0.75
    assert position[1] > 0.5
    assert np.linalg.norm(velocity) == pytest.approx(0.5)
    assert np.linalg.norm(acceleration) == pytest.approx(math.sqrt(1.0 + 0.25**2))


def test_circle_reference_returns_to_start_after_one_loop_distance():
    circle = build_test_circle()
    time_s = circle["hold_before_s"] + circle["accel_time_s"] + (
        circle["circumference"] - circle["accel_distance"]
    ) / circle["peak_velocity"]

    position, velocity, acceleration = compute_circle_reference(circle, TEST_START_POSITION, time_s)

    assert_vector_close(position, [-0.75, 0.5, 1.05])
    assert_vector_close(velocity, [0.0, 1.0, 0.0])
    assert_vector_close(acceleration, [1.0, 0.0, 0.0])


def test_circle_reference_holds_after_repeats():
    circle = build_test_circle()
    elapsed_s = circle["hold_before_s"] + circle["moving_duration_s"] + 0.25 * circle["hold_after_s"]

    position, velocity, acceleration = compute_circle_reference(circle, TEST_START_POSITION, elapsed_s)

    assert_vector_close(position, [-0.75, 0.5, 1.05])
    assert_vector_close(velocity, [0.0, 0.0, 0.0])
    assert_vector_close(acceleration, [0.0, 0.0, 0.0])


def test_circle_reference_quarter_turn():
    circle = build_test_circle()
    elapsed_s = circle["hold_before_s"] + circle["accel_time_s"] + (
        0.25 * circle["circumference"] - circle["accel_distance"]
    ) / circle["peak_velocity"]

    position, velocity, acceleration = compute_circle_reference(circle, TEST_START_POSITION, elapsed_s)

    assert_vector_close(position, [0.25, 1.5, 1.05])
    assert_vector_close(velocity, [1.0, 0.0, 0.0])
    assert_vector_close(acceleration, [0.0, -1.0, 0.0])


def test_circle_reference_decelerates_tangentially_near_end():
    circle = build_test_circle()
    elapsed_s = (
        circle["hold_before_s"]
        + circle["accel_time_s"]
        + circle["cruise_time_s"]
        + 0.5 * circle["accel_time_s"]
    )

    _, velocity, acceleration = compute_circle_reference(circle, TEST_START_POSITION, elapsed_s)

    assert np.linalg.norm(velocity) == pytest.approx(0.5)
    assert np.linalg.norm(acceleration) == pytest.approx(math.sqrt(1.0 + 0.25**2))


def test_circle_reference_acceleration_is_continuous_at_motion_boundaries():
    circle = build_test_circle()
    epsilon_s = 1.0e-6

    _, _, start_acceleration = compute_circle_reference(
        circle,
        TEST_START_POSITION,
        circle["hold_before_s"] + epsilon_s,
    )
    _, _, end_acceleration = compute_circle_reference(
        circle,
        TEST_START_POSITION,
        circle["hold_before_s"] + circle["moving_duration_s"] - epsilon_s,
    )

    assert np.linalg.norm(start_acceleration) < 1.0e-5
    assert np.linalg.norm(end_acceleration) < 1.0e-5


def test_attitude_reference_aligns_body_z_with_required_force():
    quaternion = compute_attitude_quaternion_from_acceleration([0.0, -1.0, 0.0], [0.0, 0.0, -9.81], 0.0)
    rotation = quat_to_rotmat(quaternion)
    body_z = rotation[:, 2]
    required_force_direction = np.array([0.0, -1.0, 9.81], dtype=float)
    required_force_direction /= np.linalg.norm(required_force_direction)

    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert body_z == pytest.approx(required_force_direction, abs=1e-12)


def test_circle_duration_uses_repeat_count():
    circle = build_test_circle()

    assert circle["period_s"] == pytest.approx(2.0 * math.pi)
    assert circle["moving_duration_s"] == pytest.approx(4.5 * math.pi)
    assert circle["total_duration_s"] == pytest.approx(5.0 + 4.5 * math.pi)


def make_colored_line(ax, x, y, speed, title, x_label, y_label):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_speed = 0.5 * (speed[:-1] + speed[1:])
    line_collection = LineCollection(segments, cmap="viridis", linewidth=2.5)
    line_collection.set_array(segment_speed)
    ax.add_collection(line_collection)
    ax.scatter(x[0], y[0], color="black", s=35, marker="o", label="start", zorder=3)
    ax.scatter(x[-1], y[-1], color="black", s=35, marker="x", label="end", zorder=3)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return line_collection


def build_circle_plot_samples(circle, start_position, sample_count):
    times = np.linspace(0.0, circle["total_duration_s"], sample_count)
    positions = []
    velocities = []
    for time_s in times:
        position, velocity, _ = compute_circle_reference(circle, start_position, float(time_s))
        positions.append(position)
        velocities.append(velocity)
    positions_array = np.asarray(positions, dtype=float)
    velocities_array = np.asarray(velocities, dtype=float)
    speed = np.linalg.norm(velocities_array, axis=1)
    return positions_array, speed


def build_collision_plot_samples(collision_config):
    setpoints = collision_config["setpoints"]
    if not isinstance(setpoints, list) or len(setpoints) < 2:
        raise ValueError("trajectory.collision.setpoints must contain at least two setpoints for plotting")

    positions = []
    velocities = []
    for setpoint in setpoints:
        positions.append(setpoint["position_rel_gap"])
        velocities.append(setpoint["velocity"])

    positions_array = np.asarray(positions, dtype=float)
    velocities_array = np.asarray(velocities, dtype=float)
    if positions_array.shape[1] != 3 or velocities_array.shape[1] != 3:
        raise ValueError("trajectory.collision setpoint positions and velocities must be 3D")

    speed = np.linalg.norm(velocities_array, axis=1)
    return positions_array, speed


def test_visualize_yaml_trajectories_top_view():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    trajectory_cfg = cfg["trajectory"]
    circle = parse_circle_config(trajectory_cfg["circle"])

    circle_positions, circle_speed = build_circle_plot_samples(circle, VISUALIZATION_START_POSITION, 1000)
    collision_positions, collision_speed = build_collision_plot_samples(trajectory_cfg["collision"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "yaml_trajectories_top_view.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    circle_line = make_colored_line(
        axes[0],
        circle_positions[:, 0],
        circle_positions[:, 1],
        circle_speed,
        "circle: top view relative to NMPC start",
        "x rel initial [m]",
        "y rel initial [m]",
    )
    collision_line = make_colored_line(
        axes[1],
        collision_positions[:, 0],
        collision_positions[:, 1],
        collision_speed,
        "collision setpoints: top view",
        "x rel gap [m]",
        "y rel gap [m]",
    )
    fig.colorbar(circle_line, ax=axes[0], label="speed [m/s]")
    fig.colorbar(collision_line, ax=axes[1], label="configured speed [m/s]")
    fig.suptitle(f"Trajectories from {CONFIG_PATH.name}; active={trajectory_cfg['active']}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
