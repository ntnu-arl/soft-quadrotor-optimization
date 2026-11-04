import math


REQUIRED_CIRCLE_KEYS = (
    "peak_velocity",
    "max_acceleration",
    "trajectory_diameter_x",
    "hold_before_s",
    "hold_after_s",
    "repeat_count",
)


def require_finite_number(name, value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def require_positive_number(name, value):
    parsed = require_finite_number(name, value)
    if not parsed > 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def require_nonnegative_number(name, value):
    parsed = require_finite_number(name, value)
    if not parsed >= 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def require_positive_int(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_circle_config(circle):
    for key in REQUIRED_CIRCLE_KEYS:
        if key not in circle:
            raise KeyError(f"trajectory.circle.{key} is required")

    parsed = {
        "peak_velocity": require_positive_number("trajectory.circle.peak_velocity", circle["peak_velocity"]),
        "max_acceleration": require_positive_number(
            "trajectory.circle.max_acceleration",
            circle["max_acceleration"],
        ),
        "trajectory_diameter_x": require_positive_number(
            "trajectory.circle.trajectory_diameter_x",
            circle["trajectory_diameter_x"],
        ),
        "hold_before_s": require_nonnegative_number("trajectory.circle.hold_before_s", circle["hold_before_s"]),
        "hold_after_s": require_nonnegative_number("trajectory.circle.hold_after_s", circle["hold_after_s"]),
        "repeat_count": require_positive_int("trajectory.circle.repeat_count", circle["repeat_count"]),
    }
    parsed["radius"] = 0.5 * parsed["trajectory_diameter_x"]
    parsed["omega"] = parsed["peak_velocity"] / parsed["radius"]
    parsed["period_s"] = 2.0 * math.pi / parsed["omega"]
    parsed["circumference"] = 2.0 * math.pi * parsed["radius"]
    parsed["path_length"] = parsed["repeat_count"] * parsed["circumference"]
    parsed["accel_time_s"] = math.pi * parsed["peak_velocity"] / (2.0 * parsed["max_acceleration"])
    parsed["accel_distance"] = 0.5 * parsed["peak_velocity"] * parsed["accel_time_s"]
    if 2.0 * parsed["accel_distance"] > parsed["path_length"]:
        raise ValueError("trajectory.circle path is too short to reach peak_velocity with max_acceleration")
    parsed["cruise_distance"] = parsed["path_length"] - 2.0 * parsed["accel_distance"]
    parsed["cruise_time_s"] = parsed["cruise_distance"] / parsed["peak_velocity"]
    parsed["moving_duration_s"] = 2.0 * parsed["accel_time_s"] + parsed["cruise_time_s"]
    parsed["total_duration_s"] = parsed["hold_before_s"] + parsed["moving_duration_s"] + parsed["hold_after_s"]
    return parsed


def compute_circle_distance_and_speed(circle, moving_elapsed_s):
    if moving_elapsed_s < circle["accel_time_s"]:
        phase = math.pi * moving_elapsed_s / circle["accel_time_s"]
        speed = 0.5 * circle["peak_velocity"] * (1.0 - math.cos(phase))
        distance = 0.5 * circle["peak_velocity"] * (
            moving_elapsed_s - circle["accel_time_s"] * math.sin(phase) / math.pi
        )
        tangential_acceleration = circle["max_acceleration"] * math.sin(phase)
        return distance, speed, tangential_acceleration

    cruise_end_s = circle["accel_time_s"] + circle["cruise_time_s"]
    if moving_elapsed_s < cruise_end_s:
        cruise_elapsed_s = moving_elapsed_s - circle["accel_time_s"]
        distance = circle["accel_distance"] + circle["peak_velocity"] * cruise_elapsed_s
        return distance, circle["peak_velocity"], 0.0

    decel_elapsed_s = moving_elapsed_s - cruise_end_s
    phase = math.pi * decel_elapsed_s / circle["accel_time_s"]
    speed = 0.5 * circle["peak_velocity"] * (1.0 + math.cos(phase))
    distance = (
        circle["accel_distance"]
        + circle["cruise_distance"]
        + 0.5 * circle["peak_velocity"] * (
            decel_elapsed_s + circle["accel_time_s"] * math.sin(phase) / math.pi
        )
    )
    tangential_acceleration = -circle["max_acceleration"] * math.sin(phase)
    return distance, speed, tangential_acceleration


def compute_attitude_quaternion_from_acceleration(acceleration_world, gravity_world, yaw_ref):
    if len(acceleration_world) != 3:
        raise ValueError("acceleration_world must contain three values")
    if len(gravity_world) != 3:
        raise ValueError("gravity_world must contain three values")

    acceleration = [
        require_finite_number(f"acceleration_world[{index}]", value)
        for index, value in enumerate(acceleration_world)
    ]
    gravity = [
        require_finite_number(f"gravity_world[{index}]", value)
        for index, value in enumerate(gravity_world)
    ]
    yaw = require_finite_number("yaw_ref", yaw_ref)

    thrust_direction = [
        acceleration[0] - gravity[0],
        acceleration[1] - gravity[1],
        acceleration[2] - gravity[2],
    ]
    thrust_norm = math.sqrt(sum(component * component for component in thrust_direction))
    if not math.isfinite(thrust_norm) or not thrust_norm > 1e-9:
        raise ValueError(f"invalid thrust direction norm={thrust_norm}")

    body_z = [component / thrust_norm for component in thrust_direction]
    yaw_x = [math.cos(yaw), math.sin(yaw), 0.0]
    body_y_unnormalized = [
        body_z[1] * yaw_x[2] - body_z[2] * yaw_x[1],
        body_z[2] * yaw_x[0] - body_z[0] * yaw_x[2],
        body_z[0] * yaw_x[1] - body_z[1] * yaw_x[0],
    ]
    body_y_norm = math.sqrt(sum(component * component for component in body_y_unnormalized))
    if not math.isfinite(body_y_norm) or not body_y_norm > 1e-9:
        raise ValueError(f"invalid body_y norm={body_y_norm}")

    body_y = [component / body_y_norm for component in body_y_unnormalized]
    body_x = [
        body_y[1] * body_z[2] - body_y[2] * body_z[1],
        body_y[2] * body_z[0] - body_y[0] * body_z[2],
        body_y[0] * body_z[1] - body_y[1] * body_z[0],
    ]

    rotation = (
        (body_x[0], body_y[0], body_z[0]),
        (body_x[1], body_y[1], body_z[1]),
        (body_x[2], body_y[2], body_z[2]),
    )
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * scale,
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
        ]
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        quaternion = [
            (rotation[2][1] - rotation[1][2]) / scale,
            0.25 * scale,
            (rotation[0][1] + rotation[1][0]) / scale,
            (rotation[0][2] + rotation[2][0]) / scale,
        ]
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        quaternion = [
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[0][1] + rotation[1][0]) / scale,
            0.25 * scale,
            (rotation[1][2] + rotation[2][1]) / scale,
        ]
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        quaternion = [
            (rotation[1][0] - rotation[0][1]) / scale,
            (rotation[0][2] + rotation[2][0]) / scale,
            (rotation[1][2] + rotation[2][1]) / scale,
            0.25 * scale,
        ]

    quat_norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isfinite(quat_norm) or not quat_norm > 1e-9:
        raise ValueError(f"invalid quaternion norm={quat_norm}")
    return [component / quat_norm for component in quaternion]


def compute_circle_reference(circle, start_position, elapsed_s):
    elapsed = require_nonnegative_number("elapsed_s", elapsed_s)
    if len(start_position) != 3:
        raise ValueError("start_position must contain three values")
    start = [require_finite_number(f"start_position[{index}]", value) for index, value in enumerate(start_position)]
    position = [
        start[0],
        start[1],
        start[2],
    ]
    velocity = [0.0, 0.0, 0.0]
    acceleration = [0.0, 0.0, 0.0]

    if elapsed < circle["hold_before_s"]:
        return position, velocity, acceleration

    moving_elapsed_s = elapsed - circle["hold_before_s"]
    if not moving_elapsed_s < circle["moving_duration_s"]:
        return position, velocity, acceleration

    distance, speed, tangential_acceleration = compute_circle_distance_and_speed(circle, moving_elapsed_s)
    theta = math.fmod(distance / circle["radius"], 2.0 * math.pi)
    position = [
        start[0] + circle["radius"] * (1.0 - math.cos(theta)),
        start[1] + circle["radius"] * math.sin(theta),
        start[2],
    ]
    velocity = [
        speed * math.sin(theta),
        speed * math.cos(theta),
        0.0,
    ]
    centripetal_acceleration = (speed * speed) / circle["radius"]
    acceleration = [
        tangential_acceleration * math.sin(theta) + centripetal_acceleration * math.cos(theta),
        tangential_acceleration * math.cos(theta) - centripetal_acceleration * math.sin(theta),
        0.0,
    ]
    return position, velocity, acceleration
