from __future__ import annotations
import numpy as np
from pathlib import Path
from airframe_encoding import (
    PARAM_BOUNDS,
    AIRFRAME_CONSTANTS,
    TORQUE_RESPONSE_MAX_ANGLE_DEG,
    TORQUE_RESPONSE_RAMP_ANGLE_DEG,
    build_torque_response_functions,
    Airframe,
)
from rotor_dynamics import manufacturerComponentData

def validate_airframe_force_distribution_with_qp(airframe: Airframe, desired_wrench: np.ndarray, verbose: bool, wrench_tolerance: float, max_min_motor_force_use: tuple[float,float]) -> tuple[bool, str]:
    """Validate airframe design using QP to find feasible motor forces.

    Solves: minimize ||B @ f - w_desired||^2
            subject to f_min <= f <= f_max
            f_min = max_min_motor_force_use[0] * (f_max_motor - f_min_motor) + f_min_motor
            f_max = max_min_motor_force_use[1] * (f_max_motor - f_min_motor) + f_min_motor

    Args:
        airframe: Airframe object
        desired_wrench: (6,) array of desired [fx, fy, fz, tx, ty, tz]
        verbose: if True, print detailed validation info
        wrench_tolerance: Maximum allowed wrench error (N or N·m) for validation

    Returns:
        (is_valid, message): Tuple of validation result and message
    """
    from scipy.optimize import minimize
    if verbose:
        from utils import _visualize_airframe

    try:
        alloc_data = airframe.compute_allocation_matrix()
        B = alloc_data['B']
        mass = alloc_data['mass']
        motor_positions = alloc_data['motor_positions']
        motor_thrust_directions = alloc_data['motor_thrust_directions']
        motor_directions = alloc_data['motor_directions'].reshape(-1, 1)
        min_forces = alloc_data['min_forces']
        max_forces = alloc_data['max_forces']
        thrust_to_torque_ratio = alloc_data['thrust_to_torque_ratio']
        n_motors = B.shape[1]

        def objective(f):
            wrench_error = B @ f - desired_wrench
            return 0.5 * np.sum(wrench_error ** 2)

        def jacobian(f):
            return B.T @ (B @ f - desired_wrench)

        adjusted_min_forces = max_min_motor_force_use[0] * (max_forces - min_forces) + min_forces
        adjusted_max_forces = max_min_motor_force_use[1] * (max_forces - min_forces) + min_forces
        bounds = [(adjusted_min_forces[i], adjusted_max_forces[i]) for i in range(n_motors)]

        best_result = None
        best_objective_value = np.inf
        rng = np.random.RandomState(42)
        for _ in range(10):
            f_init = rng.uniform(adjusted_min_forces, adjusted_max_forces)
            result = minimize(
                objective,
                f_init,
                method='L-BFGS-B',
                jac=jacobian,
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9}
            )
            if result.success and result.fun < best_objective_value:
                best_result = result
                best_objective_value = result.fun


        result = best_result if best_result is not None else result

        if not result.success:
            msg = f"QP optimization failed: {result.message}"
            if verbose:
                print(f"[Validation QP] ✗ FAILED: {msg}")
            return False, msg

        optimal_forces = result.x
        achieved_wrench = B @ optimal_forces
        wrench_error = achieved_wrench - desired_wrench
        wrench_error_norm = np.linalg.norm(wrench_error)
        max_wrench_component_error = np.abs(wrench_error).max()

        if wrench_error_norm > wrench_tolerance * np.sqrt(6):
            msg = f"Cannot achieve desired wrench (error={wrench_error_norm:.3f}N, max_component={max_wrench_component_error:.3f}N)"
            if verbose:
                print(f"[Validation QP] ✗ INVALID: {msg}")
                print(f"  Desired wrench: [{', '.join(f'{w:.3f}' for w in desired_wrench)}]")
                print(f"  Achieved wrench: [{', '.join(f'{w:.3f}' for w in achieved_wrench)}]")
                print(f"  Wrench error: [{', '.join(f'{e:.3f}' for e in wrench_error)}]")
                print(f"  Motor force limits (adjusted): [{adjusted_min_forces.min():.3f}, {adjusted_max_forces.max():.3f}]N")
                print(f"  Motor force limits (absolute): [{min_forces.min():.3f}, {max_forces.max():.3f}]N")
                print(f"  Force usage range: {max_min_motor_force_use[0]*100:.0f}%-{max_min_motor_force_use[1]*100:.0f}%")
                print(f"  Number of motors: {n_motors}")
                _visualize_airframe(motor_positions, motor_thrust_directions, motor_directions, optimal_forces, thrust_to_torque_ratio, desired_wrench, achieved_wrench, mass, (adjusted_min_forces, adjusted_max_forces))
            return False, msg

        if verbose:
            motor_pct_use = (optimal_forces - min_forces) / (max_forces - min_forces) * 100
            min_motor_pct_use = motor_pct_use.min()
            max_motor_pct_use = motor_pct_use.max()
            accelerations = (optimal_forces[:, np.newaxis] / mass) * motor_thrust_directions
            total_accel = accelerations.sum(axis=0)
            num_at_lower = np.sum(np.abs(optimal_forces - adjusted_min_forces) < 1e-3)
            num_at_upper = np.sum(np.abs(optimal_forces - adjusted_max_forces) < 1e-3)

            print(f"[Validation QP] ✓ VALID: Can achieve desired wrench within tolerance")
            print(f"  Desired wrench: [{', '.join(f'{w:.3f}' for w in desired_wrench)}]")
            print(f"  Achieved wrench: [{', '.join(f'{w:.3f}' for w in achieved_wrench)}]")
            print(f"  Wrench error norm: {wrench_error_norm:.6f}N")
            print(f"  Max wrench component error: {max_wrench_component_error:.6f}N")
            print(f"  Motor force limits (adjusted): [{adjusted_min_forces.min():.3f}, {adjusted_max_forces.max():.3f}]N")
            print(f"  Motor force limits (absolute): [{min_forces.min():.3f}, {max_forces.max():.3f}]N")
            print(f"  Force usage range: {max_min_motor_force_use[0]*100:.0f}%-{max_min_motor_force_use[1]*100:.0f}%")
            print(f"  Number of motors: {n_motors}")
            print(f"  Motors at lower bound: {num_at_lower}")
            print(f"  Motors at upper bound: {num_at_upper}")
            print(f"  Minimum motor % use: {min_motor_pct_use:.1f}% ∈ ({max_min_motor_force_use[0]*100:.0f}%, {max_min_motor_force_use[1]*100:.0f}%)")
            print(f"  Maximum motor % use: {max_motor_pct_use:.1f}% ∈ ({max_min_motor_force_use[0]*100:.0f}%, {max_min_motor_force_use[1]*100:.0f}%)")
            print(f"  Total acceleration: [{total_accel[0]:.3f}, {total_accel[1]:.3f}, {total_accel[2]:.3f}] m/s²")
            print(f"  Per-motor forces and accelerations:")
            for i, (force, accel, thrust_dir, adj_min, adj_max) in enumerate(zip(optimal_forces, accelerations, motor_thrust_directions, adjusted_min_forces, adjusted_max_forces)):
                at_bound = ""
                if abs(force - adj_min) < 1e-3:
                    at_bound = f" [AT MIN={adj_min:.3f}N]"
                elif abs(force - adj_max) < 1e-3:
                    at_bound = f" [AT MAX={adj_max:.3f}N]"
                print(f"    Motor {i}: force={force:.3f}N{at_bound} thrust_dir=[{thrust_dir[0]:.3f}, {thrust_dir[1]:.3f}, {thrust_dir[2]:.3f}] → accel=[{accel[0]:.3f}, {accel[1]:.3f}, {accel[2]:.3f}] m/s²")

            _visualize_airframe(motor_positions, motor_thrust_directions, motor_directions, optimal_forces, thrust_to_torque_ratio, desired_wrench, achieved_wrench, mass, (adjusted_min_forces, adjusted_max_forces))

        return True, "Valid airframe"

    except Exception as e:
        msg = f"Validation error: {str(e)}"
        if verbose:
            print(f"[Validation QP] ✗ ERROR: {msg}")
            import traceback
            traceback.print_exc()
        return False, msg

def generate_airframe_from_solution_without_usd(x: list | np.ndarray, output_json_path: str) -> dict:
    """Generate airframe properties JSON directly from parameters without USD generation.

    Args:
        x: Full parameter vector (31 normalized 0-1 values)
        output_json_path: Path where to save the JSON file

    Returns:
        Dictionary containing airframe data
    """
    import json
    import math
    import os

    if all(0 <= val <= 1 for val in x):
        params = Airframe.from_x_0_1(x).params
    else:
        params = dict(zip(PARAM_BOUNDS.keys(), x))

    arm_width = AIRFRAME_CONSTANTS['arm_width']
    arm_height = AIRFRAME_CONSTANTS['arm_height']
    arm_density = AIRFRAME_CONSTANTS['arm_density']
    motor_mass = AIRFRAME_CONSTANTS['motor_mass']
    base_mass = AIRFRAME_CONSTANTS['base_mass']
    side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']

    front_arm_len = params['front_arm_length']
    back_arm_len = params['back_arm_length']
    side_seg2_len_left = params['side_arm_seg2_length_left']
    side_seg2_len_right = params['side_arm_seg2_length_right']

    front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
    back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
    side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
    side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
    side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density

    total_mass = base_mass + 4 * motor_mass + front_arm_mass + back_arm_mass + 2 * side_seg1_mass + side_seg2_mass_left + side_seg2_mass_right

    def rpy_to_rotation_matrix(roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        return R_z @ R_y @ R_x

    motor_positions = []
    motor_thrust_directions = []

    front_prop_rot_x = math.radians(params['front_prop_rotation_x'])
    front_prop_rot_y = math.radians(params['front_prop_rotation_y'])
    front_prop_rot_z = math.radians(params['front_prop_rotation_z'])
    back_prop_rot_x = math.radians(params['back_prop_rotation_x'])
    back_prop_rot_y = math.radians(params['back_prop_rotation_y'])
    back_prop_rot_z = math.radians(params['back_prop_rotation_z'])
    side_base_rot_x_left = math.radians(params['side_base_rotation_x_left'])
    side_base_rot_y_left = math.radians(params['side_base_rotation_y_left'])
    side_base_rot_z_left = math.radians(params['side_base_rotation_z_left'])
    side_base_rot_x_right = math.radians(params['side_base_rotation_x_right'])
    side_base_rot_y_right = math.radians(params['side_base_rotation_y_right'])
    side_base_rot_z_right = math.radians(params['side_base_rotation_z_right'])
    side_motor_rot_x_left = math.radians(params['side_motor_rotation_x_left'])
    side_motor_rot_y_left = math.radians(params['side_motor_rotation_y_left'])
    side_motor_rot_z_left = math.radians(params['side_motor_rotation_z_left'])
    side_motor_rot_x_right = math.radians(params['side_motor_rotation_x_right'])
    side_motor_rot_y_right = math.radians(params['side_motor_rotation_y_right'])
    side_motor_rot_z_right = math.radians(params['side_motor_rotation_z_right'])

    motor_0_pos = np.array([0.035 + front_arm_len, 0.0, 0.0])
    motor_0_thrust = rpy_to_rotation_matrix(front_prop_rot_x, front_prop_rot_y, front_prop_rot_z) @ np.array([0, 0, 1])
    motor_positions.append(motor_0_pos.tolist())
    motor_thrust_directions.append(motor_0_thrust.tolist())

    R_base_1 = rpy_to_rotation_matrix(side_base_rot_x_left, side_base_rot_y_left, side_base_rot_z_left + np.pi/2)
    base_attachment_1 = np.array([0, 0.025, 0])
    joint_1_pos = base_attachment_1 + R_base_1 @ np.array([side_seg1_len, 0, 0])
    joint_1_axis = (R_base_1 @ np.array([0, 0, 1])).tolist()
    motor_1_pos = base_attachment_1 + R_base_1 @ np.array([side_seg1_len + side_seg2_len_left, 0, 0])
    R_motor_1 = R_base_1 @ rpy_to_rotation_matrix(side_motor_rot_x_left, side_motor_rot_y_left, side_motor_rot_z_left)
    motor_1_thrust = R_motor_1 @ np.array([0, 0, 1])
    motor_positions.append(motor_1_pos.tolist())
    motor_thrust_directions.append(motor_1_thrust.tolist())

    R_back = rpy_to_rotation_matrix(0, 0, np.pi)
    pos_back = np.array([-0.035, 0, 0]) + R_back @ np.array([back_arm_len, 0, 0])
    R_motor_2 = R_back @ rpy_to_rotation_matrix(back_prop_rot_x, back_prop_rot_y, back_prop_rot_z)
    motor_2_thrust = R_motor_2 @ np.array([0, 0, 1])
    motor_positions.append(pos_back.tolist())
    motor_thrust_directions.append(motor_2_thrust.tolist())

    R_base_3 = rpy_to_rotation_matrix(side_base_rot_x_right, side_base_rot_y_right, side_base_rot_z_right - np.pi/2)
    base_attachment_3 = np.array([0, -0.025, 0])
    joint_3_pos = base_attachment_3 + R_base_3 @ np.array([side_seg1_len, 0, 0])
    joint_3_axis = (R_base_3 @ np.array([0, 0, 1])).tolist()
    motor_3_pos = base_attachment_3 + R_base_3 @ np.array([side_seg1_len + side_seg2_len_right, 0, 0])
    R_motor_3 = R_base_3 @ rpy_to_rotation_matrix(side_motor_rot_x_right, side_motor_rot_y_right, side_motor_rot_z_right)
    motor_3_thrust = R_motor_3 @ np.array([0, 0, 1])
    motor_positions.append(motor_3_pos.tolist())
    motor_thrust_directions.append(motor_3_thrust.tolist())

    airframe_data = {
        "total_mass": total_mass,
        "num_motors": 4,
        "motor_positions_body_frame": motor_positions,
        "motor_thrust_directions": motor_thrust_directions,
        "motor_directions": AIRFRAME_CONSTANTS['motor_directions'],
        "motor_idx_list": AIRFRAME_CONSTANTS['motor_idx_list'],
        "x_0_1": list(x) if isinstance(x, np.ndarray) else x,
        "parameters": params,
        "joint_1_position": joint_1_pos.tolist(),
        "joint_3_position": joint_3_pos.tolist(),
        "joint_1_axis": joint_1_axis,
        "joint_3_axis": joint_3_axis,
        "side_seg1_length": side_seg1_len,
        "side_seg2_length_left": side_seg2_len_left,
        "side_seg2_length_right": side_seg2_len_right
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(airframe_data, f, indent=2)

    return airframe_data

def get_arm_torque_info(airframe: Airframe, verbose: bool) -> dict:
    """Get information about collision-derived arm torque responses.

    Args:
        airframe: Airframe object
        verbose: if True, print detailed info

    Returns:
        Dictionary with torque parameter info for both arms
    """
    angle_b = airframe.compute_end_angle_b()
    torque_fn_left, torque_fn_right = build_torque_response_functions(
        angle_b["end_angle_b_left"], angle_b["end_angle_b_right"]
    )
    info = {
        'start_angle_b_left': angle_b['start_angle_b_left'],
        'start_angle_b_right': angle_b['start_angle_b_right'],
        'end_angle_b_left': angle_b['end_angle_b_left'],
        'end_angle_b_right': angle_b['end_angle_b_right'],
        'torque_fn_left': torque_fn_left,
        'torque_fn_right': torque_fn_right,
    }
    if verbose:
        print(f"[Arm Torque Info]")
        print(f"  Left arm: start_angle_b={info['start_angle_b_left']:.1f}°, end_angle_b={info['end_angle_b_left']:.1f}°")
        print(f"  Right arm: start_angle_b={info['start_angle_b_right']:.1f}°, end_angle_b={info['end_angle_b_right']:.1f}°")
    return info


def build_collision_torque_response(airframe: Airframe, delta_deg: float, verbose: bool) -> dict:
    """Build torque response functions from collision-limited arm travel."""
    max_left_deg, max_right_deg, aabb_info = get_max_arm_angles_no_collision(
        airframe,
        delta_deg=delta_deg,
        verbose=verbose,
    )
    end_angle_b_left = min(max_left_deg, TORQUE_RESPONSE_MAX_ANGLE_DEG)
    end_angle_b_right = min(abs(max_right_deg), TORQUE_RESPONSE_MAX_ANGLE_DEG)
    torque_fn_left, torque_fn_right = build_torque_response_functions(end_angle_b_left, end_angle_b_right)
    aabb_info["torque_fn_left"] = torque_fn_left
    aabb_info["torque_fn_right"] = torque_fn_right
    return {
        "aabb_info": aabb_info,
        "max_left_deg": max_left_deg,
        "max_right_deg": max_right_deg,
        "end_angle_b_left": end_angle_b_left,
        "end_angle_b_right": end_angle_b_right,
        "torque_fn_left": torque_fn_left,
        "torque_fn_right": torque_fn_right,
    }

def validate_solution_hover_capability_fast(airframe: Airframe, verbose: bool, max_min_motor_force_use: tuple[float, float]) -> tuple[bool, str]:
    """Validate airframe hover capability.

    Args:
        airframe: Airframe object
        verbose: if True, print detailed validation info
        max_min_motor_force_use: (min, max) motor force usage range in [0,1]

    Returns:
        (is_valid, message): Tuple of validation result and message
    """
    total_mass = airframe.compute_total_mass()
    desired_wrench = np.array([0, 0, total_mass * 9.81, 0, 0, 0])
    return validate_airframe_force_distribution_with_qp(airframe, desired_wrench, verbose=verbose, wrench_tolerance=0.01, max_min_motor_force_use=max_min_motor_force_use)

def validate_solution_arm_deflection_fast(airframe: Airframe, verbose: bool) -> bool:
    """Check that arm parameters are consistent (max_angle_a < start_angle_b).

    Args:
        airframe: Airframe object
        verbose: if True, print detailed validation info

    Returns:
        is_valid: Boolean indicating if arm parameters are valid
    """
    return airframe.validate_arm_deflection(verbose)


def find_end_angle_b(airframe: Airframe, verbose: bool) -> dict:
    """Compute end_angle_b for both arms based on collision detection.

    Args:
        airframe: Airframe object
        verbose: if True, print detailed info

    Returns:
        Dictionary containing end_angle_b_left/right and start_angle_b_left/right
    """
    result = airframe.compute_end_angle_b()
    if verbose:
        print(f"[Find end_angle_b]")
        print(f"  Left arm: ramp_angle={TORQUE_RESPONSE_RAMP_ANGLE_DEG:.1f}°, end_angle_b={result['end_angle_b_left']:.1f}°")
        print(f"  Right arm: ramp_angle={TORQUE_RESPONSE_RAMP_ANGLE_DEG:.1f}°, end_angle_b={result['end_angle_b_right']:.1f}°")
    return result


def validate_solution_hover_capability(airframe: Airframe, verbose: bool, max_min_motor_force_use: tuple[float, float]) -> tuple[bool, str]:
    """Validate airframe hover capability.

    Args:
        airframe: Airframe object
        verbose: if True, print detailed validation info
        max_min_motor_force_use: (min, max) motor force usage range in [0,1]

    Returns:
        (is_valid, message): Tuple of validation result and message
    """
    total_mass = airframe.compute_total_mass()
    desired_wrench = np.array([0, 0, total_mass * 9.81, 0, 0, 0])
    return validate_airframe_force_distribution_with_qp(airframe, desired_wrench, verbose=verbose, wrench_tolerance=0.05, max_min_motor_force_use=max_min_motor_force_use)

def validate_airframe_hover_feasibility(airframe: Airframe, verbose: bool) -> tuple[bool, str]:
    """Validate if an airframe design can generate the required hover wrench using pseudoinverse.

    Args:
        airframe: Airframe object
        verbose: if True, print detailed validation info

    Returns:
        (is_valid, message): Tuple of validation result and message
    """
    if verbose:
        from utils import _visualize_airframe

    try:
        alloc_data = airframe.compute_allocation_matrix()
        B = alloc_data['B']
        mass = alloc_data['mass']
        motor_positions = alloc_data['motor_positions']
        motor_thrust_directions = alloc_data['motor_thrust_directions']
        motor_directions = alloc_data['motor_directions'].reshape(-1, 1)
        min_forces = alloc_data['min_forces']
        max_forces = alloc_data['max_forces']
        thrust_to_torque_ratio = alloc_data['thrust_to_torque_ratio']

        B_pinv = np.linalg.pinv(B)
        hover_wrench = np.array([0, 0, mass * 9.81, 0, 0, 0])
        desired_thrusts = B_pinv @ hover_wrench

        if np.any(np.isnan(desired_thrusts)) or np.any(np.isinf(desired_thrusts)):
            msg = "Cannot compute valid hover thrust (NaN or Inf detected)"
            if verbose:
                print(f"[Validation] ✗ INVALID: {msg}")
            return False, msg

        forces_out_of_bounds = (desired_thrusts < min_forces) | (desired_thrusts > max_forces)
        num_out_of_bounds = forces_out_of_bounds.sum()

        if num_out_of_bounds > 0:
            out_of_bounds_ratio = forces_out_of_bounds.mean()
            msg = f"{out_of_bounds_ratio*100:.1f}% of motors out of bounds"
            if verbose:
                accelerations = (desired_thrusts[:, np.newaxis] / mass) * motor_thrust_directions
                total_accel = accelerations.sum(axis=0)
                print(f"[Validation] ✗ INVALID: {msg}")
                print(f"  Required hover force: {hover_wrench[2]:.3f}N (mass={mass:.3f}kg)")
                print(f"  Motor force limits: [{min_forces.min():.3f}, {max_forces.max():.3f}]N")
                print(f"  Number of motors: {len(desired_thrusts)}")
                print(f"  Total acceleration: [{total_accel[0]:.3f}, {total_accel[1]:.3f}, {total_accel[2]:.3f}] m/s²")
                for i, (thrust, accel, thrust_dir, oob, f_min, f_max) in enumerate(zip(desired_thrusts, accelerations, motor_thrust_directions, forces_out_of_bounds, min_forces, max_forces)):
                    if oob:
                        if thrust < f_min:
                            print(f"    Motor {i}: force={thrust:.3f}N (below min by {f_min - thrust:.3f}N)")
                        else:
                            print(f"    Motor {i}: force={thrust:.3f}N (above max by {thrust - f_max:.3f}N)")
                _visualize_airframe(motor_positions, motor_thrust_directions, motor_directions, desired_thrusts, thrust_to_torque_ratio, hover_wrench, None, mass, (min_forces, max_forces))
            return False, msg

        if verbose:
            margin_below = (desired_thrusts - min_forces).min()
            margin_above = (max_forces - desired_thrusts).min()
            accelerations = (desired_thrusts[:, np.newaxis] / mass) * motor_thrust_directions
            total_accel = accelerations.sum(axis=0)
            print(f"[Validation] ✓ VALID: All motors within bounds")
            print(f"  Required hover force: {hover_wrench[2]:.3f}N (mass={mass:.3f}kg)")
            print(f"  Motor force limits: [{min_forces.min():.3f}, {max_forces.max():.3f}]N")
            print(f"  Minimum margin to lower limit: {margin_below:.3f}N")
            print(f"  Minimum margin to upper limit: {margin_above:.3f}N")
            _visualize_airframe(motor_positions, motor_thrust_directions, motor_directions, desired_thrusts, thrust_to_torque_ratio, hover_wrench, None, mass, (min_forces, max_forces))

        return True, "Valid airframe"

    except Exception as e:
        msg = f"Validation error: {str(e)}"
        if verbose:
            print(f"[Validation] ✗ ERROR: {msg}")
            import traceback
            traceback.print_exc()
        return False, msg

def repair_solution_to_track_certain_wrench_list_casadi(airframe: Airframe, wrench_list: list, gravity_offset: bool, tolerance: float, verbose: bool, max_min_motor_force_use_list: list, ipopt_options: dict) -> tuple[Airframe, float, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """CasADi/IPOPT version using scipy.optimize.minimize with IPOPT interface.

    Uses IPOPT solver through scipy's minimize interface for potentially faster optimization.

    Args:
        airframe: Airframe object to repair
        wrench_list: List of 6D wrenches [Fx, Fy, Fz, Tx, Ty, Tz] to track
        gravity_offset: If True, add mass*g to Fz component of each wrench
        tolerance: Maximum allowed wrench error for non-None components
        verbose: If True, display detailed info
        max_min_motor_force_use_list: List of (min, max) motor force usage ranges
        ipopt_options: Dict of IPOPT solver options

    Returns:
        Same as repair_solution_to_track_certain_wrench_list
    """
    from scipy.optimize import minimize

    x_0_1 = airframe.x_0_1
    n_wrenches = len(wrench_list)

    processed_wrenches = []
    wrench_masks = []
    for i, w in enumerate(wrench_list):
        w_arr = np.array([float(x) if x is not None else np.nan for x in w])
        mask = ~np.isnan(w_arr)
        w_arr = np.nan_to_num(w_arr, nan=0.0)
        processed_wrenches.append(w_arr)
        wrench_masks.append(mask)

    original_data = airframe.compute_allocation_matrix()
    mass_orig = original_data['mass']
    min_forces_orig = original_data['min_forces']
    max_forces_orig = original_data['max_forces']
    n_params = len(x_0_1)
    n_motors = len(min_forces_orig)

    B_orig = original_data['B']
    achieved_wrenches_orig = []
    u_init_list = []

    for i in range(n_wrenches):
        w_target = processed_wrenches[i].copy()
        if gravity_offset and wrench_masks[i][2]:
            w_target[2] += mass_orig * 9.81

        min_use, max_use = max_min_motor_force_use_list[i]
        adj_min = min_use * (max_forces_orig - min_forces_orig) + min_forces_orig
        adj_max = max_use * (max_forces_orig - min_forces_orig) + min_forces_orig
        u_opt = _solve_bounded_least_squares(B_orig, w_target, adj_min, adj_max)
        u_norm = (u_opt - adj_min) / (adj_max - adj_min + 1e-10)
        u_init_list.append(np.clip(u_norm, 0, 1))
        achieved_wrenches_orig.append(B_orig @ u_opt)

    z_init = np.concatenate([np.zeros(n_params)] + u_init_list)

    def objective(z):
        delta = z[:n_params]
        return np.linalg.norm(delta)

    def constraints_func(z):
        delta = z[:n_params]
        x_new = np.clip(x_0_1 + delta, 0, 1)
        data = Airframe.from_x_0_1(x_new).compute_allocation_matrix()
        B = data['B']
        mass = data['mass']
        min_f, max_f = data['min_forces'], data['max_forces']

        constraint_vals = []
        for i in range(n_wrenches):
            u_norm = z[n_params + i * n_motors : n_params + (i + 1) * n_motors]
            min_use, max_use = max_min_motor_force_use_list[i]
            adj_min = min_use * (max_f - min_f) + min_f
            adj_max = max_use * (max_f - min_f) + min_f
            u = adj_min + u_norm * (adj_max - adj_min)

            w_target = processed_wrenches[i].copy()
            if gravity_offset and wrench_masks[i][2]:
                w_target[2] += mass * 9.81

            achieved = B @ u
            error_vec = (achieved - w_target)[wrench_masks[i]]
            error_norm = np.linalg.norm(error_vec)
            constraint_vals.append(tolerance - error_norm)

        return np.array(constraint_vals)

    bounds = [(-x_0_1[i], 1 - x_0_1[i]) for i in range(n_params)]
    bounds += [(0, 1)] * (n_wrenches * n_motors)

    constraints = {'type': 'ineq', 'fun': constraints_func}

    if ipopt_options is None:
        ipopt_options = {
            'maxiter': 1000,
            'disp': verbose,
            'ftol': tolerance * 0.01,
        }

    result = minimize(objective, z_init, method='SLSQP', bounds=bounds,
                     constraints=constraints, options=ipopt_options)

    z_final = result.x
    delta_final = z_final[:n_params]
    repaired_airframe = Airframe.from_x_0_1(np.clip(x_0_1 + delta_final, 0, 1))

    repaired_data = repaired_airframe.compute_allocation_matrix()
    B_repaired = repaired_data['B']
    min_forces_repaired = repaired_data['min_forces']
    max_forces_repaired = repaired_data['max_forces']

    motor_thrusts_list = []
    achieved_wrenches_repaired = []

    for i in range(n_wrenches):
        u_norm = z_final[n_params + i * n_motors : n_params + (i + 1) * n_motors]
        min_use, max_use = max_min_motor_force_use_list[i]
        adj_min = min_use * (max_forces_repaired - min_forces_repaired) + min_forces_repaired
        adj_max = max_use * (max_forces_repaired - min_forces_repaired) + min_forces_repaired
        u_final = adj_min + u_norm * (adj_max - adj_min)
        motor_thrusts_list.append(u_final)
        achieved_wrenches_repaired.append(B_repaired @ u_final)

    l2_norm_difference = np.linalg.norm(delta_final)

    return repaired_airframe, l2_norm_difference, motor_thrusts_list, achieved_wrenches_repaired, achieved_wrenches_orig


def repair_solution_to_track_certain_wrench_list_casadi_analytical(airframe: Airframe, wrench_list: list, gravity_offset: bool, tolerance: float, verbose: bool, max_min_motor_force_use_list: list, ipopt_options: dict, minimum_arm_torque: float) -> tuple[Airframe, float, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """CasADi version with analytical gradients using symbolic allocation matrix.

    Uses CasADi symbolic variables to compute the allocation matrix and its gradients
    analytically, then solves with IPOPT for efficient constrained optimization.

    Args:
        airframe: Airframe object to repair
        wrench_list: List of 6D wrenches [Fx, Fy, Fz, Tx, Ty, Tz] to track
        gravity_offset: If True, add mass*g to Fz component of each wrench
        tolerance: Maximum allowed wrench error for non-None components
        verbose: If True, display detailed info
        max_min_motor_force_use_list: List of (min, max) motor force usage ranges
        ipopt_options: Dict of IPOPT solver options
        minimum_arm_torque: Minimum torque (Nm) that side motors must produce on their
            revolute joints at 100% thrust. Left motor must produce <= -minimum_arm_torque,
            right motor must produce >= minimum_arm_torque.

    Returns:
        Same as repair_solution_to_track_certain_wrench_list
    """
    import casadi as ca

    param_bounds = list(PARAM_BOUNDS.values())

    arm_width = AIRFRAME_CONSTANTS['arm_width']
    arm_height = AIRFRAME_CONSTANTS['arm_height']
    arm_density = AIRFRAME_CONSTANTS['arm_density']
    motor_mass = AIRFRAME_CONSTANTS['motor_mass']
    base_mass = AIRFRAME_CONSTANTS['base_mass']
    side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
    motor_directions = AIRFRAME_CONSTANTS['motor_directions']
    thrust_coefficient = AIRFRAME_CONSTANTS['thrust_coefficient']
    max_rps = AIRFRAME_CONSTANTS['max_rps']
    min_rps = AIRFRAME_CONSTANTS['min_rps']
    cq = AIRFRAME_CONSTANTS['cq']
    thrust_to_torque_ratio = cq
    max_force = thrust_coefficient * max_rps ** 2
    min_force = thrust_coefficient * min_rps ** 2


    n_motors = 4
    g = 9.81

    def rpy_to_rotation_matrix_casadi(roll, pitch, yaw):
        cr, sr = ca.cos(roll), ca.sin(roll)
        cp, sp = ca.cos(pitch), ca.sin(pitch)
        cy, sy = ca.cos(yaw), ca.sin(yaw)
        R_x = ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, cr, -sr), ca.horzcat(0, sr, cr))
        R_y = ca.vertcat(ca.horzcat(cp, 0, sp), ca.horzcat(0, 1, 0), ca.horzcat(-sp, 0, cp))
        R_z = ca.vertcat(ca.horzcat(cy, -sy, 0), ca.horzcat(sy, cy, 0), ca.horzcat(0, 0, 1))
        return R_z @ R_y @ R_x

    def compute_allocation_matrix_and_mass_casadi(x_0_1_sym):
        params = [param_bounds[i][0] + x_0_1_sym[i] * (param_bounds[i][1] - param_bounds[i][0]) for i in range(len(param_bounds))]
        front_arm_len = params[0]
        front_prop_rot = [params[i] * ca.pi / 180 for i in [1, 2, 3]]
        back_arm_len = params[4]
        back_prop_rot = [params[i] * ca.pi / 180 for i in [5, 6, 7]]
        side_seg2_len_left = params[8]
        side_seg2_len_right = params[9]
        side_base_rot_left = [params[i] * ca.pi / 180 for i in [10, 11, 12]]
        side_base_rot_right = [params[i] * ca.pi / 180 for i in [13, 14, 15]]
        side_motor_rot_left = [params[i] * ca.pi / 180 for i in [16, 17, 18]]
        side_motor_rot_right = [params[i] * ca.pi / 180 for i in [19, 20, 21]]

        front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
        back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
        side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
        side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
        side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density
        total_mass = base_mass + 4 * motor_mass + front_arm_mass + back_arm_mass + 2 * side_seg1_mass + side_seg2_mass_left + side_seg2_mass_right

        motor_0_pos = ca.vertcat(0.035 + front_arm_len, 0.0, 0.0)
        R_front = rpy_to_rotation_matrix_casadi(front_prop_rot[0], front_prop_rot[1], front_prop_rot[2])
        motor_0_thrust = R_front @ ca.vertcat(0, 0, 1)

        R_base_1 = rpy_to_rotation_matrix_casadi(side_base_rot_left[0], side_base_rot_left[1], side_base_rot_left[2] + ca.pi/2)
        base_1 = ca.vertcat(0, 0.025, 0)
        joint_1_pos = base_1 + R_base_1 @ ca.vertcat(side_seg1_len, 0, 0)
        joint_1_axis = R_base_1 @ ca.vertcat(0, 0, 1)
        motor_1_pos = base_1 + R_base_1 @ ca.vertcat(side_seg1_len + side_seg2_len_left, 0, 0)
        R_motor_1 = R_base_1 @ rpy_to_rotation_matrix_casadi(side_motor_rot_left[0], side_motor_rot_left[1], side_motor_rot_left[2])
        motor_1_thrust = R_motor_1 @ ca.vertcat(0, 0, 1)

        R_back = rpy_to_rotation_matrix_casadi(0, 0, ca.pi)
        motor_2_pos = ca.vertcat(-0.035, 0, 0) + R_back @ ca.vertcat(back_arm_len, 0, 0)
        R_motor_2 = R_back @ rpy_to_rotation_matrix_casadi(back_prop_rot[0], back_prop_rot[1], back_prop_rot[2])
        motor_2_thrust = R_motor_2 @ ca.vertcat(0, 0, 1)

        R_base_3 = rpy_to_rotation_matrix_casadi(side_base_rot_right[0], side_base_rot_right[1], side_base_rot_right[2] - ca.pi/2)
        base_3 = ca.vertcat(0, -0.025, 0)
        joint_3_pos = base_3 + R_base_3 @ ca.vertcat(side_seg1_len, 0, 0)
        joint_3_axis = R_base_3 @ ca.vertcat(0, 0, 1)
        motor_3_pos = base_3 + R_base_3 @ ca.vertcat(side_seg1_len + side_seg2_len_right, 0, 0)
        R_motor_3 = R_base_3 @ rpy_to_rotation_matrix_casadi(side_motor_rot_right[0], side_motor_rot_right[1], side_motor_rot_right[2])
        motor_3_thrust = R_motor_3 @ ca.vertcat(0, 0, 1)

        motor_positions = [motor_0_pos, motor_1_pos, motor_2_pos, motor_3_pos]
        motor_thrusts = [motor_0_thrust, motor_1_thrust, motor_2_thrust, motor_3_thrust]

        B = ca.MX.zeros(6, n_motors)
        for i in range(n_motors):
            r = motor_positions[i]
            thrust_vec = motor_thrusts[i]
            torque_contrib = ca.cross(r, thrust_vec)
            yaw_torque = motor_directions[i] * thrust_to_torque_ratio * thrust_vec
            B[0:3, i] = thrust_vec
            B[3:6, i] = torque_contrib + yaw_torque

        joint_info = {
            'joint_1_pos': joint_1_pos, 'joint_1_axis': joint_1_axis,
            'joint_3_pos': joint_3_pos, 'joint_3_axis': joint_3_axis,
            'motor_1_pos': motor_1_pos, 'motor_1_thrust': motor_1_thrust,
            'motor_3_pos': motor_3_pos, 'motor_3_thrust': motor_3_thrust,
        }
        return B, total_mass, joint_info

    x_0_1 = airframe.x_0_1
    n_wrenches = len(wrench_list)
    n_params = len(x_0_1)

    processed_wrenches, wrench_masks = [], []
    for w in wrench_list:
        w_arr = np.array([float(x) if x is not None else np.nan for x in w])
        mask = ~np.isnan(w_arr)
        w_arr = np.nan_to_num(w_arr, nan=0.0)
        processed_wrenches.append(w_arr)
        wrench_masks.append(mask)

    original_data = airframe.compute_allocation_matrix()
    mass_orig = original_data['mass']
    min_forces_orig, max_forces_orig = original_data['min_forces'], original_data['max_forces']
    B_orig = original_data['B']

    achieved_wrenches_orig, u_init_list = [], []
    for i in range(n_wrenches):
        w_target = processed_wrenches[i].copy()
        if gravity_offset and wrench_masks[i][2]:
            w_target[2] += mass_orig * g
        min_use, max_use = max_min_motor_force_use_list[i]
        adj_min = min_use * (max_forces_orig - min_forces_orig) + min_forces_orig
        adj_max = max_use * (max_forces_orig - min_forces_orig) + min_forces_orig
        u_opt = _solve_bounded_least_squares(B_orig, w_target, adj_min, adj_max)
        u_norm = (u_opt - adj_min) / (adj_max - adj_min + 1e-10)
        u_init_list.append(np.clip(u_norm, 0, 1))
        achieved_wrenches_orig.append(B_orig @ u_opt)

    delta = ca.MX.sym('delta', n_params)
    u_vars = [ca.MX.sym(f'u_{i}', n_motors) for i in range(n_wrenches)]

    x_new = ca.DM(x_0_1) + delta
    B_sym, mass_sym, joint_info = compute_allocation_matrix_and_mass_casadi(x_new)

    min_f_np, max_f_np = np.array([min_force] * n_motors), np.array([max_force] * n_motors)
    objective = ca.sqrt(ca.dot(delta, delta) + 1e-12)

    constraints, lbg, ubg = [], [], []
    for i in range(n_wrenches):
        u_norm = u_vars[i]
        min_use, max_use = max_min_motor_force_use_list[i]
        adj_min = min_use * (max_f_np - min_f_np) + min_f_np
        adj_max = max_use * (max_f_np - min_f_np) + min_f_np
        u = ca.DM(adj_min) + u_norm * ca.DM(adj_max - adj_min)

        w_base = processed_wrenches[i].copy()
        if gravity_offset and wrench_masks[i][2]:
            w_target = ca.vertcat(w_base[0], w_base[1], w_base[2] + mass_sym * g, w_base[3], w_base[4], w_base[5])
        else:
            w_target = ca.DM(w_base)

        achieved = B_sym @ u
        error_vec = achieved - w_target
        mask_indices = np.where(wrench_masks[i])[0]
        for j in mask_indices:
            constraints.append(error_vec[j])
            lbg.append(-tolerance)
            ubg.append(tolerance)

    if minimum_arm_torque > 0:
        r_1_from_joint = joint_info['motor_1_pos'] - joint_info['joint_1_pos']
        thrust_force_1 = max_force * joint_info['motor_1_thrust']
        torque_thrust_1 = ca.dot(ca.cross(r_1_from_joint, thrust_force_1), joint_info['joint_1_axis'])
        counter_torque_1 = motor_directions[1] * thrust_to_torque_ratio * max_force * ca.dot(joint_info['motor_1_thrust'], joint_info['joint_1_axis'])
        total_torque_1 = torque_thrust_1 + counter_torque_1
        constraints.append(total_torque_1 + minimum_arm_torque)
        lbg.append(-ca.inf)
        ubg.append(0)

        r_3_from_joint = joint_info['motor_3_pos'] - joint_info['joint_3_pos']
        thrust_force_3 = max_force * joint_info['motor_3_thrust']
        torque_thrust_3 = ca.dot(ca.cross(r_3_from_joint, thrust_force_3), joint_info['joint_3_axis'])
        counter_torque_3 = motor_directions[3] * thrust_to_torque_ratio * max_force * ca.dot(joint_info['motor_3_thrust'], joint_info['joint_3_axis'])
        total_torque_3 = torque_thrust_3 + counter_torque_3
        constraints.append(minimum_arm_torque - total_torque_3)
        lbg.append(-ca.inf)
        ubg.append(0)

    all_vars = ca.vertcat(delta, *u_vars)
    all_constraints = ca.vertcat(*constraints)

    lbx = list(-x_0_1) + [0] * (n_wrenches * n_motors)
    ubx = list(1 - x_0_1) + [1] * (n_wrenches * n_motors)
    x_init = np.concatenate([np.zeros(n_params)] + u_init_list)

    nlp = {'x': all_vars, 'f': objective, 'g': all_constraints}
    opts = {
        'ipopt.print_level': 5 if verbose else 0,
        'print_time': verbose,
        'ipopt.max_iter': 3000,
        'ipopt.tol': 1e-12,
        'ipopt.constr_viol_tol': 1e-14,
        'ipopt.acceptable_tol': 1e-10,
        'ipopt.acceptable_constr_viol_tol': 1e-12,
    }
    if ipopt_options:
        for k, v in ipopt_options.items():
            opts[f'ipopt.{k}'] = v

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    solver_stats = solver.stats()
    return_status = solver_stats.get('return_status', 'unknown')

    z_final = np.array(sol['x']).flatten()
    delta_final = z_final[:n_params]
    g_val = np.array(sol['g']).flatten()

    violated_constraints = []
    for i, (g, lb, ub) in enumerate(zip(g_val, lbg, ubg)):
        lb_val = float(lb) if lb != -ca.inf else -np.inf
        ub_val = float(ub) if ub != ca.inf else np.inf
        if g < lb_val - 1e-6 or g > ub_val + 1e-6:
            violated_constraints.append((i, g, lb_val, ub_val))

    if verbose:
        print(f"[CasADi] Solver status: {return_status}")
        print(f"[CasADi] Constraint values: {g_val}")
        print(f"[CasADi] Violated constraints: {len(violated_constraints)}")

    if return_status in ['Infeasible_Problem_Detected', 'Invalid_Problem_Definition'] or violated_constraints:
        n_wrench_constraints = sum(len(np.where(m)[0]) for m in wrench_masks)
        error_lines = [f"Repair optimization failed: {return_status}"]
        error_lines.append(f"Violated {len(violated_constraints)} of {len(g_val)} constraints:")
        for idx, g, lb, ub in violated_constraints[:10]:
            if idx < n_wrench_constraints:
                wrench_idx, component_idx = 0, idx
                for wi, m in enumerate(wrench_masks):
                    n_comp = len(np.where(m)[0])
                    if component_idx < n_comp:
                        wrench_idx = wi
                        actual_comp = np.where(m)[0][component_idx]
                        break
                    component_idx -= n_comp
                comp_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
                error_lines.append(f"  Wrench {wrench_idx} {comp_names[actual_comp]}: value={g:.6f}, bounds=[{lb:.6f}, {ub:.6f}]")
            else:
                arm_idx = idx - n_wrench_constraints
                side = 'left' if arm_idx == 0 else 'right'
                error_lines.append(f"  Arm torque {side}: value={g:.6f}, bounds=[{lb:.6f}, {ub:.6f}]")
        if len(violated_constraints) > 10:
            error_lines.append(f"  ... and {len(violated_constraints) - 10} more")
        raise RuntimeError('\n'.join(error_lines))

    repaired_airframe = Airframe.from_x_0_1(np.clip(x_0_1 + delta_final, 0, 1))

    repaired_data = repaired_airframe.compute_allocation_matrix()
    B_repaired = repaired_data['B']
    min_forces_repaired, max_forces_repaired = repaired_data['min_forces'], repaired_data['max_forces']

    motor_thrusts_list, achieved_wrenches_repaired = [], []
    for i in range(n_wrenches):
        u_norm = z_final[n_params + i * n_motors : n_params + (i + 1) * n_motors]
        min_use, max_use = max_min_motor_force_use_list[i]
        adj_min = min_use * (max_forces_repaired - min_forces_repaired) + min_forces_repaired
        adj_max = max_use * (max_forces_repaired - min_forces_repaired) + min_forces_repaired
        u_final = adj_min + u_norm * (adj_max - adj_min)
        motor_thrusts_list.append(u_final)
        achieved_wrenches_repaired.append(B_repaired @ u_final)

    l2_norm_difference = np.linalg.norm(delta_final)
    return repaired_airframe, l2_norm_difference, motor_thrusts_list, achieved_wrenches_repaired, achieved_wrenches_orig


def _solve_bounded_least_squares(B: np.ndarray, target: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Solve bounded least squares: min ||B @ u - target||^2 s.t. lb <= u <= ub."""
    from scipy.optimize import lsq_linear
    result = lsq_linear(B, target, bounds=(lb, ub), method='bvls')
    return result.x

def repair_solution_force_torque(airframe: Airframe, verbose: bool, tolerance: float, acceptable_authority_loss_vs_planar: float, safety_margin: float, minimum_arm_torque: float) -> Airframe:
    """Repair a solution to maintain force/torque capability by applying wrench tracking repair.

    Args:
        airframe: Airframe object
        verbose: Whether to print repair details
        tolerance: Tolerance for wrench tracking repair
        acceptable_authority_loss_vs_planar: Acceptable authority loss compared to planar
        safety_margin: Safety margin for torque parameter computation
        minimum_arm_torque: Minimum torque (Nm) that side motors must produce on their
            revolute joints at 100% thrust. Left motor must produce <= -minimum_arm_torque,
            right motor must produce >= minimum_arm_torque.

    Returns:
        Repaired Airframe with torque_params set
    """
    gravity_force    = 2.931
    max_lift_planar  = 8.233
    max_roll_planar  = 0.1948
    max_pitch_planar = 0.1814
    max_yaw_planar   = 0.0394

    s = 1.0 - acceptable_authority_loss_vs_planar
    wrench_list = [
        [   0,    0, max_lift_planar*s,                   0,                   0,                0],
        [None, None,   gravity_force*s,   max_roll_planar*s,                   0,                0],
        [None, None,   gravity_force*s,  -max_roll_planar*s,                   0,                0],
        [None, None,   gravity_force*s,                   0,  max_pitch_planar*s,                0],
        [None, None,   gravity_force*s,                   0, -max_pitch_planar*s,                0],
        [None, None,              None,                   0,                   0, max_yaw_planar*s],
        [None, None,              None,                   0,                   0,-max_yaw_planar*s],
    ]

    motor_limits_list = [(0.0, 1.0)] * 7
    repaired_airframe, _, _, _, _ = repair_solution_to_track_certain_wrench_list_casadi_analytical(
        airframe=airframe, wrench_list=wrench_list, gravity_offset=False,
        tolerance=tolerance, verbose=verbose, max_min_motor_force_use_list=motor_limits_list,
        ipopt_options={}, minimum_arm_torque=minimum_arm_torque
    )
    return repaired_airframe


def analyze_airframe_actuation(airframe: Airframe, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Analyze actuation capabilities of an airframe design.

    Computes:
    1. Max actuation capability in each axis (Fx, Fy, Fz, Tx, Ty, Tz)
    2. Coupling matrix between axes (6x6)

    Args:
        airframe: Airframe object
        verbose: If True, display visualization

    Returns:
        Tuple of (max_actuation_6d, coupling_6x6)
    """
    from scipy.optimize import linprog

    alloc_data = airframe.compute_allocation_matrix()
    B = alloc_data['B']
    min_forces = alloc_data['min_forces']
    max_forces = alloc_data['max_forces']
    n_motors = B.shape[1]

    max_actuation = np.zeros(6)
    optimal_forces = np.zeros((6, n_motors))
    axis_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

    for axis in range(6):
        c = -B[axis, :]
        bounds = [(min_forces[i], max_forces[i]) for i in range(n_motors)]
        result = linprog(c, bounds=bounds, method='highs')
        if result.success:
            max_actuation[axis] = -result.fun
            optimal_forces[axis, :] = result.x
        else:
            result_neg = linprog(-c, bounds=bounds, method='highs')
            if result_neg.success:
                max_actuation[axis] = max(0, result_neg.fun)
                optimal_forces[axis, :] = result_neg.x

    coupling = np.zeros((6, 6))
    for axis in range(6):
        achieved_wrench = B @ optimal_forces[axis, :]
        for j in range(6):
            if axis != j:
                coupling[axis, j] = achieved_wrench[j] / max(max_actuation[axis], 1e-6) if max_actuation[axis] > 1e-6 else 0

    if verbose:
        _plot_actuation_analysis(max_actuation, coupling, axis_labels)

    return max_actuation, coupling


def _plot_actuation_analysis(max_actuation: np.ndarray, coupling: np.ndarray, axis_labels: list):
    """Plot actuation capability and coupling visualization."""
    import matplotlib.pyplot as plt

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hexagon_angles = np.array([150, 30, 90, 210, 330, 270]) * np.pi / 180
    radius = 1.5
    positions = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in hexagon_angles])

    min_circle, max_circle = 0.15, 0.6
    sizes = np.zeros(6)
    max_force = max_actuation[:3].max()
    max_torque = max_actuation[3:].max()
    if max_force > 0:
        sizes[:3] = min_circle + (max_actuation[:3] / max_force) * (max_circle - min_circle)
    if max_torque > 0:
        sizes[3:] = min_circle + (max_actuation[3:] / max_torque) * (max_circle - min_circle)

    for i in range(6):
        circle = plt.Circle(positions[i], sizes[i], fill=False, ec='darkblue', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(positions[i][0], positions[i][1], f'{axis_labels[i]}\n{max_actuation[i]:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')

    max_coupling = np.abs(coupling).max()
    if max_coupling > 0:
        for i in range(6):
            for j in range(i + 1, 6):
                coupling_val = (coupling[i, j] + coupling[j, i]) / 2
                if abs(coupling_val) > 0.01:
                    linewidth = 1 + 4 * abs(coupling_val) / max_coupling
                    color = 'red' if coupling_val < 0 else 'blue'
                    ax1.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], color=color, linewidth=linewidth, alpha=0.6)

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('Actuation Capability & Coupling\n(circle size = max force/torque, line = coupling)')
    ax1.text(0, -2.2, 'Blue lines: positive coupling | Red lines: negative coupling', ha='center', fontsize=9)
    ax1.axis('off')

    im = ax2.imshow(coupling, cmap='RdBu', vmin=-np.abs(coupling).max(), vmax=np.abs(coupling).max())
    ax2.set_xticks(range(6))
    ax2.set_yticks(range(6))
    ax2.set_xticklabels(axis_labels)
    ax2.set_yticklabels(axis_labels)
    ax2.set_xlabel('Affected axis')
    ax2.set_ylabel('Actuated axis')
    ax2.set_title('Coupling Matrix\n(when maximizing row axis, column shows side effect)')
    for i in range(6):
        for j in range(6):
            ax2.text(j, i, f'{coupling[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if abs(coupling[i, j]) > np.abs(coupling).max() * 0.5 else 'black')
    plt.colorbar(im, ax=ax2, label='Coupling coefficient')

    plt.tight_layout()
    plt.show()




def analyze_actuation_while_counteracting_gravity(airframe: Airframe) -> tuple[np.ndarray, np.ndarray, float]:
    """Analyze max actuation while counteracting gravity (other axes can be any value).

    For each axis, finds max/min values subject to Fz >= mass * g.

    Args:
        airframe: Airframe object

    Returns:
        Tuple of (max_actuation_6d, min_actuation_6d, gravity_force)
    """
    from scipy.optimize import linprog

    alloc_data = airframe.compute_allocation_matrix()
    B = alloc_data['B']
    mass = alloc_data['mass']
    min_forces = alloc_data['min_forces']
    max_forces = alloc_data['max_forces']
    n_motors = B.shape[1]

    gravity_force = mass * 9.81
    max_actuation = np.zeros(6)
    min_actuation = np.zeros(6)
    bounds = [(min_forces[i], max_forces[i]) for i in range(n_motors)]

    for axis in range(6):
        A_ub = -B[2, :].reshape(1, -1)
        b_ub = np.array([-gravity_force])
        result_max = linprog(-B[axis, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        max_actuation[axis] = -result_max.fun if result_max.success else np.nan

        result_min = linprog(B[axis, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        min_actuation[axis] = result_min.fun if result_min.success else np.nan

    return max_actuation, min_actuation, gravity_force


def analyze_actuation_decoupled_while_counteracting_gravity(airframe: Airframe, tolerance_force: float, tolerance_torque: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Analyze max actuation while counteracting gravity with other axes within tolerance.

    Args:
        airframe: Airframe object
        tolerance_force: Maximum allowed deviation for force axes (Fx, Fy, Fz) in N
        tolerance_torque: Maximum allowed deviation for torque axes (Tx, Ty, Tz) in N·m

    Returns:
        Tuple of (max_actuation_6d, min_actuation_6d, gravity_force)
    """
    from scipy.optimize import linprog

    alloc_data = airframe.compute_allocation_matrix()
    B = alloc_data['B']
    mass = alloc_data['mass']
    min_forces = alloc_data['min_forces']
    max_forces = alloc_data['max_forces']
    n_motors = B.shape[1]

    gravity_force = mass * 9.81
    tolerances = np.array([tolerance_force, tolerance_force, tolerance_force, tolerance_torque, tolerance_torque, tolerance_torque])
    max_actuation = np.zeros(6)
    min_actuation = np.zeros(6)
    bounds = [(min_forces[i], max_forces[i]) for i in range(n_motors)]

    for axis in range(6):
        other_axes = [i for i in range(6) if i != axis]
        target = np.zeros(5)
        if axis != 2:
            target[other_axes.index(2)] = gravity_force
        tols = tolerances[other_axes]
        A_ub = np.vstack([B[other_axes, :], -B[other_axes, :]])
        b_ub = np.hstack([target + tols, -target + tols])

        result_max = linprog(-B[axis, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        max_actuation[axis] = -result_max.fun if result_max.success else np.nan

        result_min = linprog(B[axis, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        min_actuation[axis] = result_min.fun if result_min.success else np.nan

    return max_actuation, min_actuation, gravity_force


def analyze_actuation(airframe: Airframe, counteract_gravity: bool, decoupled: bool, tolerance_force: float, tolerance_torque: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Unified function to analyze actuation capabilities with various constraints.

    Args:
        airframe: Airframe object
        counteract_gravity: If True, add constraint Fz >= gravity (or Fz ≈ gravity if decoupled)
        decoupled: If True, constrain other axes to be within tolerance of 0 (or gravity for Fz)
        tolerance_force: Maximum allowed deviation for force axes (N)
        tolerance_torque: Maximum allowed deviation for torque axes (N·m)

    Returns:
        Tuple of (max_actuation_6d, min_actuation_6d, gravity_force)
    """
    from scipy.optimize import linprog

    alloc_data = airframe.compute_allocation_matrix()
    B = alloc_data['B']
    mass = alloc_data['mass']
    min_forces = alloc_data['min_forces']
    max_forces = alloc_data['max_forces']
    n_motors = B.shape[1]

    gravity_force = mass * 9.81
    tolerances = np.array([tolerance_force, tolerance_force, tolerance_force, tolerance_torque, tolerance_torque, tolerance_torque])
    max_actuation = np.zeros(6)
    min_actuation = np.zeros(6)
    bounds = [(min_forces[i], max_forces[i]) for i in range(n_motors)]

    for axis in range(6):
        A_ub_list, b_ub_list = [], []

        if decoupled:
            other_axes = [i for i in range(6) if i != axis]
            target = np.zeros(5)
            if counteract_gravity and axis != 2:
                target[other_axes.index(2)] = gravity_force
            tols = tolerances[other_axes]
            A_ub_list.append(B[other_axes, :])
            A_ub_list.append(-B[other_axes, :])
            b_ub_list.append(target + tols)
            b_ub_list.append(-target + tols)
        elif counteract_gravity and axis != 2:
            A_ub_list.append(-B[2, :].reshape(1, -1))
            b_ub_list.append(np.array([-gravity_force]))

        if A_ub_list:
            A_ub = np.vstack(A_ub_list)
            b_ub = np.hstack(b_ub_list)
            result_max = linprog(-B[axis, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            result_min = linprog(B[axis, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        else:
            result_max = linprog(-B[axis, :], bounds=bounds, method='highs')
            result_min = linprog(B[axis, :], bounds=bounds, method='highs')

        max_actuation[axis] = -result_max.fun if result_max.success else np.nan
        min_actuation[axis] = result_min.fun if result_min.success else np.nan

    return max_actuation, min_actuation, gravity_force


def compute_inertia_tensor(airframe: Airframe) -> np.ndarray:
    """Compute the 3x3 inertia tensor of the airframe about its center of mass.

    Uses box approximations for arms and point masses for motors.
    The inertia is computed in the body frame with origin at COM.

    Returns:
        (3, 3) inertia tensor in kg·m²
    """
    params = airframe.params
    arm_width = AIRFRAME_CONSTANTS['arm_width']
    arm_height = AIRFRAME_CONSTANTS['arm_height']
    arm_density = AIRFRAME_CONSTANTS['arm_density']
    motor_mass = AIRFRAME_CONSTANTS['motor_mass']
    base_mass = AIRFRAME_CONSTANTS['base_mass']
    base_dims = AIRFRAME_CONSTANTS['base_box_dimensions']
    side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']

    front_arm_len = params['front_arm_length']
    back_arm_len = params['back_arm_length']
    side_seg2_len_left = params['side_arm_seg2_length_left']
    side_seg2_len_right = params['side_arm_seg2_length_right']

    front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
    back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
    side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
    side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
    side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density

    I_total = np.zeros((3, 3))

    def box_inertia(m, lx, ly, lz):
        return np.diag([m/12 * (ly**2 + lz**2), m/12 * (lx**2 + lz**2), m/12 * (lx**2 + ly**2)])

    def parallel_axis(I_com, m, r):
        r = np.array(r)
        return I_com + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    I_total += box_inertia(base_mass, base_dims[0], base_dims[1], base_dims[2])

    front_arm_com = np.array([0.035 + front_arm_len/2, 0, 0])
    I_total += parallel_axis(box_inertia(front_arm_mass, front_arm_len, arm_width, arm_height), front_arm_mass, front_arm_com)
    I_total += parallel_axis(np.zeros((3,3)), motor_mass, np.array([0.035 + front_arm_len, 0, 0]))

    back_arm_com = np.array([-0.035 - back_arm_len/2, 0, 0])
    I_total += parallel_axis(box_inertia(back_arm_mass, back_arm_len, arm_width, arm_height), back_arm_mass, back_arm_com)
    I_total += parallel_axis(np.zeros((3,3)), motor_mass, np.array([-0.035 - back_arm_len, 0, 0]))

    left_seg1_com = np.array([0, 0.025 + side_seg1_len/2, 0])
    I_total += parallel_axis(box_inertia(side_seg1_mass, arm_width, side_seg1_len, arm_height), side_seg1_mass, left_seg1_com)
    left_seg2_com = np.array([0, 0.025 + side_seg1_len + side_seg2_len_left/2, 0])
    I_total += parallel_axis(box_inertia(side_seg2_mass_left, arm_width, side_seg2_len_left, arm_height), side_seg2_mass_left, left_seg2_com)
    I_total += parallel_axis(np.zeros((3,3)), motor_mass, np.array([0, 0.025 + side_seg1_len + side_seg2_len_left, 0]))

    right_seg1_com = np.array([0, -0.025 - side_seg1_len/2, 0])
    I_total += parallel_axis(box_inertia(side_seg1_mass, arm_width, side_seg1_len, arm_height), side_seg1_mass, right_seg1_com)
    right_seg2_com = np.array([0, -0.025 - side_seg1_len - side_seg2_len_right/2, 0])
    I_total += parallel_axis(box_inertia(side_seg2_mass_right, arm_width, side_seg2_len_right, arm_height), side_seg2_mass_right, right_seg2_com)
    I_total += parallel_axis(np.zeros((3,3)), motor_mass, np.array([0, -0.025 - side_seg1_len - side_seg2_len_right, 0]))

    return I_total

def _compute_inertia_tensor_casadi(x_0_1_sym):
    """Compute symbolic inertia tensor diagonal elements as a function of design parameters (CasADi).

    Returns:
        Tuple of (Ixx, Iyy, Izz) CasADi symbolic expressions
    """
    param_bounds = list(PARAM_BOUNDS.values())

    arm_width = AIRFRAME_CONSTANTS['arm_width']
    arm_height = AIRFRAME_CONSTANTS['arm_height']
    arm_density = AIRFRAME_CONSTANTS['arm_density']
    motor_mass = AIRFRAME_CONSTANTS['motor_mass']
    base_mass = AIRFRAME_CONSTANTS['base_mass']
    base_dims = AIRFRAME_CONSTANTS['base_box_dimensions']
    side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']

    front_arm_len = param_bounds[0][0] + x_0_1_sym[0] * (param_bounds[0][1] - param_bounds[0][0])
    back_arm_len = param_bounds[4][0] + x_0_1_sym[4] * (param_bounds[4][1] - param_bounds[4][0])
    side_seg2_len_left = param_bounds[8][0] + x_0_1_sym[8] * (param_bounds[8][1] - param_bounds[8][0])
    side_seg2_len_right = param_bounds[9][0] + x_0_1_sym[9] * (param_bounds[9][1] - param_bounds[9][0])

    front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
    back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
    side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
    side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
    side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density

    def box_inertia_diag(m, lx, ly, lz):
        return m/12 * (ly**2 + lz**2), m/12 * (lx**2 + lz**2), m/12 * (lx**2 + ly**2)

    def parallel_axis_diag(Ixx, Iyy, Izz, m, rx, ry, rz):
        r_sq = rx**2 + ry**2 + rz**2
        return Ixx + m * (r_sq - rx**2), Iyy + m * (r_sq - ry**2), Izz + m * (r_sq - rz**2)

    Ixx_total, Iyy_total, Izz_total = 0, 0, 0

    Ixx, Iyy, Izz = box_inertia_diag(base_mass, base_dims[0], base_dims[1], base_dims[2])
    Ixx_total += Ixx
    Iyy_total += Iyy
    Izz_total += Izz

    front_com_x = 0.035 + front_arm_len/2
    Ixx, Iyy, Izz = box_inertia_diag(front_arm_mass, front_arm_len, arm_width, arm_height)
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(Ixx, Iyy, Izz, front_arm_mass, front_com_x, 0, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p
    front_motor_x = 0.035 + front_arm_len
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(0, 0, 0, motor_mass, front_motor_x, 0, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p

    back_com_x = -0.035 - back_arm_len/2
    Ixx, Iyy, Izz = box_inertia_diag(back_arm_mass, back_arm_len, arm_width, arm_height)
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(Ixx, Iyy, Izz, back_arm_mass, back_com_x, 0, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p
    back_motor_x = -0.035 - back_arm_len
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(0, 0, 0, motor_mass, back_motor_x, 0, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p

    left_seg1_y = 0.025 + side_seg1_len/2
    Ixx, Iyy, Izz = box_inertia_diag(side_seg1_mass, arm_width, side_seg1_len, arm_height)
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(Ixx, Iyy, Izz, side_seg1_mass, 0, left_seg1_y, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p
    left_seg2_y = 0.025 + side_seg1_len + side_seg2_len_left/2
    Ixx, Iyy, Izz = box_inertia_diag(side_seg2_mass_left, arm_width, side_seg2_len_left, arm_height)
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(Ixx, Iyy, Izz, side_seg2_mass_left, 0, left_seg2_y, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p
    left_motor_y = 0.025 + side_seg1_len + side_seg2_len_left
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(0, 0, 0, motor_mass, 0, left_motor_y, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p

    right_seg1_y = -0.025 - side_seg1_len/2
    Ixx, Iyy, Izz = box_inertia_diag(side_seg1_mass, arm_width, side_seg1_len, arm_height)
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(Ixx, Iyy, Izz, side_seg1_mass, 0, right_seg1_y, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p
    right_seg2_y = -0.025 - side_seg1_len - side_seg2_len_right/2
    Ixx, Iyy, Izz = box_inertia_diag(side_seg2_mass_right, arm_width, side_seg2_len_right, arm_height)
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(Ixx, Iyy, Izz, side_seg2_mass_right, 0, right_seg2_y, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p
    right_motor_y = -0.025 - side_seg1_len - side_seg2_len_right
    Ixx_p, Iyy_p, Izz_p = parallel_axis_diag(0, 0, 0, motor_mass, 0, right_motor_y, 0)
    Ixx_total += Ixx_p
    Iyy_total += Iyy_p
    Izz_total += Izz_p

    return Ixx_total, Iyy_total, Izz_total


def analyze_actuation_accelerations(airframe: Airframe, tolerance_linear_acc: float, tolerance_angular_acc: float, configuration: list = None) -> dict:
    """Analyze actuation capability in terms of linear and angular accelerations.

    Args:
        airframe: Airframe object
        tolerance_linear_acc: Tolerance for linear acceleration constraints (m/s²)
        tolerance_angular_acc: Tolerance for angular acceleration constraints (rad/s²)
        configuration: Optional list of 7 constraint specs. Each spec is [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z].
            Values can be: 0.0 (zero with tolerance), "free" (unconstrained), 9.81 (gravity hover constraint),
            or "+lin_z", "+ang_x", "-ang_x", "+ang_y", "-ang_y", "+ang_z", "-ang_z" (optimization targets).
            If None, uses default configuration.

    Returns:
        Dictionary with max/min accelerations for each axis, plus 7 target acceleration scenarios (a_1 through a_7)
        with full 6-component acceleration vectors.
    """
    from scipy.optimize import linprog

    alloc_data = airframe.compute_allocation_matrix()
    B = alloc_data['B']
    mass = alloc_data['mass']
    min_forces = alloc_data['min_forces']
    max_forces = alloc_data['max_forces']
    n_motors = B.shape[1]

    I = compute_inertia_tensor(airframe)
    I_inv = np.linalg.inv(I)

    gravity_force = mass * 9.81
    tol_lin = tolerance_linear_acc * mass
    tol_ang_x, tol_ang_y, tol_ang_z = tolerance_angular_acc / I_inv[0, 0], tolerance_angular_acc / I_inv[1, 1], tolerance_angular_acc / I_inv[2, 2]
    bounds = [(min_forces[i], max_forces[i]) for i in range(n_motors)]

    if configuration is None:
        wrench_constraints = [
            [     0.0,      0.0, "+lin_z",       0.0,       0.0,      0.0],
            ["free", "free",     9.81,  "+ang_x",       0.0,      0.0],
            ["free", "free",     9.81,  "-ang_x",       0.0,      0.0],
            ["free", "free",     9.81,       0.0,  "+ang_y",      0.0],
            ["free", "free",     9.81,       0.0,  "-ang_y",      0.0],
            ["free", "free",   "free",       0.0,       0.0, "+ang_z"],
            ["free", "free",   "free",       0.0,       0.0, "-ang_z"],
        ]
    else:
        wrench_constraints = configuration
    wrench_constraints = [[gravity_force if c == 9.81 else c for c in row] for row in wrench_constraints]
    tol_map = [tol_lin, tol_lin, tol_lin, tol_ang_x, tol_ang_y, tol_ang_z]
    conv_map = [1.0/mass, 1.0/mass, 1.0/mass, I_inv[0, 0], I_inv[1, 1], I_inv[2, 2]]

    def solve_wrench_constraint(spec):
        opt_axis, maximize = None, True
        for i, c in enumerate(spec):
            if isinstance(c, str) and c.startswith(("+", "-")):
                opt_axis = i
                maximize = c.startswith("+")
                break
        constrained_axes, targets, tols = [], [], []
        for i, c in enumerate(spec):
            if c == "free" or (isinstance(c, str) and c.startswith(("+", "-"))):
                continue
            constrained_axes.append(i)
            targets.append(c)
            tols.append(tol_map[i])
        if not constrained_axes:
            A_ub, b_ub = None, None
        else:
            targets, tols = np.array(targets), np.array(tols)
            A_ub = np.vstack([B[constrained_axes, :], -B[constrained_axes, :]])
            b_ub = np.hstack([targets + tols, -targets + tols])
        c_vec = -B[opt_axis, :] if maximize else B[opt_axis, :]
        result = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        value = conv_map[opt_axis] * (-result.fun if maximize else result.fun) if result.success else np.nan
        return value, result

    results = [solve_wrench_constraint(spec) for spec in wrench_constraints]

    def wrench_to_accel(wrench):
        accel = np.zeros(6)
        accel[:3] = wrench[:3] / mass
        accel[3:] = I_inv @ wrench[3:]
        return accel

    def make_scenario(res, value, description, label):
        if not res.success:
            return {'value': np.nan, 'description': description, 'label': label, 'motor_forces': None, 'wrench': None, 'accel': None}
        wrench = B @ res.x
        return {'value': value, 'description': description, 'label': label, 'motor_forces': res.x, 'wrench': wrench, 'accel': wrench_to_accel(wrench)}

    return {
        'max_linear_accel': np.array([np.nan, np.nan, results[0][0]]),
        'min_linear_accel': np.array([np.nan, np.nan, np.nan]),
        'max_angular_accel': np.array([results[1][0], results[3][0], results[5][0]]),
        'min_angular_accel': np.array([results[2][0], results[4][0], results[6][0]]),
        'mass': mass,
        'inertia': I,
        'gravity_force': gravity_force,
        'a_1': make_scenario(results[0][1], results[0][0], r'$a_z^{\max}$ (maximum vertical)', r'$a_z^{\max}$'),
        'a_2': make_scenario(results[1][1], results[1][0], r'$\alpha_x^{\max}$ (positive roll at hover)', r'$\alpha_x^{\max}$'),
        'a_3': make_scenario(results[2][1], results[2][0], r'$-\alpha_x^{\max}$ (negative roll at hover)', r'$-\alpha_x^{\max}$'),
        'a_4': make_scenario(results[3][1], results[3][0], r'$\alpha_y^{\max}$ (positive pitch at hover)', r'$\alpha_y^{\max}$'),
        'a_5': make_scenario(results[4][1], results[4][0], r'$-\alpha_y^{\max}$ (negative pitch at hover)', r'$-\alpha_y^{\max}$'),
        'a_6': make_scenario(results[5][1], results[5][0], r'$\alpha_z^{\max}$ (positive yaw)', r'$\alpha_z^{\max}$'),
        'a_7': make_scenario(results[6][1], results[6][0], r'$-\alpha_z^{\max}$ (negative yaw)', r'$-\alpha_z^{\max}$'),
    }


def compute_target_wrenches(airframe: Airframe, tolerance_force: float, tolerance_torque: float) -> dict:
    """Compute the 7 target wrenches as defined in the paper.

    Returns dict with keys 'w_1' through 'w_7' containing:
        - 'value': the max/min value achieved for the target wrench component
        - 'description': human-readable description
        - 'motor_forces': optimal motor forces to achieve this wrench
        - 'wrench': full 6-component wrench vector [Fx, Fy, Fz, τx, τy, τz]
    """
    from scipy.optimize import linprog

    alloc_data = airframe.compute_allocation_matrix()
    B = alloc_data['B']
    mass = alloc_data['mass']
    min_forces = alloc_data['min_forces']
    max_forces = alloc_data['max_forces']

    gravity_force = mass * 9.81
    bounds = [(min_forces[i], max_forces[i]) for i in range(len(min_forces))]
    tol_force, tol_torque = tolerance_force, tolerance_torque

    results = {}

    def make_result(res, value, description, label):
        return {
            'value': value,
            'description': description,
            'label': label,
            'motor_forces': res.x if res.success else None,
            'wrench': B @ res.x if res.success else None
        }

    # w_1: Maximum lift with F_x=F_y=τ_x=τ_y=τ_z≈0
    A_ub = np.vstack([B[0, :], -B[0, :], B[1, :], -B[1, :], B[3, :], -B[3, :], B[4, :], -B[4, :], B[5, :], -B[5, :]])
    b_ub = np.array([tol_force, tol_force, tol_force, tol_force, tol_torque, tol_torque, tol_torque, tol_torque, tol_torque, tol_torque])
    res = linprog(-B[2, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_1'] = make_result(res, -res.fun if res.success else np.nan, r'$F_z^{\max}$ (maximum lift)', r'$F_z^{\max}$')

    # w_2: Positive roll at hover (Fz≈gravity, τy≈0, τz≈0, Fx/Fy free)
    A_ub = np.vstack([B[2, :], -B[2, :], B[4, :], -B[4, :], B[5, :], -B[5, :]])
    b_ub = np.array([gravity_force + tol_force, -gravity_force + tol_force, tol_torque, tol_torque, tol_torque, tol_torque])
    res = linprog(-B[3, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_2'] = make_result(res, -res.fun if res.success else np.nan, r'$\tau_x^{\max}$ (positive roll at hover)', r'$\tau_x^{\max}$')

    # w_3: Negative roll at hover
    res = linprog(B[3, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_3'] = make_result(res, res.fun if res.success else np.nan, r'$-\tau_x^{\max}$ (negative roll at hover)', r'$-\tau_x^{\max}$')

    # w_4: Positive pitch at hover (Fz≈gravity, τx≈0, τz≈0, Fx/Fy free)
    A_ub = np.vstack([B[2, :], -B[2, :], B[3, :], -B[3, :], B[5, :], -B[5, :]])
    b_ub = np.array([gravity_force + tol_force, -gravity_force + tol_force, tol_torque, tol_torque, tol_torque, tol_torque])
    res = linprog(-B[4, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_4'] = make_result(res, -res.fun if res.success else np.nan, r'$\tau_y^{\max}$ (positive pitch at hover)', r'$\tau_y^{\max}$')

    # w_5: Negative pitch at hover
    res = linprog(B[4, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_5'] = make_result(res, res.fun if res.success else np.nan, r'$-\tau_y^{\max}$ (negative pitch at hover)', r'$-\tau_y^{\max}$')

    # w_6: Positive yaw (τx≈0, τy≈0, Fx/Fy/Fz free)
    A_ub = np.vstack([B[3, :], -B[3, :], B[4, :], -B[4, :]])
    b_ub = np.array([tol_torque, tol_torque, tol_torque, tol_torque])
    res = linprog(-B[5, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_6'] = make_result(res, -res.fun if res.success else np.nan, r'$\tau_z^{\max}$ (positive yaw)', r'$\tau_z^{\max}$')

    # w_7: Negative yaw
    res = linprog(B[5, :], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    results['w_7'] = make_result(res, res.fun if res.success else np.nan, r'$-\tau_z^{\max}$ (negative yaw)', r'$-\tau_z^{\max}$')

    results['gravity_force'] = gravity_force
    return results


def repair_solution_acceleration(airframe: Airframe, target_accel_ratios: dict, verbose: bool, minimum_arm_torque: float, configuration: list = None) -> Airframe:
    """Repair a solution to maintain target acceleration capability.

    This function modifies the airframe design to ensure it can achieve at least
    the target percentage of the reference (planar) design's acceleration capability.

    All constraints are formulated in acceleration space (m/s² for linear, rad/s² for angular),
    properly accounting for how mass and inertia change parametrically with arm lengths.

    Args:
        airframe: Airframe to repair
        target_accel_ratios: Dict with 'linear' and 'angular' target ratios (e.g., 0.95 for 95%)
        verbose: Print progress
        minimum_arm_torque: Minimum arm joint torque constraint
        configuration: Optional list of 7 constraint specs. Each spec is [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z].
            Values can be: 0.0 (zero with tolerance), "free" (unconstrained), 9.81 (gravity hover constraint),
            or "+lin_z", "+ang_x", "-ang_x", "+ang_y", "-ang_y", "+ang_z", "-ang_z" (optimization targets).
            If None, uses default configuration.

    Returns:
        Repaired Airframe
    """
    import casadi as ca

    planar_max_linear_accel_z = 27.67
    planar_max_angular_accel_x = 230.73
    planar_max_angular_accel_y = 216.58
    planar_max_angular_accel_z = 33.63

    linear_ratio = 1.0 - target_accel_ratios['linear']
    angular_ratio = 1.0 - target_accel_ratios['angular']

    target_az = planar_max_linear_accel_z * linear_ratio
    target_alpha_x = planar_max_angular_accel_x * angular_ratio
    target_alpha_y = planar_max_angular_accel_y * angular_ratio
    target_alpha_z = planar_max_angular_accel_z * angular_ratio
    n_motors = 4
    g = 9.81

    tol_lin_accel = 0.0001
    tol_ang_accel = 0.001

    if verbose:
        print(f"\nTarget accelerations ({linear_ratio*100:.0f}% linear, {angular_ratio*100:.0f}% angular of planar):")
        print(f"  Linear z (lift): {target_az:.2f} m/s²")
        print(f"  Angular x (roll): {target_alpha_x:.2f} rad/s²")
        print(f"  Angular y (pitch): {target_alpha_y:.2f} rad/s²")
        print(f"  Angular z (yaw): {target_alpha_z:.2f} rad/s²")
        print(f"  Tolerance linear: {tol_lin_accel:.2f} m/s², angular: {tol_ang_accel:.2f} rad/s²")

    param_bounds = list(PARAM_BOUNDS.values())
    arm_width = AIRFRAME_CONSTANTS['arm_width']
    arm_height = AIRFRAME_CONSTANTS['arm_height']
    arm_density = AIRFRAME_CONSTANTS['arm_density']
    motor_mass = AIRFRAME_CONSTANTS['motor_mass']
    base_mass = AIRFRAME_CONSTANTS['base_mass']
    side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
    motor_directions = AIRFRAME_CONSTANTS['motor_directions']
    thrust_coefficient = AIRFRAME_CONSTANTS['thrust_coefficient']
    max_rps = AIRFRAME_CONSTANTS['max_rps']
    min_rps = AIRFRAME_CONSTANTS['min_rps']
    cq = AIRFRAME_CONSTANTS['cq']
    max_force = thrust_coefficient * max_rps ** 2
    min_force = thrust_coefficient * min_rps ** 2
    thrust_to_torque_ratio = cq

    def rpy_to_rotation_matrix_casadi(roll, pitch, yaw):
        cr, sr = ca.cos(roll), ca.sin(roll)
        cp, sp = ca.cos(pitch), ca.sin(pitch)
        cy, sy = ca.cos(yaw), ca.sin(yaw)
        R_x = ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, cr, -sr), ca.horzcat(0, sr, cr))
        R_y = ca.vertcat(ca.horzcat(cp, 0, sp), ca.horzcat(0, 1, 0), ca.horzcat(-sp, 0, cp))
        R_z = ca.vertcat(ca.horzcat(cy, -sy, 0), ca.horzcat(sy, cy, 0), ca.horzcat(0, 0, 1))
        return R_z @ R_y @ R_x

    def compute_allocation_and_dynamics_casadi(x_0_1_sym):
        params = [param_bounds[i][0] + x_0_1_sym[i] * (param_bounds[i][1] - param_bounds[i][0]) for i in range(len(param_bounds))]
        front_arm_len = params[0]
        front_prop_rot = [params[i] * ca.pi / 180 for i in [1, 2, 3]]
        back_arm_len = params[4]
        back_prop_rot = [params[i] * ca.pi / 180 for i in [5, 6, 7]]
        side_seg2_len_left = params[8]
        side_seg2_len_right = params[9]
        side_base_rot_left = [params[i] * ca.pi / 180 for i in [10, 11, 12]]
        side_base_rot_right = [params[i] * ca.pi / 180 for i in [13, 14, 15]]
        side_motor_rot_left = [params[i] * ca.pi / 180 for i in [16, 17, 18]]
        side_motor_rot_right = [params[i] * ca.pi / 180 for i in [19, 20, 21]]

        front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
        back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
        side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
        side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
        side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density
        total_mass = base_mass + 4 * motor_mass + front_arm_mass + back_arm_mass + 2 * side_seg1_mass + side_seg2_mass_left + side_seg2_mass_right

        motor_0_pos = ca.vertcat(0.035 + front_arm_len, 0.0, 0.0)
        R_front = rpy_to_rotation_matrix_casadi(front_prop_rot[0], front_prop_rot[1], front_prop_rot[2])
        motor_0_thrust = R_front @ ca.vertcat(0, 0, 1)

        R_base_1 = rpy_to_rotation_matrix_casadi(side_base_rot_left[0], side_base_rot_left[1], side_base_rot_left[2] + ca.pi/2)
        base_1 = ca.vertcat(0, 0.025, 0)
        joint_1_pos = base_1 + R_base_1 @ ca.vertcat(side_seg1_len, 0, 0)
        joint_1_axis = R_base_1 @ ca.vertcat(0, 0, 1)
        motor_1_pos = base_1 + R_base_1 @ ca.vertcat(side_seg1_len + side_seg2_len_left, 0, 0)
        R_motor_1 = R_base_1 @ rpy_to_rotation_matrix_casadi(side_motor_rot_left[0], side_motor_rot_left[1], side_motor_rot_left[2])
        motor_1_thrust = R_motor_1 @ ca.vertcat(0, 0, 1)

        R_back = rpy_to_rotation_matrix_casadi(0, 0, ca.pi)
        motor_2_pos = ca.vertcat(-0.035, 0, 0) + R_back @ ca.vertcat(back_arm_len, 0, 0)
        R_motor_2 = R_back @ rpy_to_rotation_matrix_casadi(back_prop_rot[0], back_prop_rot[1], back_prop_rot[2])
        motor_2_thrust = R_motor_2 @ ca.vertcat(0, 0, 1)

        R_base_3 = rpy_to_rotation_matrix_casadi(side_base_rot_right[0], side_base_rot_right[1], side_base_rot_right[2] - ca.pi/2)
        base_3 = ca.vertcat(0, -0.025, 0)
        joint_3_pos = base_3 + R_base_3 @ ca.vertcat(side_seg1_len, 0, 0)
        joint_3_axis = R_base_3 @ ca.vertcat(0, 0, 1)
        motor_3_pos = base_3 + R_base_3 @ ca.vertcat(side_seg1_len + side_seg2_len_right, 0, 0)
        R_motor_3 = R_base_3 @ rpy_to_rotation_matrix_casadi(side_motor_rot_right[0], side_motor_rot_right[1], side_motor_rot_right[2])
        motor_3_thrust = R_motor_3 @ ca.vertcat(0, 0, 1)

        motor_positions = [motor_0_pos, motor_1_pos, motor_2_pos, motor_3_pos]
        motor_thrusts = [motor_0_thrust, motor_1_thrust, motor_2_thrust, motor_3_thrust]

        B = ca.MX.zeros(6, n_motors)
        for i in range(n_motors):
            r = motor_positions[i]
            thrust_vec = motor_thrusts[i]
            torque_contrib = ca.cross(r, thrust_vec)
            yaw_torque = motor_directions[i] * thrust_to_torque_ratio * thrust_vec
            B[0:3, i] = thrust_vec
            B[3:6, i] = torque_contrib + yaw_torque

        Ixx, Iyy, Izz = _compute_inertia_tensor_casadi(x_0_1_sym)

        joint_info = {
            'joint_1_pos': joint_1_pos, 'joint_1_axis': joint_1_axis,
            'joint_3_pos': joint_3_pos, 'joint_3_axis': joint_3_axis,
            'motor_1_pos': motor_1_pos, 'motor_1_thrust': motor_1_thrust,
            'motor_3_pos': motor_3_pos, 'motor_3_thrust': motor_3_thrust,
        }
        return B, total_mass, Ixx, Iyy, Izz, joint_info

    def wrench_to_accel(wrench, mass, Ixx, Iyy, Izz):
        """Convert wrench [fx, fy, fz, tx, ty, tz] to accelerations [ax, ay, az, alpha_x, alpha_y, alpha_z]."""
        return ca.vertcat(wrench[0]/mass, wrench[1]/mass, wrench[2]/mass, wrench[3]/Ixx, wrench[4]/Iyy, wrench[5]/Izz)

    x_0_1 = np.array(airframe.x_0_1)
    n_params = len(x_0_1)

    # Known feasible planar design (always satisfies constraints)
    planar_x_0_1 = np.array([0.38, 0.5, 0.5, 0.5, 0.38, 0.5, 0.5, 0.5, 0.28, 0.28,
                             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5])

    if configuration is None:
        wrench_constraints = [
            [     0.0,      0.0, "+lin_z",       0.0,       0.0,      0.0],
            ["free", "free",     9.81,  "+ang_x",       0.0,      0.0],
            ["free", "free",     9.81,  "-ang_x",       0.0,      0.0],
            ["free", "free",     9.81,       0.0,  "+ang_y",      0.0],
            ["free", "free",     9.81,       0.0,  "-ang_y",      0.0],
            ["free", "free",   "free",       0.0,       0.0, "+ang_z"],
            ["free", "free",   "free",       0.0,       0.0, "-ang_z"],
        ]
    else:
        wrench_constraints = configuration
    wrench_constraints = [[g if c == 9.81 else c for c in row] for row in wrench_constraints]
    n_wrenches = len(wrench_constraints)
    target_map = {"+lin_z": target_az, "+ang_x": target_alpha_x, "-ang_x": target_alpha_x, "+ang_y": target_alpha_y, "-ang_y": target_alpha_y, "+ang_z": target_alpha_z, "-ang_z": target_alpha_z}
    tol_map = [tol_lin_accel, tol_lin_accel, tol_lin_accel, tol_ang_accel, tol_ang_accel, tol_ang_accel]

    current_x_0_1 = x_0_1.copy()
    max_reinit_attempts = 50
    reinit_count = 0

    while True:
        delta = ca.MX.sym('delta', n_params)
        x_new = ca.DM(current_x_0_1) + delta
        B_sym, mass_sym, Ixx_sym, Iyy_sym, Izz_sym, joint_info = compute_allocation_and_dynamics_casadi(x_new)
        objective = ca.sqrt(ca.dot(delta, delta) + 1e-12)

        u_vars = [ca.MX.sym(f'u_{i}', n_motors) for i in range(n_wrenches)]
        constraints, lbg, ubg = [], [], []

        for w_idx, spec in enumerate(wrench_constraints):
            u = u_vars[w_idx]
            f = ca.DM(min_force) + u * (max_force - min_force)
            wrench = B_sym @ f
            accel = wrench_to_accel(wrench, mass_sym, Ixx_sym, Iyy_sym, Izz_sym)

            for axis, c in enumerate(spec):
                if c == "free":
                    continue
                elif isinstance(c, str) and c.startswith("+"):
                    constraints.append(target_map[c] - accel[axis])
                    lbg.append(-ca.inf)
                    ubg.append(0)
                elif isinstance(c, str) and c.startswith("-"):
                    constraints.append(target_map[c] - (-accel[axis]))
                    lbg.append(-ca.inf)
                    ubg.append(0)
                elif c == g:
                    constraints.append(accel[axis] - g)
                    lbg.append(-tol_map[axis])
                    ubg.append(tol_map[axis])
                else:
                    constraints.append(accel[axis] - c)
                    lbg.append(-tol_map[axis])
                    ubg.append(tol_map[axis])

        if minimum_arm_torque > 0:
            r_1 = joint_info['motor_1_pos'] - joint_info['joint_1_pos']
            thrust_1 = max_force * joint_info['motor_1_thrust']
            torque_thrust_1 = ca.dot(ca.cross(r_1, thrust_1), joint_info['joint_1_axis'])
            counter_1 = motor_directions[1] * thrust_to_torque_ratio * max_force * ca.dot(joint_info['motor_1_thrust'], joint_info['joint_1_axis'])
            constraints.append(torque_thrust_1 + counter_1 + minimum_arm_torque)
            lbg.append(-ca.inf)
            ubg.append(0)

            r_3 = joint_info['motor_3_pos'] - joint_info['joint_3_pos']
            thrust_3 = max_force * joint_info['motor_3_thrust']
            torque_thrust_3 = ca.dot(ca.cross(r_3, thrust_3), joint_info['joint_3_axis'])
            counter_3 = motor_directions[3] * thrust_to_torque_ratio * max_force * ca.dot(joint_info['motor_3_thrust'], joint_info['joint_3_axis'])
            constraints.append(minimum_arm_torque - (torque_thrust_3 + counter_3))
            lbg.append(-ca.inf)
            ubg.append(0)

        all_vars = ca.vertcat(delta, *u_vars)
        all_constraints = ca.vertcat(*constraints)

        lbx = list(-current_x_0_1) + [0] * (n_wrenches * n_motors)
        ubx = list(1 - current_x_0_1) + [1] * (n_wrenches * n_motors)

        u_init_list = [np.ones(n_motors) * 0.3 for _ in range(n_wrenches)]
        x_init = np.concatenate([np.zeros(n_params)] + u_init_list)

        nlp = {'x': all_vars, 'f': objective, 'g': all_constraints}
        opts = {
            'ipopt.print_level': 5 if verbose else 0,
            'print_time': verbose,
            'ipopt.max_iter': 3000,
            'ipopt.tol': 1e-10,
            'ipopt.constr_viol_tol': 1e-12,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_constr_viol_tol': 1e-10,
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        solver_stats = solver.stats()
        return_status = solver_stats.get('return_status', 'unknown')

        z_final = np.array(sol['x']).flatten()
        delta_final = z_final[:n_params]
        g_val = np.array(sol['g']).flatten()
        lbg_arr, ubg_arr = np.array(lbg), np.array(ubg)

        if verbose:
            print(f"[CasADi] Solver status: {return_status}")
            print(f"[CasADi] Final constraint values (first 10): {g_val[:10]}")

        if return_status in ['Infeasible_Problem_Detected', 'Invalid_Problem_Definition']:
            reinit_count += 1
            if reinit_count > max_reinit_attempts:
                raise RuntimeError(f"Repair optimization failed after {max_reinit_attempts} reinitializations: {return_status}")

            violated_lower = np.where(g_val < lbg_arr - 1e-6)[0]
            violated_upper = np.where(g_val > ubg_arr + 1e-6)[0]

            current_x_0_1 = current_x_0_1 + 0.2 * (planar_x_0_1 - current_x_0_1)
            # print(f"\n[Infeasible] Reinitializing {reinit_count}/{max_reinit_attempts}: moving 20% closer to planar design")
            # print(f"  Constraints violated (lower bound): {len(violated_lower)}, (upper bound): {len(violated_upper)}")
            # for idx in violated_lower[:5]:
            #     print(f"    Constraint {idx}: value={g_val[idx]:.6f}, lb={lbg_arr[idx]:.6f}, violation={lbg_arr[idx] - g_val[idx]:.6f}")
            # for idx in violated_upper[:5]:
            #     print(f"    Constraint {idx}: value={g_val[idx]:.6f}, ub={ubg_arr[idx]:.6f}, violation={g_val[idx] - ubg_arr[idx]:.6f}")
            # print(f"  Distance to planar: {np.linalg.norm(current_x_0_1 - planar_x_0_1):.6f}")
            # print(f"  New distance to planar: {np.linalg.norm(current_x_0_1 - planar_x_0_1):.6f}")
            continue

        break

    repaired_airframe = Airframe.from_x_0_1(np.clip(current_x_0_1 + delta_final, 0, 1))
    return repaired_airframe


def get_airframe_motor_data(airframe: Airframe) -> dict:
    """Get motor positions and thrust directions from airframe.

    Args:
        airframe: Airframe object

    Returns:
        Dictionary with motor_positions, motor_thrust_directions, motor_directions, mass
    """
    motor_data = airframe.compute_motor_positions_and_thrusts()
    return {
        "motor_positions": np.array(motor_data["motor_positions"]),
        "motor_thrust_directions": np.array(motor_data["motor_thrust_directions"]),
        "motor_directions": np.array([1, -1, 1, -1]),
        "mass": airframe.compute_total_mass()
    }


def get_max_arm_angles_no_collision(airframe: Airframe, delta_deg: float, verbose: bool) -> tuple[float, float, dict]:
    """Find maximum arm bend angles before self-collision using FCL.

    Args:
        airframe: Airframe object
        delta_deg: Angle increment in degrees for collision search
        verbose: If True, display 3D visualization of the airframe

    Returns:
        Tuple of (max_left_angle_deg, max_right_angle_deg, aabb_info)
    """
    return airframe.get_max_arm_angles(delta_deg, verbose)
