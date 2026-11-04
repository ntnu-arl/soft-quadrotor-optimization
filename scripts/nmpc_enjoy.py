"""
NMPC Controller Evaluation Script for Morphy Wall Gap Task

Evaluates a nonlinear MPC position controller on the wall gap navigation task.

Usage:
  python scripts/nmpc_enjoy.py --headless --num_envs 10 \
    --usd_path cache/workspace_local/my_airframe/morphy_prog.usd

  python scripts/nmpc_enjoy.py \
    --usd_path cache/workspace_local/my_airframe/morphy_prog.usd \
    --record_video --playback_speed 0.25 --num_envs 1 --num_episodes 3
"""

import argparse
from pathlib import Path
from datetime import datetime

# Forkserver children import this module without launching Isaac or loading torch.
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from process_lifecycle import exit_with_owner
    exit_with_owner(os.getppid())

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate NMPC controller for Morphy wall gap task")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes per environment")
    parser.add_argument("--record_video", action="store_true", help="Record video of first environment")
    parser.add_argument("--playback_speed", type=float, default=0.25, help="Video playback speed multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--metrics_output", type=str, default=None, help="Path to save metrics JSON")
    parser.add_argument("--animation_capture_module", type=str, help="Explicit Python module path for recording native simulation states")
    parser.add_argument("--config-override", action="append", help="Override config values")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to USD file")
    parser.add_argument("--disable_traversal_tilt_crash", action="store_true", help="Disable the gap-zone orientation crash criterion")
    parser.add_argument("--collect_traversal_tilt", action="store_true", help="Save per-episode maximum body-z tilt inside the gap traversal zone")
    parser.add_argument("--collect_crossing_window", action="store_true", help="Save peak tilt and angular speed from the interpolated forward x=0 crossing through 0.2 s after it")
    parser.add_argument("--crossing_trace_output", type=str, help="Write native crossing traces for controller-response analysis")
    parser.add_argument("--motor_constraint_rps_min", type=float, help="Explicit lower motor-speed constraint for controlled NMPC ablations")
    parser.add_argument("--task_config", type=str, help="Path to wall-gap task config YAML")
    parser.add_argument("--nmpc_config", type=str, help="Path to NMPC controller config YAML")
    parser.add_argument("--camera_eye", type=float, nargs=3, default=[-2.5, 0.0, 3.5], help="Camera position")
    parser.add_argument("--camera_target", type=float, nargs=3, default=[-1.0, 0.0, 1.5], help="Camera target")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720], help="Video resolution")
    parser.add_argument("--video_filename", type=str, default=None, help="Custom video filename")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    args_cli.headless = True
    args_cli.enable_cameras = args_cli.record_video

    app_launcher = AppLauncher(args_cli, multi_gpu=False)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    import json
    import yaml
    import matplotlib.pyplot as plt

    from morphy_simulator import MorphySimulator
    from morphy_wall_gap_env import VecEnvRLGamesWrapper as WallGapWrapper
    from morphy_wall_gap_env import MorphyWallGapEnv
    from control_allocator import ControlAllocator
    from collision_trajectory import CollisionReferenceTrajectory
    from nmpc_controller import NMPCController
    from utils import Recorder
    from airframe_encoding import AIRFRAME_CONSTANTS
    from debug_plots_controller import NMPCLogger


def apply_config_overrides(config: dict, overrides: list):
    if not overrides:
        return {}
    special_flags = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        try:
            value = float(value)
            if value == int(value):
                value = int(value)
        except ValueError:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
        if key_path == "log_metrics":
            special_flags['log_metrics'] = value
            continue
        target = config
        for key in keys[:-1]:
            if key not in target:
                raise KeyError(f"Config section '{key}' not found in {list(target.keys())}")
            target = target[key]
        final_key = keys[-1]
        if final_key not in target:
            raise KeyError(f"Config key '{final_key}' not found. Available: {list(target.keys())}")
        print(f"[INFO] Overriding {key_path}: {target[final_key]} -> {value}")
        target[final_key] = value
    return special_flags


def create_env(num_envs, usd_path, config_overrides):
    config_path = Path(args_cli.task_config) if args_cli.task_config else Path(__file__).parent.parent / "configs/lee_wall_gap_simulation.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    special_flags = apply_config_overrides(config, config_overrides)
    sim_cfg = config['simulator']
    sim = MorphySimulator(
        usd_path=str(usd_path),
        n_envs=num_envs,
        dt=sim_cfg['dt'],
        gravity=tuple(sim_cfg['gravity']),
        motor_model=sim_cfg['motor_model'],
        motor_directions=AIRFRAME_CONSTANTS['motor_directions'],
        revolute_joint_damping_coeff=sim_cfg['revolute_joint_damping_coeff'],
        env_spacing=sim_cfg['env_spacing'],
    )
    rl_env = MorphyWallGapEnv(simulator=sim, config=config, reward_weights=None)
    return WallGapWrapper(rl_env), special_flags, config


def evaluate_nmpc(env, controller, ref_traj, allocator, rotor_dynamics, num_episodes, record_video, special_flags):
    num_envs = env.num_envs
    dt_rl = env.env.dt_rl
    max_steps = env.env.max_steps
    action_decimation = env.env.action_decimation

    print(f"\n[INFO] NMPC evaluation: {num_episodes} episodes/env, {env.env.episode_length_s}s ({max_steps} steps each)")

    import time
    total_target_episodes = num_envs * num_episodes
    progress_start_time = time.time()
    last_progress_time = progress_start_time

    recorder = None
    metrics_log_file = None
    log_metrics = special_flags.get('log_metrics', False)
    if record_video:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        if args_cli.video_filename:
            video_path = results_dir / f"{args_cli.video_filename}.mp4"
            if log_metrics:
                metrics_log_path = results_dir / f"{args_cli.video_filename}_metrics.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = results_dir / f"nmpc_wall_gap_eval_{timestamp}.mp4"
            if log_metrics:
                metrics_log_path = results_dir / f"nmpc_wall_gap_eval_{timestamp}_metrics.jsonl"
        recorder = Recorder(
            playback_speed=args_cli.playback_speed,
            mp4_path=str(video_path),
            dt=env.env.sim.dt * action_decimation,
            sim=env.env.sim.sim,
            resolution=tuple(args_cli.resolution),
            camera_eye=args_cli.camera_eye,
            camera_target=args_cli.camera_target,
        )
        print(f"[INFO] Recording video to {video_path}")

        if log_metrics:
            metrics_log_file = open(metrics_log_path, 'w')
            print(f"[INFO] Logging metrics to {metrics_log_path}")

    nmpc_logger = NMPCLogger(num_motors=allocator.num_motors, horizon_N=controller.N) if num_envs == 1 else None

    env.reset()
    if env.env.sim.motor_model != "motors_disabled":
        initial_motor_action = env.env.hover_action.unsqueeze(0).expand(num_envs, -1)
        env.env.sim.rotor_dynamics.reset_idx(torch.arange(num_envs, device=env.device), initial_motor_action)
        env.env.sim._last_motor_actions[:] = initial_motor_action
    episode_counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    active_envs = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    crash_counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    timeout_counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    success_counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    episode_rewards = torch.zeros(num_envs, device=env.device)
    all_episode_rewards = []
    position_error_sum = 0.0
    velocity_error_sum = 0.0
    attitude_error_sum = 0.0
    control_effort_sum = 0.0
    tracking_error_count = 0
    traversal_max_tilt = torch.zeros(num_envs, device=env.device)
    traversal_sample_counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    traversal_episodes = []
    crossing_window_episodes = []
    crossing_trace_records = []
    if args_cli.crossing_trace_output and not args_cli.collect_crossing_window:
        raise ValueError("Crossing traces require --collect_crossing_window")
    previous_x = torch.zeros(num_envs, device=env.device)
    previous_tilt_deg = torch.zeros(num_envs, device=env.device)
    previous_angular_speed = torch.zeros(num_envs, device=env.device)
    crossing_time = torch.full((num_envs,), float("nan"), device=env.device)
    crossing_peak_tilt_deg = torch.zeros(num_envs, device=env.device)
    crossing_peak_angular_speed = torch.zeros(num_envs, device=env.device)
    crossing_sample_counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    crossing_window_complete = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    step_count = 0
    env_time = torch.zeros(num_envs, device=env.device)

    ref_traj.reset(torch.arange(num_envs, device=env.device))
    controller.reset(torch.arange(num_envs, device=env.device))

    sim = env.env.sim
    animation_capture = None
    if args_cli.animation_capture_module:
        import importlib.util
        capture_spec = importlib.util.spec_from_file_location("animation_capture", args_cli.animation_capture_module)
        capture_module = importlib.util.module_from_spec(capture_spec)
        capture_spec.loader.exec_module(capture_module)
        animation_capture = capture_module.Capture(env.env)
    if args_cli.collect_crossing_window:
        initial_pos, initial_quat, _, initial_ang_vel = sim.get_base_state()
        relative_initial_pos = initial_pos - sim.scene.env_origins
        previous_x[:] = relative_initial_pos[:, 0]
        previous_tilt_deg[:] = torch.rad2deg(torch.acos(torch.clamp(
            1.0 - 2.0 * (initial_quat[:, 1].square() + initial_quat[:, 2].square()), -1.0, 1.0
        )))
        previous_angular_speed[:] = torch.linalg.vector_norm(initial_ang_vel, dim=-1)
    gravity_vec = torch.tensor([sim.gravity[0], sim.gravity[1], sim.gravity[2]], device=env.device).unsqueeze(0).expand(num_envs, -1)
    mass_scalar = sim.robot_mass
    mass_tensor = torch.tensor(mass_scalar, device=env.device).unsqueeze(0).expand(num_envs, 1)
    inertia_tensor = sim.get_inertia_tensor()
    print(f"[DEBUG] inertia_tensor=\n{inertia_tensor[0]}")
    print(f"[DEBUG] mass={mass_scalar:.4f}")

    if record_video:
        env.env.auto_reset = False

    with torch.no_grad():
        while active_envs.any():
            observed_pos, observed_quat, observed_lin_vel, observed_ang_vel_world = env.env.get_observed_base_state()
            true_pos, true_quat, true_lin_vel, _ = sim.get_base_state()
            controller.set_state(
                observed_pos,
                observed_quat,
                observed_lin_vel,
                observed_ang_vel_world,
                mass_tensor,
                inertia_tensor,
                gravity_vec,
            )

            ref_pos, ref_vel, ref_yaw, ref_yaw_rate, control_mode, nmpc_mode = ref_traj.get_reference(observed_pos, env_time)

            # Real flight has no arm-angle estimate, so NMPC allocation must always use the zero-angle arm geometry.
            motor_pos_body = sim.get_motor_positions_body_frame_for_allocation_matrix()
            motor_thrust_dirs = sim.get_motor_thrust_directions_body_frame_for_allocation_matrix()
            allocator.update_allocation_matrix(motor_pos_body, motor_thrust_dirs)
            controller.set_allocation_matrix(allocator.allocation_matrix.cpu().numpy())
            controller.set_com_offset(sim.get_com_offset().cpu().numpy())
            controller.set_measured_motor_rps(sim.rotor_dynamics.current_rps.cpu().numpy())

            if any(mode != "NMPC_DIRECT_ACTUATOR" for mode in control_mode):
                raise ValueError(f"unsupported collision control_mode in simulation NMPC: {control_mode}")
            motor_forces = controller.update(ref_pos, ref_vel, ref_yaw, ref_yaw_rate, nmpc_mode)
            actions = rotor_dynamics.force_to_action(motor_forces) * 2.0 - 1.0

            if nmpc_logger is not None:
                nmpc_logger.log_step(
                    step_count, env_time[0].item(),
                    observed_pos[0].cpu().tolist(), observed_lin_vel[0].cpu().tolist(),
                    ref_pos[0].cpu().tolist(), ref_vel[0].cpu().tolist(),
                    ref_yaw[0].item(), ref_yaw_rate[0].item(),
                    observed_quat[0].cpu().tolist(), observed_ang_vel_world[0].cpu().tolist(),
                    controller, allocator, actions[0].cpu().tolist(),
                )

            _, rewards, dones, infos = env.step(actions)
            if animation_capture is not None:
                animation_capture.record(dones, infos, active_envs)
            if args_cli.collect_crossing_window:
                current_x = (
                    env.env.current_position[:, 0] - sim.scene.env_origins[:, 0]
                )
                current_tilt_deg = infos['traversal_tilt_deg']
                current_angular_speed = torch.linalg.vector_norm(env.env.current_ang_vel_body, dim=-1)
                step_end_time = env_time + dt_rl
                if args_cli.crossing_trace_output:
                    trace_mask = active_envs & (current_x >= -0.5) & (
                        torch.isnan(crossing_time) | (step_end_time <= crossing_time + 0.21)
                    )
                    ids = torch.where(trace_mask)[0]
                    positions = (env.env.current_position - sim.scene.env_origins)[ids].cpu().tolist()
                    angular_velocities = env.env.current_ang_vel_body[ids].cpu().tolist()
                    for index, env_id in enumerate(ids.tolist()):
                        crossing_trace_records.append({
                            "env_id": env_id, "episode": int(episode_counts[env_id].item()),
                            "time_s": float(step_end_time[env_id].item()),
                            "position": positions[index],
                            "tilt_deg": float(current_tilt_deg[env_id].item()),
                            "omega_body_rad_s": angular_velocities[index],
                        })
                crossed_now = (
                    active_envs & torch.isnan(crossing_time)
                    & (previous_x < 0.0) & (current_x >= 0.0)
                )
                if crossed_now.any():
                    fraction = -previous_x[crossed_now] / (current_x[crossed_now] - previous_x[crossed_now])
                    crossing_time[crossed_now] = env_time[crossed_now] + fraction * dt_rl
                    interpolated_tilt = previous_tilt_deg[crossed_now] + fraction * (
                        current_tilt_deg[crossed_now] - previous_tilt_deg[crossed_now]
                    )
                    interpolated_angular_speed = previous_angular_speed[crossed_now] + fraction * (
                        current_angular_speed[crossed_now] - previous_angular_speed[crossed_now]
                    )
                    crossing_peak_tilt_deg[crossed_now] = interpolated_tilt
                    crossing_peak_angular_speed[crossed_now] = interpolated_angular_speed
                    crossing_sample_counts[crossed_now] = 1
                in_crossing_window = (
                    active_envs & torch.isfinite(crossing_time)
                    & (step_end_time >= crossing_time)
                    & (step_end_time <= crossing_time + 0.2)
                )
                crossing_peak_tilt_deg[in_crossing_window] = torch.maximum(
                    crossing_peak_tilt_deg[in_crossing_window], current_tilt_deg[in_crossing_window]
                )
                crossing_peak_angular_speed[in_crossing_window] = torch.maximum(
                    crossing_peak_angular_speed[in_crossing_window], current_angular_speed[in_crossing_window]
                )
                crossing_sample_counts[in_crossing_window] += 1
                crossed_window_end = (
                    active_envs & torch.isfinite(crossing_time)
                    & (step_end_time - dt_rl < crossing_time + 0.2)
                    & (step_end_time >= crossing_time + 0.2)
                )
                if crossed_window_end.any():
                    end_fraction = (
                        crossing_time[crossed_window_end] + 0.2 - (step_end_time[crossed_window_end] - dt_rl)
                    ) / dt_rl
                    end_tilt = previous_tilt_deg[crossed_window_end] + end_fraction * (
                        current_tilt_deg[crossed_window_end] - previous_tilt_deg[crossed_window_end]
                    )
                    end_angular_speed = previous_angular_speed[crossed_window_end] + end_fraction * (
                        current_angular_speed[crossed_window_end] - previous_angular_speed[crossed_window_end]
                    )
                    crossing_peak_tilt_deg[crossed_window_end] = torch.maximum(
                        crossing_peak_tilt_deg[crossed_window_end], end_tilt
                    )
                    crossing_peak_angular_speed[crossed_window_end] = torch.maximum(
                        crossing_peak_angular_speed[crossed_window_end], end_angular_speed
                    )
                    crossing_sample_counts[crossed_window_end] += 1
                    crossing_window_complete[crossed_window_end] = True
                previous_x[:] = current_x
                previous_tilt_deg[:] = current_tilt_deg
                previous_angular_speed[:] = current_angular_speed
            if args_cli.collect_traversal_tilt:
                in_zone = infos['in_traversal_zone'] & active_envs
                traversal_max_tilt[in_zone] = torch.maximum(
                    traversal_max_tilt[in_zone], infos['traversal_tilt_deg'][in_zone]
                )
                traversal_sample_counts[in_zone] += 1
            episode_rewards += rewards
            env_time += dt_rl

            if recorder:
                recorder.record_frame()

            position_error = torch.norm(true_pos - ref_pos, dim=-1)
            velocity_error = torch.norm(true_lin_vel - ref_vel, dim=-1)
            attitude_error = 1.0 - (true_quat * controller.desired_quat).sum(dim=-1).square()
            motor_actions_01 = rotor_dynamics.force_to_action(motor_forces)
            control_effort = (motor_actions_01 ** 2).mean(dim=-1)
            position_error_sum += position_error.sum().item()
            velocity_error_sum += velocity_error.sum().item()
            attitude_error_sum += attitude_error.sum().item()
            control_effort_sum += control_effort.sum().item()
            tracking_error_count += position_error.numel()

            if metrics_log_file and num_envs == 1:
                joint_pos, _ = sim.get_joint_state()
                joint_angles_deg = (joint_pos[0].cpu().numpy() * 180.0 / np.pi).tolist()
                metrics_entry = {
                    'timestamp': env_time[0].item(),
                    'position': true_pos[0].cpu().tolist(),
                    'velocity': true_lin_vel[0].cpu().tolist(),
                    'observed_position': observed_pos[0].cpu().tolist(),
                    'observed_velocity': observed_lin_vel[0].cpu().tolist(),
                    'setpoint_position': ref_pos[0].cpu().tolist(),
                    'setpoint_velocity': ref_vel[0].cpu().tolist(),
                    'position_error': position_error[0].item(),
                    'velocity_error': velocity_error[0].item(),
                    'attitude_error': attitude_error[0].item(),
                    'control_effort': control_effort[0].item(),
                    'joint_angles': joint_angles_deg
                }
                metrics_log_file.write(json.dumps(metrics_entry) + '\n')

            reset_envs = dones & active_envs
            if reset_envs.any():
                crashed_now = infos['crashed'] & reset_envs
                succeeded_now = infos['succeeded'] & reset_envs
                timeout_now = reset_envs & ~crashed_now & ~succeeded_now

                if args_cli.collect_traversal_tilt:
                    for env_id in torch.where(reset_envs)[0].tolist():
                        sampled = int(traversal_sample_counts[env_id].item())
                        traversal_episodes.append({
                            'env_id': env_id,
                            'episode': int(episode_counts[env_id].item()),
                            'gap_width': float(env.env.gap_widths[env_id].item()),
                            'succeeded': bool(succeeded_now[env_id].item()),
                            'crashed': bool(crashed_now[env_id].item()),
                            'traversal_samples': sampled,
                            'max_traversal_tilt_deg': float(traversal_max_tilt[env_id].item()) if sampled else None,
                        })
                    traversal_max_tilt[reset_envs] = 0.0
                    traversal_sample_counts[reset_envs] = 0

                if args_cli.collect_crossing_window:
                    for env_id in torch.where(reset_envs)[0].tolist():
                        crossed = bool(torch.isfinite(crossing_time[env_id]).item())
                        crossing_window_episodes.append({
                            'env_id': env_id,
                            'episode': int(episode_counts[env_id].item()),
                            'gap_width': float(env.env.gap_widths[env_id].item()),
                            'succeeded': bool(succeeded_now[env_id].item()),
                            'crashed': bool(crashed_now[env_id].item()),
                            'crossing_time_s': float(crossing_time[env_id].item()) if crossed else None,
                            'window_end_s': float(crossing_time[env_id].item() + 0.2) if crossed else None,
                            'complete_window': bool(crossing_window_complete[env_id].item()),
                            'window_samples_including_interpolated_crossing': int(crossing_sample_counts[env_id].item()),
                            'peak_tilt_deg': float(crossing_peak_tilt_deg[env_id].item()) if crossed else None,
                            'peak_angular_velocity_norm_rad_s': float(crossing_peak_angular_speed[env_id].item()) if crossed else None,
                        })
                    crossing_time[reset_envs] = float("nan")
                    crossing_peak_tilt_deg[reset_envs] = 0.0
                    crossing_peak_angular_speed[reset_envs] = 0.0
                    crossing_sample_counts[reset_envs] = 0
                    crossing_window_complete[reset_envs] = False

                if record_video:
                    env.env._handle_resets(dones)

                crash_counts[crashed_now] += 1
                success_counts[succeeded_now] += 1
                timeout_counts[timeout_now] += 1

                all_episode_rewards.append(episode_rewards[reset_envs].clone())
                episode_rewards[reset_envs] = 0.0
                env_time[reset_envs] = 0.0
                reset_env_ids = torch.where(reset_envs)[0]
                if env.env.sim.motor_model != "motors_disabled":
                    reset_motor_action = env.env.hover_action.unsqueeze(0).expand(len(reset_env_ids), -1)
                    env.env.sim.rotor_dynamics.reset_idx(reset_env_ids, reset_motor_action)
                    env.env.sim._last_motor_actions[reset_env_ids] = reset_motor_action
                controller.restart_acados(reset_env_ids)
                ref_traj.reset(reset_env_ids)
                controller.reset(reset_env_ids)
                if args_cli.collect_crossing_window:
                    reset_pos, reset_quat, _, reset_ang_vel = sim.get_base_state()
                    previous_x[reset_env_ids] = (
                        reset_pos[reset_env_ids, 0] - sim.scene.env_origins[reset_env_ids, 0]
                    )
                    previous_tilt_deg[reset_env_ids] = torch.rad2deg(torch.acos(torch.clamp(
                        1.0 - 2.0 * (
                            reset_quat[reset_env_ids, 1].square() + reset_quat[reset_env_ids, 2].square()
                        ), -1.0, 1.0
                    )))
                    previous_angular_speed[reset_env_ids] = torch.linalg.vector_norm(
                        reset_ang_vel[reset_env_ids], dim=-1
                    )

                episode_counts[reset_envs] += 1

                completed_envs = episode_counts >= num_episodes
                active_envs = ~completed_envs

            step_count += 1
            now = time.time()
            if now - last_progress_time >= 60:
                last_progress_time = now
                elapsed = now - progress_start_time
                completed = episode_counts.sum().item()
                pct = 100 * completed / total_target_episodes
                if completed > 0:
                    remaining = elapsed * (total_target_episodes / completed - 1)
                    eta_dt = datetime.now() + __import__('datetime').timedelta(seconds=remaining)
                    eta_str = eta_dt.strftime("%H:%M:%S") if eta_dt.date() == datetime.now().date() else eta_dt.strftime("%b %d %H:%M:%S")
                    print(f"[PROGRESS] {pct:.1f}% ({int(completed)}/{total_target_episodes} episodes) | "
                          f"Elapsed: {elapsed/60:.1f}min | Remaining: {remaining/60:.1f}min | ETA: {eta_str}")
                else:
                    print(f"[PROGRESS] {pct:.1f}% | Elapsed: {elapsed/60:.1f}min")

    if record_video:
        env.env.auto_reset = True

    if recorder:
        recorder.save()
    if animation_capture is not None:
        animation_capture.save()

    if nmpc_logger is not None:
        nmpc_logger.close()

    if metrics_log_file:
        metrics_log_file.close()
        print(f"[INFO] Metrics log saved")

    incomplete_envs = episode_counts < num_episodes
    if incomplete_envs.any():
        incomplete_indices = torch.where(incomplete_envs)[0].tolist()
        incomplete_counts = episode_counts[incomplete_envs].tolist()
        raise RuntimeError(
            f"Evaluation ended with incomplete environments: indices={incomplete_indices}, "
            f"episode_counts={incomplete_counts}, required={num_episodes}"
        )

    total_episodes = episode_counts.sum().item()
    if args_cli.crossing_trace_output:
        with Path(args_cli.crossing_trace_output).open("w") as handle:
            json.dump(crossing_trace_records, handle, allow_nan=False)
    total_crashes = crash_counts.sum().item()
    total_successes = success_counts.sum().item()
    total_timeouts = timeout_counts.sum().item()

    gap_stats = env.env.get_gap_statistics(episode_counts=episode_counts, success_counts=success_counts)

    print("\n" + "="*80)
    print("NMPC Wall Gap Evaluation Results")
    print("="*80)
    print(f"Total steps: {step_count}, Total episodes: {total_episodes}")
    print()
    print("Overall Episode Outcomes:")
    print(f"  Successes: {total_successes:4d} ({100*total_successes/total_episodes:5.1f}%)")
    print(f"  Timeouts:  {total_timeouts:4d} ({100*total_timeouts/total_episodes:5.1f}%)")
    print(f"  Crashes:   {total_crashes:4d} ({100*total_crashes/total_episodes:5.1f}%)")
    print()

    success_pcts = 100 * success_counts.float() / num_episodes
    timeout_pcts = 100 * timeout_counts.float() / num_episodes
    crash_pcts = 100 * crash_counts.float() / num_episodes
    print(f"Per-Environment Statistics ({num_episodes} episodes each):")
    print(f"  Success rate: min={success_pcts.min().item():.1f}%, max={success_pcts.max().item():.1f}%, mean={success_pcts.mean().item():.1f}%, median={success_pcts.median().item():.1f}%")
    print(f"  Timeout rate: min={timeout_pcts.min().item():.1f}%, max={timeout_pcts.max().item():.1f}%, mean={timeout_pcts.mean().item():.1f}%, median={timeout_pcts.median().item():.1f}%")
    print(f"  Crash rate:   min={crash_pcts.min().item():.1f}%, max={crash_pcts.max().item():.1f}%, mean={crash_pcts.mean().item():.1f}%, median={crash_pcts.median().item():.1f}%")
    print()

    print(f"{'Gap Width (m)':<15} {'Episodes':<12} {'Successes':<12} {'Success Rate (%)':<18}")
    print("-"*80)
    for stat in gap_stats['per_gap_stats']:
        print(f"{stat['gap_width']:<15.3f} {stat['num_episodes']:<12} {stat['num_successes']:<12} {stat['success_rate']*100:<18.1f}")
    print("-"*80)

    gap_widths = [s['gap_width'] for s in gap_stats['per_gap_stats']]
    success_rates = [s['success_rate'] * 100 for s in gap_stats['per_gap_stats']]

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "nmpc_success_rate_vs_gap_width.pdf"
    if args_cli.metrics_output:
        plot_path = Path(args_cli.metrics_output).with_suffix(".pdf")
    plt.figure(figsize=(5, 3))
    plt.plot(gap_widths, success_rates, marker='')
    plt.xlabel('Gap Width (m)')
    plt.ylabel('Success Rate (%)')
    plt.title('NMPC: Success Rate vs Gap Width')
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[INFO] Plot saved to {plot_path}")

    if gap_stats['smallest_100_percent_gap'] is not None:
        print(f"\nSmallest gap with 100% success: {gap_stats['smallest_100_percent_gap']:.3f}m")
    else:
        print(f"\nNo gap achieved 100% success rate")

    if gap_stats['largest_not_100_percent_gap'] > 0.0:
        print(f"Largest gap without 100% success: {gap_stats['largest_not_100_percent_gap']:.3f}m")
    else:
        print(f"All gaps achieved 100% success rate!")

    mean_position_error = position_error_sum / tracking_error_count if tracking_error_count > 0 else 0.0
    mean_velocity_error = velocity_error_sum / tracking_error_count if tracking_error_count > 0 else 0.0
    mean_attitude_error = attitude_error_sum / tracking_error_count if tracking_error_count > 0 else 0.0
    mean_control_effort = control_effort_sum / tracking_error_count if tracking_error_count > 0 else 0.0
    print(f"\nMean position error (vs setpoint): {mean_position_error:.4f} m")
    print(f"Mean velocity error (vs setpoint): {mean_velocity_error:.4f} m/s")
    print(f"Mean attitude error (vs desired):  {mean_attitude_error:.4f}")
    print(f"Mean control effort:               {mean_control_effort:.4f}")

    all_rewards_tensor = torch.cat(all_episode_rewards) if all_episode_rewards else torch.tensor([0.0], device=env.device)
    average_reward = all_rewards_tensor.mean().item()
    print(f"Average episode reward: {average_reward:.4f}")
    print("="*80 + "\n")

    return {
        'traversal_episodes': traversal_episodes,
        'crossing_window_episodes': crossing_window_episodes,
        'crossing_window_s': [0.0, 0.2] if args_cli.collect_crossing_window else None,
        'traversal_tilt_crash_disabled': args_cli.disable_traversal_tilt_crash,
        'gap_statistics': gap_stats['per_gap_stats'],
        'smallest_100_percent_gap': gap_stats['smallest_100_percent_gap'],
        'largest_not_100_percent_gap': gap_stats['largest_not_100_percent_gap'],
        'total_episodes': total_episodes,
        'total_successes': total_successes,
        'total_timeouts': total_timeouts,
        'total_crashes': total_crashes,
        'success_percentage': 100 * total_successes / total_episodes,
        'timeout_percentage': 100 * total_timeouts / total_episodes,
        'crash_percentage': 100 * total_crashes / total_episodes,
        'per_env_success_rate_stats': {
            'min': success_pcts.min().item(),
            'max': success_pcts.max().item(),
            'mean': success_pcts.mean().item(),
            'median': success_pcts.median().item(),
        },
        'per_env_timeout_rate_stats': {
            'min': timeout_pcts.min().item(),
            'max': timeout_pcts.max().item(),
            'mean': timeout_pcts.mean().item(),
            'median': timeout_pcts.median().item(),
        },
        'per_env_crash_rate_stats': {
            'min': crash_pcts.min().item(),
            'max': crash_pcts.max().item(),
            'mean': crash_pcts.mean().item(),
            'median': crash_pcts.median().item(),
        },
        'mean_position_error': mean_position_error,
        'mean_velocity_error': mean_velocity_error,
        'mean_attitude_error': mean_attitude_error,
        'mean_control_effort': mean_control_effort,
        'average_reward': average_reward,
    }


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    usd_path = Path(args_cli.usd_path)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    env, special_flags, wall_gap_config = create_env(args_cli.num_envs, usd_path, args_cli.config_override)
    if args_cli.disable_traversal_tilt_crash:
        env.env.x_close_to_zero_zone_termination_max_misalignment_x = float("inf")
        env.env.x_close_to_zero_zone_termination_max_misalignment_y = float("inf")
        env.env.x_close_to_zero_zone_termination_max_misalignment_z = float("inf")
        print("[INFO] Gap traversal orientation crash criterion disabled")
    device = env.device
    sim = env.env.sim

    nmpc_cfg_path = Path(args_cli.nmpc_config) if args_cli.nmpc_config else Path(__file__).parent.parent / "configs/nmpc_controller_cfg_simulation.yaml"
    with open(nmpc_cfg_path, 'r') as f:
        nmpc_cfg_all = yaml.safe_load(f)

    nmpc_cfg = nmpc_cfg_all['nmpc']
    traj_cfg = nmpc_cfg_all['trajectory']

    motor_idx_list = AIRFRAME_CONSTANTS['motor_idx_list']
    motor_directions = AIRFRAME_CONSTANTS['motor_directions']
    sim_cfg = wall_gap_config['simulator']

    from rotor_dynamics import RotorDynamics, manufacturerComponentData
    component_data = manufacturerComponentData(device, motor_idx_list)
    motor_force_min = component_data.min_force.min().item()
    motor_force_max = component_data.max_force.max().item()
    if args_cli.motor_constraint_rps_min is not None:
        rps_min = args_cli.motor_constraint_rps_min
        if not np.isfinite(rps_min) or not component_data.min_rps.max().item() <= rps_min < component_data.max_rps.min().item():
            raise ValueError("Requested motor-speed constraint is outside the motor range")
        coefficients = component_data.thrust_coefficient
        if not torch.allclose(coefficients, coefficients[0].expand_as(coefficients)):
            raise ValueError("A shared force constraint requires identical motor thrust coefficients")
        motor_force_min = coefficients[0].item() * rps_min**2
        print(f"[INFO] Explicit NMPC motor bound: {rps_min} rps, {motor_force_min} N")

    controller_dt = sim_cfg['dt'] * wall_gap_config['environment']['action_decimation']
    controller = NMPCController(
        num_envs=args_cli.num_envs,
        device=device,
        dt=controller_dt,
        nmpc_dt=nmpc_cfg['nmpc_dt'],
        N=nmpc_cfg['N'],
        Q_pos=nmpc_cfg['Q_pos'],
        Q_vel=nmpc_cfg['Q_vel'],
        Q_acc=nmpc_cfg['Q_acc'],
        Q_att=nmpc_cfg['Q_att'],
        Q_omega=nmpc_cfg['Q_omega'],
        R=nmpc_cfg['R'],
        R_delta_u=nmpc_cfg['R_delta_u'],
        terminal_factor=nmpc_cfg['terminal_factor'],
        recovery_cost_config=nmpc_cfg['recovery'],
        motor_rps_min=nmpc_cfg['motor_rps_min'],
        motor_force_min=motor_force_min,
        motor_force_max=motor_force_max,
        thrust_coefficients=component_data.thrust_coefficient.cpu().numpy(),
        motor_time_constants=component_data.motor_time_constant.cpu().numpy(),
    )
    try:
        print(f"[INFO] NMPC controller initialized: N={nmpc_cfg['N']}, nmpc_dt={nmpc_cfg['nmpc_dt']}, controller_dt={controller_dt}, horizon={nmpc_cfg['N']*nmpc_cfg['nmpc_dt']:.2f}s, motor force range=[{motor_force_min:.4f}, {motor_force_max:.4f}]")

        allocator = ControlAllocator(
            num_envs=args_cli.num_envs,
            dt=sim_cfg['dt'],
            motor_idx_list=motor_idx_list,
            device=device,
            motor_directions=motor_directions,
            motor_model=sim_cfg['motor_model'],
        )
        rotor_dynamics = allocator.motor_dynamics

        ref_traj = CollisionReferenceTrajectory(
            env_origins=sim.scene.env_origins,
            wall_x_position=env.env.wall_x_position,
            collision_config=traj_cfg['collision'],
            device=device,
        )
        ref_traj.require_direct_actuator_control()

        metrics = evaluate_nmpc(env, controller, ref_traj, allocator, rotor_dynamics, args_cli.num_episodes, args_cli.record_video, special_flags)

        if args_cli.metrics_output:
            with open(args_cli.metrics_output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"[INFO] Metrics saved to {args_cli.metrics_output}")

        print("[INFO] Evaluation complete. Exiting.")
        sys.exit(0)
    finally:
        print("[INFO] Closing NMPC solver workers", flush=True)
        controller.close()
        print("[INFO] NMPC solver workers closed", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        from isaaclab.sim import SimulationContext
        try:
            simulation_context = SimulationContext.instance()
            if simulation_context is not None:
                simulation_context.clear_all_callbacks()
                simulation_context.clear_instance()
        finally:
            simulation_app.close()
