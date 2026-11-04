"""
Airframe Objective Functions for Morphy Optimization

This module provides functions to:
1. Generate USD files from normalized parameters
2. Train RL policies for different airframe configurations
3. Evaluate trained policies
4. Compute objective function values for optimization
"""

import os
import signal
import subprocess
import sys
import json
import uuid
import torch
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List

from airframe_encoding import PARAM_NAMES


def _terminate_process_group(process, process_label):
    print(f"[{process_label}] Detected completion, terminating process group...")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


TORQUE_FN_CMD_ARG_KEYS = ("torque_fn_left", "torque_fn_right")


def split_torque_fn_cmd_args(cmd_args: Optional[Dict[str, float]]) -> tuple[str, str, Dict[str, float]]:
    if cmd_args is None:
        raise ValueError("cmd_args must include torque_fn_left and torque_fn_right")
    missing = [key for key in TORQUE_FN_CMD_ARG_KEYS if key not in cmd_args]
    if missing:
        raise ValueError(f"cmd_args missing required torque response strings: {missing}")
    filtered_cmd_args = {key: value for key, value in cmd_args.items() if key not in TORQUE_FN_CMD_ARG_KEYS}
    return str(cmd_args["torque_fn_left"]), str(cmd_args["torque_fn_right"]), filtered_cmd_args


def decode_into_airframe(
    x: Union[List[float], Dict[str, float], "numpy.ndarray"],
    workspace_dir: Union[str, Path],
    torque_fn_left: Optional[str],
    torque_fn_right: Optional[str],
    rigid: bool,
) -> Path:
    """Generate USD file from parameters.

    Args:
        x: Either:
           - List/array of normalized parameters in [0,1] range
           - Dict with parameter names (e.g., {"front_arm_length": 0.1, ...})
        workspace_dir: Directory for this airframe's workspace (created if needed)
        torque_fn_left: Left arm torque response code string, None when rigid
        torque_fn_right: Right arm torque response code string, None when rigid
        rigid: Generate the airframe with fixed side arm joints

    Returns:
        Path to generated USD file

    Raises:
        ValueError: If x doesn't contain correct parameters
        RuntimeError: If USD generation fails
    """
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).parent.parent / "scripts/convert_morphy_urdf.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--headless",
        "--output_dir", str(workspace_dir),
    ]
    if rigid:
        if torque_fn_left is not None or torque_fn_right is not None:
            raise ValueError("Rigid airframes must not receive torque response strings")
        cmd.append("--rigid")
    else:
        if torque_fn_left is None or torque_fn_right is None:
            raise ValueError("Soft airframes require torque_fn_left and torque_fn_right")
        cmd.extend(["--torque_fn_left", torque_fn_left, "--torque_fn_right", torque_fn_right])

    if isinstance(x, dict):
        for key, value in x.items():
            cmd.extend([f"--{key}", str(float(value))])
        print(f"[decode_into_airframe] Generating USD with parameters: {x}")
        params_to_save = x
    else:
        if len(x) != len(PARAM_NAMES):
            raise ValueError(f"Expected {len(PARAM_NAMES)} parameters, got {len(x)}")
        cmd.extend(["--params_list"] + [str(float(v)) for v in x])
        print(f"[decode_into_airframe] Generating USD with parameters: {[round(float(v), 3) for v in x]}")
        params_to_save = {"x_0_1": [float(v) for v in x]}

    print(f">> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"USD generation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    workspace_usd_path = workspace_dir / "morphy_prog.usd"

    if not workspace_usd_path.exists():
        raise RuntimeError(f"USD file not generated at expected location: {workspace_usd_path}")

    params_file = workspace_dir / "params.json"
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)

    print(f"[decode_into_airframe] USD saved to: {workspace_usd_path}")
    return workspace_usd_path


def eval_airframe_nmpc(
    workspace_dir: Union[str, Path],
    seed: int,
    num_envs: int,
    cmd_args: Optional[Dict[str, float]],
    device: Optional[str],
    wall_gap_size_tuple: tuple,
    num_episodes: int,
) -> Dict:
    workspace_dir = Path(workspace_dir)
    usd_path = workspace_dir / "morphy_prog.usd"

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    metrics_output = workspace_dir / "metrics_nmpc.json"
    script_path = Path(__file__).parent.parent / "scripts/nmpc_enjoy.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--headless",
        "--num_envs", str(num_envs),
        "--seed", str(seed),
        "--num_episodes", str(num_episodes),
        "--metrics_output", str(metrics_output),
        "--usd_path", str(usd_path),
        "--config-override", f"environment.min_gap_width={wall_gap_size_tuple[0]}",
        "--config-override", f"environment.max_gap_width={wall_gap_size_tuple[1]}",
        "--config-override", "log_metrics=True",
    ]

    if device is not None:
        cmd.extend(["--device", device])

    if cmd_args:
        for key, value in cmd_args.items():
            cmd.extend(["--config-override", f"{key}={value}"])

    print(f"[eval_airframe_nmpc] Evaluating with NMPC, {num_envs} envs (seed={seed})")
    print(f">> {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    saw_completion_message = False
    try:
        for line in process.stdout:
            print(line, end='')
            if "[INFO] Evaluation complete. Exiting." in line:
                saw_completion_message = True
                _terminate_process_group(process, "eval_airframe_nmpc")
                break
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if process.poll() is None:
        process.wait()

    allowed_return_codes = [0, -15, -9] if saw_completion_message else [0]
    if process.returncode not in allowed_return_codes:
        raise RuntimeError(f"NMPC evaluation failed with return code {process.returncode}")

    if not metrics_output.exists():
        raise RuntimeError(f"Metrics file not created: {metrics_output}")

    with open(metrics_output, 'r') as f:
        metrics = json.load(f)

    print(f"[eval_airframe_nmpc] Success rate: {metrics['success_percentage']:.2f}%")
    return metrics


def generate_video_airframe_nmpc(
    workspace_dir: Union[str, Path],
    seed: int,
    num_episodes: int,
    playback_speed: float,
    cmd_args: Optional[Dict[str, float]],
    resolution: tuple[int, int],
    video_filename: Optional[str]
) -> Path:
    workspace_dir = Path(workspace_dir)
    usd_path = workspace_dir / "morphy_prog.usd"

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    script_path = Path(__file__).parent.parent / "scripts/nmpc_enjoy.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--usd_path", str(usd_path),
        "--record_video",
        "--playback_speed", str(playback_speed),
        "--num_envs", "1",
        "--num_episodes", str(num_episodes),
        "--seed", str(seed),
        "--resolution", str(resolution[0]), str(resolution[1])
    ]

    if video_filename:
        cmd.extend(["--video_filename", video_filename])

    if cmd_args:
        camera_args = {}
        config_args = {}
        for k, v in cmd_args.items():
            if k.startswith("camera."):
                camera_args[k.replace("camera.", "")] = v
            else:
                config_args[k] = v

        if "eye" in camera_args:
            cmd.extend(["--camera_eye"] + [str(x) for x in camera_args["eye"]])
        if "target" in camera_args:
            cmd.extend(["--camera_target"] + [str(x) for x in camera_args["target"]])

        for key, value in config_args.items():
            cmd.extend(["--config-override", f"{key}={value}"])

    print(f"[generate_video_airframe_nmpc] Recording {num_episodes} episodes at {playback_speed}x speed (seed={seed})")
    print(f">> {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    saw_completion_message = False
    try:
        for line in process.stdout:
            print(line, end='')
            if "[INFO] Evaluation complete. Exiting." in line:
                saw_completion_message = True
                _terminate_process_group(process, "generate_video_airframe_nmpc")
                break
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if process.poll() is None:
        process.wait()

    allowed_return_codes = [0, -15, -9] if saw_completion_message else [0]
    if process.returncode not in allowed_return_codes:
        raise RuntimeError(f"Video generation failed with return code {process.returncode}")

    results_dir = Path(__file__).parent.parent / "results"

    if video_filename:
        video_path = results_dir / f"{video_filename}.mp4"
        if not video_path.exists():
            raise RuntimeError(f"Video file not found: {video_path}")
    else:
        video_files = sorted(results_dir.glob("nmpc_wall_gap_eval_*.mp4"), key=lambda p: p.stat().st_mtime)
        if not video_files:
            raise RuntimeError(f"No video file found in {results_dir}")
        video_path = video_files[-1]

    print(f"[generate_video_airframe_nmpc] Video saved to: {video_path}")
    return video_path


def objective_function_nmpc(
    x: Union[List[float], Dict[str, float], "numpy.ndarray"],
    torque_fn_left: Optional[str],
    torque_fn_right: Optional[str],
    rigid: bool,
    seed: int,
    num_envs_eval: int,
    workspace_id: Optional[str],
    cmd_args: Optional[Dict[str, float]],
    device: Optional[str],
    wall_gap_size_tuple: tuple,
    num_episodes: int,
) -> Dict:
    if workspace_id is None:
        workspace_id = f"workspace_nmpc_{uuid.uuid4().hex[:8]}"

    base_dir = Path(__file__).parent.parent / "cache/workspace_local"
    workspace_dir = base_dir / workspace_id

    print("\n" + "="*80)
    print(f"OBJECTIVE FUNCTION NMPC: {workspace_id}")
    print("="*80)

    print("\n[1/2] Generating airframe USD...")
    decode_into_airframe(x, workspace_dir, torque_fn_left, torque_fn_right, rigid)

    print("\n[2/2] Evaluating with NMPC controller...")
    metrics = eval_airframe_nmpc(
        workspace_dir=workspace_dir,
        seed=seed,
        num_envs=num_envs_eval,
        cmd_args=cmd_args,
        device=device,
        wall_gap_size_tuple=wall_gap_size_tuple,
        num_episodes=num_episodes,
    )

    print("\n" + "="*80)
    print(f"OBJECTIVE FUNCTION NMPC COMPLETE: Success Rate = {metrics['success_percentage']:.2f}%")
    print(f"Workspace: {workspace_dir}")
    print("="*80 + "\n")

    metrics['workspace_dir'] = str(workspace_dir)
    return metrics
