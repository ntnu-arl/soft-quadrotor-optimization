"""
Generate visualization video of drone airframe with rotating camera.

This script:
1. Loads a drone airframe from a workspace directory
2. Rotates camera in a circle around the drone (2 full rotations in 10 seconds)
3. First rotation: joints at neutral position
4. Second rotation: progressively move joints to max position (self-collision limit)
5. Hold max position for 1 second
6. Return to neutral during remainder

Usage:
    python scripts/generate_airframe_visualization_video.py --workspace_dir cache/workspace_local/experiment_2025_11_20_cmaes_gen57_ind6
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_visualization_video(workspace_dir: str):
    """Generate airframe visualization video using subprocess.

    Args:
        workspace_dir: Directory containing the airframe USD file
    """
    script_path = Path(__file__).parent / "_generate_airframe_visualization_impl.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--workspace_dir", workspace_dir,
    ]

    print(f"[generate_airframe_visualization_video] Starting visualization...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')
        if "[INFO] Video generation complete" in line:
            print("[generate_airframe_visualization_video] Detected completion, terminating...")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break

    if process.poll() is None:
        process.wait()

    if process.returncode not in [0, -15, -9]:
        raise RuntimeError(f"Video generation failed with return code {process.returncode}")

    print("[generate_airframe_visualization_video] Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate airframe visualization video")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Directory containing airframe USD")
    args = parser.parse_args()

    generate_visualization_video(args.workspace_dir)


if __name__ == "__main__":
    main()
