"""
Generate videos of NMPC controller airframe results with multiple camera views and metrics.

Usage example:
    python scripts/generate_collision_video_nmpc.py --workspace_dir cache/workspace_local/test_nmpc --num_episodes 1 --playback_speed 0.25 --gap_width 0.25 --seed 42
"""

import argparse
import subprocess
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from airframe_objective_functions import generate_video_airframe_nmpc


def run_cmd(cmd: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    print(f">> {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=capture_output, text=True)


def crop_video(input_video: Path, output_video: Path, crop_percent_top: float, crop_percent_bottom: float):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0",
        str(input_video)
    ]
    result = run_cmd(cmd)
    width, height = map(int, result.stdout.strip().split(','))

    crop_top_px = int(height * crop_percent_top)
    crop_bottom_px = int(height * crop_percent_bottom)
    new_height = height - crop_top_px - crop_bottom_px

    cmd = [
        "ffmpeg", "-i", str(input_video),
        "-vf", f"crop={width}:{new_height}:0:{crop_top_px}",
        "-c:v", "libx264", "-crf", "18", "-y", str(output_video)
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed: {result.stderr}")


def generate_metrics_video(metrics_log_path: Path, output_video: Path, playback_speed: float, fps: int, width: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    temp_dir = output_video.parent / "temp_metrics_frames"
    temp_dir.mkdir(exist_ok=True)

    entries = []
    with open(metrics_log_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    if not entries:
        raise ValueError("No entries in metrics log")

    times = np.array([e['timestamp'] for e in entries])
    pos_errors = np.array([e['position_error'] for e in entries])
    velocities = np.array([np.linalg.norm(e['velocity']) for e in entries])
    joint_angles_1 = np.array([e['joint_angles'][0] for e in entries])
    joint_angles_2 = np.array([e['joint_angles'][1] for e in entries])

    max_time = times[-1]
    sim_time_per_video_frame = (1.0 / fps) * playback_speed

    video_frame_times = []
    frame_idx = 0
    while True:
        frame_time = frame_idx * sim_time_per_video_frame
        if frame_time > max_time:
            break
        video_frame_times.append(frame_time)
        frame_idx += 1

    height = 540
    dpi = 100
    figsize = (width / dpi, height / dpi)

    for video_frame_idx, target_time in enumerate(video_frame_times):
        fig, axes = plt.subplots(3, 1, figsize=figsize, facecolor='black')

        for ax in axes:
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.grid(True, alpha=0.3, color='white')

        mask = times <= target_time
        times_plot = times[mask]
        has_data = len(times_plot) > 0

        axes[0].plot(times_plot, pos_errors[mask], color='#00a2a2ff', linewidth=2)
        if has_data:
            axes[0].scatter(times_plot[-1], pos_errors[mask][-1], color='#00a2a2ff', s=80, zorder=5, edgecolors='white', linewidth=1)
        axes[0].set_ylabel('$|e|$ (m)', color='white', fontsize=11)
        axes[0].set_xlim([0, max_time])
        axes[0].set_ylim([0, max(pos_errors) * 1.1])
        axes[0].axvline(x=target_time, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        axes[0].set_title('Position Error', color='white', fontsize=12)

        axes[1].plot(times_plot, velocities[mask], color='#00a2a2ff', linewidth=2)
        if has_data:
            axes[1].scatter(times_plot[-1], velocities[mask][-1], color='#00a2a2ff', s=80, zorder=5, edgecolors='white', linewidth=1)
        axes[1].set_ylabel('$|v|$ (m/s)', color='white', fontsize=11)
        axes[1].set_xlim([0, max_time])
        axes[1].set_ylim([0, max(velocities) * 1.1])
        axes[1].axvline(x=target_time, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        axes[1].set_title('Velocity Magnitude', color='white', fontsize=12)

        axes[2].plot(times_plot, joint_angles_1[mask], color='#00a2a2ff', linewidth=2, label='Joint 1')
        axes[2].plot(times_plot, joint_angles_2[mask], color='#a9a9a9ff', linewidth=2, linestyle='--', label='Joint 2')
        if has_data:
            axes[2].scatter(times_plot[-1], joint_angles_1[mask][-1], color='#00a2a2ff', s=80, zorder=5, edgecolors='white', linewidth=1)
            axes[2].scatter(times_plot[-1], joint_angles_2[mask][-1], color='#a9a9a9ff', s=80, zorder=5, edgecolors='white', linewidth=1)
        axes[2].set_ylabel('Angle (deg)', color='white', fontsize=11)
        axes[2].set_xlabel('Time (s)', color='white', fontsize=11)
        axes[2].set_xlim([0, max_time])
        y_max = max(max(abs(joint_angles_1)), max(abs(joint_angles_2))) * 1.1
        axes[2].set_ylim([-y_max, y_max])
        axes[2].axvline(x=target_time, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        axes[2].legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8)
        axes[2].set_title('Joint Angles', color='white', fontsize=12)

        plt.tight_layout()
        frame_path = temp_dir / f"frame_{video_frame_idx:06d}.png"
        plt.savefig(frame_path, dpi=dpi, facecolor='black')
        plt.close()

    cmd = [
        "ffmpeg", "-framerate", str(fps), "-i", str(temp_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-y", str(output_video)
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    shutil.rmtree(temp_dir)
    print(f"[generate_metrics_video] Metrics video saved to: {output_video}")


def compose_four_videos(top_left: Path, bottom_left: Path, right: Path, output_path: Path):
    cmd = [
        "ffmpeg",
        "-i", str(top_left),
        "-i", str(bottom_left),
        "-i", str(right),
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2[left];[left][2:v]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "18",
        "-y",
        str(output_path)
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    print(f"[compose_four_videos] Composed video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate NMPC controller videos with multiple camera views")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Directory containing airframe USD")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for video generation")
    parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes to record")
    parser.add_argument("--playback_speed", type=float, required=True, help="Video playback speed multiplier")
    parser.add_argument("--gap_width", type=float, required=True, help="Wall gap width (sets both min and max)")
    args = parser.parse_args()

    workspace_dir = Path(args.workspace_dir)
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    speed_str = f"x{args.playback_speed}".replace(".", "_")

    print(f"\n{'='*80}")
    print(f"Generating NMPC videos for: {workspace_dir}")
    print(f"Gap width: {args.gap_width}m")
    print(f"Output directory: {results_dir}")
    print(f"{'='*80}\n")

    print("[1/5] Generating top view video with metrics logging...")
    top_video = generate_video_airframe_nmpc(
        workspace_dir=workspace_dir,
        seed=args.seed,
        num_episodes=args.num_episodes,
        playback_speed=args.playback_speed,
        resolution=(960, 1080),
        video_filename=f"{timestamp}_nmpc_{speed_str}_top_view",
        cmd_args={
            "camera.eye": [-0.25, 0.0, 2.0],
            "camera.target": [-0.25, 0.0, 0.0],
            "environment.min_gap_width": args.gap_width,
            "environment.max_gap_width": args.gap_width,
            "log_metrics": True,
        }
    )
    top_video_final = results_dir / top_video.name
    metrics_log_source = top_video.parent / f"{top_video.stem}_metrics.jsonl"
    metrics_log_final = results_dir / f"{top_video.stem}_metrics.jsonl"

    if top_video != top_video_final:
        shutil.move(str(top_video), str(top_video_final))
        top_video = top_video_final

    if metrics_log_source.exists() and metrics_log_source != metrics_log_final:
        shutil.move(str(metrics_log_source), str(metrics_log_final))

    print(f"Top view saved: {top_video}\n")

    print("[2/5] Generating back view video...")
    back_video = generate_video_airframe_nmpc(
        workspace_dir=workspace_dir,
        seed=args.seed,
        num_episodes=args.num_episodes,
        playback_speed=args.playback_speed,
        resolution=(960, 1080),
        video_filename=f"{timestamp}_nmpc_{speed_str}_back_view",
        cmd_args={
            "camera.eye": [-1.6, 0.0, 1.0],
            "camera.target": [-1.0, 0.0, 0.7343],
            "environment.min_gap_width": args.gap_width,
            "environment.max_gap_width": args.gap_width,
        }
    )
    back_video_final = results_dir / back_video.name
    if back_video != back_video_final:
        shutil.move(str(back_video), str(back_video_final))
        back_video = back_video_final
    print(f"Back view saved: {back_video}\n")

    print("[3/5] Generating metrics video...")
    if not metrics_log_final.exists():
        raise RuntimeError(f"Metrics log not found at {metrics_log_final}")
    metrics_video = results_dir / f"{timestamp}_nmpc_{speed_str}_metrics.mp4"
    generate_metrics_video(metrics_log_final, metrics_video, args.playback_speed, fps=60, width=960)
    print(f"Metrics video saved: {metrics_video}\n")

    print("[4/5] Cropping top view video (25% top, 25% bottom)...")
    top_video_cropped = results_dir / f"{timestamp}_nmpc_{speed_str}_top_view_cropped.mp4"
    crop_video(top_video, top_video_cropped, crop_percent_top=0.25, crop_percent_bottom=0.25)
    print(f"Cropped top view saved: {top_video_cropped}\n")

    print("[5/5] Composing final video...")
    composed_path = results_dir / f"{timestamp}_nmpc_{speed_str}_composed.mp4"
    compose_four_videos(top_video_cropped, metrics_video, back_video, composed_path)

    print(f"\n{'='*80}")
    print(f"NMPC video generation complete!")
    print(f"Composed: {composed_path}")
    print(f"{'='*80}\n")

    print("Cleaning up intermediate files...")
    for f in [top_video, back_video, top_video_cropped, metrics_video, metrics_log_final]:
        if f and f.exists():
            f.unlink()
            print(f"  Removed: {f}")


if __name__ == "__main__":
    main()
