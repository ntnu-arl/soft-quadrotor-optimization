"""Validate the fixed-gap simulations and export the first 50 successful windows."""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results/traversal_20cm_sim2real"
OUTPUT_DIR = PROJECT_ROOT / "results/figures/data"
RUNS = {
    "reference_planar": ("reference_planar_metrics.json", "sim_traversal_20cm_reference_planar.csv"),
    "50": ("checkpoint_50_metrics.json", "sim_traversal_20cm_50.csv"),
    "200": ("checkpoint_200_metrics.json", "sim_traversal_20cm_200.csv"),
    "400": ("checkpoint_400_metrics.json", "sim_traversal_20cm_400.csv"),
    "800": ("checkpoint_800_metrics.json", "sim_traversal_20cm.csv"),
    "800_high_fidelity": ("sim2real_plant_match/metrics_motor_real_fit.json", "sim_traversal_20cm_high_fidelity.csv"),
}
EXPECTED_ATTEMPTS = 100
EXPECTED_SELECTED = 50
EXPECTED_COMPLETE_WINDOW_SAMPLES = 22


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for design, (metrics_name, output_name) in RUNS.items():
        with (RESULTS_DIR / metrics_name).open() as handle:
            metrics = json.load(handle)
        episodes = sorted(
            metrics["crossing_window_episodes"],
            key=lambda episode: (episode["env_id"], episode["episode"]),
        )
        if len(episodes) != EXPECTED_ATTEMPTS or metrics["total_episodes"] != EXPECTED_ATTEMPTS:
            raise ValueError(f"{design}: expected {EXPECTED_ATTEMPTS} retained attempts")
        selected = []
        for episode in episodes:
            complete_window = (
                episode["crossing_time_s"] is not None
                and episode["window_samples_including_interpolated_crossing"]
                == EXPECTED_COMPLETE_WINDOW_SAMPLES
            )
            if episode["succeeded"] and complete_window:
                selected.append({
                    "design": design,
                    "env_id": episode["env_id"],
                    "episode": episode["episode"],
                    "gap_width_m": episode["gap_width"],
                    "success": True,
                    "complete_window": True,
                    "crossing_time_s": episode["crossing_time_s"],
                    "window_end_s": episode["window_end_s"],
                    "window_samples": episode["window_samples_including_interpolated_crossing"],
                    "max_tilt_deg": episode["peak_tilt_deg"],
                    "max_angular_speed_rad_s": episode["peak_angular_velocity_norm_rad_s"],
                })
            if len(selected) == EXPECTED_SELECTED:
                break
        if len(selected) != EXPECTED_SELECTED:
            raise ValueError(f"{design}: fewer than {EXPECTED_SELECTED} successful complete windows")
        with (OUTPUT_DIR / output_name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        print(f"{design}: exported {len(selected)} rows to {OUTPUT_DIR / output_name}")


if __name__ == "__main__":
    main()
