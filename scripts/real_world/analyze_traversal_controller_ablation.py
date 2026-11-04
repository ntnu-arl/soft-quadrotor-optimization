"""Run fixed-seed NMPC-weight ablations without changing vehicle or motor physics.

Use --cases to select explicit conditions. real_weights_floor changes only the
coast-speed YAML setting. real_weights_bound200 also sets the active NMPC force
constraint to the force corresponding to 200 rps. Output paths must be unused.
"""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/traversal_20cm_sim2real"
OUTPUT = SOURCE / "controller_ablation"
PYTHON = sys.executable
REAL_RUNS = sorted((ROOT / "results/real_world_experiments/voxl_mocap").glob("2026_09_09_*_x10_transversal_20cm_0deg*"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, nargs="+", choices=("baseline", "delta_only", "real_weights", "real_weights_floor", "real_weights_bound200"))
    args = parser.parse_args()
    if len(REAL_RUNS) != 5:
        raise ValueError("Expected the five repeated-traversal flight logs")
    real = [yaml.safe_load((run / "nmpc_controller_cfg.yaml").read_text())["nmpc"] for run in REAL_RUNS]
    if any(config != real[0] for config in real[1:]):
        raise ValueError("Real-world controller parameters differ between flights")
    baseline = yaml.safe_load((SOURCE / "nmpc_controller_cfg_simulation.yaml").read_text())
    configs = {"baseline": copy.deepcopy(baseline), "delta_only": copy.deepcopy(baseline),
               "real_weights": copy.deepcopy(baseline), "real_weights_floor": copy.deepcopy(baseline)}
    configs["delta_only"]["nmpc"]["R_delta_u"] = real[0]["R_delta_u"]
    configs["delta_only"]["nmpc"]["recovery"]["R_delta_u"] = real[0]["recovery"]["R_delta_u"]
    for key in baseline["nmpc"]:
        if key != "motor_rps_min":
            configs["real_weights"]["nmpc"][key] = copy.deepcopy(real[0][key])
            configs["real_weights_floor"]["nmpc"][key] = copy.deepcopy(real[0][key])
    configs["real_weights_floor"]["nmpc"]["motor_rps_min"] = real[0]["motor_rps_min"]
    configs["real_weights_bound200"] = copy.deepcopy(configs["real_weights_floor"])
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    design = next(run for run in manifest["runs"] if run["design"] == "800")
    OUTPUT.mkdir(exist_ok=True)
    provenance = {"seed": 1043, "num_envs": 100, "episodes_per_env": 1,
                  "real_runs": [str(run) for run in REAL_RUNS], "runs": []}
    for name, config in configs.items():
        if name not in args.cases:
            continue
        path = OUTPUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        metrics = OUTPUT / f"{name}_metrics.json"
        if metrics.exists():
            raise FileExistsError(f"Existing ablation output: {metrics}")
        command = [PYTHON, "-u", "scripts/nmpc_enjoy.py", "--headless", "--device", "cuda:0",
                   "--kit_args=--/log/file=/dev/null", "--num_envs", "100", "--num_episodes", "1",
                   "--seed", "1043", "--usd_path", str(ROOT / design["usd_path"]),
                   "--metrics_output", str(metrics), "--task_config", str(SOURCE / "lee_wall_gap_simulation.yaml"),
                   "--nmpc_config", str(path), "--disable_traversal_tilt_crash", "--collect_crossing_window",
                   "--crossing_trace_output", str(OUTPUT / f"{name}_traces.json"),
                   "--config-override", "environment.min_gap_width=0.20",
                   "--config-override", "environment.max_gap_width=0.20",
                   "--config-override", "environment.episode_length_s=4.0"]
        if name == "real_weights_bound200":
            command.extend(["--motor_constraint_rps_min", str(real[0]["motor_rps_min"])])
        started = time.monotonic()
        print(f"Running {name}", flush=True)
        with (OUTPUT / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        provenance["runs"].append({"name": name, "command": command, "wall_time_s": time.monotonic()-started})
        (OUTPUT / ("manifest_" + "_".join(args.cases) + ".json")).write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Finished {name}", flush=True)


if __name__ == "__main__":
    main()
