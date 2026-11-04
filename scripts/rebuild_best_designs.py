"""Regenerate the USD of each shipped best design into cache/workspace_local/<workspace_id>/.

The designs are rebuilt from best_job_config.json, the job that was evaluated during
the optimization, so each workspace can be passed to the evaluation and video scripts.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airframe_objective_functions import decode_into_airframe

RUNS_ROOT = PROJECT_ROOT / "results" / "behaviour_based_nmpc_cma_es"
WORKSPACE_ROOT = PROJECT_ROOT / "cache" / "workspace_local"


def main():
    job_config_paths = sorted(RUNS_ROOT.glob("*/best_job_config.json"))
    if not job_config_paths:
        raise FileNotFoundError(f"No best_job_config.json found under {RUNS_ROOT}")
    for job_config_path in job_config_paths:
        job_config = json.loads(job_config_path.read_text())
        cmd_args = job_config["cmd_args"]
        rigid = cmd_args["rigid"]
        workspace_dir = WORKSPACE_ROOT / job_config["workspace_id"]
        workspace_dir.mkdir(parents=True, exist_ok=True)
        decode_into_airframe(
            job_config["x"],
            workspace_dir,
            None if rigid else cmd_args["torque_fn_left"],
            None if rigid else cmd_args["torque_fn_right"],
            rigid,
        )
        print(f"{job_config_path.parent.name}: {workspace_dir / 'morphy_prog.usd'}")


if __name__ == "__main__":
    main()
