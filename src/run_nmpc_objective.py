import argparse
import json
import signal
import sys
import traceback
from pathlib import Path

print("[run_nmpc_objective] Script started, parsing arguments...")
sys.stdout.flush()

_workspace_dir = None


def _signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"[run_nmpc_objective] Received signal {sig_name}", file=sys.stderr)
    sys.stderr.flush()
    if _workspace_dir:
        try:
            (_workspace_dir / "status.txt").write_text("failed")
            (_workspace_dir / "crash_reason.txt").write_text(f"python_signal_{sig_name}")
        except Exception:
            pass
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

sys.path.append(str(Path(__file__).parent))

from airframe_objective_functions import objective_function_nmpc


def main():
    global _workspace_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace_dir", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()

    workspace_dir = Path(args.workspace_dir)
    _workspace_dir = workspace_dir

    print(f"[run_nmpc_objective] Workspace: {workspace_dir}")
    sys.stdout.flush()

    status_file = workspace_dir / "status.txt"

    try:
        status_file.write_text("running")

        with open(workspace_dir / "job_config.json") as f:
            config = json.load(f)

        cmd_args = config["cmd_args"]
        wall_gap_size_tuple = tuple(cmd_args["wall_gap_size_tuple"])
        num_episodes = cmd_args["num_episodes"]
        rigid = cmd_args["rigid"]
        if not isinstance(rigid, bool):
            raise TypeError(f"cmd_args['rigid'] must be a bool, got {rigid!r}")
        if rigid:
            if "torque_fn_left" in cmd_args or "torque_fn_right" in cmd_args:
                raise ValueError("Rigid NMPC jobs must not include torque response strings")
            torque_fn_left = None
            torque_fn_right = None
        else:
            torque_fn_left = cmd_args["torque_fn_left"]
            torque_fn_right = cmd_args["torque_fn_right"]
        objective_cmd_args = {
            key: value
            for key, value in cmd_args.items()
            if key not in {"torque_fn_left", "torque_fn_right", "rigid", "wall_gap_size_tuple", "num_episodes"}
        }

        print(f"[run_nmpc_objective] workspace_id={config['workspace_id']}, num_envs_eval={config['num_envs_eval']}, seed={config['seed']}")
        sys.stdout.flush()

        metrics = objective_function_nmpc(
            x=config["x"],
            torque_fn_left=torque_fn_left,
            torque_fn_right=torque_fn_right,
            rigid=rigid,
            seed=config["seed"],
            num_envs_eval=config["num_envs_eval"],
            workspace_id=config["workspace_id"],
            cmd_args=objective_cmd_args,
            device=args.device,
            wall_gap_size_tuple=wall_gap_size_tuple,
            num_episodes=num_episodes,
        )

        with open(workspace_dir / "results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        status_file.write_text("completed")
        print(f"[run_nmpc_objective] Success rate: {metrics['success_percentage']:.2f}%")
        sys.stdout.flush()

    except Exception as e:
        status_file.write_text("failed")
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"[run_nmpc_objective] Failed: {error_msg}", file=sys.stderr)
        print(f"[run_nmpc_objective] Traceback:\n{tb}", file=sys.stderr)
        sys.stderr.flush()
        (workspace_dir / "crash_reason.txt").write_text(f"python_exception\n{error_msg}\n{tb}")
        raise


if __name__ == "__main__":
    main()
