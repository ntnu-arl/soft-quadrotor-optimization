#!/usr/bin/env python

import os
import shutil
import sys
from pathlib import Path

import yaml
from acados_template import AcadosOcpSolver


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    src_dir = repo_root / "src"
    config_path = repo_root / "configs" / "nmpc_controller_cfg.yaml"
    json_path = repo_root / "acados_ocp_quadrotor.json"
    recovery_json_path = repo_root / "acados_ocp_quadrotor_recovery.json"
    codegen_dir = repo_root / "c_generated_code"
    recovery_codegen_dir = repo_root / "c_generated_code_recovery"
    acados_source_dir = os.environ["ACADOS_SOURCE_DIR"]

    if not Path(acados_source_dir).is_dir():
        raise FileNotFoundError(f"ACADOS_SOURCE_DIR does not exist: {acados_source_dir}")

    os.environ["ACADOS_SOURCE_DIR"] = acados_source_dir
    sys.path.insert(0, str(src_dir))

    from airframe_encoding import AIRFRAME_CONSTANTS
    from nmpc_controller import build_nmpc_ocp
    from rotor_dynamics import manufacturerComponentData

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nmpc_cfg = cfg["nmpc"]
    component_data = manufacturerComponentData("cpu", AIRFRAME_CONSTANTS["motor_idx_list"])
    motor_force_max = component_data.max_force.max().item()
    motor_rps_max = component_data.max_rps.max().item()
    motor_rps_floor_min = component_data.min_rps.max().item()

    motor_rps_min = nmpc_cfg["motor_rps_min"]
    if motor_rps_min < motor_rps_floor_min or motor_rps_min >= motor_rps_max:
        raise ValueError(
            f"nmpc.motor_rps_min={motor_rps_min} must lie in "
            f"[{motor_rps_floor_min}, {motor_rps_max})"
        )
    motor_force_min = motor_force_max * (motor_rps_min / motor_rps_max) ** 2

    for path in (codegen_dir, recovery_codegen_dir):
        if path.exists():
            shutil.rmtree(path)
    for path in (json_path, recovery_json_path):
        if path.exists():
            path.unlink()

    ocp, _ = build_nmpc_ocp(
        nmpc_cfg["nmpc_dt"],
        nmpc_cfg["N"],
        nmpc_cfg["Q_pos"],
        nmpc_cfg["Q_vel"],
        nmpc_cfg["Q_acc"],
        nmpc_cfg["Q_att"],
        nmpc_cfg["Q_omega"],
        nmpc_cfg["R"],
        nmpc_cfg["R_delta_u"],
        nmpc_cfg["terminal_factor"],
        motor_force_min,
        motor_force_max,
        "quadrotor_nmpc",
    )
    ocp.code_gen_opts.code_export_directory = str(codegen_dir)
    AcadosOcpSolver.generate(ocp, json_file=str(json_path))

    recovery_cfg = nmpc_cfg["recovery"]
    recovery_ocp, _ = build_nmpc_ocp(
        nmpc_cfg["nmpc_dt"],
        nmpc_cfg["N"],
        recovery_cfg["Q_pos"],
        recovery_cfg["Q_vel"],
        recovery_cfg["Q_acc"],
        recovery_cfg["Q_att"],
        recovery_cfg["Q_omega"],
        recovery_cfg["R"],
        recovery_cfg["R_delta_u"],
        recovery_cfg["terminal_factor"],
        motor_force_min,
        motor_force_max,
        "quadrotor_nmpc_recovery",
    )
    recovery_ocp.code_gen_opts.code_export_directory = str(recovery_codegen_dir)
    AcadosOcpSolver.generate(recovery_ocp, json_file=str(recovery_json_path))

    print(f"Generated {json_path}")
    print(f"Generated {codegen_dir}")
    print(f"Generated {recovery_json_path}")
    print(f"Generated {recovery_codegen_dir}")


if __name__ == "__main__":
    main()
