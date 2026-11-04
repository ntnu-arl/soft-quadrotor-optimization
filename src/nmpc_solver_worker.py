"""CPU-only acados worker code; never import the simulator or torch here."""

import traceback

from process_lifecycle import exit_with_owner

import numpy as np
from acados_template import AcadosOcpSolver

NUM_MOTORS = 4
PARAM_MASS = 0
PARAM_INERTIA_START = 1
PARAM_GRAVITY_START = 7
PARAM_POS_REF_START = 10
PARAM_VEL_REF_START = 13
PARAM_ACC_REF_START = 16
PARAM_QUAT_REF_START = 19
PARAM_ALLOC_START = 23
PARAM_ALLOC_SIZE = 6 * NUM_MOTORS
PARAM_KF_START = PARAM_ALLOC_START + PARAM_ALLOC_SIZE
PARAM_TAU_START = PARAM_KF_START + NUM_MOTORS
PARAM_COM_START = PARAM_TAU_START + NUM_MOTORS
PARAM_DIST_FORCE_START = PARAM_COM_START + 3
PARAM_DIST_TORQUE_START = PARAM_DIST_FORCE_START + 3
PARAM_U_PREV_START = PARAM_DIST_TORQUE_START + 3
PARAM_COUNT = PARAM_U_PREV_START + NUM_MOTORS


def _reinitialize_acados_solver(solver, horizon, x0, hover_force):
    x_init = np.array(x0, dtype=np.float64, copy=True)
    u_init = np.full(NUM_MOTORS, hover_force, dtype=np.float64)
    for k in range(horizon):
        solver.set(k, "x", x_init)
        solver.set(k, "u", u_init)
    solver.set(horizon, "x", x_init)


def _solve_acados_problem(solver, horizon, x0, params_by_stage, hover_force, needs_reinit):
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    params_by_stage = np.ascontiguousarray(params_by_stage, dtype=np.float64)
    if params_by_stage.shape != (horizon + 1, PARAM_COUNT):
        raise ValueError(
            f"expected params_by_stage shape {(horizon + 1, PARAM_COUNT)}, got {params_by_stage.shape}"
        )

    if needs_reinit:
        _reinitialize_acados_solver(solver, horizon, x0, hover_force)

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    for k in range(horizon):
        solver.set(k, "p", params_by_stage[k])
    solver.set(horizon, "p", params_by_stage[horizon])

    status = solver.solve()
    solver_stats = np.asarray(solver.get_stats("statistics"), dtype=np.float64)
    qp_status = int(round(float(solver_stats[1, -1])))

    u_opt = np.array(solver.get(0, "u"), dtype=np.float64, copy=True)
    x1 = np.array(solver.get(1, "x"), dtype=np.float64, copy=True)
    x_end = np.array(solver.get(horizon, "x"), dtype=np.float64, copy=True)
    horizon_controls = np.zeros((horizon, NUM_MOTORS), dtype=np.float64)
    for k in range(horizon):
        horizon_controls[k] = solver.get(k, "u")

    return {
        "status": int(status),
        "qp_status": qp_status,
        "cost": float(solver.get_cost()),
        "u_opt": u_opt,
        "x1": x1,
        "x_end": x_end,
        "horizon_controls": horizon_controls,
    }


def _acados_solver_worker(env_i, connection, json_file, horizon, owner_pid):
    exit_with_owner(owner_pid)
    solver = None
    try:
        solver = AcadosOcpSolver(None, json_file=json_file, build=False, generate=False, verbose=False)
        connection.send(("ready", None))

        while True:
            message = connection.recv()
            command = message[0]
            if command == "solve":
                _, x0, params_by_stage, hover_force, needs_reinit = message
                result = _solve_acados_problem(solver, horizon, x0, params_by_stage, hover_force, needs_reinit)
                connection.send(("result", result))
            elif command == "close":
                connection.send(("closed", None))
                break
            else:
                raise RuntimeError(f"unknown acados worker command for env {env_i}: {command}")
    except BaseException as exc:
        payload = {
            "env_i": env_i,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            connection.send(("error", payload))
        except BaseException:
            pass
        raise
    finally:
        if solver is not None:
            del solver
        connection.close()
