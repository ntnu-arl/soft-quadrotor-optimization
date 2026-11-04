"""Integration checks using the locally generated acados solver artifacts."""

import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_evaluation_import_does_not_start_simulator():
    code = (
        "import runpy, sys; "
        f"runpy.run_path({str(ROOT / 'scripts/nmpc_enjoy.py')!r}, run_name='__mp_main__'); "
        "assert 'torch' not in sys.modules; assert 'isaaclab.app' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def make_pool(num_envs):
    from nmpc_controller import ACADOS_JSON_FILE, ParallelAcadosOcpSolvers
    horizon = json.loads(Path(ACADOS_JSON_FILE).read_text())["dims"]["N"]
    return ParallelAcadosOcpSolvers(num_envs, ACADOS_JSON_FILE, horizon)


def test_restarts_release_workers_and_do_not_inherit_parent_memory():
    import psutil
    # Dirty parent memory must not appear in workers, including after restarts.
    parent_memory = bytearray(b"x") * (128 * 1024 * 1024)
    pool = make_pool(3)
    try:
        for _ in range(3):
            previous = list(pool._processes)
            for process in previous:
                maps = psutil.Process(process.pid).memory_maps()
                assert not any("libtorch" in mapping.path for mapping in maps)
                assert psutil.Process(process.pid).memory_full_info().uss < len(parent_memory)
            pool.restart_workers([0, 1, 2])
            assert all(not process.is_alive() for process in previous)
        workers = list(pool._processes)
    finally:
        pool.close()
    assert all(not process.is_alive() for process in workers)


def test_parallel_solve_matches_in_process_solver_after_restart():
    import numpy as np
    from acados_template import AcadosOcpSolver
    from nmpc_solver_worker import (
        PARAM_COUNT, PARAM_QUAT_REF_START, PARAM_ALLOC_START, PARAM_KF_START,
        PARAM_TAU_START, PARAM_U_PREV_START, _solve_acados_problem,
    )
    pool = make_pool(2)
    try:
        params = np.zeros((pool.horizon + 1, PARAM_COUNT))
        params[:, 0] = 1.0
        params[:, [1, 4, 6]] = 0.01
        params[:, 9] = -9.81
        params[:, PARAM_QUAT_REF_START] = 1.0
        allocation = np.array([
            [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1],
            [0.1, -0.1, -0.1, 0.1], [-0.1, -0.1, 0.1, 0.1],
            [0.01, -0.01, 0.01, -0.01],
        ])
        params[:, PARAM_ALLOC_START:PARAM_ALLOC_START + 24] = allocation.ravel(order="F")
        params[:, PARAM_KF_START:PARAM_KF_START + 4] = 1e-5
        params[:, PARAM_TAU_START:PARAM_TAU_START + 4] = 0.02
        hover = 9.81 / 4
        params[:, PARAM_U_PREV_START:PARAM_U_PREV_START + 4] = hover
        x0 = np.zeros(17)
        x0[6] = 1
        x0[13:] = np.sqrt(hover / 1e-5)
        solver = AcadosOcpSolver(None, json_file=pool.json_file, build=False, generate=False, verbose=False)
        expected = _solve_acados_problem(solver, pool.horizon, x0, params, hover, True)
        assert expected["status"] == 0
        for _ in range(2):
            actual = pool.solve_all(
                np.tile(x0, (2, 1)), np.tile(params, (2, 1, 1)),
                np.full(2, hover), np.ones(2, dtype=bool),
            )
            assert np.all(actual["solver_statuses"] == 0)
            for key in ("u_opt", "x1", "x_end", "horizon_controls"):
                for env_i in range(2):
                    np.testing.assert_allclose(actual[key][env_i], expected[key], rtol=1e-10, atol=1e-10)
            pool.restart_workers([0, 1])
    finally:
        pool.close()


def test_worker_failure_cleans_entire_pool():
    pool = make_pool(3)
    workers = list(pool._processes)
    workers[0].kill()
    workers[0].join(10)
    with pytest.raises(RuntimeError):
        pool.close()
    assert all(not process.is_alive() for process in workers)


def test_partial_startup_cleans_workers(monkeypatch):
    from nmpc_controller import ParallelAcadosOcpSolvers
    original = ParallelAcadosOcpSolvers._start_worker
    started = []

    def fail_on_second(self, env_i):
        if env_i == 1:
            raise RuntimeError("injected startup failure")
        original(self, env_i)
        started.extend(self._processes)

    monkeypatch.setattr(ParallelAcadosOcpSolvers, "_start_worker", fail_on_second)
    with pytest.raises(RuntimeError, match="injected startup failure"):
        make_pool(3)
    assert started and all(not process.is_alive() for process in started)


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL])
def test_workers_exit_when_owner_is_killed(stop_signal):
    import psutil
    owner = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "owner"],
        stdout=subprocess.PIPE, text=True,
    )
    workers = []
    try:
        line = owner.stdout.readline()
        assert line, "owner exited without reporting workers"
        workers = [psutil.Process(pid) for pid in json.loads(line)]
        owner.send_signal(stop_signal)
        owner.wait(timeout=15)
        _, alive = psutil.wait_procs(workers, timeout=15)
        assert not alive
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=15)
        owner.stdout.close()
        for worker in workers:
            if worker.is_running():
                worker.kill()


if __name__ == "__main__":
    pool = make_pool(3)
    print(json.dumps([process.pid for process in pool._processes]), flush=True)
    signal.pause()
