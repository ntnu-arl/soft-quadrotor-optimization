import os
import signal
import subprocess
import time
import json
import shutil
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import sys
import uuid


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Job:
    job_id: str
    x: Union[List[float], Dict[str, float]]
    workspace_id: str
    parent_checkpoint_path: Optional[Path]
    gen: int
    ind_id: int
    seed: int
    num_envs_train: int
    max_epochs: int
    num_envs_eval: int
    cmd_args: Optional[Dict[str, Any]] = None
    status: JobStatus = JobStatus.PENDING
    worker_id: Optional[str] = None
    start_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    until_no_improvement: bool = False
    training_method: Optional[str] = None
    n_repeat_train: Optional[List[List[int]]] = None


def create_workspace(job: Job, workspace_dir: Path) -> None:
    """Create workspace folder with job config, checkpoint, run.sh, and status.txt."""
    workspace_dir.mkdir(parents=True, exist_ok=True)

    job_config = {
        'job_id': job.job_id,
        'x': job.x.tolist() if hasattr(job.x, 'tolist') else job.x,
        'workspace_id': job.workspace_id,
        'gen': int(job.gen),
        'ind_id': int(job.ind_id),
        'seed': int(job.seed),
        'num_envs_train': int(job.num_envs_train),
        'max_epochs': int(job.max_epochs),
        'num_envs_eval': int(job.num_envs_eval),
        'has_checkpoint': job.parent_checkpoint_path is not None,
        'cmd_args': job.cmd_args if job.cmd_args is not None else {},
        'until_no_improvement': job.until_no_improvement,
        'training_method': job.training_method,
        'n_repeat_train': job.n_repeat_train
    }

    with open(workspace_dir / 'job_config.json', 'w') as f:
        json.dump(job_config, f, indent=2)

    if job.parent_checkpoint_path and job.parent_checkpoint_path.exists():
        shutil.copy(job.parent_checkpoint_path, workspace_dir / 'checkpoint.pth')

    with open(workspace_dir / 'status.txt', 'w') as f:
        f.write('pending')


def read_workspace_status(workspace_dir: Path) -> str:
    """Read status from workspace status.txt."""
    status_file = workspace_dir / 'status.txt'
    if not status_file.exists():
        return 'pending'
    return status_file.read_text().strip()


def read_workspace_results(workspace_dir: Path) -> Dict[str, Any]:
    """Read results from workspace."""
    results_json = workspace_dir / 'results.json'
    with open(results_json, 'r') as f:
        results = json.load(f)

    results['workspace_dir'] = workspace_dir
    results['last_checkpoint_path'] = workspace_dir / 'last_checkpoint.pth'

    return results


class Worker:
    def __init__(self, worker_id: str, worker_type: str):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.is_free = True
        self.current_job: Optional[Job] = None
        self.process: Optional[subprocess.Popen] = None
        self.disabled = False

    def submit_job(self, job: Job) -> None:
        raise NotImplementedError

    def check_status(self) -> JobStatus:
        raise NotImplementedError

    def retrieve_result(self) -> Dict[str, Any]:
        raise NotImplementedError

    def cleanup(self) -> None:
        self.is_free = True
        self.current_job = None
        self.process = None

    def kill_process_tree(self) -> None:
        """Kill the process and all its children."""
        if self.process is None:
            return
        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait()
        except (ProcessLookupError, OSError):
            pass


class LocalWorker(Worker):
    def __init__(self, worker_id: str, gpu_id: int):
        super().__init__(worker_id, "local")
        self.base_dir = Path(__file__).parent.parent / "cache/workspace_local"
        self.gpu_id = gpu_id

    def submit_job(self, job: Job) -> None:
        self.is_free = False
        self.current_job = job
        job.status = JobStatus.RUNNING
        job.worker_id = self.worker_id
        job.start_time = time.time()

        workspace_dir = self.base_dir / job.workspace_id
        create_workspace(job, workspace_dir)

        if job.training_method != "nmpc":
            raise ValueError(f"Unsupported training method: {job.training_method}")

        run_script = workspace_dir / "run.sh"
        script_path = Path(__file__).parent / "run_nmpc_objective.py"
        repo_dir = Path(__file__).parent.parent

        run_script.write_text(f"""#!/bin/bash

WORKSPACE={shlex.quote(str(workspace_dir))}
cd "$WORKSPACE"

echo "[$(date)] Starting local evaluation" > "$WORKSPACE/output.log"
echo "running" > "$WORKSPACE/status.txt"

export CUDA_VISIBLE_DEVICES={self.gpu_id}
cd {shlex.quote(str(repo_dir))}
{shlex.quote(sys.executable)} -u {shlex.quote(str(script_path))} --workspace_dir "$WORKSPACE" --device cuda:0 >> "$WORKSPACE/output.log" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date)] Evaluation completed successfully" >> "$WORKSPACE/output.log"
    echo "completed" > "$WORKSPACE/status.txt"
else
    echo "[$(date)] Script failed" >> "$WORKSPACE/output.log"
    echo "failed" > "$WORKSPACE/status.txt"
    exit 1
fi

exit 0
""")
        run_script.chmod(0o755)

        print(f"[LocalWorker {self.worker_id}] Starting job {job.job_id[:8]} on GPU {self.gpu_id} (Gen {job.gen}, Ind {job.ind_id})")
        self.process = subprocess.Popen(
            ["bash", str(run_script)],
            cwd=str(workspace_dir),
            start_new_session=True
        )

    def check_status(self) -> JobStatus:
        if self.process is None:
            return JobStatus.PENDING

        returncode = self.process.poll()
        if returncode is None:
            return JobStatus.RUNNING

        workspace_dir = self.base_dir / self.current_job.workspace_id
        status_text = read_workspace_status(workspace_dir)

        if status_text == 'completed':
            return JobStatus.COMPLETED
        elif status_text == 'failed':
            return JobStatus.FAILED
        elif returncode != 0:
            return JobStatus.FAILED
        else:
            return JobStatus.COMPLETED

    def retrieve_result(self) -> Dict[str, Any]:
        workspace_dir = self.base_dir / self.current_job.workspace_id
        return read_workspace_results(workspace_dir)


class RemoteEvaluator:
    def __init__(self, worker_type_list: List[str], job_timeout: float):
        self.workers: List[Worker] = []
        self.jobs: Dict[str, Job] = {}
        self.job_timeout = job_timeout
        self.job_queue: List[Dict[str, Any]] = []
        self.completed_results: Dict[tuple, Dict[str, Any]] = {}
        self.job_to_metadata: Dict[str, tuple] = {}
        self.active_job_ids: List[str] = []

        for worker_type in worker_type_list:
            if worker_type != "local":
                raise ValueError(f"Unknown worker type: {worker_type}")
            worker_id = "local_0"
            self.workers.append(LocalWorker(worker_id, 0))
            print(f"[RemoteEvaluator] Registered {worker_id}")

        print(f"[RemoteEvaluator] Total workers: {len(self.workers)}")

    def submit_evaluation(
        self,
        x: Union[List[float], Dict[str, float]],
        workspace_id: str,
        parent_checkpoint_path: Optional[Union[Path, str]],
        gen: int,
        ind_id: int,
        seed: int,
        num_envs_train: int,
        max_epochs: int,
        num_envs_eval: int,
        cmd_args: Optional[Dict[str, Any]] = None,
        until_no_improvement: bool = False,
        training_method: Optional[str] = None,
        n_repeat_train: Optional[List[List[int]]] = None
    ) -> str:
        free_worker = next((w for w in self.workers if w.is_free and not w.disabled), None)

        if free_worker is None:
            raise RuntimeError("No free workers available. Call this only when a worker is free.")

        if isinstance(parent_checkpoint_path, str):
            parent_checkpoint_path = Path(parent_checkpoint_path)

        job = Job(
            job_id=str(uuid.uuid4()),
            x=x,
            workspace_id=workspace_id,
            parent_checkpoint_path=parent_checkpoint_path,
            gen=gen,
            ind_id=ind_id,
            seed=seed,
            num_envs_train=num_envs_train,
            max_epochs=max_epochs,
            num_envs_eval=num_envs_eval,
            cmd_args=cmd_args,
            until_no_improvement=until_no_improvement,
            training_method=training_method,
            n_repeat_train=n_repeat_train
        )

        self.jobs[job.job_id] = job
        free_worker.submit_job(job)

        return job.job_id

    def check_job_status(self, job_id: str) -> JobStatus:
        job = self.jobs[job_id]
        worker = next(w for w in self.workers if w.worker_id == job.worker_id)

        elapsed = time.time() - job.start_time
        if elapsed > self.job_timeout:
            print(f"[RemoteEvaluator] Job {job_id[:8]} timed out after {elapsed:.1f}s")
            if worker.process and worker.process.poll() is None:
                worker.kill_process_tree()
            job.status = JobStatus.TIMEOUT
            job.error = f"Timeout after {elapsed:.1f}s"
            worker.cleanup()
            return JobStatus.TIMEOUT

        current_status = worker.check_status()
        job.status = current_status

        if current_status == JobStatus.FAILED:
            job.error = "Job failed (check workspace output.log for details)"

        return job.status

    def get_pending_and_running_jobs(self) -> List[str]:
        return [jid for jid, job in self.jobs.items()
                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]]

    def has_free_worker(self) -> bool:
        return any(w.is_free and not w.disabled for w in self.workers)

    def retrieve_evaluation_results(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs[job_id]

        if job.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Job {job_id[:8]} is not completed (status: {job.status})")

        worker = next(w for w in self.workers if w.worker_id == job.worker_id)

        result = worker.retrieve_result()
        job.result = result

        worker.cleanup()

        return result

    def submit_job(
        self,
        x: Union[List[float], Dict[str, float]],
        workspace_id: str,
        parent_checkpoint_path: Optional[Union[Path, str]],
        gen: int,
        ind_id: int,
        seed: int,
        num_envs_train: int,
        max_epochs: int,
        num_envs_eval: int,
        cmd_args: Optional[Dict[str, Any]] = None,
        until_no_improvement: bool = False,
        training_method: Optional[str] = None,
        n_repeat_train: Optional[List[List[int]]] = None
    ) -> None:
        job_params = {
            'x': x,
            'workspace_id': workspace_id,
            'parent_checkpoint_path': parent_checkpoint_path,
            'gen': gen,
            'ind_id': ind_id,
            'seed': seed,
            'num_envs_train': num_envs_train,
            'max_epochs': max_epochs,
            'num_envs_eval': num_envs_eval,
            'cmd_args': cmd_args,
            'until_no_improvement': until_no_improvement,
            'training_method': training_method,
            'n_repeat_train': n_repeat_train
        }
        self.job_queue.append(job_params)
        self._try_launch_queued_jobs()

    def _try_launch_queued_jobs(self) -> None:
        while self.job_queue and self.has_free_worker():
            job_params = self.job_queue.pop(0)
            job_id = self.submit_evaluation(**job_params)
            self.active_job_ids.append(job_id)
            self.job_to_metadata[job_id] = (
                job_params['gen'],
                job_params['ind_id'],
                job_params['x'],
                job_params['workspace_id']
            )

    def check_jobs_still_running(self) -> bool:
        completed_this_check = []

        for job_id in list(self.active_job_ids):
            status = self.check_job_status(job_id)
            gen, ind_id, _, _ = self.job_to_metadata[job_id]
            job = self.jobs[job_id]

            if status == JobStatus.COMPLETED:
                result = self.retrieve_evaluation_results(job_id)
                success_rate = result['success_percentage']
                average_reward = result.get('average_reward', 0.0)
                eval_time = time.time() - job.start_time
                worker_id = job.worker_id

                self.completed_results[(gen, ind_id)] = {
                    'success_rate': success_rate,
                    'average_reward': average_reward,
                    'checkpoint_path': result['last_checkpoint_path'],
                    'eval_time': eval_time,
                    'metrics': result,
                    'workspace_id': job.workspace_id,
                    'worker_id': worker_id,
                }

                print(f"[Gen {gen}, Ind {ind_id}] Completed on {worker_id}: {success_rate:.2f}% in {eval_time:.1f}s")
                completed_this_check.append(job_id)

            elif status in [JobStatus.FAILED, JobStatus.TIMEOUT]:
                eval_time = time.time() - job.start_time
                worker_id = job.worker_id

                error_msg = job.error if job.error else "Unknown error"
                print(f"[Gen {gen}, Ind {ind_id}] {status.value} on {worker_id}: {error_msg}")
                raise RuntimeError(
                    f"Gen {gen}, Ind {ind_id} {status.value} on {worker_id} after {eval_time:.1f}s: {error_msg}"
                )

        for job_id in completed_this_check:
            self.active_job_ids.remove(job_id)

        self._try_launch_queued_jobs()

        if self.active_job_ids:
            print(f"\n[Status] Running jobs:")
            for job_id in self.active_job_ids:
                gen, ind_id, _, _ = self.job_to_metadata[job_id]
                job = self.jobs[job_id]
                elapsed = time.time() - job.start_time
                print(f"  Gen {gen}, Ind {ind_id} on {job.worker_id}: {elapsed:.1f}s")

        queue_size = len(self.job_queue)
        running_count = len(self.active_job_ids)
        if queue_size > 0 or running_count > 0:
            print(f"[Status] Queue: {queue_size} pending, {running_count} running\n")
            return True
        return False

    def retrieve_job_results(self, gen: int, num_individuals: int) -> tuple:
        fitness_values = [None] * num_individuals
        checkpoint_paths = [None] * num_individuals
        eval_times = [None] * num_individuals
        average_rewards = [None] * num_individuals

        for ind_id in range(num_individuals):
            key = (gen, ind_id)
            if key not in self.completed_results:
                raise RuntimeError(f"Missing results for Gen {gen}, Ind {ind_id}")

            result = self.completed_results[key]
            fitness_values[ind_id] = result['success_rate']
            checkpoint_paths[ind_id] = result['checkpoint_path']
            eval_times[ind_id] = result['eval_time']
            average_rewards[ind_id] = result['average_reward']

        for ind_id in range(num_individuals):
            del self.completed_results[(gen, ind_id)]

        return fitness_values, checkpoint_paths, eval_times, average_rewards
