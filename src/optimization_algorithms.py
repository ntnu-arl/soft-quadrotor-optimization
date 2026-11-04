from abc import ABC, abstractmethod
import pickle

import cma
import numpy as np
import pandas as pd
from botorch.acquisition.analytic import LogExpectedImprovement


CMA_VERBOSE = -1
CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA = "clip_sigma"
CMA_LIMIT_CHANGE_TYPE_CLIP_SOLUTION = "clip_solution"
CMA_LIMIT_CHANGE_TYPES = {
    CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA,
    CMA_LIMIT_CHANGE_TYPE_CLIP_SOLUTION,
}
OBJECTIVE_SENSE_MAXIMIZE = "maximize"
OBJECTIVE_SENSE_MINIMIZE = "minimize"
OBJECTIVE_SENSES = {
    OBJECTIVE_SENSE_MAXIMIZE,
    OBJECTIVE_SENSE_MINIMIZE,
}
ALGORITHM_SOBOL_BO_CMAES = "SOBOL_BO_CMAES"
SOBOL_BO_CMAES_STAGE_SOBOL = "SOBOL"
SOBOL_BO_CMAES_STAGE_BO = "BO"
SOBOL_BO_CMAES_STAGE_CMAES = "CMAES"
SOBOL_BO_CMAES_STAGE_FINISHED = "FINISHED"
CMA_SEARCH_LOWER_BOUND = 0.0
CMA_SEARCH_UPPER_BOUND = 1.0
CMA_GENERATION_ONLY_OPTIONS = {
    "maxiter": float("inf"),
    "maxfevals": float("inf"),
    "ftarget": float("-inf"),
    "tolfun": float("-inf"),
    "tolfunhist": float("-inf"),
    "tolfunrel": 0,
    "tolx": float("-inf"),
    "tolxstagnation": False,
    "tolflatfitness": float("inf"),
    "tolstagnation": 0,
    "tolfacupx": float("inf"),
    "tolupsigma": float("inf"),
    "tolconditioncov": float("inf"),
    "timeout": float("inf"),
}
WIDTH_CMA_SIGMA_MIN = 1e-8
WIDTH_CMA_SIGMA_MAX = 0.2
WIDTH_CMA_TOLX = 1e-4


class ParamConverter:
    """Handles conversion between optimized and full parameter spaces."""

    def __init__(self, param_names, param_bounds, fixed_params, log_params):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.fixed_params = fixed_params
        self.log_params = log_params
        self.optimized_param_names = [name for name in self.param_names if name not in self.fixed_params]

    def actual_to_optimized_single(self, param_name, actual_value):
        idx = self.param_names.index(param_name)
        min_val, max_val = self.param_bounds[idx]
        if param_name in self.log_params:
            return np.log(actual_value / min_val) / np.log(max_val / min_val)
        return (actual_value - min_val) / (max_val - min_val)

    def get_init_bounds_01(self, init_bounds_param_space):
        bounds_01 = []
        for name in self.optimized_param_names:
            if name in init_bounds_param_space:
                init_min, init_max = init_bounds_param_space[name]
                bounds_01.append((self.actual_to_optimized_single(name, init_min), self.actual_to_optimized_single(name, init_max)))
            else:
                bounds_01.append((CMA_SEARCH_LOWER_BOUND, CMA_SEARCH_UPPER_BOUND))
        return bounds_01

    def sample_random_optimized(self, init_bounds_param_space, rng):
        bounds_01 = self.get_init_bounds_01(init_bounds_param_space)
        return np.array([rng.uniform(lo, hi) for lo, hi in bounds_01])

    def optimized_to_full(self, x_opt):
        expected_len = len(self.optimized_param_names)
        actual_len = len(x_opt)
        assert actual_len == expected_len, (
            f"optimized_to_full() expects {expected_len} parameters (optimized space), "
            f"but received {actual_len}. "
            f"Expected params: {self.optimized_param_names}. "
            f"If you have {len(self.param_names)} parameters, they are already in full space and should not be transformed."
        )
        x_full = np.zeros(len(self.param_names))
        for i, param_name in enumerate(self.param_names):
            if param_name in self.fixed_params:
                min_val, max_val = self.param_bounds[i]
                x_full[i] = (self.fixed_params[param_name] - min_val) / (max_val - min_val)
            else:
                opt_idx = self.optimized_param_names.index(param_name)
                if param_name in self.log_params:
                    min_val, max_val = self.param_bounds[i]
                    actual = min_val * (max_val / min_val) ** x_opt[opt_idx]
                    x_full[i] = (actual - min_val) / (max_val - min_val)
                else:
                    x_full[i] = x_opt[opt_idx]
        return x_full

    def full_to_optimized(self, x_full):
        expected_len = len(self.param_names)
        actual_len = len(x_full)
        assert actual_len == expected_len, (
            f"full_to_optimized() expects {expected_len} parameters (full space), "
            f"but received {actual_len}. "
            f"All params: {self.param_names}. "
            f"If you have {len(self.optimized_param_names)} parameters, they are in optimized space and should use optimized_to_full() instead."
        )
        x_opt = []
        for param_name in self.optimized_param_names:
            full_idx = self.param_names.index(param_name)
            min_val, max_val = self.param_bounds[full_idx]
            actual = min_val + x_full[full_idx] * (max_val - min_val)
            x_opt.append(self.actual_to_optimized_single(param_name, actual))
        return x_opt


class OptimizationAlgorithm(ABC):
    def __init__(self, param_names, param_bounds, fixed_params, log_params):
        self.converter = ParamConverter(param_names, param_bounds, fixed_params, log_params)
        self.generation = 0

    @property
    def param_names(self):
        return self.converter.param_names

    @property
    def param_bounds(self):
        return self.converter.param_bounds

    @property
    def fixed_params(self):
        return self.converter.fixed_params

    @property
    def log_params(self):
        return self.converter.log_params

    @property
    def optimized_param_names(self):
        return self.converter.optimized_param_names

    def _optimized_to_full(self, x_opt):
        return self.converter.optimized_to_full(x_opt)

    def _full_to_optimized(self, x_full):
        return self.converter.full_to_optimized(x_full)

    def save_state(self, state_path, next_generation, best_fitness, best_solution):
        state = {
            "optimizer": self,
            "next_generation": int(next_generation),
            "best_fitness": float(best_fitness),
            "best_solution": None if best_solution is None else list(best_solution),
        }
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load_state(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        if "optimizer" not in state:
            raise KeyError(f"Optimizer state file does not contain an optimizer: {state_path}")
        optimizer = state["optimizer"]
        if not isinstance(optimizer, OptimizationAlgorithm):
            raise TypeError(f"Saved optimizer has unexpected type: {type(optimizer)}")
        optimizer.after_state_loaded()
        return state

    def after_state_loaded(self):
        pass

    @abstractmethod
    def ask(self):
        pass

    @abstractmethod
    def tell(self, solutions_full, fitness_values):
        pass

    @abstractmethod
    def load_resume_state(self, results_file, base_dir):
        pass

    @property
    @abstractmethod
    def mean(self):
        pass


class SigmaClippedCMAES(OptimizationAlgorithm):
    def __init__(
        self,
        x0,
        sigma0,
        max_sigma,
        opts,
        param_names,
        param_bounds,
        fixed_params,
        log_params,
        limit_change_type,
        objective_sense,
        sigma_min,
        use_generation_only_stop_options,
    ):
        super().__init__(param_names, param_bounds, fixed_params, log_params)
        self.max_sigma = max_sigma
        self.sigma_min = sigma_min
        self.limit_change_type = limit_change_type
        self.objective_sense = objective_sense
        self.use_generation_only_stop_options = use_generation_only_stop_options
        self._validate_configuration()
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, self._make_options(opts))

    def _validate_configuration(self):
        if self.limit_change_type not in CMA_LIMIT_CHANGE_TYPES:
            raise ValueError(f"Invalid limit_change_type: {self.limit_change_type}. Valid options: {sorted(CMA_LIMIT_CHANGE_TYPES)}")
        if self.objective_sense not in OBJECTIVE_SENSES:
            raise ValueError(f"Invalid objective_sense: {self.objective_sense}. Valid options: {sorted(OBJECTIVE_SENSES)}")
        if self.sigma_min is not None and self.sigma_min > self.max_sigma:
            raise ValueError(f"sigma_min must be <= max_sigma, got {self.sigma_min} > {self.max_sigma}")

    def _make_options(self, opts):
        options = dict(opts)
        if self.use_generation_only_stop_options:
            options.update(CMA_GENERATION_ONLY_OPTIONS)
        return options

    def after_state_loaded(self):
        if self.use_generation_only_stop_options:
            self.apply_generation_only_cma_options()

    def apply_generation_only_cma_options(self):
        for key, value in CMA_GENERATION_ONLY_OPTIONS.items():
            self.es.opts[key] = value
        self.es.stop(check=False).clear()

    def _limit_sigma(self):
        if self.sigma_min is None:
            self.es.sigma = min(self.es.sigma, self.max_sigma)
        else:
            self.es.sigma = np.clip(self.es.sigma, self.sigma_min, self.max_sigma)

    def ask(self):
        if self.limit_change_type == CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA:
            self._limit_sigma()
            solutions_opt = self.es.ask()
            solutions_opt_clipped = [
                np.clip(s, CMA_SEARCH_LOWER_BOUND, CMA_SEARCH_UPPER_BOUND)
                for s in solutions_opt
            ]
        elif self.limit_change_type == CMA_LIMIT_CHANGE_TYPE_CLIP_SOLUTION:
            reference = self.es.mean
            solutions_opt = self.es.ask()
            solutions_opt_clipped = [
                np.clip(s, reference - self.max_sigma, reference + self.max_sigma)
                for s in solutions_opt
            ]
            solutions_opt_clipped = [
                np.clip(s, CMA_SEARCH_LOWER_BOUND, CMA_SEARCH_UPPER_BOUND)
                for s in solutions_opt_clipped
            ]
        else:
            raise ValueError(f"Invalid limit_change_type: {self.limit_change_type}. Valid options: {sorted(CMA_LIMIT_CHANGE_TYPES)}")

        return [self._optimized_to_full(s) for s in solutions_opt_clipped]

    def tell(self, solutions_full, fitness_values):
        solutions_opt = [self._full_to_optimized(x_full) for x_full in solutions_full]
        cma_fitness_values = self._cma_fitness_values(fitness_values)
        self.es.tell(solutions_opt, cma_fitness_values)
        self._limit_sigma()
        self.generation += 1

    def _cma_fitness_values(self, fitness_values):
        if self.objective_sense == OBJECTIVE_SENSE_MAXIMIZE:
            return [-float(f) for f in fitness_values]
        if self.objective_sense == OBJECTIVE_SENSE_MINIMIZE:
            return [float(f) for f in fitness_values]
        raise ValueError(f"Invalid objective_sense: {self.objective_sense}. Valid options: {sorted(OBJECTIVE_SENSES)}")

    @property
    def result(self):
        return self.es.result

    @property
    def mean(self):
        mean_opt = self.es.mean
        return self._optimized_to_full(mean_opt)

    @property
    def sigma(self):
        return self.es.sigma

    @property
    def best_solution(self):
        return self._optimized_to_full(self.es.result.xbest)

    @property
    def best_fitness(self):
        if self.objective_sense == OBJECTIVE_SENSE_MAXIMIZE:
            return -self.es.result.fbest
        if self.objective_sense == OBJECTIVE_SENSE_MINIMIZE:
            return self.es.result.fbest
        raise ValueError(f"Invalid objective_sense: {self.objective_sense}. Valid options: {sorted(OBJECTIVE_SENSES)}")

    @property
    def stop(self):
        return self.es.stop()

    def load_resume_state(self, results_file, base_dir):
        df = pd.read_csv(results_file)

        if len(df) == 0:
            raise ValueError("CSV file is empty, cannot resume")

        last_gen = df['generation'].max()

        if 'restart_idx' in df.columns:
            current_restart_idx = df[df['generation'] == last_gen]['restart_idx'].iloc[0]
        else:
            current_restart_idx = 0

        print(f"\n{'='*80}")
        print("Reconstructing CMA-ES state from generations in CSV...")
        print(f"Current restart phase: {current_restart_idx}")
        print(f"{'='*80}\n")

        sorted_initial_population = None
        gen_minus_1 = df[df['generation'] == -1]
        if len(gen_minus_1) > 0:
            print(f"Processing generation -1 ({len(gen_minus_1)} random solutions)...")

            solutions_full = []
            fitness_values = []
            checkpoint_paths = []
            for _, row in gen_minus_1.iterrows():
                solution = [row[f'x_{i}'] for i in range(len(self.param_names))]
                solutions_full.append(solution)
                fitness_values.append(row['success_rate'])
                checkpoint_path = base_dir / row['workspace_id'].replace("cache/workspace_local/", "") / "last_checkpoint.pth"
                checkpoint_paths.append(checkpoint_path)

            sorted_indices = np.argsort(fitness_values)[::-1]
            sorted_initial_population = [{
                'solution': np.array(solutions_full[i]),
                'fitness': fitness_values[i],
                'checkpoint': checkpoint_paths[i],
                'individual_id': i
            } for i in sorted_indices]

        df_current_restart = df[df['restart_idx'] == current_restart_idx]
        if len(df_current_restart) == 0:
            raise ValueError(f"No data found for current restart phase {current_restart_idx}")

        restart_start_gen = df_current_restart['generation'].min()

        if current_restart_idx == 0:
            restart_baseline_fitness = sorted_initial_population[0]['fitness'] if sorted_initial_population else df_current_restart['success_rate'].max()
        else:
            restart_baseline_fitness = sorted_initial_population[current_restart_idx]['fitness']

        best_fitness_achieved = df_current_restart['success_rate'].max()
        best_fitness_so_far = max(restart_baseline_fitness, best_fitness_achieved)

        best_row = df_current_restart[df_current_restart['success_rate'] == best_fitness_achieved].iloc[0]
        best_solution_full = [best_row[f'x_{i}'] for i in range(len(self.param_names))]
        restart_best_checkpoint_path = base_dir / best_row['workspace_id'].replace("cache/workspace_local/", "") / "last_checkpoint.pth"

        if best_fitness_achieved < restart_baseline_fitness and sorted_initial_population:
            restart_best_checkpoint_path = sorted_initial_population[current_restart_idx]['checkpoint']
            best_solution_full = list(sorted_initial_population[current_restart_idx]['solution'])

        if restart_start_gen == -1:
            restart_solution_full = sorted_initial_population[current_restart_idx]['solution']
        elif restart_start_gen == 0:
            restart_solution_full = df[df['generation'] == 0].iloc[0]
            restart_solution_full = [restart_solution_full[f'x_{i}'] for i in range(len(self.param_names))]
        else:
            restart_row = df_current_restart.iloc[0]
            restart_solution_full = [restart_row[f'x_{i}'] for i in range(len(self.param_names))]

        restart_solution_opt = self._full_to_optimized(restart_solution_full)

        print(f"\n[Resume] Creating fresh CMA-ES from restart phase {current_restart_idx}")
        print(f"  Restart started at generation: {restart_start_gen}")
        print(f"  Restart solution: {[round(v, 3) for v in restart_solution_full]}")

        self.es = cma.CMAEvolutionStrategy(restart_solution_opt, self.es.sigma, self.es.opts)

        if last_gen >= restart_start_gen + 1:
            print(f"\nReplaying generations {restart_start_gen + 1}-{last_gen} to reconstruct covariance...")

            for gen in range(restart_start_gen + 1, last_gen + 1):
                gen_data = df[df['generation'] == gen].sort_values('individual_id')

                solutions_full = []
                fitness_values = []
                for _, row in gen_data.iterrows():
                    solution = [row[f'x_{i}'] for i in range(len(self.param_names))]
                    solutions_full.append(solution)
                    fitness_values.append(row['success_rate'])

                _ = self.ask()
                self.tell(solutions_full, fitness_values)

                print(f"  Gen {gen} replayed: {len(solutions_full)} solutions")

            next_gen = last_gen + 1
        else:
            next_gen = restart_start_gen + 1

        best_solution_opt = self._full_to_optimized(best_solution_full)
        self.es.mean = np.array(best_solution_opt)

        gens_since_last_improvement = 0
        running_best = -np.inf
        restart_gens = sorted(df_current_restart['generation'].unique())
        for gen in restart_gens:
            if gen < 0:
                continue
            gen_data = df_current_restart[df_current_restart['generation'] == gen]
            if len(gen_data) > 0:
                gen_best = gen_data['success_rate'].max()
                if gen_best > running_best:
                    running_best = gen_best
                    gens_since_last_improvement = 0
                else:
                    gens_since_last_improvement += 1

        print(f"\n[Resume] CMA-ES state fully reconstructed for restart phase {current_restart_idx}")
        print(f"[Resume] CMA-ES mean: {[round(v, 3) for v in self.mean]}")
        print(f"[Resume] Restart baseline fitness: {restart_baseline_fitness:.2f}%")
        print(f"[Resume] Best achieved in restart phase: {best_fitness_achieved:.2f}%")
        print(f"[Resume] Best fitness (baseline for comparison): {best_fitness_so_far:.2f}%")
        print(f"[Resume] Resuming from generation {next_gen}")
        print(f"[Resume] Restart-best checkpoint: {restart_best_checkpoint_path}")
        print(f"[Resume] Gens since last improvement (within restart): {gens_since_last_improvement}")

        return next_gen, restart_best_checkpoint_path, best_fitness_so_far, best_fitness_achieved, sorted_initial_population, current_restart_idx, gens_since_last_improvement


class OnePlusLambda(OptimizationAlgorithm):
    def __init__(
        self,
        x0,
        sigma,
        param_names,
        param_bounds,
        lambda_size,
        fixed_params,
        log_params,
        seed,
    ):
        super().__init__(param_names, param_bounds, fixed_params, log_params)
        self.mean_opt = np.array(x0, dtype=float)
        self.sigma = sigma
        self.lambda_size = lambda_size
        self.rng = np.random.RandomState(seed)
        self.best_fitness = -np.inf

    def ask(self):
        solutions_opt = [
            np.clip(
                self.mean_opt + self.rng.normal(0.0, self.sigma, size=len(self.mean_opt)),
                CMA_SEARCH_LOWER_BOUND,
                CMA_SEARCH_UPPER_BOUND,
            )
            for _ in range(self.lambda_size)
        ]
        return [self._optimized_to_full(s) for s in solutions_opt]

    def tell(self, solutions_full, fitness_values):
        best_idx = int(np.argmax(fitness_values))
        best_fitness = float(fitness_values[best_idx])
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.mean_opt = np.array(self._full_to_optimized(solutions_full[best_idx]), dtype=float)
        self.generation += 1

    def load_resume_state(self, results_file, base_dir):
        raise NotImplementedError("OnePlusLambda resume reconstruction is not implemented")

    @property
    def mean(self):
        return self._optimized_to_full(self.mean_opt)


class SobolBOCMAES(OptimizationAlgorithm):
    def __init__(
        self,
        cma_sigma0,
        cma_max_sigma,
        cma_opts,
        param_names,
        param_bounds,
        fixed_params,
        log_params,
        limit_change_type,
        objective_sense,
        cma_sigma_min,
        use_generation_only_stop_options,
        seed,
        bo_max_parallelism,
        n_evals_sobol,
        n_evals_bo,
        n_evals_cmaes,
    ):
        super().__init__(param_names, param_bounds, fixed_params, log_params)
        self.cma_sigma0 = cma_sigma0
        self.cma_max_sigma = cma_max_sigma
        self.cma_opts = dict(cma_opts)
        self.limit_change_type = limit_change_type
        self.objective_sense = objective_sense
        self.cma_sigma_min = cma_sigma_min
        self.use_generation_only_stop_options = use_generation_only_stop_options
        self.seed = seed
        self.bo_max_parallelism = bo_max_parallelism
        self.n_evals_sobol = n_evals_sobol
        self.n_evals_bo = n_evals_bo
        self.n_evals_cmaes = n_evals_cmaes
        self.ax_evals_completed = 0
        self.cma_evals_completed = 0
        self.pending_phase = None
        self.pending_ax_trial_indices = []
        self.pending_cma_solutions = []
        if self.objective_sense == OBJECTIVE_SENSE_MAXIMIZE:
            self.best_fitness_observed = -np.inf
        elif self.objective_sense == OBJECTIVE_SENSE_MINIMIZE:
            self.best_fitness_observed = np.inf
        else:
            raise ValueError(f"Invalid objective_sense: {self.objective_sense}. Valid options: {sorted(OBJECTIVE_SENSES)}")
        self.best_solution_observed = None
        self.cma = None
        self._validate_configuration()
        self.ax_client = self._create_ax_client()

    def _validate_configuration(self):
        if self.objective_sense not in OBJECTIVE_SENSES:
            raise ValueError(f"Invalid objective_sense: {self.objective_sense}. Valid options: {sorted(OBJECTIVE_SENSES)}")
        if self.n_evals_sobol <= 0:
            raise ValueError(f"n_evals_sobol must be > 0, got {self.n_evals_sobol}")
        if self.n_evals_bo <= 0:
            raise ValueError(f"n_evals_bo must be > 0, got {self.n_evals_bo}")
        if self.n_evals_cmaes <= 0:
            raise ValueError(f"n_evals_cmaes must be > 0, got {self.n_evals_cmaes}")
        if self.bo_max_parallelism <= 0:
            raise ValueError(f"bo_max_parallelism must be > 0, got {self.bo_max_parallelism}")
        if "popsize" not in self.cma_opts:
            raise KeyError("cma_opts must contain popsize")
        cma_popsize = int(self.cma_opts["popsize"])
        if cma_popsize <= 0:
            raise ValueError(f"CMA-ES popsize must be > 0, got {cma_popsize}")
        if self.n_evals_cmaes % cma_popsize != 0:
            raise ValueError(
                f"n_evals_cmaes must be divisible by CMA-ES popsize, got "
                f"{self.n_evals_cmaes} and {cma_popsize}"
            )

    def _create_ax_client(self):
        from ax.generation_strategy.generation_node import GenerationStep, Generators
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from ax.service.ax_client import AxClient, ObjectiveProperties

        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=self.n_evals_sobol,
                    max_parallelism=self.n_evals_sobol,
                    enforce_num_trials=True,
                    generator_kwargs={"seed": int(self.seed)},
                ),
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=self.n_evals_bo,
                    max_parallelism=1,
                    generator_kwargs={"botorch_acqf_class": LogExpectedImprovement},
                ),
            ]
        )
        ax_client = AxClient(
            generation_strategy=generation_strategy,
            verbose_logging=False,
        )
        ax_client.create_experiment(
            parameters=[
                {
                    "name": param_name,
                    "type": "range",
                    "value_type": "float",
                    "bounds": [CMA_SEARCH_LOWER_BOUND, CMA_SEARCH_UPPER_BOUND],
                }
                for param_name in self.optimized_param_names
            ],
            objectives={
                "objective_function": ObjectiveProperties(
                    minimize=self.objective_sense == OBJECTIVE_SENSE_MINIMIZE
                ),
            },
        )
        return ax_client

    @property
    def stage(self):
        if self.ax_evals_completed < self.n_evals_sobol:
            return SOBOL_BO_CMAES_STAGE_SOBOL
        if self.ax_evals_completed < self.n_evals_sobol + self.n_evals_bo:
            return SOBOL_BO_CMAES_STAGE_BO
        if self.cma_evals_completed < self.n_evals_cmaes:
            return SOBOL_BO_CMAES_STAGE_CMAES
        return SOBOL_BO_CMAES_STAGE_FINISHED

    @property
    def is_finished(self):
        return self.stage == SOBOL_BO_CMAES_STAGE_FINISHED

    @property
    def sigma(self):
        if self.cma is None:
            return np.nan
        return self.cma.sigma

    @property
    def mean(self):
        if self.cma is not None:
            return self.cma.mean
        if self.best_solution_observed is not None:
            return self.best_solution_observed
        return np.full(len(self.param_names), np.nan)

    def ask(self):
        if self.pending_phase is not None:
            raise RuntimeError(f"Cannot ask while {self.pending_phase} evaluations are pending")
        if self.stage == SOBOL_BO_CMAES_STAGE_SOBOL:
            return self._ask_ax(1)
        if self.stage == SOBOL_BO_CMAES_STAGE_BO:
            n_remaining = self.n_evals_sobol + self.n_evals_bo - self.ax_evals_completed
            return self._ask_ax(min(1, n_remaining))
        if self.stage == SOBOL_BO_CMAES_STAGE_CMAES:
            self._ensure_cma_initialized()
            return self._ask_cma()
        raise RuntimeError("Cannot ask after SOBOL_BO_CMAES has finished")

    def _ask_ax(self, n_solutions):
        self.pending_phase = self.stage
        self.pending_ax_trial_indices = []
        solutions_full = []
        for _ in range(n_solutions):
            parameters, trial_index = self.ax_client.get_next_trial()
            solution_opt = [float(parameters[param_name]) for param_name in self.optimized_param_names]
            solutions_full.append(self._optimized_to_full(solution_opt))
            self.pending_ax_trial_indices.append(int(trial_index))
        return solutions_full

    def _ask_cma(self):
        cma_popsize = int(self.cma_opts["popsize"])
        n_remaining = self.n_evals_cmaes - self.cma_evals_completed
        n_solutions = min(cma_popsize, n_remaining)
        if n_solutions != cma_popsize:
            raise RuntimeError(
                f"CMA-ES remaining evaluations ({n_solutions}) differ from popsize ({cma_popsize})"
            )
        self.pending_phase = SOBOL_BO_CMAES_STAGE_CMAES
        if self.cma.limit_change_type == CMA_LIMIT_CHANGE_TYPE_CLIP_SIGMA:
            self.cma._limit_sigma()
            solutions_opt = self.cma.es.ask(number=n_solutions)
            solutions_opt = [
                np.clip(s, CMA_SEARCH_LOWER_BOUND, CMA_SEARCH_UPPER_BOUND)
                for s in solutions_opt
            ]
        elif self.cma.limit_change_type == CMA_LIMIT_CHANGE_TYPE_CLIP_SOLUTION:
            reference = self.cma.es.mean
            solutions_opt = self.cma.es.ask(number=n_solutions)
            solutions_opt = [
                np.clip(s, reference - self.cma.max_sigma, reference + self.cma.max_sigma)
                for s in solutions_opt
            ]
            solutions_opt = [
                np.clip(s, CMA_SEARCH_LOWER_BOUND, CMA_SEARCH_UPPER_BOUND)
                for s in solutions_opt
            ]
        else:
            raise ValueError(f"Invalid limit_change_type: {self.cma.limit_change_type}. Valid options: {sorted(CMA_LIMIT_CHANGE_TYPES)}")
        self.pending_cma_solutions = [self._optimized_to_full(s) for s in solutions_opt]
        return self.pending_cma_solutions

    def tell(self, solutions_full, fitness_values):
        if self.pending_phase is None:
            raise RuntimeError("Cannot tell before ask")
        if len(solutions_full) != len(fitness_values):
            raise ValueError(
                f"solutions_full and fitness_values have different lengths: "
                f"{len(solutions_full)} != {len(fitness_values)}"
            )
        if self.pending_phase in {SOBOL_BO_CMAES_STAGE_SOBOL, SOBOL_BO_CMAES_STAGE_BO}:
            self._tell_ax(solutions_full, fitness_values)
        elif self.pending_phase == SOBOL_BO_CMAES_STAGE_CMAES:
            self._tell_cma(solutions_full, fitness_values)
        else:
            raise ValueError(f"Unknown pending phase: {self.pending_phase}")
        self._update_best_solution(solutions_full, fitness_values)
        self.pending_phase = None
        self.generation += 1

    def _tell_ax(self, solutions_full, fitness_values):
        if len(solutions_full) != len(self.pending_ax_trial_indices):
            raise ValueError(
                f"Expected {len(self.pending_ax_trial_indices)} Ax solutions, got {len(solutions_full)}"
            )
        for trial_index, solution_full, fitness_value in zip(
            self.pending_ax_trial_indices,
            solutions_full,
            fitness_values,
        ):
            self._set_ax_trial_parameters(trial_index, solution_full)
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"objective_function": float(fitness_value)},
            )
        self.ax_evals_completed += len(fitness_values)
        self.pending_ax_trial_indices = []

    def _set_ax_trial_parameters(self, trial_index, solution_full):
        solution_opt = self._full_to_optimized(solution_full)
        parameters = {
            param_name: float(value)
            for param_name, value in zip(self.optimized_param_names, solution_opt)
        }
        trial = self.ax_client.experiment.trials[trial_index]
        trial.arm._parameters = parameters

    def _tell_cma(self, solutions_full, fitness_values):
        if len(solutions_full) != len(self.pending_cma_solutions):
            raise ValueError(
                f"Expected {len(self.pending_cma_solutions)} CMA-ES solutions, got {len(solutions_full)}"
            )
        self.cma.tell(solutions_full, fitness_values)
        self.cma_evals_completed += len(fitness_values)
        self.pending_cma_solutions = []

    def _update_best_solution(self, solutions_full, fitness_values):
        if self.objective_sense == OBJECTIVE_SENSE_MAXIMIZE:
            best_idx = int(np.argmax(fitness_values))
            is_better = float(fitness_values[best_idx]) > self.best_fitness_observed
        elif self.objective_sense == OBJECTIVE_SENSE_MINIMIZE:
            best_idx = int(np.argmin(fitness_values))
            is_better = float(fitness_values[best_idx]) < self.best_fitness_observed
        else:
            raise ValueError(f"Invalid objective_sense: {self.objective_sense}. Valid options: {sorted(OBJECTIVE_SENSES)}")
        if is_better:
            self.best_fitness_observed = float(fitness_values[best_idx])
            self.best_solution_observed = np.asarray(solutions_full[best_idx], dtype=float)

    def _ensure_cma_initialized(self):
        if self.cma is not None:
            return
        if self.best_solution_observed is None:
            raise RuntimeError("Cannot initialize CMA-ES before Sobol/BO has observed a solution")
        x0 = self._full_to_optimized(self.best_solution_observed)
        self.cma = SigmaClippedCMAES(
            x0=x0,
            sigma0=self.cma_sigma0,
            max_sigma=self.cma_max_sigma,
            opts=self.cma_opts,
            param_names=self.param_names,
            param_bounds=self.param_bounds,
            fixed_params=self.fixed_params,
            log_params=self.log_params,
            limit_change_type=self.limit_change_type,
            objective_sense=self.objective_sense,
            sigma_min=self.cma_sigma_min,
            use_generation_only_stop_options=self.use_generation_only_stop_options,
        )

    def after_state_loaded(self):
        if self.cma is not None:
            self.cma.after_state_loaded()

    def load_resume_state(self, results_file, base_dir):
        raise NotImplementedError("SobolBOCMAES resume reconstruction from CSV is not implemented")
