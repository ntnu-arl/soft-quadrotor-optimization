import torch
import numpy as np


class RotorDynamics():
    n_different_motors = 7
    def __init__(self, n_envs, motor_idx_list, dt, device, motor_model="asymmetric_first_order", motor_model_pars=None):
        
        assert isinstance(motor_idx_list[0], int), "error, motor_indx should be a list of integers."
        self.dt = dt
        self.n_motors = len(motor_idx_list)
        self.n_envs = n_envs
        self.motor_idx_list = motor_idx_list
        self.device = device
        self.motor_model = motor_model

        self.component_data = manufacturerComponentData(device,  motor_idx_list)
        if motor_model_pars is not None:
            assert isinstance(motor_model_pars, dict), "error, motor_model_params should be a dictionary."
            for key, value in motor_model_pars.items():
                assert hasattr(self.component_data, key), f"error, {key} is not a valid attribute of component_data."
                setattr(self.component_data, key, torch.tensor(value, device=device))

        self.first_order_linear_mixing_factors = 1.0 - torch.exp(-self.dt / self.component_data.motor_time_constant)
        self.first_order_linear_mixing_factors_up = 1.0 - torch.exp(-self.dt / self.component_data.motor_time_constant_up)
        self.first_order_linear_mixing_factors_down = 1.0 - torch.exp(-self.dt / self.component_data.motor_time_constant_down)

        self.current_rps = torch.zeros((n_envs, self.n_motors), device=device)
        self.rps_dot = torch.zeros((n_envs, self.n_motors), device=device)  # RPS derivative for second-order dynamics
        
        self.wn_up = self.component_data.natural_freq_up.unsqueeze(0).expand(self.n_envs, self.n_motors)
        self.wn_down = self.component_data.natural_freq_down.unsqueeze(0).expand(self.n_envs, self.n_motors)
        self.zeta_up = self.component_data.damping_ratio_up.unsqueeze(0).expand(self.n_envs, self.n_motors)
        self.zeta_down = self.component_data.damping_ratio_down.unsqueeze(0).expand(self.n_envs, self.n_motors)
        self.min_rps = self.component_data.min_rps.unsqueeze(0)
        self.max_rps = self.component_data.max_rps.unsqueeze(0)
        self.max_rps_expanded = self.component_data.max_rps.unsqueeze(0).expand(self.n_envs, self.n_motors)
        self.min_rps_expanded = self.component_data.min_rps.unsqueeze(0).expand(self.n_envs, self.n_motors)

        # Rate limiter parameters
        self.max_rps_rate_up = self.component_data.max_rps_rate_up.unsqueeze(0).expand(self.n_envs, self.n_motors)
        self.max_rps_rate_down = self.component_data.max_rps_rate_down.unsqueeze(0).expand(self.n_envs, self.n_motors)

        self.reset_idx(list(range(n_envs)))

    def _get_current_forces(self):
        res_force =  self.component_data.thrust_coefficient * self.current_rps**2
        res_torque = res_force * self.component_data.cq
        return res_force, res_torque

    def _get_desired_rps_from_force(self, action_0_1):
        assert torch.all(action_0_1 < 1.0001)
        assert torch.all(action_0_1 > -0.0001)
        desired_force = self.component_data.min_force + (self.component_data.max_force - self.component_data.min_force) * action_0_1
        desired_rps = (desired_force / self.component_data.thrust_coefficient) ** 0.5
        desired_rps = torch.clamp(desired_rps, min=self.component_data.min_rps, max=self.component_data.max_rps)
        return desired_rps

    def update_current_rps(self, desired_rps):
        """
        Asymmetric second-order motor dynamics with different up/down characteristics
        """
        dt = self.dt
        if self.motor_model == "asymmetric_second_order":        
            # Calculate error to determine if we're speeding up or slowing down
            rps_error = desired_rps - self.current_rps
            speeding_up = rps_error > 0
            
            # Select parameters based on direction
            wn = torch.where(speeding_up, self.wn_up, self.wn_down)
            zeta = torch.where(speeding_up, self.zeta_up, self.zeta_down)
            
            # Second-order system: ω̈ = ωn²*(ωcmd - ω) - 2*ζ*ωn*ω̇
            rps_ddot = wn**2 * rps_error - 2*zeta*wn*self.rps_dot
            
            # Integrate using Euler method
            self.rps_dot = self.rps_dot + rps_ddot * dt
            self.current_rps = self.current_rps + self.rps_dot * dt
            
            # Clamp to physical limits
            self.current_rps = torch.clamp(
                self.current_rps, 
                min=self.min_rps, 
                max=self.max_rps
            )
        
            # Zero out positive velocity when at max RPS
            self.rps_dot = torch.where(
                (self.current_rps >= self.max_rps_expanded) & (self.rps_dot > 0),
                torch.zeros_like(self.rps_dot),
                self.rps_dot
            )
            
            # Zero out negative velocity when at min RPS
            self.rps_dot = torch.where(
                (self.current_rps <= self.min_rps_expanded) & (self.rps_dot < 0),
                torch.zeros_like(self.rps_dot),
                self.rps_dot
            )

        elif self.motor_model == "asymmetric_second_order_with_rate_limiter":
            # Calculate error to determine if we're speeding up or slowing down
            rps_error = desired_rps - self.current_rps
            speeding_up = rps_error > 0

            # Select parameters based on direction
            wn = torch.where(speeding_up, self.wn_up, self.wn_down)
            zeta = torch.where(speeding_up, self.zeta_up, self.zeta_down)
            max_rps_rate = torch.where(speeding_up, self.max_rps_rate_up, self.max_rps_rate_down)

            # Second-order system: ω̈ = ωn²*(ωcmd - ω) - 2*ζ*ωn*ω̇
            rps_ddot = wn**2 * rps_error - 2*zeta*wn*self.rps_dot

            # Integrate using Euler method
            self.rps_dot = self.rps_dot + rps_ddot * dt

            # Apply rate limiting to velocity
            self.rps_dot = torch.clamp(self.rps_dot, -max_rps_rate, max_rps_rate)

            self.current_rps = self.current_rps + self.rps_dot * dt

            # Clamp to physical limits
            self.current_rps = torch.clamp(
                self.current_rps,
                min=self.min_rps,
                max=self.max_rps
            )

            # Zero out positive velocity when at max RPS
            self.rps_dot = torch.where(
                (self.current_rps >= self.max_rps_expanded) & (self.rps_dot > 0),
                torch.zeros_like(self.rps_dot),
                self.rps_dot
            )

            # Zero out negative velocity when at min RPS
            self.rps_dot = torch.where(
                (self.current_rps <= self.min_rps_expanded) & (self.rps_dot < 0),
                torch.zeros_like(self.rps_dot),
                self.rps_dot
            )

        elif self.motor_model == "first_order":    
            delta_rps = (
                self.first_order_linear_mixing_factors * desired_rps + (1.0 - self.first_order_linear_mixing_factors) * self.current_rps
                ) - self.current_rps
            self.current_rps +=  delta_rps
        
        elif self.motor_model == "asymmetric_first_order":
            # Calculate error to determine if we're speeding up or slowing down
            rps_error = desired_rps - self.current_rps
            speeding_up = rps_error > 0

            # Select parameters based on direction
            mixing_factors = torch.where(
                speeding_up,
                self.first_order_linear_mixing_factors_up.unsqueeze(0).expand(self.n_envs, self.n_motors),
                self.first_order_linear_mixing_factors_down.unsqueeze(0).expand(self.n_envs, self.n_motors)
            )
            # First-order system: ω = (1 - α) * ω + α * ωcmd
            delta_rps = mixing_factors * desired_rps + (1.0 - mixing_factors) * self.current_rps - self.current_rps
            self.current_rps += delta_rps

        elif self.motor_model == "instantaneous_forces":
            self.current_rps = desired_rps
            self.rps_dot.zero_()

        else:
            raise ValueError(f"Unsupported motor model: {self.motor_model}. Supported models are 'asymmetric_second_order', 'first_order', 'asymmetric_first_order', 'asymmetric_second_order_with_rate_limiter', and 'instantaneous_forces'.")

    def update_state_and_get_current_forces(self, action):
        desired_rps = self._get_desired_rps_from_force(action)
        # if desired_rps.shape[0] == 1:
        #     with open(desired_rps_file_path, "a") as desired_rps_file:
        #         desired_rps_file.write("[")
        #         for i in range(desired_rps[0].size(0)):
        #             desired_rps_file.write(f"{desired_rps[0][i].item()}")
        #             if i != desired_rps[0].size(0) - 1:
        #                 desired_rps_file.write(", ")
        #         desired_rps_file.write("]\n")

        # print("desired_rps=", desired_rps)
        self.update_current_rps(desired_rps)
        return self._get_current_forces()

    def set_desired_rps_and_get_current_rps(self, desired_rps, motor_constant):
        self.first_order_linear_mixing_factors = self.dt / (self.dt + motor_constant)
        self.update_current_rps(desired_rps)
        return self.current_rps
    
    def reset_idx(self, env_ids, initial_action=None):
        if initial_action is None:
            initial_action = 0.25 + 0.15*torch.rand_like(self.current_rps[env_ids])
        elif isinstance(initial_action, float):
            initial_action = torch.full_like(self.current_rps[env_ids], initial_action)
        self.current_rps[env_ids] = self._get_desired_rps_from_force(initial_action)
        self.rps_dot[env_ids] = 0.0

    @staticmethod
    def get_motor_mass_and_prop_diameter(motor_idx_list):
        tmp = manufacturerComponentData("cpu", motor_idx_list)
        return tmp.motor_mass, tmp.prop_diameter

    @staticmethod
    def get_motor_max_force_torque_and_rps(motor_idx_list):
        tmp = manufacturerComponentData("cpu", motor_idx_list)
        return tmp.max_force.numpy(), (tmp.max_force * tmp.cq).numpy(), tmp.max_rps.numpy()

    def force_to_action(self, forces_N: torch.Tensor) -> torch.Tensor:
        """Convert forces in Newtons to normalized actions [0-1].

        Args:
            forces_N: (n_envs, n_motors) forces in Newtons

        Returns:
            actions: (n_envs, n_motors) normalized actions in [0-1]
        """
        actions = (forces_N - self.component_data.min_force) / (self.component_data.max_force - self.component_data.min_force)
        return torch.clamp(actions, 0.0, 1.0)

    def action_to_force(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions [0-1] to forces in Newtons.

        Args:
            actions: (n_envs, n_motors) normalized actions in [0-1]

        Returns:
            forces_N: (n_envs, n_motors) forces in Newtons
        """
        return self.component_data.min_force + actions * (self.component_data.max_force - self.component_data.min_force)


class manufacturerComponentData:

    def __init__(self, device, motor_idx_list):
        self.motor_dict = dict()
        self.n_motors = len(motor_idx_list)
        motor_data = {}
        motor_data[0] = {
            "max_rps": 330.431,
            "thrust_coefficient": 3.1984171295043156e-05,
            "motor_time_constant": 0.023,
            "motor_time_constant_up": 0.023,
            "motor_time_constant_down": 0.023,

            "cq": 0.135,
            "mass": 0.0374 + 0.00397,
            "prop_diameter": 0.127,
            "name": "Vertiq 23-06 2200KV 18V - APC5x4R",
            "S": [3, 4, 5, 6],
            # values below should be tunned
            "natural_freq_up": 60.0,
            "natural_freq_down": 40.0,
            "damping_ratio_up": 0.6,
            "damping_ratio_down": 0.8,
            "max_rps_rate_up": 1000.0,  # Maximum RPS change rate when speeding up
            "max_rps_rate_down": 800.0,  # Maximum RPS change rate when slowing down
        }
        motor_data[1] = {
            "max_rps": 385.7495,
            "thrust_coefficient": 3.0471406317119792e-05,
            "motor_time_constant": 0.029,
            "motor_time_constant_up": 0.029,
            "motor_time_constant_down": 0.029,

            "cq": 0.089,
            "mass": 0.0374 + 0.00311,
            "prop_diameter": 0.127,
            "name": "Vertiq 23-06 2200KV 18V - APC5x3E",
            "S": [3, 4, 5, 6],
            # values below should be tunned
            "natural_freq_up": 60.0,
            "natural_freq_down": 40.0,
            "damping_ratio_up": 0.6,
            "damping_ratio_down": 0.8,
            "max_rps_rate_up": 1000.0,  # Maximum RPS change rate when speeding up
            "max_rps_rate_down": 800.0,  # Maximum RPS change rate when slowing down
        }
        motor_data[2] = {
            "max_rps": 305.4738,
            "thrust_coefficient": 4.4871395271407915e-05,
            "motor_time_constant": 0.031,
            "motor_time_constant_up": 0.031,
            "motor_time_constant_down": 0.031,

            "cq": 0.12,
            "mass": 0.0374 + 0.003118,
            "prop_diameter": 0.1524,
            "name": "Vertiq 23-06 2200KV 18V - APC6x3R",
            "S": [3, 4, 5, 6],
            # values below should be tunned
            "natural_freq_up": 60.0,
            "natural_freq_down": 40.0,
            "damping_ratio_up": 0.6,
            "damping_ratio_down": 0.8,
            "max_rps_rate_up": 1000.0,  # Maximum RPS change rate when speeding up
            "max_rps_rate_down": 800.0,  # Maximum RPS change rate when slowing down
        }
        motor_data[3] = {
            "max_rps": 335.1818,
            "thrust_coefficient": 4.652225028058713e-05,
            "motor_time_constant": 0.033,
            "motor_time_constant_up": 0.033,
            "motor_time_constant_down": 0.033,

            "cq": 0.09,
            "mass": 0.0374 + 0.002,
            "prop_diameter": 0.127,
            "name": "Vertiq 23-06 2200KV 18V - DAL5045BNV2",
            "S": [3, 4, 5, 6],
            # values below should be tunned
            "natural_freq_up": 60.0,
            "natural_freq_down": 40.0,
            "damping_ratio_up": 0.6,
            "damping_ratio_down": 0.8,
            "max_rps_rate_up": 1000.0,  # Maximum RPS change rate when speeding up
            "max_rps_rate_down": 800.0,  # Maximum RPS change rate when slowing down
        }
        motor_data[4] = {
            "max_rps": 256.58,
            "thrust_coefficient": 6.959851185156277e-05,
            "motor_time_constant": 0.030,
            "cq": 0.1275,
            "mass": 0.0374 + 0.00510,
            "prop_diameter": 0.1524,
            "name": "Vertiq 23-06 2200KV 18V - APC6x4E",
            "S": [3, 4, 5, 6],
            # values below should be tunned
            "natural_freq_up": 60.0,
            "natural_freq_down": 40.0,
            "damping_ratio_up": 0.6,
            "damping_ratio_down": 0.8,
            "max_rps_rate_up": 1000.0,  # Maximum RPS change rate when speeding up
            "max_rps_rate_down": 800.0,  # Maximum RPS change rate when slowing down
        }
        motor_data[5] = {
            "max_rps": 400,
            "min_rps": 83,
            "thrust_coefficient": 0.00001286412,
            "motor_time_constant": 0.047,
            "cq": 0.01,
            "mass": 0.012,
            "prop_diameter": 0.0762,
            "name": "Xing2 3 inch",

            "motor_time_constant_up": 0.047,  # -> real measurements show this parameter should be 0.1                                                                                                                        
            "motor_time_constant_down": 0.047, # -> real measurements show this parameter should be 0.064

            # # Tuned to minimize L2 error from experimental data from sim2real/2025_05_7_v4_seed=9/poses_real.txt
            # "natural_freq_up": 42.2182735,      # rad/s - spin-up natural frequency (faster)
            # "natural_freq_down": 27.10310549,    # rad/s - spin-down natural frequency (slower)
            # "damping_ratio_up": 0.61670411,      # dimensionless - spin-up damping 
            # "damping_ratio_down": 0.80405314,    # dimensionless - spin-down damping



            # tuned to minimize L1 error from experimental data from sim2real/2025_05_7_v4_seed=9/poses_real.txt
            "natural_freq_up": 82.25085849829121,
            "natural_freq_down": 33.840935587535476,
            "damping_ratio_up": 0.729610638334013,
            "damping_ratio_down": 0.807618319769484,
            "max_rps_rate_up": 1849.8078473451521, 
            "max_rps_rate_down": 2377.9936624514717,
            "S": [3, 4, 5, 6],
        }
        motor_data[6] = {
            "max_rps": 627,
            "min_rps": 0, # 108,
            "thrust_coefficient": 0.0000207,
            "motor_time_constant": 0.047,
            "cq": 0.0086,
            "mass": 0.016 + 0.0025,  # The motors weigh 16 g with the nut on, and the props 2.5 g.,
            "prop_diameter": 0.09144,
            "name": "T-Motor V1507 + Gemfan GF-3630 3.6 inch",
            "motor_time_constant_up": 0.047,
            "motor_time_constant_down": 0.047,
            "S": [4],
        }

        assert len(motor_data) == RotorDynamics.n_different_motors or len(motor_data) > max(motor_idx_list), "Incomplete motor_data."

        def tensor_fill(key, default=0.0):
            return torch.full((self.n_motors,), default, device=device, dtype=torch.float32)

        self.max_rps = tensor_fill("max_rps")
        self.min_rps = tensor_fill("min_rps")
        self.thrust_coefficient = tensor_fill("thrust_coefficient")
        self.cq = tensor_fill("cq")
        self.motor_mass = [-1e8] * self.n_motors
        self.prop_diameter = [-1e8] * self.n_motors
        self.motor_time_constant = tensor_fill("motor_time_constant")
        self.motor_time_constant_up = tensor_fill("motor_time_constant_up")
        self.motor_time_constant_down = tensor_fill("motor_time_constant_down")

        # Asymmetric parameters
        self.natural_freq_up = tensor_fill("natural_freq_up")
        self.natural_freq_down = tensor_fill("natural_freq_down")
        self.damping_ratio_up = tensor_fill("damping_ratio_up")
        self.damping_ratio_down = tensor_fill("damping_ratio_down")

        # Rate limiter parameters
        self.max_rps_rate_up = tensor_fill("max_rps_rate_up", 1000.0)
        self.max_rps_rate_down = tensor_fill("max_rps_rate_down", 800.0)

        for i, idx in enumerate(motor_idx_list):
            data = motor_data[idx]
            self.max_rps[i] = data["max_rps"]
            self.min_rps[i] = data.get("min_rps", 0.0)
            self.thrust_coefficient[i] = data["thrust_coefficient"]
            self.cq[i] = data["cq"]
            self.motor_mass[i] = data["mass"]
            self.prop_diameter[i] = data["prop_diameter"]

            self.motor_time_constant[i] = data["motor_time_constant"]

            self.motor_time_constant_up[i] = data["motor_time_constant_up"]
            self.motor_time_constant_down[i] = data["motor_time_constant_down"]

            self.natural_freq_up[i] = data["natural_freq_up"]
            self.natural_freq_down[i] = data["natural_freq_down"]
            self.damping_ratio_up[i] = data["damping_ratio_up"]
            self.damping_ratio_down[i] = data["damping_ratio_down"]

            self.max_rps_rate_up[i] = data.get("max_rps_rate_up", 1000.0)
            self.max_rps_rate_down[i] = data.get("max_rps_rate_down", 800.0)

        self.max_force = self.thrust_coefficient * self.max_rps ** 2
        self.min_force = self.thrust_coefficient * self.min_rps ** 2


def optimize_motor_parameters(motor_model="asymmetric_first_order"):
        import sys
        import cma
        import numpy as np
        from plot_src import check_motor_response_from_poses_txt

        real_poses_path = "sim2real/2025_05_7_v4_seed=9/poses_real.txt"

        class ObjectiveCallback:
            def __init__(self, objective_func):
                self.objective_func = objective_func
                self.best_fitness = float('inf')
                self.best_solution = None

            def __call__(self, x):
                fitness = self.objective_func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                    print(f"New best solution found. f={self.best_fitness}")
                    print(f"x={self.best_solution.tolist()}\n" + "-"*40)
                
                return fitness


        def objective_function(x):
            
            if motor_model == "asymmetric_second_order":
                motor_model_pars = {
                "natural_freq_up": x[0],
                "natural_freq_down": x[1],
                "damping_ratio_up": x[2],
                "damping_ratio_down": x[3],
                }
            elif motor_model == "asymmetric_second_order_with_rate_limiter":
                motor_model_pars = {
                "natural_freq_up": x[0],
                "natural_freq_down": x[1],
                "damping_ratio_up": x[2],
                "damping_ratio_down": x[3],
                "max_rps_rate_up": x[4],
                "max_rps_rate_down": x[5],
                }
            elif motor_model == "asymmetric_first_order":
                motor_model_pars = {
                    "motor_time_constant_up": x[0],
                    "motor_time_constant_down": x[1],
                }
            elif motor_model == "first_order":
                motor_model_pars = {
                    "motor_time_constant": x[0],
                }
            else:
                raise ValueError(f"Unsupported motor model: {motor_model}. Supported models are 'asymmetric_second_order', 'first_order', 'asymmetric_first_order', 'asymmetric_second_order_with_rate_limiter', and 'instantaneous_forces'.")

            return check_motor_response_from_poses_txt(real_poses_path, True, motor_model_pars)

        objective_with_callback = ObjectiveCallback(objective_function)


        if motor_model == "asymmetric_second_order":
            # Initial guess for natural frequencies and damping ratios
            x0 = [60.0, 60.0, 0.7, 0.7]
            cma_stds = [15.0, 15.0, 0.1, 0.1]
        elif motor_model == "asymmetric_second_order_with_rate_limiter":
            # Initial guess for natural frequencies, damping ratios, and rate limits
            x0 = [60.0, 60.0, 0.7, 0.7, 700.0, 700.0]
            cma_stds = [15.0, 15.0, 0.1, 0.1, 100.0, 100.0]
        elif motor_model == "asymmetric_first_order":
            # Initial guess for motor time constants
            x0 = [0.02359879333607824, 0.05308359122401477]
            cma_stds = [0.005, 0.005]
        elif motor_model == "first_order":
            # Initial guess for motor time constant
            x0 = [0.047]
            cma_stds = [0.005]


        cma_sigma0 = 0.5

        result = cma.fmin(
            objective_with_callback,  # Pass the wrapped function here
            x0,
            cma_sigma0,
            {
                'bounds': [[0.0]*len(x0), [10000.0]*len(x0)],
                'CMA_stds': cma_stds,
                'popsize': 60,
            }
        )

        print("\n" + "="*50)
        print(f"Final best solution: {result[0]}")
        print(f"Final best fitness: {result[1]}")
        print("="*50)