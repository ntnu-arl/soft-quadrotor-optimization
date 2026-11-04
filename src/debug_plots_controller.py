"""
NMPC controller debug logging and plotting.

Usage:
  - NMPCLogger is imported by nmpc_enjoy.py to log per-step data to CSV.
  - Run this script directly to generate debug plots from the CSV:

    python src/debug_plots_controller.py
"""

import csv
import os
import numpy as np
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "results" / "nmpc_debug_log.csv"
PLOT_PATH = Path(__file__).parent.parent / "results" / "nmpc_debug_plots.pdf"


def _build_header(num_motors, horizon_N):
    cols = ['step', 'time']
    for v in ['pos', 'vel', 'setpoint_pos', 'setpoint_vel']:
        cols += [f'{v}_x', f'{v}_y', f'{v}_z']
    cols += ['setpoint_yaw', 'setpoint_yaw_rate']
    cols += ['quat_w', 'quat_x', 'quat_y', 'quat_z']
    cols += ['desired_quat_w', 'desired_quat_x', 'desired_quat_y', 'desired_quat_z']
    cols += ['omega_x', 'omega_y', 'omega_z']
    for i in range(num_motors):
        cols.append(f'motor_force_{i}')
    for i in range(num_motors):
        cols.append(f'prev_command_rps_{i}')
    for i in range(num_motors):
        cols.append(f'commanded_motor_rps_{i}')
    for i in range(num_motors):
        cols.append(f'thrust_coeff_{i}')
    for i in range(num_motors):
        cols.append(f'action_{i}')
    cols += ['wrench_fx', 'wrench_fy', 'wrench_fz', 'wrench_tx', 'wrench_ty', 'wrench_tz']
    cols += ['solver_status', 'solver_cost']
    cols += ['active_nmpc_mode']
    for v in ['pred_pos_end', 'pred_vel_end']:
        cols += [f'{v}_x', f'{v}_y', f'{v}_z']
    cols += ['pred_quat_end_w', 'pred_quat_end_x', 'pred_quat_end_y', 'pred_quat_end_z']
    for k_step in range(horizon_N):
        for i in range(num_motors):
            cols.append(f'horizon_u_k{k_step}_m{i}')
    return cols


class NMPCLogger:
    def __init__(self, num_motors, horizon_N):
        self.num_motors = num_motors
        self.horizon_N = horizon_N
        self.header = _build_header(num_motors, horizon_N)
        os.makedirs(CSV_PATH.parent, exist_ok=True)
        if CSV_PATH.exists():
            os.remove(CSV_PATH)
        self._file = open(CSV_PATH, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)

    def log_step(self, step, time_s, pos, vel, setpoint_pos, setpoint_vel, setpoint_yaw, setpoint_yaw_rate,
                 quat, ang_vel, controller, allocator, actions):
        dd = controller.debug_data
        env_i = 0

        row = [step, f'{time_s:.6f}']
        row += [f'{v:.6f}' for v in pos]
        row += [f'{v:.6f}' for v in vel]
        row += [f'{v:.6f}' for v in setpoint_pos]
        row += [f'{v:.6f}' for v in setpoint_vel]
        row += [f'{setpoint_yaw:.6f}', f'{setpoint_yaw_rate:.6f}']
        row += [f'{v:.6f}' for v in quat]
        row += [f'{v:.6f}' for v in controller.desired_quat[env_i].cpu().tolist()]
        row += [f'{v:.6f}' for v in ang_vel]
        row += [f'{v:.6f}' for v in controller.motor_forces[env_i].cpu().tolist()]
        row += [f'{v:.6f}' for v in dd['previous_commanded_motor_rps'][env_i].tolist()]
        row += [f'{v:.6f}' for v in dd['commanded_motor_rps'][env_i].tolist()]
        row += [f'{v:.6f}' for v in controller.thrust_coefficients.tolist()]
        row += [f'{v:.6f}' for v in actions]

        B = allocator.allocation_matrix[env_i].cpu().numpy()
        motor_forces_np = controller.motor_forces[env_i].cpu().numpy()
        wrench = B @ motor_forces_np
        row += [f'{v:.6f}' for v in wrench.tolist()]

        row.append(int(dd['solver_status'][env_i]))
        row.append(f'{dd["solver_cost"][env_i]:.6f}')
        row.append(dd['active_nmpc_mode'][env_i])
        row += [f'{v:.6f}' for v in dd['predicted_pos_end'][env_i].tolist()]
        row += [f'{v:.6f}' for v in dd['predicted_vel_end'][env_i].tolist()]
        row += [f'{v:.6f}' for v in dd['predicted_quat_end'][env_i].tolist()]

        for k in range(self.horizon_N):
            row += [f'{v:.6f}' for v in dd['horizon_controls'][env_i, k].tolist()]

        self._writer.writerow(row)

    def close(self):
        self._file.close()
        print(f"[INFO] NMPC debug log saved to {CSV_PATH}")


def generate_plots():
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    df = pd.read_csv(CSV_PATH)
    t = df['time'].values

    num_motors = 0
    while f'motor_force_{num_motors}' in df.columns:
        num_motors += 1

    motor_force_cols = [(f'motor_force_{i}', f'motor {i}') for i in range(num_motors)]
    action_cols = [(f'action_{i}', f'motor {i}') for i in range(num_motors)]

    with PdfPages(str(PLOT_PATH)) as pdf:

        # Page 1: Position tracking (actual vs setpoint vs predicted horizon end)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        for dim, label in enumerate(['x', 'y', 'z']):
            axes[dim].plot(t, df[f'pos_{label}'].values, label='actual', linewidth=0.8)
            axes[dim].plot(t, df[f'setpoint_pos_{label}'].values, label='setpoint', linewidth=0.8, linestyle='--')
            axes[dim].plot(t, df[f'pred_pos_end_{label}'].values, label='predicted (horizon end)', linewidth=0.8, linestyle=':', alpha=0.7)
            axes[dim].set_ylabel(f'{label} [m]', fontsize=9)
            axes[dim].set_title(f'Position tracking: {label}', fontsize=10)
            axes[dim].legend(fontsize=8)
            axes[dim].grid(True, alpha=0.3)
            axes[dim].set_xlabel('Time [s]', fontsize=9)
        fig.suptitle('Position: Actual vs Setpoint vs Predicted Horizon End', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Velocity tracking
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        for dim, label in enumerate(['x', 'y', 'z']):
            axes[dim].plot(t, df[f'vel_{label}'].values, label='actual', linewidth=0.8)
            axes[dim].plot(t, df[f'setpoint_vel_{label}'].values, label='setpoint', linewidth=0.8, linestyle='--')
            axes[dim].plot(t, df[f'pred_vel_end_{label}'].values, label='predicted (horizon end)', linewidth=0.8, linestyle=':', alpha=0.7)
            axes[dim].set_ylabel(f'{label} [m/s]', fontsize=9)
            axes[dim].set_title(f'Velocity tracking: {label}', fontsize=10)
            axes[dim].legend(fontsize=8)
            axes[dim].grid(True, alpha=0.3)
            axes[dim].set_xlabel('Time [s]', fontsize=9)
        fig.suptitle('Velocity: Actual vs Setpoint vs Predicted Horizon End', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Quaternion (actual vs desired)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        for dim, comp in enumerate(['w', 'x', 'y', 'z']):
            axes[dim].plot(t, df[f'quat_{comp}'].values, label='actual', linewidth=0.8)
            axes[dim].plot(t, df[f'desired_quat_{comp}'].values, label='desired (step 1)', linewidth=0.8, linestyle='--')
            axes[dim].plot(t, df[f'pred_quat_end_{comp}'].values, label='predicted (horizon end)', linewidth=0.8, linestyle=':', alpha=0.7)
            axes[dim].set_ylabel(f'q_{comp}', fontsize=9)
            axes[dim].set_title(f'Quaternion {comp}', fontsize=10)
            axes[dim].legend(fontsize=8)
            axes[dim].grid(True, alpha=0.3)
            axes[dim].set_xlabel('Time [s]', fontsize=9)
        fig.suptitle('Orientation: Actual vs Desired vs Predicted End', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Angular velocity + position/velocity error norms
        fig, axes = plt.subplots(4, 1, figsize=(14, 14))
        for dim, label in enumerate(['x', 'y', 'z']):
            axes[0].plot(t, df[f'omega_{label}'].values, label=label, linewidth=0.8)
        axes[0].set_ylabel('rad/s', fontsize=9)
        axes[0].set_title('Body Angular Velocity', fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        pos_err = np.sqrt(sum((df[f'pos_{l}'].values - df[f'setpoint_pos_{l}'].values)**2 for l in ['x', 'y', 'z']))
        axes[1].plot(t, pos_err, linewidth=0.8, color='tab:red')
        axes[1].set_ylabel('m', fontsize=9)
        axes[1].set_title('Position Error Norm', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        vel_err = np.sqrt(sum((df[f'vel_{l}'].values - df[f'setpoint_vel_{l}'].values)**2 for l in ['x', 'y', 'z']))
        axes[2].plot(t, vel_err, linewidth=0.8, color='tab:orange')
        axes[2].set_ylabel('m/s', fontsize=9)
        axes[2].set_title('Velocity Error Norm', fontsize=10)
        axes[2].grid(True, alpha=0.3)

        quat_err = 1.0 - (df['quat_w']*df['desired_quat_w'] + df['quat_x']*df['desired_quat_x'] +
                          df['quat_y']*df['desired_quat_y'] + df['quat_z']*df['desired_quat_z']).values**2
        axes[3].plot(t, quat_err, linewidth=0.8, color='tab:purple')
        axes[3].set_ylabel('1 - <q,q_d>^2', fontsize=9)
        axes[3].set_title('Attitude Error (quaternion)', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel('Time [s]', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Motor forces and actions
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for col, label in motor_force_cols:
            axes[0].plot(t, df[col].values, label=label, linewidth=0.8)
        axes[0].set_ylabel('N', fontsize=9)
        axes[0].set_title('Motor Forces (NMPC output)', fontsize=10)
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        for col, label in action_cols:
            axes[1].plot(t, df[col].values, label=label, linewidth=0.8)
        axes[1].set_ylabel('normalized', fontsize=9)
        axes[1].set_title('Actions (to simulator)', fontsize=10)
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        axes[1].axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)
        for ax in axes:
            ax.set_xlabel('Time [s]', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Wrench (forces and torques)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for comp, label in [('fx', 'Fx'), ('fy', 'Fy'), ('fz', 'Fz')]:
            axes[0].plot(t, df[f'wrench_{comp}'].values, label=label, linewidth=0.8)
        axes[0].set_ylabel('N', fontsize=9)
        axes[0].set_title('Body Wrench - Forces (B @ u)', fontsize=10)
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        for comp, label in [('tx', 'Tx'), ('ty', 'Ty'), ('tz', 'Tz')]:
            axes[1].plot(t, df[f'wrench_{comp}'].values, label=label, linewidth=0.8)
        axes[1].set_ylabel('Nm', fontsize=9)
        axes[1].set_title('Body Wrench - Torques (B @ u)', fontsize=10)
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel('Time [s]', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: Solver diagnostics
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes[0].plot(t, df['solver_status'].values, linewidth=0.8, color='tab:red', marker='.', markersize=2, linestyle='none')
        axes[0].set_ylabel('status', fontsize=9)
        axes[0].set_title('ACADOS Solver Status (0 = success)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yticks([0, 1, 2, 3, 4])

        axes[1].plot(t, df['solver_cost'].values, linewidth=0.8, color='tab:blue')
        axes[1].set_ylabel('cost', fontsize=9)
        axes[1].set_title('Optimal Cost (objective value)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel('Time [s]', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 8: Horizon control plan (motor forces over predicted horizon at select timesteps)
        n_snapshots = 6
        total_steps = len(t)
        snapshot_indices = np.linspace(0, total_steps - 1, n_snapshots, dtype=int)
        horizon_len = 0
        while f'horizon_u_k{horizon_len}_m0' in df.columns:
            horizon_len += 1

        if horizon_len > 0:
            fig, axes = plt.subplots(n_snapshots, 1, figsize=(14, 3 * n_snapshots))
            for ax_idx, step_idx in enumerate(snapshot_indices):
                ax = axes[ax_idx]
                k_steps = np.arange(horizon_len)
                for m in range(num_motors):
                    forces = [df[f'horizon_u_k{k}_m{m}'].iloc[step_idx] for k in range(horizon_len)]
                    ax.plot(k_steps, forces, label=f'motor {m}', linewidth=0.8)
                ax.set_ylabel('N', fontsize=9)
                ax.set_title(f'Horizon control plan at t={t[step_idx]:.3f}s (step {step_idx})', fontsize=10)
                ax.legend(fontsize=7, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Horizon step k', fontsize=9)
            fig.suptitle('NMPC Predicted Control Sequence (snapshots)', fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[INFO] NMPC debug plots saved to {PLOT_PATH}")


if __name__ == '__main__':
    generate_plots()
