"""
Plot timing analysis from VOXL HITL NMPC run.

Usage:
  python voxl_src/plot_voxl_timing.py --input results/voxl_hitl_metrics.json --output results/voxl_timing.pdf
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

loop_times = np.array(data['loop_times_ms'])
timing_log = data['timing_log']
nmpc_motor_traces = data['nmpc_motor_traces']

rtt_vals = np.array([t['rtt_ms'] for t in timing_log if t['rtt_ms'] > 0])
solve_vals = np.array([t['solve_us'] for t in timing_log if t['solve_us'] > 0]) / 1000.0
reinit_steps = [i for i, t in enumerate(timing_log) if t.get('flags', 0) & 0x01]
px4_vals = loop_times[:len(rtt_vals)] - rtt_vals[:len(loop_times)]
px4_vals = px4_vals[px4_vals > 0]

env_ids = sorted({trace['env'] for trace in nmpc_motor_traces})

with PdfPages(args.output) as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    ax.hist(solve_vals, bins=50, edgecolor='black', linewidth=0.5)
    ax.axvline(solve_vals.mean(), color='red', linestyle='--', label=f'mean={solve_vals.mean():.2f}ms')
    ax.set_xlabel('Solve time (ms)')
    ax.set_ylabel('Count')
    ax.set_title('VOXL Solver Time')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(px4_vals, bins=50, edgecolor='black', linewidth=0.5)
    ax.axvline(px4_vals.mean(), color='red', linestyle='--', label=f'mean={px4_vals.mean():.2f}ms')
    ax.set_xlabel('PX4 iteration time (ms)')
    ax.set_ylabel('Count')
    ax.set_title('PX4 Iteration Time (loop - RTT)')
    ax.legend()

    ax = axes[1, 0]
    freq = 1000.0 / loop_times
    ax.plot(freq, linewidth=0.5)
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='100 Hz target')
    for i, rs in enumerate(reinit_steps):
        ax.axvline(rs, color='green', linestyle=':', alpha=0.4, linewidth=0.8, label='setpoint change' if i == 0 else None)
    ax.set_xlabel('Step')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Control Loop Frequency')
    ax.legend()

    ax = axes[1, 1]
    deployed_freq = 1000.0 / solve_vals
    min_freq = deployed_freq.min()
    ax.plot(deployed_freq, linewidth=0.5)
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='100 Hz')
    ax.axhline(min_freq, color='orange', linestyle='--', alpha=0.5, label=f'{min_freq:.0f} Hz')
    for i, rs in enumerate(reinit_steps):
        ax.axvline(rs, color='green', linestyle=':', alpha=0.4, linewidth=0.8, label='setpoint change' if i == 0 else None)
    ax.set_xlabel('Step')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Deployed Frequency (VOXL only, no network)')
    ax.legend()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(len(env_ids), 2, figsize=(12, 4 * len(env_ids)), sharex=False)
    if len(env_ids) == 1:
        axes = np.array([axes])

    for row_index, env_id in enumerate(env_ids):
        env_traces = [trace for trace in nmpc_motor_traces if trace['env'] == env_id]
        time_s = np.array([trace['time_s'] for trace in env_traces], dtype=float)
        sim_actions = np.array([trace['sim_action'] for trace in env_traces], dtype=float)
        motor_commands = np.array([trace['nmpc_motor_command'] for trace in env_traces], dtype=float)
        measured_rps = np.array([trace['measured_motor_rps'] for trace in env_traces], dtype=float)

        ax = axes[row_index, 0]
        for motor_index in range(sim_actions.shape[1]):
            ax.plot(time_s, sim_actions[:, motor_index], linewidth=0.8, label=f'action_{motor_index}')
        ax.set_ylabel('Action')
        ax.set_xlabel('Time [s]')
        ax.set_title(f'Applied Simulator Actions (env {env_id})')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[row_index, 1]
        ax_rps = ax.twinx()
        for motor_index in range(motor_commands.shape[1]):
            ax.plot(time_s, motor_commands[:, motor_index], linewidth=0.8, label=f'nmpc_u{motor_index}')
            ax_rps.plot(
                time_s,
                measured_rps[:, motor_index],
                linewidth=0.8,
                linestyle=':',
                alpha=0.7,
                label=f'rps{motor_index}',
            )
        ax.set_ylabel('NMPC command')
        ax_rps.set_ylabel('RPS')
        ax.set_xlabel('Time [s]')
        ax.set_title(f'NMPC Motor Command + Measured RPS (env {env_id})')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_rps.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved timing plots to {args.output}")
