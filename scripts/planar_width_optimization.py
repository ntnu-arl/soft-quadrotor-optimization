"""Planar width optimization via grid search over fold axis tilt angles.

Rotating side_base_rotation_x by theta and compensating with
side_motor_rotation_x = -theta keeps motor position and thrust unchanged
but tilts the fold axis, enabling torque constraint satisfaction and
different folded widths.

Produces a heatmap of folded width vs (base_x_left, base_x_right),
with infeasible (torque-violating) points shown in white.

Usage:
    python scripts/planar_width_optimization.py
    python scripts/planar_width_optimization.py --plot
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from airframe_encoding import Airframe, PARAM_BOUNDS, AIRFRAME_CONSTANTS
from airframe_validations import get_max_arm_angles_no_collision

PLANAR_ARM_LENGTH_FRONT_BACK = 0.088
PLANAR_ARM_LENGTH_SIDE = 0.078

PARAM_NAMES = list(PARAM_BOUNDS.keys())
PARAM_BOUNDS_LIST = list(PARAM_BOUNDS.values())

IDX_BASE_X_LEFT = PARAM_NAMES.index('side_base_rotation_x_left')
IDX_BASE_X_RIGHT = PARAM_NAMES.index('side_base_rotation_x_right')
IDX_MOTOR_X_LEFT = PARAM_NAMES.index('side_motor_rotation_x_left')
IDX_MOTOR_X_RIGHT = PARAM_NAMES.index('side_motor_rotation_x_right')

BASE_X_LO, BASE_X_HI = PARAM_BOUNDS_LIST[IDX_BASE_X_LEFT]


@dataclass
class Config:
    experiment_name: str = f"planar_width"
    grid_resolution: int = 301
    minimum_arm_torque: float = 0.02

    def __post_init__(self):
        self.results_dir = Path(__file__).parent.parent / "results/width_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = self.results_dir / f"{self.experiment_name}.csv"
        self.heatmap_pdf = self.results_dir / f"{self.experiment_name}_heatmap.pdf"


def build_planar_x_0_1():
    x = np.array([0.5] * len(PARAM_BOUNDS))
    for name, val in [('front_arm_length', PLANAR_ARM_LENGTH_FRONT_BACK),
                      ('back_arm_length', PLANAR_ARM_LENGTH_FRONT_BACK),
                      ('side_arm_seg2_length_left', PLANAR_ARM_LENGTH_SIDE),
                      ('side_arm_seg2_length_right', PLANAR_ARM_LENGTH_SIDE)]:
        lo, hi = PARAM_BOUNDS[name]
        x[PARAM_NAMES.index(name)] = (val - lo) / (hi - lo)
    return x


def check_torque_constraint(airframe, minimum_arm_torque):
    motor_data = airframe.compute_motor_positions_and_thrusts()
    motor_directions = AIRFRAME_CONSTANTS['motor_directions']
    cq = AIRFRAME_CONSTANTS['cq']
    max_force = AIRFRAME_CONSTANTS['thrust_coefficient'] * AIRFRAME_CONSTANTS['max_rps'] ** 2

    for side, motor_idx, joint_key, axis_key in [
        ('left', 1, 'joint_1_position', 'joint_1_axis'),
        ('right', 3, 'joint_3_position', 'joint_3_axis'),
    ]:
        joint_pos = np.array(motor_data[joint_key])
        joint_axis = np.array(motor_data[axis_key])
        motor_pos = np.array(motor_data['motor_positions'][motor_idx])
        motor_thrust = np.array(motor_data['motor_thrust_directions'][motor_idx])

        r = motor_pos - joint_pos
        torque_thrust = np.dot(np.cross(r, max_force * motor_thrust), joint_axis)
        counter_torque = motor_directions[motor_idx] * cq * max_force * np.dot(motor_thrust, joint_axis)
        net_torque = torque_thrust + counter_torque

        if side == 'left' and net_torque + minimum_arm_torque > 0:
            return False
        if side == 'right' and minimum_arm_torque - net_torque > 0:
            return False
    return True


def evaluate_single_point(base_x_left_deg, base_x_right_deg, minimum_arm_torque):
    x_0_1 = build_planar_x_0_1()

    lo_bl, hi_bl = PARAM_BOUNDS_LIST[IDX_BASE_X_LEFT]
    lo_br, hi_br = PARAM_BOUNDS_LIST[IDX_BASE_X_RIGHT]
    x_0_1[IDX_BASE_X_LEFT] = (base_x_left_deg - lo_bl) / (hi_bl - lo_bl)
    x_0_1[IDX_BASE_X_RIGHT] = (base_x_right_deg - lo_br) / (hi_br - lo_br)

    lo_ml, hi_ml = PARAM_BOUNDS_LIST[IDX_MOTOR_X_LEFT]
    lo_mr, hi_mr = PARAM_BOUNDS_LIST[IDX_MOTOR_X_RIGHT]
    x_0_1[IDX_MOTOR_X_LEFT] = (-base_x_left_deg - lo_ml) / (hi_ml - lo_ml)
    x_0_1[IDX_MOTOR_X_RIGHT] = (-base_x_right_deg - lo_mr) / (hi_mr - lo_mr)

    airframe = Airframe.from_x_0_1(x_0_1)

    if not check_torque_constraint(airframe, minimum_arm_torque):
        return None, None, None

    max_left, max_right, aabb_info = get_max_arm_angles_no_collision(airframe, delta_deg=0.01, verbose=False)
    width = aabb_info['max_width']

    return width, aabb_info, airframe.x_0_1


def _worker(args):
    i, j, base_x_left_deg, base_x_right_deg, minimum_arm_torque = args
    width, aabb_info, x_0_1 = evaluate_single_point(base_x_left_deg, base_x_right_deg, minimum_arm_torque)
    unfolded_width = aabb_info['unfolded_width'] if aabb_info else None
    return i, j, base_x_left_deg, base_x_right_deg, width, unfolded_width, x_0_1


def run_grid_search(cfg):
    left_angles = np.linspace(BASE_X_LO, BASE_X_HI, cfg.grid_resolution)
    right_angles = np.linspace(BASE_X_LO, BASE_X_HI, cfg.grid_resolution)
    total = cfg.grid_resolution ** 2

    print(f"\n{'='*80}")
    print(f"Planar Width Grid Search")
    print(f"{'='*80}")
    print(f"Grid: {cfg.grid_resolution} x {cfg.grid_resolution} = {total} points")
    print(f"base_x range: [{BASE_X_LO}, {BASE_X_HI}] deg")
    print(f"{'='*80}\n")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    for i, bxl in enumerate(left_angles):
        for j, bxr in enumerate(right_angles):
            tasks.append((i, j, bxl, bxr, cfg.minimum_arm_torque))

    width_grid = np.full((cfg.grid_resolution, cfg.grid_resolution), np.nan)
    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            i, j, bxl, bxr, width, unfolded_width, x_0_1 = future.result()
            if width is not None:
                width_grid[i, j] = width
                row = {
                    'base_x_left_deg': bxl,
                    'base_x_right_deg': bxr,
                    'folded_width': width,
                    'unfolded_width': unfolded_width,
                    'feasible': True,
                }
                if x_0_1 is not None:
                    row.update({f'x_{k}': x_0_1[k] for k in range(len(x_0_1))})
                results.append(row)

            if count % 500 == 0 or count == total:
                elapsed = time.time() - start_time
                eta = elapsed / count * (total - count)
                n_feasible = np.count_nonzero(~np.isnan(width_grid))
                print(f"  [{count}/{total}] {n_feasible} feasible, elapsed {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(cfg.results_csv, index=False)

    n_feasible = np.count_nonzero(~np.isnan(width_grid))
    print(f"\nGrid search complete: {n_feasible}/{total} feasible ({n_feasible/total*100:.1f}%)")
    if n_feasible > 0:
        best_idx = np.unravel_index(np.nanargmin(width_grid), width_grid.shape)
        print(f"Best width: {width_grid[best_idx]*100:.2f} cm at "
              f"base_x_left={left_angles[best_idx[0]]:.2f} deg, "
              f"base_x_right={right_angles[best_idx[1]]:.2f} deg")
        print("x_0_1 of best planar design:")
        best_x_0_1 = df.loc[df['folded_width'].idxmin(), df.columns.str.startswith('x_')]
        for k, v in best_x_0_1.items():
            print(f"{v}", end=" ")
        print("")
    print(f"Results saved to: {cfg.results_csv}")

    return df, width_grid, left_angles, right_angles



def generate_heatmap(cfg, width_grid, left_angles, right_angles):
    fig, ax = plt.subplots(figsize=(7, 6))

    masked = np.ma.masked_invalid(width_grid * 100)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')

    im = ax.pcolormesh(right_angles, left_angles, masked, cmap=cmap, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Folded Width (cm)', fontsize=11)

    ax.set_xlabel('side_base_rotation_x_right (deg)', fontsize=11)
    ax.set_ylabel('side_base_rotation_x_left (deg)', fontsize=11)
    ax.set_aspect('equal')

    if np.any(~np.isnan(width_grid)):
        best_idx = np.unravel_index(np.nanargmin(width_grid), width_grid.shape)
        ax.plot(right_angles[best_idx[1]], left_angles[best_idx[0]],
                marker='x', color='black', markersize=12,
                markeredgewidth=2, label='Planar', zorder=5)

    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(cfg.heatmap_pdf, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {cfg.heatmap_pdf}")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planar width grid search")
    parser.add_argument("--plot", action='store_true', help="Generate plots from existing results")
    args = parser.parse_args()

    cfg = Config()

    if args.plot:
        for csv_path in sorted(cfg.results_dir.glob("planar_width_*.csv")):
            cfg.results_csv = csv_path
            cfg.heatmap_pdf = csv_path.with_name(csv_path.stem + "_heatmap.pdf")

            df = pd.read_csv(csv_path)
            if 'base_x_left_deg' not in df.columns:
                continue
            left_angles = np.sort(df['base_x_left_deg'].unique())
            right_angles = np.sort(df['base_x_right_deg'].unique())
            n_left, n_right = len(left_angles), len(right_angles)
            width_grid = np.full((n_left, n_right), np.nan)
            for _, row in df.iterrows():
                i = np.searchsorted(left_angles, row['base_x_left_deg'])
                j = np.searchsorted(right_angles, row['base_x_right_deg'])
                if i < n_left and j < n_right:
                    width_grid[i, j] = row['folded_width']
            generate_heatmap(cfg, width_grid, left_angles, right_angles)
    else:
        df, width_grid, left_angles, right_angles = run_grid_search(cfg)
        generate_heatmap(cfg, width_grid, left_angles, right_angles)
