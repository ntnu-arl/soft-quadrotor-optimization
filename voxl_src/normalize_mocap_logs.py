import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pyulog import ULog
except ModuleNotFoundError as exc:
    raise SystemExit("pyulog is required to normalize mocap logs") from exc

NUM_MOTORS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize VOXL mocap logs into a shared PX4 timestamp_us/time_s domain")
    parser.add_argument("--ulog-path", required=True)
    parser.add_argument("--solver-csv-path", required=True)
    parser.add_argument("--px4-motor-csv-path", required=True)
    parser.add_argument("--px4-flight-csv-path", required=True)
    parser.add_argument("--qualisys-csv-path", required=True)
    parser.add_argument("--battery-csv-path", help="Optional; omit to normalize logs from a run without a battery status log")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qualisys-match-max-position-error-m", required=True, type=float)
    return parser.parse_args()


def require_columns(df, columns, path):
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"{path} is missing columns: {missing_columns}")


def read_nonempty_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty")
    return df


def get_single_topic(ulog, topic_name):
    matches = [dataset for dataset in ulog.data_list if dataset.name == topic_name]
    if not matches:
        raise RuntimeError(f"Required ULog topic '{topic_name}' not found")
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one '{topic_name}' dataset, found {len(matches)}")
    return matches[0]


def find_nmpc_active_window_us(px4_flight_df):
    require_columns(px4_flight_df, ["timestamp_us", "nmpc_direct_actuator_active"], "PX4 flight CSV")
    active_indices = np.flatnonzero(px4_flight_df["nmpc_direct_actuator_active"].to_numpy(dtype=bool))
    if active_indices.size == 0:
        raise RuntimeError("No px4_flight_log rows have nmpc_direct_actuator_active=1")
    start_us = int(px4_flight_df["timestamp_us"].iloc[int(active_indices[0])])
    end_us = int(px4_flight_df["timestamp_us"].iloc[int(active_indices[-1])])
    if end_us < start_us:
        raise RuntimeError(f"Invalid NMPC active window: start_us={start_us} end_us={end_us}")
    return start_us, end_us


def add_time_s_and_trim(df, timestamp_column, start_us, end_us):
    normalized_df = df[(df[timestamp_column] >= start_us) & (df[timestamp_column] <= end_us)].copy()
    if normalized_df.empty:
        raise RuntimeError(
            f"No rows remain after trimming {timestamp_column} to the NMPC-active window "
            f"[{start_us}, {end_us}]"
        )
    normalized_df.insert(0, "time_s", (normalized_df[timestamp_column] - start_us) / 1e6)
    if timestamp_column != "timestamp_us":
        normalized_df.insert(0, "timestamp_us", normalized_df[timestamp_column].astype(np.int64))
    normalized_df.reset_index(drop=True, inplace=True)
    return normalized_df


def transformed_qualisys_position_ned(qualisys_df):
    return np.column_stack(
        [
            qualisys_df["position_y"].to_numpy(dtype=float),
            qualisys_df["position_x"].to_numpy(dtype=float),
            -qualisys_df["position_z"].to_numpy(dtype=float),
        ]
    )


def visual_odometry_position_ned(ulog_path):
    ulog = ULog(str(ulog_path), ["vehicle_visual_odometry"])
    dataset = get_single_topic(ulog, "vehicle_visual_odometry")
    timestamp_us = dataset.data["timestamp"].astype(np.int64)
    position_ned = np.column_stack(
        [
            dataset.data[f"position[{axis}]"].astype(float)
            for axis in range(3)
        ]
    )
    return timestamp_us, position_ned


def build_qualisys_px4_time_map(ulog_path, qualisys_df, match_max_position_error_m):
    require_columns(
        qualisys_df,
        ["timestamp_ns", "position_x", "position_y", "position_z"],
        "Qualisys CSV",
    )
    qualisys_timestamp_us = (qualisys_df["timestamp_ns"].to_numpy(dtype=np.int64) // 1000).astype(np.float64)
    qualisys_position_ned = transformed_qualisys_position_ned(qualisys_df)
    visual_timestamp_us, visual_position_ned = visual_odometry_position_ned(ulog_path)

    matched_qualisys_timestamp_us = []
    matched_visual_timestamp_us = []
    matched_position_errors = []
    for visual_index in range(visual_position_ned.shape[0]):
        position_errors = np.linalg.norm(qualisys_position_ned - visual_position_ned[visual_index], axis=1)
        qualisys_index = int(np.argmin(position_errors))
        position_error = float(position_errors[qualisys_index])
        if position_error > match_max_position_error_m:
            raise RuntimeError(
                "Could not match vehicle_visual_odometry to raw Qualisys position: "
                f"visual_index={visual_index}, position_error_m={position_error}"
            )
        matched_qualisys_timestamp_us.append(float(qualisys_timestamp_us[qualisys_index]))
        matched_visual_timestamp_us.append(float(visual_timestamp_us[visual_index]))
        matched_position_errors.append(position_error)

    matched_qualisys_timestamp_us = np.asarray(matched_qualisys_timestamp_us, dtype=np.float64)
    matched_visual_timestamp_us = np.asarray(matched_visual_timestamp_us, dtype=np.float64)
    matched_position_errors = np.asarray(matched_position_errors, dtype=np.float64)

    if np.any(np.diff(matched_qualisys_timestamp_us) <= 0.0):
        raise RuntimeError("Matched Qualisys timestamps are not strictly increasing")
    if np.any(np.diff(matched_visual_timestamp_us) <= 0.0):
        raise RuntimeError("Matched PX4 visual-odometry timestamps are not strictly increasing")

    return matched_qualisys_timestamp_us, matched_visual_timestamp_us, matched_position_errors


def map_epoch_us_to_px4_us(epoch_timestamp_us, matched_qualisys_us, matched_px4_us):
    mapped_interval_mask = (
        (epoch_timestamp_us >= matched_qualisys_us[0])
        & (epoch_timestamp_us <= matched_qualisys_us[-1])
    )
    if not np.any(mapped_interval_mask):
        raise RuntimeError("No samples lie inside the proven PX4 visual-odometry mapping interval")
    px4_timestamp_us = np.interp(epoch_timestamp_us[mapped_interval_mask], matched_qualisys_us, matched_px4_us)
    return mapped_interval_mask, px4_timestamp_us


def normalize_qualisys_df(qualisys_df, matched_qualisys_us, matched_px4_us, start_us, end_us):
    qualisys_timestamp_us = (qualisys_df["timestamp_ns"].to_numpy(dtype=np.int64) // 1000).astype(np.float64)
    mapped_interval_mask, px4_timestamp_us = map_epoch_us_to_px4_us(
        qualisys_timestamp_us,
        matched_qualisys_us,
        matched_px4_us,
    )
    qualisys_df = qualisys_df[mapped_interval_mask].copy()

    normalized_df = qualisys_df.copy()
    normalized_df.insert(0, "timestamp_us", np.rint(px4_timestamp_us).astype(np.int64))
    normalized_df["position_ned_x"] = normalized_df["position_y"]
    normalized_df["position_ned_y"] = normalized_df["position_x"]
    normalized_df["position_ned_z"] = -normalized_df["position_z"]
    normalized_df = add_time_s_and_trim(normalized_df, "timestamp_us", start_us, end_us)
    return normalized_df


def normalize_battery_df(battery_df, matched_qualisys_us, matched_px4_us, start_us, end_us):
    require_columns(
        battery_df,
        ["timestamp_us", "voltage_v", "current_a", "cell_count", "per_cell_voltage_v", "percent_estimate"],
        "Battery status CSV",
    )
    epoch_timestamp_us = battery_df["timestamp_us"].to_numpy(dtype=np.float64)
    mapped_interval_mask, px4_timestamp_us = map_epoch_us_to_px4_us(
        epoch_timestamp_us,
        matched_qualisys_us,
        matched_px4_us,
    )
    battery_df = battery_df[mapped_interval_mask].copy()
    battery_df["timestamp_us"] = np.rint(px4_timestamp_us).astype(np.int64)
    return add_time_s_and_trim(battery_df, "timestamp_us", start_us, end_us)


def build_alignment_df(matched_qualisys_us, matched_px4_us, matched_position_errors, start_us, end_us):
    alignment_df = pd.DataFrame(
        {
            "qualisys_timestamp_us": matched_qualisys_us.astype(np.int64),
            "px4_timestamp_us": matched_px4_us.astype(np.int64),
            "position_error_m": matched_position_errors,
        }
    )
    alignment_df = alignment_df[
        (alignment_df["px4_timestamp_us"] >= start_us) & (alignment_df["px4_timestamp_us"] <= end_us)
    ].reset_index(drop=True)
    if alignment_df.empty:
        raise RuntimeError("No Qualisys/PX4 alignment samples remain inside the NMPC-active window")
    return alignment_df


def main():
    args = parse_args()
    ulog_path = Path(args.ulog_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solver_df = read_nonempty_csv(args.solver_csv_path)
    px4_motor_df = read_nonempty_csv(args.px4_motor_csv_path)
    px4_flight_df = read_nonempty_csv(args.px4_flight_csv_path)
    qualisys_df = read_nonempty_csv(args.qualisys_csv_path)
    battery_df = read_nonempty_csv(args.battery_csv_path) if args.battery_csv_path else None

    require_columns(solver_df, ["sample_timestamp_us"], args.solver_csv_path)
    require_columns(px4_motor_df, ["timestamp_us"], args.px4_motor_csv_path)
    require_columns(px4_flight_df, ["timestamp_us", "nmpc_direct_actuator_active"], args.px4_flight_csv_path)

    active_start_us, active_end_us = find_nmpc_active_window_us(px4_flight_df)

    matched_qualisys_us, matched_px4_us, matched_position_errors = build_qualisys_px4_time_map(
        ulog_path,
        qualisys_df,
        args.qualisys_match_max_position_error_m,
    )

    solver_normalized_df = add_time_s_and_trim(solver_df, "sample_timestamp_us", active_start_us, active_end_us)
    px4_motor_normalized_df = add_time_s_and_trim(px4_motor_df, "timestamp_us", active_start_us, active_end_us)
    px4_flight_normalized_df = add_time_s_and_trim(px4_flight_df, "timestamp_us", active_start_us, active_end_us)
    qualisys_normalized_df = normalize_qualisys_df(
        qualisys_df,
        matched_qualisys_us,
        matched_px4_us,
        active_start_us,
        active_end_us,
    )
    battery_normalized_df = None
    if battery_df is not None:
        battery_normalized_df = normalize_battery_df(
            battery_df,
            matched_qualisys_us,
            matched_px4_us,
            active_start_us,
            active_end_us,
        )
    alignment_df = build_alignment_df(
        matched_qualisys_us,
        matched_px4_us,
        matched_position_errors,
        active_start_us,
        active_end_us,
    )

    solver_normalized_df.to_csv(output_dir / "nmpc_solver_data_combined_normalized.csv", index=False)
    px4_motor_normalized_df.to_csv(output_dir / "px4_motor_log_normalized.csv", index=False)
    px4_flight_normalized_df.to_csv(output_dir / "px4_flight_log_normalized.csv", index=False)
    qualisys_normalized_df.to_csv(output_dir / "qualisys_odom_normalized.csv", index=False)
    if battery_normalized_df is not None:
        battery_normalized_df.to_csv(output_dir / "battery_status_log_normalized.csv", index=False)
    alignment_df.to_csv(output_dir / "qualisys_px4_time_alignment.csv", index=False)

    print(f"Normalized logs saved to {output_dir}")
    print(f"nmpc_active_start_timestamp_us={active_start_us}")
    print(f"nmpc_active_end_timestamp_us={active_end_us}")
    print(f"nmpc_active_duration_s={(active_end_us - active_start_us) * 1e-6:.9g}")
    print(f"qualisys_alignment_max_position_error_m={alignment_df['position_error_m'].max():.9g}")


if __name__ == "__main__":
    main()
