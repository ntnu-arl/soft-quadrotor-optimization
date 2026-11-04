import argparse

import numpy as np
import pandas as pd

try:
    from pyulog import ULog
except ModuleNotFoundError as exc:
    raise SystemExit("pyulog is required to extract PX4 flight logs") from exc


def get_single_topic(ulog, topic_name):
    matches = [dataset for dataset in ulog.data_list if dataset.name == topic_name]

    if not matches:
        raise KeyError(f"Required ULog topic '{topic_name}' not found")

    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one '{topic_name}' dataset, found {len(matches)}")

    return matches[0]


def get_optional_single_topic(ulog, topic_name):
    matches = [dataset for dataset in ulog.data_list if dataset.name == topic_name]

    if not matches:
        return None

    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one '{topic_name}' dataset, found {len(matches)}")

    return matches[0]


def require_fields(dataset, field_names):
    available_fields = {field.field_name for field in dataset.field_data}

    for field_name in field_names:
        if field_name not in available_fields:
            raise KeyError(f"Required field '{field_name}' missing from topic '{dataset.name}'")


def build_offboard_control_mode_dataframe(dataset):
    required_fields = [
        "timestamp",
        "position",
        "velocity",
        "acceleration",
        "attitude",
        "body_rate",
        "actuator",
    ]
    require_fields(dataset, required_fields)

    df = pd.DataFrame(
        {
            "timestamp_us": dataset.data["timestamp"].astype(np.int64),
            "offboard_position": dataset.data["position"].astype(bool).astype(np.int8),
            "offboard_velocity": dataset.data["velocity"].astype(bool).astype(np.int8),
            "offboard_acceleration": dataset.data["acceleration"].astype(bool).astype(np.int8),
            "offboard_attitude": dataset.data["attitude"].astype(bool).astype(np.int8),
            "offboard_body_rate": dataset.data["body_rate"].astype(bool).astype(np.int8),
            "offboard_actuator": dataset.data["actuator"].astype(bool).astype(np.int8),
        }
    )
    df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    df.sort_values("timestamp_us", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_vehicle_control_mode_dataframe(dataset):
    required_fields = ["timestamp", "flag_control_offboard_enabled"]
    require_fields(dataset, required_fields)

    df = pd.DataFrame(
        {
            "timestamp_us": dataset.data["timestamp"].astype(np.int64),
            "vehicle_offboard_enabled": dataset.data["flag_control_offboard_enabled"].astype(bool).astype(np.int8),
        }
    )
    df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    df.sort_values("timestamp_us", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_vehicle_local_position_dataframe(dataset):
    required_fields = ["timestamp", "x", "y", "z", "vx", "vy", "vz"]
    require_fields(dataset, required_fields)

    df = pd.DataFrame(
        {
            "timestamp_us": dataset.data["timestamp"].astype(np.int64),
            "px4_x": dataset.data["x"].astype(float),
            "px4_y": dataset.data["y"].astype(float),
            "px4_z": dataset.data["z"].astype(float),
            "px4_vx": dataset.data["vx"].astype(float),
            "px4_vy": dataset.data["vy"].astype(float),
            "px4_vz": dataset.data["vz"].astype(float),
        }
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["px4_x", "px4_y", "px4_z", "px4_vx", "px4_vy", "px4_vz"], inplace=True)
    df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    df.sort_values("timestamp_us", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_trajectory_setpoint_dataframe(dataset):
    required_fields = ["timestamp"]
    for axis in range(3):
        required_fields.append(f"position[{axis}]")
        required_fields.append(f"velocity[{axis}]")
    require_fields(dataset, required_fields)

    df = pd.DataFrame(
        {
            "timestamp_us": dataset.data["timestamp"].astype(np.int64),
            "px4_sp_x": dataset.data["position[0]"].astype(float),
            "px4_sp_y": dataset.data["position[1]"].astype(float),
            "px4_sp_z": dataset.data["position[2]"].astype(float),
            "px4_sp_vx": dataset.data["velocity[0]"].astype(float),
            "px4_sp_vy": dataset.data["velocity[1]"].astype(float),
            "px4_sp_vz": dataset.data["velocity[2]"].astype(float),
        }
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    df.sort_values("timestamp_us", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def merge_nearest_required_timeline(base_df, source_df, source_name, tolerance_us):
    merged_df = pd.merge_asof(
        base_df,
        source_df,
        on="timestamp_us",
        direction="nearest",
        tolerance=tolerance_us,
    )

    source_columns = [column for column in source_df.columns if column != "timestamp_us"]
    if merged_df[source_columns].isna().all().all():
        raise RuntimeError(f"No {source_name} samples could be aligned with vehicle_local_position")

    return merged_df


def merge_state_timeline(base_df, source_df, source_name):
    merged_df = pd.merge_asof(
        base_df,
        source_df,
        on="timestamp_us",
        direction="backward",
    )

    source_columns = [column for column in source_df.columns if column != "timestamp_us"]
    if merged_df[source_columns].isna().all().all():
        raise RuntimeError(f"No {source_name} samples could be aligned with offboard_control_mode")

    return merged_df


def build_flight_log_dataframe(ulog_path):
    try:
        ulog = ULog(
            str(ulog_path),
            [
                "vehicle_local_position",
                "trajectory_setpoint",
                "vehicle_control_mode",
                "offboard_control_mode",
            ],
        )
        local_position_df = build_vehicle_local_position_dataframe(get_single_topic(ulog, "vehicle_local_position"))
        trajectory_setpoint_dataset = get_optional_single_topic(ulog, "trajectory_setpoint")
        if trajectory_setpoint_dataset is None:
            trajectory_setpoint_df = None
        else:
            trajectory_setpoint_df = build_trajectory_setpoint_dataframe(trajectory_setpoint_dataset)
        vehicle_control_mode_df = build_vehicle_control_mode_dataframe(get_single_topic(ulog, "vehicle_control_mode"))
        offboard_control_mode_df = build_offboard_control_mode_dataframe(get_single_topic(ulog, "offboard_control_mode"))
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc

    if local_position_df.empty:
        raise RuntimeError("No valid vehicle_local_position samples were found in the ULog")
    if trajectory_setpoint_df is not None and trajectory_setpoint_df.empty:
        raise RuntimeError("No valid trajectory_setpoint samples were found in the ULog")
    if vehicle_control_mode_df.empty:
        raise RuntimeError("No valid vehicle_control_mode samples were found in the ULog")
    if offboard_control_mode_df.empty:
        raise RuntimeError("No valid offboard_control_mode samples were found in the ULog")

    merged_df = merge_nearest_required_timeline(offboard_control_mode_df, local_position_df, "vehicle_local_position", 100000)
    merged_df = merge_state_timeline(merged_df, vehicle_control_mode_df, "vehicle_control_mode")
    if trajectory_setpoint_df is None:
        merged_df["px4_sp_x"] = np.nan
        merged_df["px4_sp_y"] = np.nan
        merged_df["px4_sp_z"] = np.nan
        merged_df["px4_sp_vx"] = np.nan
        merged_df["px4_sp_vy"] = np.nan
        merged_df["px4_sp_vz"] = np.nan
    else:
        merged_df = pd.merge_asof(
            merged_df,
            trajectory_setpoint_df,
            on="timestamp_us",
            direction="nearest",
            tolerance=100000,
        )

    merged_df.dropna(subset=["vehicle_offboard_enabled"], inplace=True)
    merged_df["vehicle_offboard_enabled"] = merged_df["vehicle_offboard_enabled"].astype(np.int8)
    offboard_df = merged_df[merged_df["vehicle_offboard_enabled"] == 1].copy()

    if offboard_df.empty:
        raise RuntimeError("No offboard samples were found in the aligned PX4 flight log")

    offboard_df.dropna(
        subset=[
            "offboard_position",
            "offboard_velocity",
            "offboard_actuator",
            "px4_x",
            "px4_y",
            "px4_z",
            "px4_vx",
            "px4_vy",
            "px4_vz",
        ],
        inplace=True,
    )

    if offboard_df.empty:
        raise RuntimeError("No offboard samples remain after requiring offboard_control_mode and vehicle_local_position")

    offboard_df["px4_position_control_active"] = (
        (offboard_df["offboard_position"].astype(bool) | offboard_df["offboard_velocity"].astype(bool))
        & ~offboard_df["offboard_actuator"].astype(bool)
    ).astype(np.int8)
    offboard_df["nmpc_direct_actuator_active"] = offboard_df["offboard_actuator"].astype(bool).astype(np.int8)

    output_columns = [
        "timestamp_us",
        "vehicle_offboard_enabled",
        "px4_position_control_active",
        "nmpc_direct_actuator_active",
        "offboard_position",
        "offboard_velocity",
        "offboard_acceleration",
        "offboard_attitude",
        "offboard_body_rate",
        "offboard_actuator",
        "px4_x",
        "px4_y",
        "px4_z",
        "px4_vx",
        "px4_vy",
        "px4_vz",
        "px4_sp_x",
        "px4_sp_y",
        "px4_sp_z",
        "px4_sp_vx",
        "px4_sp_vy",
        "px4_sp_vz",
    ]
    offboard_df = offboard_df[output_columns]
    return offboard_df


def main():
    parser = argparse.ArgumentParser(description="Extract PX4 offboard trajectory and NMPC/PX4 control-mode logs from a ULog")
    parser.add_argument("ulog_path", help="Input PX4 ULog file")
    parser.add_argument("output_csv_path", help="Output CSV path")
    args = parser.parse_args()

    try:
        offboard_df = build_flight_log_dataframe(args.ulog_path)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    offboard_df.to_csv(args.output_csv_path, index=False)
    print(f"Saved {args.output_csv_path}")


if __name__ == "__main__":
    main()
