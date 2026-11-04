import argparse

import numpy as np
import pandas as pd

try:
    from pyulog import ULog
except ModuleNotFoundError as exc:
    raise SystemExit("pyulog is required to extract PX4 motor logs") from exc


NUM_MOTORS = 4
NMPC_MIN_RPM = 4980   # must match MulticopterNmpcControl.cpp
NMPC_MAX_RPM = 24000  # must match MulticopterNmpcControl.cpp
CONNECTED_ESC_MAX = 8
ACTUATOR_FUNCTION_MOTOR1 = 101
MEASURED_MERGE_TOLERANCE_US = 100000
ACTUATOR_MOTORS_MERGE_TOLERANCE_US = 100000
ACTUATOR_MOTORS_NONREVERSIBLE_MASK = (1 << NUM_MOTORS) - 1


def get_single_topic(ulog, topic_name):
    matches = [dataset for dataset in ulog.data_list if dataset.name == topic_name]

    if not matches:
        raise KeyError(f"Required ULog topic '{topic_name}' not found")

    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one '{topic_name}' dataset, found {len(matches)}")

    return matches[0]


def find_single_topic(ulog, topic_name):
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


def get_required_initial_parameter(ulog, parameter_name):
    if parameter_name not in ulog.initial_parameters:
        raise KeyError(f"Required PX4 initial parameter '{parameter_name}' not found in ULog")

    return float(ulog.initial_parameters[parameter_name])


def build_esc_to_motor_map(ulog):
    esc_to_motor = []
    for esc_ch in range(NUM_MOTORS):
        param_name = f"VOXL_ESC_FUNC{esc_ch + 1}"
        if param_name not in ulog.initial_parameters:
            raise KeyError(f"Required PX4 initial parameter '{param_name}' not found in ULog")
        motor_index = int(ulog.initial_parameters[param_name]) - ACTUATOR_FUNCTION_MOTOR1
        if motor_index < 0 or motor_index >= NUM_MOTORS:
            raise RuntimeError(f"{param_name}={int(ulog.initial_parameters[param_name])} is not a valid motor function (expected {ACTUATOR_FUNCTION_MOTOR1}..{ACTUATOR_FUNCTION_MOTOR1 + NUM_MOTORS - 1})")
        esc_to_motor.append(motor_index)
    return tuple(esc_to_motor)


def build_requested_dataframe(dataset, esc_to_motor_map):
    required_fields = ["timestamp"] + [f"output[{motor_index}]" for motor_index in range(NUM_MOTORS)]
    require_fields(dataset, required_fields)

    requested_columns = {
        "timestamp_us": dataset.data["timestamp"].astype(np.int64),
    }

    for esc_ch in range(NUM_MOTORS):
        motor_index = esc_to_motor_map[esc_ch]
        requested_columns[f"requested_rpm{motor_index}"] = dataset.data[f"output[{esc_ch}]"].astype(float)

    requested_df = pd.DataFrame(requested_columns)
    requested_columns_order = ["timestamp_us"] + [f"requested_rpm{motor_index}" for motor_index in range(NUM_MOTORS)]
    requested_df = requested_df[requested_columns_order]
    requested_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    requested_df.dropna(subset=requested_columns_order[1:], inplace=True)
    requested_df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    requested_df.sort_values("timestamp_us", inplace=True)
    requested_df.reset_index(drop=True, inplace=True)
    return requested_df


def build_measured_dataframe(dataset):
    required_fields = ["timestamp", "esc_count"]

    for esc_index in range(CONNECTED_ESC_MAX):
        required_fields.extend(
            [
                f"esc[{esc_index}].esc_rpm",
                f"esc[{esc_index}].actuator_function",
                f"esc[{esc_index}].esc_address",
            ]
        )

    require_fields(dataset, required_fields)

    num_rows = len(dataset.data["timestamp"])
    measured_columns = {
        "timestamp_us": dataset.data["timestamp"].astype(np.int64),
    }

    for motor_index in range(NUM_MOTORS):
        measured_columns[f"measured_rpm{motor_index}"] = np.full(num_rows, np.nan, dtype=float)

    esc_count = dataset.data["esc_count"]

    for esc_index in range(CONNECTED_ESC_MAX):
        esc_rpm = dataset.data[f"esc[{esc_index}].esc_rpm"].astype(float)
        actuator_function = dataset.data[f"esc[{esc_index}].actuator_function"]
        esc_address = dataset.data[f"esc[{esc_index}].esc_address"]
        esc_present = esc_count > esc_index

        for motor_index in range(NUM_MOTORS):
            function_match = actuator_function == (ACTUATOR_FUNCTION_MOTOR1 + motor_index)
            address_match = (actuator_function == 0) & (esc_address == (motor_index + 1))
            selected_rows = esc_present & (function_match | address_match)
            measured_columns[f"measured_rpm{motor_index}"][selected_rows] = esc_rpm[selected_rows]

    measured_df = pd.DataFrame(measured_columns)
    measured_columns_order = [f"measured_rpm{motor_index}" for motor_index in range(NUM_MOTORS)]
    measured_df.dropna(subset=measured_columns_order, how="all", inplace=True)
    measured_df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    measured_df.sort_values("timestamp_us", inplace=True)
    measured_df.reset_index(drop=True, inplace=True)
    return measured_df


def build_actuator_motors_dataframe(dataset):
    required_fields = ["timestamp", "reversible_flags"] + [f"control[{motor_index}]" for motor_index in range(NUM_MOTORS)]
    require_fields(dataset, required_fields)

    reversible_flags = dataset.data["reversible_flags"]

    if np.any(reversible_flags & ACTUATOR_MOTORS_NONREVERSIBLE_MASK):
        raise RuntimeError("Reversible motor outputs are not supported by this extractor")

    actuator_columns = {
        "timestamp_us": dataset.data["timestamp"].astype(np.int64),
    }

    for motor_index in range(NUM_MOTORS):
        actuator_columns[f"actuator_motors_control{motor_index}"] = dataset.data[
            f"control[{motor_index}]"
        ].astype(float)

    actuator_df = pd.DataFrame(actuator_columns)
    actuator_columns_order = ["timestamp_us"] + [f"actuator_motors_control{motor_index}" for motor_index in range(NUM_MOTORS)]
    actuator_df = actuator_df[actuator_columns_order]
    actuator_df.drop_duplicates(subset="timestamp_us", keep="last", inplace=True)
    actuator_df.sort_values("timestamp_us", inplace=True)
    actuator_df.reset_index(drop=True, inplace=True)
    return actuator_df


def actuator_controls_to_output_rpm(actuator_controls, thrust_factor, rpm_min, rpm_max):
    values = actuator_controls.astype(float).to_numpy(copy=True)
    finite_mask = np.isfinite(values)

    if thrust_factor > 0.0:
        a = thrust_factor
        b = 1.0 - thrust_factor
        tmp1 = b / (2.0 * a)
        tmp2 = b * b / (4.0 * a * a)
        positive_mask = finite_mask & (values > 0.0)
        negative_mask = finite_mask & (values < 0.0)
        zero_mask = finite_mask & ~(positive_mask | negative_mask)

        values[positive_mask] = -tmp1 + np.sqrt(tmp2 + values[positive_mask] / a)
        values[negative_mask] = tmp1 - np.sqrt(tmp2 - values[negative_mask] / a)
        values[zero_mask] = 0.0

    negative_output_mask = finite_mask & (values < -np.finfo(np.float32).eps)
    valid_output_mask = finite_mask & ~negative_output_mask
    values[negative_output_mask] = np.nan
    values[valid_output_mask] = values[valid_output_mask] * 2.0 - 1.0

    return values * (rpm_max - rpm_min) / 2.0 + (rpm_max + rpm_min) / 2.0


def build_motor_log_dataframe(ulog_path):
    try:
        ulog = ULog(str(ulog_path), ["esc_status", "actuator_outputs_debug", "actuator_motors"])
        esc_to_motor_map = build_esc_to_motor_map(ulog)
        requested_dataset = find_single_topic(ulog, "actuator_outputs_debug")
        if requested_dataset is None:
            requested_df = None
        else:
            requested_df = build_requested_dataframe(requested_dataset, esc_to_motor_map)
        measured_df = build_measured_dataframe(get_single_topic(ulog, "esc_status"))
        actuator_motors_df = build_actuator_motors_dataframe(get_single_topic(ulog, "actuator_motors"))
        thrust_factor = get_required_initial_parameter(ulog, "THR_MDL_FAC")
        voxl_esc_rpm_min = get_required_initial_parameter(ulog, "VOXL_ESC_RPM_MIN")
        voxl_esc_rpm_max = get_required_initial_parameter(ulog, "VOXL_ESC_RPM_MAX")
        mc_nmpc_rpm_min = float(NMPC_MIN_RPM)
        mc_nmpc_rpm_max = float(NMPC_MAX_RPM)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc

    if requested_df is not None and requested_df.empty:
        raise RuntimeError("No valid requested motor RPM samples were found in actuator_outputs_debug")

    if measured_df.empty:
        raise RuntimeError("No valid measured motor RPM samples were found in esc_status")

    if actuator_motors_df.empty:
        raise RuntimeError("No valid actuator_motors samples were found in the ULog")

    if voxl_esc_rpm_max <= voxl_esc_rpm_min:
        raise RuntimeError("VOXL_ESC RPM limits are invalid in the ULog parameters")

    if mc_nmpc_rpm_max <= mc_nmpc_rpm_min:
        raise RuntimeError("MC_NMPC RPM limits are invalid in the ULog parameters")

    if requested_df is None:
        merged_df = pd.merge_asof(
            actuator_motors_df,
            measured_df,
            on="timestamp_us",
            direction="nearest",
            tolerance=MEASURED_MERGE_TOLERANCE_US,
        )
    else:
        merged_df = pd.merge_asof(
            requested_df,
            measured_df,
            on="timestamp_us",
            direction="nearest",
            tolerance=MEASURED_MERGE_TOLERANCE_US,
        )
        merged_df = pd.merge_asof(
            merged_df,
            actuator_motors_df,
            on="timestamp_us",
            direction="nearest",
            tolerance=ACTUATOR_MOTORS_MERGE_TOLERANCE_US,
        )

    if requested_df is None:
        print("actuator_outputs_debug not found; requested_rpm columns will be omitted")
    else:
        actuator_columns = [f"actuator_motors_control{motor_index}" for motor_index in range(NUM_MOTORS)]

        if merged_df[actuator_columns].isna().all().all():
            raise RuntimeError("No actuator_motors samples could be aligned with actuator_outputs_debug")

    measured_columns = [f"measured_rpm{motor_index}" for motor_index in range(NUM_MOTORS)]

    if merged_df[measured_columns].isna().all().all():
        raise RuntimeError("No esc_status samples could be aligned with the PX4 motor timeline")

    actuator_columns = [f"actuator_motors_control{motor_index}" for motor_index in range(NUM_MOTORS)]

    merged_df["param_thr_mdl_fac"] = thrust_factor
    merged_df["param_voxl_esc_rpm_min"] = voxl_esc_rpm_min
    merged_df["param_voxl_esc_rpm_max"] = voxl_esc_rpm_max
    merged_df["param_mc_nmpc_rpm_min"] = mc_nmpc_rpm_min
    merged_df["param_mc_nmpc_rpm_max"] = mc_nmpc_rpm_max

    for motor_index in range(NUM_MOTORS):
        control_column = f"actuator_motors_control{motor_index}"
        expected_column = f"expected_requested_rpm{motor_index}"
        error_column = f"requested_minus_expected_rpm{motor_index}"
        merged_df[expected_column] = actuator_controls_to_output_rpm(
            merged_df[control_column],
            thrust_factor,
            voxl_esc_rpm_min,
            voxl_esc_rpm_max,
        )
        if requested_df is not None:
            merged_df[error_column] = merged_df[f"requested_rpm{motor_index}"] - merged_df[expected_column]

    output_columns = ["timestamp_us"]
    if requested_df is not None:
        output_columns.extend([f"requested_rpm{motor_index}" for motor_index in range(NUM_MOTORS)])
    output_columns.extend([f"measured_rpm{motor_index}" for motor_index in range(NUM_MOTORS)])
    output_columns.extend([f"actuator_motors_control{motor_index}" for motor_index in range(NUM_MOTORS)])
    output_columns.extend([f"expected_requested_rpm{motor_index}" for motor_index in range(NUM_MOTORS)])
    if requested_df is not None:
        output_columns.extend([f"requested_minus_expected_rpm{motor_index}" for motor_index in range(NUM_MOTORS)])
    output_columns.extend(
        [
            "param_thr_mdl_fac",
            "param_voxl_esc_rpm_min",
            "param_voxl_esc_rpm_max",
            "param_mc_nmpc_rpm_min",
            "param_mc_nmpc_rpm_max",
        ]
    )
    merged_df = merged_df[output_columns]
    return merged_df


def main():
    parser = argparse.ArgumentParser(description="Extract PX4 requested and measured motor RPM from a ULog")
    parser.add_argument("ulog_path", help="Input PX4 ULog file")
    parser.add_argument("output_csv_path", help="Output CSV path")
    args = parser.parse_args()

    try:
        merged_df = build_motor_log_dataframe(args.ulog_path)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    merged_df.to_csv(args.output_csv_path, index=False)
    print(f"Saved {args.output_csv_path}")


if __name__ == "__main__":
    main()
