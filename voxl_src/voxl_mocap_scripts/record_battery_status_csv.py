#!/usr/bin/env python

import argparse
import csv
import re
import subprocess
import sys
import time

CSV_HEADER = [
    "timestamp_us",
    "voltage_v",
    "current_a",
    "cell_count",
    "per_cell_voltage_v",
    "percent_estimate",
]

CURVE = [
    (4.350, 100.0),
    (4.280, 95.0),
    (4.220, 90.0),
    (4.150, 85.0),
    (4.050, 83.0),
    (4.000, 75.0),
    (3.950, 65.0),
    (3.900, 55.0),
    (3.850, 45.0),
    (3.800, 35.0),
    (3.750, 25.0),
    (3.700, 18.0),
    (3.650, 12.0),
    (3.600, 8.0),
    (3.550, 4.0),
    (3.500, 0.0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Poll px4-listener battery_status and record it to a strict CSV format.")
    parser.add_argument("--sample-interval-s", required=True, type=float)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def open_csv_handle(path):
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


def voltage_to_percent(per_cell_voltage):
    if per_cell_voltage >= CURVE[0][0]:
        return CURVE[0][1]
    if per_cell_voltage <= CURVE[-1][0]:
        return CURVE[-1][1]
    for (v_hi, p_hi), (v_lo, p_lo) in zip(CURVE, CURVE[1:]):
        if v_hi >= per_cell_voltage >= v_lo:
            alpha = (per_cell_voltage - v_lo) / (v_hi - v_lo)
            return p_lo + alpha * (p_hi - p_lo)
    raise RuntimeError("Failed to interpolate LiHV battery percentage for per_cell_voltage=%s" % per_cell_voltage)


def query_battery_status():
    result = subprocess.run(
        ["px4-listener", "battery_status", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    text = result.stdout

    voltage_match = re.search(r"voltage_v:\s*([0-9.]+)", text)
    current_match = re.search(r"current_a:\s*([0-9.]+)", text)
    cell_count_match = re.search(r"cell_count:\s*(\d+)", text)

    if voltage_match is None:
        raise RuntimeError("Could not parse voltage_v from PX4 battery_status")
    if current_match is None:
        raise RuntimeError("Could not parse current_a from PX4 battery_status")
    if cell_count_match is None:
        raise RuntimeError("Could not parse cell_count from PX4 battery_status")

    voltage_v = float(voltage_match.group(1))
    current_a = float(current_match.group(1))
    cell_count = int(cell_count_match.group(1))

    if cell_count != 4:
        raise RuntimeError("Expected 4S battery, got %dS" % cell_count)

    per_cell_voltage_v = voltage_v / cell_count
    percent_estimate = voltage_to_percent(per_cell_voltage_v)

    return voltage_v, current_a, cell_count, per_cell_voltage_v, percent_estimate


def main():
    args = parse_args()

    output_handle = open_csv_handle(args.output_csv)
    writer = csv.writer(output_handle)
    writer.writerow(CSV_HEADER)
    output_handle.flush()

    try:
        while True:
            sample_start_s = time.monotonic()
            timestamp_us = int(time.time() * 1e6)
            voltage_v, current_a, cell_count, per_cell_voltage_v, percent_estimate = query_battery_status()
            writer.writerow([timestamp_us, voltage_v, current_a, cell_count, per_cell_voltage_v, percent_estimate])
            output_handle.flush()

            elapsed_s = time.monotonic() - sample_start_s
            remaining_s = args.sample_interval_s - elapsed_s
            if remaining_s > 0:
                time.sleep(remaining_s)
    finally:
        output_handle.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("record_battery_status_csv failed: %s\n" % exc)
        raise
