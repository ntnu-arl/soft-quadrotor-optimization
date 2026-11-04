import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Combine NMPC solver CSV logs into one timestamp-ordered CSV")
    parser.add_argument("output_csv_path", help="Output combined CSV path")
    parser.add_argument("input_csv_paths", nargs="+", help="Input NMPC solver CSV paths")
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input_csv_paths]
    for input_path in input_paths:
        if not input_path.is_file():
            raise SystemExit(f"Input solver CSV does not exist: {input_path}")

    frames = []
    expected_columns = None
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        if df.empty:
            raise SystemExit(f"Input solver CSV is empty: {input_path}")
        if expected_columns is None:
            expected_columns = list(df.columns)
        elif list(df.columns) != expected_columns:
            raise SystemExit(f"Input solver CSV header differs from the first file: {input_path}")
        df["source_solver_csv"] = input_path.name
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    if "sample_timestamp_us" not in combined_df.columns:
        raise SystemExit("Combined solver CSV is missing sample_timestamp_us")
    combined_df.drop_duplicates(subset=["sample_timestamp_us", "seq"], keep="last", inplace=True)
    combined_df.sort_values(["sample_timestamp_us", "seq"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv(args.output_csv_path, index=False)
    print(f"Saved {args.output_csv_path}")


if __name__ == "__main__":
    main()
