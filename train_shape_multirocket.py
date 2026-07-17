"""
Train the same MultiRocket classifier used by train_thedata_multirocket.py
on random 3-second windows from Shape_Test.xlsx.

The workbook is read like extract_timeseries_features.py: the first column
of every worksheet is treated as time, index-like columns are skipped, and
all remaining numeric columns are class signals. Both workbook tabs are used.
Train and test windows are always disjoint within each source series, even
when overlapping candidate windows are enabled.

四棱锥 (Square Pyramid) → SQP

圆锥 (Cone) → CON

菱形 (Rhombus) → RHO

六棱柱 (Hexagonal Prism) → HXP

球 (Sphere) → SPH

三棱柱 (Triangular Prism) → TRP

圆柱 (Cylinder) → CYL

长方体 (Rectangular Prism) → RPR

正方体 (Cube) → CUB

Run:
    python train_shape_multirocket.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_texture_multirocket import (
    estimate_window_seconds,
    is_index_like_column,
    sample_train_test_chunks_without_overlap,
    train_and_evaluate,
)


EXPECTED_TAB_COUNT = 2
EXPECTED_CLASS_COUNT = 9


def read_shape_test_records(
    input_path: Path,
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Read all Shape_Test.xlsx tabs using the feature-extraction convention."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    workbook = pd.read_excel(input_path, sheet_name=None)
    sheet_names = list(workbook.keys())
    print(f"Detected {len(sheet_names)} tab(s): {sheet_names}")
    if len(sheet_names) != EXPECTED_TAB_COUNT:
        print(
            f"Warning: Shape_Test.xlsx was expected to have "
            f"{EXPECTED_TAB_COUNT} tabs."
        )

    records: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    class_names: set[str] = set()

    print(f"Using all worksheets: {sheet_names}")

    for sheet_name, df in workbook.items():
        if df.shape[1] < 2:
            continue

        time_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        for column_index, column_name in enumerate(df.columns[1:], start=2):
            if is_index_like_column(column_name):
                continue

            signal = pd.to_numeric(df[column_name], errors="coerce")
            valid = time_col.notna() & signal.notna()
            time_values = time_col[valid].to_numpy(dtype=float)
            signal_values = signal[valid].to_numpy(dtype=np.float32)
            if signal_values.size < 2:
                continue

            class_name = (
                str(column_name).strip()
                if str(column_name).strip()
                else f"class_{column_index}"
            )
            records.append((sheet_name, class_name, time_values, signal_values))
            class_names.add(class_name)

    if not records:
        raise ValueError("No valid numeric shape columns were found in the workbook.")

    print(f"Loaded {len(records)} usable class-series records from workbook.")
    print(f"Detected classes ({len(class_names)}): {sorted(class_names)}")
    if len(class_names) != EXPECTED_CLASS_COUNT:
        print(
            f"Warning: expected {EXPECTED_CLASS_COUNT} shape classes, "
            f"but detected {len(class_names)}."
        )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the existing MultiRocket method on Shape_Test.xlsx"
    )
    parser.add_argument("--input", type=Path, default=Path("Shape_Test.xlsx"))
    parser.add_argument(
        "--window-points",
        type=int,
        default=10_000,
        help="Number of consecutive points per input window (default: 10,000).",
    )
    parser.add_argument(
        "--stride-points",
        type=int,
        default=500,
        help="Stride when overlap is enabled (default: 500).",
    )
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow overlapping candidate windows within each split; "
            "train/test windows remain disjoint. "
            "--no-overlap uses window length as stride."
        ),
    )
    parser.add_argument(
        "--exclude-initial-seconds",
        type=float,
        default=15.0,
        help="Ignore this many seconds at the start of every source series (default: 15).",
    )
    parser.add_argument(
        "--exclude-final-seconds",
        type=float,
        default=20.0,
        help="Ignore this many seconds at the end of every source series (default: 20).",
    )
    parser.add_argument("--max-chunks-per-series", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--num-kernels", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shape_multirocket_output"),
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Save plots only and do not display figure windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.stride_points < 1:
        raise ValueError("--stride-points must be at least 1.")
    if args.exclude_initial_seconds < 0:
        raise ValueError("--exclude-initial-seconds must be non-negative.")
    if args.exclude_final_seconds < 0:
        raise ValueError("--exclude-final-seconds must be non-negative.")

    effective_stride_points = args.stride_points if args.overlap else args.window_points
    print(
        f"Window overlap: {'enabled' if args.overlap else 'disabled'} "
        f"(stride={effective_stride_points} points)."
    )
    print("Train/test source windows: disjoint (data leakage prevented).")
    print(f"Excluded initial data: {args.exclude_initial_seconds:g} seconds per series.")
    print(f"Excluded final data: {args.exclude_final_seconds:g} seconds per series.")

    records = read_shape_test_records(args.input)
    approx_seconds = estimate_window_seconds(records, args.window_points)
    if np.isfinite(approx_seconds):
        print(
            f"Window size: {args.window_points} points "
            f"(~{approx_seconds:.3f} s based on median sample interval)."
        )
    else:
        print(f"Window size: {args.window_points} points (time step unavailable).")

    X_train, X_test, y_train, y_test = sample_train_test_chunks_without_overlap(
        records=records,
        window_points=args.window_points,
        stride_points=effective_stride_points,
        max_chunks_per_series=args.max_chunks_per_series,
        test_size=args.test_size,
        random_state=args.seed,
        exclude_initial_seconds=args.exclude_initial_seconds,
        exclude_final_seconds=args.exclude_final_seconds,
    )

    train_and_evaluate(
        X=np.concatenate((X_train, X_test)),
        y=np.concatenate((y_train, y_test)),
        test_size=args.test_size,
        num_kernels=args.num_kernels,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        show_plots=not args.no_show_plots,
        split_data=(X_train, X_test, y_train, y_test),
        input_stem=args.input.stem,
    )


if __name__ == "__main__":
    main()
