"""
Train a MultiRocket classifier on random fixed windows from an .xlsx workbook.

The script:
1. Reads all tabs from Texture_Test.xlsx by default, or another .xlsx workbook
    supplied with --input.
2. Treats each non-index column (except the first time column) as one class signal,
   matching the reading logic used in extract_timeseries_features.py.
3. Randomly extracts overlapped fixed-length windows (default: 10,000
    consecutive points, approximately 10 seconds at 1 kHz).
4. Splits data into train/test with 9/10 for training and 1/10 for testing.
5. Fits RocketClassifier with MultiRocket transform and reports metrics.
6. Saves a normalized confusion matrix figure.
7. Saves a 5-subplot figure with random test chunks and class text.

Install:
    pip install numpy pandas matplotlib scikit-learn sktime numba openpyxl

Run:
    python train_feature_multirocket.py [--input OTHER_FILE.xlsx]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import RocketClassifier


def is_index_like_column(col_name: str) -> bool:
    """Return True for unnamed/index columns often created by Excel exports."""
    name = str(col_name).strip().lower()
    return name.startswith("unnamed") or name in {"index", "id"}


def read_workbook_records(input_path: Path) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """
    Read an .xlsx workbook using the feature-extraction convention.

    Returns a list of tuples:
        (sheet_name, class_name, time_array, signal_array)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Input must be an .xlsx workbook: {input_path}")

    workbook = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
    sheet_names = list(workbook.keys())
    print(f"Detected {len(sheet_names)} tab(s): {sheet_names}")

    records: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for sheet_name, df in workbook.items():
        if df.shape[1] < 2:
            continue

        time_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")

        for i, col in enumerate(df.columns[1:], start=2):
            if is_index_like_column(col):
                continue

            signal = pd.to_numeric(df[col], errors="coerce")
            valid = time_col.notna() & signal.notna()
            t = time_col[valid].to_numpy(dtype=float)
            x = signal[valid].to_numpy(dtype=np.float32)

            if x.size < 2:
                continue

            class_name = str(col).strip() if str(col).strip() else f"class_{i}"
            records.append((sheet_name, class_name, t, x))

    if not records:
        raise ValueError("No valid numeric class columns were found in the workbook.")

    print(f"Loaded {len(records)} usable class-series records from workbook.")
    return records


def estimate_window_seconds(records: list[tuple[str, str, np.ndarray, np.ndarray]], window_points: int) -> float:
    """Estimate window duration from median positive sample interval."""
    dts: list[float] = []
    for _, _, t, _ in records:
        if t.size < 2:
            continue
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            dts.append(float(np.median(dt)))

    if not dts:
        return float("nan")

    median_dt = float(np.median(np.asarray(dts, dtype=float)))
    return window_points * median_dt


def sample_random_chunks(
    records: list[tuple[str, str, np.ndarray, np.ndarray]],
    window_points: int,
    stride_points: int,
    max_chunks_per_series: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sample overlapped fixed-length windows from each class-series."""
    if window_points < 2:
        raise ValueError("window_points must be at least 2.")
    if stride_points < 1:
        raise ValueError("stride_points must be at least 1.")
    if max_chunks_per_series < 1:
        raise ValueError("max_chunks_per_series must be at least 1.")

    rng = np.random.default_rng(random_state)
    chunk_list: list[np.ndarray] = []
    label_list: list[str] = []

    for sheet_name, class_name, _t, x in records:
        n = x.size
        if n < window_points:
            continue

        possible_starts = n - window_points + 1
        candidate_starts = np.arange(0, possible_starts, stride_points, dtype=int)
        if candidate_starts.size == 0 or candidate_starts[-1] != possible_starts - 1:
            candidate_starts = np.append(candidate_starts, possible_starts - 1)

        n_chunks = min(max_chunks_per_series, candidate_starts.size)
        if n_chunks == candidate_starts.size:
            starts = candidate_starts
        else:
            starts = rng.choice(candidate_starts, size=n_chunks, replace=False)

        for start in starts:
            end = int(start) + window_points
            chunk = x[int(start):end]
            if chunk.size != window_points:
                continue

            chunk_list.append(chunk[np.newaxis, :])
            label_list.append(class_name)

    if not chunk_list:
        raise ValueError(
            "No chunks sampled. Reduce --window-points or verify input has longer series."
        )

    X = np.asarray(chunk_list, dtype=np.float32)
    y = np.asarray(label_list)

    # Remove classes that are too small for stratified splitting.
    classes, counts = np.unique(y, return_counts=True)
    small_classes = classes[counts < 2]
    if small_classes.size > 0:
        keep_mask = ~np.isin(y, small_classes)
        X = X[keep_mask]
        y = y[keep_mask]
        print(
            "Dropped classes with <2 chunks for stratified split: "
            + ", ".join(map(str, small_classes))
        )

    if X.shape[0] < 2 or np.unique(y).size < 2:
        raise ValueError(
            "Not enough sampled data/classes after filtering. "
            "Increase chunks or reduce window size."
        )

    print(f"Sampled chunks: {X.shape[0]} | Chunk shape: {X.shape[1:]}")
    class_counts = pd.Series(y).value_counts().sort_index()
    print("Chunks per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    return X, y


def sample_train_test_chunks_without_overlap(
    records: list[tuple[str, str, np.ndarray, np.ndarray]],
    window_points: int,
    stride_points: int,
    max_chunks_per_series: int,
    test_size: float,
    random_state: int,
    exclude_initial_seconds: float = 10.0,
    exclude_final_seconds: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample disjoint train/test windows after trimming series endpoints."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if exclude_initial_seconds < 0:
        raise ValueError("exclude_initial_seconds must be non-negative.")
    if exclude_final_seconds < 0:
        raise ValueError("exclude_final_seconds must be non-negative.")

    rng = np.random.default_rng(random_state)
    train_chunks: list[np.ndarray] = []
    test_chunks: list[np.ndarray] = []
    train_labels: list[str] = []
    test_labels: list[str] = []

    for _sheet_name, class_name, t, x in records:
        if exclude_initial_seconds > 0 and t.size > 0:
            keep_from = t[0] + exclude_initial_seconds
            keep_mask = t >= keep_from
            t = t[keep_mask]
            x = x[keep_mask]

        if exclude_final_seconds > 0 and t.size > 0:
            keep_until = t[-1] - exclude_final_seconds
            keep_mask = t <= keep_until
            t = t[keep_mask]
            x = x[keep_mask]

        if x.size < window_points:
            continue

        possible_starts = x.size - window_points + 1
        candidate_starts = np.arange(0, possible_starts, stride_points, dtype=int)
        if candidate_starts.size == 0 or candidate_starts[-1] != possible_starts - 1:
            candidate_starts = np.append(candidate_starts, possible_starts - 1)

        n_chunks = min(max_chunks_per_series, candidate_starts.size)
        if n_chunks < 2:
            continue
        if n_chunks < candidate_starts.size:
            selected = np.sort(
                rng.choice(candidate_starts, size=n_chunks, replace=False)
            )
        else:
            selected = candidate_starts

        n_test = max(1, int(np.ceil(n_chunks * test_size)))
        test_starts = selected[-n_test:]
        first_test_start = int(test_starts[0])
        train_starts = selected[selected + window_points <= first_test_start]
        if train_starts.size == 0:
            continue

        for start in train_starts:
            train_chunks.append(x[int(start):int(start) + window_points][np.newaxis, :])
            train_labels.append(class_name)
        for start in test_starts:
            test_chunks.append(x[int(start):int(start) + window_points][np.newaxis, :])
            test_labels.append(class_name)

    if not train_chunks or not test_chunks:
        raise ValueError(
            "Could not create non-overlapping train/test windows. "
            "Reduce window size or increase source-series length."
        )

    X_train = np.asarray(train_chunks, dtype=np.float32)
    X_test = np.asarray(test_chunks, dtype=np.float32)
    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)
    if train_classes.size < 2 or test_classes.size < 2:
        raise ValueError("Non-overlapping split must contain at least two classes.")

    print(
        f"Sampled non-overlapping chunks: training={len(y_train)}, "
        f"test={len(y_test)} | Chunk shape: {X_train.shape[1:]}"
    )
    return X_train, X_test, y_train, y_test


def plot_random_test_chunks(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    labels_sorted: np.ndarray,
    output_path: Path,
    random_state: int,
    show_plots: bool,
) -> None:
    """Plot 9 random test chunks in a 3x3 grid with class text."""
    rng = np.random.default_rng(random_state)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 10),
        sharex=True,
        constrained_layout=True,
    )
    axes_flat = axes.flat

    for i, axis in enumerate(axes_flat):
        if i >= len(labels_sorted):
            axis.axis("off")
            continue

        class_name = labels_sorted[i]
        candidate_idx = np.flatnonzero(y_test == class_name)
        if candidate_idx.size == 0:
            axis.text(0.02, 0.5, f"Class: {class_name} | No test chunk", transform=axis.transAxes)
            axis.set_ylabel(str(class_name))
            axis.grid(alpha=0.25)
            continue

        idx = int(rng.choice(candidate_idx))
        signal = X_test[idx, 0]
        pred_name = y_pred[idx]

        axis.plot(signal, linewidth=1.0)
        axis.set_ylabel(str(class_name))
        axis.grid(alpha=0.25)
        axis.text(
            0.01,
            0.95,
            f"True: {class_name} | Pred: {pred_name}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[-1, 0].set_xlabel("Point index in chunk")
    axes[-1, 1].set_xlabel("Point index in chunk")
    axes[-1, 2].set_xlabel("Point index in chunk")
    fig.suptitle("Random test chunks with class text")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    num_kernels: int,
    random_state: int,
    n_jobs: int,
    output_dir: Path,
    show_plots: bool,
    split_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    input_stem: str = "input",
) -> None:
    """Train MultiRocket and save metrics/plots."""
    if split_data is None:
        n_classes = np.unique(y).size
        requested_test_count = int(np.ceil(len(y) * test_size))
        effective_test_count = max(requested_test_count, n_classes)
        if effective_test_count >= len(y):
            raise ValueError(
                "Test set would consume all samples. Increase sampled chunks or reduce classes."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=effective_test_count,
            stratify=y,
            random_state=random_state,
        )
    else:
        X_train, X_test, y_train, y_test = split_data

    print(f"Before training: training chunks = {len(y_train)}, test chunks = {len(y_test)}")

    classifier = RocketClassifier(
        rocket_transform="multirocket",
        num_kernels=num_kernels,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    t0 = perf_counter()
    classifier.fit(X_train, y_train)
    fit_seconds = perf_counter() - t0

    t1 = perf_counter()
    y_pred = classifier.predict(X_test)
    predict_seconds = perf_counter() - t1

    accuracy = accuracy_score(y_test, y_pred)
    labels_sorted = np.unique(y)
    dataset_name = input_stem.removesuffix("_Test")

    print(f"\nTraining samples: {len(y_train)}")
    print(f"Test samples:     {len(y_test)}")
    print(f"Input shape:      {X.shape}")
    print(f"Fit time (s):     {fit_seconds:.3f}")
    print(f"Predict time (s): {predict_seconds:.3f}")
    print(f"Test accuracy:    {accuracy:.4f}\n")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=labels_sorted,
            digits=4,
            zero_division=0,
        )
    )

    fig, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels_sorted,
        display_labels=labels_sorted,
        normalize="true",
        values_format=".2f",
        xticks_rotation=30,
        ax=axis,
    )
    axis.set_title(
        f"{dataset_name} MultiRocket confusion matrix, accuracy={accuracy:.3f}"
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = output_dir / (
        f"{dataset_name.lower()}_multirocket_{input_stem}_confusion_matrix.png"
    )
    fig.savefig(confusion_path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    print(f"Saved confusion matrix to: {confusion_path.resolve()}")

    test_chunks_path = output_dir / (
        f"{dataset_name.lower()}_multirocket_{input_stem}_random_test_chunks_by_class.png"
    )
    plot_random_test_chunks(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        labels_sorted=labels_sorted,
        output_path=test_chunks_path,
        random_state=random_state,
        show_plots=show_plots,
    )
    print(f"Saved random test chunks to: {test_chunks_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MultiRocket on random windows from an .xlsx workbook "
            "(default: Texture_Test.xlsx)"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Texture_Test.xlsx"),
        help="Input .xlsx workbook (default: Texture_Test.xlsx).",
    )
    parser.add_argument("--window-points", type=int, default=5_000)
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
        help="Allow overlapping windows; --no-overlap uses window length as stride.",
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
        default=25.0,
        help="Ignore this many seconds at the end of every source series (default: 25).",
    )
    parser.add_argument("--max-chunks-per-series", type=int, default=1_000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--num-kernels", type=int, default=5_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("texture_multirocket_output"),
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
    print(f"Excluded initial data: {args.exclude_initial_seconds:g} seconds per series.")
    print(f"Excluded final data: {args.exclude_final_seconds:g} seconds per series.")

    records = read_workbook_records(args.input)
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
