import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dominant_frequency(x: np.ndarray, sampling_rate: float = 1.0) -> float:
    """Return dominant non-DC frequency using FFT."""
    n = x.size
    if n < 2:
        return float("nan")

    x_centered = x - np.mean(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    spectrum = np.abs(np.fft.rfft(x_centered))

    if spectrum.size <= 1:
        return float("nan")

    # Ignore the DC term at index 0.
    peak_idx = 1 + np.argmax(spectrum[1:])
    return float(freqs[peak_idx])


def find_peak_indices(x: np.ndarray) -> np.ndarray:
    """Simple local-maximum peak detector without external dependencies."""
    if x.size < 3:
        return np.array([], dtype=int)

    left = x[1:-1] > x[:-2]
    right = x[1:-1] >= x[2:]
    peaks = np.where(left & right)[0] + 1
    return peaks


def peak_to_peak_interval(x: np.ndarray) -> float:
    """Mean sample interval between consecutive peaks."""
    peaks = find_peak_indices(x)
    if peaks.size < 2:
        return float("nan")
    intervals = np.diff(peaks)
    return float(np.mean(intervals))


def peak_to_peak_value(x: np.ndarray) -> float:
    """Signal amplitude peak-to-peak value: max - min."""
    if x.size == 0:
        return float("nan")
    return float(np.max(x) - np.min(x))


def mean_value(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def skewness_value(x: np.ndarray) -> float:
    """Sample skewness computed from central moments."""
    n = x.size
    if n < 3:
        return float("nan")

    mu = np.mean(x)
    centered = x - mu
    m2 = np.mean(centered ** 2)
    if m2 == 0:
        return 0.0

    m3 = np.mean(centered ** 3)
    return float(m3 / (m2 ** 1.5))


def compute_features_for_series(x: np.ndarray, sampling_rate: float) -> dict:
    return {
        "dominant_frequency": dominant_frequency(x, sampling_rate=sampling_rate),
        "peak_to_peak_interval": peak_to_peak_interval(x),
        "peak_to_peak_value": peak_to_peak_value(x),
        "mean_value": mean_value(x),
        "skewness": skewness_value(x),
    }


def time_span_from_time_axis(t: np.ndarray) -> float:
    """Return span as max(t) - min(t) for valid time values."""
    if t.size < 2:
        return float("nan")
    return float(np.max(t) - np.min(t))


def is_index_like_column(col_name: str) -> bool:
    """Return True for unnamed/index columns created by Excel exports."""
    name = str(col_name).strip().lower()
    return name.startswith("unnamed") or name in {"index", "id"}


def plot_random_10s_by_class(class_records: dict, chunk_sec: float = 10.0) -> pd.DataFrame:
    """Plot one random fixed-duration chunk per class and return one column per chunk."""
    class_names = sorted(class_records.keys())
    if not class_names:
        return pd.DataFrame()

    plot_chunks = {}

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)

    for i in range(5):
        ax = axes[i]
        if i >= len(class_names):
            ax.axis("off")
            continue

        class_name = class_names[i]
        series_list = class_records[class_name]

        eligible = [rec for rec in series_list if rec[0].size >= 2 and (rec[0][-1] - rec[0][0]) >= chunk_sec]
        if eligible:
            t, x = random.choice(eligible)
            t_start_min = t[0]
            t_start_max = t[-1] - chunk_sec
            t_start = random.uniform(t_start_min, t_start_max)
            start_idx = int(np.searchsorted(t, t_start, side="left"))
            end_idx = int(np.searchsorted(t, t_start + chunk_sec, side="right"))
            chunk_t = t[start_idx:end_idx]
            chunk_x = x[start_idx:end_idx]

            if chunk_t.size < 2:
                chunk_t = t
                chunk_x = x
        else:
            t, x = max(series_list, key=lambda rec: rec[0].size)
            chunk_t = t
            chunk_x = x

        t_axis = chunk_t - chunk_t[0]
        ax.plot(t_axis, chunk_x, linewidth=1.2)
        plot_chunks[f"{class_name}_chunk"] = pd.Series(chunk_x, dtype=float)
        ax.set_title(f"Class: {class_name}")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Random 10.0s chunk per class", fontsize=13)
    plt.show()
    return pd.DataFrame(plot_chunks)


def main(input_path: Path, output_path: Path, plot_output_path: Path, sampling_rate: float) -> None:
    class_signals = {}
    class_records = {}
    class_time_spans = {}

    # Read and process every worksheet tab.
    workbook = pd.read_excel(input_path, sheet_name=None)
    for sheet_name, df in workbook.items():
        if df.shape[1] == 0:
            continue

        # Use the first column as time if it can be interpreted numerically.
        time_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")

        for i, col in enumerate(df.columns[1:], start=2):
            if is_index_like_column(col):
                continue

            signal = pd.to_numeric(df[col], errors="coerce")
            valid = time_col.notna() & signal.notna()
            t = time_col[valid].to_numpy(dtype=float)
            x = signal[valid].to_numpy(dtype=float)

            if x.size == 0:
                continue

            time_span = time_span_from_time_axis(t)
            class_name = str(col) if str(col).strip() else f"class_{i}"

            class_signals.setdefault(class_name, []).append(x)
            class_records.setdefault(class_name, []).append((t, x))
            class_time_spans[class_name] = class_time_spans.get(class_name, 0.0)
            if np.isfinite(time_span):
                class_time_spans[class_name] += float(time_span)

    rows = []
    for class_name in sorted(class_signals.keys()):
        x_all = np.concatenate(class_signals[class_name])
        features = compute_features_for_series(x_all, sampling_rate=sampling_rate)
        rows.append(
            {
                "class": class_name,
                "time_span": class_time_spans.get(class_name, float("nan")),
                **features,
            }
        )

    plot_df = plot_random_10s_by_class(class_records, chunk_sec=10.0)

    if not rows:
        raise ValueError("No valid numeric time-series columns found in the input file.")

    out_df = pd.DataFrame(rows)
    out_df.to_excel(output_path, index=False)
    plot_df.to_excel(plot_output_path, index=False)

    print(f"Processed {len(rows)} pooled classes across {len(workbook)} tab(s).")
    print("Time span by class:")
    for row in rows:
        span = row["time_span"]
        span_text = f"{span:.6f}" if np.isfinite(span) else "nan"
        print(f"  {row['class']}: {span_text}")
    print(f"Saved features to: {output_path}")
    print(f"Saved plotted chunks to: {plot_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract time-series features from each column in an Excel file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Thedata.xlsx"),
        help="Path to input Excel file (default: Thedata.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Thedata_features.xlsx"),
        help="Path to output Excel file (default: Thedata_features.xlsx)",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("Thedata_plot_chunks.xlsx"),
        help="Path to plotted chunk output Excel file (default: Thedata_plot_chunks.xlsx)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1.0,
        help="Sampling rate used for dominant frequency calculation (default: 1.0)",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.plot_output, args.sampling_rate)
