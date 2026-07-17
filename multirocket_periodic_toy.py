"""
Toy example: periodic 1-channel time-series classification with MultiRocket.

The script:
1. Generates five classes of noisy periodic signals.
2. Splits them into training and test sets with stratification.
3. Fits MultiRocket followed by RidgeClassifierCV through sktime's
   RocketClassifier wrapper.
4. Reports accuracy and a classification report.
5. Saves example-signal and confusion-matrix figures.

Install:
    pip install numpy matplotlib scikit-learn sktime numba

Run:
    python multirocket_periodic_toy.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import RocketClassifier


CLASS_NAMES = np.array(
    ["sine", "harmonic", "am_modulated", "chirped", "pulse_like"]
)


def generate_periodic_dataset(
    samples_per_class: int = 120,
    series_length: int = 500,
    sampling_rate: float = 100.0,
    noise_std: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate five classes of periodic, one-channel signals.

    Returns
    -------
    X : ndarray, shape (n_samples, 1, series_length)
        Time-series collection in sktime-compatible 3D format.
    y : ndarray, shape (n_samples,)
        Integer class labels.
    time : ndarray, shape (series_length,)
        Time vector in seconds.
    """
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2.")
    if series_length < 30:
        raise ValueError("series_length must be at least 30.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")

    rng = np.random.default_rng(random_state)
    time = np.arange(series_length, dtype=np.float64) / sampling_rate

    signals: list[np.ndarray] = []
    labels: list[int] = []

    for class_id in range(len(CLASS_NAMES)):
        for _ in range(samples_per_class):
            # Randomized amplitude, phase, frequency, offset, and slow drift
            # prevent the classification problem from being trivially fixed.
            amplitude = rng.uniform(0.8, 1.2)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            offset = rng.normal(0.0, 0.10)
            drift = rng.normal(0.0, 0.04) * np.linspace(-1.0, 1.0, series_length)

            if class_id == 0:
                # Nearly pure periodic oscillation around 4 Hz.
                frequency = rng.normal(4.0, 0.12)
                clean = amplitude * np.sin(
                    2.0 * np.pi * frequency * time + phase
                )

            elif class_id == 1:
                # Fundamental plus harmonics, producing a nonsinusoidal shape.
                frequency = rng.normal(4.0, 0.12)
                theta = 2.0 * np.pi * frequency * time + phase
                clean = amplitude * (
                    np.sin(theta)
                    + 0.45 * np.sin(2.0 * theta + 0.3)
                    + 0.20 * np.sin(3.0 * theta - 0.4)
                )

            elif class_id == 2:
                # Periodic carrier with a slow periodic amplitude envelope.
                carrier_frequency = rng.normal(5.0, 0.15)
                modulation_frequency = rng.normal(0.65, 0.03)
                envelope = 1.0 + 0.50 * np.sin(
                    2.0 * np.pi * modulation_frequency * time
                    + rng.uniform(0.0, 2.0 * np.pi)
                )
                clean = amplitude * envelope * np.sin(
                    2.0 * np.pi * carrier_frequency * time + phase
                )

            elif class_id == 3:
                # Repeated frequency sweep within each one-second cycle.
                base_frequency = rng.normal(2.0, 0.08)
                sweep_rate = rng.normal(5.0, 0.15)
                cycle_time = np.mod(time, 1.0)
                chirp_phase = (
                    2.0 * np.pi
                    * (
                        base_frequency * cycle_time
                        + 0.5 * sweep_rate * cycle_time**2
                    )
                    + phase
                )
                clean = amplitude * np.sin(chirp_phase)

            else:
                # Smooth pulse-like periodic signal using a tanh-compressed sine.
                frequency = rng.normal(3.0, 0.10)
                theta = 2.0 * np.pi * frequency * time + phase
                clean = amplitude * np.tanh(3.0 * np.sin(theta))

            noise = rng.normal(0.0, noise_std, series_length)
            signal = clean + offset + drift + noise

            signals.append(signal.astype(np.float32))
            labels.append(class_id)

    # sktime collection format: (instances, channels, time points)
    X = np.asarray(signals, dtype=np.float32)[:, np.newaxis, :]
    y = np.asarray(labels, dtype=np.int64)

    # Shuffle once before returning.
    order = rng.permutation(len(y))
    return X[order], y[order], time


def plot_examples(
    X: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    output_path: Path,
) -> None:
    """Save one generated example from every class."""
    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(10, 9), sharex=True)

    for class_id, axis in enumerate(axes):
        sample_index = np.flatnonzero(y == class_id)[0]
        axis.plot(time, X[sample_index, 0])
        axis.set_ylabel(CLASS_NAMES[class_id])
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Generated periodic one-channel time series")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    num_kernels: int = 10_000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train MultiRocket and return test labels, predictions, and accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # RocketClassifier creates MultiRocket features and fits RidgeClassifierCV.
    classifier = RocketClassifier(
        rocket_transform="multirocket",
        num_kernels=num_kernels,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples:     {len(y_test)}")
    print(f"Input shape:      {X.shape}")
    print(f"Test accuracy:    {accuracy:.4f}\n")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    return y_test, predictions, accuracy


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float,
    output_path: Path,
) -> None:
    """Save a row-normalized confusion matrix."""
    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=CLASS_NAMES,
        normalize="true",
        values_format=".2f",
        xticks_rotation=30,
        ax=axis,
    )
    axis.set_title(f"MultiRocket confusion matrix, accuracy = {accuracy:.3f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toy MultiRocket classifier for periodic signals."
    )
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--series-length", type=int, default=500)
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--num-kernels", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("multirocket_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y, time = generate_periodic_dataset(
        samples_per_class=args.samples_per_class,
        series_length=args.series_length,
        sampling_rate=args.sampling_rate,
        noise_std=args.noise_std,
        random_state=args.seed,
    )

    examples_path = args.output_dir / "periodic_examples.png"
    confusion_path = args.output_dir / "confusion_matrix.png"

    plot_examples(X, y, time, examples_path)

    y_test, predictions, accuracy = train_and_evaluate(
        X,
        y,
        test_size=args.test_size,
        num_kernels=args.num_kernels,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    plot_confusion_matrix(
        y_test,
        predictions,
        accuracy,
        confusion_path,
    )

    print(f"Saved example signals to:   {examples_path.resolve()}")
    print(f"Saved confusion matrix to: {confusion_path.resolve()}")


if __name__ == "__main__":
    main()
