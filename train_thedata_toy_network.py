import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class TinyMambaBlock(nn.Module):
    """
    A simple MambaTS-style block for demonstration.
    Real Mamba uses selective state-space modeling.
    This toy block uses gated temporal mixing.
    """

    def __init__(self, d_model, kernel_size=7):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        residual = x

        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)

        u = u.transpose(1, 2)  # [B, D, T]
        u = self.conv(u)
        u = u.transpose(1, 2)  # [B, T, D]

        x = u * torch.sigmoid(gate)
        x = self.out_proj(x)

        return x + residual


class MambaTSInstrumentClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=32, num_classes=10, depth=5, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList([TinyMambaBlock(d_model) for _ in range(depth)])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, T, 1]
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)

        x = self.norm(x)

        # Global temporal pooling
        x = x.mean(dim=1)
        x = self.dropout(x)

        logits = self.classifier(x)

        return logits


class ChunkDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]
        return x, torch.tensor(y).long()


def read_thedata(input_path: Path):
    # Read Thedata in the same style as extract_timeseries_features.py.
    df = pd.read_excel(input_path)
    if df.shape[1] < 2:
        raise ValueError("Thedata must contain at least 2 columns: time + one signal column.")

    time_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    signal_columns = df.columns[1:]

    classes = []
    for col in signal_columns:
        signal = pd.to_numeric(df[col], errors="coerce")
        valid = time_col.notna() & signal.notna()

        t = time_col[valid].to_numpy(dtype=float)
        x = signal[valid].to_numpy(dtype=float)

        if t.size < 4 or x.size < 4:
            continue

        order = np.argsort(t)
        t = t[order]
        x = x[order]

        classes.append((str(col), t, x))

    if not classes:
        raise ValueError("No usable signal columns found after numeric conversion.")

    all_dt = []
    for _, t, _ in classes:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            all_dt.append(dt)

    if not all_dt:
        raise ValueError("Could not infer sampling interval from time column.")

    dt_concat = np.concatenate(all_dt)
    dt_median = float(np.median(dt_concat))
    sampling_rate = 1.0 / dt_median

    return classes, sampling_rate


def build_random_chunks(classes, sampling_rate, chunks_per_class=400, chunk_sec=10.0):
    # Fixed 10-second chunks keep a consistent duration and avoid temporal distortion.
    chunk_len = max(8, int(round(chunk_sec * sampling_rate)))

    samples = []
    labels = []
    class_names = []

    for class_idx, (class_name, _, x) in enumerate(classes):
        n = x.size
        if n < chunk_len:
            continue

        class_names.append(class_name)

        for _ in range(chunks_per_class):
            start = random.randint(0, n - chunk_len)
            chunk = x[start : start + chunk_len].astype(np.float32)

            # Per-chunk normalization keeps scale differences from dominating training.
            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-6)
            samples.append(torch.from_numpy(chunk).unsqueeze(-1).float())
            labels.append(class_idx)

    if not samples:
        raise ValueError("No chunks were generated. Check data length and sampling rate.")

    return samples, labels, class_names, chunk_len


def split_train_test(samples, labels, test_ratio=0.1):
    total = len(samples)
    if total < 2:
        raise ValueError("Need at least two chunks to split into train and test sets.")

    indices = list(range(total))
    random.shuffle(indices)

    test_size = max(1, int(total * test_ratio))
    test_size = min(test_size, total - 1)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    train_samples = [samples[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return (train_samples, train_labels), (test_samples, test_labels)


def compute_test_time_span_by_class(test_labels, class_names, chunk_sec):
    counts = np.bincount(np.array(test_labels, dtype=int), minlength=len(class_names))
    spans_sec = counts.astype(float) * float(chunk_sec)
    return counts, spans_sec


def plot_test_time_span_by_class(class_names, spans_sec):
    if len(class_names) == 0:
        return

    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), 4.8), constrained_layout=True)
    ax.bar(x, spans_sec, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("Estimated test time span (s)")
    ax.set_title("Per-class test time span")
    ax.grid(axis="y", alpha=0.3)
    plt.show()


def mixup_batch(x, y, alpha=0.2):
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def train_model(
    train_loader,
    test_loader,
    num_classes,
    epochs=40,
    learning_rate=2e-3,
    weight_decay=1e-4,
    mixup_prob=0.6,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MambaTSInstrumentClassifier(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if random.random() < mixup_prob and x.size(0) > 1:
                x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
                logits = model(x_mixed)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                pred = logits.argmax(dim=-1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        best_acc = max(best_acc, acc)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss:.4f} | "
            f"Test Acc: {acc:.3f} | Best: {best_acc:.3f} | LR: {current_lr:.6f}"
        )

    return model


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes, sampling_rate = read_thedata(args.input)
    chunk_sec = 10.0
    samples, labels, class_names, seq_len = build_random_chunks(
        classes=classes,
        sampling_rate=sampling_rate,
        chunks_per_class=args.chunks_per_class,
        chunk_sec=chunk_sec,
    )

    (train_samples, train_labels), (test_samples, test_labels) = split_train_test(
        samples,
        labels,
        test_ratio=0.1,
    )

    train_set = ChunkDataset(train_samples, train_labels)
    test_set = ChunkDataset(test_samples, test_labels)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    test_counts, test_spans_sec = compute_test_time_span_by_class(
        test_labels=test_labels,
        class_names=class_names,
        chunk_sec=chunk_sec,
    )

    print(f"Loaded classes: {len(class_names)}")
    print(f"Estimated sampling rate: {sampling_rate:.6f} Hz")
    print("Chunk duration: 10.00s (fixed, no temporal resampling)")
    print(f"Training sequence length: {seq_len} samples")
    print(f"Train/Test chunks: {len(train_set)}/{len(test_set)}")
    print("Estimated test time span by class:")
    for class_name, count, span_sec in zip(class_names, test_counts, test_spans_sec):
        print(f"  {class_name}: {int(count)} chunks -> {span_sec:.2f}s")

    plot_test_time_span_by_class(class_names, test_spans_sec)

    _ = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=len(class_names),
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        mixup_prob=args.mixup_prob,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the toy MambaTS-style network on Thedata fixed 10s chunks."
    )
    parser.add_argument("--input", type=Path, default=Path("Thedata.xlsx"), help="Input Excel file")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--chunks-per-class", type=int, default=400, help="Random chunks per class")
    parser.add_argument("--mixup-prob", type=float, default=0.6, help="Probability of mixup per batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
