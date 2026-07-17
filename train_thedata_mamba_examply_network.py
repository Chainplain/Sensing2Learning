import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class TinyMambaBlock(nn.Module):
    """
    Minimal Mamba-style block with input-dependent selective SSM parameters.
    """

    def __init__(self, d_model, d_state=16, expand=2, kernel_size=4):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.d_inner,
        )

        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state)
        self.dt_proj = nn.Linear(1, self.d_inner)

        a = torch.arange(1, d_state + 1).float()
        a = a.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def selective_scan(self, x, delta, b_param, c_param):
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        a = -torch.exp(self.A_log)

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            delta_t = delta[:, t, :]
            b_t = b_param[:, t, :]
            c_t = c_param[:, t, :]

            a_t = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))

            b_t = b_t.unsqueeze(1)
            x_t_expanded = x_t.unsqueeze(-1)
            input_term = delta_t.unsqueeze(-1) * b_t * x_t_expanded

            h = a_t * h + input_term
            y_t = torch.sum(h * c_t.unsqueeze(1), dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv(x_ssm)[:, :, : x.shape[1]]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        params = self.x_proj(x_ssm)
        delta_raw = params[..., :1]
        b_param = params[..., 1 : 1 + self.d_state]
        c_param = params[..., 1 + self.d_state :]

        delta = F.softplus(self.dt_proj(delta_raw))

        y = self.selective_scan(x_ssm, delta, b_param, c_param)
        y = y + x_ssm * self.D

        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual


class MambaTSInstrumentClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=10, num_classes=10, depth=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TinyMambaBlock(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)

        x = self.norm(x)
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


def build_fixed_chunks(classes, sampling_rate, chunks_per_class=400, chunk_sec=4.0):
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

            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-6)
            samples.append(torch.from_numpy(chunk).unsqueeze(-1).float())
            labels.append(class_idx)

    if not samples:
        raise ValueError("No chunks were generated. Check data length and sampling rate.")

    return samples, labels, class_names, chunk_len


def plot_random_chunks(classes, sampling_rate, chunk_sec=4.0, n_plots=4):
    chunk_len = max(8, int(round(chunk_sec * sampling_rate)))

    eligible = []
    for class_name, _, x in classes:
        if x.size >= chunk_len:
            eligible.append((class_name, x))

    if not eligible:
        raise ValueError("No class has enough samples for plotting fixed-length chunks.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    t_axis = np.arange(chunk_len) / sampling_rate

    for i in range(min(n_plots, 4)):
        class_name, x = random.choice(eligible)
        start = random.randint(0, x.size - chunk_len)
        chunk = x[start : start + chunk_len]

        axes[i].plot(t_axis, chunk, linewidth=1.2)
        axes[i].set_title(f"Class: {class_name}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(alpha=0.3)

    for i in range(min(n_plots, 4), 4):
        axes[i].axis("off")

    fig.suptitle("Random 4-second chunks", fontsize=13)
    plt.show()


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

    if args.plot_random_chunks:
        plot_random_chunks(
            classes=classes,
            sampling_rate=sampling_rate,
            chunk_sec=4.0,
            n_plots=4,
        )

    samples, labels, class_names, seq_len = build_fixed_chunks(
        classes=classes,
        sampling_rate=sampling_rate,
        chunks_per_class=args.chunks_per_class,
        chunk_sec=4.0,
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

    print(f"Loaded classes: {len(class_names)}")
    print(f"Estimated sampling rate: {sampling_rate:.6f} Hz")
    print("Chunk duration: 4.00s (fixed, no temporal resampling)")
    print(f"Training sequence length: {seq_len} samples")
    print(f"Train/Test chunks: {len(train_set)}/{len(test_set)}")

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
        description="Train the Mamba examply network on Thedata fixed 4s chunks."
    )
    parser.add_argument("--input", type=Path, default=Path("Thedata.xlsx"), help="Input Excel file")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--chunks-per-class", type=int, default=400, help="Random chunks per class")
    parser.add_argument("--mixup-prob", type=float, default=0.6, help="Probability of mixup per batch")
    parser.add_argument(
        "--plot-random-chunks",
        action="store_true",
        help="Plot 4 random fixed-length chunks in a 2x2 grid before training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
