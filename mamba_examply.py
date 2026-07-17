import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import random
import tempfile
import wave
import torch.nn.functional as F

try:
    import winsound
except ImportError:
    winsound = None


# -----------------------------
# 1. Toy one-channel instrument data
# -----------------------------
class ToyInstrumentDataset(Dataset):
    def __init__(self, n_samples=1200, seq_len=512, num_classes=10):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.num_classes = num_classes

    def generate_waveform(self, label, seq_len=None, deterministic=False):
        if seq_len is None:
            seq_len = self.seq_len

        t = torch.linspace(0, 1, seq_len)

        # 10 class-specific base frequencies to keep classes distinct.
        base_freqs = [
            3.5, 4.8, 6.2, 7.8, 9.6,
            11.8, 14.2, 17.0, 20.2, 24.0
        ]
        base_f = base_freqs[label % len(base_freqs)]
        if deterministic:
            f = base_f
            phase = 0.0
        else:
            f = random.uniform(base_f * 0.93, base_f * 1.07)
            phase = random.uniform(0, 2 * math.pi)

        if label == 0:
            # Pure low tone.
            envelope = 0.9 + 0.1 * torch.cos(2 * math.pi * 0.6 * t)
            x = torch.sin(2 * math.pi * f * t + phase)
            x += 0.10 * torch.sin(2 * math.pi * 2 * f * t)
            x = envelope * x

        elif label == 1:
            # Tone with medium vibrato.
            vib_depth = 0.22 if deterministic else random.uniform(0.18, 0.28)
            vib_rate = 4.0 if deterministic else random.uniform(3.6, 4.6)
            vibrato = vib_depth * torch.sin(2 * math.pi * vib_rate * t)
            x = torch.sin(2 * math.pi * (f + vibrato) * t + phase)
            x += 0.20 * torch.sin(2 * math.pi * 2 * f * t)

        elif label == 2:
            # Pluck-like decay.
            decay_rate = 3.2 if deterministic else random.uniform(2.8, 3.8)
            envelope = torch.exp(-decay_rate * t)
            x = envelope * torch.sin(2 * math.pi * f * t + phase)
            x += 0.18 * envelope * torch.sin(2 * math.pi * 2 * f * t)

        elif label == 3:
            # Tremolo AM tone.
            am_depth = 0.55 if deterministic else random.uniform(0.45, 0.65)
            am_rate = 3.5 if deterministic else random.uniform(3.0, 4.0)
            am = 1.0 + am_depth * torch.sin(2 * math.pi * am_rate * t)
            x = am * torch.sin(2 * math.pi * f * t + phase)

        elif label == 4:
            # Bright harmonic stack.
            x = torch.sin(2 * math.pi * f * t + phase)
            x += 0.45 * torch.sin(2 * math.pi * 2 * f * t)
            x += 0.30 * torch.sin(2 * math.pi * 3 * f * t)

        elif label == 5:
            # Narrow pulse train style (odd harmonics dominant).
            x = torch.sin(2 * math.pi * f * t + phase)
            x += 0.35 * torch.sin(2 * math.pi * 3 * f * t)
            x += 0.20 * torch.sin(2 * math.pi * 5 * f * t)

        elif label == 6:
            # Upward chirp.
            k = 3.5 if deterministic else random.uniform(3.0, 4.2)
            phase_t = 2 * math.pi * (f * t + 0.5 * k * t * t)
            x = torch.sin(phase_t + phase)

        elif label == 7:
            # Downward chirp.
            k = 3.0 if deterministic else random.uniform(2.6, 3.5)
            phase_t = 2 * math.pi * ((f + k) * t - 0.5 * k * t * t)
            x = torch.sin(phase_t + phase)

        elif label == 8:
            # Two partials with beating.
            delta = 0.35 if deterministic else random.uniform(0.25, 0.45)
            x = 0.65 * torch.sin(2 * math.pi * (f - delta) * t + phase)
            x += 0.65 * torch.sin(2 * math.pi * (f + delta) * t)

        else:
            # Transient onset + sustain tail.
            attack = 1.0 - torch.exp(-18 * t)
            tail = torch.exp(-1.6 * t)
            x = attack * tail * torch.sin(2 * math.pi * f * t + phase)
            x += 0.22 * tail * torch.sin(2 * math.pi * 2 * f * t)

        # Keep shared noise modest so classes remain separable.
        noise_level = 0.025 if deterministic else random.uniform(0.015, 0.035)
        x += noise_level * torch.randn_like(x)

        # Normalize each sample.
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = random.randint(0, self.num_classes - 1)
        x = self.generate_waveform(label)

        # Shape: [T, 1]
        return x.unsqueeze(-1).float(), torch.tensor(label).long()


def play_waveform_once(x, sample_rate=16000):
    if winsound is None:
        return

    # Convert normalized float waveform to 16-bit PCM WAV and play it.
    x_pcm = torch.clamp(x, -1.0, 1.0)
    x_pcm = (x_pcm * 32767).short().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(x_pcm.tobytes())

        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    finally:
        try:
            import os
            os.remove(wav_path)
        except OSError:
            pass


def preview_class_sounds(dataset, sample_rate=16000):
    print("Previewing one sound per class before learning...")
    for label in range(dataset.num_classes):
        print(f"Playing class {label} sample")
        x = dataset.generate_waveform(label=label, seq_len=sample_rate, deterministic=True)
        play_waveform_once(x, sample_rate=sample_rate)


# ---------------------------------------------
# 2. Tiny MambaTS-style temporal block
# ---------------------------------------------
class TinyMambaBlock(nn.Module):
    """
    A minimal Mamba-style block following the paper's selective SSM idea.

    Core selective SSM:
        h_t = A_t h_{t-1} + B_t x_t
        y_t = C_t h_t

    where:
        A is learned,
        Δ_t, B_t, C_t are generated from the current input,
        A_t = exp(Δ_t A).

    This implementation uses a simple Python loop for clarity.
    Real Mamba uses a hardware-aware selective scan.
    """

    def __init__(self, d_model, d_state=16, expand=2, kernel_size=4):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        self.norm = nn.LayerNorm(d_model)

        # Mamba-style input projection:
        # one branch goes to SSM, one branch is used as a gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # Local depthwise convolution before SSM
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.d_inner
        )

        # Generate input-dependent Δ_t, B_t, C_t
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state)

        # Project scalar Δ_t to each channel
        self.dt_proj = nn.Linear(1, self.d_inner)

        # Learned continuous-time A
        # Shape: [D_inner, N]
        A = torch.arange(1, d_state + 1).float()
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection inside SSM branch
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def selective_scan(self, x, delta, B, C):
        """
        x:     [B, T, D_inner]
        delta: [B, T, D_inner]
        B:     [B, T, N]
        C:     [B, T, N]

        returns:
        y:     [B, T, D_inner]
        """

        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # A should be negative for stable dynamics
        # A: [D_inner, N]
        A = -torch.exp(self.A_log)

        # hidden state h: [B, D_inner, N]
        h = torch.zeros(
            batch,
            d_inner,
            d_state,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]          # [B, D_inner]
            delta_t = delta[:, t, :]  # [B, D_inner]
            B_t = B[:, t, :]          # [B, N]
            C_t = C[:, t, :]          # [B, N]

            # Discretize A:
            # A_t = exp(Δ_t A)
            # Shape: [B, D_inner, N]
            A_t = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))

            # Discretized input effect.
            # This is a simplified version of the ZOH B discretization.
            # Shape: [B, D_inner, N]
            B_t = B_t.unsqueeze(1)
            x_t_expanded = x_t.unsqueeze(-1)
            input_term = delta_t.unsqueeze(-1) * B_t * x_t_expanded

            # State update:
            # h_t = A_t h_{t-1} + B_t x_t
            h = A_t * h + input_term

            # Output:
            # y_t = C_t h_t
            # C_t: [B, N], h: [B, D_inner, N]
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        return y

    def forward(self, x):
        """
        x: [B, T, D_model]
        """

        residual = x

        x = self.norm(x)

        # Split into SSM branch and gate branch
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Depthwise causal convolution
        x_ssm = x_ssm.transpose(1, 2)              # [B, D_inner, T]
        x_ssm = self.conv(x_ssm)[:, :, :x.shape[1]]
        x_ssm = x_ssm.transpose(1, 2)              # [B, T, D_inner]

        x_ssm = F.silu(x_ssm)

        # Generate Δ, B, C from current input
        params = self.x_proj(x_ssm)

        delta_raw = params[..., :1]                # [B, T, 1]
        B = params[..., 1:1 + self.d_state]         # [B, T, N]
        C = params[..., 1 + self.d_state:]          # [B, T, N]

        # Positive Δ
        delta = F.softplus(self.dt_proj(delta_raw)) # [B, T, D_inner]

        # Selective SSM
        y = self.selective_scan(x_ssm, delta, B, C)

        # Skip connection inside SSM branch
        y = y + x_ssm * self.D

        # Gate branch
        y = y * F.silu(z)

        y = self.out_proj(y)

        return y + residual
    
# --------------------------------
# 3. MambaTS-style classifier
# --------------------------------
class MambaTSInstrumentClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=10, num_classes=10, depth=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList([
            TinyMambaBlock(d_model) for _ in range(depth)
        ])

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


# -----------------------------
# 4. Training
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

train_set = ToyInstrumentDataset(n_samples=3000, num_classes=NUM_CLASSES)
test_set = ToyInstrumentDataset(n_samples=800, num_classes=NUM_CLASSES)

preview_class_sounds(train_set)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

model = MambaTSInstrumentClassifier(num_classes=NUM_CLASSES).to(device)

def mixup_batch(x, y, alpha=0.2):
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


epochs = 40
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
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

        if random.random() < 0.6:
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