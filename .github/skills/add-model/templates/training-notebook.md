# Training Notebook Template

Reference template for `{ModelName}_Training.ipynb`. Based on `WaveFollower_Training.ipynb` — the fastest and most optimized training notebook in the project. All new training notebooks MUST match these optimizations.

## Cell Structure

### Cell 1: Header (Markdown)

```markdown
# {ModelName} Training Notebook
## {Architecture description} — GPU Training

**{ModelName}** is a {description of what the model does}.

**Data**: Reuses the same processed parquet data from the MTF pipeline.

**OANDA Account**: `{account_id}` (separate from other models)
```

### Cell 2: Imports & GPU Check

```python
import sys, os, time, json, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path.cwd()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")
if device.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"CUDA version    : {torch.version.cuda}")
    # Performance flags — ALL mandatory
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"cuDNN benchmark : enabled")
    print(f"TF32            : enabled")
else:
    print("WARNING: No GPU detected — training will use CPU (slow).")
```

### Cell 3: Configuration

```python
from wavetrader.{model_module} import {ModelClass}, {ModelConfig}

config = {ModelConfig}(
    learning_rate=2e-4,   # Scale with batch size: lr *= batch/128
    batch_size=640,       # Max for T4 16GB; reduce if OOM
    epochs=60,
    dropout=0.15,
)

PROCESSED_DIR = Path("processed_data")
CHECKPOINT_DIR = Path("checkpoints/{model_tag}")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = ["GBPJPY", "EURJPY", "USDJPY", "GBPUSD"]
PAIR_NAMES = {"GBPJPY": "GBP/JPY", "EURJPY": "EUR/JPY",
              "USDJPY": "USD/JPY", "GBPUSD": "GBP/USD"}
```

### Cell 4: Check Processed Data

```python
def check_processed_data(processed_dir, pairs, timeframes):
    status = {}
    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        for pair in pairs:
            for tf in timeframes:
                tf_short = tf.replace("min", "m")
                path = split_dir / f"{pair}_{tf_short}.parquet"
                status[f"{split}/{pair}_{tf_short}"] = path.exists()

    found = sum(v for v in status.values())
    total = len(status)
    print(f"Processed data: {found}/{total} files found")
    return all(status.values()), status
```

### Cell 5: Load Data from Parquets

```python
from wavetrader.data import _normalise_df

def load_split_data(split, pairs, timeframes, processed_dir):
    all_pair_data = {}
    for pair_tag in pairs:
        dfs = {}
        for tf in timeframes:
            tf_short = tf.replace("min", "m")
            path = processed_dir / split / f"{pair_tag}_{tf_short}.parquet"
            df = pd.read_parquet(path)
            df = _normalise_df(df)
            dfs[tf] = df
        all_pair_data[pair_tag] = dfs
    return all_pair_data
```

### Cell 6: Verify Data Compatibility

```python
from wavetrader.dataset import prepare_features

# Check all required feature groups exist
REQUIRED_FEATURES = {
    "ohlcv":     ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"],
    "structure": [f"structure_{i}" for i in range(8)],
    "rsi":       ["rsi_norm", "rsi_delta_norm", "rsi_accel_norm"],
    "volume":    ["volume_norm", "volume_ratio", "volume_delta"],
}
```

### Cell 7: PreCachedMTFDataset + DataLoaders (CRITICAL for speed)

```python
from wavetrader.dataset import MTFForexDataset, mtf_collate_fn
from torch.utils.data import ConcatDataset, Dataset

class PreCachedMTFDataset(Dataset):
    """
    Wraps MTFForexDataset and pre-caches ALL samples as contiguous tensors.
    __getitem__ becomes a pure tensor index — zero pandas overhead per batch.
    Trades ~2-4 GB RAM for massive training speed improvement.
    """

    def __init__(self, mtf_dataset: MTFForexDataset):
        print(f"    Pre-caching {len(mtf_dataset):,} samples...", end=" ", flush=True)
        t0 = time.time()

        sample = mtf_dataset[0]
        self.timeframes = [k for k in sample if k not in ("label", "trend_label", "add_target")]
        self.feat_keys = {tf: list(sample[tf].keys()) for tf in self.timeframes}

        n = len(mtf_dataset)
        self.tensors = {}
        for tf in self.timeframes:
            self.tensors[tf] = {}
            for feat in self.feat_keys[tf]:
                shape = sample[tf][feat].shape
                self.tensors[tf][feat] = torch.empty(n, *shape, dtype=torch.float32)
        self.labels = torch.empty(n, dtype=torch.long)

        for i in range(n):
            item = mtf_dataset[i]
            for tf in self.timeframes:
                for feat in self.feat_keys[tf]:
                    self.tensors[tf][feat][i] = item[tf][feat]
            self.labels[i] = item["label"]

        print(f"done in {time.time() - t0:.0f}s")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        result = {}
        for tf in self.timeframes:
            result[tf] = {feat: self.tensors[tf][feat][idx] for feat in self.feat_keys[tf]}
        result["label"] = self.labels[idx]
        return result


# DataLoaders — GPU-optimized settings
NUM_WORKERS = min(4, os.cpu_count() or 1)
PREFETCH = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=mtf_collate_fn,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
    drop_last=True,  # Avoids small last-batch slowdown
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size * 2,  # 2x batch for val (no grads = less VRAM)
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=mtf_collate_fn,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
)
```

### Cell 8: Model + torch.compile

```python
model = {ModelClass}(config).to(device)

# torch.compile — fuses kernels, eliminates Python overhead (15-30% speedup)
if device.type == "cuda" and hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile: enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"torch.compile: skipped ({e})")

# Sanity check — also triggers compile warmup
sample_batch = next(iter(train_loader))
model_input = {
    tf: {k: v.to(device) for k, v in sample_batch[tf].items()}
    for tf in config.timeframes
    if tf in sample_batch and isinstance(sample_batch[tf], dict)
}
with torch.no_grad():
    out = model(model_input)
print("Forward pass OK — output shapes:", {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
```

### Cell 9: Loss + Optimizer + Scheduler

```python
from wavetrader.train_{model_module} import {LossClass}

criterion = {LossClass}().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    fused=(device.type == "cuda"),  # Fused AdamW — fewer kernel launches
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01,
)

use_amp = device.type == "cuda"
scaler = GradScaler("cuda", enabled=use_amp)
```

### Cell 10: Training Loop (INLINE — not delegated)

The training loop MUST be inline in the notebook, not delegated to `train_model()` or
`train_mtf_model()`. This gives full control over AMP, gradient clipping, logging,
and checkpointing. The older `WaveTrader_Colab.ipynb` delegates to those functions
and misses all the GPU optimizations.

```python
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, config, use_amp):
    model.train()
    total_loss, correct, n_samples = 0.0, 0, 0

    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        model_input = {
            tf: {k: v.to(device, non_blocking=True) for k, v in batch[tf].items()}
            for tf in config.timeframes
            if tf in batch and isinstance(batch[tf], dict)
        }

        optimizer.zero_grad(set_to_none=True)  # Avoids memset overhead

        with autocast("cuda", enabled=use_amp):
            out = model(model_input)
            losses = criterion(out, labels)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)  # MUST come before clip_grad_norm_
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["total"].item()
        correct += (out["signal_logits"].argmax(-1) == labels).sum().item()
        n_samples += labels.size(0)

    n_batches = max(len(loader), 1)
    return {"loss": total_loss / n_batches, "accuracy": correct / max(n_samples, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device, config, use_amp):
    model.eval()
    total_loss, correct, n_samples = 0.0, 0, 0

    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        model_input = {
            tf: {k: v.to(device, non_blocking=True) for k, v in batch[tf].items()}
            for tf in config.timeframes
            if tf in batch and isinstance(batch[tf], dict)
        }
        with autocast("cuda", enabled=use_amp):
            out = model(model_input)
            losses = criterion(out, labels)
        total_loss += losses["total"].item()
        correct += (out["signal_logits"].argmax(-1) == labels).sum().item()
        n_samples += labels.size(0)

    n_batches = max(len(loader), 1)
    return {"loss": total_loss / n_batches, "accuracy": correct / max(n_samples, 1)}


# Main loop with best-checkpoint saving
best_val_acc = 0.0
for epoch in range(config.epochs):
    train_m = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config, use_amp)
    val_m = validate(model, val_loader, criterion, device, config, use_amp)
    scheduler.step()

    if val_m["accuracy"] > best_val_acc:
        best_val_acc = val_m["accuracy"]
        torch.save({"model_state_dict": model.state_dict(), ...}, checkpoint_path)
```

### Cell 11: Training Curves (Plotly)

Plot loss, accuracy, LR, and overfit gap (train_acc - val_acc) on dark theme.

### Cell 12: Test Evaluation

Reload best checkpoint, evaluate on test set, print confusion matrix and per-class accuracy.

### Cell 13: Save Checkpoint + Google Drive Copy

See checkpoint save pattern in SKILL.md Phase 2.

## GPU Memory Guide

| GPU | VRAM | Max Batch Size | Notes |
|-----|------|----------------|-------|
| T4 | 16 GB | 640 | Colab free/pro default |
| A100 | 40 GB | 2048 | Colab Pro+, scale LR |
| V100 | 16 GB | 640 | Same as T4 |
| RTX 3090 | 24 GB | 1024 | Local training |
| CPU | N/A | 32-64 | Emergency only, very slow |

If OOM: halve batch size, enable gradient accumulation:
```python
ACCUM_STEPS = 2  # Effective batch = batch_size * ACCUM_STEPS
optimizer.zero_grad(set_to_none=True)
for i, batch in enumerate(train_loader):
    with autocast("cuda"):
        loss = criterion(model(inputs), labels)["total"] / ACCUM_STEPS
    scaler.scale(loss).backward()
    if (i + 1) % ACCUM_STEPS == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```
