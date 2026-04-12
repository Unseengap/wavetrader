"""
Training loop for the WaveFollower trend-following model.

Extends the base SignalLoss with an additional trend classification loss
and an add-score calibration loss.

Usage:
    python -m wavetrader.train_wave_follower \\
        --pair GBP/JPY --epochs 60 --checkpoint wavefollower_best.pt
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .wave_follower import WaveFollower, WaveFollowerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class TrendFollowerLoss(nn.Module):
    """
    Combined loss for WaveFollower:

      1. **Signal CE** (BUY/SELL/HOLD) — weighted, HOLD down-weighted.
      2. **Trend CE** (UP/DOWN/NEUTRAL) — teaches the model macro direction.
      3. **Confidence MSE** — push confidence toward 1 when correct, 0 when wrong.
      4. **Add-score calibration** — positive add_score when price moves in trend
         direction after a pullback, negative otherwise.
    """

    def __init__(
        self,
        signal_weight: float = 1.0,
        trend_weight: float = 0.5,
        conf_weight: float = 0.1,
        add_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.signal_weight = signal_weight
        self.trend_weight = trend_weight
        self.conf_weight = conf_weight
        self.add_weight = add_weight

        self.signal_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.3])
        )
        self.trend_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.3])
        )

    def forward(
        self,
        output: Dict[str, Tensor],
        signal_labels: Tensor,
        trend_labels: Optional[Tensor] = None,
        add_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        # 1. Signal loss
        signal_loss = self.signal_ce(output["signal_logits"], signal_labels)

        # 2. Trend loss (if labels provided)
        trend_loss = torch.tensor(0.0, device=signal_loss.device)
        if trend_labels is not None:
            trend_loss = self.trend_ce(output["trend_logits"], trend_labels)

        # 3. Confidence calibration
        probs = F.softmax(output["signal_logits"], dim=-1)
        correct = (probs.argmax(-1) == signal_labels).float()
        conf = output["confidence"].squeeze(-1)
        conf_loss = F.mse_loss(conf, correct)

        # 4. Add-score calibration (if targets provided)
        add_loss = torch.tensor(0.0, device=signal_loss.device)
        if add_targets is not None:
            add_loss = F.mse_loss(output["add_score"].squeeze(-1), add_targets)

        total = (
            self.signal_weight * signal_loss
            + self.trend_weight * trend_loss
            + self.conf_weight * conf_loss
            + self.add_weight * add_loss
        )

        return {
            "total": total,
            "signal_loss": signal_loss,
            "trend_loss": trend_loss,
            "conf_loss": conf_loss,
            "add_loss": add_loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_wavefollower(
    model: WaveFollower,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: WaveFollowerConfig,
    device: torch.device,
    checkpoint: str = "wavefollower_best.pt",
) -> Dict[str, List[float]]:
    """Train WaveFollower; saves best checkpoint by val accuracy."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )
    criterion = TrendFollowerLoss().to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
    }
    best_val_acc = 0.0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining WaveFollower  ({n_params:,} parameters)")
    print(f"Timeframes : {config.timeframes}")
    print(f"Device     : {device}")
    print("-" * 60)

    for epoch in range(config.epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss, val_acc = _val_epoch(model, val_loader, criterion, device, config)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

        print(
            f"Epoch {epoch + 1:3d}/{config.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}"
        )

    print("-" * 60)
    print(f"Best val accuracy: {best_val_acc:.2%}")
    return history


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TrendFollowerLoss,
    device: torch.device,
    config: WaveFollowerConfig,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        signal_labels = batch["label"].to(device)
        trend_labels = batch.get("trend_label")
        if trend_labels is not None:
            trend_labels = trend_labels.to(device)
        add_targets = batch.get("add_target")
        if add_targets is not None:
            add_targets = add_targets.to(device)

        model_input = {
            tf: {k: v.to(device) for k, v in batch[tf].items()}
            for tf in config.timeframes
            if tf in batch and isinstance(batch[tf], dict)
        }

        optimizer.zero_grad()
        out = model(model_input)
        losses = criterion(out, signal_labels, trend_labels, add_targets)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += losses["total"].item()
    return total / max(len(loader), 1)


def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TrendFollowerLoss,
    device: torch.device,
    config: WaveFollowerConfig,
) -> Tuple[float, float]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            signal_labels = batch["label"].to(device)
            trend_labels = batch.get("trend_label")
            if trend_labels is not None:
                trend_labels = trend_labels.to(device)

            model_input = {
                tf: {k: v.to(device) for k, v in batch[tf].items()}
                for tf in config.timeframes
                if tf in batch and isinstance(batch[tf], dict)
            }

            out = model(model_input)
            losses = criterion(out, signal_labels, trend_labels)
            total += losses["total"].item()
            correct += (out["signal_logits"].argmax(-1) == signal_labels).sum().item()
            n += signal_labels.size(0)
    return total / max(len(loader), 1), correct / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import pandas as pd
    from torch.utils.data import DataLoader, random_split
    from .dataset import MTFForexDataset
    from .data import load_data

    parser = argparse.ArgumentParser(description="Train WaveFollower model")
    parser.add_argument("--pair", default="GBP/JPY")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint", default="wavefollower_best.pt")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    config = WaveFollowerConfig(
        pair=args.pair,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load multi-TF data
    dfs: Dict[str, pd.DataFrame] = {}
    for tf in config.timeframes:
        df = load_data(args.pair, timeframe=tf, data_dir=args.data_dir)
        if df is not None and not df.empty:
            dfs[tf] = df
        else:
            raise RuntimeError(f"No data found for {args.pair} {tf}")

    # Build dataset
    dataset = MTFForexDataset(dfs, config, lookahead=10, pair=args.pair)
    print(f"Dataset size: {len(dataset)} samples")

    # Train/val split (80/20)
    n = len(dataset)
    n_val = int(n * 0.2)
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = WaveFollower(config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Train
    history = train_wavefollower(
        model, train_loader, val_loader, config, device,
        checkpoint=args.checkpoint,
    )

    print(f"\nCheckpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
