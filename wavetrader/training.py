"""
Training loops and loss for FluxSignal and WaveTraderMTF.

Walk-forward validation
────────────────────────
Use walk_forward_splits() instead of a single train/val split to get an
honest out-of-sample estimate across multiple market regimes.  Each fold
trains on an expanding window and tests on the next fixed-size window,
with a gap (purge_pct) to prevent label overlap.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .config import MTFConfig, SignalConfig


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class SignalLoss(nn.Module):
    """
    Combined loss:
      • Weighted cross-entropy for BUY/SELL/HOLD
        (HOLD weight = 0.3 — we care more about directional errors)
      • MSE confidence calibration
        (push confidence toward 1 on correct predictions, 0 on wrong ones)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.3])
        )

    def forward(
        self, output: Dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor]:
        signal_loss = self.ce_loss(output["signal_logits"], labels)

        probs      = F.softmax(output["signal_logits"], dim=-1)
        correct    = (probs.argmax(-1) == labels).float()
        conf       = output["confidence"].squeeze(-1)
        conf_loss  = F.mse_loss(conf, correct)

        total = signal_loss + 0.1 * conf_loss
        return {"total": total, "signal_loss": signal_loss, "conf_loss": conf_loss}


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward split utility
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(
    n: int,
    n_folds:    int   = 5,
    val_pct:    float = 0.15,
    purge_pct:  float = 0.02,
) -> Iterator[Tuple[range, range]]:
    """
    Yield (train_indices, val_indices) for purged walk-forward CV.

    Each fold:
      - train on all data up to the split point
      - skip a purge gap (avoids label overlap from lookahead windows)
      - validate on the next val_pct fraction

    Args:
        n          Total number of samples.
        n_folds    Number of test folds (default 5).
        val_pct    Fraction used as validation window per fold.
        purge_pct  Fraction purged between train end and val start.
    """
    val_size   = int(n * val_pct)
    purge_size = int(n * purge_pct)
    fold_step  = val_size

    start = n - n_folds * fold_step
    if start < val_size:
        raise ValueError(
            f"Not enough data for {n_folds} folds: "
            f"need >{n_folds * fold_step + purge_size} samples, got {n}."
        )

    for fold in range(n_folds):
        val_end   = n - (n_folds - fold - 1) * fold_step
        val_start = val_end - val_size
        train_end = val_start - purge_size
        yield range(train_end), range(val_start, val_end)


# ─────────────────────────────────────────────────────────────────────────────
# Single-timeframe training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:       SignalConfig,
    device:       torch.device,
    checkpoint:   str = "flux_signal_best.pt",
) -> Dict[str, List[float]]:
    """Train FluxSignal; saves best checkpoint by val accuracy."""

    model     = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = SignalLoss().to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_accuracy": []
    }
    best_val_acc = 0.0

    print(f"\nTraining FluxSignal  ({_count(model):,} parameters)")
    print(f"Device : {device}")
    print("-" * 60)

    for epoch in range(config.epochs):
        train_loss = _train_epoch_single(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = _val_epoch_single(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{config.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}"
            )

    print("-" * 60)
    print(f"Best val accuracy: {best_val_acc:.2%}")
    return history


def _train_epoch_single(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SignalLoss,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        optimizer.zero_grad()
        losses = criterion(model(batch), labels)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += losses["total"].item()
    return total / max(len(loader), 1)


def _val_epoch_single(
    model: nn.Module,
    loader: DataLoader,
    criterion: SignalLoss,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            out    = model(batch)
            total += criterion(out, labels)["total"].item()
            correct += (out["signal_logits"].argmax(-1) == labels).sum().item()
            n       += labels.size(0)
    return total / max(len(loader), 1), correct / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_mtf_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:       MTFConfig,
    device:       torch.device,
    checkpoint:   str = "wavetrader_mtf_best.pt",
) -> Dict[str, List[float]]:
    """Train WaveTraderMTF; saves best checkpoint by val accuracy."""

    model     = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = SignalLoss().to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_accuracy": []
    }
    best_val_acc = 0.0

    print(f"\nTraining WaveTraderMTF  ({_count(model):,} parameters)")
    print(f"Timeframes : {config.timeframes}")
    print(f"Device     : {device}")
    print("-" * 60)

    for epoch in range(config.epochs):
        train_loss = _train_epoch_mtf(model, train_loader, optimizer, criterion, device, config)
        val_loss, val_acc = _val_epoch_mtf(model, val_loader, criterion, device, config)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{config.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}"
            )

    print("-" * 60)
    print(f"Best val accuracy: {best_val_acc:.2%}")
    return history


def _train_epoch_mtf(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SignalLoss,
    device: torch.device,
    config: MTFConfig,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        labels = batch["label"].to(device)
        model_input = {
            tf: {k: v.to(device) for k, v in batch[tf].items()}
            for tf in config.timeframes
        }
        optimizer.zero_grad()
        losses = criterion(model(model_input), labels)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += losses["total"].item()
    return total / max(len(loader), 1)


def _val_epoch_mtf(
    model: nn.Module,
    loader: DataLoader,
    criterion: SignalLoss,
    device: torch.device,
    config: MTFConfig,
) -> Tuple[float, float]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            model_input = {
                tf: {k: v.to(device) for k, v in batch[tf].items()}
                for tf in config.timeframes
            }
            out    = model(model_input)
            total += criterion(out, labels)["total"].item()
            correct += (out["signal_logits"].argmax(-1) == labels).sum().item()
            n       += labels.size(0)
    return total / max(len(loader), 1), correct / max(n, 1)


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Synaptic Intelligence — online continual learning
# ─────────────────────────────────────────────────────────────────────────────

class SynapticIntelligence:
    """
    Online parameter importance tracking for continual learning.
    (Zenke, Poole & Ganguli, 2017 — "Continual Learning Through Synaptic Intelligence")

    Unlike EWC, SI accumulates importance *during* each forward/backward pass
    rather than requiring a separate Fisher estimation step.  This makes it
    compatible with streaming forex data where regime shifts are gradual and
    task boundaries are unknown.

    Storage cost: 3 tensors per parameter (omega, theta_star, theta_prev)
    ≈ 3× model size in RAM.  For a 10M-param model this is ~120 MB FP32.

    Usage pattern (online learning loop):
        si = SynapticIntelligence(model, si_lambda=0.1)
        for batch in stream:
            optimizer.zero_grad()
            loss = criterion(model(batch), labels) + si.penalty()
            loss.backward()
            si.update()          # accumulate importance before step
            optimizer.step()
        si.consolidate()         # call periodically (every ~500 batches)
    """

    def __init__(
        self,
        model:     nn.Module,
        si_lambda: float = 0.1,
        epsilon:   float = 1e-3,
    ) -> None:
        self.model     = model
        self.si_lambda = si_lambda
        self.epsilon   = epsilon

        # Snapshot of parameters at last consolidation (theta*)
        self.theta_star = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # Previous-step snapshot — tracks delta per update
        self.theta_prev = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # Accumulated path integral: sum of -grad * delta (= importance proxy)
        self._w = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # Final importance weights (omega): normalised by param displacement
        self.omega = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self) -> None:
        """
        Accumulate importance after each backward pass (call BEFORE optimizer.step()).
        Accumulates -grad * delta_param as a path-integral approximation of the
        contribution of each parameter to the loss reduction.
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta             = p.data - self.theta_prev[n]
                # Negative sign: lower loss = positive contribution
                self._w[n]       -= p.grad.detach() * delta
                self.theta_prev[n] = p.data.clone()

    def consolidate(self) -> None:
        """
        Snapshot current parameters as the new anchor (theta*).
        Normalise accumulated path integrals into omega (importance weights).
        Call periodically (e.g. every `SIConfig.consolidate_every` batches).
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                delta_sq       = (p.data - self.theta_star[n]).pow(2)
                self.omega[n] += self._w[n] / (delta_sq + self.epsilon)
                self.omega[n]  = self.omega[n].clamp(min=0.0)
                self._w[n].zero_()
                self.theta_star[n] = p.data.clone()
                self.theta_prev[n] = p.data.clone()

    def penalty(self) -> Tensor:
        """
        SI regularisation term to add to the task loss.
        Penalises changes to parameters proportional to their accumulated
        importance omega.

        Returns a scalar Tensor (gradients flow through model parameters).
        """
        device = next(self.model.parameters()).device
        loss   = torch.zeros(1, device=device)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss = loss + (
                    self.omega[n].to(device) * (p - self.theta_star[n].to(device)).pow(2)
                ).sum()
        return self.si_lambda * loss
