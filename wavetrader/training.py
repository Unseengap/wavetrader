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

        if True:
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

        if True:
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


# ─────────────────────────────────────────────────────────────────────────────
# v2 Training: Focal Loss, per-class metrics, early stopping on BUY/SELL F1
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): down-weights easy examples, focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    With gamma=2.0, correctly classified samples with p > 0.6 contribute < 10%
    of their original loss, forcing the model to focus on uncertain predictions.
    """

    def __init__(
        self,
        alpha:   Optional[List[float]] = None,
        gamma:   float = 2.0,
        weight:  Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        if weight is not None:
            self.register_buffer("class_weight", weight)
        else:
            self.class_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        logits:  [B, C] raw logits
        targets: [B] class indices
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_one_hot).sum(-1)   # [B]
        focal_weight = (1.0 - pt) ** self.gamma

        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets_one_hot * log_probs).sum(-1)  # [B]

        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()


class SignalLossV2(nn.Module):
    """
    v2 loss function:
      • Focal Loss for BUY/SELL/HOLD classification
      • MSE confidence calibration (same as v1)
    """

    def __init__(
        self,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.use_focal = use_focal
        if use_focal:
            alpha = focal_alpha or [1.0, 1.0, 0.3]
            self.signal_loss = FocalLoss(alpha=alpha, gamma=focal_gamma)
        else:
            self.signal_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(focal_alpha or [1.0, 1.0, 0.3])
            )

    def forward(
        self, output: Dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor]:
        signal_loss = self.signal_loss(output["signal_logits"], labels)

        probs      = F.softmax(output["signal_logits"], dim=-1)
        correct    = (probs.argmax(-1) == labels).float()
        conf       = output["confidence"].squeeze(-1)
        conf_loss  = F.mse_loss(conf, correct)

        total = signal_loss + 0.1 * conf_loss
        return {"total": total, "signal_loss": signal_loss, "conf_loss": conf_loss}


def _compute_per_class_metrics(
    all_preds: List[int], all_labels: List[int], n_classes: int = 3
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per class. Returns {cls: {prec, rec, f1}}."""
    import numpy as np
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    metrics = {}
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        class_name = ["BUY", "SELL", "HOLD"][c]
        metrics[class_name] = {"precision": prec, "recall": rec, "f1": f1}
    return metrics


def _directional_f1(metrics: Dict[str, Dict[str, float]]) -> float:
    """Weighted F1 of BUY + SELL only (ignoring HOLD)."""
    buy_f1  = metrics.get("BUY", {}).get("f1", 0.0)
    sell_f1 = metrics.get("SELL", {}).get("f1", 0.0)
    return (buy_f1 + sell_f1) / 2.0


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for `warmup_epochs`, then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        import math
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


def train_mtf_model_v2(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config,       # MTFv2Config
    device:       torch.device,
    checkpoint:   str = "wavetrader_mtf_v2_best.pt",
) -> Dict[str, List]:
    """
    Train WaveTraderMTFv2 with:
      - Focal Loss (or weighted CE)
      - LR warmup + cosine annealing
      - Early stopping on BUY/SELL directional F1
      - Per-class precision/recall/F1 logging every epoch
    """
    model     = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )

    warmup = getattr(config, 'warmup_epochs', 5)
    scheduler = WarmupCosineScheduler(optimizer, warmup, config.epochs)

    use_focal = getattr(config, 'use_focal_loss', True)
    gamma     = getattr(config, 'focal_gamma', 2.0)
    alpha     = getattr(config, 'focal_alpha', [1.0, 1.0, 0.3])
    criterion = SignalLossV2(use_focal=use_focal, focal_gamma=gamma, focal_alpha=alpha).to(device)

    patience    = getattr(config, 'early_stopping_patience', 10)
    best_f1     = 0.0
    epochs_no_improve = 0

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "val_f1_directional": [], "per_class_metrics": [],
    }

    print(f"\nTraining WaveTraderMTFv2  ({_count(model):,} parameters)")
    print(f"Timeframes : {config.timeframes}")
    print(f"Focal Loss : {'ON' if use_focal else 'OFF'} (gamma={gamma})")
    print(f"Early Stop : patience={patience} on directional F1")
    print(f"LR Warmup  : {warmup} epochs → cosine annealing")
    print(f"Device     : {device}")
    print("-" * 70)

    for epoch in range(config.epochs):
        train_loss = _train_epoch_mtf(model, train_loader, optimizer, criterion, device, config)
        val_loss, val_acc, per_class = _val_epoch_mtf_v2(model, val_loader, criterion, device, config)
        scheduler.step()

        dir_f1 = _directional_f1(per_class)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1_directional"].append(dir_f1)
        history["per_class_metrics"].append(per_class)

        # Early stopping on directional F1
        if dir_f1 > best_f1:
            best_f1 = dir_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        buy_f1  = per_class.get("BUY", {}).get("f1", 0.0)
        sell_f1 = per_class.get("SELL", {}).get("f1", 0.0)
        hold_f1 = per_class.get("HOLD", {}).get("f1", 0.0)

        print(
            f"Epoch {epoch+1:3d}/{config.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}  "
            f"F1[B/S/H]={buy_f1:.2f}/{sell_f1:.2f}/{hold_f1:.2f}  "
            f"dir_f1={dir_f1:.3f}  lr={lr:.1e}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best checkpoint
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    print("-" * 70)
    print(f"Best directional F1: {best_f1:.3f}")
    return history


def _val_epoch_mtf_v2(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config,
) -> Tuple[float, float, Dict]:
    """Validation with per-class metrics."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            model_input = {
                tf: {k: v.to(device) for k, v in batch[tf].items()}
                for tf in config.timeframes
            }
            out    = model(model_input)
            total += criterion(out, labels)["total"].item()
            preds  = out["signal_logits"].argmax(-1)
            correct += (preds == labels).sum().item()
            n       += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = total / max(len(loader), 1)
    val_acc  = correct / max(n, 1)
    per_class = _compute_per_class_metrics(all_preds, all_labels)
    return val_loss, val_acc, per_class
