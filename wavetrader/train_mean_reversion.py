"""
Training loss for the MeanReversion model.

Extends the base SignalLoss with:
  - Extension z-score regression (teaches the model to estimate overextension)
  - Regime calibration (push regime_score toward 1 during ranging, 0 during trends)

Usage:
    python -m wavetrader.train_mean_reversion \\
        --pair GBP/JPY --epochs 60 --checkpoint mean_reversion_best.pt
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MeanRevLoss(nn.Module):
    """
    Combined loss for MeanReversion:

      1. **Signal CE** (BUY/SELL/HOLD) — weighted, HOLD down-weighted.
      2. **Confidence MSE** — push confidence toward 1 when correct, 0 when wrong.
      3. **Extension regression** — L1 loss against true z-score (if provided).
      4. **Regime calibration** — BCE against ranging/trending label (if provided).
    """

    def __init__(
        self,
        signal_weight: float = 1.0,
        conf_weight: float = 0.1,
        extension_weight: float = 0.3,
        regime_weight: float = 0.2,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.signal_weight = signal_weight
        self.conf_weight = conf_weight
        self.extension_weight = extension_weight
        self.regime_weight = regime_weight

        # Down-weight HOLD to encourage the model to take trades
        # Label smoothing prevents overconfident logits → better generalisation
        self.signal_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.3]),
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        output: Dict[str, Tensor],
        signal_labels: Tensor,
        extension_targets: Optional[Tensor] = None,
        regime_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        # 1. Signal loss
        signal_loss = self.signal_ce(output["signal_logits"], signal_labels)

        # 2. Confidence calibration
        probs = F.softmax(output["signal_logits"], dim=-1)
        correct = (probs.argmax(-1) == signal_labels).float()
        conf = output["confidence"].squeeze(-1)
        conf_loss = F.mse_loss(conf, correct)

        # 3. Extension z-score regression
        ext_loss = torch.tensor(0.0, device=signal_loss.device)
        if extension_targets is not None:
            ext_loss = F.l1_loss(
                output["extension"].squeeze(-1), extension_targets
            )

        # 4. Regime calibration
        regime_loss = torch.tensor(0.0, device=signal_loss.device)
        if regime_targets is not None:
            regime_loss = F.binary_cross_entropy(
                output["regime_score"].squeeze(-1), regime_targets
            )

        total = (
            self.signal_weight * signal_loss
            + self.conf_weight * conf_loss
            + self.extension_weight * ext_loss
            + self.regime_weight * regime_loss
        )

        return {
            "total": total,
            "signal_loss": signal_loss,
            "conf_loss": conf_loss,
            "ext_loss": ext_loss,
            "regime_loss": regime_loss,
        }
