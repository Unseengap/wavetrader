"""
Training loss for the AMDScalper model.

Extends the base SignalLoss with:
  - Phase classification (ACCUM/MANIP/DIST/INVALID)
  - Per-entry-model auxiliary losses (S&R, S&D, ORB)
  - Gate diversity regularisation (prevent collapse to single entry model)
  - Confidence calibration
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AMDScalperLoss(nn.Module):
    """
    Combined loss for AMDScalper:

      1. **Signal CE** (BUY/SELL/HOLD) — weighted, HOLD down-weighted.
      2. **Phase CE** (ACCUM/MANIP/DIST/INVALID) — teaches session awareness.
      3. **Per-entry-model CE** — auxiliary loss on each entry head (S&R, S&D, ORB).
      4. **Gate entropy** — penalise collapsed gates (encourages using all 3 models).
      5. **Confidence MSE** — push confidence toward 1 when correct, 0 when wrong.
    """

    def __init__(
        self,
        signal_weight: float = 1.0,
        phase_weight: float = 0.3,
        entry_aux_weight: float = 0.15,
        gate_entropy_weight: float = 0.05,
        conf_weight: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.signal_weight = signal_weight
        self.phase_weight = phase_weight
        self.entry_aux_weight = entry_aux_weight
        self.gate_entropy_weight = gate_entropy_weight
        self.conf_weight = conf_weight

        # Signal: down-weight HOLD
        self.signal_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.3]),
            label_smoothing=label_smoothing,
        )

        # Phase: weight INVALID lower (noise label), DISTRIBUTION higher (trade zone)
        self.phase_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([0.8, 1.0, 1.2, 0.3]),
            label_smoothing=label_smoothing,
        )

        # Auxiliary entry CEs (same weighting as signal)
        self.entry_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.3]),
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        output: Dict[str, Tensor],
        signal_labels: Tensor,
        phase_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        device = output["signal_logits"].device

        # 1. Signal loss (gated ensemble output)
        signal_loss = self.signal_ce(output["signal_logits"], signal_labels)

        # 2. Phase loss (if phase labels available)
        phase_loss = torch.tensor(0.0, device=device)
        if phase_labels is not None:
            phase_loss = self.phase_ce(output["phase_logits"], phase_labels)

        # 3. Auxiliary entry model losses (each head should independently be correct)
        sr_loss = self.entry_ce(output["sr_logits"], signal_labels)
        sd_loss = self.entry_ce(output["sd_logits"], signal_labels)
        orb_loss = self.entry_ce(output["orb_logits"], signal_labels)
        entry_aux_loss = (sr_loss + sd_loss + orb_loss) / 3.0

        # 4. Gate entropy regularisation: H(gate) — encourage diversity
        gate = output["gate_weights"]  # [B, 3]
        gate_entropy = -(gate * (gate + 1e-8).log()).sum(dim=-1).mean()
        # Max entropy for 3 classes = log(3) ≈ 1.099
        # We want to maximise entropy → minimise negative entropy
        gate_entropy_loss = -gate_entropy  # negative → maximise entropy

        # 5. Confidence calibration
        probs = F.softmax(output["signal_logits"], dim=-1)
        correct = (probs.argmax(-1) == signal_labels).float()
        conf = output["confidence"].squeeze(-1)
        conf_loss = F.mse_loss(conf, correct)

        total = (
            self.signal_weight * signal_loss
            + self.phase_weight * phase_loss
            + self.entry_aux_weight * entry_aux_loss
            + self.gate_entropy_weight * gate_entropy_loss
            + self.conf_weight * conf_loss
        )

        return {
            "total": total,
            "signal_loss": signal_loss,
            "phase_loss": phase_loss,
            "entry_aux_loss": entry_aux_loss,
            "sr_loss": sr_loss,
            "sd_loss": sd_loss,
            "orb_loss": orb_loss,
            "gate_entropy": gate_entropy,
            "conf_loss": conf_loss,
        }
