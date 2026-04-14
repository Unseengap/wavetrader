"""
MeanReversion — Pair-Agnostic Multi-Timeframe Mean-Reversion Model.

Design Principles
─────────────────
1. **Mean-reversion**: Detects when price is overextended relative to its
   rolling mean (Bollinger Bands / Z-score) and trades the snap-back.
2. **Multi-timeframe**: Higher TFs (4H, 1D) confirm the ranging regime;
   lower TFs (15m, 1H) identify the exact reversal entry.
3. **High win-rate**: Targets ~71 % win rate with tighter TP (back to mean)
   and wider SL (allow overshoot before reversal).
4. **Regime awareness**: Should *not* trade during strong trends — the model
   learns to output HOLD when higher-TF structure is clearly trending.

Architecture
────────────
Reuses WaveFollower's encoder stack (PriceEncoder, StructureTrendEncoder,
MomentumEncoder, TrendTimeframeEncoder, HierarchicalTrendFusion) because
the input features are identical — only the output heads and loss differ.

Output heads:
  ┌───────────────┐  ┌────────────────┐  ┌────────────────┐
  │ SignalHead     │  │ ConfidenceHead │  │ RiskHead       │
  │ BUY/SELL/HOLD │  │ calibrated 0-1 │  │ SL/TP/trailing │
  └───────────────┘  └────────────────┘  └────────────────┘
  ┌───────────────┐
  │ ExtensionHead │
  │ z-score est.  │
  └───────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MeanRevConfig
from .wave_follower import (
    CausalConv1d,
    ConfidenceHead,
    HierarchicalTrendFusion,
    MomentumEncoder,
    PriceEncoder,
    RiskHead,
    StructureTrendEncoder,
    TrendPredictor,
    TrendTimeframeEncoder,
    _causal_mask,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mean-Reversion-specific heads
# ─────────────────────────────────────────────────────────────────────────────

class MeanRevSignalHead(nn.Module):
    """
    BUY / SELL / HOLD classifier tuned for mean-reversion entries.
    Slightly deeper than PullbackHead because reversal timing is hard.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, D] → [B, 3] logits for BUY/SELL/HOLD"""
        return self.net(x)


class ExtensionHead(nn.Module):
    """
    Estimates how overextended price is from its mean (z-score proxy).
    Output ∈ ℝ — positive = above mean, negative = below mean.
    Used as an auxiliary training signal and an inference filter.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, D] → [B, 1] estimated z-score."""
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Adapted TF Encoder for MeanRevConfig
# ─────────────────────────────────────────────────────────────────────────────

class MeanRevTimeframeEncoder(nn.Module):
    """Same architecture as TrendTimeframeEncoder but accepts MeanRevConfig."""

    def __init__(self, config: MeanRevConfig) -> None:
        super().__init__()
        D = config.tf_wave_dim
        self.price_enc = PriceEncoder(wave_dim=config.price_conv_dim, dropout=config.dropout)
        self.structure_enc = StructureTrendEncoder(
            wave_dim=config.structure_emb_dim, dropout=config.dropout,
        )
        self.momentum_enc = MomentumEncoder(wave_dim=48, dropout=config.dropout)

        combined = config.price_conv_dim + config.structure_emb_dim + 48
        self.fusion = nn.Sequential(
            nn.Linear(combined, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.temporal_attn = nn.MultiheadAttention(
            D, num_heads=4, batch_first=True, dropout=config.dropout,
        )
        self.temporal_norm = nn.LayerNorm(D)

    def forward(
        self,
        ohlcv: Tensor,
        structure: Tensor,
        rsi: Tensor,
        volume: Tensor,
    ) -> Tensor:
        """All inputs [B, T, features] → [B, D] summary."""
        price_h = self.price_enc(ohlcv)
        struct_h = self.structure_enc(structure)
        mom_h = self.momentum_enc(rsi, volume)

        combined = torch.cat([price_h, struct_h, mom_h], dim=-1)
        fused = self.fusion(combined)

        mask = _causal_mask(fused.size(1), fused.device)
        attn_out, _ = self.temporal_attn(fused, fused, fused, attn_mask=mask)
        fused = self.temporal_norm(fused + attn_out)

        return fused[:, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
# Adapted Fusion for MeanRevConfig
# ─────────────────────────────────────────────────────────────────────────────

class MeanRevFusion(nn.Module):
    """
    Hierarchical TF fusion for mean-reversion.
    Same structure as HierarchicalTrendFusion but operates on MeanRevConfig dims.
    """

    def __init__(self, config: MeanRevConfig) -> None:
        super().__init__()
        D = config.tf_wave_dim
        N = len(config.timeframes)
        out = config.fused_dim

        self.tf_weights = nn.Parameter(
            torch.tensor([0.5, 1.0, 1.5, 2.0][:N], dtype=torch.float32)
        )

        self.cross_attn = nn.MultiheadAttention(
            D, num_heads=4, batch_first=True, dropout=config.dropout,
        )
        self.cross_norm = nn.LayerNorm(D)

        self.fusion = nn.Sequential(
            nn.Linear(D * N, out * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(out * 2, out),
            nn.LayerNorm(out),
        )

        self.regime_head = nn.Sequential(
            nn.Linear(D * N, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, tf_waves: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        tf_waves: list of [B, D] tensors.
        Returns (fused [B, fused_dim], regime_score [B, 1]).
        regime_score ≈ 1 means ranging (good for MR), ≈ 0 means trending.
        """
        weights = F.softmax(self.tf_weights, dim=0)
        weighted = [w * wave for w, wave in zip(weights, tf_waves)]

        stacked = torch.stack(weighted, dim=1)
        query = weighted[0].unsqueeze(1)
        attn_out, _ = self.cross_attn(query, stacked, stacked)
        entry_refined = self.cross_norm(query + attn_out).squeeze(1)
        weighted[0] = entry_refined

        cat = torch.cat(weighted, dim=-1)
        fused = self.fusion(cat)
        regime_score = self.regime_head(cat)

        return fused, regime_score


# ─────────────────────────────────────────────────────────────────────────────
# Adapted Predictor for MeanRevConfig
# ─────────────────────────────────────────────────────────────────────────────

class MeanRevPredictor(nn.Module):
    """Small transformer that refines the fused wave before heads."""

    def __init__(self, config: MeanRevConfig) -> None:
        super().__init__()
        from .wave_follower import TrendPredictorBlock

        D = config.fused_dim
        self.input_proj = nn.Linear(D, config.predictor_hidden)
        self.blocks = nn.ModuleList([
            TrendPredictorBlock(
                config.predictor_hidden, config.predictor_heads,
                config.predictor_ff_dim, config.dropout,
            )
            for _ in range(config.predictor_layers)
        ])
        self.output_proj = nn.Linear(config.predictor_hidden, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.norm(self.output_proj(x).squeeze(1))


# ─────────────────────────────────────────────────────────────────────────────
# Main Model: MeanReversion
# ─────────────────────────────────────────────────────────────────────────────

class MeanReversion(nn.Module):
    """
    Pair-agnostic multi-timeframe mean-reversion model.

    Inputs:   batch = {tf: {ohlcv, structure, rsi, volume}, ...}
    Outputs:  {
        "signal_logits":  [B, 3]   BUY / SELL / HOLD
        "confidence":     [B, 1]
        "regime_score":   [B, 1]   ranging ≈ 1, trending ≈ 0
        "extension":      [B, 1]   z-score estimate
        "risk_params":    [B, 3]   SL / TP / trailing (raw positive)
        "wave_state":     [B, D]   for resonance buffer
    }
    """

    def __init__(self, config: Optional[MeanRevConfig] = None) -> None:
        super().__init__()
        self.config = config or MeanRevConfig()

        # Shared TF encoder
        self.tf_encoder = MeanRevTimeframeEncoder(self.config)

        # Fusion
        self.fusion = MeanRevFusion(self.config)

        # Predictor
        self.predictor = MeanRevPredictor(self.config)

        # Heads
        D = self.config.fused_dim
        self.signal_head = MeanRevSignalHead(D, self.config.dropout)
        self.confidence_head = ConfidenceHead(D)
        self.extension_head = ExtensionHead(D)
        self.risk_head = RiskHead(D)

    def forward(self, batch: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        tf_waves: List[Tensor] = []

        for tf in self.config.timeframes:
            if tf not in batch:
                raise KeyError(f"Missing timeframe '{tf}' in batch")
            d = batch[tf]
            wave = self.tf_encoder(
                ohlcv=d["ohlcv"],
                structure=d["structure"],
                rsi=d["rsi"],
                volume=d["volume"],
            )
            tf_waves.append(wave)

        fused, regime_score = self.fusion(tf_waves)
        refined = self.predictor(fused)

        signal_logits = self.signal_head(refined)
        confidence = self.confidence_head(refined)
        extension = self.extension_head(refined)
        risk_params = self.risk_head(refined)

        return {
            "signal_logits": signal_logits,
            "confidence":    confidence,
            "regime_score":  regime_score,
            "extension":     extension,
            "risk_params":   risk_params,
            "wave_state":    refined,
        }

    # ── Inference helper ──────────────────────────────────────────────────

    def predict(
        self,
        batch: Dict[str, Dict[str, Tensor]],
        entry_price: float = 0.0,
    ) -> Dict[str, Any]:
        """Single-sample inference helper for dashboard / streaming."""
        from .config import DEFAULT_RISK_SCALING

        self.eval()
        with torch.no_grad():
            out = self.forward(batch)

        signal_idx = out["signal_logits"].argmax(-1).item()
        signal_name = ["BUY", "SELL", "HOLD"][signal_idx]
        confidence = out["confidence"].item()
        regime = out["regime_score"].item()
        extension = out["extension"].item()

        risk = out["risk_params"][0]
        rs = DEFAULT_RISK_SCALING

        return {
            "signal": signal_name,
            "confidence": round(confidence * (0.5 + 0.5 * regime), 4),
            "regime_score": round(regime, 4),
            "extension": round(extension, 4),
            "sl_pips": round(rs.sl_pips(risk[0].item()), 1),
            "tp_pips": round(rs.tp_pips(risk[1].item()), 1),
            "trailing_pct": round(rs.trailing_pct(risk[2].item()), 4),
            "entry_price": entry_price,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
