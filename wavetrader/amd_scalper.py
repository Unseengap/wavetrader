"""
AMDScalper — 5-Minute Session Manipulation Reversal Model.

Uses the Accumulation–Manipulation–Distribution (AMD) framework to identify
high-probability scalping entries during the New York session.

Architecture
────────────
  ┌─────────────────────────────────────────────────────────────┐
  │  Per-TF Encoder (shared weights)                            │
  │  ── PriceConv → StructureMLP → Momentum → AMD Features ─→  │ [B, T, tf_dim]
  │  ── Causal self-attention ─────────────────────────────→    │ [B, tf_dim]
  └──────────────────────────┬──────────────────────────────────┘
                             │ × N timeframes (5m, 15m, 1h, 4h)
                   ┌─────────▼─────────┐
                   │  Session Fusion    │
                   │  (AMD-aware       │
                   │   cross-attention)│
                   └─────────┬─────────┘
                             │ [B, fused_dim]
         ┌───────────────────┼───────────────────────┐
         ▼                   ▼                       ▼
  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐
  │ PhaseHead    │  │ EntryGate      │  │ RiskHead             │
  │ ACCUM/MANIP/ │  │ selects among  │  │ SL/TP/trail          │
  │ DIST/INVALID │  │ 3 entry models │  │ (for scalp targets)  │
  └──────────────┘  └───────┬────────┘  └──────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ S&R Head   │  │ S&D Head   │  │ ORB Head   │
     │ flip zones │  │ demand+FVG │  │ breakout   │
     └────────────┘  └────────────┘  └────────────┘

Three entry models activated by a learned gating network:
  1. Support & Resistance — horizontal flip zones + engulfing
  2. Supply & Demand — demand/supply zones + FVG + engulfing
  3. Open Range Breakout — NY ORB + retracement entry
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import AMDScalperConfig


# ─────────────────────────────────────────────────────────────────────────────
# Causal primitives (local copies — same as WaveFollower)
# ─────────────────────────────────────────────────────────────────────────────

def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# AMD-Specific Encoders
# ─────────────────────────────────────────────────────────────────────────────

class PriceEncoder(nn.Module):
    """Normalized OHLCV → wave via causal convolutions."""

    def __init__(self, wave_dim: int = 112, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(5)
        self.conv_stack = nn.Sequential(
            CausalConv1d(5, 64, kernel_size=5),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(64, wave_dim, kernel_size=3),
            nn.GELU(),
        )
        self.out_norm = nn.LayerNorm(wave_dim)

    def forward(self, ohlcv: Tensor) -> Tensor:
        x = self.norm(ohlcv.transpose(1, 2))
        x = self.conv_stack(x)
        return self.out_norm(x.transpose(1, 2))


class StructureEncoder(nn.Module):
    """8-dim market structure → wave representation."""

    def __init__(self, wave_dim: int = 56, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, wave_dim),
            nn.LayerNorm(wave_dim),
        )
        self.seq_conv = nn.Sequential(
            CausalConv1d(wave_dim, wave_dim, kernel_size=5),
            nn.GELU(),
            CausalConv1d(wave_dim, wave_dim, kernel_size=3),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, structure: Tensor) -> Tensor:
        h = self.mlp(structure)
        h_conv = self.seq_conv(h.transpose(1, 2))
        return self.norm(h + h_conv.transpose(1, 2))


class MomentumEncoder(nn.Module):
    """RSI (3-dim) + Volume (3-dim) → single momentum vector."""

    def __init__(self, wave_dim: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        self.rsi_proj = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, wave_dim // 2),
        )
        self.vol_proj = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, wave_dim // 2),
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, rsi: Tensor, volume: Tensor) -> Tensor:
        return self.norm(torch.cat([self.rsi_proj(rsi), self.vol_proj(volume)], dim=-1))


class AMDFeatureEncoder(nn.Module):
    """
    Encodes AMD-specific features: Asian range (5), London sweep (4),
    engulfing (3), FVG (4), S&R (3), ORB (4) = 23 features total.
    """

    def __init__(self, wave_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(23, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, wave_dim),
            nn.LayerNorm(wave_dim),
        )
        self.context_conv = nn.Sequential(
            CausalConv1d(wave_dim, wave_dim, kernel_size=5),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, amd_feats: Tensor) -> Tensor:
        """amd_feats: [B, T, 23] → [B, T, wave_dim]"""
        h = self.encoder(amd_feats)
        h_conv = self.context_conv(h.transpose(1, 2))
        return self.norm(h + h_conv.transpose(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Per-Timeframe Encoder
# ─────────────────────────────────────────────────────────────────────────────

class AMDTimeframeEncoder(nn.Module):
    """Compresses one timeframe's features into a summary vector."""

    def __init__(self, config: AMDScalperConfig) -> None:
        super().__init__()
        D = config.tf_wave_dim
        self.price_enc = PriceEncoder(wave_dim=config.price_conv_dim, dropout=config.dropout)
        self.structure_enc = StructureEncoder(wave_dim=config.structure_emb_dim, dropout=config.dropout)
        self.momentum_enc = MomentumEncoder(wave_dim=48, dropout=config.dropout)
        self.amd_enc = AMDFeatureEncoder(wave_dim=64, dropout=config.dropout)

        combined = config.price_conv_dim + config.structure_emb_dim + 48 + 64
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
        amd_feats: Tensor,
    ) -> Tensor:
        """All inputs [B, T, features] → [B, D] summary."""
        price_h = self.price_enc(ohlcv)
        struct_h = self.structure_enc(structure)
        mom_h = self.momentum_enc(rsi, volume)
        amd_h = self.amd_enc(amd_feats)

        combined = torch.cat([price_h, struct_h, mom_h, amd_h], dim=-1)
        fused = self.fusion(combined)

        mask = _causal_mask(fused.size(1), fused.device)
        attn_out, _ = self.temporal_attn(fused, fused, fused, attn_mask=mask)
        fused = self.temporal_norm(fused + attn_out)

        return fused[:, -1, :]  # last position = summary


# ─────────────────────────────────────────────────────────────────────────────
# Session-Aware Fusion
# ─────────────────────────────────────────────────────────────────────────────

class SessionAwareFusion(nn.Module):
    """
    Fuses multi-TF vectors with session context.

    5min is the entry TF; higher TFs provide directional bias.
    The fusion is session-aware: it conditions on the AMD phase.
    """

    def __init__(self, config: AMDScalperConfig) -> None:
        super().__init__()
        D = config.tf_wave_dim
        N = len(config.timeframes)
        out = config.fused_dim

        # Learnable TF hierarchy: entry TF has lowest weight (refined by attention)
        self.tf_weights = nn.Parameter(
            torch.tensor([0.5, 1.0, 1.5, 2.0][:N], dtype=torch.float32)
        )

        # Cross-attention: entry TF queries higher TFs
        self.cross_attn = nn.MultiheadAttention(
            D, num_heads=4, batch_first=True, dropout=config.dropout,
        )
        self.cross_norm = nn.LayerNorm(D)

        # Session conditioning — FiLM (regime: session flags + ATR) [4 features]
        self.session_gain = nn.Linear(4, D)
        self.session_bias = nn.Linear(4, D)
        nn.init.zeros_(self.session_gain.weight)
        nn.init.zeros_(self.session_gain.bias)
        nn.init.zeros_(self.session_bias.weight)
        nn.init.zeros_(self.session_bias.bias)

        # Final projection
        self.fusion = nn.Sequential(
            nn.Linear(D * N, out * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(out * 2, out),
            nn.LayerNorm(out),
        )

        # Alignment head
        self.alignment_head = nn.Sequential(
            nn.Linear(D * N, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, tf_waves: List[Tensor], regime: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        tf_waves: list of [B, D], ordered [5m, 15m, 1h, 4h]
        regime:   [B, 4] session flags + ATR
        Returns:  (fused [B, fused_dim], alignment [B, 1])
        """
        weights = F.softmax(self.tf_weights, dim=0)
        weighted = [w * wave for w, wave in zip(weights, tf_waves)]

        stacked = torch.stack(weighted, dim=1)
        query = weighted[0].unsqueeze(1)
        attn_out, _ = self.cross_attn(query, stacked, stacked)
        entry_refined = self.cross_norm(query + attn_out).squeeze(1)

        # Session conditioning on entry TF
        gamma = self.session_gain(regime)
        beta = self.session_bias(regime)
        entry_refined = entry_refined * (1.0 + gamma) + beta

        weighted[0] = entry_refined

        cat = torch.cat(weighted, dim=-1)
        fused = self.fusion(cat)
        alignment = self.alignment_head(cat)

        return fused, alignment


# ─────────────────────────────────────────────────────────────────────────────
# Output Heads
# ─────────────────────────────────────────────────────────────────────────────

class PhaseHead(nn.Module):
    """Classifies the current AMD phase: ACCUM=0, MANIP=1, DIST=2, INVALID=3."""

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EntryModelHead(nn.Module):
    """
    Single entry model head (S&R, S&D, or ORB).
    Outputs signal logits: BUY / SELL / HOLD.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EntryGate(nn.Module):
    """
    Learned gating network that selects among the 3 entry models.
    Outputs soft weights over [S&R, S&D, ORB] based on market context.
    """

    def __init__(self, input_dim: int, n_models: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_models),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, D] → [B, n_models] softmax weights."""
        return F.softmax(self.gate(x), dim=-1)


class ConfidenceHead(nn.Module):
    """Calibrated confidence ∈ [0, 1]."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RiskHead(nn.Module):
    """SL / TP / trailing factors (positive via softplus)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor (small transformer)
# ─────────────────────────────────────────────────────────────────────────────

class ScalperPredictorBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        mask = _causal_mask(x.size(1), x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.attn_norm(x + attn_out)
        return self.ff_norm(x + self.ff(x))


class ScalperPredictor(nn.Module):
    def __init__(self, config: AMDScalperConfig) -> None:
        super().__init__()
        D = config.fused_dim
        self.input_proj = nn.Linear(D, config.predictor_hidden)
        self.blocks = nn.ModuleList([
            ScalperPredictorBlock(
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
# Main Model: AMDScalper
# ─────────────────────────────────────────────────────────────────────────────

class AMDScalper(nn.Module):
    """
    5-Minute AMD Scalper — session manipulation reversal model.

    Inputs:   batch = {
        tf: {ohlcv, structure, rsi, volume, amd_feats, regime}, ...
    }
    Outputs:  {
        "signal_logits":    [B, 3]   BUY / SELL / HOLD (gated ensemble)
        "phase_logits":     [B, 4]   ACCUM / MANIP / DIST / INVALID
        "confidence":       [B, 1]
        "alignment":        [B, 1]   cross-TF agreement
        "gate_weights":     [B, 3]   soft weights over S&R, S&D, ORB
        "sr_logits":        [B, 3]   S&R entry model logits
        "sd_logits":        [B, 3]   S&D entry model logits
        "orb_logits":       [B, 3]   ORB entry model logits
        "risk_params":      [B, 3]   SL / TP / trailing
        "wave_state":       [B, D]   for resonance buffer
    }
    """

    def __init__(self, config: Optional[AMDScalperConfig] = None) -> None:
        super().__init__()
        self.config = config or AMDScalperConfig()

        # Shared TF encoder
        self.tf_encoder = AMDTimeframeEncoder(self.config)

        # Session-aware fusion
        self.fusion = SessionAwareFusion(self.config)

        # Predictor
        self.predictor = ScalperPredictor(self.config)

        # Output heads
        D = self.config.fused_dim
        self.phase_head = PhaseHead(D, self.config.dropout)
        self.confidence_head = ConfidenceHead(D)
        self.risk_head = RiskHead(D)

        # Three entry model heads
        self.sr_head = EntryModelHead(D, self.config.entry_head_dim, self.config.dropout)
        self.sd_head = EntryModelHead(D, self.config.entry_head_dim, self.config.dropout)
        self.orb_head = EntryModelHead(D, self.config.entry_head_dim, self.config.dropout)

        # Gating network
        self.entry_gate = EntryGate(D, n_models=3, dropout=self.config.dropout)

    def forward(self, batch: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        tf_waves: List[Tensor] = []

        # Get regime from entry timeframe
        entry_tf = self.config.entry_timeframe
        if entry_tf not in batch:
            raise KeyError(f"Missing entry timeframe '{entry_tf}' in batch")

        for tf in self.config.timeframes:
            if tf not in batch:
                raise KeyError(f"Missing timeframe '{tf}' in batch")
            d = batch[tf]
            wave = self.tf_encoder(
                ohlcv=d["ohlcv"],
                structure=d["structure"],
                rsi=d["rsi"],
                volume=d["volume"],
                amd_feats=d["amd_feats"],
            )
            tf_waves.append(wave)

        # Regime from entry TF (session flags + ATR)
        regime = batch[entry_tf]["regime"]
        if regime.dim() == 3:
            regime = regime[:, -1, :]  # last bar's regime

        # Fuse all timeframes with session conditioning
        fused, alignment = self.fusion(tf_waves, regime)

        # Refine
        refined = self.predictor(fused)

        # Phase classification
        phase_logits = self.phase_head(refined)

        # Three entry models
        sr_logits = self.sr_head(refined)
        sd_logits = self.sd_head(refined)
        orb_logits = self.orb_head(refined)

        # Gate selects among entry models
        gate_weights = self.entry_gate(refined)   # [B, 3]

        # Gated ensemble: weighted combination of entry model logits
        stacked_logits = torch.stack([sr_logits, sd_logits, orb_logits], dim=1)  # [B, 3, 3]
        signal_logits = torch.einsum("bm,bmc->bc", gate_weights, stacked_logits)  # [B, 3]

        # Confidence and risk
        confidence = self.confidence_head(refined)
        risk_params = self.risk_head(refined)

        return {
            "signal_logits": signal_logits,
            "phase_logits": phase_logits,
            "confidence": confidence,
            "alignment": alignment,
            "gate_weights": gate_weights,
            "sr_logits": sr_logits,
            "sd_logits": sd_logits,
            "orb_logits": orb_logits,
            "risk_params": risk_params,
            "wave_state": refined,
        }

    # ── Inference helper ──────────────────────────────────────────────────

    def predict(
        self,
        batch: Dict[str, Dict[str, Tensor]],
        entry_price: float = 0.0,
    ) -> Dict[str, Any]:
        """Single-sample inference for dashboard / streaming."""
        from .config import DEFAULT_RISK_SCALING

        self.eval()
        with torch.no_grad():
            out = self.forward(batch)

        signal_idx = out["signal_logits"].argmax(-1).item()
        signal_name = ["BUY", "SELL", "HOLD"][signal_idx]
        phase_idx = out["phase_logits"].argmax(-1).item()
        phase_name = ["ACCUMULATION", "MANIPULATION", "DISTRIBUTION", "INVALID"][phase_idx]
        confidence = out["confidence"].item()
        alignment = out["alignment"].item()

        gate = out["gate_weights"][0]
        gate_names = ["S&R", "S&D", "ORB"]
        dominant_model = gate_names[gate.argmax().item()]

        risk = out["risk_params"][0]
        rs = DEFAULT_RISK_SCALING

        return {
            "signal": signal_name,
            "phase": phase_name,
            "entry_model": dominant_model,
            "confidence": round(confidence * (0.5 + 0.5 * alignment), 4),
            "alignment": round(alignment, 4),
            "gate_weights": {
                "sr": round(gate[0].item(), 3),
                "sd": round(gate[1].item(), 3),
                "orb": round(gate[2].item(), 3),
            },
            "sl_pips": round(rs.sl_pips(risk[0].item()), 1),
            "tp_pips": round(rs.tp_pips(risk[1].item()), 1),
            "trailing_pct": round(rs.trailing_pct(risk[2].item()), 4),
            "entry_price": entry_price,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
