"""
WaveFollower — Pair-Agnostic Multi-Timeframe Trend Following Model.

Design Principles
─────────────────
1. **Pair-agnostic**: Uses only normalized price structure (HH/HL/LL/LH),
   RSI, volume, and ATR — never hard-codes pair-specific values.
2. **Multi-timeframe**: Higher TFs (4H, 1D) determine the macro trend
   direction; lower TFs (15m, 1H) identify pullback entries.
3. **Trend classification**: Each timeframe independently classifies
   the current trend (UP / DOWN / NEUTRAL) using structure bias and
   swing-point sequences.
4. **Entry on pullbacks**: Enters when lower TF pulls back against the
   higher-TF trend (HL in uptrend, LH in downtrend).
5. **Pyramid / double-up**: The model outputs an ``add_score`` indicating
   whether to add to a winning position on valid pullbacks.
6. **Exit on structure break**: Closes when the structure breaks against
   the trade (LH→LL in uptrend, HL→HH in downtrend).

Architecture
────────────
  ┌──────────────────────────────────────────────────────┐
  │  Per-TF Encoder (shared weights across all TFs)      │
  │  ── PriceConv → StructureMLPEncoder → RSI/Vol enc ─→ │ [B, T, tf_dim]
  │  ── Causal self-attention ────────────────────────→  │ [B, tf_dim]
  └───────────────────────────┬──────────────────────────┘
                              │ × N timeframes
                    ┌─────────▼─────────┐
                    │  Trend Fusion     │
                    │  (hierarchical    │
                    │   cross-attention)│
                    └─────────┬─────────┘
                              │ [B, fused_dim]
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐
  │ TrendHead    │  │ PullbackHead   │  │ RiskHead       │
  │ UP/DOWN/NEUT │  │ entry + add    │  │ SL/TP/trailing │
  └──────────────┘  └────────────────┘  └────────────────┘
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WaveFollowerConfig:
    """Hyperparameters for the WaveFollower trend-following model."""

    # --- Timeframes (small → large) ---
    timeframes: List[str] = field(
        default_factory=lambda: ["15min", "1h", "4h", "1d"]
    )
    lookbacks: Dict[str, int] = field(
        default_factory=lambda: {
            "15min": 120,    # ~30 hours — captures intraday pullbacks
            "1h":    120,    # ~5 days
            "4h":    100,    # ~16 days — sees trend legs
            "1d":    60,     # ~3 months — macro direction
        }
    )
    entry_timeframe: str = "15min"

    # --- Encoder ---
    tf_wave_dim: int = 192          # Per-TF encoder output (lighter than MTF)
    structure_emb_dim: int = 64     # Structure-specific embedding
    price_conv_dim: int = 128       # Price conv channels

    # --- Fusion ---
    fused_dim: int = 384            # After hierarchical fusion

    # --- Predictor ---
    predictor_hidden: int = 384
    predictor_heads: int = 6
    predictor_layers: int = 3
    predictor_ff_dim: int = 1536

    # --- Outputs ---
    n_trend_classes: int = 3        # UP=0, DOWN=1, NEUTRAL=2
    n_signal_classes: int = 3       # BUY=0, SELL=1, HOLD=2

    # --- Training ---
    dropout: float = 0.15
    learning_rate: float = 5e-5
    batch_size: int = 16
    epochs: int = 60

    # --- Pair-agnostic flag (no pair embedding) ---
    pair: str = ""  # empty = any pair

    @property
    def output_wave_dim(self) -> int:
        return self.fused_dim


# ─────────────────────────────────────────────────────────────────────────────
# Causal building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Upper-triangular causal attention mask (True = masked)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class CausalConv1d(nn.Module):
    """Left-padded 1-D convolution — no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Structure-Aware Encoder (pair-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class StructureTrendEncoder(nn.Module):
    """
    Encodes the 8-dim structure features (one-hot HH/HL/LL/LH/NONE + bars_since
    + swing_rate + trend_bias) into a trend-aware representation.

    This is the heart of the trend-following model — it learns to read
    swing-point sequences to determine trend direction and pullback state.
    """

    def __init__(self, wave_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, wave_dim),
            nn.LayerNorm(wave_dim),
        )
        # Causal conv to capture swing-point sequences (HH→HL→HH→HL = uptrend)
        self.seq_conv = nn.Sequential(
            CausalConv1d(wave_dim, wave_dim, kernel_size=7),
            nn.GELU(),
            CausalConv1d(wave_dim, wave_dim, kernel_size=5),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, structure: Tensor) -> Tensor:
        """structure: [B, T, 8] → [B, T, wave_dim]"""
        h = self.mlp(structure)                        # [B, T, D]
        h_conv = self.seq_conv(h.transpose(1, 2))      # [B, D, T]
        return self.norm(h + h_conv.transpose(1, 2))    # [B, T, D] residual


class PriceEncoder(nn.Module):
    """Encodes normalized OHLCV via causal convolution stack."""

    def __init__(self, wave_dim: int = 128, dropout: float = 0.1) -> None:
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
        """ohlcv: [B, T, 5] → [B, T, wave_dim]"""
        x = self.norm(ohlcv.transpose(1, 2))  # BN over channel dim
        x = self.conv_stack(x)                  # [B, D, T]
        return self.out_norm(x.transpose(1, 2))


class MomentumEncoder(nn.Module):
    """Encodes RSI (3-dim) and volume (3-dim) into a single momentum vector."""

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
        """rsi: [B,T,3], volume: [B,T,3] → [B,T, wave_dim]"""
        return self.norm(torch.cat([self.rsi_proj(rsi), self.vol_proj(volume)], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Per-Timeframe Encoder (shared across all TFs — pair-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class TrendTimeframeEncoder(nn.Module):
    """
    Compresses one timeframe's features into a single summary vector.

    The structure encoder gets a disproportionate share of dimensions
    because trend direction is derived primarily from swing-point
    sequences (HH/HL vs LL/LH).
    """

    def __init__(self, config: WaveFollowerConfig) -> None:
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

        # Causal temporal attention to distill the sequence into one vector
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
        price_h = self.price_enc(ohlcv)            # [B, T, price_dim]
        struct_h = self.structure_enc(structure)    # [B, T, struct_dim]
        mom_h = self.momentum_enc(rsi, volume)      # [B, T, 48]

        combined = torch.cat([price_h, struct_h, mom_h], dim=-1)  # [B, T, combined]
        fused = self.fusion(combined)                              # [B, T, D]

        # Causal self-attention
        mask = _causal_mask(fused.size(1), fused.device)
        attn_out, _ = self.temporal_attn(fused, fused, fused, attn_mask=mask)
        fused = self.temporal_norm(fused + attn_out)

        # Last position = summary of the entire history
        return fused[:, -1, :]   # [B, D]


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Trend Fusion
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalTrendFusion(nn.Module):
    """
    Fuses timeframe vectors with hierarchical bias:
      1D direction dominates → 4H confirms → 1H refines → 15m triggers.

    Higher TFs get higher learned prior weight.  Cross-attention lets
    the entry TF attend to higher-TF context.
    """

    def __init__(self, config: WaveFollowerConfig) -> None:
        super().__init__()
        D = config.tf_wave_dim
        N = len(config.timeframes)
        out = config.fused_dim

        # Learnable TF hierarchy weights (initialise with higher TF bias)
        self.tf_weights = nn.Parameter(
            torch.tensor([0.5, 1.0, 1.5, 2.0][:N], dtype=torch.float32)
        )

        # Cross-attention: entry TF queries higher TFs
        self.cross_attn = nn.MultiheadAttention(
            D, num_heads=4, batch_first=True, dropout=config.dropout,
        )
        self.cross_norm = nn.LayerNorm(D)

        # Final projection
        self.fusion = nn.Sequential(
            nn.Linear(D * N, out * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(out * 2, out),
            nn.LayerNorm(out),
        )

        # Trend alignment: how strongly do all timeframes agree on direction?
        self.alignment_head = nn.Sequential(
            nn.Linear(D * N, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, tf_waves: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        tf_waves: list of [B, D] tensors, ordered [entry, confirm, trend, bias].
        Returns (fused [B, fused_dim],  alignment [B, 1]).
        """
        # Weight each TF
        weights = F.softmax(self.tf_weights, dim=0)
        weighted = [w * wave for w, wave in zip(weights, tf_waves)]

        # Stack for cross-attention: [B, N, D]
        stacked = torch.stack(weighted, dim=1)

        # Entry TF attends to all TFs (including itself)
        query = weighted[0].unsqueeze(1)                  # [B, 1, D]
        attn_out, _ = self.cross_attn(query, stacked, stacked)
        entry_refined = self.cross_norm(query + attn_out).squeeze(1)  # [B, D]

        # Replace entry vector with cross-attended version
        weighted[0] = entry_refined

        # Concatenate all and fuse
        cat = torch.cat(weighted, dim=-1)                  # [B, D*N]
        fused = self.fusion(cat)                           # [B, fused_dim]
        alignment = self.alignment_head(cat)               # [B, 1]

        return fused, alignment


# ─────────────────────────────────────────────────────────────────────────────
# Output Heads
# ─────────────────────────────────────────────────────────────────────────────

class TrendHead(nn.Module):
    """
    Classifies the current macro trend:
      0 = UP   (HH + HL dominant on higher TFs)
      1 = DOWN (LL + LH dominant on higher TFs)
      2 = NEUTRAL / ranging
    """

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, D] → [B, 3] logits for UP/DOWN/NEUTRAL"""
        return self.net(x)


class PullbackHead(nn.Module):
    """
    Outputs:
      - entry_logits [B, 3]: BUY / SELL / HOLD (should we enter now on pullback?)
      - add_score    [B, 1]: sigmoid score for pyramiding (0=don't add, 1=add)
    """

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.entry_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.add_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: [B, D] → (entry_logits [B,3], add_score [B,1])"""
        return self.entry_net(x), self.add_net(x)


class ConfidenceHead(nn.Module):
    """Calibrated confidence ∈ [0, 1] for the signal."""

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
    """
    Outputs learned risk parameters (positive via softplus):
      [0] SL factor,  [1] TP factor,  [2] trailing pct
    """

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
# Transformer predictor (lighter version for trend following)
# ─────────────────────────────────────────────────────────────────────────────

class TrendPredictorBlock(nn.Module):
    """Pre-LN transformer block with causal self-attention."""

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


class TrendPredictor(nn.Module):
    """Small transformer that refines the fused wave before heads."""

    def __init__(self, config: WaveFollowerConfig) -> None:
        super().__init__()
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
        """x: [B, D] → [B, D]   (single vector, unsqueeze/squeeze for attn)."""
        x = self.input_proj(x).unsqueeze(1)   # [B, 1, H]
        for block in self.blocks:
            x = block(x)
        return self.norm(self.output_proj(x).squeeze(1))  # [B, D]


# ─────────────────────────────────────────────────────────────────────────────
# Main Model: WaveFollower
# ─────────────────────────────────────────────────────────────────────────────

class WaveFollower(nn.Module):
    """
    Pair-agnostic multi-timeframe trend-following model.

    Inputs:   batch = {tf: {ohlcv, structure, rsi, volume}, ...}
    Outputs:  {
        "signal_logits":  [B, 3]   BUY / SELL / HOLD
        "trend_logits":   [B, 3]   UP / DOWN / NEUTRAL
        "confidence":     [B, 1]
        "alignment":      [B, 1]   cross-TF agreement
        "add_score":      [B, 1]   pyramid / double-up score
        "risk_params":    [B, 3]   SL / TP / trailing (raw positive)
        "wave_state":     [B, D]   for resonance buffer
    }
    """

    def __init__(self, config: Optional[WaveFollowerConfig] = None) -> None:
        super().__init__()
        self.config = config or WaveFollowerConfig()

        # Shared TF encoder (pair-agnostic: same weights for every pair/TF)
        self.tf_encoder = TrendTimeframeEncoder(self.config)

        # Fusion
        self.fusion = HierarchicalTrendFusion(self.config)

        # Predictor
        self.predictor = TrendPredictor(self.config)

        # Heads
        D = self.config.fused_dim
        self.trend_head = TrendHead(D, self.config.dropout)
        self.pullback_head = PullbackHead(D, self.config.dropout)
        self.confidence_head = ConfidenceHead(D)
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

        # Hierarchical fusion with trend alignment
        fused, alignment = self.fusion(tf_waves)   # [B, fused_dim], [B, 1]

        # Refine via transformer predictor
        refined = self.predictor(fused)             # [B, fused_dim]

        # Output heads
        trend_logits = self.trend_head(refined)                    # [B, 3]
        entry_logits, add_score = self.pullback_head(refined)      # [B, 3], [B, 1]
        confidence = self.confidence_head(refined)                  # [B, 1]
        risk_params = self.risk_head(refined)                       # [B, 3]

        return {
            "signal_logits": entry_logits,       # BUY / SELL / HOLD
            "trend_logits":  trend_logits,        # UP / DOWN / NEUTRAL
            "confidence":    confidence,
            "alignment":     alignment,
            "add_score":     add_score,
            "risk_params":   risk_params,
            "wave_state":    refined,
        }

    # ── Inference helper ──────────────────────────────────────────────────

    def predict(
        self,
        batch: Dict[str, Dict[str, Tensor]],
        entry_price: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Single-sample inference helper.

        Returns a plain dict with human-readable values, compatible with
        the dashboard's LiveService and the streaming engine.
        """
        from .config import DEFAULT_RISK_SCALING

        self.eval()
        with torch.no_grad():
            out = self.forward(batch)

        signal_idx = out["signal_logits"].argmax(-1).item()
        signal_name = ["BUY", "SELL", "HOLD"][signal_idx]
        trend_idx = out["trend_logits"].argmax(-1).item()
        trend_name = ["UP", "DOWN", "NEUTRAL"][trend_idx]
        confidence = out["confidence"].item()
        alignment = out["alignment"].item()
        add_score = out["add_score"].item()

        risk = out["risk_params"][0]
        rs = DEFAULT_RISK_SCALING

        return {
            "signal": signal_name,
            "trend": trend_name,
            "confidence": round(confidence * (0.5 + 0.5 * alignment), 4),
            "alignment": round(alignment, 4),
            "add_score": round(add_score, 4),
            "sl_pips": round(rs.sl_pips(risk[0].item()), 1),
            "tp_pips": round(rs.tp_pips(risk[1].item()), 1),
            "trailing_pct": round(rs.trailing_pct(risk[2].item()), 4),
            "entry_price": entry_price,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
