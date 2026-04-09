"""
Wave encoder modules: CausalConv1d, per-modality encoders, and CausalWaveChainer.
All encoders are strictly causal (no future bar leakage during inference).
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    1-D convolution that only attends to the past.
    Achieved by left-padding (kernel_size - 1) zeros before the conv.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, channels, seq]
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


def create_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Upper-triangular mask preventing attention to future positions."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality wave encoders
# ─────────────────────────────────────────────────────────────────────────────

class PriceWaveEncoder(nn.Module):
    """
    OHLCV → wave representation.
    Stack of causal convolutions captures local candlestick patterns;
    causal self-attention captures medium-range price interactions.
    """

    def __init__(self, wave_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.input_norm = nn.BatchNorm1d(5)

        self.conv_stack = nn.Sequential(
            CausalConv1d(5, 64, 3),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(64, 128, 3),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(128, wave_dim, 3),
            nn.GELU(),
        )

        self.pattern_attn = nn.MultiheadAttention(
            wave_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.pattern_norm = nn.LayerNorm(wave_dim)

    def forward(self, ohlcv: Tensor) -> Tensor:
        """ohlcv: [B, T, 5]  →  [B, T, wave_dim]"""
        x = self.input_norm(ohlcv.transpose(1, 2))   # [B, 5, T]
        x = self.conv_stack(x).transpose(1, 2)        # [B, T, wave_dim]

        mask = create_causal_mask(x.size(1), x.device)
        attn_out, _ = self.pattern_attn(x, x, x, attn_mask=mask)
        return self.pattern_norm(x + attn_out)


class StructureWaveEncoder(nn.Module):
    """
    Market-structure feature vector → wave.
    Input is the 8-dim output from indicators.classify_structure:
      [one_hot(5), bars_since, swing_rate, trend_bias]
    Sequential causal convolutions learn HH→HL→HH = uptrend patterns.
    """

    def __init__(self, wave_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, wave_dim),
        )

        self.structure_conv = nn.Sequential(
            CausalConv1d(wave_dim, wave_dim, 5),
            nn.GELU(),
            CausalConv1d(wave_dim, wave_dim, 3),
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, structure: Tensor) -> Tensor:
        """structure: [B, T, 8]  →  [B, T, wave_dim]"""
        x = self.encoder(structure)
        conv_out = self.structure_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + conv_out)


class RSIWaveEncoder(nn.Module):
    """
    RSI + first/second derivatives → wave.
    Long causal conv (kernel=7) detects divergence patterns.
    """

    def __init__(self, wave_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.encoder = nn.Sequential(
            nn.Linear(3, 32),   # RSI, RSI_delta, RSI_accel
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, wave_dim),
        )
        self.divergence_conv = CausalConv1d(wave_dim, wave_dim, 7)
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, rsi_data: Tensor) -> Tensor:
        """rsi_data: [B, T, 3]  →  [B, T, wave_dim]"""
        x = self.encoder(rsi_data)
        div_out = self.divergence_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + div_out)


class VolumeWaveEncoder(nn.Module):
    """
    Tick volume + normalised volume features → wave.
    Volume spikes often precede breakouts; causal conv captures clustering.
    """

    def __init__(self, wave_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.encoder = nn.Sequential(
            nn.Linear(3, 32),   # volume_norm, vol/MA ratio, vol delta
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, wave_dim),
        )
        self.volume_conv = CausalConv1d(wave_dim, wave_dim, 5)
        self.norm = nn.LayerNorm(wave_dim)

    def forward(self, volume_data: Tensor) -> Tensor:
        """volume_data: [B, T, 3]  →  [B, T, wave_dim]"""
        x = self.encoder(volume_data)
        conv_out = self.volume_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + conv_out)


class RegimeEncoder(nn.Module):
    """
    Encode market-regime context: session flags + ATR percentile.
    Injects external temporal context that the price encoders cannot infer.

    Input (per bar): [tokyo, london, newyork, atr_pct]  →  4 features
    """

    def __init__(self, wave_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, wave_dim),
        )

    def forward(self, regime: Tensor) -> Tensor:
        """regime: [B, T, 4]  →  [B, T, wave_dim]"""
        return self.encoder(regime)


# ─────────────────────────────────────────────────────────────────────────────
# Causal Wave Chainer (CWC)
# ─────────────────────────────────────────────────────────────────────────────

class CausalWaveChainer(nn.Module):
    """
    Augments the fused wave with learned causal direction features and
    sinusoidal positional encoding, then applies causal self-attention.

    This is the adaptation of the CWC from FLUX-LM to the trading domain.
    It ensures temporal ordering is baked into the representation before the
    WavePredictor makes its signal prediction.
    """

    def __init__(
        self,
        wave_dim:   int = 432,
        causal_dim: int = 176,
        hidden_dim: int = 512,
        n_heads:    int = 8,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.wave_dim   = wave_dim
        self.causal_dim = causal_dim
        self.output_dim = wave_dim + causal_dim

        self.direction_encoder = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, causal_dim),
        )

        self.causal_attn = nn.MultiheadAttention(
            self.output_dim, n_heads, batch_first=True, dropout=dropout
        )

        self.register_buffer(
            "pos_enc", self._build_pos_encoding(1024, self.output_dim)
        )
        self.norm = nn.LayerNorm(self.output_dim)

    def _build_pos_encoding(self, max_len: int, d_model: int) -> Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # [1, max_len, d_model]

    def forward(self, wave: Tensor) -> Tensor:
        """wave: [B, T, wave_dim]  →  [B, T, wave_dim + causal_dim]"""
        causal_feats = self.direction_encoder(wave)
        x = torch.cat([wave, causal_feats], dim=-1)
        x = x + self.pos_enc[:, : x.size(1), :]

        mask = create_causal_mask(x.size(1), x.device)
        attn_out, _ = self.causal_attn(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# Regime-Gated Layer (FiLM conditioning)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeGatedLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) — conditions a wave tensor on
    externally supplied regime features (e.g. ATR percentile, session flags).

    The model learns *how* to modulate itself per regime end-to-end; no manual
    hyperparameter tuning or runtime dropout mutation required.

    Reference: Perez et al. (2018), "FiLM: Visual Reasoning with a General
    Conditioning Layer."
    """

    def __init__(self, dim: int, n_features: int = 4) -> None:
        super().__init__()
        self.gain = nn.Linear(n_features, dim)
        self.bias = nn.Linear(n_features, dim)
        # Initialise close to identity so early training is stable
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)

    def forward(self, x: Tensor, regime_features: Tensor) -> Tensor:
        """
        x:               [B, dim]  — wave vector to modulate
        regime_features: [B, n_features]  — regime context
        Returns:         [B, dim]
        """
        gamma = self.gain(regime_features)   # [B, dim]
        beta  = self.bias(regime_features)   # [B, dim]
        return x * (1.0 + gamma) + beta


# ─────────────────────────────────────────────────────────────────────────────
# Candle Pattern Encoder (v3) — specialised for 3-candle reversal detection
# ─────────────────────────────────────────────────────────────────────────────

class CandlePatternEncoder(nn.Module):
    """
    Encode a window of 4H candles with special attention to the 3-candle
    reversal pattern.

    Input channels per bar:
      - OHLCV (5) + pattern features (8) + structure (8) = 21 features
      - Or just OHLCV (5) + pattern (8) = 13 if structure not provided

    Architecture:
      1. Linear projection of per-bar features → hidden_dim
      2. Causal conv stack (kernel=3) to capture local candle interactions
      3. Causal self-attention over the full lookback window
      4. Last 3 positions (the reversal window) get special attention via
         a small cross-attention from a learned "pattern query"
      5. Output: pattern embedding [B, wave_dim] + pattern_confidence [B, 1]
    """

    def __init__(
        self,
        wave_dim:     int = 128,
        ohlcv_dim:    int = 5,
        pattern_dim:  int = 8,
        structure_dim: int = 8,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.wave_dim = wave_dim
        in_dim = ohlcv_dim + pattern_dim + structure_dim  # 21

        # Per-bar projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, wave_dim),
        )

        # Causal conv stack for local pattern interactions
        self.conv_stack = nn.Sequential(
            CausalConv1d(wave_dim, wave_dim, 3),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(wave_dim, wave_dim, 3),
            nn.GELU(),
        )

        # Full-sequence causal self-attention
        self.temporal_attn = nn.MultiheadAttention(
            wave_dim, num_heads=4, batch_first=True, dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(wave_dim)

        # Pattern query — learned queries that attend to the last 3 positions
        self.pattern_query = nn.Parameter(torch.randn(1, 1, wave_dim) * 0.02)
        self.pattern_attn = nn.MultiheadAttention(
            wave_dim, num_heads=4, batch_first=True, dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(wave_dim)

        # Pattern confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(wave_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ohlcv:     Tensor,
        pattern:   Tensor,
        structure: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        ohlcv:     [B, T, 5]   — normalised OHLCV
        pattern:   [B, T, 8]   — reversal pattern features from detect_reversal_pattern
        structure: [B, T, 8]   — market structure features

        Returns:
            embedding:  [B, wave_dim]  — pattern-aware representation
            confidence: [B, 1]         — how confident the encoder is about a pattern
        """
        B, T, _ = ohlcv.shape

        # Concatenate all inputs
        x = torch.cat([ohlcv, pattern, structure], dim=-1)  # [B, T, 21]
        x = self.input_proj(x)                               # [B, T, wave_dim]

        # Causal conv for local patterns
        conv_out = self.conv_stack(x.transpose(1, 2)).transpose(1, 2)
        x = x + conv_out

        # Full-sequence causal self-attention
        mask = create_causal_mask(T, x.device)
        attn_out, _ = self.temporal_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Pattern query attends to last 3 bars (the reversal window)
        last_3 = x[:, -3:, :]  # [B, 3, wave_dim]
        query = self.pattern_query.expand(B, -1, -1)  # [B, 1, wave_dim]
        pattern_out, _ = self.pattern_attn(query, last_3, last_3)
        pattern_vec = self.norm2(pattern_out.squeeze(1))  # [B, wave_dim]

        confidence = self.confidence_head(pattern_vec)  # [B, 1]

        return pattern_vec, confidence
