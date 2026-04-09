"""
Full model definitions:
  Single-TF : FluxSignal
  Multi-TF  : WaveTraderMTF
  Multi-Pair: FluxSignalFabric (cross-pair attention + FiLM regime gating)
  Shared    : WavePredictor, SignalHead, CrossPairAttention
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MTFConfig, SignalConfig, DEFAULT_RISK_SCALING
from .encoders import (
    CausalWaveChainer,
    CandlePatternEncoder,
    PriceWaveEncoder,
    RegimeEncoder,
    RegimeGatedLayer,
    RSIWaveEncoder,
    StructureWaveEncoder,
    VolumeWaveEncoder,
    CausalConv1d,
    create_causal_mask,
)
from .types import Signal, TradeSignal

_DEFAULT_SINGLE_CONFIG = SignalConfig()
_DEFAULT_MTF_CONFIG    = MTFConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Shared transformer backbone
# ─────────────────────────────────────────────────────────────────────────────

class WavePredictorBlock(nn.Module):
    """Pre-LN transformer block with causal self-attention."""

    def __init__(
        self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn      = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        mask = create_causal_mask(x.size(1), x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.attn_norm(x + attn_out)
        return self.ff_norm(x + self.ff(x))


class WavePredictor(nn.Module):
    """
    Transformer that maps (causal) wave sequence → next-state wave sequence.
    Projects to predictor_hidden internally; projects back to input_dim on output.
    """

    def __init__(self, config: SignalConfig) -> None:
        super().__init__()
        input_dim  = config.output_wave_dim
        hidden_dim = config.predictor_hidden

        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            WavePredictorBlock(
                hidden_dim,
                config.predictor_heads,
                config.predictor_ff_dim,
                config.dropout,
            )
            for _ in range(config.predictor_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.norm        = nn.LayerNorm(input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, input_dim]  →  [B, T, input_dim]"""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(self.output_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Signal output head
# ─────────────────────────────────────────────────────────────────────────────

class SignalHead(nn.Module):
    """
    Maps the last-position wave vector to:
      - signal_logits  [B, 3]       BUY / SELL / HOLD
      - confidence     [B, 1]       calibrated probability
      - risk_params    [B, 3]       SL / TP / trailing_pct (all positive)
    """

    def __init__(self, wave_dim: int = 608, dropout: float = 0.1) -> None:
        super().__init__()

        self.signal_classifier = nn.Sequential(
            nn.Linear(wave_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softplus(),   # ensures SL/TP/trailing > 0
        )

    def forward(self, wave: Tensor) -> Dict[str, Tensor]:
        if wave.dim() == 3:
            wave = wave[:, -1, :]   # use last position only
        return {
            "signal_logits": self.signal_classifier(wave),
            "confidence":    self.confidence_head(wave),
            "risk_params":   self.risk_head(wave),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Single-timeframe model: FluxSignal
# ─────────────────────────────────────────────────────────────────────────────

class FluxSignal(nn.Module):
    """
    FLUX-Signal — single-timeframe wave trading model.

    Inputs per bar:
      ohlcv     [B, T, 5]
      structure [B, T, 8]
      rsi       [B, T, 3]
      volume    [B, T, 3]
      regime    [B, T, 4]  (optional: session flags + ATR percentile)

    Outputs: signal_logits, confidence, risk_params
    """

    def __init__(self, config: Optional[SignalConfig] = None) -> None:
        super().__init__()
        self.config = config or _DEFAULT_SINGLE_CONFIG
        cfg = self.config

        # Wave encoders
        self.price_encoder     = PriceWaveEncoder(cfg.price_wave_dim, cfg.dropout)
        self.structure_encoder = StructureWaveEncoder(cfg.structure_wave_dim, cfg.dropout)
        self.rsi_encoder       = RSIWaveEncoder(cfg.rsi_wave_dim, cfg.dropout)
        self.volume_encoder    = VolumeWaveEncoder(cfg.volume_wave_dim, cfg.dropout)
        self.regime_encoder    = RegimeEncoder(wave_dim=16, dropout=cfg.dropout)

        # Fusion — note: total_input_dim + 16 (regime)
        fuse_in = cfg.total_input_dim + 16
        self.wave_fusion = nn.Sequential(
            nn.Linear(fuse_in, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.cwc       = CausalWaveChainer(cfg.fused_wave_dim, cfg.causal_dim, dropout=cfg.dropout)
        self.predictor = WavePredictor(cfg)
        self.signal_head = SignalHead(cfg.output_wave_dim, cfg.dropout)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        batch keys: ohlcv, structure, rsi, volume
                    optionally: regime  [B, T, 4]
        """
        price_wave     = self.price_encoder(batch["ohlcv"])
        structure_wave = self.structure_encoder(batch["structure"])
        rsi_wave       = self.rsi_encoder(batch["rsi"])
        volume_wave    = self.volume_encoder(batch["volume"])

        parts = [price_wave, structure_wave, rsi_wave, volume_wave]

        if "regime" in batch:
            regime_wave = self.regime_encoder(batch["regime"])
            parts.append(regime_wave)
        else:
            # Zero-out regime when not provided (graceful degradation)
            B, T, _ = price_wave.shape
            parts.append(torch.zeros(B, T, 16, device=price_wave.device))

        combined  = torch.cat(parts, dim=-1)
        fused     = self.wave_fusion(combined)
        causal    = self.cwc(fused)
        predicted = self.predictor(causal)
        return self.signal_head(predicted)

    def _encode_wave(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Run encoders + CWC + predictor and return the last-position wave vector.
        Shape: [B, output_wave_dim]  (no signal head applied).
        Used by FluxSignalFabric to collect per-pair wave states before
        cross-pair attention.
        """
        price_wave     = self.price_encoder(batch["ohlcv"])
        structure_wave = self.structure_encoder(batch["structure"])
        rsi_wave       = self.rsi_encoder(batch["rsi"])
        volume_wave    = self.volume_encoder(batch["volume"])

        parts = [price_wave, structure_wave, rsi_wave, volume_wave]
        if "regime" in batch:
            parts.append(self.regime_encoder(batch["regime"]))
        else:
            B, T, _ = price_wave.shape
            parts.append(torch.zeros(B, T, 16, device=price_wave.device))

        combined  = torch.cat(parts, dim=-1)
        fused     = self.wave_fusion(combined)
        causal    = self.cwc(fused)
        predicted = self.predictor(causal)
        return predicted[:, -1, :]   # [B, output_wave_dim]

    def predict(self, batch: Dict[str, Tensor]) -> TradeSignal:
        """Single-sample inference → TradeSignal."""
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)
            signal_idx = out["signal_logits"].argmax(-1).item()
            confidence = out["confidence"].item()
            risk       = out["risk_params"][0]
            _rs = DEFAULT_RISK_SCALING
            return TradeSignal(
                signal=Signal(signal_idx),
                confidence=confidence,
                entry_price=0.0,
                stop_loss=_rs.sl_pips(float(risk[0].item())),
                take_profit=_rs.tp_pips(float(risk[1].item())),
                trailing_stop_pct=_rs.trailing_pct(float(risk[2].item())),
                timestamp=datetime.now(),
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe components
# ─────────────────────────────────────────────────────────────────────────────

class TimeframeEncoder(nn.Module):
    """
    Compress one timeframe's [B, T, features] → [B, tf_wave_dim] summary vector.
    Uses the same four modality encoders + causal temporal self-attention.
    """

    def __init__(self, wave_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.price_conv = nn.Sequential(
            CausalConv1d(5, 64, 3),
            nn.GELU(),
            CausalConv1d(64, 128, 3),
            nn.GELU(),
            CausalConv1d(128, wave_dim // 2, 3),
        )
        self.structure_enc = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(), nn.Linear(64, wave_dim // 4)
        )
        self.rsi_enc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, wave_dim // 8)
        )
        self.volume_enc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, wave_dim // 8)
        )

        combined_dim = wave_dim // 2 + wave_dim // 4 + wave_dim // 8 + wave_dim // 8
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, wave_dim),
            nn.LayerNorm(wave_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_attn = nn.MultiheadAttention(
            wave_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(
        self,
        ohlcv:     Tensor,
        structure: Tensor,
        rsi:       Tensor,
        volume:    Tensor,
    ) -> Tensor:
        """All inputs [B, T, features]  →  [B, wave_dim]"""
        price_wave     = self.price_conv(ohlcv.transpose(1, 2)).transpose(1, 2)
        structure_wave = self.structure_enc(structure)
        rsi_wave       = self.rsi_enc(rsi)
        volume_wave    = self.volume_enc(volume)

        combined = torch.cat([price_wave, structure_wave, rsi_wave, volume_wave], dim=-1)
        fused    = self.fusion(combined)

        mask = create_causal_mask(fused.size(1), fused.device)
        attn_out, _ = self.temporal_attn(fused, fused, fused, attn_mask=mask)
        fused = self.norm(fused + attn_out)

        return fused[:, -1, :]   # [B, wave_dim]


class MultiTimeframeFusion(nn.Module):
    """
    Cross-attention + weighted concat fusion of N timeframe vectors.

    The entry timeframe (index 0) attends to higher timeframes as context.
    A learned alignment score indicates how much all TFs agree — used to
    scale confidence at inference time.
    """

    def __init__(
        self,
        tf_wave_dim:  int = 256,
        output_dim:   int = 512,
        n_timeframes: int = 4,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()

        self.tf_weights = nn.Parameter(torch.ones(n_timeframes) / n_timeframes)

        self.cross_attn = nn.MultiheadAttention(
            tf_wave_dim, num_heads=4, batch_first=True, dropout=dropout
        )

        self.fusion = nn.Sequential(
            nn.Linear(tf_wave_dim * n_timeframes, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.alignment_head = nn.Sequential(
            nn.Linear(tf_wave_dim * n_timeframes, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, tf_waves: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        tf_waves: list of [B, wave_dim] in order [entry, confirm, trend, bias]
        Returns: (fused [B, output_dim], alignment [B, 1])
        """
        batch = tf_waves[0].size(0)

        stacked = torch.stack(tf_waves, dim=1)          # [B, n_tf, wave_dim]
        entry   = tf_waves[0].unsqueeze(1)               # [B, 1, wave_dim]
        higher  = stacked[:, 1:, :]                      # [B, n_tf-1, wave_dim]

        attended, _ = self.cross_attn(entry, higher, higher)
        attended = attended.squeeze(1)                   # [B, wave_dim]

        weights  = F.softmax(self.tf_weights, dim=0)
        weighted = stacked * weights.view(1, -1, 1)
        concat   = weighted.view(batch, -1)              # [B, n_tf * wave_dim]

        fused     = self.fusion(concat)
        alignment = self.alignment_head(concat)
        return fused, alignment


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe model: WaveTraderMTF
# ─────────────────────────────────────────────────────────────────────────────

class WaveTraderMTF(nn.Module):
    """
    Multi-Timeframe WaveTrader.

    One TimeframeEncoder per TF → MultiTimeframeFusion → CWC → WavePredictor → SignalHead.
    Uses 4 TFs by default: 15min (entry), 1H (confirmation), 4H (trend), Daily (bias).
    """

    def __init__(self, config: Optional[MTFConfig] = None) -> None:
        super().__init__()
        self.config = config or _DEFAULT_MTF_CONFIG
        cfg = self.config

        self.tf_encoders = nn.ModuleDict({
            tf: TimeframeEncoder(cfg.tf_wave_dim, cfg.dropout)
            for tf in cfg.timeframes
        })

        self.mtf_fusion = MultiTimeframeFusion(
            tf_wave_dim=cfg.tf_wave_dim,
            output_dim=cfg.fused_wave_dim,
            n_timeframes=len(cfg.timeframes),
            dropout=cfg.dropout,
        )

        self.cwc = CausalWaveChainer(
            wave_dim=cfg.fused_wave_dim,
            causal_dim=176,
            dropout=cfg.dropout,
        )

        # Build a minimal SignalConfig to drive WavePredictor
        _pred_cfg = SignalConfig(
            fused_wave_dim=cfg.fused_wave_dim,
            causal_dim=176,
            predictor_hidden=cfg.predictor_hidden,
            predictor_heads=cfg.predictor_heads,
            predictor_layers=cfg.predictor_layers,
            predictor_ff_dim=cfg.predictor_ff_dim,
            dropout=cfg.dropout,
        )
        self.predictor   = WavePredictor(_pred_cfg)
        self.signal_head = SignalHead(cfg.output_wave_dim, cfg.dropout)

        self.last_alignment: Optional[Tensor] = None

    def forward(self, batch: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        batch: {tf: {ohlcv, structure, rsi, volume}, ...}
        """
        tf_waves = [
            self.tf_encoders[tf](
                batch[tf]["ohlcv"],
                batch[tf]["structure"],
                batch[tf]["rsi"],
                batch[tf]["volume"],
            )
            for tf in self.config.timeframes
        ]

        fused, alignment   = self.mtf_fusion(tf_waves)
        self.last_alignment = alignment

        fused  = fused.unsqueeze(1)         # [B, 1, fused_dim]
        causal = self.cwc(fused)
        output = self.signal_head(self.predictor(causal).squeeze(1))
        output["alignment"] = alignment
        return output

    def predict(self, batch: Dict[str, Dict[str, Tensor]]) -> TradeSignal:
        """Single-sample inference → TradeSignal."""
        self.eval()
        with torch.no_grad():
            out       = self.forward(batch)
            signal_idx = out["signal_logits"].argmax(-1).item()
            base_conf  = out["confidence"].item()
            alignment  = out["alignment"].item()
            confidence = base_conf * (0.5 + 0.5 * alignment)

            risk = out["risk_params"][0]
            _rs = DEFAULT_RISK_SCALING
            return TradeSignal(
                signal=Signal(signal_idx),
                confidence=confidence,
                entry_price=0.0,
                stop_loss=_rs.sl_pips(float(risk[0].item())),
                take_profit=_rs.tp_pips(float(risk[1].item())),
                trailing_stop_pct=_rs.trailing_pct(float(risk[2].item())),
                timestamp=datetime.now(),
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe model v2: WaveTraderMTFv2
# ─────────────────────────────────────────────────────────────────────────────

class TimeframeEncoderV2(nn.Module):
    """
    v2 TimeframeEncoder: can return either the last-position summary [B, dim]
    or the full temporal sequence [B, T, dim] (for CWC temporal processing).
    """

    def __init__(self, wave_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.wave_dim = wave_dim

        self.price_conv = nn.Sequential(
            CausalConv1d(5, 64, 3),
            nn.GELU(),
            CausalConv1d(64, 128, 3),
            nn.GELU(),
            CausalConv1d(128, wave_dim // 2, 3),
        )
        self.structure_enc = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(), nn.Linear(64, wave_dim // 4)
        )
        self.rsi_enc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, wave_dim // 8)
        )
        self.volume_enc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, wave_dim // 8)
        )

        combined_dim = wave_dim // 2 + wave_dim // 4 + wave_dim // 8 + wave_dim // 8
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, wave_dim),
            nn.LayerNorm(wave_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_attn = nn.MultiheadAttention(
            wave_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(wave_dim)

    def forward(
        self,
        ohlcv:     Tensor,
        structure: Tensor,
        rsi:       Tensor,
        volume:    Tensor,
        return_sequence: bool = False,
    ) -> Tensor:
        """
        All inputs [B, T, features]
        Returns [B, wave_dim] if return_sequence=False (default)
        Returns [B, T, wave_dim] if return_sequence=True (for CWC)
        """
        price_wave     = self.price_conv(ohlcv.transpose(1, 2)).transpose(1, 2)
        structure_wave = self.structure_enc(structure)
        rsi_wave       = self.rsi_enc(rsi)
        volume_wave    = self.volume_enc(volume)

        combined = torch.cat([price_wave, structure_wave, rsi_wave, volume_wave], dim=-1)
        fused    = self.fusion(combined)

        mask = create_causal_mask(fused.size(1), fused.device)
        attn_out, _ = self.temporal_attn(fused, fused, fused, attn_mask=mask)
        fused = self.norm(fused + attn_out)

        if return_sequence:
            return fused            # [B, T, wave_dim]
        return fused[:, -1, :]      # [B, wave_dim]


class WaveTraderMTFv2(nn.Module):
    """
    WaveTrader Multi-Timeframe v2 — targeting 55-70% win rate.

    Key improvements over v1:
      1. CWC bottleneck fixed: entry-TF temporal sequence [B, T, 256] goes through
         CWC (not a squeezed single vector), preserving temporal reasoning.
      2. RegimeGatedLayer (FiLM): conditions fused wave on regime features
         (ATR + ADX + session + day-of-week).
      3. Compatible with v2 dataset: extra features, ATR-adaptive labels.

    Architecture:
      Entry TF encoder (full sequence) → CWC → last position
      Higher TF encoders (summaries) → concat with entry CWC output
      → MultiTimeframeFusion → RegimeGatedLayer (FiLM) → WavePredictor → SignalHead
    """

    def __init__(self, config=None) -> None:
        super().__init__()
        from .config import MTFv2Config
        self.config = config or MTFv2Config()
        cfg = self.config

        # Per-TF encoders (v2 encoder supports return_sequence)
        self.tf_encoders = nn.ModuleDict({
            tf: TimeframeEncoderV2(cfg.tf_wave_dim, cfg.dropout)
            for tf in cfg.timeframes
        })

        # CWC for entry-TF temporal sequence
        self.entry_cwc = CausalWaveChainer(
            wave_dim=cfg.tf_wave_dim,
            causal_dim=176,
            dropout=cfg.dropout,
        )
        entry_cwc_dim = cfg.tf_wave_dim + 176  # 256 + 176 = 432

        # Fusion: entry CWC output + higher-TF summaries via cross-attention
        self.mtf_fusion = MultiTimeframeFusion(
            tf_wave_dim=cfg.tf_wave_dim,
            output_dim=cfg.fused_wave_dim,
            n_timeframes=len(cfg.timeframes),
            dropout=cfg.dropout,
        )

        # Project entry CWC output to match fusion output for concat
        self.entry_proj = nn.Sequential(
            nn.Linear(entry_cwc_dim, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
            nn.GELU(),
        )

        # Merge entry CWC stream + fusion stream
        self.stream_merge = nn.Sequential(
            nn.Linear(cfg.fused_wave_dim * 2, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # FiLM regime gating (conditions on ATR + ADX + session + DOW)
        regime_dim = getattr(cfg, 'regime_dim', 7)
        self.use_regime_gating = getattr(cfg, 'use_regime_gating', True)
        if self.use_regime_gating:
            self.regime_gate = RegimeGatedLayer(cfg.fused_wave_dim, n_features=regime_dim)

        # Main CWC + predictor stack
        self.cwc = CausalWaveChainer(
            wave_dim=cfg.fused_wave_dim,
            causal_dim=176,
            dropout=cfg.dropout,
        )

        _pred_cfg = SignalConfig(
            fused_wave_dim=cfg.fused_wave_dim,
            causal_dim=176,
            predictor_hidden=cfg.predictor_hidden,
            predictor_heads=cfg.predictor_heads,
            predictor_layers=cfg.predictor_layers,
            predictor_ff_dim=cfg.predictor_ff_dim,
            dropout=cfg.dropout,
        )
        self.predictor   = WavePredictor(_pred_cfg)
        self.signal_head = SignalHead(cfg.output_wave_dim, cfg.dropout)

        self.last_alignment: Optional[Tensor] = None

    def forward(self, batch: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        batch: {tf: {ohlcv, structure, rsi, volume, regime(optional)}, ...}
        """
        entry_tf = self.config.entry_timeframe

        # 1. Entry-TF: get FULL temporal sequence, run through CWC
        entry_data = batch[entry_tf]
        entry_seq = self.tf_encoders[entry_tf](
            entry_data["ohlcv"],
            entry_data["structure"],
            entry_data["rsi"],
            entry_data["volume"],
            return_sequence=True,  # [B, T, tf_wave_dim]
        )
        entry_causal = self.entry_cwc(entry_seq)  # [B, T, tf_wave_dim + 176]
        entry_vec = entry_causal[:, -1, :]         # [B, entry_cwc_dim]
        entry_proj = self.entry_proj(entry_vec)    # [B, fused_wave_dim]

        # 2. All TFs: get summary vectors (last position) for fusion
        tf_waves = []
        for tf in self.config.timeframes:
            if tf == entry_tf:
                # Use last position from the sequence we already computed
                tf_waves.append(entry_seq[:, -1, :])
            else:
                tf_waves.append(
                    self.tf_encoders[tf](
                        batch[tf]["ohlcv"],
                        batch[tf]["structure"],
                        batch[tf]["rsi"],
                        batch[tf]["volume"],
                        return_sequence=False,
                    )
                )

        # 3. Multi-TF fusion
        fused, alignment   = self.mtf_fusion(tf_waves)
        self.last_alignment = alignment

        # 4. Merge entry CWC stream + fusion stream
        merged = self.stream_merge(torch.cat([entry_proj, fused], dim=-1))

        # 5. FiLM regime gating
        if self.use_regime_gating and "regime" in batch.get(entry_tf, {}):
            regime = batch[entry_tf]["regime"]
            if regime.dim() == 3:
                regime = regime[:, -1, :]  # [B, regime_dim]
            merged = self.regime_gate(merged, regime)

        # 6. Main CWC + predictor + signal head
        merged = merged.unsqueeze(1)        # [B, 1, fused_dim]
        causal = self.cwc(merged)
        output = self.signal_head(self.predictor(causal).squeeze(1))
        output["alignment"] = alignment
        return output

    def predict(self, batch: Dict[str, Dict[str, Tensor]]) -> TradeSignal:
        """Single-sample inference → TradeSignal."""
        self.eval()
        with torch.no_grad():
            out        = self.forward(batch)
            signal_idx = out["signal_logits"].argmax(-1).item()
            base_conf  = out["confidence"].item()
            alignment  = out["alignment"].item()
            confidence = base_conf * (0.5 + 0.5 * alignment)

            risk = out["risk_params"][0]
            _rs = DEFAULT_RISK_SCALING
            return TradeSignal(
                signal=Signal(signal_idx),
                confidence=confidence,
                entry_price=0.0,
                stop_loss=_rs.sl_pips(float(risk[0].item())),
                take_profit=_rs.tp_pips(float(risk[1].item())),
                trailing_stop_pct=_rs.trailing_pct(float(risk[2].item())),
                timestamp=datetime.now(),
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-pair attention
# ─────────────────────────────────────────────────────────────────────────────

class CrossPairAttention(nn.Module):
    """
    Cross-pair attention: a primary trade pair attends to correlated peer pairs.

    Economically motivated — GBP/JPY shares yen-rate exposure with USD/JPY and
    EUR/JPY; sterling exposure with GBP/USD.  When the BoJ intervenes, all yen
    pairs move together.  This layer lets the model discover those correlations
    without hard-coded correlation matrices.

    Mechanism:
        Q = primary wave  [B, 1, dim]
        K = peer waves    [B, n_peers, dim]
        V = peer waves    [B, n_peers, dim]
        Output: gated residual addition of attended context onto primary wave.
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, primary: Tensor, peers: List[Tensor]) -> Tensor:
        """
        primary: [B, dim]
        peers:   list of [B, dim]  (variable length >= 1)
        Returns: [B, dim]
        """
        q          = self.q_proj(primary).unsqueeze(1)   # [B, 1, dim]
        peer_stack = torch.stack(peers, dim=1)            # [B, n_peers, dim]
        k          = self.k_proj(peer_stack)
        v          = self.v_proj(peer_stack)

        context, _ = self.cross_attn(q, k, v)
        context    = context.squeeze(1)                   # [B, dim]

        gate_val = self.gate(torch.cat([primary, context], dim=-1))
        return self.norm(primary + gate_val * context)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-pair model: FluxSignalFabric
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PEER_PAIRS = ["USD/JPY", "EUR/JPY", "GBP/USD"]


class FluxSignalFabric(nn.Module):
    """
    Cross-pair aware, regime-conditioned trading model (target: ~15M params).

    Architecture:
        For each pair (primary + peers):
            Shared encoders (price / structure / RSI / volume / regime)
            + pair identity injected as learned embedding at the fusion layer
            + shared CWC + WavePredictor
            → last-position wave vector [B, output_wave_dim]

        CrossPairAttention:
            Primary wave attends to peer waves  → enhanced primary wave

        RegimeGatedLayer (FiLM):
            Enhanced wave modulated by regime features (ATR percentile, session flags)

        SignalHead:
            → signal_logits  [B, 3]
            → confidence     [B, 1]
            → risk_params    [B, 3]

    Sharing encoder weights across pairs reduces parameter count ~3x vs fully
    independent encoders while still learning pair-specific representations via
    the pair embedding injection.

    Input format for forward():
        {
            "GBP/JPY": {"ohlcv": ..., "structure": ..., "rsi": ...,
                        "volume": ..., "regime": ...},
            "USD/JPY": {...},
            ...
        }
        Peer pairs absent from the dict are silently skipped (graceful degradation
        when only primary data is available at inference time).
    """

    def __init__(
        self,
        config:         Optional[SignalConfig] = None,
        peer_pairs:     List[str] = _DEFAULT_PEER_PAIRS,
        pair_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.config     = config or SignalConfig()
        cfg             = self.config
        self.peer_pairs = peer_pairs
        self.all_pairs  = [cfg.pair] + peer_pairs
        self.pair_index = {p: i for i, p in enumerate(self.all_pairs)}

        # ── Shared encoders (all pairs use the same weights) ──────────────
        self.price_encoder     = PriceWaveEncoder(cfg.price_wave_dim, cfg.dropout)
        self.structure_encoder = StructureWaveEncoder(cfg.structure_wave_dim, cfg.dropout)
        self.rsi_encoder       = RSIWaveEncoder(cfg.rsi_wave_dim, cfg.dropout)
        self.volume_encoder    = VolumeWaveEncoder(cfg.volume_wave_dim, cfg.dropout)
        self.regime_encoder    = RegimeEncoder(wave_dim=16, dropout=cfg.dropout)

        # ── Pair identity embedding → injected at fusion ──────────────────
        n_pairs = len(self.all_pairs)
        self.pair_embedding = nn.Embedding(n_pairs, pair_embed_dim)

        fuse_in = cfg.total_input_dim + 16 + pair_embed_dim
        self.wave_fusion = nn.Sequential(
            nn.Linear(fuse_in, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # ── Shared CWC + predictor ────────────────────────────────────────
        self.cwc       = CausalWaveChainer(cfg.fused_wave_dim, cfg.causal_dim, dropout=cfg.dropout)
        self.predictor = WavePredictor(cfg)

        # ── Cross-pair attention ──────────────────────────────────────────
        self.cross_pair_attn = CrossPairAttention(
            dim=cfg.output_wave_dim, n_heads=8, dropout=cfg.dropout
        )

        # ── FiLM regime gating ────────────────────────────────────────────
        self.regime_gate = RegimeGatedLayer(cfg.output_wave_dim, n_features=4)

        # ── Signal head ───────────────────────────────────────────────────
        self.signal_head = SignalHead(cfg.output_wave_dim, cfg.dropout)

    def _encode_pair(self, batch: Dict[str, Tensor], pair: str) -> Tensor:
        """Encode one pair's batch using shared weights → [B, output_wave_dim]."""
        device = batch["ohlcv"].device
        B, T   = batch["ohlcv"].shape[:2]

        price_wave     = self.price_encoder(batch["ohlcv"])
        structure_wave = self.structure_encoder(batch["structure"])
        rsi_wave       = self.rsi_encoder(batch["rsi"])
        volume_wave    = self.volume_encoder(batch["volume"])

        parts = [price_wave, structure_wave, rsi_wave, volume_wave]
        if "regime" in batch:
            parts.append(self.regime_encoder(batch["regime"]))
        else:
            parts.append(torch.zeros(B, T, 16, device=device))

        combined = torch.cat(parts, dim=-1)       # [B, T, total_input_dim + 16]

        idx      = torch.tensor(
            [self.pair_index[pair]], dtype=torch.long, device=device
        ).expand(B)
        pair_emb = self.pair_embedding(idx).unsqueeze(1).expand(-1, T, -1)

        extended = torch.cat([combined, pair_emb], dim=-1)
        fused    = self.wave_fusion(extended)
        causal   = self.cwc(fused)
        pred     = self.predictor(causal)
        return pred[:, -1, :]   # [B, output_wave_dim]

    def forward(self, batches: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        primary      = self.config.pair
        primary_wave = self._encode_pair(batches[primary], primary)

        peer_waves = [
            self._encode_pair(batches[p], p)
            for p in self.peer_pairs
            if p in batches
        ]

        enhanced = (
            self.cross_pair_attn(primary_wave, peer_waves)
            if peer_waves
            else primary_wave
        )

        if "regime" in batches[primary]:
            regime_feats = batches[primary]["regime"][:, -1, :]   # [B, 4]
            enhanced     = self.regime_gate(enhanced, regime_feats)

        return self.signal_head(enhanced)

    def predict(self, batches: Dict[str, Dict[str, Tensor]]) -> TradeSignal:
        """Single-sample inference → TradeSignal."""
        self.eval()
        with torch.no_grad():
            out        = self.forward(batches)
            signal_idx = out["signal_logits"].argmax(-1).item()
            confidence = out["confidence"].item()
            risk       = out["risk_params"][0]
            _rs = DEFAULT_RISK_SCALING
            return TradeSignal(
                signal=Signal(signal_idx),
                confidence=confidence,
                entry_price=0.0,
                stop_loss=_rs.sl_pips(float(risk[0].item())),
                take_profit=_rs.tp_pips(float(risk[1].item())),
                trailing_stop_pct=_rs.trailing_pct(float(risk[2].item())),
                timestamp=datetime.now(),
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Signal output head v3 — no TP output
# ─────────────────────────────────────────────────────────────────────────────

class SignalHeadV3(nn.Module):
    """
    V3 signal head — outputs SL + trailing only (no TP).

    Maps the last-position wave vector to:
      - signal_logits  [B, 3]       BUY / SELL / HOLD
      - confidence     [B, 1]       calibrated probability
      - risk_params    [B, 2]       SL / trailing_pct (positive values)

    No take-profit output — V3 exits on opposite signal or trailing SL only.
    """

    def __init__(self, wave_dim: int = 608, dropout: float = 0.1) -> None:
        super().__init__()

        self.signal_classifier = nn.Sequential(
            nn.Linear(wave_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),     # Only SL + trailing (no TP)
            nn.Softplus(),
        )

    def forward(self, wave: Tensor) -> Dict[str, Tensor]:
        if wave.dim() == 3:
            wave = wave[:, -1, :]
        return {
            "signal_logits": self.signal_classifier(wave),
            "confidence":    self.confidence_head(wave),
            "risk_params":   self.risk_head(wave),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe model v3: WaveTraderMTFv3 — 4H Reversal Swing Model
# ─────────────────────────────────────────────────────────────────────────────

class WaveTraderMTFv3(nn.Module):
    """
    WaveTrader Multi-Timeframe v3 — 4H reversal swing trading model.

    Detects 3-candle reversal patterns on 4H, confirmed by Daily trend,
    with 1H entry timing. Holds until opposite signal fires.

    Architecture:
      1. Daily Trend Encoder (TimeframeEncoderV2): Daily bars → trend context [B, dim]
      2. 4H Pattern Encoder (CandlePatternEncoder): Last N 4H bars → reversal pattern
         embedding [B, pattern_dim] + pattern_confidence [B, 1]
      3. 4H Sequence Encoder (TimeframeEncoderV2): Full 4H history → context [B, dim]
      4. 1H Entry Encoder (TimeframeEncoderV2, return_sequence=True): 1H bars →
         entry timing [B, T, dim] → CWC → [B, entry_cwc_dim]
      5. Pattern-Guided Fusion: Cross-attention where pattern embedding queries
         all TF summaries + entry CWC
      6. RegimeGatedLayer (FiLM conditioning)
      7. Main CWC + WavePredictor
      8. SignalHeadV3: signal_logits + confidence + risk_params (SL + trailing only)
    """

    def __init__(self, config=None) -> None:
        super().__init__()
        from .config import MTFv3Config
        self.config = config or MTFv3Config()
        cfg = self.config

        # Per-TF encoders (v2 encoder supports return_sequence)
        self.tf_encoders = nn.ModuleDict({
            tf: TimeframeEncoderV2(cfg.tf_wave_dim, cfg.dropout)
            for tf in cfg.timeframes
        })

        # 4H Pattern Encoder — specialised for reversal detection
        self.pattern_encoder = CandlePatternEncoder(
            wave_dim=cfg.pattern_wave_dim,
            dropout=cfg.dropout,
        )

        # CWC for entry-TF (1H) temporal sequence
        self.entry_cwc = CausalWaveChainer(
            wave_dim=cfg.tf_wave_dim,
            causal_dim=176,
            dropout=cfg.dropout,
        )
        entry_cwc_dim = cfg.tf_wave_dim + 176  # 256 + 176 = 432

        # Project entry CWC to match fusion dim
        self.entry_proj = nn.Sequential(
            nn.Linear(entry_cwc_dim, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
            nn.GELU(),
        )

        # Project pattern embedding to match fusion dim
        self.pattern_proj = nn.Sequential(
            nn.Linear(cfg.pattern_wave_dim, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
            nn.GELU(),
        )

        # Multi-source fusion: entry CWC + 4H context + daily context + pattern
        # Uses cross-attention where pattern queries all other sources
        n_sources = 4  # entry_proj, 4h_summary, daily_summary, pattern_proj
        self.fusion_attn = nn.MultiheadAttention(
            cfg.fused_wave_dim, num_heads=8, batch_first=True, dropout=cfg.dropout,
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(cfg.fused_wave_dim * n_sources, cfg.fused_wave_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fused_wave_dim * 2, cfg.fused_wave_dim),
            nn.LayerNorm(cfg.fused_wave_dim),
        )

        # Alignment head
        self.alignment_head = nn.Sequential(
            nn.Linear(cfg.fused_wave_dim * n_sources, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Project higher-TF summaries to fused_wave_dim
        self.tf_proj = nn.ModuleDict({
            tf: nn.Linear(cfg.tf_wave_dim, cfg.fused_wave_dim)
            for tf in cfg.timeframes if tf != cfg.entry_timeframe
        })

        # FiLM regime gating
        regime_dim = getattr(cfg, 'regime_dim', 7)
        self.use_regime_gating = getattr(cfg, 'use_regime_gating', True)
        if self.use_regime_gating:
            self.regime_gate = RegimeGatedLayer(cfg.fused_wave_dim, n_features=regime_dim)

        # Main CWC + predictor
        self.cwc = CausalWaveChainer(
            wave_dim=cfg.fused_wave_dim,
            causal_dim=176,
            dropout=cfg.dropout,
        )

        _pred_cfg = SignalConfig(
            fused_wave_dim=cfg.fused_wave_dim,
            causal_dim=176,
            predictor_hidden=cfg.predictor_hidden,
            predictor_heads=cfg.predictor_heads,
            predictor_layers=cfg.predictor_layers,
            predictor_ff_dim=cfg.predictor_ff_dim,
            dropout=cfg.dropout,
        )
        self.predictor = WavePredictor(_pred_cfg)
        self.signal_head = SignalHeadV3(cfg.output_wave_dim, cfg.dropout)

        self.last_alignment: Optional[Tensor] = None

    def forward(self, batch: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        batch: {
            "1h":  {ohlcv, structure, rsi, volume, regime},
            "4h":  {ohlcv, structure, rsi, volume, regime, pattern},
            "1d":  {ohlcv, structure, rsi, volume, regime, trend},
        }
        """
        entry_tf = self.config.entry_timeframe  # "1h"

        # 1. Entry-TF (1H): full temporal sequence → CWC → last position
        entry_data = batch[entry_tf]
        entry_seq = self.tf_encoders[entry_tf](
            entry_data["ohlcv"],
            entry_data["structure"],
            entry_data["rsi"],
            entry_data["volume"],
            return_sequence=True,
        )
        entry_causal = self.entry_cwc(entry_seq)    # [B, T, cwc_dim]
        entry_vec = entry_causal[:, -1, :]           # [B, entry_cwc_dim]
        entry_proj = self.entry_proj(entry_vec)      # [B, fused_wave_dim]

        # 2. 4H Pattern Encoder — uses OHLCV + pattern features + structure
        h4_data = batch["4h"]
        h4_pattern_feat = h4_data.get("pattern", torch.zeros(
            h4_data["ohlcv"].shape[0], h4_data["ohlcv"].shape[1], 8,
            device=h4_data["ohlcv"].device,
        ))
        pattern_vec, pattern_conf = self.pattern_encoder(
            h4_data["ohlcv"], h4_pattern_feat, h4_data["structure"],
        )
        pattern_proj = self.pattern_proj(pattern_vec)  # [B, fused_wave_dim]

        # 3. Higher-TF summary vectors (4H context, Daily context)
        tf_summaries = {}
        for tf in self.config.timeframes:
            if tf == entry_tf:
                continue
            tf_data = batch[tf]
            summary = self.tf_encoders[tf](
                tf_data["ohlcv"],
                tf_data["structure"],
                tf_data["rsi"],
                tf_data["volume"],
                return_sequence=False,
            )
            tf_summaries[tf] = self.tf_proj[tf](summary)  # [B, fused_wave_dim]

        # 4. Fusion: stack all sources and use cross-attention
        B = entry_proj.size(0)
        sources = [entry_proj]
        for tf in self.config.timeframes:
            if tf != entry_tf and tf in tf_summaries:
                sources.append(tf_summaries[tf])
        sources.append(pattern_proj)

        # Stack for cross-attention: pattern queries the rest
        stacked = torch.stack(sources, dim=1)           # [B, n_sources, fused_dim]
        query = pattern_proj.unsqueeze(1)                # [B, 1, fused_dim]
        attended, _ = self.fusion_attn(query, stacked, stacked)
        attended = attended.squeeze(1)                    # [B, fused_dim]

        # Weighted concat fusion
        concat = stacked.view(B, -1)                     # [B, n_sources * fused_dim]
        fused = self.fusion_proj(concat)                  # [B, fused_wave_dim]
        alignment = self.alignment_head(concat)           # [B, 1]
        self.last_alignment = alignment

        # 5. FiLM regime gating
        if self.use_regime_gating and "regime" in batch.get(entry_tf, {}):
            regime = batch[entry_tf]["regime"]
            if regime.dim() == 3:
                regime = regime[:, -1, :]
            fused = self.regime_gate(fused, regime)

        # 6. Main CWC + predictor + signal head
        fused = fused.unsqueeze(1)                        # [B, 1, fused_dim]
        causal = self.cwc(fused)
        output = self.signal_head(self.predictor(causal).squeeze(1))
        output["alignment"] = alignment
        output["pattern_confidence"] = pattern_conf
        return output

    def predict(self, batch: Dict[str, Dict[str, Tensor]]) -> TradeSignal:
        """Single-sample inference → TradeSignal (no TP, opposite-signal exit)."""
        self.eval()
        with torch.no_grad():
            out        = self.forward(batch)
            signal_idx = out["signal_logits"][0].argmax(-1).item()
            base_conf  = out["confidence"][0].item()
            alignment  = out["alignment"][0].item()
            pat_conf   = out["pattern_confidence"][0].item()

            # Confidence = base × alignment × pattern_confidence
            confidence = base_conf * (0.5 + 0.5 * alignment) * (0.5 + 0.5 * pat_conf)

            risk = out["risk_params"][0]
            _rs = DEFAULT_RISK_SCALING
            return TradeSignal(
                signal=Signal(signal_idx),
                confidence=confidence,
                entry_price=0.0,
                stop_loss=_rs.sl_pips(float(risk[0].item())),
                take_profit=0.0,  # No TP for V3
                trailing_stop_pct=max(
                    _rs.trailing_pct(float(risk[1].item())),
                    self.config.default_trailing_pct,
                ),
                timestamp=datetime.now(),
                exit_mode="opposite_signal",
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
