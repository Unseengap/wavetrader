"""
Configuration classes for WaveTrader.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SignalConfig:
    """Configuration for single-timeframe FLUX-Signal model."""
    # Wave dimensions
    price_wave_dim: int = 128
    structure_wave_dim: int = 64
    rsi_wave_dim: int = 32
    volume_wave_dim: int = 32
    fused_wave_dim: int = 432
    causal_dim: int = 176       # Added by CWC

    # Model architecture — start small (2-4 layers) for tabular time-series
    predictor_hidden: int = 512
    predictor_heads: int = 8
    predictor_layers: int = 4   # Reduced from 8; prevents overfitting on low SNR data
    predictor_ff_dim: int = 2048

    # Sequence
    lookback: int = 100         # Bars of history fed to model

    # Training
    dropout: float = 0.2        # Higher dropout than default to combat overfitting
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50

    # Trading
    pair: str = "GBP/JPY"
    timeframe: str = "15min"

    @property
    def total_input_dim(self) -> int:
        return (
            self.price_wave_dim
            + self.structure_wave_dim
            + self.rsi_wave_dim
            + self.volume_wave_dim
        )

    @property
    def output_wave_dim(self) -> int:
        return self.fused_wave_dim + self.causal_dim  # 608


@dataclass
class MTFConfig:
    """Configuration for Multi-Timeframe WaveTrader."""
    # Timeframes ordered from entry (fastest) to bias (slowest)
    timeframes: List[str] = field(
        default_factory=lambda: ["15min", "1h", "4h", "1d"]
    )
    lookbacks: Dict[str, int] = field(
        default_factory=lambda: {
            "15min": 100,   # ~25 hours
            "1h":    100,   # ~4 days
            "4h":    100,   # ~16 days
            "1d":    50,    # ~2.5 months
        }
    )

    # Wave dimensions
    tf_wave_dim: int = 256
    fused_wave_dim: int = 512

    # Model architecture
    predictor_hidden: int = 512
    predictor_heads: int = 8
    predictor_layers: int = 4   # Lean; deeper is not better on tabular FX data
    predictor_ff_dim: int = 2048

    # Training
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 16        # Smaller: each sample contains 4× timeframe data
    epochs: int = 50

    # Trading
    pair: str = "GBP/JPY"
    entry_timeframe: str = "15min"

    @property
    def output_wave_dim(self) -> int:
        return self.fused_wave_dim + 176  # fused + causal dim


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""
    initial_balance: float = 10_000.0   # USD
    risk_per_trade: float = 0.02        # 2 % of balance per trade
    leverage: float = 30.0              # 30:1 standard retail forex
    spread_pips: float = 2.0            # GBP/JPY typical spread
    commission_per_lot: float = 7.0     # USD per standard lot round-trip
    pip_value: float = 6.5              # USD per pip per std lot (GBP/JPY approx)
    min_confidence: float = 0.60        # Skip signals below this confidence

    # Circuit breakers
    atr_halt_multiplier: float = 3.0    # Halt if current range > multiplier × 20-bar mean
    drawdown_reduce_threshold: float = 0.05  # Halve risk when drawdown exceeds this fraction


@dataclass
class ResonanceConfig:
    """Configuration for the ResonanceBuffer (episodic memory)."""
    capacity: int = 100                 # Max stored wave states
    wave_dim: int = 608                 # Must match model's output_wave_dim
    salience_threshold: float = 2.0     # Std deviations above mean |PnL| to qualify
    top_k: int = 5                      # States retrieved per inference


@dataclass
class SIConfig:
    """Synaptic Intelligence continual learning configuration."""
    si_lambda: float = 0.1             # Importance penalty weight (gentle for streaming FX)
    epsilon: float = 1e-3              # Denominator stability term
    consolidate_every: int = 500       # Batches between consolidation checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Unified risk scaling — single source of truth for SL/TP/trailing conversion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskScaling:
    """
    Maps raw model risk_params (Softplus output, ~0-1+ range) to pip values.
    Used by model.predict(), backtest.run_backtest(), and streaming.py.
    """
    sl_mult:    float = 50.0    # SL pips = risk[0] * sl_mult + sl_floor
    sl_floor:   float = 10.0
    tp_mult:    float = 100.0   # TP pips = risk[1] * tp_mult + tp_floor
    tp_floor:   float = 20.0
    trail_mult: float = 0.5     # trailing % = risk[2] * trail_mult

    def sl_pips(self, raw: float) -> float:
        return raw * self.sl_mult + self.sl_floor

    def tp_pips(self, raw: float) -> float:
        return raw * self.tp_mult + self.tp_floor

    def trailing_pct(self, raw: float) -> float:
        return raw * self.trail_mult


# Default risk scaling — matches model.py predict() (was mismatched in backtest.py)
DEFAULT_RISK_SCALING = RiskScaling()


# ─────────────────────────────────────────────────────────────────────────────
# MTF v2 Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MTFv2Config:
    """
    Configuration for WaveTraderMTFv2 — improved MTF model targeting 55-70% win rate.
    
    Key differences from MTFConfig:
      - ATR-adaptive labels (threshold = atr_k × ATR instead of fixed pips)
      - Extra features: ADX, day-of-week cyclical encoding
      - Regime gating via FiLM conditioning
      - Focal loss for training
      - Early stopping on BUY/SELL F1
      - ADX trend filter for trade execution
    """
    # Timeframes (same as v1)
    timeframes: List[str] = field(
        default_factory=lambda: ["15min", "1h", "4h", "1d"]
    )
    lookbacks: Dict[str, int] = field(
        default_factory=lambda: {
            "15min": 100,
            "1h":    100,
            "4h":    100,
            "1d":    50,
        }
    )

    # Wave dimensions
    tf_wave_dim: int = 256
    fused_wave_dim: int = 512

    # Model architecture
    predictor_hidden: int = 512
    predictor_heads: int = 8
    predictor_layers: int = 4
    predictor_ff_dim: int = 2048

    # Training
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    warmup_epochs: int = 5

    # Trading
    pair: str = "GBP/JPY"
    entry_timeframe: str = "15min"

    # ── v2 Label generation ───────────────────────────────────────────────
    label_atr_k: float = 1.5           # Label threshold = atr_k × ATR(14)
    label_lookahead: int = 10           # Bars ahead for label computation

    # ── v2 Extra features ─────────────────────────────────────────────────
    extra_features: List[str] = field(
        default_factory=lambda: ["adx", "dow"]  # ADX + day-of-week
    )
    regime_dim: int = 7                 # 4 base (session+atr_pct) + 1 ADX + 2 DOW

    # ── v2 Architecture flags ─────────────────────────────────────────────
    use_regime_gating: bool = True      # FiLM conditioning after fusion
    use_temporal_cwc: bool = True       # Pass entry-TF sequence through CWC (fix bottleneck)

    # ── v2 Training flags ─────────────────────────────────────────────────
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 0.3]
    )
    early_stopping_patience: int = 10   # On BUY/SELL weighted F1
    use_augmentation: bool = True
    augment_noise_prob: float = 0.3     # Probability of adding Gaussian noise
    augment_dropout_prob: float = 0.1   # Probability of zeroing a modality

    # ── v2 Backtest / execution ───────────────────────────────────────────
    adx_filter_threshold: float = 20.0  # Skip trades when ADX < this
    max_hold_bars: int = 50             # Close trade after this many bars
    risk_scaling: str = "fixed"         # "fixed" (default scaling) or "atr"

    @property
    def output_wave_dim(self) -> int:
        return self.fused_wave_dim + 176  # fused + causal dim


# ─────────────────────────────────────────────────────────────────────────────
# MTF v3 Configuration — 4H Reversal Swing Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MTFv3Config:
    """
    Configuration for WaveTraderMTFv3 — 4H reversal swing trading model.

    Strategy: Detect 3-candle reversal patterns on 4H, confirmed by Daily trend
    direction, with 1H entry timing. Holds trades until opposite reversal signal
    fires — no fixed TP, only trailing SL.

    Key differences from v1/v2:
      - 3 timeframes (1h, 4h, 1d) instead of 4
      - CandlePatternEncoder for 3-candle reversal detection on 4H
      - No take-profit output — exit on opposite signal or trailing SL only
      - Fewer but higher-conviction trades
    """
    # Timeframes: 1h (entry), 4h (confirmation/pattern), 1d (core trend)
    timeframes: List[str] = field(
        default_factory=lambda: ["1h", "4h", "1d"]
    )
    lookbacks: Dict[str, int] = field(
        default_factory=lambda: {
            "1h":  100,   # ~4 days of 1h bars
            "4h":   50,   # ~8 days of 4h bars
            "1d":   30,   # ~6 weeks of daily bars
        }
    )

    # Wave dimensions
    tf_wave_dim: int = 256
    fused_wave_dim: int = 512

    # Pattern encoder
    pattern_lookback: int = 3       # 3-candle reversal window
    pattern_wave_dim: int = 128     # Pattern embedding dim

    # Model architecture
    predictor_hidden: int = 512
    predictor_heads: int = 8
    predictor_layers: int = 4
    predictor_ff_dim: int = 2048

    # Training
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    warmup_epochs: int = 5

    # Trading
    pair: str = "GBP/JPY"
    entry_timeframe: str = "1h"

    # ── v3 Exit mode ──────────────────────────────────────────────────────
    exit_mode: str = "opposite_signal"  # "tp_sl" (v1/v2) or "opposite_signal"
    default_trailing_pct: float = 0.4   # Wider trail for swing trades

    # ── v3 Label generation ───────────────────────────────────────────────
    label_method: str = "reversal_pattern"
    label_atr_k: float = 1.5
    label_lookahead: int = 20      # Longer lookahead for swing trades
    trend_lookback: int = 10       # Daily bars for trend direction

    # ── v3 Extra features ─────────────────────────────────────────────────
    extra_features: List[str] = field(
        default_factory=lambda: ["adx", "dow"]
    )
    regime_dim: int = 7             # session(3) + atr_pct + adx + dow_sin + dow_cos

    # ── v3 Architecture flags ─────────────────────────────────────────────
    use_regime_gating: bool = True
    use_temporal_cwc: bool = True

    # ── v3 Training flags ─────────────────────────────────────────────────
    use_focal_loss: bool = True
    focal_gamma: float = 3.0        # Higher gamma — patterns are rare, focus on hard examples
    focal_alpha: List[float] = field(
        default_factory=lambda: [1.5, 1.5, 0.2]  # BUY/SELL weighted higher, HOLD suppressed
    )
    early_stopping_patience: int = 15  # More patience for sparse signals
    use_augmentation: bool = True
    augment_noise_prob: float = 0.3
    augment_dropout_prob: float = 0.1

    # ── v3 Backtest / execution ───────────────────────────────────────────
    adx_filter_threshold: float = 20.0
    max_hold_bars: int = 0          # 0 = no max hold (hold until opposite signal)
    risk_scaling: str = "fixed"

    @property
    def output_wave_dim(self) -> int:
        return self.fused_wave_dim + 176  # fused + causal dim
