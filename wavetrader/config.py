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
    cooldown_bars: int = 2              # Minimum bars between new trades (avoids churn)

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


@dataclass
class MeanRevConfig:
    """Hyperparameters for the MeanReversion model.

    Detects overextended price relative to rolling means and Bollinger-style
    bands, then trades the snap-back.  Higher win rate (~71 %) but smaller
    per-trade expectancy than trend-following.
    """

    # --- Timeframes ---
    timeframes: List[str] = field(
        default_factory=lambda: ["15min", "1h", "4h", "1d"]
    )
    lookbacks: Dict[str, int] = field(
        default_factory=lambda: {
            "15min": 120,
            "1h":    120,
            "4h":    100,
            "1d":     60,
        }
    )
    entry_timeframe: str = "15min"

    # --- Encoder ---
    tf_wave_dim: int = 192
    structure_emb_dim: int = 64
    price_conv_dim: int = 128

    # --- Fusion ---
    fused_dim: int = 384

    # --- Predictor ---
    predictor_hidden: int = 384
    predictor_heads: int = 6
    predictor_layers: int = 3
    predictor_ff_dim: int = 1536

    # --- Outputs ---
    n_signal_classes: int = 3       # BUY / SELL / HOLD

    # --- Mean-reversion specific ---
    bb_window: int = 20             # Bollinger Band lookback
    bb_std: float = 2.0             # Bollinger Band width (std devs)
    zscore_entry: float = 2.0       # Z-score threshold for entry
    zscore_exit: float = 0.5        # Z-score threshold for exit (snap-back)

    # --- Training ---
    dropout: float = 0.30
    learning_rate: float = 2e-4
    batch_size: int = 320
    epochs: int = 30

    # --- Pair ---
    pair: str = "GBP/JPY"

    @property
    def output_wave_dim(self) -> int:
        return self.fused_dim


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
    trail_mult: float = 0.2     # trailing % = risk[2] * trail_mult (reduced from 0.5)
    min_trail_pips: float = 20.0  # Minimum trail distance in pips (floor)

    def sl_pips(self, raw: float) -> float:
        return raw * self.sl_mult + self.sl_floor

    def tp_pips(self, raw: float) -> float:
        return raw * self.tp_mult + self.tp_floor

    def trailing_pct(self, raw: float) -> float:
        return raw * self.trail_mult


# Default risk scaling — matches model.py predict() (was mismatched in backtest.py)
DEFAULT_RISK_SCALING = RiskScaling()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Arbiter configuration (re-exported from llm_arbiter for convenience)
# ─────────────────────────────────────────────────────────────────────────────
from .llm_arbiter import LLMArbiterConfig  # noqa: E402
