"""
Base classes for all trading strategies.

Every strategy implements BaseStrategy and returns StrategySetup (or None)
on each bar evaluation.  The setup feeds into the AI confirmation layer
and then into the backtest engine or OANDA live execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..types import Signal


# ─────────────────────────────────────────────────────────────────────────────
# Strategy metadata — static info about the strategy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyMeta:
    """Immutable metadata for a strategy — displayed in dashboard."""
    id: str                     # "amd_session"
    name: str                   # "AMD Session Scalper"
    author: str                 # "Dectrick McGee"
    version: str                # "1.0.0"
    description: str            # One-liner
    category: str               # "scalper" | "swing" | "trend" | "mean_reversion"
    timeframes: List[str]       # Required TFs: ["5min", "15min", "1h", "4h"]
    pairs: List[str]            # Supported pairs: ["GBP/JPY", "EUR/JPY"]
    entry_timeframe: str = "15min"  # Which TF triggers entries


# ─────────────────────────────────────────────────────────────────────────────
# Indicator bundle — computed once per bar, shared across strategies
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorBundle:
    """All technical indicators computed for the current state.

    Populated by strategies.indicators.compute_all_indicators() and passed
    to every strategy's evaluate() method.  Each field is a dict keyed by
    timeframe (e.g. "5min", "1h") unless noted otherwise.
    """
    # ── Per-timeframe arrays (keyed by TF string) ────────────────────────
    rsi: Dict[str, np.ndarray] = field(default_factory=dict)          # (n,) RSI values
    atr: Dict[str, np.ndarray] = field(default_factory=dict)          # (n,) ATR values
    adx: Dict[str, np.ndarray] = field(default_factory=dict)          # (n,) ADX values
    structure: Dict[str, np.ndarray] = field(default_factory=dict)    # (n,8) market structure
    ema_20: Dict[str, np.ndarray] = field(default_factory=dict)       # (n,) EMA 20
    ema_50: Dict[str, np.ndarray] = field(default_factory=dict)       # (n,) EMA 50
    ema_200: Dict[str, np.ndarray] = field(default_factory=dict)      # (n,) EMA 200
    bollinger_upper: Dict[str, np.ndarray] = field(default_factory=dict)
    bollinger_lower: Dict[str, np.ndarray] = field(default_factory=dict)
    bollinger_mid: Dict[str, np.ndarray] = field(default_factory=dict)

    # ── Entry-TF specific features (from amd_features.py) ───────────────
    asian_range: Optional[np.ndarray] = None        # (n,5) — from compute_asian_range
    london_sweep: Optional[np.ndarray] = None       # (n,4) — from compute_london_sweep
    engulfing: Optional[np.ndarray] = None           # (n,3) — from compute_engulfing_patterns
    fair_value_gaps: Optional[np.ndarray] = None     # (n,4) — from compute_fair_value_gaps
    sr_zones: Optional[np.ndarray] = None            # (n,3) — from compute_sr_zones
    orb_features: Optional[np.ndarray] = None        # (n,4) — from compute_orb_features

    # ── Metadata ────────────────────────────────────────────────────────
    entry_tf: str = "15min"
    pair: str = "GBP/JPY"
    current_bar_idx: int = -1   # Index of the current bar being evaluated


# ─────────────────────────────────────────────────────────────────────────────
# Strategy setup — returned when a strategy detects a valid entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategySetup:
    """A concrete entry signal produced by a strategy.

    This is the strategy's "opinion" — direction, SL, TP, and reason.
    The AI confirmation layer may accept or reject it.  The SL/TP come
    from structural analysis (zones, levels) not statistical averages.
    """
    direction: Signal                # BUY or SELL (never HOLD)
    entry_price: float               # Current price at setup detection
    sl_pips: float                   # Stop loss in pips (structure-based)
    tp_pips: float                   # Take profit in pips (structure-based)
    confidence: float                # Strategy's own confidence 0.0–1.0
    reason: str                      # "London swept Asian low → bullish reversal at demand zone"
    trailing_stop_pct: float = 0.0   # Optional trailing (0 = no trail)
    exit_mode: str = "tp_sl"          # "tp_sl" | "geometric_trail"
    timestamp: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)  # Extra data for LLM narrative
    tp_levels: List[Tuple[float, float]] = field(default_factory=list)  # [(pips, fraction), ...]

    def __post_init__(self):
        if self.direction == Signal.HOLD:
            raise ValueError("StrategySetup direction cannot be HOLD")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter schema — describes tunable parameters for dashboard config panel
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamSpec:
    """Describes a single tunable parameter for dynamic UI generation."""
    name: str           # "asian_range_max_pips"
    label: str          # "Max Asian Range (pips)"
    type: str           # "float" | "int" | "bool"
    default: Any        # 80.0
    min_val: Any = None # 20.0
    max_val: Any = None # 200.0
    step: Any = None    # 5.0
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Base strategy — abstract class all strategies implement
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Subclasses MUST:
      1. Set `meta` class attribute (StrategyMeta)
      2. Implement `evaluate()` → StrategySetup | None
      3. Implement `param_schema()` → list of ParamSpec
      4. Implement `default_params()` → dict
    """

    meta: StrategyMeta

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = {**self.default_params()}
        if params:
            self.params.update(params)

    @abstractmethod
    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
        """Evaluate the strategy on the current bar.

        Args:
            candles: Dict of TF → DataFrame with columns [date, open, high, low, close, volume].
            indicators: Pre-computed IndicatorBundle.
            current_bar_idx: Index into the entry-timeframe DataFrame.

        Returns:
            StrategySetup if a valid entry is detected, None otherwise.
            Returning None on most bars is expected and correct.
        """

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Return default parameter values."""

    @abstractmethod
    def param_schema(self) -> List[ParamSpec]:
        """Return parameter specs for dynamic dashboard form generation."""

    def reset(self) -> None:
        """Reset any internal state between backtests.  Override if needed."""
        pass

    def __repr__(self) -> str:
        return f"<{self.meta.name} v{self.meta.version} by {self.meta.author}>"
