"""
Price Action Close Reversal V2 — by Dectrick McGee

A price-action swing strategy for 4H charts that identifies trend reversals
by watching for a candle that closes against the prevailing trend, then waits
for the NEXT candle to confirm the direction before entering.

Downtrend Buy Signal:
  1. N consecutive candles each closing below their predecessor (downtrend).
  2. A new candle closes ABOVE the last downtrend candle's close (reversal).
  3. NEXT candle also closes higher than it opened (confirmation) → BUY.
  Stop loss: below the lowest low of the ENTIRE downtrend.

Uptrend Sell Signal:
  1. N consecutive candles each closing above their predecessor (uptrend).
  2. A new candle closes BELOW the last uptrend candle's close (reversal).
  3. NEXT candle also closes lower than it opened (confirmation) → SELL.
  Stop loss: above the highest high of the ENTIRE uptrend.

V2 changes from V1:
  - Next-bar confirmation eliminates false reversals
  - SL at trend extreme (not just prev candle) = entering at the true peak
  - No RSI filter — peaks/troughs are inherently oversold/overbought
  - Bigger SL = bigger position room = rides further with geometric trail
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


@dataclass
class _PendingReversal:
    """Stored when a reversal candle is detected; awaits next-bar confirmation."""
    direction: Signal
    bar_idx: int
    entry_close: float       # Close of the reversal bar
    trend_length: int
    trend_low: float         # Lowest low of the entire downtrend (for BUY SL)
    trend_high: float        # Highest high of the entire uptrend (for SELL SL)
    body_ratio: float
    confidence: float


class PriceActionReversalStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="price_action_reversal",
        name="Price Action Close Reversal",
        author="Dectrick McGee",
        version="2.0.0",
        description="4H close reversal with next-bar confirmation — SL at trend extreme, geometric trail",
        category="swing",
        timeframes=["4h", "1d"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="4h",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self._pending: Optional[_PendingReversal] = None

    def reset(self) -> None:
        self._pending = None

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Trend detection ──────────────────────────────────────
            "min_trend_candles": 4,          # Min consecutive trending candles
            "max_trend_candles": 30,         # Ignore extremely extended trends
            # ── Reversal candle quality ──────────────────────────────
            "require_body_ratio": 0.30,      # Reversal candle body >= 30% of range
            "min_reversal_body_atr": 0.25,   # Reversal candle body >= 0.25 × ATR
            # ── Confirmation bar ─────────────────────────────────────
            "require_confirmation": True,    # Wait for next bar to confirm
            "confirm_body_atr_min": 0.15,    # Confirm bar body >= 0.15 × ATR
            # ── Optional filters ─────────────────────────────────────
            "use_ema_200_filter": False,
            "use_adx_filter": False,
            "adx_min": 20,
            # ── Risk management ──────────────────────────────────────
            "trailing_stop_pct": 0.3,        # Geometric trail: trail at 30% of SL distance
            "trail_activate_pnl": 5.0,       # Activate trail at $5 PnL
            "sl_buffer_pips": 5.0,           # Extra buffer below/above trend extreme
            "min_sl_pips": 15.0,             # Floor SL
            "max_sl_pips": 300.0,            # Cap SL (wide — we want room)
            # ── General ──────────────────────────────────────────────
            "min_confidence": 0.4,
            "min_atr_pips": 3.0,
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("min_trend_candles", "Min Trend Candles", "int", 3, 2, 10, 1,
                      "Minimum consecutive candles in a trend before reversal is valid"),
            ParamSpec("require_confirmation", "Next-Bar Confirm", "bool", True, None, None, None,
                      "Wait for next candle to confirm reversal direction"),
            ParamSpec("trailing_stop_pct", "Trail Distance %", "float", 0.3, 0.1, 0.8, 0.05,
                      "Trail distance as fraction of SL distance"),
            ParamSpec("trail_activate_pnl", "Trail Activate PnL ($)", "float", 5.0, 1.0, 20.0, 1.0,
                      "Activate geometric trail at this profit level"),
            ParamSpec("require_body_ratio", "Min Body Ratio", "float", 0.30, 0.1, 0.7, 0.05,
                      "Reversal candle body as ratio of candle range"),
            ParamSpec("sl_buffer_pips", "SL Buffer (pips)", "float", 5.0, 0.0, 20.0, 1.0,
                      "Extra pips below/above the trend extreme SL level"),
            ParamSpec("use_ema_200_filter", "EMA-200 Filter", "bool", False, None, None, None,
                      "Only trade reversals toward the 200 EMA"),
            ParamSpec("min_sl_pips", "Min SL (pips)", "float", 15.0, 5.0, 50.0, 5.0,
                      "Minimum stop loss in pips"),
            ParamSpec("max_sl_pips", "Max SL (pips)", "float", 300.0, 100.0, 500.0, 25.0,
                      "Maximum stop loss in pips"),
        ]

    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
        """Detect trend reversal with next-bar confirmation on 4H chart."""
        p = self.params
        etf = self.meta.entry_timeframe  # "4h"

        if etf not in candles:
            return None

        df = candles[etf]
        i = current_bar_idx
        if i < 200 or i >= len(df):
            return None

        bar = df.iloc[i]
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        open_ = bar["open"]

        pair = indicators.pair if hasattr(indicators, "pair") else "GBP/JPY"
        pip_size = 0.0001 if "USD" in pair else 0.01

        # ── Get ATR ──────────────────────────────────────────────────────
        atr_arr = indicators.atr.get(etf, np.array([]))
        atr_val = atr_arr[i] if i < len(atr_arr) else 0.0
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        atr_pips = atr_val / pip_size

        if atr_pips < p["min_atr_pips"]:
            return None

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: Check pending confirmation from previous bar
        # ══════════════════════════════════════════════════════════════════
        if self._pending is not None:
            pend = self._pending
            self._pending = None  # Always consume

            # Must be the very next bar
            if pend.bar_idx == i - 1:
                body = abs(close - open_)
                body_atr = body / atr_val if atr_val > 0 else 0

                confirmed = False
                if (pend.direction == Signal.BUY
                        and close > open_
                        and body_atr >= p["confirm_body_atr_min"]):
                    confirmed = True
                elif (pend.direction == Signal.SELL
                      and close < open_
                      and body_atr >= p["confirm_body_atr_min"]):
                    confirmed = True

                if confirmed:
                    return self._build_setup(pend, close, pip_size, atr_pips)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: Detect new reversal candle → store as pending
        # ══════════════════════════════════════════════════════════════════

        # Count consecutive trending candles and track extremes
        down_count = 0
        trend_low = df.iloc[i - 1]["low"]  # Start with prev candle
        for j in range(i - 1, 0, -1):
            if df.iloc[j]["close"] < df.iloc[j - 1]["close"]:
                down_count += 1
                trend_low = min(trend_low, df.iloc[j]["low"])
            else:
                break

        up_count = 0
        trend_high = df.iloc[i - 1]["high"]
        for j in range(i - 1, 0, -1):
            if df.iloc[j]["close"] > df.iloc[j - 1]["close"]:
                up_count += 1
                trend_high = max(trend_high, df.iloc[j]["high"])
            else:
                break

        prev_close = df.iloc[i - 1]["close"]
        direction = None
        trend_length = 0

        if (down_count >= p["min_trend_candles"]
                and down_count <= p["max_trend_candles"]
                and close > prev_close):
            direction = Signal.BUY
            trend_length = down_count
        elif (up_count >= p["min_trend_candles"]
              and up_count <= p["max_trend_candles"]
              and close < prev_close):
            direction = Signal.SELL
            trend_length = up_count

        if direction is None:
            return None

        # ── Reversal candle quality ──────────────────────────────────────
        candle_range = high - low
        if candle_range <= 0:
            return None

        body = abs(close - open_)
        body_ratio = body / candle_range
        if body_ratio < p["require_body_ratio"]:
            return None

        body_atr_ratio = body / atr_val if atr_val > 0 else 0
        if body_atr_ratio < p["min_reversal_body_atr"]:
            return None

        # ── Confidence ───────────────────────────────────────────────────
        confidence = 0.50
        if trend_length >= 6:
            confidence += 0.15
        elif trend_length >= 5:
            confidence += 0.10
        elif trend_length >= 4:
            confidence += 0.05
        if body_ratio > 0.6:
            confidence += 0.10
        elif body_ratio > 0.5:
            confidence += 0.05

        # ── EMA-200 filter (optional) ────────────────────────────────────
        if p["use_ema_200_filter"]:
            ema200_arr = indicators.ema_200.get(etf, np.array([]))
            if i < len(ema200_arr):
                ema200 = ema200_arr[i]
                if not np.isnan(ema200):
                    if direction == Signal.BUY and close > ema200:
                        return None
                    if direction == Signal.SELL and close < ema200:
                        return None
                    confidence += 0.05

        # ── ADX filter (optional) ────────────────────────────────────────
        if p["use_adx_filter"]:
            adx_arr = indicators.adx.get(etf, np.array([]))
            if i < len(adx_arr):
                adx_val = adx_arr[i]
                if not np.isnan(adx_val) and adx_val < p["adx_min"]:
                    return None

        confidence = min(confidence, 1.0)
        if confidence < p["min_confidence"]:
            return None

        # ── Store pending or enter immediately ───────────────────────────
        pending = _PendingReversal(
            direction=direction,
            bar_idx=i,
            entry_close=close,
            trend_length=trend_length,
            trend_low=trend_low,
            trend_high=trend_high,
            body_ratio=body_ratio,
            confidence=confidence,
        )

        if p["require_confirmation"]:
            self._pending = pending
            return None  # Wait for next bar
        else:
            return self._build_setup(pending, close, pip_size, atr_pips)

    def _build_setup(
        self,
        pend: _PendingReversal,
        entry_close: float,
        pip_size: float,
        atr_pips: float,
    ) -> Optional[StrategySetup]:
        """Build final StrategySetup from a confirmed (or immediate) reversal."""
        p = self.params

        if pend.direction == Signal.BUY:
            # SL below the lowest low of the ENTIRE downtrend
            sl_level = pend.trend_low
            sl_pips = (entry_close - sl_level) / pip_size + p["sl_buffer_pips"]
        else:
            # SL above the highest high of the ENTIRE uptrend
            sl_level = pend.trend_high
            sl_pips = (sl_level - entry_close) / pip_size + p["sl_buffer_pips"]

        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])

        # No fixed TP — geometric trail handles exit
        tp_pips = 9999.0

        dir_label = "BUY" if pend.direction == Signal.BUY else "SELL"
        trend_dir = "downtrend" if pend.direction == Signal.BUY else "uptrend"

        return StrategySetup(
            direction=pend.direction,
            entry_price=entry_close,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(pend.confidence, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            exit_mode="geometric_trail",
            reason=(
                f"4H confirmed reversal {dir_label}: {pend.trend_length} candle "
                f"{trend_dir} broken + next bar confirmed | "
                f"SL {sl_pips:.0f}p at trend extreme | geometric trail"
            ),
            context={
                "trend_length": pend.trend_length,
                "body_ratio": round(pend.body_ratio, 3),
                "sl_level": round(sl_level, 5),
            },
        )
