"""
Price Action Close Reversal — by Dectrick McGee

A clean price-action strategy for 4H charts that identifies trend reversals
by watching for a candle that closes against the prevailing trend.

Downtrend Buy Signal:
  After N consecutive candles each closing below their predecessor, a new
  candle closes ABOVE the last downtrend candle's close → BUY.
  Stop loss: below the lowest low of the last downtrend candle.

Uptrend Sell Signal:
  After N consecutive candles each closing above their predecessor, a new
  candle closes BELOW the last uptrend candle's close → SELL.
  Stop loss: above the highest high of the last uptrend candle.

Designed for 4H–1D timeframes to filter noise and catch significant reversals.
Optional EMA-200 filter to only trade reversals back toward the long-term mean.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


class PriceActionReversalStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="price_action_reversal",
        name="Price Action Close Reversal",
        author="Dectrick McGee",
        version="1.0.0",
        description="4H candlestick close reversal — trend break via single candle close against direction",
        category="swing",
        timeframes=["4h", "1d"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="4h",
    )

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Trend detection ──────────────────────────────────────
            "min_trend_candles": 3,          # Minimum consecutive trending candles
            "max_trend_candles": 30,         # Ignore trends that are too extended
            # ── Reversal candle quality ──────────────────────────────
            "require_body_ratio": 0.35,      # Reversal candle body must be >= 35% of range
            "min_reversal_body_atr": 0.3,    # Reversal candle body >= 0.3 × ATR
            # ── Optional filters ─────────────────────────────────────
            "use_ema_200_filter": False,      # Only take reversals toward EMA-200
            "use_rsi_filter": True,           # RSI must be in oversold/overbought zone
            "rsi_oversold": 35,              # RSI below this → oversold (buy ok)
            "rsi_overbought": 65,            # RSI above this → overbought (sell ok)
            "use_adx_filter": False,          # Require ADX > threshold (strong trend)
            "adx_min": 20,                   # Minimum ADX for trend confirmation
            # ── Risk management ──────────────────────────────────────
            "min_rr_ratio": 2.0,             # Minimum reward:risk ratio
            "trailing_stop_pct": 0.4,        # Trail 40% of initial risk
            "sl_buffer_pips": 3.0,           # Extra buffer below/above SL level
            "min_sl_pips": 10.0,             # Floor SL
            "max_sl_pips": 150.0,            # Cap SL (wider for 4H)
            # ── General ──────────────────────────────────────────────
            "min_confidence": 0.4,           # Minimum confidence to enter
            "min_atr_pips": 3.0,             # Skip dead markets
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("min_trend_candles", "Min Trend Candles", "int", 3, 2, 10, 1,
                      "Minimum consecutive candles in a trend before reversal is valid"),
            ParamSpec("min_rr_ratio", "Min R:R Ratio", "float", 2.0, 1.0, 5.0, 0.5,
                      "Minimum reward-to-risk ratio for entry"),
            ParamSpec("trailing_stop_pct", "Trailing Stop %", "float", 0.4, 0.1, 0.8, 0.05,
                      "Trail distance as fraction of initial risk"),
            ParamSpec("require_body_ratio", "Min Body Ratio", "float", 0.35, 0.1, 0.7, 0.05,
                      "Reversal candle body as ratio of candle range"),
            ParamSpec("use_rsi_filter", "Use RSI Filter", "bool", True, None, None, None,
                      "Require RSI oversold/overbought for reversal"),
            ParamSpec("rsi_oversold", "RSI Oversold", "int", 35, 15, 45, 5,
                      "RSI threshold for oversold (buy signals)"),
            ParamSpec("rsi_overbought", "RSI Overbought", "int", 65, 55, 85, 5,
                      "RSI threshold for overbought (sell signals)"),
            ParamSpec("sl_buffer_pips", "SL Buffer (pips)", "float", 3.0, 0.0, 15.0, 1.0,
                      "Extra pips below/above the SL level for safety"),
            ParamSpec("use_ema_200_filter", "EMA-200 Filter", "bool", False, None, None, None,
                      "Only trade reversals toward the 200 EMA"),
            ParamSpec("min_sl_pips", "Min SL (pips)", "float", 10.0, 5.0, 30.0, 1.0,
                      "Minimum stop loss in pips"),
            ParamSpec("max_sl_pips", "Max SL (pips)", "float", 150.0, 50.0, 300.0, 10.0,
                      "Maximum stop loss in pips"),
        ]

    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
        """Detect trend reversal via candlestick close pattern on 4H chart."""
        p = self.params
        etf = self.meta.entry_timeframe  # "4h"

        if etf not in candles:
            return None

        df = candles[etf]
        i = current_bar_idx
        min_lookback = p["min_trend_candles"] + 1
        if i < max(min_lookback, 200) or i >= len(df):
            return None

        bar = df.iloc[i]
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        open_ = bar["open"]

        # Determine pip size from pair
        pair = indicators.pair if hasattr(indicators, "pair") else "GBP/JPY"
        pip_size = 0.0001 if "USD" in pair else 0.01

        # ── Get indicators ───────────────────────────────────────────────
        atr_arr = indicators.atr.get(etf, np.array([]))
        atr_val = atr_arr[i] if i < len(atr_arr) else 0.0
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        atr_pips = atr_val / pip_size

        if atr_pips < p["min_atr_pips"]:
            return None

        # ── Count consecutive trending candles before current bar ────────
        down_count = 0
        up_count = 0

        # Count consecutive lower closes (downtrend)
        for j in range(i - 1, 0, -1):
            if df.iloc[j]["close"] < df.iloc[j - 1]["close"]:
                down_count += 1
            else:
                break

        # Count consecutive higher closes (uptrend)
        for j in range(i - 1, 0, -1):
            if df.iloc[j]["close"] > df.iloc[j - 1]["close"]:
                up_count += 1
            else:
                break

        # ── Check for reversal ───────────────────────────────────────────
        prev_close = df.iloc[i - 1]["close"]
        prev_low = df.iloc[i - 1]["low"]
        prev_high = df.iloc[i - 1]["high"]

        direction = None
        trend_length = 0

        # Downtrend reversal → BUY: current close > previous close after downtrend
        if (down_count >= p["min_trend_candles"]
                and down_count <= p["max_trend_candles"]
                and close > prev_close):
            direction = Signal.BUY
            trend_length = down_count

        # Uptrend reversal → SELL: current close < previous close after uptrend
        elif (up_count >= p["min_trend_candles"]
              and up_count <= p["max_trend_candles"]
              and close < prev_close):
            direction = Signal.SELL
            trend_length = up_count

        if direction is None:
            return None

        # ── Reversal candle quality checks ───────────────────────────────
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

        # ── Compute stop loss from the last trend candle's extreme ───────
        if direction == Signal.BUY:
            # SL below the lowest low of the last downtrend candle
            sl_level = prev_low
            sl_pips = (close - sl_level) / pip_size + p["sl_buffer_pips"]
        else:
            # SL above the highest high of the last uptrend candle
            sl_level = prev_high
            sl_pips = (sl_level - close) / pip_size + p["sl_buffer_pips"]

        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])

        # ── TP from R:R ──────────────────────────────────────────────────
        tp_pips = sl_pips * p["min_rr_ratio"]

        # ── Confidence scoring ───────────────────────────────────────────
        confidence = 0.45

        # Longer trends = stronger reversal signal
        if trend_length >= 5:
            confidence += 0.10
        elif trend_length >= 4:
            confidence += 0.05

        # Strong body = more conviction
        if body_ratio > 0.6:
            confidence += 0.10
        elif body_ratio > 0.5:
            confidence += 0.05

        # ── Optional RSI filter ──────────────────────────────────────────
        if p["use_rsi_filter"]:
            rsi_arr = indicators.rsi.get(etf, np.array([]))
            rsi_val = rsi_arr[i] if i < len(rsi_arr) else 50.0
            if not np.isnan(rsi_val):
                if direction == Signal.BUY and rsi_val > p["rsi_oversold"]:
                    return None  # Not oversold enough for buy reversal
                if direction == Signal.SELL and rsi_val < p["rsi_overbought"]:
                    return None  # Not overbought enough for sell reversal
                # Deep oversold/overbought = higher confidence
                if direction == Signal.BUY and rsi_val < 25:
                    confidence += 0.10
                elif direction == Signal.SELL and rsi_val > 75:
                    confidence += 0.10

        # ── Optional EMA-200 filter ──────────────────────────────────────
        if p["use_ema_200_filter"]:
            ema200_arr = indicators.ema_200.get(etf, np.array([]))
            if i < len(ema200_arr):
                ema200 = ema200_arr[i]
                if not np.isnan(ema200):
                    # Buy only if price below EMA-200 (reverting to mean)
                    if direction == Signal.BUY and close > ema200:
                        return None
                    # Sell only if price above EMA-200
                    if direction == Signal.SELL and close < ema200:
                        return None
                    confidence += 0.05

        # ── Optional ADX filter ──────────────────────────────────────────
        if p["use_adx_filter"]:
            adx_arr = indicators.adx.get(etf, np.array([]))
            if i < len(adx_arr):
                adx_val = adx_arr[i]
                if not np.isnan(adx_val) and adx_val < p["adx_min"]:
                    return None  # Trend not strong enough to reverse from

        confidence = min(confidence, 1.0)

        if confidence < p["min_confidence"]:
            return None

        dir_label = "BUY" if direction == Signal.BUY else "SELL"
        trend_dir = "downtrend" if direction == Signal.BUY else "uptrend"

        return StrategySetup(
            direction=direction,
            entry_price=close,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(confidence, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            reason=(
                f"4H close reversal {dir_label}: {trend_length} candle {trend_dir} "
                f"broken by close {'above' if direction == Signal.BUY else 'below'} "
                f"prev close | body ratio {body_ratio:.0%} | "
                f"SL {sl_pips:.1f} pips below structure"
            ),
            context={
                "trend_length": trend_length,
                "body_ratio": round(body_ratio, 3),
                "sl_level": round(sl_level, 5),
            },
        )
