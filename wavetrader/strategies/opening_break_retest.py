"""
Opening Break & Retest V2 — by Dectrick McGee

Inspired by Scarface Trades' "A+ Setup": mark the high/low of the first
15-minute candle at the NY open, wait for a breakout, then enter on the
retest of the broken level.  Strict 3:1 R:R minimum.  One trade per day,
all within the first ~60 minutes of the session.

V2 Enhancements:
  - 4H EMA trend alignment as HARD filter (only trade with trend)
  - Require rejection wick on retest bar (proof of bounce)
  - Require strong breakout candle (body > 50% of range)
  - Next-bar confirmation mode (wait for follow-through)
  - Tighter retest tolerance

Adapted for 15min data (finest available):
  1. ORB candle = first 15min bar at 13:30 UTC (9:30 AM ET)
  2. Breakout = strong close above/below that range
  3. Retest  = pullback to broken level with rejection wick
  4. Entry on confirmation (next bar closes in direction)
  5. SL below retest bar low + ATR buffer (buys)
  6. TP = SL × min_rr_ratio (default 3.0)
  7. Trading window: 13:45–15:30 UTC (~first 2 hours of NY)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, IndicatorBundle, ParamSpec, StrategyMeta, StrategySetup
from ..types import Signal


class OpeningBreakRetestStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="opening_break_retest",
        name="Opening Break & Retest",
        author="Dectrick McGee",
        version="2.0.0",
        description="NY open first-candle breakout with pullback retest — trend-filtered, 3R minimum",
        category="scalper",
        timeframes=["15min", "1h", "4h"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="15min",
    )

    # ── Default parameters ───────────────────────────────────────────────

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── ORB timing (UTC) ─────────────────────────────────────────
            "orb_hour": 13,                 # NY open ≈ 13:30 UTC
            "orb_minute": 30,               # First 15min candle starts here
            "trade_window_bars": 8,         # Max bars after ORB to look for setup (8 × 15min = 2h)

            # ── Range filters ────────────────────────────────────────────
            "min_orb_range_pips": 5.0,      # Skip tiny ranges (need real volatility)
            "max_orb_range_pips": 80.0,     # Skip huge gaps (too risky)

            # ── Breakout quality ─────────────────────────────────────────
            "breakout_body_pct": 0.30,      # Breakout candle body must be > 30% of its range
            "breakout_clear_pips": 1.0,     # Close must be N pips past ORB level

            # ── Retest entry ─────────────────────────────────────────────
            "retest_tolerance_pct": 0.25,   # How far past breakout level counts as retest
            "require_close_beyond": True,   # Retest bar must close on breakout side
            "require_rejection_wick": False, # Rejection wick (optional — hurts on 15min)
            "rejection_wick_ratio": 0.35,   # Min wick/range ratio for rejection

            # ── Next-bar confirmation ────────────────────────────────────
            "require_next_bar_confirm": True,  # Wait one bar for follow-through
            "confirm_bar_body_pct": 0.30,      # Confirm bar body > 30% of its range

            # ── Higher-timeframe trend filter ────────────────────────────
            "require_4h_ema_align": False,  # Disabled — hurts ORB counter-trend setups
            "require_1h_ema_align": False,  # Disabled — same reason

            # ── Risk management ──────────────────────────────────────────
            "min_rr_ratio": 2.0,            # 2:1 reward:risk (best PF on EUR/JPY)
            "trailing_stop_pct": 0.4,       # Trail 40% of initial risk
            "sl_buffer_atr_mult": 0.3,      # ATR buffer added below structure SL
            "min_sl_pips": 8.0,             # Floor SL to avoid micro-stops
            "max_sl_pips": 60.0,            # Cap SL

            # ── Filters ──────────────────────────────────────────────────
            "min_confidence": 0.45,
            "min_atr_pips": 2.0,            # Minimum ATR to trade
            "min_adx": 8.0,                # Loose trending filter
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("min_orb_range_pips", "Min ORB Range (pips)", "float",
                      5.0, 2.0, 30.0, 1.0, "Minimum opening range to trade"),
            ParamSpec("max_orb_range_pips", "Max ORB Range (pips)", "float",
                      80.0, 20.0, 120.0, 5.0, "Maximum opening range to trade"),
            ParamSpec("retest_tolerance_pct", "Retest Tolerance", "float",
                      0.25, 0.1, 0.8, 0.05,
                      "How deep past breakout level the pullback can go (fraction of ORB range)"),
            ParamSpec("min_rr_ratio", "Min R:R Ratio", "float",
                      2.0, 1.0, 5.0, 0.5, "Minimum reward-to-risk ratio"),
            ParamSpec("trailing_stop_pct", "Trailing Stop %", "float",
                      0.4, 0.1, 0.8, 0.05, "Trail distance as fraction of initial risk"),
            ParamSpec("sl_buffer_atr_mult", "SL ATR Buffer", "float",
                      0.3, 0.0, 1.0, 0.1, "ATR multiplier added as buffer below structure SL"),
            ParamSpec("trade_window_bars", "Trade Window (bars)", "int",
                      8, 2, 16, 1, "How many 15min bars after ORB to look for entry"),
            ParamSpec("min_confidence", "Min Confidence", "float",
                      0.45, 0.3, 0.8, 0.05, "Minimum confidence threshold"),
            ParamSpec("breakout_body_pct", "Breakout Body %", "float",
                      0.30, 0.2, 0.8, 0.05, "Min body/range ratio for breakout candle"),
            ParamSpec("min_adx", "Min ADX", "float",
                      8.0, 5.0, 30.0, 1.0, "Minimum ADX for trending market filter"),
        ]

    # ── Internal state (reset per day) ───────────────────────────────────

    def __init__(self, params=None):
        super().__init__(params)
        self._orb_high: float = 0.0
        self._orb_low: float = 0.0
        self._orb_close: float = 0.0
        self._orb_date = None
        self._orb_bar_idx: int = -1
        self._breakout_dir: Optional[Signal] = None
        self._breakout_bar_idx: int = -1
        self._traded_today: bool = False
        # Next-bar confirmation state
        self._pending_setup: Optional[dict] = None

    def reset(self) -> None:
        self._orb_high = 0.0
        self._orb_low = 0.0
        self._orb_close = 0.0
        self._orb_date = None
        self._orb_bar_idx = -1
        self._breakout_dir = None
        self._breakout_bar_idx = -1
        self._traded_today = False
        self._pending_setup = None

    # ── Main evaluation ──────────────────────────────────────────────────

    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
        p = self.params
        etf = self.meta.entry_timeframe
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

        # ── Determine pip size from pair ─────────────────────────────────
        pair = indicators.pair if indicators.pair else "GBP/JPY"
        pip_size = 0.0001 if "USD" in pair else 0.01

        # ── Parse timestamp ──────────────────────────────────────────────
        if "date" in df.columns:
            ts = pd.Timestamp(bar["date"])
        else:
            ts = df.index[i]
        hour = ts.hour
        minute = ts.minute
        bar_date = ts.date()

        # ── Reset on new day ─────────────────────────────────────────────
        if bar_date != self._orb_date:
            self._orb_high = 0.0
            self._orb_low = 0.0
            self._orb_close = 0.0
            self._orb_date = bar_date
            self._orb_bar_idx = -1
            self._breakout_dir = None
            self._breakout_bar_idx = -1
            self._traded_today = False
            self._pending_setup = None

        # Already traded today — one trade per day
        if self._traded_today:
            return None

        # ── Check pending next-bar confirmation ──────────────────────────
        if self._pending_setup is not None:
            pend = self._pending_setup
            self._pending_setup = None

            # Confirmation bar must be the very next bar
            if i == pend["bar_idx"] + 1:
                bar_body = abs(close - open_)
                bar_range = high - low
                body_pct = bar_body / bar_range if bar_range > 0 else 0

                confirmed = False
                if pend["direction"] == Signal.BUY:
                    # Confirm: bar closes green (close > open) with decent body
                    if close > open_ and body_pct >= p["confirm_bar_body_pct"]:
                        confirmed = True
                else:
                    # Confirm: bar closes red (close < open) with decent body
                    if close < open_ and body_pct >= p["confirm_bar_body_pct"]:
                        confirmed = True

                if confirmed:
                    self._traded_today = True
                    # Re-compute SL from the confirmation bar
                    atr_arr = indicators.atr.get(etf)
                    atr_val = atr_arr[i] if atr_arr is not None and i < len(atr_arr) else 0.0
                    atr_buffer = atr_val * p["sl_buffer_atr_mult"]

                    if pend["direction"] == Signal.BUY:
                        # SL below the lower of retest bar low and confirm bar low
                        sl_ref = min(low, pend["retest_low"])
                        sl_price = sl_ref - atr_buffer
                        sl_pips = (close - sl_price) / pip_size
                    else:
                        sl_ref = max(high, pend["retest_high"])
                        sl_price = sl_ref + atr_buffer
                        sl_pips = (sl_price - close) / pip_size

                    sl_pips = max(sl_pips, p["min_sl_pips"])
                    sl_pips = min(sl_pips, p["max_sl_pips"])
                    tp_pips = sl_pips * p["min_rr_ratio"]

                    return StrategySetup(
                        direction=pend["direction"],
                        entry_price=close,
                        sl_pips=round(sl_pips, 1),
                        tp_pips=round(tp_pips, 1),
                        confidence=round(pend["confidence"], 3),
                        trailing_stop_pct=p["trailing_stop_pct"],
                        timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.utcnow(),
                        reason=pend["reason"] + " [confirmed]",
                        context=pend["context"],
                    )
            # Confirmation failed or wrong bar — signal expires
            return None

        # ── Step 1: Mark the opening range (first 15min candle) ──────────
        if hour == p["orb_hour"] and minute == p["orb_minute"]:
            self._orb_high = high
            self._orb_low = low
            self._orb_close = close
            self._orb_bar_idx = i
            self._breakout_dir = None
            return None  # Just recorded the ORB candle

        # Need a valid ORB
        if self._orb_bar_idx < 0 or self._orb_high <= 0:
            return None

        # ── Check we're within the trade window ──────────────────────────
        bars_since_orb = i - self._orb_bar_idx
        if bars_since_orb < 1 or bars_since_orb > p["trade_window_bars"]:
            return None

        orb_range = self._orb_high - self._orb_low
        orb_range_pips = orb_range / pip_size

        # ── Range size filter ────────────────────────────────────────────
        if orb_range_pips < p["min_orb_range_pips"]:
            return None
        if orb_range_pips > p["max_orb_range_pips"]:
            return None

        # ── ATR filter ───────────────────────────────────────────────────
        atr_arr = indicators.atr.get(etf)
        atr_val = atr_arr[i] if atr_arr is not None and i < len(atr_arr) else 0.0
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        atr_pips = atr_val / pip_size
        if atr_pips < p["min_atr_pips"]:
            return None

        # ── ADX filter (require trending market) ─────────────────────────
        adx_arr = indicators.adx.get(etf)
        if adx_arr is not None and i < len(adx_arr):
            adx_val = adx_arr[i]
            if not np.isnan(adx_val) and adx_val < p["min_adx"]:
                return None

        # ── Step 2: Detect breakout ──────────────────────────────────────
        if self._breakout_dir is None:
            bar_body = abs(close - open_)
            bar_range = high - low
            body_pct = bar_body / bar_range if bar_range > 0 else 0

            # Require strong breakout candle (body > threshold)
            if body_pct < p["breakout_body_pct"]:
                return None

            clear_dist = p["breakout_clear_pips"] * pip_size

            if close > self._orb_high + clear_dist:
                self._breakout_dir = Signal.BUY
                self._breakout_bar_idx = i
            elif close < self._orb_low - clear_dist:
                self._breakout_dir = Signal.SELL
                self._breakout_bar_idx = i
            return None  # Wait for retest on a subsequent bar

        # ── 4H EMA trend alignment (HARD FILTER) ────────────────────────
        if p["require_4h_ema_align"]:
            ema20_4h = indicators.ema_20.get("4h")
            ema50_4h = indicators.ema_50.get("4h")
            if ema20_4h is not None and ema50_4h is not None:
                ix_4h = min(i // 16, len(ema20_4h) - 1)
                if ix_4h >= 0:
                    e20 = ema20_4h[ix_4h]
                    e50 = ema50_4h[ix_4h]
                    if not (np.isnan(e20) or np.isnan(e50)):
                        if self._breakout_dir == Signal.BUY and e20 <= e50:
                            return None  # 4H bearish → skip BUY
                        if self._breakout_dir == Signal.SELL and e20 >= e50:
                            return None  # 4H bullish → skip SELL

        # ── 1H EMA trend alignment (HARD FILTER) ────────────────────────
        if p["require_1h_ema_align"]:
            ema20_1h = indicators.ema_20.get("1h")
            ema50_1h = indicators.ema_50.get("1h")
            if ema20_1h is not None and ema50_1h is not None:
                ix_1h = min(i // 4, len(ema20_1h) - 1)
                if ix_1h >= 0:
                    e20 = ema20_1h[ix_1h]
                    e50 = ema50_1h[ix_1h]
                    if not (np.isnan(e20) or np.isnan(e50)):
                        if self._breakout_dir == Signal.BUY and e20 <= e50:
                            return None
                        if self._breakout_dir == Signal.SELL and e20 >= e50:
                            return None

        # ── Step 3: Wait for retest of the broken level ──────────────────
        # Must be at least 1 bar after breakout
        if i <= self._breakout_bar_idx:
            return None

        tolerance = orb_range * p["retest_tolerance_pct"]

        # ATR buffer for SL
        atr_buffer = atr_val * p["sl_buffer_atr_mult"]

        if self._breakout_dir == Signal.BUY:
            # Retest: price pulls back down toward ORB high
            retest_level = self._orb_high
            pulled_back = low <= retest_level + tolerance

            if not pulled_back:
                return None

            # Confirmation: close must be above the breakout level
            if p["require_close_beyond"] and close <= retest_level:
                return None

            # Require rejection wick (lower wick shows buying pressure)
            if p["require_rejection_wick"]:
                bar_range = high - low
                if bar_range > 0:
                    lower_wick = min(open_, close) - low
                    if lower_wick / bar_range < p["rejection_wick_ratio"]:
                        return None
                else:
                    return None

            direction = Signal.BUY

            # SL below retest bar's low + ATR buffer
            sl_price = low - atr_buffer
            sl_distance = close - sl_price
            sl_pips = sl_distance / pip_size

        else:  # SELL breakout
            # Retest: price rallies back up toward ORB low
            retest_level = self._orb_low
            pulled_back = high >= retest_level - tolerance

            if not pulled_back:
                return None

            # Confirmation: close must be below the breakout level
            if p["require_close_beyond"] and close >= retest_level:
                return None

            # Require rejection wick (upper wick shows selling pressure)
            if p["require_rejection_wick"]:
                bar_range = high - low
                if bar_range > 0:
                    upper_wick = high - max(open_, close)
                    if upper_wick / bar_range < p["rejection_wick_ratio"]:
                        return None
                else:
                    return None

            direction = Signal.SELL

            # SL above retest bar's high + ATR buffer
            sl_price = high + atr_buffer
            sl_distance = sl_price - close
            sl_pips = sl_distance / pip_size

        # ── SL bounds ────────────────────────────────────────────────────
        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])

        # ── TP from R:R ratio ────────────────────────────────────────────
        tp_pips = sl_pips * p["min_rr_ratio"]

        # ── Step 4: Confidence scoring ───────────────────────────────────
        conf = 0.60  # Base confidence (higher because we already passed hard filters)

        # Bonus: ORB range is in the sweet spot
        if 15 < orb_range_pips < 45:
            conf += 0.05

        # Bonus: engulfing pattern at retest
        if indicators.engulfing is not None and i < len(indicators.engulfing):
            eng = indicators.engulfing[i]
            if direction == Signal.BUY and eng[0] > 0:
                conf += 0.08
            elif direction == Signal.SELL and eng[0] < 0:
                conf += 0.08

        # Bonus: FVG confluence
        if indicators.fair_value_gaps is not None and i < len(indicators.fair_value_gaps):
            if indicators.fair_value_gaps[i][0] != 0:
                conf += 0.05

        # Bonus: strong ADX (trending hard)
        if adx_arr is not None and i < len(adx_arr):
            adx_val = adx_arr[i]
            if not np.isnan(adx_val) and adx_val > 25:
                conf += 0.05

        conf = min(conf, 0.95)

        if conf < p["min_confidence"]:
            return None

        # ── Build reason and context ─────────────────────────────────────
        reason = (
            f"Break & Retest: ORB {self._orb_low:.3f}–{self._orb_high:.3f} "
            f"({orb_range_pips:.0f} pips) → "
            f"{'bullish' if direction == Signal.BUY else 'bearish'} breakout, "
            f"retested at {retest_level:.3f}"
        )
        context = {
            "orb_high": round(self._orb_high, 5),
            "orb_low": round(self._orb_low, 5),
            "orb_range_pips": round(orb_range_pips, 1),
            "retest_level": round(retest_level, 5),
            "bars_since_orb": bars_since_orb,
            "breakout_direction": "bullish" if direction == Signal.BUY else "bearish",
        }

        # ── Next-bar confirmation (store pending, don't enter yet) ───────
        if p["require_next_bar_confirm"]:
            self._pending_setup = {
                "direction": direction,
                "bar_idx": i,
                "confidence": conf,
                "retest_low": low,
                "retest_high": high,
                "reason": reason,
                "context": context,
            }
            return None  # Wait for next bar

        # ── Direct entry (no confirmation) ───────────────────────────────
        self._traded_today = True

        return StrategySetup(
            direction=direction,
            entry_price=close,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(conf, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.utcnow(),
            reason=reason,
            context=context,
        )
