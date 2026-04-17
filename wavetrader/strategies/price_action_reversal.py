"""
Price Action Structure V3 — by Dectrick McGee

A full market-structure strategy for 4H charts that catches entries at proven
support/resistance zones using HH/HL/LL/LH swing structure.

Entry Types:
  1. SUPPORT BOUNCE (BUY): In uptrend, price pulls back to a prior swing low
     zone, candle rejects → next bar confirms → BUY at support.
  2. RESISTANCE REJECTION (SELL): In downtrend, price rallies to a prior swing
     high zone, candle rejects → next bar confirms → SELL at resistance.
  3. BREAK & RETEST BUY: Price breaks above resistance, retests the level as
     new support, next bar confirms → BUY.
  4. BREAK & RETEST SELL: Price breaks below support, retests as new resistance,
     next bar confirms → SELL.
  5. TREND REVERSAL: After extended trend, structure shifts (first HL after LL
     sequence or first LH after HH sequence) at a major zone → enter.

SL is placed just beyond the zone (tight!), giving massive R:R vs the geometric
trailing stop. Entries at structure = tight SL + long runway = big R multiples.

V3 changes from V2:
  - Multiple entry types instead of just close-reversal
  - Own swing-point and S/R zone tracking for precise levels
  - Structure trend from IndicatorBundle for direction bias
  - SL at structure zone (tight) instead of trend extreme (wide)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


# ─────────────────────────────────────────────────────────────────────────────
# Internal data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SwingPoint:
    """A detected swing high or swing low."""
    idx: int
    price: float
    is_high: bool  # True = swing high, False = swing low
    tested: int = 0  # How many times price returned to this level


@dataclass
class _PendingEntry:
    """Stored signal awaiting next-bar confirmation."""
    direction: Signal
    bar_idx: int
    entry_type: str  # "support_bounce", "resistance_reject", "break_retest", "reversal"
    zone_price: float  # The S/R zone price
    sl_level: float  # Exact SL price
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


class PriceActionReversalStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="price_action_reversal",
        name="Price Action Close Reversal",
        author="Dectrick McGee",
        version="3.0.0",
        description="4H market structure — S/R zones, break & retest, trend reversals with geometric trail",
        category="swing",
        timeframes=["4h", "1d"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="4h",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self._pending: Optional[_PendingEntry] = None
        self._swing_highs: List[_SwingPoint] = []
        self._swing_lows: List[_SwingPoint] = []
        self._broken_levels: List[_SwingPoint] = []  # Levels that price broke through
        self._last_swing_check: int = -1

    def reset(self) -> None:
        self._pending = None
        self._swing_highs = []
        self._swing_lows = []
        self._broken_levels = []
        self._last_swing_check = -1

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Swing detection ──────────────────────────────────────
            "swing_lookback": 7,             # Bars left/right for pivot detection (7 = major swings only)
            "zone_tolerance_pct": 0.002,     # 0.2% — price within this = "at zone"
            "max_zones": 15,                 # Keep this many recent zones
            "min_zone_age": 5,               # Zone must be at least 5 bars old (20h on 4H)
            "min_zone_tests": 0,             # Zone must have been tested N times before
            # ── Confirmation ─────────────────────────────────────────
            "require_confirmation": True,
            "confirm_body_atr_min": 0.40,    # Confirm bar body >= 0.40 × ATR (strong confirmation)
            # ── Entry types (all on by default) ──────────────────────
            "enable_support_bounce": True,
            "enable_resistance_reject": True,
            "enable_break_retest": True,
            "enable_reversal": True,
            # ── Trend ────────────────────────────────────────────────
            "trend_bias_threshold": 0.15,    # |bias| > this = trending
            "require_trend_alignment": True, # Support bounce only in uptrend, resist reject only in downtrend
            "min_trend_candles_reversal": 5, # For reversal entries only
            # ── Rejection quality ────────────────────────────────────
            "min_wick_atr_ratio": 0.7,       # Rejection wick >= 0.7 × ATR (strong rejection)
            # ── Risk management ──────────────────────────────────────
            "trailing_stop_pct": 0.3,
            "sl_buffer_pips": 5.0,           # Buffer beyond zone for SL
            "min_sl_pips": 10.0,
            "max_sl_pips": 200.0,
            "exit_mode": "multi_tp_trail",   # "geometric_trail" or "multi_tp_trail"
            # ── General ──────────────────────────────────────────────
            "min_confidence": 0.50,
            "min_atr_pips": 3.0,
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("swing_lookback", "Swing Lookback", "int", 3, 2, 7, 1,
                      "Bars left/right for pivot point detection"),
            ParamSpec("zone_tolerance_pct", "Zone Tolerance %", "float", 0.003, 0.001, 0.008, 0.001,
                      "Price within this % of zone counts as 'at zone'"),
            ParamSpec("trailing_stop_pct", "Trail Distance %", "float", 0.3, 0.1, 0.8, 0.05,
                      "Trail distance as fraction of SL distance"),
            ParamSpec("enable_support_bounce", "Support Bounce", "bool", True, None, None, None,
                      "Buy at support in uptrend"),
            ParamSpec("enable_resistance_reject", "Resistance Reject", "bool", True, None, None, None,
                      "Sell at resistance in downtrend"),
            ParamSpec("enable_break_retest", "Break & Retest", "bool", True, None, None, None,
                      "Enter on break & retest of S/R"),
            ParamSpec("enable_reversal", "Trend Reversal", "bool", True, None, None, None,
                      "Enter on trend structure shift"),
            ParamSpec("sl_buffer_pips", "SL Buffer (pips)", "float", 5.0, 0.0, 20.0, 1.0,
                      "Extra pips beyond zone for SL"),
            ParamSpec("min_sl_pips", "Min SL (pips)", "float", 10.0, 5.0, 30.0, 5.0,
                      "Minimum stop loss in pips"),
            ParamSpec("max_sl_pips", "Max SL (pips)", "float", 200.0, 50.0, 400.0, 25.0,
                      "Maximum stop loss in pips"),
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Swing detection
    # ─────────────────────────────────────────────────────────────────────

    def _update_swings(self, df: pd.DataFrame, i: int) -> None:
        """Detect new swing points up to bar i (must be called in order)."""
        p = self.params
        lb = p["swing_lookback"]

        # Only check bars we haven't checked yet, and leave room for right side
        start = max(self._last_swing_check + 1, lb)
        end = i - lb  # Need lb bars to the right to confirm pivot

        for j in range(start, end + 1):
            high_j = df.iloc[j]["high"]
            low_j = df.iloc[j]["low"]

            # Swing high: higher than lb bars on each side
            is_sh = True
            for k in range(1, lb + 1):
                if j - k < 0 or j + k >= len(df):
                    is_sh = False
                    break
                if df.iloc[j - k]["high"] >= high_j or df.iloc[j + k]["high"] >= high_j:
                    is_sh = False
                    break
            if is_sh:
                self._swing_highs.append(_SwingPoint(idx=j, price=high_j, is_high=True))

            # Swing low: lower than lb bars on each side
            is_sl = True
            for k in range(1, lb + 1):
                if j - k < 0 or j + k >= len(df):
                    is_sl = False
                    break
                if df.iloc[j - k]["low"] <= low_j or df.iloc[j + k]["low"] <= low_j:
                    is_sl = False
                    break
            if is_sl:
                self._swing_lows.append(_SwingPoint(idx=j, price=low_j, is_high=False))

        self._last_swing_check = end

        # Trim to max_zones most recent
        max_z = p["max_zones"]
        if len(self._swing_highs) > max_z:
            self._swing_highs = self._swing_highs[-max_z:]
        if len(self._swing_lows) > max_z:
            self._swing_lows = self._swing_lows[-max_z:]

    def _find_nearby_zone(
        self, price: float, zones: List[_SwingPoint], tolerance: float, min_age: int, current_idx: int
    ) -> Optional[_SwingPoint]:
        """Find the nearest zone within tolerance, that's old enough."""
        best = None
        best_dist = float("inf")
        for z in reversed(zones):  # Recent first
            if current_idx - z.idx < min_age:
                continue
            dist = abs(price - z.price) / price
            if dist < tolerance and dist < best_dist:
                best = z
                best_dist = dist
        return best

    def _check_level_broken(
        self, df: pd.DataFrame, i: int, level: float, direction: str
    ) -> bool:
        """Check if a level was cleanly broken (close beyond it) in recent bars."""
        lookback = 10
        for j in range(max(0, i - lookback), i):
            close_j = df.iloc[j]["close"]
            if direction == "above" and close_j > level:
                return True
            if direction == "below" and close_j < level:
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main evaluate
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
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

        # ── ATR ──────────────────────────────────────────────────────────
        atr_arr = indicators.atr.get(etf, np.array([]))
        atr_val = atr_arr[i] if i < len(atr_arr) else 0.0
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        atr_pips = atr_val / pip_size
        if atr_pips < p["min_atr_pips"]:
            return None

        # ── Update swing points ──────────────────────────────────────────
        self._update_swings(df, i)

        # ── Get trend bias from structure ────────────────────────────────
        struct_arr = indicators.structure.get(etf, np.array([]))
        trend_bias = 0.0
        if i < len(struct_arr):
            trend_bias = struct_arr[i][7]  # [-1, 1] trend bias

        thr = p["trend_bias_threshold"]
        is_uptrend = trend_bias > thr
        is_downtrend = trend_bias < -thr
        tolerance = p["zone_tolerance_pct"]
        min_age = p["min_zone_age"]

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: Check pending confirmation
        # ══════════════════════════════════════════════════════════════════
        if self._pending is not None:
            pend = self._pending
            self._pending = None

            if pend.bar_idx == i - 1:
                body = abs(close - open_)
                body_atr = body / atr_val if atr_val > 0 else 0

                confirmed = False
                if pend.direction == Signal.BUY and close > open_ and body_atr >= p["confirm_body_atr_min"]:
                    confirmed = True
                elif pend.direction == Signal.SELL and close < open_ and body_atr >= p["confirm_body_atr_min"]:
                    confirmed = True

                if confirmed:
                    return self._build_setup(pend, close, pip_size, atr_pips)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: Scan for new entry signals
        # ══════════════════════════════════════════════════════════════════

        # ── Type 1: Support Bounce (BUY in uptrend) ──────────────────────
        trend_required = p["require_trend_alignment"]
        if p["enable_support_bounce"] and (is_uptrend or not trend_required):
            zone = self._find_nearby_zone(low, self._swing_lows, tolerance, min_age, i)
            if zone is not None and zone.tested >= p["min_zone_tests"]:
                # Check rejection: wick below zone but close above
                wick_below = zone.price - low
                if wick_below > 0 and close > zone.price and wick_below / atr_val >= p["min_wick_atr_ratio"]:
                    conf = 0.55
                    if is_uptrend:
                        conf += 0.10
                    if zone.tested > 0:
                        conf += 0.05  # Zone has been tested before = stronger
                    zone.tested += 1
                    sl_level = zone.price - p["sl_buffer_pips"] * pip_size
                    pending = _PendingEntry(
                        direction=Signal.BUY, bar_idx=i,
                        entry_type="support_bounce", zone_price=zone.price,
                        sl_level=sl_level, confidence=conf,
                        context={"zone_idx": zone.idx, "trend_bias": round(trend_bias, 2)},
                    )
                    if p["require_confirmation"]:
                        self._pending = pending
                        return None
                    return self._build_setup(pending, close, pip_size, atr_pips)

        # ── Type 2: Resistance Rejection (SELL in downtrend) ─────────────
        if p["enable_resistance_reject"] and (is_downtrend or not trend_required):
            zone = self._find_nearby_zone(high, self._swing_highs, tolerance, min_age, i)
            if zone is not None and zone.tested >= p["min_zone_tests"]:
                wick_above = high - zone.price
                if wick_above > 0 and close < zone.price and wick_above / atr_val >= p["min_wick_atr_ratio"]:
                    conf = 0.55
                    if is_downtrend:
                        conf += 0.10
                    if zone.tested > 0:
                        conf += 0.05
                    zone.tested += 1
                    sl_level = zone.price + p["sl_buffer_pips"] * pip_size
                    pending = _PendingEntry(
                        direction=Signal.SELL, bar_idx=i,
                        entry_type="resistance_reject", zone_price=zone.price,
                        sl_level=sl_level, confidence=conf,
                        context={"zone_idx": zone.idx, "trend_bias": round(trend_bias, 2)},
                    )
                    if p["require_confirmation"]:
                        self._pending = pending
                        return None
                    return self._build_setup(pending, close, pip_size, atr_pips)

        # ── Type 3: Break & Retest ──────────────────────────────────────
        if p["enable_break_retest"]:
            # BUY: price previously broke above a swing high, now retesting it as support
            for sh in reversed(self._swing_highs[-10:]):
                if i - sh.idx < 5:
                    continue  # Too recent
                # Was it broken? Check if any bar closed above it after the swing
                if self._check_level_broken(df, i, sh.price, "above"):
                    # Is price retesting? (close is near the broken level, from above)
                    dist_pct = abs(close - sh.price) / close
                    if dist_pct < tolerance and close >= sh.price and low <= sh.price + atr_val * 0.5:
                        # Rejection wick: touched the level and bounced
                        wick = sh.price - low if low < sh.price else 0
                        if wick / atr_val >= p["min_wick_atr_ratio"] * 0.5 or close > open_:
                            conf = 0.55
                            if is_uptrend:
                                conf += 0.10
                            sl_level = sh.price - p["sl_buffer_pips"] * pip_size
                            pending = _PendingEntry(
                                direction=Signal.BUY, bar_idx=i,
                                entry_type="break_retest", zone_price=sh.price,
                                sl_level=sl_level, confidence=conf,
                                context={"zone_idx": sh.idx, "break_type": "resistance_to_support"},
                            )
                            if p["require_confirmation"]:
                                self._pending = pending
                                return None
                            return self._build_setup(pending, close, pip_size, atr_pips)
                        break  # Only check the most recent broken level

            # SELL: price broke below a swing low, now retesting it as resistance
            for sl_pt in reversed(self._swing_lows[-10:]):
                if i - sl_pt.idx < 5:
                    continue
                if self._check_level_broken(df, i, sl_pt.price, "below"):
                    dist_pct = abs(close - sl_pt.price) / close
                    if dist_pct < tolerance and close <= sl_pt.price and high >= sl_pt.price - atr_val * 0.5:
                        wick = high - sl_pt.price if high > sl_pt.price else 0
                        if wick / atr_val >= p["min_wick_atr_ratio"] * 0.5 or close < open_:
                            conf = 0.55
                            if is_downtrend:
                                conf += 0.10
                            sl_level = sl_pt.price + p["sl_buffer_pips"] * pip_size
                            pending = _PendingEntry(
                                direction=Signal.SELL, bar_idx=i,
                                entry_type="break_retest", zone_price=sl_pt.price,
                                sl_level=sl_level, confidence=conf,
                                context={"zone_idx": sl_pt.idx, "break_type": "support_to_resistance"},
                            )
                            if p["require_confirmation"]:
                                self._pending = pending
                                return None
                            return self._build_setup(pending, close, pip_size, atr_pips)
                        break

        # ── Type 4: Trend Reversal (enhanced V2 logic) ──────────────────
        if p["enable_reversal"]:
            min_tc = p["min_trend_candles_reversal"]
            down_count = 0
            trend_low = df.iloc[i - 1]["low"]
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
            body = abs(close - open_)
            candle_range = high - low
            if candle_range > 0 and body / candle_range >= 0.30:
                if down_count >= min_tc and close > prev_close:
                    conf = 0.45 + min(down_count - min_tc, 3) * 0.05
                    sl_level = trend_low - p["sl_buffer_pips"] * pip_size
                    pending = _PendingEntry(
                        direction=Signal.BUY, bar_idx=i,
                        entry_type="reversal", zone_price=trend_low,
                        sl_level=sl_level, confidence=conf,
                        context={"trend_length": down_count},
                    )
                    if p["require_confirmation"]:
                        self._pending = pending
                        return None
                    return self._build_setup(pending, close, pip_size, atr_pips)

                elif up_count >= min_tc and close < prev_close:
                    conf = 0.45 + min(up_count - min_tc, 3) * 0.05
                    sl_level = trend_high + p["sl_buffer_pips"] * pip_size
                    pending = _PendingEntry(
                        direction=Signal.SELL, bar_idx=i,
                        entry_type="reversal", zone_price=trend_high,
                        sl_level=sl_level, confidence=conf,
                        context={"trend_length": up_count},
                    )
                    if p["require_confirmation"]:
                        self._pending = pending
                        return None
                    return self._build_setup(pending, close, pip_size, atr_pips)

        return None

    # ─────────────────────────────────────────────────────────────────────
    # Build final setup
    # ─────────────────────────────────────────────────────────────────────

    def _build_setup(
        self,
        pend: _PendingEntry,
        entry_close: float,
        pip_size: float,
        atr_pips: float,
    ) -> Optional[StrategySetup]:
        p = self.params

        if pend.direction == Signal.BUY:
            sl_pips = (entry_close - pend.sl_level) / pip_size
        else:
            sl_pips = (pend.sl_level - entry_close) / pip_size

        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])

        if pend.confidence < p["min_confidence"]:
            return None

        tp_pips = 9999.0  # Geometric trail handles exit

        dir_label = "BUY" if pend.direction == Signal.BUY else "SELL"

        return StrategySetup(
            direction=pend.direction,
            entry_price=entry_close,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(pend.confidence, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            exit_mode=p["exit_mode"],
            reason=(
                f"4H {pend.entry_type} {dir_label} at {pend.zone_price:.3f} | "
                f"SL {sl_pips:.0f}p | {p['exit_mode']}"
            ),
            context={
                "entry_type": pend.entry_type,
                "zone_price": round(pend.zone_price, 5),
                **pend.context,
            },
        )
