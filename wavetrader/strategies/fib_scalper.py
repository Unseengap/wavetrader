"""
Fibonacci Scalper — by Dectrick McGee

Inspired by "The 1-Minute Fibonacci Scalping Strategy" (The Moving Average).
Adapted from 1-min to 15-min (finest available timeframe).

Core concept:
1. Detect short-term trend via swing structure (HH/HL = uptrend, LL/LH = downtrend)
2. Wait for a Break of Structure (BOS) — price breaks a prior swing point
3. Plot Fibonacci from the impulse swing (high-to-low or low-to-high)
4. Enter when price retraces into the "Golden Zone" (0.50–0.618 retracement)
5. SL at the 1.0 Fibonacci level (start of the impulse move) + ATR buffer
6. TP at the previous swing extreme (the BOS target)

Session filter focuses on London/NY overlap for maximum volatility.
EMA 200 provides a macro trend bias — only trade in the trend direction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


class _FibSetup:
    """A pending Fibonacci golden-zone retracement setup."""
    __slots__ = (
        "direction", "swing_start", "swing_end", "fib_50", "fib_618",
        "sl_level", "tp_level", "created_idx", "impulse_pips",
    )

    def __init__(
        self, direction: Signal, swing_start: float, swing_end: float,
        fib_50: float, fib_618: float, sl_level: float, tp_level: float,
        created_idx: int, impulse_pips: float,
    ):
        self.direction = direction
        self.swing_start = swing_start
        self.swing_end = swing_end
        self.fib_50 = fib_50
        self.fib_618 = fib_618
        self.sl_level = sl_level
        self.tp_level = tp_level
        self.created_idx = created_idx
        self.impulse_pips = impulse_pips


class FibScalperStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="fib_scalper",
        name="Fibonacci Scalper",
        author="Dectrick McGee",
        version="1.1.0",
        description="Golden Zone (0.5–0.618) Fibonacci retracement scalper with BOS detection",
        category="scalper",
        timeframes=["1min", "15min", "1h", "4h"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="1min",
    )

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Swing detection ──────────────────────────────────────
            "swing_lookback": 3,            # Bars each side for swing detection
            "min_impulse_atr": 1.0,         # Impulse move must be > N × ATR
            "max_swing_age": 120,           # Max bars since BOS to still trade

            # ── Fibonacci levels ─────────────────────────────────────
            "fib_entry_low": 0.50,          # Golden zone start
            "fib_entry_high": 0.618,        # Golden zone end (preferred entry)
            "max_active_setups": 5,         # Max pending fib setups at once

            # ── Trend filter ─────────────────────────────────────────
            "use_ema_200_filter": True,      # Only trade in EMA 200 direction
            "ema_200_tf": "15min",           # Use 15min EMA 200 (not 1min)
            "use_htf_bias": False,           # 1H structure bias (slow on 1min)
            "htf_bias_min": 0.15,            # Min |bias| on 1H structure

            # ── Session filter ───────────────────────────────────────
            "session_start_hour": 7,         # London open
            "session_end_hour": 20,          # NY close
            "overlap_start_hour": 13,        # London/NY overlap
            "overlap_end_hour": 17,
            "overlap_conf_bonus": 0.08,

            # ── Risk management ──────────────────────────────────────
            "min_rr_ratio": 1.5,            # Video: 1:1 to 1:1.5 typical
            "trailing_stop_pct": 0.40,      # Trail 40% of initial risk
            "sl_atr_buffer": 0.3,           # Buffer beyond 1.0 fib level
            "min_sl_pips": 5.0,             # Floor SL (lower for 1min)
            "max_sl_pips": 40.0,            # Cap SL

            # ── Filters ──────────────────────────────────────────────
            "min_confidence": 0.45,
            "min_atr_pips": 0.5,            # 1min ATR is small (~1-2 pips)
            "use_rsi_filter": True,
            "rsi_buy_max": 70,              # Don't buy overbought
            "rsi_sell_min": 30,             # Don't sell oversold
            "require_confirmation": True,    # Bar must close with rejection
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("swing_lookback", "Swing Lookback", "int",
                      3, 2, 10, 1, "Bars each side for swing high/low detection"),
            ParamSpec("min_impulse_atr", "Min Impulse (ATR)", "float",
                      1.0, 0.5, 3.0, 0.1, "Impulse size in ATR multiples"),
            ParamSpec("max_swing_age", "Max Setup Age", "int",
                      120, 30, 200, 10, "Max bars before fib setup expires"),
            ParamSpec("fib_entry_low", "Fib Entry Low", "float",
                      0.50, 0.38, 0.618, 0.01, "Lower bound of golden zone"),
            ParamSpec("fib_entry_high", "Fib Entry High", "float",
                      0.618, 0.50, 0.786, 0.01, "Upper bound of golden zone"),
            ParamSpec("use_ema_200_filter", "EMA 200 Filter", "bool",
                      True, description="Only trade in direction of EMA 200"),
            ParamSpec("min_rr_ratio", "Min R:R", "float",
                      1.5, 1.0, 3.0, 0.25, "Minimum reward:risk ratio"),
            ParamSpec("trailing_stop_pct", "Trailing Stop %", "float",
                      0.40, 0.1, 0.8, 0.05, "Trail distance as fraction of initial risk"),
            ParamSpec("sl_atr_buffer", "SL ATR Buffer", "float",
                      0.3, 0.1, 0.8, 0.1, "Buffer beyond Fib 1.0 for SL"),
            ParamSpec("min_sl_pips", "Min SL", "float",
                      8.0, 3.0, 20.0, 1.0, "Floor SL pips"),
            ParamSpec("max_sl_pips", "Max SL", "float",
                      60.0, 20.0, 100.0, 5.0, "Cap SL pips"),
            ParamSpec("min_confidence", "Min Confidence", "float",
                      0.45, 0.2, 0.8, 0.05, "Confidence threshold"),
            ParamSpec("session_start_hour", "Session Start", "int",
                      7, 0, 23, 1, "UTC hour start"),
            ParamSpec("session_end_hour", "Session End", "int",
                      20, 0, 23, 1, "UTC hour end"),
            ParamSpec("rsi_buy_max", "RSI Buy Max", "float",
                      70, 55, 85, 5, "Avoid buying overbought"),
            ParamSpec("rsi_sell_min", "RSI Sell Min", "float",
                      30, 15, 45, 5, "Avoid selling oversold"),
            ParamSpec("require_confirmation", "Require Confirm", "bool",
                      True, description="Bar must close with rejection candle"),
        ]

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        # Swing points: list of (price, bar_index)
        self._swing_highs: List[Tuple[float, int]] = []
        self._swing_lows: List[Tuple[float, int]] = []
        self._active_setups: List[_FibSetup] = []
        self._last_swing_scan: int = -1
        # Track which swing indices have already triggered a BOS
        self._broken_high_idxs: Set[int] = set()
        self._broken_low_idxs: Set[int] = set()

    def reset(self) -> None:
        self._swing_highs = []
        self._swing_lows = []
        self._active_setups = []
        self._last_swing_scan = -1
        self._broken_high_idxs = set()
        self._broken_low_idxs = set()

    # ─── Swing detection (no look-ahead) ─────────────────────────────

    def _detect_swings(self, highs: np.ndarray, lows: np.ndarray, i: int) -> None:
        """Detect confirmed swing highs/lows up to bar i.

        A swing high at bar j requires bars [j-lb..j+lb] to all have
        lower highs. Only confirms where j + lb <= i (no look-ahead).
        Uses >= for swing high neighbors and <= for swing low neighbors
        to be more permissive (allows ties on edges).
        """
        lb = self.params["swing_lookback"]
        start = max(lb, self._last_swing_scan + 1)
        end = i - lb + 1

        if start >= end:
            return

        n = len(highs)
        for j in range(start, end):
            w_start = max(0, j - lb)
            w_end = min(n, j + lb + 1)

            # Check swing high: j is highest in window
            is_sh = True
            for k in range(w_start, w_end):
                if k != j and highs[k] >= highs[j]:
                    is_sh = False
                    break
            if is_sh:
                self._swing_highs.append((float(highs[j]), j))

            # Check swing low: j is lowest in window
            is_sl = True
            for k in range(w_start, w_end):
                if k != j and lows[k] <= lows[j]:
                    is_sl = False
                    break
            if is_sl:
                self._swing_lows.append((float(lows[j]), j))

        self._last_swing_scan = end - 1

        # Memory management: keep recent swings
        max_keep = 80
        if len(self._swing_highs) > max_keep:
            removed = self._swing_highs[:-max_keep]
            self._swing_highs = self._swing_highs[-max_keep:]
            for _, idx in removed:
                self._broken_high_idxs.discard(idx)
        if len(self._swing_lows) > max_keep:
            removed = self._swing_lows[:-max_keep]
            self._swing_lows = self._swing_lows[-max_keep:]
            for _, idx in removed:
                self._broken_low_idxs.discard(idx)

    # ─── Break-of-structure detection (one-time per swing) ───────────

    def _detect_bos(
        self, close: float, bar_high: float, bar_low: float,
        atr: float, pip: float, current_idx: int,
    ) -> Optional[_FibSetup]:
        """Detect a break of structure on this bar.

        A BOS fires when the bar's range crosses a swing level that
        hasn't been broken before (tracked in _broken sets).
        Uses bar_low for bearish breaks, bar_high for bullish breaks.
        """
        p = self.params
        min_impulse = atr * p["min_impulse_atr"]

        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return None

        # ── Bearish BOS: bar low breaks below a swing low ────────────
        for i in range(len(self._swing_lows) - 1, -1, -1):
            sw_price, sw_idx = self._swing_lows[i]

            if sw_idx in self._broken_low_idxs:
                continue  # Already broken

            if bar_low < sw_price:
                # Fresh break! Find the swing high that forms the impulse
                relevant_high = None
                for j in range(len(self._swing_highs) - 1, -1, -1):
                    sh_price, sh_idx = self._swing_highs[j]
                    if sh_idx < sw_idx and sh_price > sw_price:
                        relevant_high = (sh_price, sh_idx)
                        break

                if relevant_high is None:
                    continue

                impulse = relevant_high[0] - sw_price
                if impulse < min_impulse:
                    continue

                # Mark as broken
                self._broken_low_idxs.add(sw_idx)

                # Fib levels: retracement goes UP from the broken low
                fib_50 = sw_price + impulse * p["fib_entry_low"]
                fib_618 = sw_price + impulse * p["fib_entry_high"]

                return _FibSetup(
                    direction=Signal.SELL,
                    swing_start=relevant_high[0],
                    swing_end=sw_price,
                    fib_50=fib_50,
                    fib_618=fib_618,
                    sl_level=relevant_high[0] + atr * p["sl_atr_buffer"],
                    tp_level=sw_price,
                    created_idx=current_idx,
                    impulse_pips=impulse / pip,
                )

        # ── Bullish BOS: bar high breaks above a swing high ────────
        for i in range(len(self._swing_highs) - 1, -1, -1):
            sw_price, sw_idx = self._swing_highs[i]

            if sw_idx in self._broken_high_idxs:
                continue

            if bar_high > sw_price:
                # Fresh break! Find the swing low that forms the impulse
                relevant_low = None
                for j in range(len(self._swing_lows) - 1, -1, -1):
                    sl_price, sl_idx = self._swing_lows[j]
                    if sl_idx < sw_idx and sl_price < sw_price:
                        relevant_low = (sl_price, sl_idx)
                        break

                if relevant_low is None:
                    continue

                impulse = sw_price - relevant_low[0]
                if impulse < min_impulse:
                    continue

                self._broken_high_idxs.add(sw_idx)

                # Fib levels: retracement goes DOWN from the broken high
                fib_50 = sw_price - impulse * p["fib_entry_low"]
                fib_618 = sw_price - impulse * p["fib_entry_high"]

                return _FibSetup(
                    direction=Signal.BUY,
                    swing_start=relevant_low[0],
                    swing_end=sw_price,
                    fib_50=fib_50,
                    fib_618=fib_618,
                    sl_level=relevant_low[0] - atr * p["sl_atr_buffer"],
                    tp_level=sw_price,
                    created_idx=current_idx,
                    impulse_pips=impulse / pip,
                )

        return None

    # ─── Check active setups for golden zone entry ───────────────────

    def _check_golden_zone_entry(
        self, close: float, high: float, low: float, o: float,
        i: int, atr: float,
    ) -> Optional[_FibSetup]:
        """Check if current bar price has entered any active golden zone."""
        p = self.params

        expired = []
        triggered = None

        for idx, setup in enumerate(self._active_setups):
            # Expire old setups
            if i - setup.created_idx > p["max_swing_age"]:
                expired.append(idx)
                continue

            zone_low = min(setup.fib_50, setup.fib_618)
            zone_high = max(setup.fib_50, setup.fib_618)

            if setup.direction == Signal.SELL:
                # SELL: golden zone is ABOVE the BOS break (price retraced up)
                # Entry when price wicks into or through the zone
                if high >= zone_low:
                    if p["require_confirmation"]:
                        # Bearish rejection: close below zone_high, bearish bar
                        if close < zone_high and close < o:
                            triggered = setup
                            expired.append(idx)
                            break
                    else:
                        if high >= zone_low:  # Any touch
                            triggered = setup
                            expired.append(idx)
                            break

                # Invalidate: price broke back above the swing_start (1.0 fib)
                if high > setup.swing_start:
                    expired.append(idx)

            elif setup.direction == Signal.BUY:
                # BUY: golden zone is BELOW the BOS break (price retraced down)
                if low <= zone_high:
                    if p["require_confirmation"]:
                        # Bullish rejection: close above zone_low, bullish bar
                        if close > zone_low and close > o:
                            triggered = setup
                            expired.append(idx)
                            break
                    else:
                        if low <= zone_high:
                            triggered = setup
                            expired.append(idx)
                            break

                # Invalidate: price broke below the swing_start (1.0 fib)
                if low < setup.swing_start:
                    expired.append(idx)

        # Clean expired
        for idx in sorted(set(expired), reverse=True):
            if idx < len(self._active_setups):
                self._active_setups.pop(idx)

        return triggered

    # ─── Main evaluate ───────────────────────────────────────────────

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
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        o = float(bar["open"])
        pip = 0.01 if "JPY" in indicators.pair else 0.0001

        # ── ATR ──────────────────────────────────────────────────────
        atr_a = indicators.atr.get(etf)
        if atr_a is None or i >= len(atr_a):
            return None
        atr = float(atr_a[i])
        if np.isnan(atr) or atr <= 0:
            return None
        atr_pips = atr / pip
        if atr_pips < p["min_atr_pips"]:
            return None

        # ── Session filter ───────────────────────────────────────────
        ts = bar.get("date")
        hour = 12
        if ts is not None:
            hour = pd.Timestamp(ts).hour
            if hour < p["session_start_hour"] or hour >= p["session_end_hour"]:
                # Still track swings + BOS even outside session
                self._detect_swings(
                    df["high"].values, df["low"].values, i,
                )
                # Detect BOS outside session too (setup created, entry later)
                bos = self._detect_bos(close, high, low, atr, pip, i)
                if bos is not None:
                    if len(self._active_setups) >= p["max_active_setups"]:
                        self._active_setups.pop(0)
                    self._active_setups.append(bos)
                return None

        # ── Detect swings (incremental, no look-ahead) ──────────────
        highs_arr = df["high"].values
        lows_arr = df["low"].values
        self._detect_swings(highs_arr, lows_arr, i)

        # ── Detect new BOS → create Fibonacci setup ──────────────────
        bos = self._detect_bos(close, high, low, atr, pip, i)
        if bos is not None:
            if len(self._active_setups) >= p["max_active_setups"]:
                self._active_setups.pop(0)
            self._active_setups.append(bos)

        # ── Check if price enters any active golden zone ─────────────
        triggered = self._check_golden_zone_entry(close, high, low, o, i, atr)
        if triggered is None:
            return None

        # ══════════════════════════════════════════════════════════════
        # ENTRY CONFIRMED — build the trade setup
        # ══════════════════════════════════════════════════════════════
        d = triggered.direction

        # ── EMA 200 trend filter ─────────────────────────────────────
        if p["use_ema_200_filter"]:
            ema_tf = p.get("ema_200_tf", etf)
            ema200 = indicators.ema_200.get(ema_tf)
            if ema200 is not None:
                # Map entry bar index to EMA TF index
                if ema_tf == etf:
                    ema_idx = i
                elif ema_tf == "15min" and etf == "1min":
                    ema_idx = min(i // 15, len(ema200) - 1)
                elif ema_tf == "1h":
                    ema_idx = min(i // 60, len(ema200) - 1)
                else:
                    ema_idx = min(i, len(ema200) - 1)

                if ema_idx >= 0 and ema_idx < len(ema200):
                    ema_val = float(ema200[ema_idx])
                    if not np.isnan(ema_val):
                        if d == Signal.BUY and close < ema_val:
                            return None
                        if d == Signal.SELL and close > ema_val:
                            return None

        # ── 1H structure bias filter ─────────────────────────────────
        if p["use_htf_bias"]:
            htf = "1h"
            if htf in indicators.structure:
                # Map entry bar to 1H bar index
                if etf == "1min":
                    htf_idx = min(i // 60, len(indicators.structure[htf]) - 1)
                elif etf == "15min":
                    htf_idx = min(i // 4, len(indicators.structure[htf]) - 1)
                else:
                    htf_idx = min(i, len(indicators.structure[htf]) - 1)
                if htf_idx >= 0:
                    bias = float(indicators.structure[htf][htf_idx, 7])
                    if d == Signal.BUY and bias < p["htf_bias_min"]:
                        return None
                    if d == Signal.SELL and bias > -p["htf_bias_min"]:
                        return None

        # ── RSI filter ───────────────────────────────────────────────
        if p["use_rsi_filter"]:
            rsi_a = indicators.rsi.get(etf)
            if rsi_a is not None and i < len(rsi_a):
                rsi = float(rsi_a[i])
                if not np.isnan(rsi):
                    if d == Signal.BUY and rsi > p["rsi_buy_max"]:
                        return None
                    if d == Signal.SELL and rsi < p["rsi_sell_min"]:
                        return None

        # ── Compute SL/TP from Fibonacci levels ─────────────────────
        if d == Signal.BUY:
            sl_dist = close - triggered.sl_level
            tp_dist = triggered.tp_level - close
        else:
            sl_dist = triggered.sl_level - close
            tp_dist = close - triggered.tp_level

        if sl_dist <= 0 or tp_dist <= 0:
            return None

        sl_pips = sl_dist / pip
        tp_pips = tp_dist / pip

        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])

        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        if rr < p["min_rr_ratio"]:
            return None

        # ── Confidence scoring ───────────────────────────────────────
        conf = 0.50

        # Bonus: price near 0.618 (preferred entry per video)
        zone_range = abs(triggered.fib_50 - triggered.fib_618)
        if zone_range > 0:
            if d == Signal.BUY:
                depth = (triggered.fib_50 - close) / zone_range
            else:
                depth = (close - triggered.fib_50) / zone_range
            conf += min(max(depth, 0) * 0.10, 0.10)

        # Bonus: London/NY overlap
        in_overlap = p["overlap_start_hour"] <= hour < p["overlap_end_hour"]
        if in_overlap:
            conf += p["overlap_conf_bonus"]

        # Bonus: large impulse
        if triggered.impulse_pips > atr_pips * 2.0:
            conf += 0.06

        # Bonus: 1H structure alignment
        if "1h" in indicators.structure:
            if etf == "1min":
                htf_idx = min(i // 60, len(indicators.structure["1h"]) - 1)
            elif etf == "15min":
                htf_idx = min(i // 4, len(indicators.structure["1h"]) - 1)
            else:
                htf_idx = min(i, len(indicators.structure["1h"]) - 1)
            if htf_idx >= 0:
                bias = float(indicators.structure["1h"][htf_idx, 7])
                if (d == Signal.BUY and bias > 0.3) or (d == Signal.SELL and bias < -0.3):
                    conf += 0.05

        # Bonus: engulfing at golden zone
        if indicators.engulfing is not None and i < len(indicators.engulfing):
            eng = float(indicators.engulfing[i, 0])
            if (d == Signal.BUY and eng > 0) or (d == Signal.SELL and eng < 0):
                conf += 0.07

        conf = min(conf, 0.95)
        if conf < p["min_confidence"]:
            return None

        side = "Bull" if d == Signal.BUY else "Bear"
        fib_zone = f"{triggered.fib_618:.2f}–{triggered.fib_50:.2f}"
        return StrategySetup(
            direction=d,
            entry_price=close,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(conf, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            reason=(
                f"{side} Fib Golden Zone @ {fib_zone} | "
                f"BOS {triggered.swing_start:.2f}→{triggered.swing_end:.2f} | "
                f"R:R={rr:.1f} | ovl={in_overlap}"
            ),
            context={
                "fib_50": round(triggered.fib_50, 5),
                "fib_618": round(triggered.fib_618, 5),
                "swing_start": round(triggered.swing_start, 5),
                "swing_end": round(triggered.swing_end, 5),
                "impulse_pips": round(triggered.impulse_pips, 1),
            },
        )
