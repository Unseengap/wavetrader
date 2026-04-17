"""
Harmonic Pattern Scanner — by Dectrick McGee

Full harmonic pattern detection on 1H and 4H timeframes.  Scans for XABCD
Fibonacci structures and enters at the Potential Reversal Zone (D point).

Detected patterns (Big 7 + AB=CD):
  1. Gartley    — XAB=0.618, XAD=0.786
  2. Butterfly  — XAB=0.786, XAD=1.272–1.618
  3. Bat        — XAB=0.382–0.50, XAD=0.886
  4. Alt Bat    — XAB=0.382, XAD=1.13
  5. Crab       — XAB=0.382–0.618, XAD=1.618
  6. Deep Crab  — XAB=0.886, XAD=1.618
  7. Cypher     — XAB=0.382–0.618, XAD=0.786 of XC
  8. Shark      — ABC=1.13–1.618, XAD=0.886–1.13
  9. AB=CD      — CD leg equals AB leg (±tolerance)

Each pattern auto-generates:
  - SL beyond the X point (or D extension)
  - TP1 at 0.382 retracement of AD (close 33%)
  - TP2 at 0.618 retracement of AD (close 33%)
  - TP3 at A point / full AD retracement (close 34%, rides with trail)
  - Geometric trailing stop on remainder

Per-pattern win-rate tracking via trade context for optimization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic pattern definitions — Fibonacci ratio constraints
# Each pattern defines (min, max) for each leg ratio.
# None means that ratio is not checked for this pattern.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _PatternDef:
    """Defines Fibonacci ratio constraints for one harmonic pattern type."""
    name: str
    xab: Optional[Tuple[float, float]]  # B retracement of XA
    abc: Optional[Tuple[float, float]]  # C retracement of AB
    bcd: Optional[Tuple[float, float]]  # D extension of BC
    xad: Optional[Tuple[float, float]]  # D retracement/extension of XA
    # For Cypher/Shark: XAD is measured as retracement of XC instead of XA
    xad_of_xc: bool = False


_HARMONIC_PATTERNS: List[_PatternDef] = [
    _PatternDef("gartley",     xab=(0.618, 0.618), abc=(0.382, 0.886), bcd=(1.272, 1.618), xad=(0.786, 0.786)),
    _PatternDef("butterfly",   xab=(0.786, 0.786), abc=(0.382, 0.886), bcd=(1.618, 2.618), xad=(1.272, 1.618)),
    _PatternDef("bat",         xab=(0.382, 0.500), abc=(0.382, 0.886), bcd=(1.618, 2.618), xad=(0.886, 0.886)),
    _PatternDef("alt_bat",     xab=(0.382, 0.382), abc=(0.382, 0.886), bcd=(2.000, 3.618), xad=(1.130, 1.130)),
    _PatternDef("crab",        xab=(0.382, 0.618), abc=(0.382, 0.886), bcd=(2.240, 3.618), xad=(1.618, 1.618)),
    _PatternDef("deep_crab",   xab=(0.886, 0.886), abc=(0.382, 0.886), bcd=(2.000, 3.618), xad=(1.618, 1.618)),
    _PatternDef("cypher",      xab=(0.382, 0.618), abc=(1.130, 1.414), bcd=None,           xad=(0.786, 0.786), xad_of_xc=True),
    _PatternDef("shark",       xab=None,           abc=(1.130, 1.618), bcd=(1.618, 2.240), xad=(0.886, 1.130)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Swing point for zigzag
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Swing:
    """A confirmed swing high or low."""
    idx: int
    price: float
    is_high: bool  # True = swing high, False = swing low


@dataclass
class _HarmonicMatch:
    """A detected harmonic pattern ready for signal generation."""
    pattern_name: str
    direction: str          # "bullish" or "bearish"
    x: float
    a: float
    b: float
    c: float
    d: float                # Projected D price
    x_idx: int
    a_idx: int
    b_idx: int
    c_idx: int
    d_idx: int
    fib_accuracy: float     # Average ratio accuracy (0–1, higher = tighter)
    timeframe: str


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicScannerStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="harmonic_scanner",
        name="Harmonic Pattern Scanner",
        author="Dectrick McGee",
        version="1.0.0",
        description="Fibonacci harmonic pattern scanner — Gartley, Butterfly, Bat, Crab, Cypher, Shark, AB=CD with multi-TP",
        category="swing",
        timeframes=["1h", "4h", "1d"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="1h",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        # Per-timeframe swing storage
        self._swings: Dict[str, List[_Swing]] = {}
        self._last_swing_check: Dict[str, int] = {}
        self._used_patterns: set = set()  # Track (pattern, d_idx, tf) to avoid duplicates

    def reset(self) -> None:
        self._swings = {}
        self._last_swing_check = {}
        self._used_patterns = set()

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Swing detection ──────────────────────────────────────────
            "swing_lookback": 3,              # Bars left/right for pivot
            "min_swing_atr_mult": 0.3,        # Min swing size as ATR multiple
            "max_swings": 80,                 # Keep N most recent swings per TF
            # ── Pattern matching ─────────────────────────────────────────
            "fib_tolerance": 0.05,            # ±5% ratio tolerance
            "prz_tolerance_atr": 0.5,         # D within 0.5 × ATR of current price
            "max_pattern_bars": 200,          # Max bars X→D span
            "min_pattern_bars": 10,           # Min bars X→D span
            # ── Risk management ──────────────────────────────────────────
            "sl_buffer_atr_mult": 0.3,        # SL buffer beyond pattern extreme
            "trailing_stop_pct": 0.40,        # Trail on remainder after TPs
            "min_sl_pips": 10.0,
            "max_sl_pips": 300.0,
            # ── Multi-TP Fibonacci levels ────────────────────────────────
            "tp1_fib": 0.382,                 # TP1 = 0.382 retrace of AD
            "tp2_fib": 0.618,                 # TP2 = 0.618 retrace of AD
            "tp3_fib": 1.0,                   # TP3 = full A point
            "tp1_fraction": 0.33,             # Close 33% at TP1
            "tp2_fraction": 0.33,             # Close 33% at TP2
            # Remaining ~34% rides trail
            # ── Confidence ───────────────────────────────────────────────
            "min_confidence": 0.45,
            "min_atr_pips": 3.0,
            # ── Scan timeframes ──────────────────────────────────────────
            "scan_timeframes": ["1h", "4h"],
            # ── Per-pattern enables ──────────────────────────────────────
            "enable_gartley": True,
            "enable_butterfly": True,
            "enable_bat": True,
            "enable_alt_bat": True,
            "enable_crab": True,
            "enable_deep_crab": True,
            "enable_cypher": True,
            "enable_shark": True,
            "enable_abcd": True,
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("swing_lookback", "Swing Lookback", "int", 5, 3, 10, 1,
                      "Bars left/right for pivot detection"),
            ParamSpec("fib_tolerance", "Fib Tolerance", "float", 0.03, 0.01, 0.08, 0.005,
                      "Ratio tolerance (±) for pattern matching"),
            ParamSpec("trailing_stop_pct", "Trail Distance %", "float", 0.40, 0.1, 0.8, 0.05,
                      "Trail distance as fraction of SL distance"),
            ParamSpec("tp1_fib", "TP1 Fib Level", "float", 0.382, 0.236, 0.500, 0.01,
                      "First TP as Fibonacci retracement of AD"),
            ParamSpec("tp2_fib", "TP2 Fib Level", "float", 0.618, 0.382, 0.786, 0.01,
                      "Second TP as Fibonacci retracement of AD"),
            ParamSpec("tp1_fraction", "TP1 Close %", "float", 0.33, 0.20, 0.50, 0.01,
                      "Fraction of position to close at TP1"),
            ParamSpec("tp2_fraction", "TP2 Close %", "float", 0.33, 0.20, 0.50, 0.01,
                      "Fraction of position to close at TP2"),
            ParamSpec("sl_buffer_atr_mult", "SL Buffer (ATR×)", "float", 0.3, 0.1, 1.0, 0.05,
                      "SL buffer beyond X point as ATR multiple"),
            ParamSpec("min_sl_pips", "Min SL (pips)", "float", 10.0, 5.0, 30.0, 5.0,
                      "Minimum stop loss in pips"),
            ParamSpec("max_sl_pips", "Max SL (pips)", "float", 300.0, 50.0, 500.0, 25.0,
                      "Maximum stop loss in pips"),
            ParamSpec("enable_gartley", "Gartley", "bool", True, None, None, None, "Enable Gartley pattern"),
            ParamSpec("enable_butterfly", "Butterfly", "bool", True, None, None, None, "Enable Butterfly pattern"),
            ParamSpec("enable_bat", "Bat", "bool", True, None, None, None, "Enable Bat pattern"),
            ParamSpec("enable_alt_bat", "Alt Bat", "bool", True, None, None, None, "Enable Alternate Bat pattern"),
            ParamSpec("enable_crab", "Crab", "bool", True, None, None, None, "Enable Crab pattern"),
            ParamSpec("enable_deep_crab", "Deep Crab", "bool", True, None, None, None, "Enable Deep Crab pattern"),
            ParamSpec("enable_cypher", "Cypher", "bool", True, None, None, None, "Enable Cypher pattern"),
            ParamSpec("enable_shark", "Shark", "bool", True, None, None, None, "Enable Shark pattern"),
            ParamSpec("enable_abcd", "AB=CD", "bool", True, None, None, None, "Enable AB=CD pattern"),
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Layer 1: Zigzag swing detection (incremental, no look-ahead)
    # ─────────────────────────────────────────────────────────────────────

    def _update_swings(self, df: pd.DataFrame, tf: str, current_idx: int,
                       atr_arr: np.ndarray) -> None:
        """Detect new swing highs/lows up to current_idx on the given timeframe."""
        p = self.params
        lb = p["swing_lookback"]

        if tf not in self._swings:
            self._swings[tf] = []
            self._last_swing_check[tf] = lb - 1

        start = max(self._last_swing_check[tf] + 1, lb)
        end = current_idx - lb  # Need lb bars to right to confirm

        if end < start:
            return

        highs = df["high"].values
        lows = df["low"].values

        for j in range(start, end + 1):
            atr_val = atr_arr[j] if j < len(atr_arr) else 0.0
            if np.isnan(atr_val) or atr_val <= 0:
                continue
            min_swing = atr_val * p["min_swing_atr_mult"]

            high_j = highs[j]
            low_j = lows[j]

            # Swing high: higher than lb bars on each side
            is_sh = True
            for k in range(1, lb + 1):
                if j - k < 0 or j + k >= len(df):
                    is_sh = False
                    break
                if highs[j - k] >= high_j or highs[j + k] >= high_j:
                    is_sh = False
                    break

            # Swing low: lower than lb bars on each side
            is_sl = True
            for k in range(1, lb + 1):
                if j - k < 0 or j + k >= len(df):
                    is_sl = False
                    break
                if lows[j - k] <= low_j or lows[j + k] <= low_j:
                    is_sl = False
                    break

            swings = self._swings[tf]

            if is_sh:
                # Check minimum swing size from last swing low
                last_low = None
                for s in reversed(swings):
                    if not s.is_high:
                        last_low = s
                        break
                if last_low is None or abs(high_j - last_low.price) >= min_swing:
                    # Enforce alternation: if last was also a high, replace if this is higher
                    if swings and swings[-1].is_high:
                        if high_j > swings[-1].price:
                            swings[-1] = _Swing(idx=j, price=high_j, is_high=True)
                    else:
                        swings.append(_Swing(idx=j, price=high_j, is_high=True))

            if is_sl:
                last_high = None
                for s in reversed(swings):
                    if s.is_high:
                        last_high = s
                        break
                if last_high is None or abs(last_high.price - low_j) >= min_swing:
                    # Enforce alternation: if last was also a low, replace if this is lower
                    if swings and not swings[-1].is_high:
                        if low_j < swings[-1].price:
                            swings[-1] = _Swing(idx=j, price=low_j, is_high=False)
                    else:
                        swings.append(_Swing(idx=j, price=low_j, is_high=False))

        self._last_swing_check[tf] = end

        # Trim to max_swings
        max_s = p["max_swings"]
        if len(self._swings[tf]) > max_s:
            self._swings[tf] = self._swings[tf][-max_s:]

    # ─────────────────────────────────────────────────────────────────────
    # Layer 2: Pattern matching engine
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ratio_match(actual: float, expected_min: float, expected_max: float,
                     tolerance: float) -> Tuple[bool, float]:
        """Check if actual ratio is within [expected_min - tol, expected_max + tol].
        Returns (match, accuracy) where accuracy is 1.0 at center, decreasing toward edges."""
        lo = expected_min - tolerance
        hi = expected_max + tolerance
        if actual < lo or actual > hi:
            return False, 0.0
        center = (expected_min + expected_max) / 2.0
        span = (expected_max - expected_min) / 2.0 + tolerance
        deviation = abs(actual - center) / span if span > 0 else 0.0
        accuracy = max(0.0, 1.0 - deviation)
        return True, accuracy

    def _find_xabcd_patterns(self, swings: List[_Swing], tf: str,
                             current_idx: int) -> List[_HarmonicMatch]:
        """Scan recent swings for XABCD harmonic patterns.

        Tries all valid 5-point combinations from recent swings where
        points alternate H/L.  D must be among the most recent swings.
        """
        p = self.params
        tol = p["fib_tolerance"]
        matches: List[_HarmonicMatch] = []
        n = len(swings)
        if n < 5:
            return matches

        # D candidates: last few swings
        max_d_look = min(4, n - 4)

        for di in range(n - 1, n - 1 - max_d_look, -1):
            d_sw = swings[di]

            # Walk backwards through swings to find C, B, A, X
            # C is the nearest preceding swing of opposite type from D
            # B is the nearest preceding swing of same type as D
            # A is the nearest preceding swing of same type as C
            # X is the nearest preceding swing of same type as B (= same as D)
            # This ensures alternation: X(H/L), A(L/H), B(H/L), C(L/H), D(H/L)

            # Try multiple C candidates
            for ci in range(di - 1, max(di - 8, -1), -1):
                if ci < 0:
                    break
                c_sw = swings[ci]
                if c_sw.is_high == d_sw.is_high:
                    continue  # C must be opposite type from D

                for bi in range(ci - 1, max(ci - 8, -1), -1):
                    if bi < 0:
                        break
                    b_sw = swings[bi]
                    if b_sw.is_high != d_sw.is_high:
                        continue  # B must be same type as D

                    for ai in range(bi - 1, max(bi - 8, -1), -1):
                        if ai < 0:
                            break
                        a_sw = swings[ai]
                        if a_sw.is_high != c_sw.is_high:
                            continue  # A must be same type as C

                        for xi in range(ai - 1, max(ai - 8, -1), -1):
                            if xi < 0:
                                break
                            x_sw = swings[xi]
                            if x_sw.is_high != b_sw.is_high:
                                continue  # X must be same type as B/D

                            # Span check
                            bar_span = d_sw.idx - x_sw.idx
                            if bar_span < p["min_pattern_bars"] or bar_span > p["max_pattern_bars"]:
                                continue

                            x, a, b, c, d = x_sw.price, a_sw.price, b_sw.price, c_sw.price, d_sw.price

                            xa = abs(a - x)
                            ab = abs(b - a)
                            bc = abs(c - b)
                            cd = abs(d - c)
                            if xa < 1e-9 or ab < 1e-9 or bc < 1e-9:
                                continue

                            # Direction: bullish if X is low & A is high
                            is_bullish = a > x
                            direction = "bullish" if is_bullish else "bearish"

                            xab_ratio = ab / xa
                            abc_ratio = bc / ab
                            xad_ratio = abs(d - x) / xa
                            bcd_ratio = cd / bc if bc > 1e-9 else 0.0

                            xc = abs(c - x)
                            xad_of_xc_ratio = abs(d - x) / xc if xc > 1e-9 else 0.0

                            self._try_match_patterns(
                                matches, p, tol, direction, x, a, b, c, d,
                                x_sw, a_sw, b_sw, c_sw, d_sw,
                                xab_ratio, abc_ratio, bcd_ratio, xad_ratio,
                                xad_of_xc_ratio, ab, cd, tf,
                            )
                            break  # Only best X per A
                        break  # Only best A per B
                    break  # Only best B per C

        return matches

    def _try_match_patterns(
        self, matches: List[_HarmonicMatch], p: dict, tol: float,
        direction: str, x: float, a: float, b: float, c: float, d: float,
        x_sw: _Swing, a_sw: _Swing, b_sw: _Swing, c_sw: _Swing, d_sw: _Swing,
        xab_ratio: float, abc_ratio: float, bcd_ratio: float, xad_ratio: float,
        xad_of_xc_ratio: float, ab: float, cd: float, tf: str,
    ) -> None:
        """Check all pattern definitions against computed ratios."""
        for pdef in _HARMONIC_PATTERNS:
            enable_key = f"enable_{pdef.name}"
            if not p.get(enable_key, True):
                continue

            accuracies = []
            ok = True

            if pdef.xab is not None:
                m, acc = self._ratio_match(xab_ratio, pdef.xab[0], pdef.xab[1], tol)
                if not m:
                    ok = False
                else:
                    accuracies.append(acc)

            if ok and pdef.abc is not None:
                m, acc = self._ratio_match(abc_ratio, pdef.abc[0], pdef.abc[1], tol)
                if not m:
                    ok = False
                else:
                    accuracies.append(acc)

            if ok and pdef.bcd is not None:
                m, acc = self._ratio_match(bcd_ratio, pdef.bcd[0], pdef.bcd[1], tol)
                if not m:
                    ok = False
                else:
                    accuracies.append(acc)

            if ok and pdef.xad is not None:
                ratio_to_check = xad_of_xc_ratio if pdef.xad_of_xc else xad_ratio
                m, acc = self._ratio_match(ratio_to_check, pdef.xad[0], pdef.xad[1], tol)
                if not m:
                    ok = False
                else:
                    accuracies.append(acc)

            if ok and accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                matches.append(_HarmonicMatch(
                    pattern_name=pdef.name,
                    direction=direction,
                    x=x, a=a, b=b, c=c, d=d,
                    x_idx=x_sw.idx, a_idx=a_sw.idx,
                    b_idx=b_sw.idx, c_idx=c_sw.idx, d_idx=d_sw.idx,
                    fib_accuracy=avg_acc,
                    timeframe=tf,
                ))

        # AB=CD check
        if p.get("enable_abcd", True) and cd > 1e-9:
            abcd_ratio = cd / ab
            m, acc = self._ratio_match(abcd_ratio, 0.95, 1.05, tol)
            if not m:
                m1, acc1 = self._ratio_match(abcd_ratio, 1.272, 1.272, tol)
                m2, acc2 = self._ratio_match(abcd_ratio, 1.618, 1.618, tol)
                if m1:
                    m, acc = True, acc1
                elif m2:
                    m, acc = True, acc2
            if m:
                matches.append(_HarmonicMatch(
                    pattern_name="abcd",
                    direction=direction,
                    x=x, a=a, b=b, c=c, d=d,
                    x_idx=x_sw.idx, a_idx=a_sw.idx,
                    b_idx=b_sw.idx, c_idx=c_sw.idx, d_idx=d_sw.idx,
                    fib_accuracy=acc,
                    timeframe=tf,
                ))

    # ─────────────────────────────────────────────────────────────────────
    # Layer 3: Signal generation
    # ─────────────────────────────────────────────────────────────────────

    def _score_pattern(self, match: _HarmonicMatch, indicators: IndicatorBundle,
                       current_idx: int, close: float) -> float:
        """Compute confidence score for a detected pattern."""
        conf = 0.45  # Base

        # +0.10 for tight ratio accuracy (>95%)
        if match.fib_accuracy > 0.95:
            conf += 0.10
        elif match.fib_accuracy > 0.85:
            conf += 0.05

        # +0.10 for 4H EMA trend alignment
        ema20 = indicators.ema_20.get("4h", np.array([]))
        ema50 = indicators.ema_50.get("4h", np.array([]))
        ema200 = indicators.ema_200.get("4h", np.array([]))

        # Map 1h bar index to approximate 4h index
        tf = match.timeframe
        if tf == "1h":
            idx_4h = current_idx // 4
        else:
            idx_4h = current_idx

        if (idx_4h < len(ema20) and idx_4h < len(ema50) and idx_4h < len(ema200)
                and not np.isnan(ema20[idx_4h]) and not np.isnan(ema50[idx_4h])
                and not np.isnan(ema200[idx_4h])):
            if match.direction == "bullish" and ema20[idx_4h] > ema50[idx_4h] > ema200[idx_4h]:
                conf += 0.10
            elif match.direction == "bearish" and ema20[idx_4h] < ema50[idx_4h] < ema200[idx_4h]:
                conf += 0.10

        # +0.05 for RSI divergence at D
        rsi_arr = indicators.rsi.get(tf, np.array([]))
        d_idx = match.d_idx
        if d_idx < len(rsi_arr) and not np.isnan(rsi_arr[d_idx]):
            rsi_val = rsi_arr[d_idx]
            if match.direction == "bullish" and rsi_val < 35:
                conf += 0.05
            elif match.direction == "bearish" and rsi_val > 65:
                conf += 0.05

        # +0.05 for 4H timeframe patterns (higher TF = stronger)
        if match.timeframe == "4h":
            conf += 0.05

        return min(conf, 1.0)

    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
        p = self.params
        etf = self.meta.entry_timeframe  # "1h"

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

        pair = indicators.pair if hasattr(indicators, "pair") else "GBP/JPY"
        pip_size = 0.0001 if "USD" in pair else 0.01

        # ── ATR check ────────────────────────────────────────────────────
        atr_arr = indicators.atr.get(etf, np.array([]))
        atr_val = atr_arr[i] if i < len(atr_arr) else 0.0
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        atr_pips = atr_val / pip_size
        if atr_pips < p["min_atr_pips"]:
            return None

        # ── Update swings on all scan timeframes ─────────────────────────
        scan_tfs = p.get("scan_timeframes", ["1h", "4h"])
        for tf in scan_tfs:
            if tf not in candles:
                continue
            tf_atr = indicators.atr.get(tf, np.array([]))
            tf_df = candles[tf]
            # Map entry TF bar index to this TF's bar index
            if tf == etf:
                tf_idx = i
            elif tf == "4h" and etf == "1h":
                tf_idx = i // 4
            elif tf == "1h" and etf == "4h":
                tf_idx = i * 4
            else:
                tf_idx = i  # fallback
            if tf_idx >= len(tf_df):
                tf_idx = len(tf_df) - 1
            self._update_swings(tf_df, tf, tf_idx, tf_atr)

        # ── Scan for patterns on all timeframes ──────────────────────────
        best_match: Optional[_HarmonicMatch] = None
        best_conf: float = 0.0

        for tf in scan_tfs:
            if tf not in self._swings:
                continue
            swings = self._swings[tf]
            if tf == etf:
                tf_idx = i
            elif tf == "4h" and etf == "1h":
                tf_idx = i // 4
            else:
                tf_idx = i

            patterns = self._find_xabcd_patterns(swings, tf, tf_idx)

            for match in patterns:
                # Deduplicate: don't signal same pattern twice
                key = (match.pattern_name, match.d_idx, tf)
                if key in self._used_patterns:
                    continue

                # PRZ check: is current price near the D point?
                prz_tol = atr_val * p["prz_tolerance_atr"]
                if abs(close - match.d) > prz_tol:
                    continue

                # Direction validation: bullish D should be a low, bearish D should be a high
                if match.direction == "bullish" and close > match.a:
                    continue  # Price already past A, D reversal missed
                if match.direction == "bearish" and close < match.a:
                    continue

                conf = self._score_pattern(match, indicators, i, close)
                if conf > best_conf:
                    best_conf = conf
                    best_match = match

        if best_match is None or best_conf < p["min_confidence"]:
            return None

        # ── Mark pattern as used ─────────────────────────────────────────
        key = (best_match.pattern_name, best_match.d_idx, best_match.timeframe)
        self._used_patterns.add(key)

        # ── Build signal ─────────────────────────────────────────────────
        return self._build_setup(best_match, best_conf, close, pip_size, atr_val, pair)

    def _build_setup(
        self,
        match: _HarmonicMatch,
        confidence: float,
        entry_price: float,
        pip_size: float,
        atr_val: float,
        pair: str,
    ) -> Optional[StrategySetup]:
        p = self.params

        is_buy = match.direction == "bullish"
        direction = Signal.BUY if is_buy else Signal.SELL

        # ── SL: beyond the pattern extreme ───────────────────────────────
        # For patterns where D is between X and A (gartley, bat): SL beyond X
        # For patterns where D extends beyond X (butterfly, crab, alt_bat): SL beyond D
        sl_buffer = atr_val * p["sl_buffer_atr_mult"]
        if is_buy:
            # Bullish: X is a low, D is a low. SL below the lower of X and D
            sl_anchor = min(match.x, match.d)
            sl_price = sl_anchor - sl_buffer
            sl_pips = (entry_price - sl_price) / pip_size
        else:
            # Bearish: X is a high, D is a high. SL above the higher of X and D
            sl_anchor = max(match.x, match.d)
            sl_price = sl_anchor + sl_buffer
            sl_pips = (sl_price - entry_price) / pip_size

        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])

        # ── Multi-TP levels (Fibonacci retracements of AD) ───────────────
        ad_distance = abs(match.a - match.d)
        ad_pips = ad_distance / pip_size

        tp1_pips = ad_pips * p["tp1_fib"]
        tp2_pips = ad_pips * p["tp2_fib"]
        tp3_pips = ad_pips * p["tp3_fib"]

        tp_levels = [
            (tp1_pips, p["tp1_fraction"]),
            (tp2_pips, p["tp2_fraction"]),
            (tp3_pips, 1.0 - p["tp1_fraction"] - p["tp2_fraction"]),
        ]

        # Primary TP is TP3 (for the single-TP field)
        tp_pips = tp3_pips

        dir_label = "BUY" if is_buy else "SELL"

        return StrategySetup(
            direction=direction,
            entry_price=entry_price,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(confidence, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            exit_mode="multi_tp_trail",
            reason=(
                f"{match.timeframe.upper()} {match.pattern_name.replace('_', ' ').title()} "
                f"{dir_label} at D={match.d:.5f} | "
                f"SL {sl_pips:.0f}p beyond X={match.x:.5f} | "
                f"TP1={tp1_pips:.0f}p TP2={tp2_pips:.0f}p TP3={tp3_pips:.0f}p"
            ),
            tp_levels=tp_levels,
            context={
                "pattern_name": match.pattern_name,
                "pattern_direction": match.direction,
                "timeframe": match.timeframe,
                "fib_accuracy": round(match.fib_accuracy, 3),
                "x": round(match.x, 5),
                "a": round(match.a, 5),
                "b": round(match.b, 5),
                "c": round(match.c, 5),
                "d": round(match.d, 5),
                "x_idx": match.x_idx,
                "a_idx": match.a_idx,
                "d_idx": match.d_idx,
                "pair": pair,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern stats utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_pattern_stats(trades) -> Dict[str, Dict[str, Any]]:
    """Compute per-pattern performance statistics from a list of Trade objects.

    Args:
        trades: List of Trade objects (from BacktestResults.trades).

    Returns:
        Dict keyed by pattern name with stats: count, wins, losses,
        win_rate, profit_factor, avg_pnl, best, worst.
    """
    from collections import defaultdict
    buckets: Dict[str, list] = defaultdict(list)

    for t in trades:
        ctx = getattr(t, 'context', {})
        name = ctx.get("pattern_name", "unknown")
        buckets[name].append(t)

    stats: Dict[str, Dict[str, Any]] = {}
    for name, bucket in sorted(buckets.items()):
        n = len(bucket)
        wins = sum(1 for t in bucket if t.pnl > 0)
        losses = n - wins
        total_win = sum(t.pnl for t in bucket if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in bucket if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999.0
        avg_pnl = sum(t.pnl for t in bucket) / n if n > 0 else 0.0
        best = max(t.pnl for t in bucket) if bucket else 0.0
        worst = min(t.pnl for t in bucket) if bucket else 0.0

        stats[name] = {
            "count": n,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / n if n > 0 else 0.0,
            "profit_factor": round(pf, 2),
            "avg_pnl": round(avg_pnl, 2),
            "best": round(best, 2),
            "worst": round(worst, 2),
        }

    return stats


def print_pattern_stats(trades) -> None:
    """Print a formatted per-pattern stats table."""
    stats = compute_pattern_stats(trades)
    if not stats:
        print("No trades with pattern context found.")
        return

    print(f"\n{'Pattern':<14} {'Trades':>6} {'WR':>6} {'PF':>6} {'AvgPnL':>8} {'Best':>8} {'Worst':>8}")
    print("-" * 58)
    for name, s in stats.items():
        print(f"{name:<14} {s['count']:>6} {s['win_rate']:>5.0%} {s['profit_factor']:>6.2f} "
              f"${s['avg_pnl']:>7.2f} ${s['best']:>7.2f} ${s['worst']:>7.2f}")

    # Totals
    total_trades = sum(s["count"] for s in stats.values())
    total_wins = sum(s["wins"] for s in stats.values())
    total_win_pnl = sum(s["best"] * s["count"] for s in stats.values())  # rough
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    print("-" * 58)
    print(f"{'TOTAL':<14} {total_trades:>6} {overall_wr:>5.0%}")
