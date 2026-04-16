"""
News Catalyst Order Block Strategy v4 — by Dectrick McGee

Inspired by Ali Trades' 3-pillar approach: fundamental bias + order block
precision + news volatility catalysts.

V4 Key Change: NEXT-BAR CONFIRMATION
Instead of entering on the OB retest bar itself, we wait one bar:
  - Bar N: price retests OB zone (dips in) → stored as PENDING
  - Bar N+1: price confirms bounce (closes outside zone) → ENTRY
This eliminates false retests where price just passes through the zone.

SL is based on the retest bar's extreme (structural level) rather than
pure ATR, giving tighter stops at meaningful levels.

Pillar 1 — Macro Bias: 4H EMA alignment (20>50>200 or inverse).
Pillar 2 — Order Blocks: 15min OB detection, retest + next-bar confirmation.
Pillar 3 — Session + Momentum: London/NY hours, RSI alignment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


class _OB:
    __slots__ = ("direction", "hi", "lo", "idx", "impulse", "tested")
    def __init__(self, direction: str, hi: float, lo: float, idx: int, impulse: float):
        self.direction = direction
        self.hi = hi
        self.lo = lo
        self.idx = idx
        self.impulse = impulse
        self.tested = False


class _Pending:
    """Pending OB retest awaiting next-bar confirmation."""
    __slots__ = ("ob", "direction", "bar_idx", "bar_high", "bar_low", "atr", "atr_pips")
    def __init__(self, ob: _OB, direction: Signal, bar_idx: int,
                 bar_high: float, bar_low: float, atr: float, atr_pips: float):
        self.ob = ob
        self.direction = direction
        self.bar_idx = bar_idx
        self.bar_high = bar_high
        self.bar_low = bar_low
        self.atr = atr
        self.atr_pips = atr_pips


class NewsCatalystOBStrategy(BaseStrategy):

    meta = StrategyMeta(
        id="news_catalyst_ob",
        name="News Catalyst OB",
        author="Dectrick McGee",
        version="4.0.0",
        description="Macro bias + 15min OB retest + next-bar confirmation",
        category="scalper",
        timeframes=["15min", "1h", "4h", "1d"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="15min",
    )

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Pillar 1: Macro bias ─────────────────────────────────────
            "require_4h_ema_align": True,
            "htf_trend_bias_min": 0.1,

            # ── Pillar 2: 15min Order Blocks ─────────────────────────────
            "impulse_atr_mult": 0.8,        # Body > 0.8 × ATR → impulse
            "ob_max_age_bars": 384,          # 96 hours = 4 days
            "ob_retest_tolerance": 0.6,      # Price enters top 60% of zone
            "ob_max_active": 10,
            "ob_body_pct_min": 0.35,         # OB candle body:range ratio

            # ── V4: Confirmation ─────────────────────────────────────────
            "require_confirmation": True,    # Wait for next bar to confirm
            "confirm_body_atr_min": 0.3,     # Confirm bar body > 0.3 × ATR

            # ── Pillar 3: Session + Entry ────────────────────────────────
            "session_start_hour": 13,        # Overlap hours only (best edge)
            "session_end_hour": 17,
            "overlap_start_hour": 13,
            "overlap_end_hour": 17,
            "overlap_confidence_bonus": 0.08,
            "use_rsi_filter": True,
            "rsi_buy_max": 65,
            "rsi_sell_min": 35,

            # ── Risk management ──────────────────────────────────────────
            "min_rr_ratio": 3.0,            # 3R target — clean wins via TP
            "trailing_stop_pct": 0.50,       # Trail tightens 50% of risk
            "sl_buffer_atr": 0.3,           # Buffer below retest low for SL
            "min_sl_pips": 10.0,
            "max_sl_pips": 60.0,

            # ── Filters ──────────────────────────────────────────────────
            "min_confidence": 0.45,
            "min_atr_pips": 3.0,
            "min_adx": 12.0,
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            ParamSpec("require_4h_ema_align", "Require 4H EMA Align", "bool",
                      True, description="4H EMA stack determines direction"),
            ParamSpec("htf_trend_bias_min", "HTF Trend Bias", "float",
                      0.1, 0.0, 0.8, 0.05, "Min 4H structure bias"),
            ParamSpec("impulse_atr_mult", "Impulse ATR Mult", "float",
                      0.8, 0.3, 2.0, 0.1, "15min body > N × ATR = impulse"),
            ParamSpec("ob_max_age_bars", "OB Max Age (bars)", "int",
                      384, 48, 576, 48, "Max 15min bars before OB expires"),
            ParamSpec("ob_retest_tolerance", "OB Retest Tolerance", "float",
                      0.6, 0.2, 1.0, 0.1, "Price enters top N% of OB zone"),
            ParamSpec("require_confirmation", "Next-Bar Confirm", "bool",
                      True, description="Wait for bar after retest to confirm bounce"),
            ParamSpec("confirm_body_atr_min", "Confirm Body Min", "float",
                      0.3, 0.0, 1.0, 0.1, "Confirmation bar body > N × ATR"),
            ParamSpec("min_rr_ratio", "Min R:R", "float",
                      2.0, 1.0, 5.0, 0.5, "Reward:risk target"),
            ParamSpec("trailing_stop_pct", "Trailing Stop %", "float",
                      0.35, 0.1, 0.8, 0.05, "Trail fraction"),
            ParamSpec("sl_buffer_atr", "SL Buffer (ATR)", "float",
                      0.3, 0.1, 1.0, 0.1, "Buffer below retest bar extreme for SL"),
            ParamSpec("min_sl_pips", "Min SL", "float",
                      10.0, 5.0, 25.0, 1.0, "Floor SL pips"),
            ParamSpec("max_sl_pips", "Max SL", "float",
                      60.0, 20.0, 100.0, 5.0, "Cap SL pips"),
            ParamSpec("min_confidence", "Min Confidence", "float",
                      0.45, 0.2, 0.8, 0.05, "Confidence threshold"),
            ParamSpec("min_adx", "Min ADX", "float",
                      12.0, 5.0, 40.0, 2.0, "Trend strength"),
            ParamSpec("rsi_buy_max", "RSI Buy Max", "float",
                      65, 55, 80, 5, "Avoid buying overbought"),
            ParamSpec("rsi_sell_min", "RSI Sell Min", "float",
                      35, 20, 45, 5, "Avoid selling oversold"),
            ParamSpec("session_start_hour", "Session Start", "int",
                      7, 0, 23, 1, "UTC hour"),
            ParamSpec("session_end_hour", "Session End", "int",
                      20, 0, 23, 1, "UTC hour"),
        ]

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self._bull_obs: List[_OB] = []
        self._bear_obs: List[_OB] = []
        self._pending: Optional[_Pending] = None

    def reset(self) -> None:
        self._bull_obs = []
        self._bear_obs = []
        self._pending = None

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
        c = float(bar["close"])
        h = float(bar["high"])
        l = float(bar["low"])
        o = float(bar["open"])
        pip = 0.01 if "JPY" in indicators.pair else 0.0001

        # ATR
        atr_a = indicators.atr.get(etf)
        if atr_a is None or i >= len(atr_a):
            return None
        atr = float(atr_a[i])
        if np.isnan(atr) or atr <= 0:
            return None
        atr_pips = atr / pip
        if atr_pips < p["min_atr_pips"]:
            return None

        # Session
        ts = bar["date"] if "date" in bar.index else None
        hour = 12
        if ts is not None:
            hour = pd.Timestamp(ts).hour
            if hour < p["session_start_hour"] or hour >= p["session_end_hour"]:
                self._pending = None  # Clear pending if out of session
                return None

        # ═══ PILLAR 1: Macro Bias ═════════════════════════════════════
        d = self._macro_bias(indicators, i, candles)
        if d is None:
            return None

        # ═══ PILLAR 2: OB Detection + Maintenance ════════════════════
        self._detect_obs(df, i, atr)
        self._maintain(i, c)

        # ═══ CHECK PENDING CONFIRMATION (V4) ══════════════════════════
        setup = None
        if p["require_confirmation"] and self._pending is not None:
            pend = self._pending
            self._pending = None  # Consume (will re-set if new retest found)

            # Only confirm if from previous bar and direction still matches
            if pend.bar_idx == i - 1 and pend.direction == d:
                confirmed = False
                if d == Signal.BUY and c > pend.ob.hi:
                    confirmed = True
                elif d == Signal.SELL and c < pend.ob.lo:
                    confirmed = True

                # Confirmation bar must have meaningful body
                if confirmed and p["confirm_body_atr_min"] > 0:
                    body = abs(c - o)
                    if body < atr * p["confirm_body_atr_min"]:
                        confirmed = False

                # Confirmation bar must close in the right direction
                if confirmed:
                    if d == Signal.BUY and c <= o:
                        confirmed = False
                    elif d == Signal.SELL and c >= o:
                        confirmed = False

                if confirmed:
                    setup = self._build_confirmed_setup(
                        d, c, h, l, o, i, hour, atr, atr_pips, pip,
                        pend, indicators,
                    )

        # ═══ FIND NEW RETESTS ════════════════════════════════════════
        ob = self._find_retest(d, c, h, l, i)
        if ob is not None:
            if p["require_confirmation"]:
                # Store as pending — don't enter yet
                self._pending = _Pending(ob, d, i, h, l, atr, atr_pips)
                ob.tested = True
            elif setup is None:
                # Direct entry (V3 fallback when confirmation disabled)
                setup = self._build_direct_setup(
                    d, c, h, l, o, i, hour, atr, atr_pips, pip,
                    ob, indicators,
                )

        return setup

    # ─── Build confirmed entry (V4) ──────────────────────────────────

    def _build_confirmed_setup(
        self, d: Signal, c: float, h: float, l: float, o: float,
        i: int, hour: int, atr: float, atr_pips: float, pip: float,
        pend: _Pending, indicators: IndicatorBundle,
    ) -> Optional[StrategySetup]:
        p = self.params
        etf = self.meta.entry_timeframe

        # RSI filter
        if p["use_rsi_filter"]:
            rsi_a = indicators.rsi.get(etf)
            if rsi_a is not None and i < len(rsi_a):
                rsi = float(rsi_a[i])
                if not np.isnan(rsi):
                    if d == Signal.BUY and rsi > p["rsi_buy_max"]:
                        return None
                    if d == Signal.SELL and rsi < p["rsi_sell_min"]:
                        return None

        # ADX filter
        adx_a = indicators.adx.get(etf)
        adx = 20.0
        if adx_a is not None and i < len(adx_a):
            v = float(adx_a[i])
            if not np.isnan(v):
                adx = v
                if adx < p["min_adx"]:
                    return None

        # Confidence
        ob = pend.ob
        conf = 0.50
        conf += min(ob.impulse / (atr * 3), 0.12)
        if p["overlap_start_hour"] <= hour < p["overlap_end_hour"]:
            conf += p["overlap_confidence_bonus"]
        if adx > 25:
            conf += 0.05
        if indicators.sr_zones is not None and i < len(indicators.sr_zones):
            if float(indicators.sr_zones[i, 0]) > 0.7:
                conf += 0.06
        if indicators.engulfing is not None and i < len(indicators.engulfing):
            eng = float(indicators.engulfing[i, 0])
            if (d == Signal.BUY and eng > 0) or (d == Signal.SELL and eng < 0):
                conf += 0.08
        if "4h" in indicators.structure:
            idx4 = min(i // 16, len(indicators.structure["4h"]) - 1)
            if idx4 >= 0:
                bias = float(indicators.structure["4h"][idx4, 7])
                if (d == Signal.BUY and bias > 0.3) or (d == Signal.SELL and bias < -0.3):
                    conf += 0.05
        conf = min(conf, 0.95)
        if conf < p["min_confidence"]:
            return None

        # SL: below the retest bar's extreme + buffer
        buf = atr * p["sl_buffer_atr"]
        if d == Signal.BUY:
            sl_price = pend.bar_low - buf
            sl_dist = c - sl_price
        else:
            sl_price = pend.bar_high + buf
            sl_dist = sl_price - c

        if sl_dist <= 0:
            return None

        sl = sl_dist / pip
        sl = max(sl, p["min_sl_pips"])
        sl = min(sl, p["max_sl_pips"])
        tp = sl * p["min_rr_ratio"]

        in_ovl = p["overlap_start_hour"] <= hour < p["overlap_end_hour"]
        return StrategySetup(
            direction=d, entry_price=c,
            sl_pips=round(sl, 1), tp_pips=round(tp, 1),
            confidence=round(conf, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            reason=f"{'Bull' if d == Signal.BUY else 'Bear'} OB confirmed "
                   f"@ {ob.lo:.2f}–{ob.hi:.2f} | ovl={in_ovl}",
        )

    # ─── Build direct entry (V3 fallback) ────────────────────────────

    def _build_direct_setup(
        self, d: Signal, c: float, h: float, l: float, o: float,
        i: int, hour: int, atr: float, atr_pips: float, pip: float,
        ob: _OB, indicators: IndicatorBundle,
    ) -> Optional[StrategySetup]:
        p = self.params
        etf = self.meta.entry_timeframe

        # Strong close filter (V3 behavior)
        bar_range = h - l
        if bar_range < 1e-10:
            return None
        close_pos = (c - l) / bar_range
        if d == Signal.BUY and close_pos < 0.65:
            return None
        if d == Signal.SELL and close_pos > 0.35:
            return None

        # RSI filter
        if p["use_rsi_filter"]:
            rsi_a = indicators.rsi.get(etf)
            if rsi_a is not None and i < len(rsi_a):
                rsi = float(rsi_a[i])
                if not np.isnan(rsi):
                    if d == Signal.BUY and rsi > p["rsi_buy_max"]:
                        return None
                    if d == Signal.SELL and rsi < p["rsi_sell_min"]:
                        return None

        # ADX filter
        adx_a = indicators.adx.get(etf)
        adx = 20.0
        if adx_a is not None and i < len(adx_a):
            v = float(adx_a[i])
            if not np.isnan(v):
                adx = v
                if adx < p["min_adx"]:
                    return None

        conf = 0.50
        conf += min(ob.impulse / (atr * 3), 0.12)
        if p["overlap_start_hour"] <= hour < p["overlap_end_hour"]:
            conf += p["overlap_confidence_bonus"]
        if adx > 25:
            conf += 0.05
        conf = min(conf, 0.95)
        if conf < p["min_confidence"]:
            return None

        # ATR-based SL (V3 behavior)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        sl = atr_pips * sl_atr_mult
        sl = max(sl, p["min_sl_pips"])
        sl = min(sl, p["max_sl_pips"])
        tp = sl * p["min_rr_ratio"]

        ob.tested = True
        in_ovl = p["overlap_start_hour"] <= hour < p["overlap_end_hour"]
        return StrategySetup(
            direction=d, entry_price=c,
            sl_pips=round(sl, 1), tp_pips=round(tp, 1),
            confidence=round(conf, 3),
            trailing_stop_pct=p["trailing_stop_pct"],
            reason=f"{'Bull' if d == Signal.BUY else 'Bear'} OB "
                   f"@ {ob.lo:.2f}–{ob.hi:.2f} | ovl={in_ovl}",
        )

    # ─── Macro bias ──────────────────────────────────────────────────

    def _macro_bias(self, ind: IndicatorBundle, i: int,
                    candles: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        p = self.params
        if p["require_4h_ema_align"] and "4h" in candles:
            ix = min(i // 16, len(ind.ema_20.get("4h", [])) - 1)
            if ix < 0:
                return None
            a20, a50, a200 = ind.ema_20.get("4h"), ind.ema_50.get("4h"), ind.ema_200.get("4h")
            if a20 is None or a50 is None or a200 is None:
                return None
            if ix >= len(a20) or ix >= len(a50) or ix >= len(a200):
                return None
            e20, e50, e200 = float(a20[ix]), float(a50[ix]), float(a200[ix])
            if np.isnan(e20) or np.isnan(e50) or np.isnan(e200):
                return None
            if e20 > e50 > e200:
                d = Signal.BUY
            elif e20 < e50 < e200:
                d = Signal.SELL
            else:
                return None
        else:
            a20 = ind.ema_20.get("15min")
            a50 = ind.ema_50.get("15min")
            a200 = ind.ema_200.get("15min")
            if a20 is None or a50 is None or a200 is None:
                return None
            if i >= len(a20):
                return None
            e20, e50, e200 = float(a20[i]), float(a50[i]), float(a200[i])
            if np.isnan(e20) or np.isnan(e50) or np.isnan(e200):
                return None
            if e20 > e50 > e200:
                d = Signal.BUY
            elif e20 < e50 < e200:
                d = Signal.SELL
            else:
                return None

        if p["htf_trend_bias_min"] > 0 and "4h" in ind.structure:
            ix = min(i // 16, len(ind.structure["4h"]) - 1)
            if ix >= 0:
                b = float(ind.structure["4h"][ix, 7])
                if d == Signal.BUY and b < p["htf_trend_bias_min"]:
                    return None
                if d == Signal.SELL and b > -p["htf_trend_bias_min"]:
                    return None
        return d

    # ─── OB detection on 15min ───────────────────────────────────────

    def _detect_obs(self, df: pd.DataFrame, i: int, atr: float) -> None:
        p = self.params
        if i < 3:
            return
        # Check bar i-1 (impulse) and i-2 (OB candidate)
        imp = df.iloc[i - 1]
        cand = df.iloc[i - 2]

        ic, io = float(imp["close"]), float(imp["open"])
        ih, il = float(imp["high"]), float(imp["low"])
        ibody = abs(ic - io)
        irange = ih - il

        if ibody < atr * p["impulse_atr_mult"]:
            return
        if irange > 0 and ibody / irange < 0.4:
            return

        cc, co = float(cand["close"]), float(cand["open"])
        ch, cl = float(cand["high"]), float(cand["low"])
        crange = ch - cl
        cbody = abs(cc - co)
        if crange > 0 and cbody / crange < p["ob_body_pct_min"]:
            return

        # Green impulse + Red OB = Bullish OB
        if ic > io and cc < co:
            self._bull_obs.append(_OB("bull", ch, cl, i - 2, ibody))
            if len(self._bull_obs) > p["ob_max_active"]:
                self._bull_obs = self._bull_obs[-p["ob_max_active"]:]
        # Red impulse + Green OB = Bearish OB
        if ic < io and cc > co:
            self._bear_obs.append(_OB("bear", ch, cl, i - 2, ibody))
            if len(self._bear_obs) > p["ob_max_active"]:
                self._bear_obs = self._bear_obs[-p["ob_max_active"]:]

    def _maintain(self, i: int, close: float) -> None:
        """Expire old OBs. Invalidate only on CLOSE through zone (not wicks)."""
        age = self.params["ob_max_age_bars"]
        self._bull_obs = [
            ob for ob in self._bull_obs
            if not ob.tested and (i - ob.idx) <= age and close >= ob.lo
        ]
        self._bear_obs = [
            ob for ob in self._bear_obs
            if not ob.tested and (i - ob.idx) <= age and close <= ob.hi
        ]

    def _find_retest(self, d: Signal, c: float, h: float,
                     l: float, i: int) -> Optional[_OB]:
        tol = self.params["ob_retest_tolerance"]
        hits: List[_OB] = []
        if d == Signal.BUY:
            for ob in self._bull_obs:
                if ob.tested:
                    continue
                depth = ob.hi - ob.lo
                if depth <= 0:
                    continue
                lvl = ob.hi - depth * tol
                # Bar's low dips into OB zone, close stays in upper portion
                if l <= ob.hi and c >= lvl:
                    hits.append(ob)
        else:
            for ob in self._bear_obs:
                if ob.tested:
                    continue
                depth = ob.hi - ob.lo
                if depth <= 0:
                    continue
                lvl = ob.lo + depth * tol
                if h >= ob.lo and c <= lvl:
                    hits.append(ob)
        if not hits:
            return None
        hits.sort(key=lambda o: (o.impulse, -o.idx), reverse=True)
        return hits[0]
