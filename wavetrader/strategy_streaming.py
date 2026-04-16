"""
Strategy-based live streaming engine.

Replaces model inference in the streaming loop with:
  Strategy.evaluate() → AIConfirmer.confirm() → LLM Arbiter → OANDA execute

Reuses OANDA client, candle polling, position management, and trailing
stop logic from the existing streaming infrastructure.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, DEFAULT_RISK_SCALING
from .oanda import Candle, OANDAClient, OANDAConfig, tf_to_granularity
from .strategies.base import BaseStrategy, IndicatorBundle
from .strategies.indicators import compute_all_indicators
from .strategies.ai_confirmer import AIConfirmer
from .types import Signal, TradeSignal

logger = logging.getLogger(__name__)

_TF_SECONDS = {
    "1min": 60, "5min": 300, "15min": 900,
    "30min": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}

_HISTORY_BARS = {
    "1min": 300, "5min": 250, "15min": 200,
    "30min": 200, "1h": 200, "4h": 200, "1d": 100,
}

_PIP_SIZE = {"GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01, "GBP/USD": 0.0001}
_PIP_VALUE = {"GBP/JPY": 6.5, "EUR/JPY": 6.5, "USD/JPY": 7.0, "GBP/USD": 10.0}
_LOT_SIZE = 100_000


def _candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    rows = [{
        "date": c.timestamp, "open": c.open, "high": c.high,
        "low": c.low, "close": c.close, "volume": float(c.volume),
    } for c in candles]
    return pd.DataFrame(rows)


class StrategyStreamingEngine:
    """Live trading engine driven by a rule-based strategy + AI confirmation."""

    def __init__(
        self,
        strategy: BaseStrategy,
        oanda_demo: OANDAClient,
        pair: str = "GBP/JPY",
        ai_confirmer: Optional[AIConfirmer] = None,
        bt_config: Optional[BacktestConfig] = None,
        oanda_live: Optional[OANDAClient] = None,
    ) -> None:
        self.strategy = strategy
        self.ai_confirmer = ai_confirmer
        self.oanda_demo = oanda_demo
        self.oanda_live = oanda_live
        self.pair = pair
        self.bt_config = bt_config or BacktestConfig()

        self.entry_tf = strategy.meta.entry_timeframe
        self.timeframes = strategy.meta.timeframes

        # State
        self.bar_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.signals_generated = 0
        self.signals_confirmed = 0

        # Position tracking
        self.open_trade_id: Optional[str] = None
        self.open_trade_direction: Optional[Signal] = None
        self._bars_since_close: int = getattr(self.bt_config, "cooldown_bars", 3)
        self._recent_ranges: deque = deque(maxlen=20)

        # Candle history per timeframe
        self._history: Dict[str, pd.DataFrame] = {}

        # LLM Arbiter (optional)
        self._arbiter = None
        arbiter_enabled = os.environ.get("LLM_ARBITER_ENABLED", "true").lower() == "true"
        if arbiter_enabled:
            try:
                from .llm_arbiter import LLMArbiter, LLMArbiterConfig
                arbiter_cfg = LLMArbiterConfig(
                    enabled=True,
                    authority_mode=os.environ.get("LLM_AUTHORITY_MODE", "advisory"),
                    model=os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
                )
                self._arbiter = LLMArbiter(arbiter_cfg)
                logger.info("LLM Arbiter enabled (strategy voice mode)")
            except Exception as e:
                logger.warning("LLM Arbiter init failed: %s", e)

        self._last_bar_time: Optional[datetime] = None
        self._running = False

    # ── Warmup ────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Fetch historical candles for all required timeframes."""
        for tf in self.timeframes:
            try:
                gran = tf_to_granularity(tf)
                n = _HISTORY_BARS.get(tf, 200)
                candles = self.oanda_demo.get_latest_candles(self.pair, gran, n)
                if candles:
                    self._history[tf] = _candles_to_df(candles)
                    logger.info("Warmed up %s: %d bars", tf, len(candles))
            except Exception as e:
                logger.warning("Warmup %s failed: %s", tf, e)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main polling loop."""
        self._running = True
        poll_interval = _TF_SECONDS.get(self.entry_tf, 900)

        logger.info(
            "Strategy streaming started: %s on %s (entry=%s, poll=%ds)",
            self.strategy.meta.name, self.pair, self.entry_tf, poll_interval,
        )

        while self._running:
            try:
                gran = tf_to_granularity(self.entry_tf)
                candles = self.oanda_demo.get_latest_candles(self.pair, gran, 2)
                if candles and len(candles) >= 2:
                    latest = candles[-2]  # Last complete candle
                    if self._last_bar_time is None or latest.timestamp > self._last_bar_time:
                        self._process_bar(latest)
            except Exception as e:
                logger.error("Strategy loop error: %s", e)

            time.sleep(min(poll_interval // 3, 30))

    def stop(self) -> None:
        self._running = False

    # ── Process bar ───────────────────────────────────────────────────────

    def _process_bar(self, candle: Candle) -> None:
        """Process a new entry-TF bar using strategy evaluation."""
        self.bar_count += 1
        self._bars_since_close += 1
        self._last_bar_time = candle.timestamp

        # Update entry-TF history
        new_row = pd.DataFrame([{
            "date": candle.timestamp, "open": candle.open,
            "high": candle.high, "low": candle.low,
            "close": candle.close, "volume": float(candle.volume),
        }])
        if self.entry_tf in self._history:
            self._history[self.entry_tf] = pd.concat(
                [self._history[self.entry_tf], new_row], ignore_index=True,
            ).tail(_HISTORY_BARS.get(self.entry_tf, 200))
        else:
            self._history[self.entry_tf] = new_row

        self._recent_ranges.append(candle.high - candle.low)

        # Refresh higher TFs
        self._refresh_higher_tfs()

        # Volatility check
        if self._is_volatility_halted(candle.high, candle.low):
            logger.warning("Volatility halt — skipping")
            return

        # Skip if we have an open position
        if self.open_trade_id:
            return

        # Cooldown
        if self._bars_since_close < getattr(self.bt_config, "cooldown_bars", 3):
            return

        # Need enough history
        if self.entry_tf not in self._history or len(self._history[self.entry_tf]) < 50:
            return

        # ── Strategy evaluation ───────────────────────────────────────────
        try:
            indicators = compute_all_indicators(
                self._history, entry_tf=self.entry_tf, pair=self.pair,
            )
            bar_idx = len(self._history[self.entry_tf]) - 1
            setup = self.strategy.evaluate(self._history, indicators, bar_idx)
        except Exception as e:
            logger.error("Strategy evaluate error: %s", e)
            return

        if setup is None:
            return

        self.signals_generated += 1
        if setup.timestamp is None:
            setup.timestamp = candle.timestamp
        if setup.entry_price == 0.0:
            setup.entry_price = candle.close

        logger.info(
            "Strategy signal: %s %s conf=%.3f — %s",
            self.strategy.meta.name, setup.direction.name,
            setup.confidence, setup.reason[:80],
        )

        # ── AI confirmation ───────────────────────────────────────────────
        trade_signal = None
        if self.ai_confirmer is not None:
            confirmed = self.ai_confirmer.confirm(setup, self._history)
            if confirmed is None:
                logger.info("AI Confirmer REJECTED signal")
                return
            self.signals_confirmed += 1
            trade_signal = confirmed.trade_signal
        else:
            self.signals_confirmed += 1
            trade_signal = TradeSignal(
                signal=setup.direction,
                confidence=setup.confidence,
                entry_price=setup.entry_price,
                stop_loss=setup.sl_pips,
                take_profit=setup.tp_pips,
                trailing_stop_pct=setup.trailing_stop_pct,
                timestamp=setup.timestamp,
            )

        # ── LLM Arbiter ──────────────────────────────────────────────────
        if self._arbiter and self._arbiter.config.enabled:
            try:
                from .llm_arbiter import ArbiterContext
                ctx = ArbiterContext(
                    signal=trade_signal.signal.name,
                    confidence=trade_signal.confidence,
                    alignment=0.0,
                    sl_pips=trade_signal.stop_loss,
                    tp_pips=trade_signal.take_profit,
                    entry_price=trade_signal.entry_price,
                    model_id="strategy",
                    pair=self.pair,
                    timeframe=self.entry_tf,
                    strategy_id=self.strategy.meta.id,
                    strategy_name=self.strategy.meta.name,
                    strategy_author=self.strategy.meta.author,
                    strategy_reason=setup.reason,
                    strategy_context=setup.context,
                )
                decision = self._arbiter.evaluate(ctx)
                if decision.action == "VETO":
                    logger.info("LLM VETOED: %s", decision.reasoning[:80])
                    return
                if decision.narrative:
                    logger.info("LLM Narrative: %s", decision.narrative[:120])
            except Exception as e:
                logger.error("Arbiter failed: %s — proceeding", e)

        # ── Execute ───────────────────────────────────────────────────────
        self._execute_signal(trade_signal, candle)

    def _execute_signal(self, signal: TradeSignal, candle: Candle) -> None:
        """Execute via OANDA."""
        if signal.signal == Signal.HOLD:
            return
        if signal.confidence < self.bt_config.min_confidence:
            return

        pip = _PIP_SIZE.get(self.pair, 0.01)
        pip_val = _PIP_VALUE.get(self.pair, 6.5)
        risk_amount = self.bt_config.initial_balance * self.bt_config.risk_per_trade
        lot = risk_amount / max(signal.stop_loss * pip_val, 1e-9)
        lot = max(0.01, min(5.0, round(lot, 2)))

        units = int(lot * _LOT_SIZE)
        if signal.signal == Signal.SELL:
            units = -units

        if signal.signal == Signal.BUY:
            sl_price = round(candle.close - signal.stop_loss * pip, 5)
            tp_price = round(candle.close + signal.take_profit * pip, 5)
        else:
            sl_price = round(candle.close + signal.stop_loss * pip, 5)
            tp_price = round(candle.close - signal.take_profit * pip, 5)

        try:
            result = self.oanda_demo.place_market_order(
                self.pair, units, sl_price=sl_price, tp_price=tp_price,
            )
            if result:
                self.open_trade_id = result.get("id")
                self.open_trade_direction = signal.signal
                self.total_trades += 1
                self._bars_since_close = 0
                logger.info(
                    "OPENED %s %s %.2f lots @ %.3f SL=%.3f TP=%.3f",
                    signal.signal.name, self.pair, lot,
                    candle.close, sl_price, tp_price,
                )
        except Exception as e:
            logger.error("OANDA order failed: %s", e)

    def _refresh_higher_tfs(self) -> None:
        entry_secs = _TF_SECONDS.get(self.entry_tf, 900)
        for tf in self.timeframes:
            if tf == self.entry_tf:
                continue
            tf_secs = _TF_SECONDS.get(tf, 3600)
            interval = max(1, tf_secs // entry_secs)
            if self.bar_count % interval == 0:
                try:
                    gran = tf_to_granularity(tf)
                    candles = self.oanda_demo.get_latest_candles(
                        self.pair, gran, _HISTORY_BARS.get(tf, 200),
                    )
                    if candles:
                        self._history[tf] = _candles_to_df(candles)
                except Exception as e:
                    logger.warning("Failed to refresh %s: %s", tf, e)

    def _is_volatility_halted(self, high: float, low: float) -> bool:
        if len(self._recent_ranges) < 5:
            return False
        mean_range = np.mean(list(self._recent_ranges))
        return (high - low) > self.bt_config.atr_halt_multiplier * mean_range

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "strategy": self.strategy.meta.id,
            "strategy_name": self.strategy.meta.name,
            "pair": self.pair,
            "bar_count": self.bar_count,
            "signals_generated": self.signals_generated,
            "signals_confirmed": self.signals_confirmed,
            "total_trades": self.total_trades,
            "has_open_trade": self.open_trade_id is not None,
        }
