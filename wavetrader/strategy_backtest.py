"""
Strategy-aware bar-by-bar backtest engine.

Replaces model-based inference with:
  Strategy.evaluate() → AIConfirmer.confirm() → BacktestEngine.open_position()

Uses the same BacktestEngine for position management, trailing stops, and
circuit breakers — all existing analytics and equity curve output unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .backtest import BacktestEngine, _print_results
from .config import BacktestConfig
from .strategies.base import BaseStrategy, IndicatorBundle
from .strategies.indicators import compute_all_indicators
from .strategies.ai_confirmer import AIConfirmer, ConfirmedSignal
from .types import BacktestResults, Signal


def run_strategy_backtest(
    strategy: BaseStrategy,
    candles: Dict[str, pd.DataFrame],
    bt_config: Optional[BacktestConfig] = None,
    ai_confirmer: Optional[AIConfirmer] = None,
    pair: str = "GBP/JPY",
    indicator_recompute_interval: int = 1,
    verbose: bool = True,
) -> BacktestResults:
    """Run a full bar-by-bar strategy backtest.

    Args:
        strategy: An instantiated BaseStrategy subclass.
        candles: Dict of TF → DataFrame with [date, open, high, low, close, volume].
        bt_config: Backtest configuration (defaults, risk, account size, etc.).
        ai_confirmer: Optional AI confirmer (WaveTrader MTF). If None, strategies
                      trade without AI confirmation.
        pair: Currency pair for metadata.
        indicator_recompute_interval: How often to recompute indicators (every N bars).
        verbose: Print progress and results.

    Returns:
        BacktestResults with trades, equity curve, and aggregate stats.
    """
    bt_config = bt_config or BacktestConfig()
    # Auto-detect pip_size from pair if caller didn't override
    if bt_config.pip_size == 0.01 and "USD" in pair.split("/")[-1]:
        bt_config.pip_size = 0.0001
    elif bt_config.pip_size == 0.0001 and "JPY" in pair:
        bt_config.pip_size = 0.01
    engine = BacktestEngine(bt_config)
    entry_tf = strategy.meta.entry_timeframe

    # Graceful TF fallback: if the entry TF isn't in the data, pick the
    # finest available timeframe that *is* present.
    _tf_order = ["1min", "5min", "15min", "30min", "1h", "4h", "1d"]
    if entry_tf not in candles:
        available = [t for t in _tf_order if t in candles]
        if not available:
            raise ValueError(
                f"Strategy requires entry TF '{entry_tf}' but candles only have: "
                f"{list(candles.keys())}"
            )
        entry_tf = available[0]  # finest available
        if verbose:
            print(f"[backtest] entry TF '{strategy.meta.entry_timeframe}' not in data — "
                  f"falling back to '{entry_tf}'")

    base_df = candles[entry_tf]
    n_bars = len(base_df)

    # Pre-compute indicators once for the full dataset
    indicators = compute_all_indicators(
        candles, entry_tf=entry_tf, pair=pair, compute_amd=True,
    )

    # Reset strategy state (important for strategies with internal state like ORB)
    strategy.reset()

    if verbose:
        print("\n" + "=" * 70)
        print(f"STRATEGY BACKTEST: {strategy.meta.name}")
        print(f"Author: {strategy.meta.author}  |  Pair: {pair}  |  Entry TF: {entry_tf}")
        print(f"Bars: {n_bars}  |  Initial Balance: ${bt_config.initial_balance:,.2f}")
        print(f"AI Confirmer: {'ON' if ai_confirmer else 'OFF'}")
        print("=" * 70)

    # ── Track strategy signals for analysis ──────────────────────────────
    signals_generated = 0
    signals_confirmed = 0
    signals_rejected = 0

    # ── Main loop: iterate over entry-TF bars ────────────────────────────
    # Skip first 200 bars for indicator warmup
    start_bar = max(200, 0)

    for i in range(start_bar, n_bars):
        bar = base_df.iloc[i]
        timestamp = bar["date"] if "date" in bar.index else datetime.utcnow()

        # 1. Register bar for volatility circuit breaker
        engine.record_bar(bar["high"], bar["low"])

        # 2. Update any open trade (trailing SL, check SL/TP hits)
        if engine.open_trade is not None:
            closed = engine.update_trade(
                bar["high"], bar["low"], bar["close"], timestamp,
            )
            # Trade was closed on this bar — continue to next bar
            # (don't open a new position on the same bar a trade closes)
            if closed is not None:
                continue

        # 3. If flat, evaluate strategy
        if engine.open_trade is None:
            # Update the current bar index in the indicator bundle
            indicators.current_bar_idx = i

            setup = strategy.evaluate(candles, indicators, i)

            if setup is not None:
                signals_generated += 1

                # Set timestamp if strategy didn't
                if setup.timestamp is None:
                    setup.timestamp = timestamp

                # Set entry price to current close if not set
                if setup.entry_price == 0.0:
                    setup.entry_price = bar["close"]

                # 4. AI confirmation (if enabled)
                if ai_confirmer is not None:
                    confirmed = ai_confirmer.confirm(setup, candles)
                    if confirmed is None:
                        signals_rejected += 1
                        continue
                    signals_confirmed += 1
                    trade_signal = confirmed.trade_signal
                else:
                    # No AI confirmer — use strategy signal directly
                    signals_confirmed += 1
                    from .types import TradeSignal
                    trade_signal = TradeSignal(
                        signal=setup.direction,
                        confidence=setup.confidence,
                        entry_price=setup.entry_price,
                        stop_loss=setup.sl_pips,
                        take_profit=setup.tp_pips,
                        trailing_stop_pct=setup.trailing_stop_pct,
                        timestamp=timestamp,
                        exit_mode=getattr(setup, 'exit_mode', 'tp_sl'),
                        context=getattr(setup, 'context', {}),
                        tp_levels=getattr(setup, 'tp_levels', []),
                    )

                # 5. Open position via engine
                engine.open_position(
                    trade_signal,
                    bar["close"],
                    timestamp,
                    current_high=bar["high"],
                    current_low=bar["low"],
                )

        # Progress reporting
        if verbose and (i - start_bar + 1) % 2000 == 0:
            print(
                f"  Bar {i+1:>6}/{n_bars}  "
                f"Trades: {len(engine.closed_trades):>4}  "
                f"Balance: ${engine.balance:>10,.2f}  "
                f"Signals: {signals_generated} gen / {signals_confirmed} conf"
            )

    # ── Close any open trade at end of data ──────────────────────────────
    if engine.open_trade is not None:
        last = base_df.iloc[-1]
        engine.close_position(
            last["close"],
            last["date"] if "date" in last.index else datetime.utcnow(),
            "End of Backtest",
        )

    results = engine.get_results()

    if verbose:
        _print_results(results, bt_config.initial_balance)
        print(f"\nStrategy Signals: {signals_generated} generated, "
              f"{signals_confirmed} confirmed, {signals_rejected} rejected")

    return results
