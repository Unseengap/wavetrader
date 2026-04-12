"""
Backtest engine for WaveFollower — trend-following with pyramid (add) positions.

Key differences from the base BacktestEngine:
  1. **Pyramiding**: up to ``max_adds`` add-on positions when the model's
     ``add_score`` is high and the trend remains intact.
  2. **Structure-break exits**: closes the *entire* position (initial +
     add-ons) when HH/HL → LL/LH structure break is detected, OR when
     the model emits a signal reversal.
  3. **Pair-agnostic**: pip size / value are computed from the pair string
     at runtime rather than hard-coded.
  4. **Opposite-signal exits**: when the model flips from BUY→SELL or
     SELL→BUY, the whole pyramid is closed and a new trend position opens.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .config import BacktestConfig, DEFAULT_RISK_SCALING
from .dataset import MTFForexDataset
from .types import BacktestResults, Signal, Trade, TradeSignal


# ─────────────────────────────────────────────────────────────────────────────
# Pair-agnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

_PIP_SIZE = {
    "GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01,
    "GBP/USD": 0.0001, "EUR/USD": 0.0001, "AUD/USD": 0.0001,
}
_PIP_VALUE = {
    "GBP/JPY": 6.5, "EUR/JPY": 6.7, "USD/JPY": 6.5,
    "GBP/USD": 10.0, "EUR/USD": 10.0, "AUD/USD": 10.0,
}


def _pip_size(pair: str) -> float:
    return _PIP_SIZE.get(pair, 0.01 if "JPY" in pair else 0.0001)


def _pip_value(pair: str) -> float:
    return _PIP_VALUE.get(pair, 6.5 if "JPY" in pair else 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Pyramid position — groups initial entry + add-on legs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PyramidLeg:
    """One leg of a pyramid (initial or add-on)."""
    entry_price: float
    size: float
    entry_time: datetime
    leg_index: int = 0


@dataclass
class PyramidPosition:
    """Group of legs all in the same direction (one effective position)."""
    direction: Signal
    legs: List[PyramidLeg] = field(default_factory=list)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_pct: float = 0.0
    current_sl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    _min_trail_abs: float = 0.0

    @property
    def total_size(self) -> float:
        return sum(leg.size for leg in self.legs)

    @property
    def avg_entry(self) -> float:
        if not self.legs:
            return 0.0
        total_notional = sum(leg.entry_price * leg.size for leg in self.legs)
        return total_notional / self.total_size

    @property
    def n_legs(self) -> int:
        return len(self.legs)


# ─────────────────────────────────────────────────────────────────────────────
# TrendFollowerEngine
# ─────────────────────────────────────────────────────────────────────────────

class TrendFollowerEngine:
    """
    Bar-by-bar backtest engine with pyramid support for the WaveFollower model.

    Each bar:
      1. Record bar range for volatility circuit breaker.
      2. If position open: update trailing SL, check SL/TP hits.
      3. If position open AND model says add_score > threshold: add a leg.
      4. If position open AND signal reverses: close all, open new direction.
      5. If flat: run inference, open initial position if signal != HOLD.
    """

    def __init__(
        self,
        pair: str = "GBP/JPY",
        config: Optional[BacktestConfig] = None,
        max_adds: int = 2,
        add_threshold: float = 0.65,
    ) -> None:
        self.pair = pair
        self.config = config or BacktestConfig()
        self.max_adds = max_adds
        self.add_threshold = add_threshold

        self.pip = _pip_size(pair)
        self.pip_val = _pip_value(pair)

        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.peak_equity = self.config.initial_balance
        self.max_drawdown = 0.0

        self.position: Optional[PyramidPosition] = None
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [self.config.initial_balance]
        self._recent_ranges: deque = deque(maxlen=20)
        self._min_trail_abs: float = DEFAULT_RISK_SCALING.min_trail_pips * self.pip

    # ── Circuit breakers ─────────────────────────────────────────────────

    def record_bar(self, high: float, low: float) -> None:
        self._recent_ranges.append(high - low)

    def _is_volatility_halted(self, high: float, low: float) -> bool:
        if len(self._recent_ranges) < 5:
            return False
        mean_range = np.mean(list(self._recent_ranges))
        return (high - low) > self.config.atr_halt_multiplier * mean_range

    def _risk_multiplier(self) -> float:
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        return 0.5 if dd >= self.config.drawdown_reduce_threshold else 1.0

    # ── Position sizing (same formula as base engine, but pair-aware) ───

    def _lot_size(self, sl_pips: float) -> float:
        effective_risk = self.config.risk_per_trade * self._risk_multiplier()
        risk_amount = self.balance * effective_risk
        lot = risk_amount / max(sl_pips * self.pip_val, 1e-9)
        lot = max(0.01, min(5.0, lot))
        margin_required = (lot * 100_000) / self.config.leverage
        if margin_required > self.balance * 0.5:
            lot = (self.balance * 0.5 * self.config.leverage) / 100_000
        return round(lot, 2)

    # ── Open initial position ────────────────────────────────────────────

    def open_position(
        self,
        signal: Signal,
        confidence: float,
        current_price: float,
        sl_pips: float,
        tp_pips: float,
        trailing_pct: float,
        timestamp: datetime,
        high: float,
        low: float,
    ) -> bool:
        """Open a new pyramid position (first leg). Returns True if opened."""
        if self.position is not None:
            return False
        if signal == Signal.HOLD:
            return False
        if confidence < self.config.min_confidence:
            return False
        if self._is_volatility_halted(high, low):
            return False

        spread = self.config.spread_pips * self.pip
        if signal == Signal.BUY:
            entry = current_price + spread / 2
            sl = entry - sl_pips * self.pip
            tp = entry + tp_pips * self.pip
        else:
            entry = current_price - spread / 2
            sl = entry + sl_pips * self.pip
            tp = entry - tp_pips * self.pip

        lot = self._lot_size(sl_pips)
        self.balance -= self.config.commission_per_lot * lot

        self.position = PyramidPosition(
            direction=signal,
            legs=[PyramidLeg(entry_price=entry, size=lot, entry_time=timestamp, leg_index=0)],
            stop_loss=sl,
            take_profit=tp,
            trailing_stop_pct=trailing_pct,
            current_sl=sl,
            highest_price=entry,
            lowest_price=entry,
            _min_trail_abs=self._min_trail_abs,
        )
        return True

    # ── Add to position (pyramid) ────────────────────────────────────────

    def add_to_position(
        self,
        add_score: float,
        current_price: float,
        sl_pips: float,
        timestamp: datetime,
    ) -> bool:
        """
        Add a leg to an existing position on a pullback.
        The add_score comes from the model's PullbackHead.
        Returns True if a leg was added.
        """
        if self.position is None:
            return False
        if self.position.n_legs > self.max_adds:
            return False
        if add_score < self.add_threshold:
            return False

        # Only add if price has pulled back from the peak (proving it's a pullback)
        pos = self.position
        if pos.direction == Signal.BUY:
            pullback_depth = pos.highest_price - current_price
            if pullback_depth < 5 * self.pip:  # Need at least 5 pips of pullback
                return False
        else:
            pullback_depth = current_price - pos.lowest_price
            if pullback_depth < 5 * self.pip:
                return False

        lot = self._lot_size(sl_pips)
        self.balance -= self.config.commission_per_lot * lot

        pos.legs.append(PyramidLeg(
            entry_price=current_price,
            size=lot,
            entry_time=timestamp,
            leg_index=pos.n_legs,
        ))

        # Tighten SL to the new leg's SL level (protective)
        if pos.direction == Signal.BUY:
            new_sl = current_price - sl_pips * self.pip
            pos.current_sl = max(pos.current_sl, new_sl)
        else:
            new_sl = current_price + sl_pips * self.pip
            pos.current_sl = min(pos.current_sl, new_sl)

        return True

    # ── Update (trailing, SL/TP check) ──────────────────────────────────

    def update_bar(
        self, high: float, low: float, close: float, timestamp: datetime,
    ) -> Optional[Trade]:
        """
        Advance one bar. Returns a closed Trade if the position was stopped
        out or hit TP.
        """
        if self.position is None:
            self._update_equity(close)
            return None

        pos = self.position

        if pos.direction == Signal.BUY:
            # Update peak
            if high > pos.highest_price:
                pos.highest_price = high
                if pos.trailing_stop_pct > 0:
                    initial_risk = pos.avg_entry - pos.stop_loss
                    trail_dist = initial_risk * (1.0 - pos.trailing_stop_pct)
                    trail_dist = max(trail_dist, self._min_trail_abs)
                    new_sl = pos.highest_price - trail_dist
                    if new_sl > pos.current_sl:
                        pos.current_sl = new_sl

            # SL hit
            if low <= pos.current_sl:
                return self._close_all(pos.current_sl, timestamp, "Stop Loss")
            # TP hit
            if high >= pos.take_profit:
                return self._close_all(pos.take_profit, timestamp, "Take Profit")

        else:  # SELL
            if low < pos.lowest_price:
                pos.lowest_price = low
                if pos.trailing_stop_pct > 0:
                    initial_risk = pos.stop_loss - pos.avg_entry
                    trail_dist = initial_risk * (1.0 - pos.trailing_stop_pct)
                    trail_dist = max(trail_dist, self._min_trail_abs)
                    new_sl = pos.lowest_price + trail_dist
                    if new_sl < pos.current_sl:
                        pos.current_sl = new_sl

            if high >= pos.current_sl:
                return self._close_all(pos.current_sl, timestamp, "Stop Loss")
            if low <= pos.take_profit:
                return self._close_all(pos.take_profit, timestamp, "Take Profit")

        self._update_equity(close)
        return None

    # ── Close on signal reversal ─────────────────────────────────────────

    def close_on_reversal(
        self, close: float, timestamp: datetime,
    ) -> Optional[Trade]:
        """Close the entire pyramid when the model flips direction."""
        if self.position is None:
            return None
        return self._close_all(close, timestamp, "Signal Reversal")

    def close_on_structure_break(
        self, close: float, timestamp: datetime,
    ) -> Optional[Trade]:
        """Close the entire pyramid on a structure break."""
        if self.position is None:
            return None
        return self._close_all(close, timestamp, "Structure Break")

    # ── Internal close ───────────────────────────────────────────────────

    def _close_all(
        self, exit_price: float, timestamp: datetime, reason: str,
    ) -> Trade:
        """Close the entire pyramid (all legs) at exit_price."""
        pos = self.position
        total_pnl = 0.0

        for leg in pos.legs:
            if pos.direction == Signal.BUY:
                pips = (exit_price - leg.entry_price) / self.pip
            else:
                pips = (leg.entry_price - exit_price) / self.pip
            total_pnl += pips * self.pip_val * leg.size

        self.balance += total_pnl

        # Record as a single Trade for results compatibility
        trade = Trade(
            entry_time=pos.legs[0].entry_time,
            entry_price=pos.avg_entry,
            direction=pos.direction,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            trailing_stop_pct=pos.trailing_stop_pct,
            size=pos.total_size,
            exit_time=timestamp,
            exit_price=exit_price,
            pnl=total_pnl,
            exit_reason=f"{reason} ({pos.n_legs} legs)",
        )
        self.closed_trades.append(trade)
        self.position = None
        self._update_equity(exit_price)
        return trade

    # ── Equity tracking ──────────────────────────────────────────────────

    def _update_equity(self, current_price: float) -> None:
        unrealized = 0.0
        if self.position:
            pos = self.position
            for leg in pos.legs:
                if pos.direction == Signal.BUY:
                    unrealized += (current_price - leg.entry_price) / self.pip * self.pip_val * leg.size
                else:
                    unrealized += (leg.entry_price - current_price) / self.pip * self.pip_val * leg.size

        self.equity = self.balance + unrealized
        self.equity_curve.append(self.equity)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    # ── Results ──────────────────────────────────────────────────────────

    def get_results(self) -> BacktestResults:
        if not self.closed_trades:
            return BacktestResults(
                final_balance=self.balance,
                equity_curve=self.equity_curve,
            )

        winning = [t for t in self.closed_trades if t.pnl > 0]
        losing = [t for t in self.closed_trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        returns = [t.pnl / self.config.initial_balance for t in self.closed_trades]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if len(returns) > 1 and np.std(returns) > 0
            else 0.0
        )

        return BacktestResults(
            total_trades=len(self.closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=sum(t.pnl for t in self.closed_trades),
            max_drawdown=self.max_drawdown,
            win_rate=len(winning) / len(self.closed_trades),
            profit_factor=profit_factor,
            sharpe_ratio=round(sharpe, 3),
            final_balance=self.balance,
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner (mirrors backtest.run_backtest but for WaveFollower)
# ─────────────────────────────────────────────────────────────────────────────

def run_wavefollower_backtest(
    model: Any,
    df: Dict[str, pd.DataFrame],
    config: "WaveFollowerConfig",
    bt_config: Optional[BacktestConfig] = None,
    pair: str = "GBP/JPY",
    device: Optional[torch.device] = None,
    max_adds: int = 2,
    add_threshold: float = 0.65,
) -> BacktestResults:
    """
    Run a full bar-by-bar backtest using WaveFollower with pyramid positions.

    The model is expected to output:
      signal_logits, trend_logits, confidence, alignment,
      add_score, risk_params
    """
    from .wave_follower import WaveFollowerConfig

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bt_config = bt_config or BacktestConfig()
    engine = TrendFollowerEngine(
        pair=pair, config=bt_config,
        max_adds=max_adds, add_threshold=add_threshold,
    )

    # Build MTF dataset using the WaveFollower's config
    dataset = MTFForexDataset(df, config, lookahead=10, pair=pair)
    base_df = dataset.prepared[dataset.entry_tf]

    model = model.to(device)
    model.eval()

    rs = DEFAULT_RISK_SCALING

    print("\n" + "=" * 70)
    print(f"WAVEFOLLOWER BACKTEST: {pair}  TFs={config.timeframes}")
    print(f"Initial Balance : ${bt_config.initial_balance:,.2f}")
    print(f"Risk per Trade  : {bt_config.risk_per_trade:.1%}")
    print(f"Pyramiding      : max {max_adds} adds, threshold {add_threshold:.0%}")
    print("=" * 70)

    with torch.no_grad():
        for i in range(len(dataset)):
            actual = dataset.valid_indices[i]
            bar = base_df.iloc[actual]
            timestamp = bar["date"] if "date" in bar.index else datetime.utcnow()

            engine.record_bar(bar["high"], bar["low"])

            # Update open position
            if engine.position:
                closed = engine.update_bar(
                    bar["high"], bar["low"], bar["close"], timestamp,
                )
                if closed:
                    pass  # Position was stopped out or hit TP

            # Run inference every bar (to detect reversals and add-ons)
            sample = dataset[i]
            model_input = {
                tf: {k: v.unsqueeze(0).to(device) for k, v in sample[tf].items()
                     if isinstance(v, torch.Tensor)}
                for tf in config.timeframes
                if tf in sample and isinstance(sample[tf], dict)
            }

            out = model(model_input)
            sig_idx = out["signal_logits"].argmax(-1).item()
            signal = Signal(sig_idx)
            conf = out["confidence"].item()
            risk = out["risk_params"][0]
            add_score = out["add_score"].item()
            trend_idx = out["trend_logits"].argmax(-1).item()

            sl_pips = rs.sl_pips(float(risk[0].item()))
            tp_pips = rs.tp_pips(float(risk[1].item()))
            trailing_pct = rs.trailing_pct(float(risk[2].item()))

            # ── Decision logic ──────────────────────────────────────

            if engine.position:
                pos = engine.position

                # Signal reversal → close all + open opposite
                if signal != Signal.HOLD and signal != pos.direction:
                    if conf >= bt_config.min_confidence:
                        engine.close_on_reversal(bar["close"], timestamp)
                        # Open in new direction immediately
                        engine.open_position(
                            signal, conf, bar["close"],
                            sl_pips, tp_pips, trailing_pct,
                            timestamp, bar["high"], bar["low"],
                        )
                # Same direction + add score high → pyramid
                elif signal == pos.direction and add_score >= add_threshold:
                    engine.add_to_position(
                        add_score, bar["close"], sl_pips, timestamp,
                    )
            else:
                # Flat → open new position
                if signal != Signal.HOLD and conf >= bt_config.min_confidence:
                    engine.open_position(
                        signal, conf, bar["close"],
                        sl_pips, tp_pips, trailing_pct,
                        timestamp, bar["high"], bar["low"],
                    )

            if (i + 1) % 1000 == 0:
                n_legs = engine.position.n_legs if engine.position else 0
                print(
                    f"  Bar {i + 1:>6}/{len(dataset)}  "
                    f"Trades: {len(engine.closed_trades):>4}  "
                    f"Balance: ${engine.balance:>10,.2f}  "
                    f"Legs: {n_legs}"
                )

    # Close any remaining position
    if engine.position:
        last = base_df.iloc[-1]
        engine._close_all(
            last["close"],
            last["date"] if "date" in last.index else datetime.utcnow(),
            "End of Backtest",
        )

    results = engine.get_results()
    _print_wf_results(results, bt_config.initial_balance)
    return results


def _print_wf_results(results: BacktestResults, initial: float) -> None:
    print("\n" + "=" * 70)
    print("WAVEFOLLOWER BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total Trades   : {results.total_trades}")
    print(f"Winning        : {results.winning_trades}")
    print(f"Losing         : {results.losing_trades}")
    print(f"Win Rate       : {results.win_rate:.1%}")
    print("-" * 40)
    print(f"Total P&L      : ${results.total_pnl:,.2f}")
    print(f"Final Balance  : ${results.final_balance:,.2f}")
    print(f"Return         : {(results.final_balance / initial - 1) * 100:.1f}%")
    print("-" * 40)
    print(f"Max Drawdown   : {results.max_drawdown:.1%}")
    print(f"Profit Factor  : {results.profit_factor:.2f}")
    print(f"Sharpe Ratio   : {results.sharpe_ratio:.2f}")
    print("=" * 70)

    if results.trades:
        # Count pyramid stats
        legs = [int(t.exit_reason.split("(")[-1].split(" ")[0]) if "legs" in t.exit_reason else 1
                for t in results.trades]
        avg_legs = np.mean(legs) if legs else 1
        max_legs = max(legs) if legs else 1
        print(f"\nPyramid stats  : avg {avg_legs:.1f} legs, max {max_legs}")

        print("\nLast 10 Trades:")
        print("-" * 70)
        for t in results.trades[-10:]:
            side = "BUY " if t.direction == Signal.BUY else "SELL"
            pnl = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
            print(
                f"  {side} @ {t.entry_price:.3f} → {t.exit_price:.3f}  "
                f"size={t.size:.2f}  {t.exit_reason:25s}  {pnl}"
            )
