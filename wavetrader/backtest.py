"""
Backtesting engine: realistic forex simulation with spread, commission,
dynamic position sizing, and trailing stop management.

Assumes GBP/JPY-style pair conventions by default; adjust BacktestConfig
for other pairs (pip_value, spread_pips, etc.).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import BacktestConfig, SignalConfig, DEFAULT_RISK_SCALING
from .dataset import ForexDataset
from .types import BacktestResults, Signal, Trade, TradeSignal


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven bar-by-bar backtest.

    Each bar:
      1. Update any open trade (trailing SL, check SL/TP hits).
      2. If flat, run model inference and optionally open a new position.

    Position sizing: fixed-fractional risk management
      lot_size = (balance x risk_pct) / (SL_pips x pip_value)

    Circuit breakers (configurable in BacktestConfig):
      - Volatility halt: skip trade if current bar's H-L range exceeds
        `atr_halt_multiplier` times the 20-bar rolling mean range.
      - Drawdown step-down: halve risk_per_trade when live drawdown exceeds
        `drawdown_reduce_threshold`.

    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config      = config or BacktestConfig()
        self.balance     = self.config.initial_balance
        self.equity      = self.config.initial_balance
        self.open_trade: Optional[Trade]    = None
        self.closed_trades                  = []
        self.equity_curve                   = [self.config.initial_balance]
        self.peak_equity                    = self.config.initial_balance
        self.max_drawdown                   = 0.0
        self._recent_ranges: deque = deque(maxlen=20)
        self._bars_in_trade: int = 0
        self._min_trail_abs: float = DEFAULT_RISK_SCALING.min_trail_pips * 0.01  # absolute price

    # ── Circuit breakers ───────────────────────────────────────────────

    def record_bar(self, high: float, low: float) -> None:
        """Register a bar's high-low range. Call once per bar regardless of position."""
        self._recent_ranges.append(high - low)

    def _is_volatility_halted(self, current_high: float, current_low: float) -> bool:
        """True when current bar range exceeds atr_halt_multiplier * rolling mean."""
        if len(self._recent_ranges) < 5:
            return False
        mean_range   = np.mean(list(self._recent_ranges))
        current_range = current_high - current_low
        return current_range > self.config.atr_halt_multiplier * mean_range

    def _risk_multiplier(self) -> float:
        """Returns 0.5 when current drawdown exceeds threshold, else 1.0."""
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        return 0.5 if dd >= self.config.drawdown_reduce_threshold else 1.0

    # ── Position sizing ────────────────────────────────────────────────────

    def _lot_size(self, sl_pips: float, confidence: float = 1.0) -> float:
        effective_risk  = self.config.risk_per_trade * self._risk_multiplier()
        risk_amount     = self.balance * effective_risk
        lot             = risk_amount / max(sl_pips * self.config.pip_value, 1e-9)

        lot             = max(0.01, min(5.0, lot))
        margin_required = (lot * 100_000) / self.config.leverage
        if margin_required > self.balance * 0.5:
            lot = (self.balance * 0.5 * self.config.leverage) / 100_000
        return round(lot, 2)

    # ── Open ───────────────────────────────────────────────────────────────

    def open_position(
        self,
        signal:        TradeSignal,
        current_price: float,
        timestamp:     datetime,
        current_high:  Optional[float] = None,
        current_low:   Optional[float] = None,
    ) -> Optional[Trade]:
        if self.open_trade is not None:
            return None
        if signal.signal == Signal.HOLD:
            return None
        if signal.confidence < self.config.min_confidence:
            return None
        # Volatility circuit breaker
        if (
            current_high is not None
            and current_low is not None
            and self._is_volatility_halted(current_high, current_low)
        ):
            return None
        spread = self.config.spread_pips * 0.01
        pip    = 0.01   # GBP/JPY convention

        if signal.signal == Signal.BUY:
            entry = current_price + spread / 2
            sl    = entry - signal.stop_loss * pip
            tp    = entry + signal.take_profit * pip
        else:
            entry = current_price - spread / 2
            sl    = entry + signal.stop_loss * pip
            tp    = entry - signal.take_profit * pip

        lot = self._lot_size(signal.stop_loss, signal.confidence)
        self.balance -= self.config.commission_per_lot * lot

        self.open_trade = Trade(
            entry_time=timestamp,
            entry_price=entry,
            direction=signal.signal,
            stop_loss=sl,
            take_profit=tp,
            trailing_stop_pct=signal.trailing_stop_pct,
            size=lot,
            exit_mode=getattr(signal, 'exit_mode', 'tp_sl'),
        )
        self._bars_in_trade = 0
        return self.open_trade

    # ── Update ────────────────────────────────────────────────────────────

    def update_trade(
        self,
        current_high:  float,
        current_low:   float,
        current_close: float,
        timestamp:     datetime,
    ) -> Optional[Trade]:
        """
        Advance the open trade by one bar.
        Returns the closed Trade if SL or TP was hit, else None.
        """
        if self.open_trade is None:
            return None

        t = self.open_trade
        self._bars_in_trade += 1
        activate_r = self.config.trail_activate_r

        is_opposite_exit = getattr(t, 'exit_mode', 'tp_sl') == 'opposite_signal'
        is_geometric = getattr(t, 'exit_mode', 'tp_sl') == 'geometric_trail'
        is_multi_tp = getattr(t, 'exit_mode', 'tp_sl') == 'multi_tp_trail'

        if is_geometric:
            return self._update_geometric_trail(t, current_high, current_low, current_close, timestamp)

        if is_multi_tp:
            return self._update_multi_tp_trail(t, current_high, current_low, current_close, timestamp)

        if t.direction == Signal.BUY:
            if current_high > t.highest_price:
                t.highest_price = current_high
                if t.trailing_stop_pct > 0:
                    initial_risk = t.entry_price - t.stop_loss
                    # Only start trailing after price moves >= activate_r × R in our favor
                    unrealised_r = (t.highest_price - t.entry_price) / max(initial_risk, 1e-9)
                    if unrealised_r >= activate_r:
                        trail_distance = initial_risk * (1.0 - t.trailing_stop_pct)
                        # Floor = 50% of initial risk (scales with trade size)
                        min_trail = initial_risk * 0.5
                        trail_distance = max(trail_distance, min_trail)
                        new_sl = t.highest_price - trail_distance
                        if new_sl > t.current_sl:
                            t.current_sl = new_sl

            if current_low <= t.current_sl:
                reason = "Trailing Stop" if t.current_sl > t.stop_loss else "Stop Loss"
                return self.close_position(t.current_sl, timestamp, reason)
            # Skip TP check for opposite_signal exit mode
            if not is_opposite_exit and current_high >= t.take_profit:
                return self.close_position(t.take_profit, timestamp, "Take Profit")

        else:  # SELL
            if current_low < t.lowest_price:
                t.lowest_price = current_low
                if t.trailing_stop_pct > 0:
                    initial_risk = t.stop_loss - t.entry_price
                    # Only start trailing after price moves >= activate_r × R in our favor
                    unrealised_r = (t.entry_price - t.lowest_price) / max(initial_risk, 1e-9)
                    if unrealised_r >= activate_r:
                        trail_distance = initial_risk * (1.0 - t.trailing_stop_pct)
                        min_trail = initial_risk * 0.5
                        trail_distance = max(trail_distance, min_trail)
                        new_sl = t.lowest_price + trail_distance
                        if new_sl < t.current_sl:
                            t.current_sl = new_sl

            if current_high >= t.current_sl:
                reason = "Trailing Stop" if t.current_sl < t.stop_loss else "Stop Loss"
                return self.close_position(t.current_sl, timestamp, reason)
            # Skip TP check for opposite_signal exit mode
            if not is_opposite_exit and current_low <= t.take_profit:
                return self.close_position(t.take_profit, timestamp, "Take Profit")

        self._update_equity(current_close)
        return None

    # ── Geometric Trailing Stop ───────────────────────────────────────────

    def _update_geometric_trail(
        self,
        t: Trade,
        current_high: float,
        current_low: float,
        current_close: float,
        timestamp: datetime,
    ) -> Optional[Trade]:
        """
        Geometric ratcheting trailing stop:
          - No TP — ride winners via trailing stop only.
          - Activate when unrealized PnL >= 1R (the dollar risk on this trade).
          - Trail distance = trailing_stop_pct × initial SL distance.
          - Floor locks at each R multiple: 1R, 2R, 4R, 8R, 16R...
          - SL never moves backward.
        """
        pip = 0.01  # JPY pairs
        pip_value = self.config.pip_value

        if t.direction == Signal.BUY:
            initial_risk_price = t.entry_price - t.stop_loss  # SL distance in price
            trail_pct = t.trailing_stop_pct if t.trailing_stop_pct > 0 else 0.5
            trail_distance = initial_risk_price * trail_pct
            t.highest_price = max(t.highest_price, current_high)

            # Dollar risk on this trade (1R)
            risk_pips = initial_risk_price / pip
            one_r = risk_pips * pip_value * t.size

            # Unrealized PnL at peak
            peak_pips = (t.highest_price - t.entry_price) / pip
            peak_pnl = peak_pips * pip_value * t.size

            if one_r > 0 and peak_pnl >= one_r:
                # Trail activated: SL follows peak at trail_distance
                trail_sl = t.highest_price - trail_distance

                # R-step floor: lock at every R (1R, 2R, 3R, 4R...)
                # At 3.5R, floor = 3R. Guarantees whole-R profit on pullback.
                r_multiple = peak_pnl / one_r
                floor_pnl = int(r_multiple) * one_r
                t.trail_lock_pnl = max(t.trail_lock_pnl, floor_pnl)

                # Floor SL: guarantee locked PnL minimum
                floor_pips = t.trail_lock_pnl / max(pip_value * t.size, 1e-9)
                floor_sl = t.entry_price + floor_pips * pip
                new_sl = max(trail_sl, floor_sl)

                if new_sl > t.current_sl:
                    t.current_sl = new_sl

            # Exit check
            if current_low <= t.current_sl:
                reason = "Trailing Stop" if t.current_sl > t.stop_loss else "Stop Loss"
                return self.close_position(t.current_sl, timestamp, reason)

        else:  # SELL
            initial_risk_price = t.stop_loss - t.entry_price
            trail_pct = t.trailing_stop_pct if t.trailing_stop_pct > 0 else 0.5
            trail_distance = initial_risk_price * trail_pct
            t.lowest_price = min(t.lowest_price, current_low)

            risk_pips = initial_risk_price / pip
            one_r = risk_pips * pip_value * t.size

            trough_pips = (t.entry_price - t.lowest_price) / pip
            trough_pnl = trough_pips * pip_value * t.size

            if one_r > 0 and trough_pnl >= one_r:
                trail_sl = t.lowest_price + trail_distance

                # R-step floor: 1R, 2R, 3R, 4R...
                r_multiple = trough_pnl / one_r
                floor_pnl = int(r_multiple) * one_r
                t.trail_lock_pnl = max(t.trail_lock_pnl, floor_pnl)

                floor_pips = t.trail_lock_pnl / max(pip_value * t.size, 1e-9)
                floor_sl = t.entry_price - floor_pips * pip
                new_sl = min(trail_sl, floor_sl)

                if new_sl < t.current_sl:
                    t.current_sl = new_sl

            # Exit check
            if current_high >= t.current_sl:
                reason = "Trailing Stop" if t.current_sl < t.stop_loss else "Stop Loss"
                return self.close_position(t.current_sl, timestamp, reason)

        self._update_equity(current_close)
        return None

    # ── Multi-TP Geometric Trail ──────────────────────────────────────────

    def _update_multi_tp_trail(
        self,
        t: Trade,
        current_high: float,
        current_low: float,
        current_close: float,
        timestamp: datetime,
    ) -> Optional[Trade]:
        """
        Multi-TP with trailing runner:
          - TP1 at 1R: close 25%, move SL to break-even
          - TP2 at 2R: close 25%, SL locks at 1R
          - TP3 at 3R: close 25%, SL locks at 2R
          - Remaining 25%: rides geometric trailing stop
        """
        pip = 0.01  # JPY pairs
        pip_value = self.config.pip_value

        if t.direction == Signal.BUY:
            initial_risk_price = t.entry_price - t.stop_loss
            t.highest_price = max(t.highest_price, current_high)

            risk_pips = initial_risk_price / pip
            one_r = risk_pips * pip_value * t.original_size  # Based on original size

            # Current unrealized R at peak
            peak_pips = (t.highest_price - t.entry_price) / pip
            peak_r = (peak_pips * pip_value * t.original_size) / one_r if one_r > 0 else 0

            # Check partial TPs
            tp_levels = self.config.multi_tp_levels
            while t.partial_closes < len(tp_levels) and one_r > 0:
                target_r, portion = tp_levels[t.partial_closes]
                tp_price = t.entry_price + target_r * initial_risk_price
                if current_high >= tp_price:
                    # Partial close: bank portion of the position
                    close_size = t.original_size * portion
                    close_pips = (tp_price - t.entry_price) / pip
                    close_pnl = close_pips * pip_value * close_size
                    t.partial_pnl += close_pnl
                    t.size = round(t.size - close_size, 4)
                    self.balance += close_pnl
                    t.partial_closes += 1

                    # Move SL up: lock at (target_r - 1)R or break-even
                    lock_r = max(target_r - 1.0, 0.0)
                    new_sl = t.entry_price + lock_r * initial_risk_price
                    if new_sl > t.current_sl:
                        t.current_sl = new_sl
                else:
                    break

            # Runner portion: trail with geometric logic if we've taken all partial TPs
            if t.partial_closes >= len(tp_levels) and t.size > 0.005:
                trail_pct = t.trailing_stop_pct if t.trailing_stop_pct > 0 else 0.5
                trail_distance = initial_risk_price * trail_pct

                # Trail from peak
                trail_sl = t.highest_price - trail_distance

                # R-step floor for runner
                runner_peak_pips = (t.highest_price - t.entry_price) / pip
                runner_peak_pnl = runner_peak_pips * pip_value * t.original_size
                if one_r > 0:
                    r_multiple = runner_peak_pnl / one_r
                    floor_r = max(int(r_multiple), len(tp_levels))  # At least last TP level
                    floor_pnl = floor_r * one_r
                    t.trail_lock_pnl = max(t.trail_lock_pnl, floor_pnl)

                    floor_pips = t.trail_lock_pnl / max(pip_value * t.original_size, 1e-9)
                    floor_sl = t.entry_price + floor_pips * pip
                    trail_sl = max(trail_sl, floor_sl)

                if trail_sl > t.current_sl:
                    t.current_sl = trail_sl

            # Exit check
            if current_low <= t.current_sl:
                reason = "Trailing Stop" if t.current_sl > t.stop_loss else "Stop Loss"
                # Add partial PnL to final close
                remaining_pips = (t.current_sl - t.entry_price) / pip
                remaining_pnl = remaining_pips * pip_value * t.size
                t.size = t.original_size  # Restore for PnL calc
                return self._close_multi_tp(t, t.current_sl, timestamp, reason)

        else:  # SELL
            initial_risk_price = t.stop_loss - t.entry_price
            t.lowest_price = min(t.lowest_price, current_low)

            risk_pips = initial_risk_price / pip
            one_r = risk_pips * pip_value * t.original_size

            tp_levels = self.config.multi_tp_levels
            while t.partial_closes < len(tp_levels) and one_r > 0:
                target_r, portion = tp_levels[t.partial_closes]
                tp_price = t.entry_price - target_r * initial_risk_price
                if current_low <= tp_price:
                    close_size = t.original_size * portion
                    close_pips = (t.entry_price - tp_price) / pip
                    close_pnl = close_pips * pip_value * close_size
                    t.partial_pnl += close_pnl
                    t.size = round(t.size - close_size, 4)
                    self.balance += close_pnl
                    t.partial_closes += 1

                    lock_r = max(target_r - 1.0, 0.0)
                    new_sl = t.entry_price - lock_r * initial_risk_price
                    if new_sl < t.current_sl:
                        t.current_sl = new_sl
                else:
                    break

            # Runner trail
            if t.partial_closes >= len(tp_levels) and t.size > 0.005:
                trail_pct = t.trailing_stop_pct if t.trailing_stop_pct > 0 else 0.5
                trail_distance = initial_risk_price * trail_pct
                trail_sl = t.lowest_price + trail_distance

                runner_trough_pips = (t.entry_price - t.lowest_price) / pip
                runner_trough_pnl = runner_trough_pips * pip_value * t.original_size
                if one_r > 0:
                    r_multiple = runner_trough_pnl / one_r
                    floor_r = max(int(r_multiple), len(tp_levels))
                    floor_pnl = floor_r * one_r
                    t.trail_lock_pnl = max(t.trail_lock_pnl, floor_pnl)

                    floor_pips = t.trail_lock_pnl / max(pip_value * t.original_size, 1e-9)
                    floor_sl = t.entry_price - floor_pips * pip
                    trail_sl = min(trail_sl, floor_sl)

                if trail_sl < t.current_sl:
                    t.current_sl = trail_sl

            # Exit check
            if current_high >= t.current_sl:
                reason = "Trailing Stop" if t.current_sl < t.stop_loss else "Stop Loss"
                t.size = t.original_size
                return self._close_multi_tp(t, t.current_sl, timestamp, reason)

        self._update_equity(current_close)
        return None

    def _close_multi_tp(
        self, t: Trade, exit_price: float, timestamp: datetime, reason: str
    ) -> Trade:
        """Close remaining position and combine with partial PnL."""
        t.exit_time = timestamp
        t.exit_price = exit_price
        t.exit_reason = reason

        pip = 0.01
        # Calculate PnL for the remaining runner portion
        tp_levels = self.config.multi_tp_levels
        remaining_size = t.original_size - sum(
            t.original_size * tp_levels[i][1]
            for i in range(min(t.partial_closes, len(tp_levels)))
        )
        remaining_size = max(remaining_size, 0.0)

        if t.direction == Signal.BUY:
            remaining_pips = (exit_price - t.entry_price) / pip
        else:
            remaining_pips = (t.entry_price - exit_price) / pip

        remaining_pnl = remaining_pips * self.config.pip_value * remaining_size

        # Total PnL = partial closes + runner
        t.pnl = t.partial_pnl + remaining_pnl
        t.size = t.original_size  # Restore for display

        self.closed_trades.append(t)
        self.open_trade = None
        self._update_equity(exit_price)
        return t

    # ── Close ─────────────────────────────────────────────────────────────

    def close_position(
        self, exit_price: float, timestamp: datetime, reason: str
    ) -> Trade:
        t = self.open_trade

        # Multi-TP trades have partial PnL already banked
        if getattr(t, 'exit_mode', 'tp_sl') == 'multi_tp_trail' and t.partial_pnl != 0:
            t.size = t.original_size
            return self._close_multi_tp(t, exit_price, timestamp, reason)

        t.exit_time   = timestamp
        t.exit_price  = exit_price
        t.exit_reason = reason

        pip = 0.01
        if t.direction == Signal.BUY:
            pips = (exit_price - t.entry_price) / pip
        else:
            pips = (t.entry_price - exit_price) / pip

        t.pnl      = pips * self.config.pip_value * t.size
        self.balance += t.pnl
        self.closed_trades.append(t)
        self.open_trade = None
        self._update_equity(exit_price)
        return t

    def _update_equity(self, current_price: float) -> None:
        unrealized = 0.0
        if self.open_trade:
            t = self.open_trade
            if t.direction == Signal.BUY:
                unrealized = (current_price - t.entry_price) / 0.01 * self.config.pip_value * t.size
            else:
                unrealized = (t.entry_price - current_price) / 0.01 * self.config.pip_value * t.size

        self.equity = self.balance + unrealized
        self.equity_curve.append(self.equity)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    # ── Results ───────────────────────────────────────────────────────────

    def get_results(self) -> BacktestResults:
        if not self.closed_trades:
            return BacktestResults(
                final_balance=self.balance,
                equity_curve=self.equity_curve,
            )

        winning = [t for t in self.closed_trades if t.pnl > 0]
        losing  = [t for t in self.closed_trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in winning)
        gross_loss   = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        returns = [t.pnl / self.config.initial_balance for t in self.closed_trades]
        sharpe  = (
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
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    model:     Any,
    df:        Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    config:    Union[SignalConfig, 'MTFConfig'],
    bt_config: Optional[BacktestConfig] = None,
    device:    Optional[torch.device]   = None,
) -> BacktestResults:
    """
    Run a full bar-by-bar backtest on `df` using `model` for signal generation.
    Supports single-TF and MTF backtesting.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bt_config = bt_config or BacktestConfig()

    engine = BacktestEngine(bt_config)

    if hasattr(config, 'timeframes'):
        from .dataset import MTFForexDataset
        dataset = MTFForexDataset(df, config, lookahead=10, pair=config.pair)
        base_df = dataset.prepared[dataset.entry_tf]
        is_mtf = True
    else:
        dataset = ForexDataset(df, lookback=config.lookback, lookahead=10, pair=config.pair)
        base_df = dataset.df
        is_mtf = False

    model = model.to(device)
    model.eval()

    print("\n" + "=" * 70)
    print(f"BACKTEST: {config.pair}  {getattr(config, 'timeframe', getattr(config, 'timeframes', 'MTF'))}")
    print(f"Initial Balance : ${bt_config.initial_balance:,.2f}")
    print(f"Risk per Trade  : {bt_config.risk_per_trade:.1%}")
    print("=" * 70)

    with torch.no_grad():
        for i in range(len(dataset)):
            actual      = dataset.valid_indices[i]
            current_bar = base_df.iloc[actual]
            timestamp   = (
                current_bar["date"]
                if "date" in current_bar.index
                else datetime.utcnow()
            )

            # Register bar range for volatility circuit breaker every bar
            engine.record_bar(current_bar["high"], current_bar["low"])

            if engine.open_trade:
                engine.update_trade(
                    current_bar["high"],
                    current_bar["low"],
                    current_bar["close"],
                    timestamp,
                )

            # Run inference when no open trade
            if engine.open_trade is None:
                sample = dataset[i]
                if is_mtf:
                    model_input = {
                        k: {feat: v.unsqueeze(0).to(device) for feat, v in val.items() if isinstance(v, torch.Tensor)} if isinstance(val, dict) else val.to(device) if isinstance(val, torch.Tensor) else val
                        for k, val in sample.items() if k != "label"
                    }
                else:
                    model_input = {
                        k: v.unsqueeze(0).to(device)
                        for k, v in sample.items()
                        if k != "label"
                    }
                
                out      = model(model_input)
                sig_idx  = out["signal_logits"].argmax(-1).item()
                conf     = out["confidence"].item()
                risk     = out["risk_params"][0]

                if sig_idx != Signal.HOLD.value and conf >= bt_config.min_confidence:
                    new_signal = Signal(sig_idx)

                    if engine.open_trade is None:
                        _rs = DEFAULT_RISK_SCALING
                        trade_signal = TradeSignal(
                            signal=new_signal,
                            confidence=conf,
                            entry_price=current_bar["close"],
                            stop_loss=_rs.sl_pips(float(risk[0].item())),
                            take_profit=_rs.tp_pips(float(risk[1].item())),
                            trailing_stop_pct=_rs.trailing_pct(float(risk[2].item())),
                            timestamp=timestamp,
                        )
                        engine.open_position(
                            trade_signal,
                            current_bar["close"],
                            timestamp,
                            current_high=current_bar["high"],
                            current_low=current_bar["low"],
                        )

            if (i + 1) % 1000 == 0:
                print(
                    f"  Bar {i+1:>6}/{len(dataset)}  "
                    f"Trades: {len(engine.closed_trades):>4}  "
                    f"Balance: ${engine.balance:>10,.2f}"
                )

    if engine.open_trade:
        last = base_df.iloc[-1]
        engine.close_position(
            last["close"],
            last["date"] if "date" in last.index else datetime.utcnow(),
            "End of Backtest",
        )

    results = engine.get_results()
    _print_results(results, bt_config.initial_balance)
    return results


def _print_results(results: BacktestResults, initial: float) -> None:
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
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
        print("\nLast 10 Trades:")
        print("-" * 70)
        for t in results.trades[-10:]:
            side = "BUY " if t.direction == Signal.BUY else "SELL"
            pnl  = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
            print(
                f"  {side} @ {t.entry_price:.3f} → {t.exit_price:.3f}  "
                f"{t.exit_reason:12s}  {pnl}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_backtest(
    model:        Any,
    df:           pd.DataFrame,
    config:       SignalConfig,
    bt_config:    Optional[BacktestConfig] = None,
    device:       Optional[torch.device]   = None,
    n_folds:      int   = 5,
    min_train_pct: float = 0.40,
) -> List[BacktestResults]:
    """
    Expanding-window walk-forward backtest.

    The dataset is divided into `n_folds + 1` equal slices.  Each fold trains
    on an expanding prefix and tests on the *next* slice.  This gives a more
    honest performance estimate than a single train/test split.

    Fold layout (example, n_folds=5):
        Fold 1: train [0, 20%)   → test [20%, 40%)
        Fold 2: train [0, 40%)   → test [40%, 60%)   (expanding train)
        Fold 3: train [0, 60%)   → test [60%, 80%)
        ...

    NOTE: This function does NOT retrain the model — it evaluates a fixed
    checkpoint across rolling test windows to assess stability over time.
    To retrain per fold, call train_model() separately with each fold's data
    before passing the retrained model here.

    Args:
        model:          Trained FluxSignal (or any compatible model).
        df:             Full dataset DataFrame (chronological).
        config:         SignalConfig used during training.
        bt_config:      BacktestConfig; defaults to BacktestConfig().
        device:         Torch device; auto-detected if None.
        n_folds:        Number of test folds (default 5).
        min_train_pct:  Minimum fraction of data required for the first fold's
                        training set.  Raises ValueError if violated.

    Returns:
        List of BacktestResults, one per fold (in chronological order).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bt_config  = bt_config or BacktestConfig()
    n          = len(df)
    slice_size = n // (n_folds + 1)

    if slice_size < config.lookback + 100:
        raise ValueError(
            f"walk_forward_backtest: each fold slice has only {slice_size} bars "
            f"(need >= {config.lookback + 100}).  Reduce n_folds or use more data."
        )

    if min_train_pct * n > slice_size:
        raise ValueError(
            f"walk_forward_backtest: first training window ({slice_size} bars) is "
            f"smaller than min_train_pct={min_train_pct:.0%} of total data "
            f"({int(min_train_pct * n)} bars)."
        )

    fold_results: List[BacktestResults] = []

    print(f"\nWalk-Forward Backtest  |  {n_folds} folds  |  ~{slice_size:,} bars/fold")
    print("=" * 70)

    for fold in range(n_folds):
        test_start = (fold + 1) * slice_size
        test_end   = test_start + slice_size
        test_df    = df.iloc[test_start:test_end].reset_index(drop=True)

        print(
            f"\nFold {fold + 1}/{n_folds}  "
            f"train=[0, {test_start:,})  "
            f"test=[{test_start:,}, {test_end:,})"
        )

        engine  = BacktestEngine(bt_config)
        dataset = ForexDataset(
            test_df, lookback=config.lookback, lookahead=10, pair=config.pair
        )

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for i in range(len(dataset)):
                actual      = dataset.valid_indices[i]
                current_bar = dataset.df.iloc[actual]
                timestamp   = (
                    current_bar["date"]
                    if "date" in current_bar.index
                    else datetime.utcnow()
                )

                engine.record_bar(current_bar["high"], current_bar["low"])

                if engine.open_trade:
                    engine.update_trade(
                        current_bar["high"],
                        current_bar["low"],
                        current_bar["close"],
                        timestamp,
                    )

                if engine.open_trade is None:
                    sample = dataset[i]
                    model_input = {
                        k: v.unsqueeze(0).to(device)
                        for k, v in sample.items()
                        if k != "label"
                    }
                    out     = model(model_input)
                    sig_idx = out["signal_logits"].argmax(-1).item()
                    conf    = out["confidence"].item()
                    risk    = out["risk_params"][0]

                    if sig_idx != Signal.HOLD.value and conf >= bt_config.min_confidence:
                        _rs = DEFAULT_RISK_SCALING
                        ts = TradeSignal(
                            signal=Signal(sig_idx),
                            confidence=conf,
                            entry_price=current_bar["close"],
                            stop_loss=_rs.sl_pips(float(risk[0].item())),
                            take_profit=_rs.tp_pips(float(risk[1].item())),
                            trailing_stop_pct=_rs.trailing_pct(float(risk[2].item())),
                            timestamp=timestamp,
                        )
                        engine.open_position(
                            ts,
                            current_bar["close"],
                            timestamp,
                            current_high=current_bar["high"],
                            current_low=current_bar["low"],
                        )

        if engine.open_trade:
            last = dataset.df.iloc[-1]
            engine.close_position(
                last["close"],
                last["date"] if "date" in last.index else datetime.utcnow(),
                "End of Fold",
            )

        results = engine.get_results()
        fold_results.append(results)

        print(
            f"  Trades: {results.total_trades:>4}  "
            f"WR: {results.win_rate:.1%}  "
            f"PF: {results.profit_factor:.2f}  "
            f"Sharpe: {results.sharpe_ratio:.2f}  "
            f"DD: {results.max_drawdown:.1%}"
        )

    # Summary across folds
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)
    avg_sharpe = np.mean([r.sharpe_ratio for r in fold_results])
    avg_wr     = np.mean([r.win_rate for r in fold_results])
    avg_dd     = np.mean([r.max_drawdown for r in fold_results])
    avg_pf     = np.mean([r.profit_factor for r in fold_results])
    print(f"Avg Sharpe     : {avg_sharpe:.2f}")
    print(f"Avg Win Rate   : {avg_wr:.1%}")
    print(f"Avg Max DD     : {avg_dd:.1%}")
    print(f"Avg Profit Factor: {avg_pf:.2f}")
    print("=" * 70)

    return fold_results
