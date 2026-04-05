"""
StreamingEngine — live trading loop for WaveTrader.

Main loop:
  1. Poll OANDA for the latest complete 15m candle
  2. Also fetch 1H, 4H, Daily candles for MTF context
  3. Build feature tensors (same pipeline as dataset.py)
  4. Run model inference → TradeSignal
  5. Execute signal: open/close positions via OANDA REST API
  6. Update trailing stops on open positions
  7. Checkpoint state every N bars
  8. Send monitoring events

Entry point:
  python -m wavetrader.streaming
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .config import BacktestConfig, MTFConfig, ResonanceConfig
from .dataset import ResonanceBuffer, prepare_features
from .model import WaveTraderMTF
from .monitor import Monitor, MonitorConfig
from .oanda import Candle, OANDAClient, OANDAConfig, tf_to_granularity
from .state import LiveState, StateManager
from .types import Signal, TradeSignal
from .copytrade import CopyTradeManager, UserRegistry

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PAIRS = ["GBP/JPY", "EUR/JPY", "GBP/USD", "USD/JPY"]
_DEFAULT_TFS = ["15min", "1h", "4h", "1d"]
_TF_SECONDS = {
    "1min": 60, "5min": 300, "15min": 900,
    "30min": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}

# How many bars of history to keep in memory per timeframe
_HISTORY_BARS = {
    "1min": 300, "5min": 250, "15min": 200,
    "30min": 200, "1h": 200, "4h": 200, "1d": 100,
}

# Pip conventions
_PIP_SIZE = {"GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01, "GBP/USD": 0.0001}
_PIP_VALUE = {"GBP/JPY": 6.5, "EUR/JPY": 6.5, "USD/JPY": 7.0, "GBP/USD": 10.0}
_LOT_SIZE = 100_000  # Standard lot


def _candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    """Convert OANDA candle list to DataFrame matching wavetrader format."""
    rows = []
    for c in candles:
        rows.append({
            "date": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": float(c.volume),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class StreamingEngine:
    """
    Live trading engine for WaveTrader.

    Polls OANDA for new candles, runs inference, and manages positions.
    All state is checkpointed to disk for crash recovery.
    """

    def __init__(
        self,
        model: WaveTraderMTF,
        oanda: OANDAClient,
        pair: str = "GBP/JPY",
        config: Optional[MTFConfig] = None,
        bt_config: Optional[BacktestConfig] = None,
        res_config: Optional[ResonanceConfig] = None,
        checkpoint_dir: str = "/data/checkpoints",
        checkpoint_interval: int = 100,
        monitor: Optional[Monitor] = None,
        paper_trading: bool = True,
        copy_trade_mgr: Optional[CopyTradeManager] = None,
    ) -> None:
        self.model = model
        self.oanda = oanda
        self.pair = pair
        self.copy_trade_mgr = copy_trade_mgr
        self.config = config or MTFConfig(pair=pair)
        self.bt_config = bt_config or BacktestConfig()
        self.res_config = res_config or ResonanceConfig()
        self.paper_trading = paper_trading
        self.checkpoint_interval = checkpoint_interval

        # State
        self.balance = self.bt_config.initial_balance
        self.equity = self.bt_config.initial_balance
        self.peak_equity = self.bt_config.initial_balance
        self.max_drawdown = 0.0
        self.bar_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # Current position tracking
        self.open_trade_id: Optional[str] = None
        self.open_trade_direction: Optional[Signal] = None
        self.open_trade_entry: Optional[float] = None
        self.open_trade_sl: Optional[float] = None
        self.open_trade_tp: Optional[float] = None
        self.open_trade_trailing_pct: float = 0.0
        self.open_trade_peak: float = 0.0  # For trailing stop

        # Circuit breaker state
        self._recent_ranges: deque = deque(maxlen=20)

        # Per-timeframe candle history (DataFrames)
        self._history: Dict[str, pd.DataFrame] = {}

        # Resonance buffer
        self.resonance = ResonanceBuffer(
            capacity=self.res_config.capacity,
            wave_dim=self.config.output_wave_dim,
        )

        # State persistence
        self.state_mgr = StateManager(checkpoint_dir)

        # Monitoring
        self.monitor = monitor

        # Last processed candle timestamp (to detect new bars)
        self._last_bar_time: Optional[datetime] = None

        # Shutdown flag
        self._running = False

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ── Warmup ────────────────────────────────────────────────────────────

    def warmup(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Initialise the engine:
          1. Try to load a checkpoint (restores model + state)
          2. If no checkpoint, fetch historical candles for warmup
          3. Sync with OANDA account state
        """
        # Try loading checkpoint
        ckpt = self.state_mgr.load_checkpoint(checkpoint_path)
        if ckpt:
            self._restore_from_checkpoint(ckpt)
            logger.info("Restored from checkpoint (bar_count=%d)", self.bar_count)
        else:
            logger.info("Cold start — fetching historical candles for warmup")

        # Fetch historical candles for each timeframe
        for tf in self.config.timeframes:
            n_bars = _HISTORY_BARS[tf]
            granularity = tf_to_granularity(tf)
            candles = self.oanda.get_latest_candles(self.pair, granularity, n_bars)
            if candles:
                self._history[tf] = _candles_to_df(candles)
                logger.info("  %s: fetched %d bars", tf, len(candles))
            else:
                logger.warning("  %s: no candles returned", tf)

        # Sync balance from OANDA
        try:
            account = self.oanda.get_account_summary()
            if not self.paper_trading:
                self.balance = account.balance
                self.equity = account.nav
            logger.info(
                "OANDA account: balance=%.2f nav=%.2f open_trades=%d",
                account.balance, account.nav, account.open_trade_count,
            )
        except Exception as e:
            logger.warning("Could not sync OANDA account: %s", e)

        # Check for existing open positions
        try:
            open_trades = self.oanda.get_open_trades(self.pair)
            if open_trades:
                t = open_trades[0]
                self.open_trade_id = t.trade_id
                self.open_trade_direction = Signal.BUY if t.units > 0 else Signal.SELL
                self.open_trade_entry = t.price
                self.open_trade_sl = t.stop_loss
                self.open_trade_tp = t.take_profit
                logger.info("Existing position found: %s %s @ %.3f", t.trade_id, self.open_trade_direction.name, t.price)
        except Exception as e:
            logger.warning("Could not check open trades: %s", e)

        if self._history.get(self.config.entry_timeframe) is not None:
            self._last_bar_time = self._history[self.config.entry_timeframe]["date"].iloc[-1]
            logger.info("Last bar time: %s", self._last_bar_time)

    def _restore_from_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        """Restore all state from a checkpoint dict."""
        self.state_mgr.restore_model(self.model, ckpt)
        self.model.to(self.device)
        self.model.eval()

        ls = ckpt.get("live_state", {})
        self.bar_count = ls.get("bar_count", 0)
        self.balance = ls.get("balance", self.bt_config.initial_balance)
        self.equity = ls.get("equity", self.balance)
        self.peak_equity = ls.get("peak_equity", self.equity)
        self.max_drawdown = ls.get("max_drawdown", 0.0)
        self.total_trades = ls.get("total_trades", 0)
        self.winning_trades = ls.get("winning_trades", 0)
        self.losing_trades = ls.get("losing_trades", 0)
        self.total_pnl = ls.get("total_pnl", 0.0)

        self.open_trade_id = ls.get("open_trade_id")
        if ls.get("open_trade_direction"):
            self.open_trade_direction = Signal[ls["open_trade_direction"]]
        self.open_trade_entry = ls.get("open_trade_entry")

        # Restore resonance buffer
        self.state_mgr.restore_resonance_buffer(self.resonance, ckpt)

        # Restore recent ranges
        for r in ckpt.get("recent_ranges", []):
            self._recent_ranges.append(r)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main polling loop. Runs until SIGINT/SIGTERM.

        Flow per iteration:
          1. Check if market is open
          2. Poll for new 15m candle
          3. If new candle → process bar
          4. Sleep until next expected candle
        """
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("=" * 60)
        logger.info("WaveTrader StreamingEngine STARTED")
        logger.info("  Pair:  %s", self.pair)
        logger.info("  Mode:  %s", "PAPER" if self.paper_trading else "LIVE")
        logger.info("  Device: %s", self.device)
        logger.info("=" * 60)

        if self.monitor:
            self.monitor.send_info(
                f"WaveTrader started — {self.pair} ({'PAPER' if self.paper_trading else 'LIVE'})"
            )

        while self._running:
            try:
                if not self.oanda.is_market_open():
                    logger.debug("Market closed — sleeping 5min")
                    time.sleep(300)
                    continue

                new_candle = self._poll_new_candle()
                if new_candle:
                    self._process_bar(new_candle)
                else:
                    # No new bar yet — wait 30s and re-check
                    time.sleep(30)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception("Error in main loop: %s", e)
                if self.monitor:
                    self.monitor.send_alert(f"Error in main loop: {e}")
                time.sleep(60)  # Back off on error

        self._shutdown()

    def _poll_new_candle(self) -> Optional[Candle]:
        """Check if a new complete candle is available for the entry timeframe."""
        entry_tf = self.config.entry_timeframe
        granularity = tf_to_granularity(entry_tf)
        candles = self.oanda.get_candles(self.pair, granularity, count=2)
        complete = [c for c in candles if c.complete]
        if not complete:
            return None

        latest = complete[-1]
        if self._last_bar_time and latest.timestamp <= self._last_bar_time:
            return None  # Already processed

        return latest

    # ── Bar processing ────────────────────────────────────────────────────

    def _process_bar(self, candle: Candle) -> None:
        """Process a new entry-TF bar: update history, run inference, manage positions."""
        self.bar_count += 1
        self._last_bar_time = candle.timestamp
        entry_tf = self.config.entry_timeframe

        # Update entry-TF history
        new_row = pd.DataFrame([{
            "date": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": float(candle.volume),
        }])
        if entry_tf in self._history:
            self._history[entry_tf] = pd.concat(
                [self._history[entry_tf], new_row], ignore_index=True
            ).tail(_HISTORY_BARS.get(entry_tf, 200))
        else:
            self._history[entry_tf] = new_row

        # Record bar range for circuit breaker
        self._recent_ranges.append(candle.high - candle.low)

        # Refresh higher timeframes periodically
        self._refresh_higher_tfs()

        # Update trailing stop on open position
        if self.open_trade_id:
            self._update_trailing_stop(candle)

        # Circuit breaker check
        if self._is_volatility_halted(candle.high, candle.low):
            logger.warning("Volatility circuit breaker — skipping signal")
            if self.monitor:
                self.monitor.send_alert("Circuit breaker: volatility halt triggered")
            return

        # Build model input tensors
        batch = self._build_batch()
        if batch is None:
            logger.debug("Insufficient history for inference — skipping")
            return

        # Run inference
        t0 = time.time()
        trade_signal = self._infer(batch, candle.close)
        latency_ms = (time.time() - t0) * 1000

        logger.info(
            "Bar %d | %s | Signal=%s conf=%.3f | latency=%.1fms",
            self.bar_count, candle.timestamp.strftime("%Y-%m-%d %H:%M"),
            trade_signal.signal.name, trade_signal.confidence, latency_ms,
        )

        if self.monitor:
            self.monitor.record_inference(latency_ms, trade_signal)

        # Execute signal
        self._execute_signal(trade_signal, candle)

        # Checkpoint
        if self.bar_count % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def _refresh_higher_tfs(self) -> None:
        """Refresh non-entry timeframe candle history at appropriate intervals."""
        entry_tf = self.config.entry_timeframe
        entry_secs = _TF_SECONDS.get(entry_tf, 60)
        for tf in self.config.timeframes:
            if tf == entry_tf:
                continue  # Entry TF is updated bar-by-bar
            tf_secs = _TF_SECONDS.get(tf, 3600)
            # Refresh every N entry bars where N = higher_tf_seconds / entry_tf_seconds
            interval = max(1, tf_secs // entry_secs)
            if self.bar_count % interval == 0:
                try:
                    granularity = tf_to_granularity(tf)
                    n_bars = _HISTORY_BARS[tf]
                    candles = self.oanda.get_latest_candles(self.pair, granularity, n_bars)
                    if candles:
                        self._history[tf] = _candles_to_df(candles)
                except Exception as e:
                    logger.warning("Failed to refresh %s candles: %s", tf, e)

    # ── Feature building ──────────────────────────────────────────────────

    def _build_batch(self) -> Optional[Dict[str, Dict[str, Tensor]]]:
        """Build multi-timeframe feature tensors from candle history."""
        batch: Dict[str, Dict[str, Tensor]] = {}

        for tf in self.config.timeframes:
            if tf not in self._history or len(self._history[tf]) < 20:
                return None

            df = self._history[tf].copy()
            lookback = self.config.lookbacks[tf]

            # Prepare features using the same pipeline as training
            try:
                prepared = prepare_features(df, lookahead=1, pair=self.pair)
            except Exception as e:
                logger.warning("Feature prep failed for %s: %s", tf, e)
                return None

            # Take the last `lookback` bars
            if len(prepared) < lookback:
                # Pad with repeats of first row
                pad_n = lookback - len(prepared)
                padded = pd.concat(
                    [pd.concat([prepared.iloc[:1]] * pad_n)] + [prepared],
                    ignore_index=True,
                )
            else:
                padded = prepared.iloc[-lookback:]

            # Convert to tensors
            ohlcv = torch.tensor(
                padded[["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]].values,
                dtype=torch.float32,
            ).unsqueeze(0)  # [1, T, 5]

            structure = torch.tensor(
                padded[[f"structure_{i}" for i in range(8)]].values,
                dtype=torch.float32,
            ).unsqueeze(0)  # [1, T, 8]

            rsi = torch.tensor(
                padded[["rsi_norm", "rsi_delta_norm", "rsi_accel_norm"]].values,
                dtype=torch.float32,
            ).unsqueeze(0)  # [1, T, 3]

            volume = torch.tensor(
                padded[["volume_norm", "volume_ratio", "volume_delta"]].values,
                dtype=torch.float32,
            ).unsqueeze(0)  # [1, T, 3]

            batch[tf] = {
                "ohlcv": ohlcv.to(self.device),
                "structure": structure.to(self.device),
                "rsi": rsi.to(self.device),
                "volume": volume.to(self.device),
            }

        return batch

    # ── Inference ─────────────────────────────────────────────────────────

    def _infer(self, batch: Dict[str, Dict[str, Tensor]], current_price: float) -> TradeSignal:
        """Run model inference and return a TradeSignal."""
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward(batch)

            signal_idx = out["signal_logits"].argmax(-1).item()
            base_conf = out["confidence"].item()
            alignment = out.get("alignment", torch.tensor([1.0])).item()
            confidence = base_conf * (0.5 + 0.5 * alignment)

            risk = out["risk_params"][0]
            pip = _PIP_SIZE.get(self.pair, 0.01)

            sl_pips = float(risk[0].item() * 50 + 10)
            tp_pips = float(risk[1].item() * 100 + 20)
            trailing = float(risk[2].item() * 0.5)

            sig = Signal(signal_idx)
            if sig == Signal.BUY:
                entry = current_price
                sl_price = entry - sl_pips * pip
                tp_price = entry + tp_pips * pip
            elif sig == Signal.SELL:
                entry = current_price
                sl_price = entry + sl_pips * pip
                tp_price = entry - tp_pips * pip
            else:
                sl_price = 0.0
                tp_price = 0.0
                entry = current_price

            return TradeSignal(
                signal=sig,
                confidence=confidence,
                entry_price=entry,
                stop_loss=sl_pips,
                take_profit=tp_pips,
                trailing_stop_pct=trailing,
                timestamp=datetime.now(timezone.utc),
            )

    # ── Execution ─────────────────────────────────────────────────────────

    def _execute_signal(self, signal: TradeSignal, candle: Candle) -> None:
        """Execute a trade signal: open, hold, or close positions."""
        # Skip low-confidence signals
        if signal.confidence < self.bt_config.min_confidence:
            return

        if signal.signal == Signal.HOLD:
            return

        # If we have an open position in the opposite direction, close it first
        if self.open_trade_id and self.open_trade_direction != signal.signal:
            self._close_position("Signal reversal")

        # If flat, open new position
        if self.open_trade_id is None and signal.signal != Signal.HOLD:
            self._open_position(signal, candle)

    def _open_position(self, signal: TradeSignal, candle: Candle) -> None:
        """Open a new position via OANDA."""
        pip = _PIP_SIZE.get(self.pair, 0.01)
        pip_value = _PIP_VALUE.get(self.pair, 6.5)

        # Position sizing: fixed-fractional risk
        risk_mult = self._risk_multiplier()
        effective_risk = self.bt_config.risk_per_trade * risk_mult
        risk_amount = self.balance * effective_risk
        lot = risk_amount / max(signal.stop_loss * pip_value, 1e-9)
        lot = max(0.01, min(5.0, lot))

        # Convert lots to units
        units = int(lot * _LOT_SIZE)
        if signal.signal == Signal.SELL:
            units = -units

        # Calculate absolute SL/TP prices
        if signal.signal == Signal.BUY:
            sl_price = candle.close - signal.stop_loss * pip
            tp_price = candle.close + signal.take_profit * pip
        else:
            sl_price = candle.close + signal.stop_loss * pip
            tp_price = candle.close - signal.take_profit * pip

        logger.info(
            "Opening %s %s: units=%d SL=%.3f TP=%.3f conf=%.3f",
            signal.signal.name, self.pair, units, sl_price, tp_price, signal.confidence,
        )

        if self.paper_trading:
            # Paper trade: simulate fill
            self.open_trade_id = f"paper_{self.bar_count}"
            self.open_trade_direction = signal.signal
            self.open_trade_entry = candle.close
            self.open_trade_sl = sl_price
            self.open_trade_tp = tp_price
            self.open_trade_trailing_pct = signal.trailing_stop_pct
            self.open_trade_peak = candle.close
            self.total_trades += 1
        else:
            # Live trade via OANDA
            try:
                order = self.oanda.place_market_order(
                    self.pair, units, sl=sl_price, tp=tp_price,
                )
                if order.status == "FILLED":
                    self.open_trade_id = order.trade_id
                    self.open_trade_direction = signal.signal
                    self.open_trade_entry = order.price
                    self.open_trade_sl = sl_price
                    self.open_trade_tp = tp_price
                    self.open_trade_trailing_pct = signal.trailing_stop_pct
                    self.open_trade_peak = order.price
                    self.total_trades += 1
                    logger.info("Order filled: trade_id=%s price=%.3f", order.trade_id, order.price)
                else:
                    logger.error("Order rejected: %s", order.status)
            except Exception as e:
                logger.error("Failed to place order: %s", e)
                if self.monitor:
                    self.monitor.send_alert(f"Order failed: {e}")

        if self.monitor and self.open_trade_id:
            self.monitor.send_trade(
                f"OPEN {signal.signal.name} {self.pair} @ {candle.close:.3f} "
                f"SL={sl_price:.3f} TP={tp_price:.3f} conf={signal.confidence:.3f}"
            )

        # Copy trade to followers
        if self.copy_trade_mgr and self.open_trade_id:
            self.copy_trade_mgr.copy_open(signal, candle.close)

        # Broadcast signal to Telegram channel subscribers
        if self.monitor and self.open_trade_id:
            self.monitor.broadcast_signal(signal, self.pair, candle.close)

    def _close_position(self, reason: str) -> None:
        """Close the current open position."""
        if not self.open_trade_id:
            return

        logger.info("Closing position %s: %s", self.open_trade_id, reason)

        if self.paper_trading:
            # Get current price for PnL calculation
            try:
                price_data = self.oanda.get_price(self.pair)
                if self.open_trade_direction == Signal.BUY:
                    exit_price = price_data["bid"]
                else:
                    exit_price = price_data["ask"]
            except Exception:
                exit_price = self.open_trade_entry or 0.0

            pnl = self._calc_pnl(exit_price)
            self.balance += pnl
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        else:
            try:
                self.oanda.close_trade(self.open_trade_id)
                # Sync balance after close
                account = self.oanda.get_account_summary()
                pnl = account.balance - self.balance
                self.balance = account.balance
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            except Exception as e:
                logger.error("Failed to close trade: %s", e)
                if self.monitor:
                    self.monitor.send_alert(f"Close failed: {e}")
                return

        if self.monitor:
            self.monitor.send_trade(
                f"CLOSE {self.pair} — {reason} — PnL: ${pnl:.2f}"
            )

        # Close follower positions too
        if self.copy_trade_mgr:
            self.copy_trade_mgr.copy_close(reason)

        # Reset position state
        self.open_trade_id = None
        self.open_trade_direction = None
        self.open_trade_entry = None
        self.open_trade_sl = None
        self.open_trade_tp = None
        self.open_trade_trailing_pct = 0.0
        self.open_trade_peak = 0.0

        # Update equity tracking
        self._update_equity()

    def _calc_pnl(self, exit_price: float) -> float:
        """Calculate PnL for paper trading."""
        if not self.open_trade_entry:
            return 0.0
        pip = _PIP_SIZE.get(self.pair, 0.01)
        pip_value = _PIP_VALUE.get(self.pair, 6.5)
        if self.open_trade_direction == Signal.BUY:
            pips = (exit_price - self.open_trade_entry) / pip
        else:
            pips = (self.open_trade_entry - exit_price) / pip
        # Rough PnL estimate (1 lot * pip_value * pips)
        return pips * pip_value

    # ── Trailing stop ─────────────────────────────────────────────────────

    def _update_trailing_stop(self, candle: Candle) -> None:
        """Update trailing stop on open position."""
        if not self.open_trade_id or self.open_trade_trailing_pct <= 0:
            return

        if self.open_trade_direction == Signal.BUY:
            if candle.high > self.open_trade_peak:
                self.open_trade_peak = candle.high
                move = self.open_trade_peak - (self.open_trade_entry or candle.close)
                new_sl = (self.open_trade_entry or candle.close) + move * self.open_trade_trailing_pct
                if self.open_trade_sl and new_sl > self.open_trade_sl:
                    self.open_trade_sl = new_sl
                    if not self.paper_trading and self.open_trade_id:
                        self.oanda.modify_trade(self.open_trade_id, sl=new_sl)
                        logger.info("Trailing SL updated to %.3f", new_sl)

            # Check if SL hit
            if self.open_trade_sl and candle.low <= self.open_trade_sl:
                self._close_position("Trailing stop hit")
                return

        else:  # SELL
            if candle.low < self.open_trade_peak or self.open_trade_peak == 0:
                if self.open_trade_peak == 0 or candle.low < self.open_trade_peak:
                    self.open_trade_peak = candle.low
                move = (self.open_trade_entry or candle.close) - self.open_trade_peak
                new_sl = (self.open_trade_entry or candle.close) - move * self.open_trade_trailing_pct
                if self.open_trade_sl and new_sl < self.open_trade_sl:
                    self.open_trade_sl = new_sl
                    if not self.paper_trading and self.open_trade_id:
                        self.oanda.modify_trade(self.open_trade_id, sl=new_sl)
                        logger.info("Trailing SL updated to %.3f", new_sl)

            # Check if SL hit
            if self.open_trade_sl and candle.high >= self.open_trade_sl:
                self._close_position("Trailing stop hit")
                return

    # ── Circuit breakers ──────────────────────────────────────────────────

    def _is_volatility_halted(self, high: float, low: float) -> bool:
        """True when current bar range exceeds threshold × rolling mean."""
        if len(self._recent_ranges) < 5:
            return False
        mean_range = np.mean(list(self._recent_ranges))
        current_range = high - low
        return current_range > self.bt_config.atr_halt_multiplier * mean_range

    def _risk_multiplier(self) -> float:
        """Returns 0.5 when drawdown exceeds threshold, else 1.0."""
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        return 0.5 if dd >= self.bt_config.drawdown_reduce_threshold else 1.0

    def _update_equity(self) -> None:
        """Update equity and drawdown tracking."""
        self.equity = self.balance
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        """Save current state to disk."""
        live_state = LiveState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            bar_count=self.bar_count,
            balance=self.balance,
            equity=self.equity,
            peak_equity=self.peak_equity,
            max_drawdown=self.max_drawdown,
            open_trade_id=self.open_trade_id,
            open_trade_direction=(
                self.open_trade_direction.name if self.open_trade_direction else None
            ),
            open_trade_entry=self.open_trade_entry,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            total_pnl=self.total_pnl,
            model_version="2.0.1",
        )

        resonance_waves = list(self.resonance._waves) if self.resonance._waves else None
        resonance_outcomes = list(self.resonance._outcomes) if self.resonance._outcomes else None

        try:
            self.state_mgr.save_checkpoint(
                model=self.model,
                live_state=live_state,
                resonance_waves=resonance_waves,
                resonance_outcomes=resonance_outcomes,
                recent_ranges=list(self._recent_ranges),
            )
        except Exception as e:
            logger.error("Checkpoint save failed: %s", e)
            if self.monitor:
                self.monitor.send_alert(f"CRITICAL: Checkpoint save failed: {e}")

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM gracefully."""
        logger.info("Shutdown signal received (%s)", signum)
        self._running = False

    def _shutdown(self) -> None:
        """Graceful shutdown: save state, close positions if configured."""
        logger.info("Shutting down StreamingEngine...")
        self._save_checkpoint()
        if self.monitor:
            self.monitor.send_info(
                f"WaveTrader stopped — {self.pair}\n"
                f"Balance: ${self.balance:.2f} | Trades: {self.total_trades} | "
                f"PnL: ${self.total_pnl:.2f}"
            )
        logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# Module entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for `python -m wavetrader.streaming`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration from environment
    pair = os.environ.get("PAIR", "GBP/JPY")
    paper = os.environ.get("PAPER_TRADING", "true").lower() == "true"
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
    checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", "100"))
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")

    # Model config
    mtf_config = MTFConfig(pair=pair)
    bt_config = BacktestConfig(
        initial_balance=float(os.environ.get("INITIAL_BALANCE", "10000")),
    )

    # Load model
    model = WaveTraderMTF(mtf_config)

    # If a trained checkpoint exists, load weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Loaded model weights from %s", checkpoint_path)

    # OANDA client
    oanda = OANDAClient()

    # Monitor
    monitor_config = MonitorConfig.from_env()
    monitor = Monitor(monitor_config) if monitor_config.telegram_token else None

    # Create and run engine
    engine = StreamingEngine(
        model=model,
        oanda=oanda,
        pair=pair,
        config=mtf_config,
        bt_config=bt_config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        monitor=monitor,
        paper_trading=paper,
    )

    engine.warmup(checkpoint_path)
    engine.run()


if __name__ == "__main__":
    main()
