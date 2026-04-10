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

from .config import BacktestConfig, MTFConfig, ResonanceConfig, SIConfig, DEFAULT_RISK_SCALING
from .dataset import ResonanceBuffer, prepare_features
from .model import WaveTraderMTF
from .monitor import Monitor, MonitorConfig
from .oanda import Candle, OANDAClient, OANDAConfig, tf_to_granularity
from .state import LiveState, StateManager
from .training import SynapticIntelligence
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
        oanda_demo: OANDAClient,
        pair: str = "GBP/JPY",
        config: Optional[MTFConfig] = None,
        bt_config: Optional[BacktestConfig] = None,
        res_config: Optional[ResonanceConfig] = None,
        checkpoint_dir: str = "/data/checkpoints",
        checkpoint_interval: int = 100,
        monitor: Optional[Monitor] = None,
        oanda_live: Optional[OANDAClient] = None,
        copy_trade_mgr: Optional[CopyTradeManager] = None,
    ) -> None:
        self.model = model
        self.oanda_demo = oanda_demo
        self.oanda_live = oanda_live  # None if live creds not configured
        self.pair = pair
        self.copy_trade_mgr = copy_trade_mgr
        self.config = config or MTFConfig(pair=pair)
        self.bt_config = bt_config or BacktestConfig()
        self.res_config = res_config or ResonanceConfig()
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

        # Current position tracking (demo account)
        self.open_trade_id: Optional[str] = None
        self.open_trade_direction: Optional[Signal] = None
        self.open_trade_entry: Optional[float] = None
        self.open_trade_sl: Optional[float] = None
        self.open_trade_initial_sl: Optional[float] = None  # immutable original SL
        self.open_trade_tp: Optional[float] = None
        self.open_trade_trailing_pct: float = 0.0
        self.open_trade_peak: float = 0.0  # For trailing stop

        # Current position tracking (live account)
        self.live_trade_id: Optional[str] = None
        self.live_trade_direction: Optional[Signal] = None
        self.live_trade_entry: Optional[float] = None
        self.live_trade_sl: Optional[float] = None
        self.live_trade_tp: Optional[float] = None

        # Trade cooldown (bars since last close)
        self._bars_since_close: int = self.bt_config.cooldown_bars  # allow first trade immediately

        # Circuit breaker state
        self._recent_ranges: deque = deque(maxlen=20)

        # Per-timeframe candle history (DataFrames)
        self._history: Dict[str, pd.DataFrame] = {}

        # Resonance buffer
        self.resonance = ResonanceBuffer(
            capacity=self.res_config.capacity,
            wave_dim=self.config.output_wave_dim,
        )
        self._last_wave_state: Optional[Tensor] = None  # For resonance storage on close

        # Synaptic Intelligence — online continual learning
        self.si_config = SIConfig()
        self.si = SynapticIntelligence(
            model=self.model,
            si_lambda=self.si_config.si_lambda,
            epsilon=self.si_config.epsilon,
        )
        self._si_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-5, weight_decay=1e-6,
        )
        self._online_batch_count: int = 0

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
            candles = self.oanda_demo.get_latest_candles(self.pair, granularity, n_bars)
            if candles:
                self._history[tf] = _candles_to_df(candles)
                logger.info("  %s: fetched %d bars", tf, len(candles))
            else:
                logger.warning("  %s: no candles returned", tf)

        # Sync balance from OANDA demo account
        try:
            account = self.oanda_demo.get_account_summary()
            self.balance = account.balance
            self.equity = account.nav
            logger.info(
                "OANDA account: balance=%.2f nav=%.2f open_trades=%d",
                account.balance, account.nav, account.open_trade_count,
            )
        except Exception as e:
            logger.warning("Could not sync OANDA account: %s", e)

        # Check for existing open positions on demo account
        try:
            open_trades = self.oanda_demo.get_open_trades(self.pair)
            if open_trades:
                t = open_trades[0]
                self.open_trade_id = t.trade_id
                self.open_trade_direction = Signal.BUY if t.units > 0 else Signal.SELL
                self.open_trade_entry = t.price
                self.open_trade_sl = t.stop_loss
                self.open_trade_initial_sl = t.stop_loss
                self.open_trade_tp = t.take_profit
                logger.info("Existing demo position: %s %s @ %.3f", t.trade_id, self.open_trade_direction.name, t.price)
        except Exception as e:
            logger.warning("Could not check demo open trades: %s", e)

        # Check for existing open positions on live account
        if self.oanda_live:
            try:
                live_trades = self.oanda_live.get_open_trades(self.pair)
                if live_trades:
                    t = live_trades[0]
                    self.live_trade_id = t.trade_id
                    self.live_trade_direction = Signal.BUY if t.units > 0 else Signal.SELL
                    self.live_trade_entry = t.price
                    self.live_trade_sl = t.stop_loss
                    self.live_trade_tp = t.take_profit
                    logger.info("Existing live position: %s %s @ %.3f", t.trade_id, self.live_trade_direction.name, t.price)
            except Exception as e:
                logger.warning("Could not check live open trades: %s", e)

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
        logger.info("  Mode:  DEMO%s", " + LIVE" if self.oanda_live else " only")
        logger.info("  Device: %s", self.device)
        logger.info("=" * 60)

        if self.monitor:
            mode = "DEMO + LIVE" if self.oanda_live else "DEMO"
            self.monitor.send_info(
                f"WaveTrader started — {self.pair} ({mode})"
            )

        while self._running:
            try:
                if not self.oanda_demo.is_market_open():
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
        candles = self.oanda_demo.get_candles(self.pair, granularity, count=2)
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
        self._bars_since_close += 1
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
                    candles = self.oanda_demo.get_latest_candles(self.pair, granularity, n_bars)
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

            try:
                prepared = prepare_features(df, lookahead=1, pair=self.pair)
            except Exception as e:
                logger.warning("Feature prep failed for %s: %s", tf, e)
                return None

            # Take the last `lookback` bars
            if len(prepared) < lookback:
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
            ).unsqueeze(0)

            structure = torch.tensor(
                padded[[f"structure_{i}" for i in range(8)]].values,
                dtype=torch.float32,
            ).unsqueeze(0)

            rsi = torch.tensor(
                padded[["rsi_norm", "rsi_delta_norm", "rsi_accel_norm"]].values,
                dtype=torch.float32,
            ).unsqueeze(0)

            volume = torch.tensor(
                padded[["volume_norm", "volume_ratio", "volume_delta"]].values,
                dtype=torch.float32,
            ).unsqueeze(0)

            tf_batch = {
                "ohlcv": ohlcv.to(self.device),
                "structure": structure.to(self.device),
                "rsi": rsi.to(self.device),
                "volume": volume.to(self.device),
            }

            # Regime context: session flags + ATR percentile
            regime_cols = ["session_tokyo", "session_london", "session_newyork", "atr_pct"]
            if all(c in padded.columns for c in regime_cols):
                regime = torch.tensor(
                    padded[regime_cols].values, dtype=torch.float32,
                ).unsqueeze(0)
                tf_batch["regime"] = regime.to(self.device)

            batch[tf] = tf_batch

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

            risk = out["risk_params"][0]
            pip = _PIP_SIZE.get(self.pair, 0.01)
            _rs = DEFAULT_RISK_SCALING

            confidence = base_conf * (0.5 + 0.5 * alignment)
            sl_pips = _rs.sl_pips(float(risk[0].item()))
            tp_pips = _rs.tp_pips(float(risk[1].item()))
            trailing = _rs.trailing_pct(float(risk[2].item()))

            # ResonanceBuffer: retrieve similar past waves for confidence calibration
            wave_state = out.get("wave_state")
            if wave_state is not None:
                self._last_wave_state = wave_state.detach()
                res = self.resonance.retrieve_with_outcomes(wave_state, k=self.res_config.top_k)
                if res is not None:
                    _, past_outcomes = res
                    win_rate = sum(1 for o in past_outcomes if o > 0) / len(past_outcomes)
                    # Scale confidence: if similar past waves mostly lost, dampen;
                    # if they mostly won, slightly boost.
                    resonance_bias = 0.5 + 0.5 * (win_rate - 0.5)  # range: 0.25-0.75
                    confidence *= resonance_bias
                    logger.debug(
                        "Resonance: %d/%d similar waves won (bias=%.2f)",
                        sum(1 for o in past_outcomes if o > 0), len(past_outcomes),
                        resonance_bias,
                    )

            sig = Signal(signal_idx)
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
        """Execute a trade signal: open, hold, or close positions on both accounts."""
        logger.info(
            "Evaluating signal: %s conf=%.4f threshold=%.2f price=%.3f",
            signal.signal.name, signal.confidence, self.bt_config.min_confidence,
            candle.close,
        )

        # Skip HOLD signals
        if signal.signal == Signal.HOLD:
            logger.info("Skipping HOLD signal")
            return

        # Skip low-confidence signals
        if signal.confidence < self.bt_config.min_confidence:
            logger.warning(
                "Skipping %s: confidence %.4f < threshold %.2f",
                signal.signal.name, signal.confidence, self.bt_config.min_confidence,
            )
            return

        # Cooldown: wait N bars after last trade close before opening new
        if self.open_trade_id is None and self._bars_since_close < self.bt_config.cooldown_bars:
            logger.info(
                "Cooldown: %d/%d bars since last close — skipping %s",
                self._bars_since_close, self.bt_config.cooldown_bars, signal.signal.name,
            )
            return

        # Same direction as existing position → hold, don't duplicate
        if self.open_trade_id and self.open_trade_direction == signal.signal:
            logger.info("Already %s — holding position %s", signal.signal.name, self.open_trade_id)
            return

        # Opposite direction → close then immediately open (atomic reversal)
        if (self.open_trade_id and self.open_trade_direction != signal.signal) or \
           (self.live_trade_id and self.live_trade_direction != signal.signal):
            self._close_position("Signal reversal")
            self._open_position(signal, candle)
            return

        # Flat → open new position
        if self.open_trade_id is None:
            self._open_position(signal, candle)

    def _open_position(self, signal: TradeSignal, candle: Candle) -> None:
        """Open a new position via OANDA on both demo and live accounts."""
        pip = _PIP_SIZE.get(self.pair, 0.01)
        pip_value = _PIP_VALUE.get(self.pair, 6.5)

        # Calculate absolute SL/TP prices
        is_opposite_exit = getattr(signal, 'exit_mode', 'tp_sl') == 'opposite_signal'
        if signal.signal == Signal.BUY:
            sl_price = candle.close - signal.stop_loss * pip
            tp_price = None if is_opposite_exit else candle.close + signal.take_profit * pip
        else:
            sl_price = candle.close + signal.stop_loss * pip
            tp_price = None if is_opposite_exit else candle.close - signal.take_profit * pip

        # Place on demo account
        self._place_order_on_account(
            "demo", self.oanda_demo, signal, candle, sl_price, tp_price, pip_value,
        )

        # Place on live account if available
        if self.oanda_live:
            self._place_order_on_account(
                "live", self.oanda_live, signal, candle, sl_price, tp_price, pip_value,
            )

        if self.monitor and (self.open_trade_id or self.live_trade_id):
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

    def _place_order_on_account(
        self,
        account: str,
        client: OANDAClient,
        signal: TradeSignal,
        candle: Candle,
        sl_price: float,
        tp_price: float,
        pip_value: float,
    ) -> None:
        """Place an order on a specific OANDA account (demo or live)."""
        # Position sizing: fixed-fractional risk based on account balance
        try:
            acct = client.get_account_summary()
            balance = acct.balance
        except Exception:
            balance = self.balance

        # Check available margin before placing
        try:
            if acct.margin_available <= 0:
                logger.warning(
                    "No margin available [%s] (margin_avail=%.2f) — skipping order",
                    account.upper(), acct.margin_available,
                )
                return
        except Exception:
            pass  # acct may not exist if balance fallback was used

        risk_amount = balance * self.bt_config.risk_per_trade
        lot = risk_amount / max(signal.stop_loss * pip_value, 1e-9)
        lot = max(0.01, lot)  # no artificial upper cap — OANDA enforces margin

        units = int(lot * _LOT_SIZE)
        if signal.signal == Signal.SELL:
            units = -units

        logger.info(
            "Opening %s %s [%s]: units=%d SL=%.3f TP=%.3f conf=%.3f",
            signal.signal.name, self.pair, account.upper(),
            units, sl_price, tp_price, signal.confidence,
        )

        try:
            order = client.place_market_order(
                self.pair, units, sl=sl_price, tp=tp_price,
            )
            if order.status == "FILLED":
                if account == "demo":
                    self.open_trade_id = order.trade_id
                    self.open_trade_direction = signal.signal
                    self.open_trade_entry = order.price
                    self.open_trade_sl = sl_price
                    self.open_trade_initial_sl = sl_price
                    self.open_trade_tp = tp_price
                    self.open_trade_trailing_pct = signal.trailing_stop_pct
                    self.open_trade_peak = order.price
                else:
                    self.live_trade_id = order.trade_id
                    self.live_trade_direction = signal.signal
                    self.live_trade_entry = order.price
                    self.live_trade_sl = sl_price
                    self.live_trade_tp = tp_price
                self.total_trades += 1
                logger.info(
                    "Order filled [%s]: trade_id=%s price=%.3f",
                    account.upper(), order.trade_id, order.price,
                )
            else:
                logger.error("Order rejected [%s]: %s", account.upper(), order.status)
        except Exception as e:
            logger.error("Failed to place order [%s]: %s", account.upper(), e)
            if self.monitor:
                self.monitor.send_alert(f"Order failed [{account}]: {e}")

    def _close_position(self, reason: str) -> None:
        """Close the current open positions on both accounts."""
        if not self.open_trade_id and not self.live_trade_id:
            return

        logger.info("Closing positions: %s", reason)
        pnl = 0.0

        # Close demo account position
        if self.open_trade_id:
            try:
                self.oanda_demo.close_trade(self.open_trade_id)
                account = self.oanda_demo.get_account_summary()
                pnl = account.balance - self.balance
                self.balance = account.balance
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                logger.info("Demo trade closed: %s PnL=%.2f", self.open_trade_id, pnl)
            except Exception as e:
                logger.error("Failed to close demo trade: %s", e)
                if self.monitor:
                    self.monitor.send_alert(f"Demo close failed: {e}")

        # Close live account position
        if self.live_trade_id and self.oanda_live:
            try:
                self.oanda_live.close_trade(self.live_trade_id)
                logger.info("Live trade closed: %s", self.live_trade_id)
            except Exception as e:
                logger.error("Failed to close live trade: %s", e)
                if self.monitor:
                    self.monitor.send_alert(f"Live close failed: {e}")

        if self.monitor:
            self.monitor.send_trade(
                f"CLOSE {self.pair} — {reason} — PnL: ${pnl:.2f}"
            )

        # Close follower positions too
        if self.copy_trade_mgr:
            self.copy_trade_mgr.copy_close(reason)

        # Store wave state + outcome in ResonanceBuffer for episodic memory
        if self._last_wave_state is not None and pnl != 0.0:
            stored = self.resonance.store(self._last_wave_state, pnl)
            if stored:
                logger.info(
                    "Resonance: stored wave state (PnL=%.2f, buffer=%d/%d)",
                    pnl, len(self.resonance._waves), self.resonance.capacity,
                )

        # Reset demo position state
        self.open_trade_id = None
        self.open_trade_direction = None
        self.open_trade_entry = None
        self.open_trade_sl = None
        self.open_trade_initial_sl = None
        self.open_trade_tp = None
        self.open_trade_trailing_pct = 0.0
        self.open_trade_peak = 0.0

        # Reset live position state
        self.live_trade_id = None
        self.live_trade_direction = None
        self.live_trade_entry = None
        self.live_trade_sl = None
        self.live_trade_tp = None

        # Reset cooldown counter
        self._bars_since_close = 0

        # Update equity tracking
        self._update_equity()

        # Online learning step: adapt model to recent trade outcome
        if pnl != 0.0:
            self._online_learn(pnl)

    def _online_learn(self, pnl: float) -> None:
        """One-step online learning using Synaptic Intelligence.

        After each closed trade, do a single gradient step to nudge the model
        toward the correct signal (if it was wrong) while SI regularisation
        prevents catastrophic forgetting of older knowledge.
        """
        batch = self._build_batch()
        if batch is None:
            return

        try:
            self.model.train()
            self._si_optimizer.zero_grad()

            out = self.model.forward(batch)
            signal_logits = out["signal_logits"]  # [1, 3]

            # Construct target: if pnl > 0, the signal was correct → reinforce it.
            # If pnl < 0, the opposite signal (or HOLD) would have been better.
            pred_signal = signal_logits.argmax(-1).item()
            if pnl > 0:
                target = torch.tensor([pred_signal], device=signal_logits.device)
            else:
                # Penalise the predicted signal; nudge toward HOLD
                target = torch.tensor([Signal.HOLD.value], device=signal_logits.device)

            task_loss = torch.nn.functional.cross_entropy(signal_logits, target)
            si_loss = self.si.penalty()
            total_loss = task_loss + si_loss

            total_loss.backward()
            self.si.update()
            self._si_optimizer.step()
            self._online_batch_count += 1

            # Periodically consolidate importance weights
            if self._online_batch_count % self.si_config.consolidate_every == 0:
                self.si.consolidate()
                logger.info("SI consolidated at batch %d", self._online_batch_count)

            self.model.eval()
            logger.info(
                "Online learn: task_loss=%.4f si_loss=%.4f pnl=%.2f",
                task_loss.item(), si_loss.item(), pnl,
            )
        except Exception as e:
            self.model.eval()
            logger.warning("Online learning failed: %s", e)

    # ── Trailing stop ─────────────────────────────────────────────────────

    def _update_trailing_stop(self, candle: Candle) -> None:
        """Update trailing stop on open position.

        Uses the same formula as backtest.py: trail_distance = initial_risk * (1 - pct),
        floored by min_trail_pips so the stop never gets tighter than that distance
        from the peak price.
        """
        if not self.open_trade_id or self.open_trade_trailing_pct <= 0:
            return

        from wavetrader.config import DEFAULT_RISK_SCALING
        pip = _PIP_SIZE.get(self.pair, 0.01)
        min_trail = DEFAULT_RISK_SCALING.min_trail_pips * pip

        if self.open_trade_direction == Signal.BUY:
            if candle.high > self.open_trade_peak:
                self.open_trade_peak = candle.high
            entry = self.open_trade_entry or candle.close
            initial_risk = entry - (self.open_trade_initial_sl or entry)
            if initial_risk <= 0:
                return
            trail_distance = initial_risk * (1.0 - self.open_trade_trailing_pct)
            trail_distance = max(trail_distance, min_trail)
            new_sl = self.open_trade_peak - trail_distance
            if self.open_trade_sl and new_sl > self.open_trade_sl:
                self.open_trade_sl = new_sl
                if self.open_trade_id:
                    self.oanda_demo.modify_trade(self.open_trade_id, sl=new_sl)
                if self.live_trade_id and self.oanda_live:
                    self.oanda_live.modify_trade(self.live_trade_id, sl=new_sl)
                logger.info("Trailing SL updated to %.3f (dist=%.3f)", new_sl, trail_distance)

            # Check if SL hit
            if self.open_trade_sl and candle.low <= self.open_trade_sl:
                self._close_position("Trailing stop hit")
                return

        else:  # SELL
            if candle.low < self.open_trade_peak or self.open_trade_peak == 0:
                self.open_trade_peak = candle.low
            entry = self.open_trade_entry or candle.close
            initial_risk = (self.open_trade_initial_sl or entry) - entry
            if initial_risk <= 0:
                return
            trail_distance = initial_risk * (1.0 - self.open_trade_trailing_pct)
            trail_distance = max(trail_distance, min_trail)
            new_sl = self.open_trade_peak + trail_distance
            if self.open_trade_sl and new_sl < self.open_trade_sl:
                self.open_trade_sl = new_sl
                if self.open_trade_id:
                    self.oanda_demo.modify_trade(self.open_trade_id, sl=new_sl)
                if self.live_trade_id and self.oanda_live:
                    self.oanda_live.modify_trade(self.live_trade_id, sl=new_sl)
                logger.info("Trailing SL updated to %.3f (dist=%.3f)", new_sl, trail_distance)

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
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
    checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", "100"))
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")

    # Model config
    mtf_config = MTFConfig(pair=pair)
    bt_config = BacktestConfig(
        initial_balance=float(os.environ.get("INITIAL_BALANCE", "10000")),
        min_confidence=float(os.environ.get("MIN_CONFIDENCE", "0.30")),
    )

    # Load model
    model = WaveTraderMTF(mtf_config)

    # If a trained checkpoint exists, load weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint_path, weights_only=False, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Loaded model weights from %s", checkpoint_path)

    # OANDA clients (demo always, live if configured)
    oanda_demo = OANDAClient(OANDAConfig.demo_from_env())
    logger.info("Demo account: %s", oanda_demo.config.account_id)

    oanda_live = None
    live_cfg = OANDAConfig.live_from_env()
    if live_cfg is not None:
        try:
            oanda_live = OANDAClient(live_cfg)
            logger.info("Live account: %s", oanda_live.config.account_id)
        except Exception as e:
            logger.warning("Live OANDA init failed (continuing demo-only): %s", e)
    else:
        logger.info("No live OANDA credentials — trading demo only")

    # Monitor
    monitor_config = MonitorConfig.from_env()
    monitor = Monitor(monitor_config) if monitor_config.telegram_token else None

    # Create and run engine
    engine = StreamingEngine(
        model=model,
        oanda_demo=oanda_demo,
        pair=pair,
        config=mtf_config,
        bt_config=bt_config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        monitor=monitor,
        oanda_live=oanda_live,
    )

    engine.warmup(checkpoint_path)
    engine.run()


if __name__ == "__main__":
    main()
