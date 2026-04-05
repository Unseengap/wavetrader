"""
Live streaming service — bridges OANDA price feed + model inference
to the dashboard via Server-Sent Events (SSE).

Runs a background thread that:
  1. Polls OANDA for the latest candle (every ~5s for tick-level updates)
  2. Streams completed candles to connected SSE clients
  3. Runs model inference on new complete bars
  4. Pushes trade signals + account state to the frontend

Thread-safe: multiple SSE clients can subscribe simultaneously.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveTick:
    """A single price update pushed to the frontend."""
    time: int            # UNIX timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool       # True = bar is closed
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0


@dataclass
class LiveSignal:
    """Model prediction pushed after each complete bar."""
    signal: str          # "BUY" | "SELL" | "HOLD"
    confidence: float
    sl_pips: float
    tp_pips: float
    alignment: float
    entry_price: float
    timestamp: str


@dataclass
class AccountState:
    """OANDA account snapshot."""
    balance: float
    nav: float
    unrealized_pnl: float
    margin_used: float
    open_trades: int
    market_open: bool


# ─────────────────────────────────────────────────────────────────────────────
# SSE Broadcaster
# ─────────────────────────────────────────────────────────────────────────────

class SSEBroadcaster:
    """Fan-out: one producer → many SSE consumer queues."""

    def __init__(self, maxsize: int = 200):
        self._lock = threading.Lock()
        self._clients: List[queue.Queue] = []
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients = [c for c in self._clients if c is not q]

    def publish(self, event: str, data: dict) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        with self._lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


# ─────────────────────────────────────────────────────────────────────────────
# Live Service
# ─────────────────────────────────────────────────────────────────────────────

class LiveService:
    """
    Singleton-ish service managing OANDA streaming + model inference.

    Lifecycle:
      svc = LiveService()
      svc.start("GBP/JPY", "15min")  # spawns background thread
      ...
      svc.stop()
    """

    def __init__(self) -> None:
        self.broadcaster = SSEBroadcaster()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pair: str = "GBP/JPY"
        self._timeframe: str = "15min"
        self._oanda = None
        self._model = None
        self._model_config = None
        self._device = None
        self._last_complete_time: Optional[datetime] = None
        self._last_broadcast_time: int = 0  # UNIX ts of last candle sent via SSE
        self._status: str = "stopped"  # stopped | starting | running | error
        self._error_msg: str = ""

        # ── Auto-trade state ──────────────────────────────────────────────
        self._auto_trade: bool = False
        self._paper_trading: bool = True
        self._open_trade_id: Optional[str] = None
        self._open_trade_direction: Optional[str] = None  # "BUY" or "SELL"
        self._open_trade_entry: float = 0.0
        self._open_trade_sl: float = 0.0
        self._open_trade_tp: float = 0.0
        self._min_confidence: float = 0.55
        self._risk_per_trade: float = 0.01
        self._trade_log: List[dict] = []  # record of executed trades

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> dict:
        return {
            "status": self._status,
            "pair": self._pair,
            "timeframe": self._timeframe,
            "clients": self.broadcaster.client_count,
            "error": self._error_msg,
            "model_loaded": self._model is not None,
        }

    def start(self, pair: str = "GBP/JPY", timeframe: str = "15min") -> dict:
        """Start the live streaming loop in a background thread."""
        if self.is_running:
            return {"status": "already_running", "pair": self._pair}

        self._pair = pair
        self._timeframe = timeframe
        self._running = True
        self._status = "starting"
        self._error_msg = ""

        # Lazy-init OANDA client
        if self._oanda is None:
            try:
                from wavetrader.oanda import OANDAClient
                self._oanda = OANDAClient()
                logger.info("OANDA client initialised")
            except Exception as e:
                self._status = "error"
                self._error_msg = f"OANDA init failed: {e}"
                self._running = False
                return {"status": "error", "error": str(e)}

        # Lazy-load model
        if self._model is None:
            self._load_model()

        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="live-stream"
        )
        self._thread.start()
        return {"status": "started", "pair": pair, "timeframe": timeframe}

    def stop(self) -> dict:
        """Stop the streaming loop."""
        self._running = False
        self._status = "stopped"
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        return {"status": "stopped"}

    def get_live_candles(self, pair: str, granularity: str, count: int = 300) -> list:
        """Fetch recent candles from OANDA for the chart (one-shot, not streaming)."""
        if self._oanda is None:
            try:
                from wavetrader.oanda import OANDAClient
                self._oanda = OANDAClient()
            except Exception as e:
                logger.error("Cannot init OANDA: %s", e)
                return []

        from wavetrader.oanda import tf_to_granularity
        try:
            gran = tf_to_granularity(granularity)
        except ValueError:
            gran = granularity

        try:
            candles = self._oanda.get_candles(pair, gran, count=count)
            return [
                {
                    "time": int(c.timestamp.timestamp()),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "complete": c.complete,
                }
                for c in candles
            ]
        except Exception as e:
            logger.error("get_live_candles failed: %s", e)
            return []

    def get_account(self) -> dict:
        """Fetch current OANDA account state."""
        if not self._oanda:
            return {"error": "OANDA not connected"}
        try:
            acct = self._oanda.get_account_summary()
            return {
                "balance": acct.balance,
                "nav": acct.nav,
                "unrealized_pnl": acct.unrealized_pnl,
                "margin_used": acct.margin_used,
                "open_trades": acct.open_trade_count,
                "currency": acct.currency,
                "market_open": self._oanda.is_market_open(),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_open_trades(self) -> list:
        """Fetch open trades from OANDA."""
        if not self._oanda:
            return []
        try:
            trades = self._oanda.get_open_trades(self._pair)
            return [
                {
                    "trade_id": t.trade_id,
                    "instrument": t.instrument,
                    "units": t.units,
                    "price": t.price,
                    "unrealized_pnl": t.unrealized_pnl,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "direction": "BUY" if t.units > 0 else "SELL",
                }
                for t in trades
            ]
        except Exception as e:
            logger.error("get_open_trades: %s", e)
            return []

    def sse_stream(self) -> Generator[str, None, None]:
        """Generator for SSE endpoint — yields event strings."""
        q = self.broadcaster.subscribe()
        try:
            # Send initial status
            yield f"event: status\ndata: {json.dumps(self.status)}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    # Heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            self.broadcaster.unsubscribe(q)

    # ── Auto-Trade ────────────────────────────────────────────────────────

    @property
    def auto_trade_status(self) -> dict:
        return {
            "enabled": self._auto_trade,
            "paper_trading": self._paper_trading,
            "open_trade": self._open_trade_id,
            "open_direction": self._open_trade_direction,
            "min_confidence": self._min_confidence,
            "risk_per_trade": self._risk_per_trade,
            "recent_trades": self._trade_log[-10:],
        }

    def set_auto_trade(self, enabled: bool, paper: bool = True) -> dict:
        """Enable or disable automatic trade execution."""
        self._auto_trade = enabled
        self._paper_trading = paper
        logger.info("Auto-trade %s (paper=%s)", "ENABLED" if enabled else "DISABLED", paper)
        return self.auto_trade_status

    def _execute_signal(self, signal_dict: dict, current_price: float) -> None:
        """Execute a trade based on model signal — mirrors StreamingEngine logic."""
        if not self._auto_trade:
            return

        sig = signal_dict["signal"]  # "BUY", "SELL", "HOLD"
        conf = signal_dict["confidence"]

        # Skip low-confidence or HOLD
        if conf < self._min_confidence or sig == "HOLD":
            return

        # Close opposite position first
        if self._open_trade_id and self._open_trade_direction != sig:
            self._close_live_position("Signal reversal")

        # Open new if flat
        if self._open_trade_id is None:
            self._open_live_position(signal_dict, current_price)

    def _open_live_position(self, signal_dict: dict, current_price: float) -> None:
        """Open a position via OANDA (or paper)."""
        _PIP_SIZE = {"GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01, "GBP/USD": 0.0001}
        _PIP_VALUE = {"GBP/JPY": 6.5, "EUR/JPY": 6.7, "USD/JPY": 6.5, "GBP/USD": 10.0}

        sig = signal_dict["signal"]
        pip = _PIP_SIZE.get(self._pair, 0.01)
        pip_value = _PIP_VALUE.get(self._pair, 6.5)

        # Get balance from OANDA
        try:
            acct = self.get_account()
            balance = acct.get("balance", 25000)
        except Exception:
            balance = 25000

        # Position sizing
        sl_pips = signal_dict["sl_pips"]
        tp_pips = signal_dict["tp_pips"]
        risk_amount = balance * self._risk_per_trade
        lot = risk_amount / max(sl_pips * pip_value, 1e-9)
        lot = max(0.01, min(5.0, lot))
        units = int(lot * 100000)
        if sig == "SELL":
            units = -units

        # Absolute SL/TP
        if sig == "BUY":
            sl_price = current_price - sl_pips * pip
            tp_price = current_price + tp_pips * pip
        else:
            sl_price = current_price + sl_pips * pip
            tp_price = current_price - tp_pips * pip

        trade_record = {
            "signal": sig,
            "pair": self._pair,
            "entry_price": round(current_price, 5),
            "units": units,
            "sl": round(sl_price, 5),
            "tp": round(tp_price, 5),
            "confidence": signal_dict["confidence"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "paper": self._paper_trading,
        }

        if self._paper_trading:
            self._open_trade_id = f"paper_{int(time.time())}"
            self._open_trade_direction = sig
            self._open_trade_entry = current_price
            self._open_trade_sl = sl_price
            self._open_trade_tp = tp_price
            trade_record["trade_id"] = self._open_trade_id
            trade_record["status"] = "filled_paper"
            logger.info("Paper trade opened: %s %s @ %.3f", sig, self._pair, current_price)
        else:
            try:
                order = self._oanda.place_market_order(
                    self._pair, units, sl=sl_price, tp=tp_price,
                )
                if order.status == "FILLED":
                    self._open_trade_id = order.trade_id
                    self._open_trade_direction = sig
                    self._open_trade_entry = order.price
                    self._open_trade_sl = sl_price
                    self._open_trade_tp = tp_price
                    trade_record["trade_id"] = order.trade_id
                    trade_record["status"] = "filled"
                    logger.info("OANDA order filled: %s @ %.3f", order.trade_id, order.price)
                else:
                    trade_record["status"] = f"rejected: {order.status}"
                    logger.error("Order rejected: %s", order.status)
            except Exception as e:
                trade_record["status"] = f"error: {e}"
                logger.error("Failed to place order: %s", e)

        self._trade_log.append(trade_record)
        # Broadcast the trade event to frontend
        self.broadcaster.publish("trade_executed", trade_record)

    def _close_live_position(self, reason: str) -> None:
        """Close the current open position."""
        if not self._open_trade_id:
            return

        close_record = {
            "trade_id": self._open_trade_id,
            "direction": self._open_trade_direction,
            "pair": self._pair,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._paper_trading:
            logger.info("Paper trade closed: %s (%s)", self._open_trade_id, reason)
        else:
            try:
                self._oanda.close_trade(self._open_trade_id)
                logger.info("OANDA trade closed: %s (%s)", self._open_trade_id, reason)
            except Exception as e:
                logger.error("Failed to close trade %s: %s", self._open_trade_id, e)

        self._open_trade_id = None
        self._open_trade_direction = None
        self._open_trade_entry = 0.0
        self.broadcaster.publish("trade_closed", close_record)

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Try to load the latest WaveTraderMTF checkpoint."""
        try:
            import torch
            from wavetrader.config import MTFConfig
            from wavetrader.model import WaveTraderMTF

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model_config = MTFConfig(pair=self._pair)
            self._model = WaveTraderMTF(self._model_config)

            # Search for latest checkpoint
            ckpt_dirs = [
                Path("checkpoints"),
                Path("/checkpoints"),
                Path("data/live_checkpoints"),
            ]
            loaded = False
            for ckpt_dir in ckpt_dirs:
                if not ckpt_dir.is_dir():
                    continue
                for d in sorted(ckpt_dir.iterdir(), reverse=True):
                    weights = d / "model_weights.pt"
                    if weights.exists():
                        state = torch.load(
                            str(weights), weights_only=False, map_location=self._device
                        )
                        if "model_state_dict" in state:
                            self._model.load_state_dict(state["model_state_dict"])
                        else:
                            self._model.load_state_dict(state)
                        self._model.to(self._device)
                        self._model.eval()
                        loaded = True
                        logger.info("Loaded model from %s", weights)
                        break
                if loaded:
                    break

            if not loaded:
                logger.warning("No model checkpoint found — signals will not be generated")
                self._model = None

        except Exception as e:
            logger.error("Model load failed: %s", e)
            self._model = None

    # ── Inference ─────────────────────────────────────────────────────────

    def _run_inference(self, candles_by_tf: Dict[str, list]) -> Optional[dict]:
        """Run model on latest candles, return signal dict or None."""
        if self._model is None:
            return None

        try:
            import torch
            import pandas as pd
            from wavetrader.dataset import prepare_features

            batch = {}
            for tf in self._model_config.timeframes:
                if tf not in candles_by_tf or not candles_by_tf[tf]:
                    return None

                rows = candles_by_tf[tf]
                df = pd.DataFrame(rows)
                if "date" not in df.columns and "time" in df.columns:
                    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)

                lookback = self._model_config.lookbacks[tf]
                try:
                    prepared = prepare_features(df, lookahead=1, pair=self._pair)
                except Exception:
                    return None

                if len(prepared) < lookback:
                    return None

                prepared = prepared.iloc[-lookback:]

                ohlcv = torch.tensor(
                    prepared[["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]].values,
                    dtype=torch.float32,
                ).unsqueeze(0).to(self._device)

                structure = torch.tensor(
                    prepared[[f"structure_{i}" for i in range(8)]].values,
                    dtype=torch.float32,
                ).unsqueeze(0).to(self._device)

                rsi = torch.tensor(
                    prepared[["rsi_norm", "rsi_delta_norm", "rsi_accel_norm"]].values,
                    dtype=torch.float32,
                ).unsqueeze(0).to(self._device)

                volume = torch.tensor(
                    prepared[["volume_norm", "volume_ratio", "volume_delta"]].values,
                    dtype=torch.float32,
                ).unsqueeze(0).to(self._device)

                batch[tf] = {
                    "ohlcv": ohlcv,
                    "structure": structure,
                    "rsi": rsi,
                    "volume": volume,
                }

            self._model.eval()
            with torch.no_grad():
                out = self._model.forward(batch)

            signal_idx = out["signal_logits"].argmax(-1).item()
            signal_name = ["BUY", "SELL", "HOLD"][signal_idx]
            confidence = out["confidence"].item()
            alignment = out.get("alignment", torch.tensor([1.0])).item()
            risk = out["risk_params"][0]

            return {
                "signal": signal_name,
                "confidence": round(confidence * (0.5 + 0.5 * alignment), 4),
                "alignment": round(alignment, 4),
                "sl_pips": round(float(risk[0].item() * 50 + 10), 1),
                "tp_pips": round(float(risk[1].item() * 100 + 20), 1),
                "trailing_pct": round(float(risk[2].item() * 0.5), 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Inference failed: %s", e)
            return None

    # ── Stream loop ───────────────────────────────────────────────────────

    def _stream_loop(self) -> None:
        """Background thread: poll OANDA, push candles + signals via SSE."""
        from wavetrader.oanda import tf_to_granularity

        self._status = "running"
        logger.info("Live stream started: %s %s", self._pair, self._timeframe)

        poll_interval = 5  # seconds between polls
        _TF_MAP = {"15min": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}

        # Keep a small history per TF for inference
        tf_history: Dict[str, list] = {tf: [] for tf in self._model_config.timeframes} if self._model_config else {}

        # Initial candle fetch for all timeframes
        try:
            for tf in (self._model_config.timeframes if self._model_config else [self._timeframe]):
                gran = tf_to_granularity(tf)
                candles = self._oanda.get_candles(self._pair, gran, count=300)
                if candles:
                    tf_history[tf] = [
                        {
                            "time": int(c.timestamp.timestamp()),
                            "date": c.timestamp.isoformat(),
                            "open": c.open, "high": c.high,
                            "low": c.low, "close": c.close,
                            "volume": c.volume, "complete": c.complete,
                        }
                        for c in candles if c.complete
                    ]
        except Exception as e:
            logger.error("Initial fetch failed: %s", e)

        # Push account state once on start
        try:
            acct = self.get_account()
            self.broadcaster.publish("account", acct)
        except Exception:
            pass

        while self._running:
            try:
                # Poll for new entry-TF candle
                gran = tf_to_granularity(self._timeframe)
                candles = self._oanda.get_candles(self._pair, gran, count=3)

                for c in candles:
                    candle_time = int(c.timestamp.timestamp())
                    # Only broadcast candles >= last broadcast time to avoid
                    # "Cannot update oldest data" in lightweight-charts
                    if candle_time < self._last_broadcast_time:
                        continue
                    tick = {
                        "time": candle_time,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                        "complete": c.complete,
                    }
                    self.broadcaster.publish("candle", tick)
                    self._last_broadcast_time = candle_time

                    # Track completed bars
                    if c.complete:
                        if (self._last_complete_time is None
                                or c.timestamp > self._last_complete_time):
                            self._last_complete_time = c.timestamp

                            # Append to entry-TF history
                            if self._timeframe in tf_history:
                                entry = {
                                    "time": int(c.timestamp.timestamp()),
                                    "date": c.timestamp.isoformat(),
                                    "open": c.open, "high": c.high,
                                    "low": c.low, "close": c.close,
                                    "volume": c.volume, "complete": True,
                                }
                                tf_history[self._timeframe].append(entry)
                                tf_history[self._timeframe] = tf_history[self._timeframe][-300:]

                            # Refresh higher TFs
                            for tf in tf_history:
                                if tf != self._timeframe:
                                    try:
                                        g = tf_to_granularity(tf)
                                        htf_candles = self._oanda.get_candles(self._pair, g, count=200)
                                        tf_history[tf] = [
                                            {
                                                "time": int(hc.timestamp.timestamp()),
                                                "date": hc.timestamp.isoformat(),
                                                "open": hc.open, "high": hc.high,
                                                "low": hc.low, "close": hc.close,
                                                "volume": hc.volume, "complete": True,
                                            }
                                            for hc in htf_candles if hc.complete
                                        ]
                                    except Exception as e:
                                        logger.debug("HTF refresh %s: %s", tf, e)

                            # Run inference
                            signal = self._run_inference(tf_history)
                            if signal:
                                self.broadcaster.publish("signal", signal)
                                # Auto-execute trade if enabled
                                self._execute_signal(signal, c.close)

                # Also push current price
                try:
                    price = self._oanda.get_price(self._pair)
                    self.broadcaster.publish("price", {
                        "bid": price["bid"],
                        "ask": price["ask"],
                        "spread": round(price["spread"], 5),
                        "mid": round((price["bid"] + price["ask"]) / 2, 5),
                        "time": price["time"],
                    })
                except Exception:
                    pass

                # Periodic account update (every 30s ≈ every 6th poll)
                if self._last_complete_time and int(time.time()) % 30 < poll_interval:
                    try:
                        acct = self.get_account()
                        self.broadcaster.publish("account", acct)
                        trades = self.get_open_trades()
                        self.broadcaster.publish("trades", {"trades": trades})
                    except Exception:
                        pass

                time.sleep(poll_interval)

            except Exception as e:
                logger.error("Stream loop error: %s", e)
                self._error_msg = str(e)
                time.sleep(15)

        self._status = "stopped"
        logger.info("Live stream stopped")


# ── Module-level singleton ────────────────────────────────────────────────────
_live_service: Optional[LiveService] = None


def get_live_service() -> LiveService:
    """Return (or create) the global LiveService singleton."""
    global _live_service
    if _live_service is None:
        _live_service = LiveService()
    return _live_service
