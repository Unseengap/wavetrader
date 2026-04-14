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
import uuid
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
    Per-model service managing OANDA data viewing + model inference.

    Each model gets its own LiveService instance pointed at a specific
    OANDA account.  Trading is handled by the standalone wavetrader
    containers — this service is for *viewing* data and streaming it
    to the dashboard frontend.

    Lifecycle:
      svc = LiveService(model_id="mtf")
      svc.start("GBP/JPY", "15min")  # spawns background thread
      ...
      svc.stop()
    """

    def __init__(self, model_id: str = "mtf") -> None:
        from .model_registry import get_model_registry

        self.model_id = model_id
        self._model_entry = get_model_registry().get(model_id)
        self.broadcaster = SSEBroadcaster()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pair: str = self._model_entry.pair if self._model_entry else "GBP/JPY"
        self._timeframe: str = "15min"
        self._oanda_demo = None   # OANDAClient for practice account (always)
        self._oanda_live = None   # OANDAClient for live account (optional)
        self._live_available: bool = False
        self._model = None
        self._model_config = None
        self._device = None
        self._last_complete_time: Optional[datetime] = None
        self._last_broadcast_time: int = 0  # UNIX ts of last candle sent via SSE
        self._status: str = "stopped"  # stopped | starting | running | error
        self._error_msg: str = ""

        # ── Auto-trade state (always on — no toggle) ─────────────────────
        self._min_confidence: float = float(os.environ.get("MIN_CONFIDENCE", "0.52"))
        self._risk_per_trade: float = float(os.environ.get("RISK_PER_TRADE", "0.10"))
        self._trade_log: List[dict] = []  # record of executed trades
        self._signal_history: List[dict] = []  # last N signals for diagnostics

        # Per-account position tracking
        self._demo_trade_id: Optional[str] = None
        self._demo_trade_direction: Optional[str] = None
        self._demo_trade_entry: float = 0.0
        self._demo_trade_sl: float = 0.0
        self._demo_trade_tp: float = 0.0
        self._demo_trade_initial_sl: float = 0.0
        self._demo_trade_trailing_pct: float = 0.0
        self._demo_trade_highest: float = 0.0
        self._demo_trade_lowest: float = float("inf")

        self._live_trade_id: Optional[str] = None
        self._live_trade_direction: Optional[str] = None
        self._live_trade_entry: float = 0.0
        self._live_trade_sl: float = 0.0
        self._live_trade_tp: float = 0.0
        self._live_trade_initial_sl: float = 0.0
        self._live_trade_trailing_pct: float = 0.0
        self._live_trade_highest: float = 0.0
        self._live_trade_lowest: float = float("inf")

        # ── LLM Arbiter ──────────────────────────────────────────────────
        from wavetrader.llm_arbiter import LLMArbiter, LLMArbiterConfig, ArbiterContext
        from wavetrader.calendar import get_calendar
        from wavetrader.llm_logger import get_decision_log

        arbiter_enabled = os.environ.get("LLM_ARBITER_ENABLED", "true").lower() == "true"
        arbiter_mode = os.environ.get("LLM_AUTHORITY_MODE", "override")
        arbiter_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
        self._arbiter_config = LLMArbiterConfig(
            enabled=arbiter_enabled,
            authority_mode=arbiter_mode,
            model=arbiter_model,
        )
        self._arbiter = LLMArbiter(self._arbiter_config)
        self._calendar = get_calendar()
        self._decision_log = get_decision_log()
        self._arbiter_decisions: List[dict] = []  # recent decisions for dashboard

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> dict:
        return {
            "status": self._status,
            "model_id": self.model_id,
            "model_name": self._model_entry.name if self._model_entry else self.model_id,
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

        # Lazy-init OANDA clients using model-specific env vars
        if self._oanda_demo is None:
            try:
                from wavetrader.oanda import OANDAClient, OANDAConfig
                me = self._model_entry
                if me and me.demo_api_key and me.demo_account_id:
                    demo_cfg = OANDAConfig(
                        api_key=me.demo_api_key,
                        account_id=me.demo_account_id,
                        environment="practice",
                    )
                else:
                    demo_cfg = OANDAConfig.demo_from_env()
                self._oanda_demo = OANDAClient(demo_cfg)
                logger.info("OANDA demo client initialised for model '%s' (account %s)",
                            self.model_id, demo_cfg.account_id)
            except Exception as e:
                self._status = "error"
                self._error_msg = f"OANDA demo init failed: {e}"
                self._running = False
                return {"status": "error", "error": str(e)}

        if self._oanda_live is None and not self._live_available:
            try:
                from wavetrader.oanda import OANDAClient, OANDAConfig
                me = self._model_entry
                if me and me.live_api_key and me.live_account_id:
                    live_cfg = OANDAConfig(
                        api_key=me.live_api_key,
                        account_id=me.live_account_id,
                        environment="live",
                    )
                else:
                    live_cfg = OANDAConfig.live_from_env()
                if live_cfg is not None:
                    self._oanda_live = OANDAClient(live_cfg)
                    self._live_available = True
                    logger.info("OANDA live client initialised for model '%s' (account %s)",
                                self.model_id, live_cfg.account_id)
                else:
                    logger.info("No live OANDA credentials for model '%s' — demo only", self.model_id)
            except Exception as e:
                logger.warning("OANDA live init failed for model '%s' (continuing demo-only): %s",
                               self.model_id, e)

        # Lazy-load model
        if self._model is None:
            self._load_model()

        # Sync existing open positions from OANDA so we don't duplicate
        self._sync_open_positions()

        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name=f"live-stream-{self.model_id}"
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
        if self._oanda_demo is None:
            try:
                from wavetrader.oanda import OANDAClient, OANDAConfig
                me = self._model_entry
                if me and me.demo_api_key and me.demo_account_id:
                    demo_cfg = OANDAConfig(
                        api_key=me.demo_api_key,
                        account_id=me.demo_account_id,
                        environment="practice",
                    )
                else:
                    demo_cfg = OANDAConfig.demo_from_env()
                self._oanda_demo = OANDAClient(demo_cfg)
            except Exception as e:
                logger.error("Cannot init OANDA demo for model '%s': %s", self.model_id, e)
                return []

        from wavetrader.oanda import tf_to_granularity
        try:
            gran = tf_to_granularity(granularity)
        except ValueError:
            gran = granularity

        try:
            candles = self._oanda_demo.get_candles(pair, gran, count=count)
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
        """Fetch current OANDA account state (both demo and live)."""
        result: dict = {}

        # Demo account (always)
        if self._oanda_demo:
            try:
                acct = self._oanda_demo.get_account_summary()
                result["demo"] = {
                    "balance": acct.balance,
                    "nav": acct.nav,
                    "unrealized_pnl": acct.unrealized_pnl,
                    "margin_used": acct.margin_used,
                    "margin_available": acct.margin_available,
                    "open_trades": acct.open_trade_count,
                    "currency": acct.currency,
                }
                # For backward compat, also set top-level fields from demo
                result["balance"] = acct.balance
                result["nav"] = acct.nav
                result["unrealized_pnl"] = acct.unrealized_pnl
                result["margin_used"] = acct.margin_used
                result["margin_available"] = acct.margin_available
                result["open_trades"] = acct.open_trade_count
                result["currency"] = acct.currency
                result["market_open"] = self._oanda_demo.is_market_open()
            except Exception as e:
                result["demo"] = {"error": str(e)}
        else:
            result["demo"] = {"error": "OANDA demo not connected"}

        # Live account (optional)
        if self._oanda_live:
            try:
                acct = self._oanda_live.get_account_summary()
                result["live"] = {
                    "balance": acct.balance,
                    "nav": acct.nav,
                    "unrealized_pnl": acct.unrealized_pnl,
                    "margin_used": acct.margin_used,
                    "open_trades": acct.open_trade_count,
                    "currency": acct.currency,
                }
            except Exception as e:
                result["live"] = {"error": str(e)}
        else:
            result["live"] = {"status": "not_configured"}

        result["live_available"] = self._live_available
        return result

    def get_open_trades(self) -> list:
        """Fetch open trades from both OANDA accounts."""
        all_trades = []

        for label, client in [("demo", self._oanda_demo), ("live", self._oanda_live)]:
            if not client:
                continue
            try:
                trades = client.get_open_trades(self._pair)
                for t in trades:
                    # Look up our tracked initial SL and trailing state
                    if label == "demo" and self._demo_trade_id == t.trade_id:
                        initial_sl = self._demo_trade_initial_sl or t.stop_loss
                        tsl = self._demo_trade_sl if self._demo_trade_sl != self._demo_trade_initial_sl else None
                    elif label == "live" and self._live_trade_id == t.trade_id:
                        initial_sl = self._live_trade_initial_sl or t.stop_loss
                        tsl = self._live_trade_sl if self._live_trade_sl != self._live_trade_initial_sl else None
                    else:
                        initial_sl = t.stop_loss
                        tsl = None

                    all_trades.append({
                        "trade_id": t.trade_id,
                        "instrument": t.instrument,
                        "units": t.units,
                        "price": t.price,
                        "unrealized_pnl": t.unrealized_pnl,
                        "stop_loss": t.stop_loss,
                        "take_profit": t.take_profit,
                        "initial_stop_loss": initial_sl,
                        "trailing_stop_loss": tsl or t.trailing_stop_loss,
                        "direction": "BUY" if t.units > 0 else "SELL",
                        "account": label,
                    })
            except Exception as e:
                logger.error("get_open_trades (%s): %s", label, e)

        return all_trades

    def get_pending_orders(self) -> list:
        """Fetch pending orders from both OANDA accounts."""
        all_orders = []

        for label, client in [("demo", self._oanda_demo), ("live", self._oanda_live)]:
            if not client:
                continue
            try:
                orders = client.get_pending_orders(self._pair)
                for o in orders:
                    o["account"] = label
                all_orders.extend(orders)
            except Exception as e:
                logger.error("get_pending_orders (%s): %s", label, e)

        return all_orders

    def get_trade_history(self, pair: Optional[str] = None, count: int = 50) -> list:
        """Fetch trade history from both OANDA accounts (open + closed)."""
        all_trades = []

        # Build a lookup from trade_id → initial_sl from our own trade log
        initial_sl_map = {}
        for rec in self._trade_log:
            tid = rec.get("trade_id")
            if tid and "initial_sl" in rec:
                initial_sl_map[tid] = rec["initial_sl"]

        for label, client in [("demo", self._oanda_demo), ("live", self._oanda_live)]:
            if not client:
                continue
            try:
                trades = client.get_trade_history(pair=pair or self._pair, state="ALL", count=count)
                for t in trades:
                    t["account"] = label
                    # Enrich with initial_sl from our trade log if available
                    tid = t.get("trade_id")
                    if tid in initial_sl_map:
                        t["initial_sl"] = initial_sl_map[tid]
                    else:
                        # Fallback: if no TSL data, initial_sl = current sl
                        t.setdefault("initial_sl", t.get("sl"))
                all_trades.extend(trades)
            except Exception as e:
                logger.error("get_trade_history (%s): %s", label, e)

        # Sort by open_time descending
        all_trades.sort(key=lambda t: t.get("open_time", ""), reverse=True)
        return all_trades

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

    # ── Position sync ────────────────────────────────────────────

    def _sync_open_positions(self) -> None:
        """Check OANDA for existing open positions so we don't duplicate trades."""
        # Demo account
        if self._oanda_demo and self._demo_trade_id is None:
            try:
                trades = self._oanda_demo.get_open_trades(self._pair)
                if trades:
                    t = trades[0]
                    self._demo_trade_id = t.trade_id
                    self._demo_trade_direction = "BUY" if t.units > 0 else "SELL"
                    self._demo_trade_entry = t.price
                    self._demo_trade_sl = t.stop_loss or 0.0
                    self._demo_trade_tp = t.take_profit or 0.0
                    self._demo_trade_initial_sl = t.stop_loss or 0.0
                    self._demo_trade_highest = t.price
                    self._demo_trade_lowest = t.price
                    logger.info(
                        "Synced existing demo position: %s %s @ %.3f",
                        t.trade_id, self._demo_trade_direction, t.price,
                    )
            except Exception as e:
                logger.warning("Could not sync demo positions: %s", e)

        # Live account
        if self._oanda_live and self._live_trade_id is None:
            try:
                trades = self._oanda_live.get_open_trades(self._pair)
                if trades:
                    t = trades[0]
                    self._live_trade_id = t.trade_id
                    self._live_trade_direction = "BUY" if t.units > 0 else "SELL"
                    self._live_trade_entry = t.price
                    self._live_trade_sl = t.stop_loss or 0.0
                    self._live_trade_tp = t.take_profit or 0.0
                    self._live_trade_initial_sl = t.stop_loss or 0.0
                    self._live_trade_highest = t.price
                    self._live_trade_lowest = t.price
                    logger.info(
                        "Synced existing live position: %s %s @ %.3f",
                        t.trade_id, self._live_trade_direction, t.price,
                    )
            except Exception as e:
                logger.warning("Could not sync live positions: %s", e)

    # ── Auto-Trade (always on) ───────────────────────────────────

    def update_config(self, cfg: dict) -> dict:
        """Update live trading configuration at runtime. Returns the new config."""
        if "min_confidence" in cfg:
            self._min_confidence = float(cfg["min_confidence"])
        if "risk_per_trade" in cfg:
            self._risk_per_trade = float(cfg["risk_per_trade"])
        if "atr_halt_multiplier" in cfg:
            self._atr_halt_multiplier = float(cfg["atr_halt_multiplier"])
        if "drawdown_reduce_threshold" in cfg:
            self._drawdown_reduce_threshold = float(cfg["drawdown_reduce_threshold"])
        if "friction" in cfg:
            f = cfg["friction"]
            if "slippage_min" in f:
                self._friction_slippage_min = float(f["slippage_min"])
            if "slippage_max" in f:
                self._friction_slippage_max = float(f["slippage_max"])
            if "spread_offhours_extra" in f:
                self._friction_spread_offhours = float(f["spread_offhours_extra"])
            if "news_spike_prob" in f:
                self._friction_news_prob = float(f["news_spike_prob"])
            if "news_spike_extra" in f:
                self._friction_news_extra = float(f["news_spike_extra"])
            if "lot_cap" in f:
                self._friction_lot_cap = float(f["lot_cap"])
        logger.info("Live config updated: conf=%.2f risk=%.2f", self._min_confidence, self._risk_per_trade)
        return self.live_config

    @property
    def live_config(self) -> dict:
        """Return the current live trading configuration."""
        return {
            "min_confidence": self._min_confidence,
            "risk_per_trade": self._risk_per_trade,
            "atr_halt_multiplier": getattr(self, "_atr_halt_multiplier", 3.0),
            "drawdown_reduce_threshold": getattr(self, "_drawdown_reduce_threshold", 0.10),
            "friction": {
                "slippage_min": getattr(self, "_friction_slippage_min", 0.5),
                "slippage_max": getattr(self, "_friction_slippage_max", 3.0),
                "spread_offhours_extra": getattr(self, "_friction_spread_offhours", 2.5),
                "news_spike_prob": getattr(self, "_friction_news_prob", 0.05),
                "news_spike_extra": getattr(self, "_friction_news_extra", 5.0),
                "lot_cap": getattr(self, "_friction_lot_cap", 2.0),
            },
        }

    @property
    def auto_trade_status(self) -> dict:
        return {
            "enabled": True,
            "demo_active": self._oanda_demo is not None,
            "live_active": self._live_available,
            "demo_trade": self._demo_trade_id,
            "live_trade": self._live_trade_id,
            "demo_direction": self._demo_trade_direction,
            "live_direction": self._live_trade_direction,
            "min_confidence": self._min_confidence,
            "risk_per_trade": self._risk_per_trade,
            "recent_trades": self._trade_log[-10:],
            "recent_signals": self._signal_history[-10:],
        }

    # ── LLM Arbiter helpers ───────────────────────────────────────────

    def _evaluate_with_arbiter(
        self,
        signal_dict: dict,
        tf_history: dict,
        current_price: float,
    ):
        """Build ArbiterContext and call the LLM arbiter."""
        from wavetrader.llm_arbiter import ArbiterContext

        # Determine current session
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 0 <= hour < 9:
            session = "Tokyo"
        elif 7 <= hour < 16:
            session = "London"
        elif 12 <= hour < 21:
            session = "New York"
        else:
            session = "Off-hours"

        # Get recent bars from entry timeframe
        entry_bars = tf_history.get(self._timeframe, [])[-self._arbiter_config.recent_bars_count:]

        # Calendar events
        events = []
        has_high_impact = False
        try:
            events = self._calendar.get_upcoming(self._pair, hours_ahead=4)
            has_high_impact = any(e.impact == "high" for e in events)
        except Exception as e:
            logger.debug("Calendar fetch for arbiter failed: %s", e)

        # Portfolio state
        balance = 0.0
        unrealized_pnl = 0.0
        try:
            acct = self.get_account()
            balance = acct.get("balance", 0)
            unrealized_pnl = acct.get("unrealized_pnl", 0)
        except Exception:
            pass

        # Open positions
        open_positions = []
        try:
            open_positions = self.get_open_trades()
        except Exception:
            pass

        # Win rate from trade log
        total = len(self._trade_log)
        wins = sum(1 for t in self._trade_log if float(t.get("realized_pl", 0)) > 0)
        win_rate = wins / total if total > 0 else 0

        ctx = ArbiterContext(
            signal=signal_dict.get("signal", "HOLD"),
            confidence=signal_dict.get("confidence", 0),
            alignment=signal_dict.get("alignment", 0),
            sl_pips=signal_dict.get("sl_pips", 0),
            tp_pips=signal_dict.get("tp_pips", 0),
            entry_price=current_price,
            model_id=self.model_id,
            pair=self._pair,
            timeframe=self._timeframe,
            recent_bars=entry_bars,
            balance=balance,
            unrealized_pnl=unrealized_pnl,
            open_positions=[p for p in open_positions],
            max_drawdown=0.0,
            win_rate=win_rate,
            total_trades=total,
            recent_trades=self._trade_log[-self._arbiter_config.recent_trades_count:],
            calendar_events=[e.to_dict() for e in events],
            has_high_impact_event=has_high_impact,
            current_session=session,
        )

        return self._arbiter.evaluate(ctx)

    @property
    def arbiter_status(self) -> dict:
        """Return current LLM arbiter configuration and stats."""
        stats = self._decision_log.get_stats()
        return {
            "enabled": self._arbiter_config.enabled,
            "authority_mode": self._arbiter_config.authority_mode,
            "model": self._arbiter_config.model,
            "escalation_model": self._arbiter_config.escalation_model,
            "stats": stats,
        }

    def get_arbiter_decisions(self, count: int = 50) -> list:
        """Return recent LLM arbiter decisions."""
        return self._decision_log.get_recent(count)

    def update_arbiter_config(self, cfg: dict) -> dict:
        """Update arbiter configuration at runtime (mode locked to override)."""
        if "model" in cfg:
            self._arbiter_config.model = cfg["model"]
        logger.info(
            "Arbiter config updated: enabled=%s mode=%s model=%s",
            self._arbiter_config.enabled,
            self._arbiter_config.authority_mode,
            self._arbiter_config.model,
        )
        return self.arbiter_status

    def run_inspection(self) -> dict:
        """
        Run a manual LLM market inspection across ALL models.

        Gathers context from every registered model's LiveService, then calls
        the arbiter's ``inspect()`` method for a comprehensive analysis.
        If the LLM recommends a trade, execute it through the specified model.
        """
        from .model_registry import get_model_registry
        from wavetrader.calendar import get_calendar

        registry = get_model_registry()
        cal = get_calendar()
        pair = self._pair or "GBP/JPY"

        # Determine session
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 0 <= hour < 9:
            session = "Tokyo"
        elif 7 <= hour < 16:
            session = "London"
        elif 12 <= hour < 21:
            session = "New York"
        else:
            session = "Off-hours"

        # Collect recent bars from this service's OANDA connection
        recent_bars = []
        try:
            # Ensure OANDA client is available for fetching bars
            oanda = self._oanda_demo
            if oanda is None:
                from wavetrader.oanda import OANDAClient, OANDAConfig
                me = self._model_entry
                if me and me.demo_api_key and me.demo_account_id:
                    demo_cfg = OANDAConfig(
                        api_key=me.demo_api_key,
                        account_id=me.demo_account_id,
                        environment="practice",
                    )
                else:
                    demo_cfg = OANDAConfig.demo_from_env()
                oanda = OANDAClient(demo_cfg)
            candles = oanda.get_candles(pair, "M15", count=40)
            for c in candles:
                recent_bars.append({
                    "time": c.timestamp.isoformat() if c.timestamp else "",
                    "open": c.open, "high": c.high, "low": c.low,
                    "close": c.close, "volume": c.volume,
                })
        except Exception as e:
            logger.warning("Inspection: could not fetch bars: %s", e)

        # Collect calendar events
        events = []
        try:
            raw_events = cal.get_upcoming(pair, hours_ahead=4)
            events = [e.to_dict() for e in raw_events]
        except Exception:
            pass

        # Collect per-model state
        models = {}
        models_info = {}
        for entry in registry.list_models():
            mid = entry.id
            models_info[mid] = {
                "name": entry.name,
                "description": entry.description or f"{entry.model_type} model",
                "pair": entry.pair,
            }
            try:
                svc = get_live_service(mid)
                acct = {}
                try:
                    acct = svc.get_account()
                except Exception:
                    pass

                positions = []
                try:
                    positions = svc.get_open_trades()
                except Exception:
                    pass

                recent_trades = []
                try:
                    recent_trades = svc.get_trade_history(pair=pair, count=10)
                except Exception:
                    pass

                recent_signals = getattr(svc, "_signal_history", [])[-5:]

                models[mid] = {
                    "name": entry.name,
                    "balance": acct.get("balance", 0),
                    "nav": acct.get("nav", 0),
                    "unrealized_pnl": acct.get("unrealized_pnl", 0),
                    "margin_used": acct.get("margin_used", 0),
                    "margin_available": acct.get("margin_available", 0),
                    "open_positions": positions,
                    "recent_trades": recent_trades,
                    "recent_signals": recent_signals,
                }
            except Exception as e:
                logger.warning("Inspection: could not load model %s: %s", mid, e)
                models[mid] = {"name": entry.name, "balance": 0, "nav": 0,
                               "unrealized_pnl": 0, "margin_used": 0,
                               "margin_available": 0, "open_positions": [],
                               "recent_trades": [], "recent_signals": []}

        context = {
            "pair": pair,
            "current_session": session,
            "recent_bars": recent_bars,
            "models": models,
            "models_info": models_info,
            "calendar_events": events,
        }

        # Run the LLM inspection
        result = self._arbiter.inspect(context)

        # If the LLM recommends a trade, execute it
        trade_executed = None
        ta = result.get("trade_action")
        if ta and ta.get("signal") in ("BUY", "SELL"):
            target_model = ta["model_id"]
            try:
                target_svc = get_live_service(target_model)

                # Ensure OANDA client is available on the target service
                if target_svc._oanda_demo is None:
                    from wavetrader.oanda import OANDAClient, OANDAConfig
                    me = target_svc._model_entry
                    if me and me.demo_api_key and me.demo_account_id:
                        demo_cfg = OANDAConfig(
                            api_key=me.demo_api_key,
                            account_id=me.demo_account_id,
                            environment="practice",
                        )
                    else:
                        demo_cfg = OANDAConfig.demo_from_env()
                    target_svc._oanda_demo = OANDAClient(demo_cfg)

                # Sync target service state with OANDA before executing —
                # the streaming engine may have opened positions that the
                # dashboard's LiveService doesn't know about.
                try:
                    existing = target_svc._oanda_demo.get_open_trades(pair)
                    if existing:
                        t = existing[0]
                        direction = "BUY" if t.units > 0 else "SELL"
                        target_svc._demo_trade_id = t.trade_id
                        target_svc._demo_trade_direction = direction
                        target_svc._demo_trade_entry = t.price
                        logger.info(
                            "Inspection sync: found existing %s position %s on %s",
                            direction, t.trade_id, target_model,
                        )
                    else:
                        target_svc._demo_trade_id = None
                        target_svc._demo_trade_direction = None
                except Exception as sync_err:
                    logger.warning("Inspection OANDA sync failed: %s", sync_err)

                # If LLM says close_first, close the existing position before
                # placing the new trade so margin is freed up.
                if ta.get("close_first") and target_svc._demo_trade_id:
                    logger.info(
                        "Inspection: closing existing %s position %s first (LLM requested)",
                        target_svc._demo_trade_direction, target_svc._demo_trade_id,
                    )
                    target_svc._close_position_on("demo", "LLM inspection: close before reversal")

                # Get current price
                price_data = target_svc._oanda_demo.get_price(pair)
                current_price = round((price_data["bid"] + price_data["ask"]) / 2, 5)

                signal_dict = {
                    "signal": ta["signal"],
                    "confidence": ta.get("confidence", 0.7),
                    "alignment": 0.0,
                    "sl_pips": ta["sl_pips"],
                    "tp_pips": ta["tp_pips"],
                    "trailing_pct": 0.0,
                    "timestamp": now.isoformat(),
                    "_source": "llm_inspection",
                }
                target_svc._execute_signal(signal_dict, current_price)
                trade_executed = {
                    "model_id": target_model,
                    "signal": ta["signal"],
                    "price": current_price,
                    "sl_pips": ta["sl_pips"],
                    "tp_pips": ta["tp_pips"],
                }
                logger.info(
                    "LLM Inspection trade executed: %s on %s @ %.3f",
                    ta["signal"], target_model, current_price,
                )
            except Exception as e:
                logger.error("LLM Inspection trade execution failed: %s", e)
                trade_executed = {"error": str(e)}

        result["trade_executed"] = trade_executed

        # Broadcast to SSE
        self.broadcaster.publish("inspection", result)

        # Log the inspection
        self._decision_log.log_decision({
            "decision_id": result.get("inspection_id", ""),
            "timestamp": result.get("timestamp", ""),
            "model_id": "inspection",
            "pair": pair,
            "original_signal": "INSPECT",
            "original_confidence": 0,
            "action": "TRADE" if trade_executed and "error" not in trade_executed else "ANALYSIS",
            "reasoning": (result.get("analysis", "") or "")[:500],
            "confidence_adjustment": 0,
            "modified_signal": ta["signal"] if ta else None,
            "modified_sl_pips": ta["sl_pips"] if ta else None,
            "modified_tp_pips": ta["tp_pips"] if ta else None,
            "risk_notes": "; ".join(result.get("risk_warnings", [])),
            "model_used": result.get("model_used", ""),
            "latency_ms": result.get("latency_ms", 0),
            "entry_price": trade_executed.get("price", 0) if trade_executed and isinstance(trade_executed, dict) else 0,
            "trade_placed": bool(trade_executed and "error" not in (trade_executed or {})),
        })

        return result

    def _execute_signal(self, signal_dict: dict, current_price: float) -> None:
        """Execute a trade based on model signal on both accounts."""
        sig = signal_dict["signal"]  # "BUY", "SELL", "HOLD"
        conf = signal_dict["confidence"]

        # Track all signals for diagnostics
        self._signal_history.append({
            "signal": sig,
            "confidence": conf,
            "alignment": signal_dict.get("alignment", 0),
            "timestamp": signal_dict.get("timestamp", ""),
            "price": current_price,
        })
        self._signal_history = self._signal_history[-50:]

        logger.info(
            "Signal received: %s conf=%.4f align=%.4f price=%.3f (threshold=%.2f)",
            sig, conf, signal_dict.get("alignment", 0), current_price, self._min_confidence,
        )

        # Skip HOLD signals
        if sig == "HOLD":
            logger.info("Skipping HOLD signal")
            return

        # Skip low-confidence signals
        if conf < self._min_confidence:
            logger.warning(
                "Skipping %s signal: confidence %.4f < threshold %.2f",
                sig, conf, self._min_confidence,
            )
            return

        # Same direction as existing position → hold, don't duplicate
        if self._demo_trade_id and self._demo_trade_direction == sig:
            logger.info("Already %s — holding demo position %s", sig, self._demo_trade_id)
            return
        if self._live_trade_id and self._live_trade_direction == sig:
            logger.info("Already %s — holding live position %s", sig, self._live_trade_id)
            return

        # Opposite direction → close then immediately open (atomic reversal)
        if self._demo_trade_id and self._demo_trade_direction != sig:
            self._close_position_on("demo", "Signal reversal")
        if self._live_trade_id and self._live_trade_direction != sig:
            self._close_position_on("live", "Signal reversal")

        # Open new positions (now flat after reversal, or was already flat)
        if self._demo_trade_id is None:
            self._open_position_on("demo", signal_dict, current_price)
        if self._live_available and self._live_trade_id is None:
            self._open_position_on("live", signal_dict, current_price)

    def _open_position_on(self, account: str, signal_dict: dict, current_price: float) -> None:
        """Open a position via OANDA on the specified account (demo or live)."""
        client = self._oanda_demo if account == "demo" else self._oanda_live
        if not client:
            return

        # Safety: check OANDA for existing positions to prevent FIFO violations
        try:
            existing = client.get_open_trades(self._pair)
            if existing:
                t = existing[0]
                direction = "BUY" if t.units > 0 else "SELL"
                if account == "demo":
                    self._demo_trade_id = t.trade_id
                    self._demo_trade_direction = direction
                    self._demo_trade_entry = t.price
                else:
                    self._live_trade_id = t.trade_id
                    self._live_trade_direction = direction
                    self._live_trade_entry = t.price
                logger.info(
                    "Already have open %s position [%s] %s @ %.3f — skipping new order",
                    direction, account.upper(), t.trade_id, t.price,
                )
                return
        except Exception as e:
            logger.warning("Could not check existing trades [%s]: %s", account, e)

        _PIP_SIZE = {"GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01, "GBP/USD": 0.0001}
        _PIP_VALUE = {"GBP/JPY": 6.5, "EUR/JPY": 6.7, "USD/JPY": 6.5, "GBP/USD": 10.0}

        sig = signal_dict["signal"]
        pip = _PIP_SIZE.get(self._pair, 0.01)
        pip_value = _PIP_VALUE.get(self._pair, 6.5)

        # Get balance and check margin
        try:
            acct = client.get_account_summary()
            balance = acct.balance
            if acct.margin_available <= 0:
                logger.warning(
                    "No margin available [%s] (margin_avail=%.2f) — skipping order",
                    account.upper(), acct.margin_available,
                )
                return
        except Exception:
            balance = 25000

        # Position sizing — no artificial cap, OANDA enforces margin
        sl_pips = signal_dict["sl_pips"]
        tp_pips = signal_dict["tp_pips"]
        risk_amount = balance * self._risk_per_trade
        lot = risk_amount / max(sl_pips * pip_value, 1e-9)
        lot = max(0.01, lot)
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
            "initial_sl": round(sl_price, 5),
            "tp": round(tp_price, 5),
            "confidence": signal_dict["confidence"],
            "trailing_pct": signal_dict.get("trailing_pct", 0.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": account,
        }

        logger.info(
            "Placing order [%s]: %s %d units %s SL=%.3f TP=%.3f",
            account.upper(), sig, units, self._pair, sl_price, tp_price,
        )
        try:
            order = client.place_market_order(
                self._pair, units, sl=sl_price, tp=tp_price,
            )
            if order.status == "FILLED":
                trade_record["trade_id"] = order.trade_id
                trade_record["status"] = "filled"
                trailing_pct = signal_dict.get("trailing_pct", 0.0)
                if account == "demo":
                    self._demo_trade_id = order.trade_id
                    self._demo_trade_direction = sig
                    self._demo_trade_entry = order.price
                    self._demo_trade_sl = sl_price
                    self._demo_trade_tp = tp_price
                    self._demo_trade_initial_sl = sl_price
                    self._demo_trade_trailing_pct = trailing_pct
                    self._demo_trade_highest = order.price
                    self._demo_trade_lowest = order.price
                else:
                    self._live_trade_id = order.trade_id
                    self._live_trade_direction = sig
                    self._live_trade_entry = order.price
                    self._live_trade_sl = sl_price
                    self._live_trade_tp = tp_price
                    self._live_trade_initial_sl = sl_price
                    self._live_trade_trailing_pct = trailing_pct
                    self._live_trade_highest = order.price
                    self._live_trade_lowest = order.price
                logger.info(
                    "OANDA [%s] order filled: %s @ %.3f",
                    account.upper(), order.trade_id, order.price,
                )
            else:
                trade_record["status"] = f"rejected: {order.status}"
                logger.error("OANDA [%s] order rejected: %s", account.upper(), order.status)
        except Exception as e:
            trade_record["status"] = f"error: {e}"
            logger.error("OANDA [%s] failed to place order: %s", account.upper(), e)

        self._trade_log.append(trade_record)
        self.broadcaster.publish("trade_executed", trade_record)

    def _close_position_on(self, account: str, reason: str) -> None:
        """Close the current open position on the specified account."""
        if account == "demo":
            trade_id = self._demo_trade_id
            direction = self._demo_trade_direction
        else:
            trade_id = self._live_trade_id
            direction = self._live_trade_direction

        if not trade_id:
            return

        client = self._oanda_demo if account == "demo" else self._oanda_live

        close_record = {
            "trade_id": trade_id,
            "direction": direction,
            "pair": self._pair,
            "reason": reason,
            "account": account,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if client:
            try:
                client.close_trade(trade_id)
                logger.info("OANDA [%s] trade closed: %s (%s)", account.upper(), trade_id, reason)
            except Exception as e:
                logger.error("OANDA [%s] failed to close trade %s: %s", account.upper(), trade_id, e)

        if account == "demo":
            self._demo_trade_id = None
            self._demo_trade_direction = None
            self._demo_trade_entry = 0.0
            self._demo_trade_initial_sl = 0.0
            self._demo_trade_trailing_pct = 0.0
            self._demo_trade_highest = 0.0
            self._demo_trade_lowest = float("inf")
        else:
            self._live_trade_id = None
            self._live_trade_direction = None
            self._live_trade_entry = 0.0
            self._live_trade_initial_sl = 0.0
            self._live_trade_trailing_pct = 0.0
            self._live_trade_highest = 0.0
            self._live_trade_lowest = float("inf")

        self.broadcaster.publish("trade_closed", close_record)

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Try to load the latest model checkpoint (type-aware via registry)."""
        try:
            import torch

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Determine model type from registry
            model_type = "mtf"
            if self._model_entry:
                model_type = getattr(self._model_entry, "model_type", "mtf")

            # Search for latest checkpoint
            ckpt_dirs = [
                Path("checkpoints"),
                Path("/checkpoints"),
                Path("data/live_checkpoints"),
            ]
            # If the registry entry specifies a checkpoint dir, try that first
            if self._model_entry and self._model_entry.checkpoint_dir:
                ckpt_dirs.insert(0, Path(self._model_entry.checkpoint_dir))

            loaded = False
            for ckpt_dir in ckpt_dirs:
                if not ckpt_dir.is_dir():
                    continue
                # Filter checkpoint subdirs by model type prefix
                if model_type == "wavefollower":
                    prefix = "wavefollower_"
                elif model_type == "meanrev":
                    prefix = "mean_reversion_"
                elif model_type == "amd_scalper":
                    prefix = "amd_scalper_"
                else:
                    prefix = "wavetrader_mtf_"
                subdirs = sorted(
                    (d for d in ckpt_dir.iterdir()
                     if d.is_dir() and d.name.startswith(prefix)),
                    reverse=True,
                )
                # Fallback: if no prefix-matched dirs, try all subdirs
                if not subdirs:
                    subdirs = sorted(
                        (d for d in ckpt_dir.iterdir() if d.is_dir()),
                        reverse=True,
                    )
                for d in subdirs:
                    weights = d / "model_weights.pt"
                    if weights.exists():
                        if model_type == "wavefollower":
                            from wavetrader.wave_follower import WaveFollower, WaveFollowerConfig
                            self._model_config = WaveFollowerConfig(pair=self._pair)
                            self._model = WaveFollower(self._model_config)
                        elif model_type == "meanrev":
                            from wavetrader.mean_reversion import MeanReversion
                            from wavetrader.config import MeanRevConfig
                            self._model_config = MeanRevConfig(pair=self._pair)
                            self._model = MeanReversion(self._model_config)
                        elif model_type == "amd_scalper":
                            from wavetrader.amd_scalper import AMDScalper
                            from wavetrader.config import AMDScalperConfig
                            self._model_config = AMDScalperConfig(pair=self._pair)
                            self._model = AMDScalper(self._model_config)
                        else:
                            from wavetrader.config import MTFConfig
                            from wavetrader.model import WaveTraderMTF
                            self._model_config = MTFConfig(pair=self._pair)
                            self._model = WaveTraderMTF(self._model_config)

                        state = torch.load(
                            str(weights), weights_only=False, map_location=self._device
                        )
                        if "model_state_dict" in state:
                            raw_sd = state["model_state_dict"]
                        else:
                            raw_sd = state
                        # Strip _orig_mod. prefix from torch.compile'd checkpoints
                        cleaned = {k.replace("_orig_mod.", ""): v
                                   for k, v in raw_sd.items()}
                        self._model.load_state_dict(cleaned, strict=False)
                        self._model.to(self._device)
                        self._model.eval()
                        loaded = True
                        logger.info("Loaded %s model from %s", model_type, weights)
                        break
                if loaded:
                    break

            if not loaded:
                logger.warning("No model checkpoint found for '%s' — signals will not be generated",
                               self.model_id)
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

            result = {
                "signal": signal_name,
                "confidence": round(confidence * (0.5 + 0.5 * alignment), 4),
                "alignment": round(alignment, 4),
                "sl_pips": round(float(risk[0].item() * 50 + 10), 1),
                "tp_pips": round(float(risk[1].item() * 100 + 20), 1),
                "trailing_pct": round(float(risk[2].item() * 0.5), 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # WaveFollower-specific outputs
            if "trend_logits" in out:
                trend_idx = out["trend_logits"].argmax(-1).item()
                result["trend"] = ["UP", "DOWN", "NEUTRAL"][trend_idx]
            if "add_score" in out:
                result["add_score"] = round(out["add_score"].item(), 4)

            return result

        except Exception as e:
            logger.error("Inference failed: %s", e)
            return None

    # ── Trailing stop management ──────────────────────────────────────────

    _MIN_TRAIL_PIPS = {"GBP/JPY": 5.0, "EUR/JPY": 5.0, "USD/JPY": 5.0, "GBP/USD": 5.0}
    _PIP_SIZE_MAP = {"GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01, "GBP/USD": 0.0001}

    def _update_trailing_stops(self, bid: float, ask: float) -> None:
        """Check and update trailing stop losses for all open positions."""
        pip = self._PIP_SIZE_MAP.get(self._pair, 0.01)
        min_trail = self._MIN_TRAIL_PIPS.get(self._pair, 5.0) * pip

        for account in ("demo", "live"):
            if account == "demo":
                trade_id = self._demo_trade_id
                direction = self._demo_trade_direction
                entry = self._demo_trade_entry
                current_sl = self._demo_trade_sl
                initial_sl = self._demo_trade_initial_sl
                trailing_pct = self._demo_trade_trailing_pct
                highest = self._demo_trade_highest
                lowest = self._demo_trade_lowest
                client = self._oanda_demo
            else:
                trade_id = self._live_trade_id
                direction = self._live_trade_direction
                entry = self._live_trade_entry
                current_sl = self._live_trade_sl
                initial_sl = self._live_trade_initial_sl
                trailing_pct = self._live_trade_trailing_pct
                highest = self._live_trade_highest
                lowest = self._live_trade_lowest
                client = self._oanda_live

            if not trade_id or not client or trailing_pct <= 0 or initial_sl == 0:
                continue

            initial_risk = abs(entry - initial_sl)
            if initial_risk <= 0:
                continue

            trail_dist = initial_risk * (1.0 - trailing_pct)
            trail_dist = max(trail_dist, min_trail)

            new_sl = current_sl
            if direction == "BUY":
                current_high = ask  # Best price for long
                if current_high > highest:
                    if account == "demo":
                        self._demo_trade_highest = current_high
                    else:
                        self._live_trade_highest = current_high
                    highest = current_high
                candidate_sl = highest - trail_dist
                if candidate_sl > current_sl:
                    new_sl = round(candidate_sl, 3 if "JPY" in self._pair else 5)
            elif direction == "SELL":
                current_low = bid  # Best price for short
                if current_low < lowest:
                    if account == "demo":
                        self._demo_trade_lowest = current_low
                    else:
                        self._live_trade_lowest = current_low
                    lowest = current_low
                candidate_sl = lowest + trail_dist
                if candidate_sl < current_sl:
                    new_sl = round(candidate_sl, 3 if "JPY" in self._pair else 5)

            # Only modify if SL actually moved
            if new_sl != current_sl:
                try:
                    client.modify_trade(trade_id, sl=new_sl)
                    if account == "demo":
                        self._demo_trade_sl = new_sl
                    else:
                        self._live_trade_sl = new_sl
                    logger.info(
                        "TSL updated [%s]: %s SL %.3f → %.3f (initial: %.3f)",
                        account.upper(), trade_id, current_sl, new_sl, initial_sl,
                    )
                except Exception as e:
                    logger.error("Failed to update TSL [%s] %s: %s", account, trade_id, e)

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
                candles = self._oanda_demo.get_candles(self._pair, gran, count=300)
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
                candles = self._oanda_demo.get_candles(self._pair, gran, count=3)

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
                                        htf_candles = self._oanda_demo.get_candles(self._pair, g, count=200)
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
                                # ── LLM Arbiter evaluation ────────────
                                arbiter_decision = None
                                if self._arbiter_config.enabled:
                                    try:
                                        arbiter_decision = self._evaluate_with_arbiter(
                                            signal, tf_history, c.close,
                                        )
                                    except Exception as e:
                                        logger.error("Arbiter context build failed: %s", e)
                                        # Create a fallback APPROVE so we still broadcast
                                        from wavetrader.llm_arbiter import ArbiterDecision
                                        from datetime import datetime, timezone
                                        arbiter_decision = ArbiterDecision(
                                            decision_id=str(uuid.uuid4())[:12],
                                            action="APPROVE",
                                            reasoning=f"Arbiter context error — defaulting to APPROVE: {e}",
                                            model_used="none",
                                            timestamp=datetime.now(timezone.utc).isoformat(),
                                            original_signal=signal.get("signal", "?"),
                                            original_confidence=signal.get("confidence", 0),
                                            entry_price=c.close,
                                        )
                                    try:
                                        # Apply decision to signal
                                        signal = self._arbiter.apply_decision(
                                            signal, arbiter_decision,
                                        )
                                        # Broadcast arbiter decision via SSE
                                        from dataclasses import asdict
                                        dec_dict = asdict(arbiter_decision)
                                        dec_dict["trade_placed"] = signal.get("signal") != "HOLD"
                                        self.broadcaster.publish("arbiter", dec_dict)
                                        # Log it
                                        dec_dict["model_id"] = self.model_id
                                        dec_dict["pair"] = self._pair
                                        self._decision_log.log_decision(dec_dict)
                                        self._arbiter_decisions.append(dec_dict)
                                        self._arbiter_decisions = self._arbiter_decisions[-100:]
                                    except Exception as e:
                                        logger.error("Arbiter broadcast/log failed: %s", e)

                                self.broadcaster.publish("signal", signal)
                                # Auto-execute trade on both accounts
                                self._execute_signal(signal, c.close)

                # Also push current price and update trailing stops
                try:
                    price = self._oanda_demo.get_price(self._pair)
                    mid = round((price["bid"] + price["ask"]) / 2, 5)
                    self.broadcaster.publish("price", {
                        "bid": price["bid"],
                        "ask": price["ask"],
                        "spread": round(price["spread"], 5),
                        "mid": mid,
                        "time": price["time"],
                    })
                    # Update trailing stops based on current price
                    self._update_trailing_stops(price["bid"], price["ask"])
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


# ── Module-level service registry ─────────────────────────────────────────────
_live_services: Dict[str, LiveService] = {}


def get_live_service(model_id: Optional[str] = None) -> LiveService:
    """Return (or create) the LiveService for the given model.

    Falls back to the default model if *model_id* is not specified.
    """
    from .model_registry import get_model_registry

    if model_id is None:
        model_id = get_model_registry().default_id

    if model_id not in _live_services:
        _live_services[model_id] = LiveService(model_id=model_id)
    return _live_services[model_id]
