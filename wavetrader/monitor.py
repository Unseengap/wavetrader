"""
Monitoring and alerting for WaveTrader live trading.

Supports:
  - Telegram notifications (trades, alerts, daily summaries)
  - Console logging (always on)
  - Metrics tracking (inference latency, signal distribution)

Setup:
  1. Create a Telegram bot via @BotFather → get token
  2. Send a message to your bot, then fetch your chat_id
  3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Monitoring configuration."""
    telegram_token: str = ""
    telegram_chat_id: str = ""           # Admin notifications
    telegram_channel_id: str = ""        # Public signal broadcast channel
    daily_summary_hour: int = 22  # UTC hour to send daily summary
    max_alerts_per_hour: int = 10  # Rate limit alerts

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        return cls(
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            telegram_channel_id=os.environ.get("TELEGRAM_CHANNEL_ID", ""),
        )


@dataclass
class InferenceMetrics:
    """Rolling metrics for monitoring."""
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    signal_counts: Dict[str, int] = field(default_factory=lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})
    trade_pnls: List[float] = field(default_factory=list)
    last_summary_date: Optional[str] = None


class Monitor:
    """
    Monitoring and alerting system.

    Sends Telegram messages for:
      - Trade opens/closes
      - Critical alerts (circuit breakers, errors, checkpoint failures)
      - Daily performance summaries
      - Info messages (start/stop)
    """

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self.config = config or MonitorConfig.from_env()
        self.metrics = InferenceMetrics()
        self._alert_timestamps: deque = deque(maxlen=100)
        self._telegram_ok = bool(self.config.telegram_token and self.config.telegram_chat_id)
        self._channel_ok = bool(self.config.telegram_token and self.config.telegram_channel_id)

        if self._telegram_ok:
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram not configured — console-only monitoring")
        if self._channel_ok:
            logger.info("Telegram signal channel enabled: %s", self.config.telegram_channel_id)

    # ── Telegram ──────────────────────────────────────────────────────────

    def _send_telegram(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API."""
        if not self._telegram_ok:
            return False

        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": self.config.telegram_chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Telegram send failed: %s", resp.text)
                return False
            return True
        except Exception as e:
            logger.warning("Telegram error: %s", e)
            return False

    # ── Public API ────────────────────────────────────────────────────────

    def send_trade(self, message: str) -> None:
        """Notify about a trade open/close."""
        formatted = f"📊 <b>TRADE</b>\n{message}"
        logger.info("[TRADE] %s", message)
        self._send_telegram(formatted)

    def broadcast_signal(self, signal: Any, pair: str, price: float) -> None:
        """
        Broadcast a trade signal to the public Telegram channel.

        This is the signal-as-a-service feature: subscribers see signals
        and can choose to execute manually on their own accounts.
        """
        if not self._channel_ok:
            return

        pip_size = 0.01 if "JPY" in pair else 0.0001
        sig = signal.signal if hasattr(signal, "signal") else signal
        sig_name = sig.name if hasattr(sig, "name") else str(sig)

        if sig_name == "HOLD":
            return

        sl_price = price - signal.stop_loss * pip_size if sig_name == "BUY" else price + signal.stop_loss * pip_size
        tp_price = price + signal.take_profit * pip_size if sig_name == "BUY" else price - signal.take_profit * pip_size

        emoji = "🟢" if sig_name == "BUY" else "🔴"
        stars = "⭐" * min(5, max(1, int(signal.confidence * 5)))

        msg = (
            f"{emoji} <b>WaveTrader Signal</b>\n\n"
            f"<b>{sig_name} {pair}</b>\n"
            f"Entry: <code>{price:.3f}</code>\n"
            f"Stop Loss: <code>{sl_price:.3f}</code> ({signal.stop_loss:.0f} pips)\n"
            f"Take Profit: <code>{tp_price:.3f}</code> ({signal.take_profit:.0f} pips)\n"
            f"Confidence: {signal.confidence:.1%} {stars}\n\n"
            f"<i>⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>\n"
            f"<i>Risk management: never risk more than 1-2% per trade</i>"
        )

        self._send_to_channel(msg)

    def broadcast_close(self, pair: str, reason: str, pnl: float) -> None:
        """Broadcast a position close to the signal channel."""
        if not self._channel_ok:
            return
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} <b>Close Signal</b>\n\n"
            f"<b>{pair}</b> — {reason}\n"
            f"Result: <b>${pnl:+.2f}</b>"
        )
        self._send_to_channel(msg)

    def _send_to_channel(self, text: str) -> bool:
        """Send a message to the public Telegram signal channel."""
        if not self._channel_ok:
            return False
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": self.config.telegram_channel_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Channel broadcast failed: %s", resp.text)
                return False
            return True
        except Exception as e:
            logger.warning("Channel broadcast error: %s", e)
            return False

    def send_alert(self, message: str) -> None:
        """Send a critical alert (rate-limited)."""
        now = time.time()
        # Rate limit: max N alerts per hour
        recent = [t for t in self._alert_timestamps if now - t < 3600]
        if len(recent) >= self.config.max_alerts_per_hour:
            logger.warning("Alert rate limit reached — suppressing: %s", message)
            return

        self._alert_timestamps.append(now)
        formatted = f"🚨 <b>ALERT</b>\n{message}"
        logger.warning("[ALERT] %s", message)
        self._send_telegram(formatted)

    def send_info(self, message: str) -> None:
        """Send an informational message."""
        formatted = f"ℹ️ {message}"
        logger.info("[INFO] %s", message)
        self._send_telegram(formatted)

    def record_inference(self, latency_ms: float, signal: Any) -> None:
        """Record inference metrics."""
        self.metrics.latencies.append(latency_ms)
        signal_name = signal.signal.name if hasattr(signal, "signal") else str(signal)
        self.metrics.signal_counts[signal_name] = (
            self.metrics.signal_counts.get(signal_name, 0) + 1
        )

        # Check if daily summary is due
        self._check_daily_summary()

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a closed trade's PnL."""
        self.metrics.trade_pnls.append(pnl)

    # ── Daily summary ─────────────────────────────────────────────────────

    def _check_daily_summary(self) -> None:
        """Send daily summary at configured hour."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        if (
            now.hour == self.config.daily_summary_hour
            and self.metrics.last_summary_date != today
        ):
            self._send_daily_summary()
            self.metrics.last_summary_date = today

    def _send_daily_summary(self) -> None:
        """Compile and send daily performance summary."""
        latencies = list(self.metrics.latencies)
        avg_lat = sum(latencies) / max(len(latencies), 1)
        max_lat = max(latencies) if latencies else 0

        pnls = self.metrics.trade_pnls
        day_pnl = sum(pnls[-50:]) if pnls else 0  # Last 50 trades
        total_signals = sum(self.metrics.signal_counts.values())

        summary = (
            f"📈 <b>Daily Summary</b> ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})\n\n"
            f"Signals: {total_signals} total\n"
            f"  BUY:  {self.metrics.signal_counts.get('BUY', 0)}\n"
            f"  SELL: {self.metrics.signal_counts.get('SELL', 0)}\n"
            f"  HOLD: {self.metrics.signal_counts.get('HOLD', 0)}\n\n"
            f"Latency: avg={avg_lat:.1f}ms  max={max_lat:.1f}ms\n"
            f"Recent PnL: ${day_pnl:.2f}\n"
            f"Total trades logged: {len(pnls)}"
        )

        logger.info("[DAILY SUMMARY]\n%s", summary)
        self._send_telegram(summary)

    # ── Health check ──────────────────────────────────────────────────────

    def get_health(self) -> Dict[str, Any]:
        """Return current health metrics."""
        latencies = list(self.metrics.latencies)
        return {
            "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
            "max_latency_ms": max(latencies) if latencies else 0,
            "total_signals": sum(self.metrics.signal_counts.values()),
            "signal_counts": dict(self.metrics.signal_counts),
            "total_trade_pnls": len(self.metrics.trade_pnls),
            "net_pnl": sum(self.metrics.trade_pnls),
        }
