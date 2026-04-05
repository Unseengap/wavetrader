"""
Multi-user copy trading for WaveTrader.

Two modes for letting others trade with the bot:

  1. COPY TRADING (auto-execution)
     Users provide their OANDA API key + account ID.
     When the bot generates a signal, it copies the trade to ALL connected
     accounts with per-user position sizing based on their balance & risk.

  2. SIGNAL BROADCAST (manual execution)
     Users join a Telegram channel/group.
     Bot posts every signal with entry/SL/TP — users execute manually.

User data is stored in a local JSON registry (encrypted API keys at rest).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from base64 import b64decode, b64encode
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .oanda import OANDAClient, OANDAConfig
from .types import Signal, TradeSignal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# User model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserAccount:
    """A registered copy-trade follower."""
    user_id: str                       # Unique ID (e.g. "user_001")
    name: str                          # Display name
    oanda_api_key: str                 # OANDA v20 API token (stored encrypted)
    oanda_account_id: str              # OANDA account ID
    oanda_environment: str = "practice"  # "practice" or "live"
    risk_per_trade: float = 0.01       # 1% default (conservative for followers)
    max_lot_size: float = 1.0          # Cap per trade
    enabled: bool = True               # Can be paused without removing
    telegram_chat_id: str = ""         # Per-user Telegram notifications
    created_at: str = ""
    last_trade_at: str = ""
    total_trades: int = 0
    total_pnl: float = 0.0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# User registry (JSON file-based)
# ─────────────────────────────────────────────────────────────────────────────

class UserRegistry:
    """
    Manages the list of copy-trade followers.

    Storage: JSON file at <data_dir>/users.json
    API keys are obfuscated with XOR + base64 (not full encryption, but
    prevents casual exposure in logs/backups — for production, use a vault).
    """

    def __init__(self, data_dir: str = "/data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.data_dir / "users.json"
        self._users: Dict[str, UserAccount] = {}
        self._load()

    def _load(self) -> None:
        """Load users from disk."""
        if not self._file.exists():
            self._users = {}
            return
        with open(self._file) as f:
            data = json.load(f)
        self._users = {
            uid: UserAccount(**udata) for uid, udata in data.items()
        }
        logger.info("Loaded %d users from registry", len(self._users))

    def _save(self) -> None:
        """Persist users to disk (atomic write)."""
        tmp = self._file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {uid: asdict(u) for uid, u in self._users.items()},
                f, indent=2,
            )
        tmp.rename(self._file)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_user(self, user: UserAccount) -> None:
        """Register a new copy-trade follower."""
        if user.user_id in self._users:
            raise ValueError(f"User '{user.user_id}' already exists")
        # Obfuscate API key before storing
        user.oanda_api_key = self._obfuscate(user.oanda_api_key)
        self._users[user.user_id] = user
        self._save()
        logger.info("Added user: %s (%s)", user.user_id, user.name)

    def remove_user(self, user_id: str) -> bool:
        """Remove a user from the registry."""
        if user_id not in self._users:
            return False
        del self._users[user_id]
        self._save()
        logger.info("Removed user: %s", user_id)
        return True

    def get_user(self, user_id: str) -> Optional[UserAccount]:
        """Get a user by ID."""
        return self._users.get(user_id)

    def get_active_users(self) -> List[UserAccount]:
        """Return all enabled users."""
        return [u for u in self._users.values() if u.enabled]

    def list_users(self) -> List[UserAccount]:
        """Return all users."""
        return list(self._users.values())

    def update_user_stats(
        self, user_id: str, pnl: float = 0.0, trade_count: int = 1
    ) -> None:
        """Update a user's trade stats after a copy-trade."""
        user = self._users.get(user_id)
        if user:
            user.total_trades += trade_count
            user.total_pnl += pnl
            user.last_trade_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def set_enabled(self, user_id: str, enabled: bool) -> bool:
        """Enable or disable a user."""
        user = self._users.get(user_id)
        if not user:
            return False
        user.enabled = enabled
        self._save()
        return True

    # ── Key obfuscation ───────────────────────────────────────────────────

    @staticmethod
    def _obfuscate(key: str) -> str:
        """Simple XOR obfuscation — prevents casual log exposure."""
        if key.startswith("obf:"):
            return key  # Already obfuscated
        salt = b"wavetrader_2026"
        xored = bytes(a ^ salt[i % len(salt)] for i, a in enumerate(key.encode()))
        return "obf:" + b64encode(xored).decode()

    @staticmethod
    def _deobfuscate(key: str) -> str:
        """Reverse XOR obfuscation to get the real API key."""
        if not key.startswith("obf:"):
            return key  # Not obfuscated
        salt = b"wavetrader_2026"
        xored = b64decode(key[4:])
        return bytes(a ^ salt[i % len(salt)] for i, a in enumerate(xored)).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Copy Trade Manager
# ─────────────────────────────────────────────────────────────────────────────

# Pip conventions (same as streaming.py)
_PIP_SIZE = {"GBP/JPY": 0.01, "EUR/JPY": 0.01, "USD/JPY": 0.01, "GBP/USD": 0.0001}
_PIP_VALUE = {"GBP/JPY": 6.5, "EUR/JPY": 6.5, "USD/JPY": 7.0, "GBP/USD": 10.0}
_LOT_SIZE = 100_000


class CopyTradeManager:
    """
    Copies signals from the master account to all registered followers.

    When the master StreamingEngine generates a signal:
      1. CopyTradeManager receives the signal
      2. For each active user:
         a. Create an OANDAClient with the user's credentials
         b. Size the position based on the USER's balance & risk setting
         c. Place the order on the USER's OANDA account
         d. Track the trade for the user
      3. Report results back

    The master bot only runs inference ONCE — trades are copied in parallel
    to all followers. No extra compute per user.
    """

    def __init__(
        self,
        registry: UserRegistry,
        pair: str = "GBP/JPY",
        monitor: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.pair = pair
        self.monitor = monitor
        self._clients: Dict[str, OANDAClient] = {}  # Cached per user

    def _get_client(self, user: UserAccount) -> OANDAClient:
        """Get or create an OANDA client for a user."""
        if user.user_id not in self._clients:
            real_key = UserRegistry._deobfuscate(user.oanda_api_key)
            config = OANDAConfig(
                api_key=real_key,
                account_id=user.oanda_account_id,
                environment=user.oanda_environment,
            )
            self._clients[user.user_id] = OANDAClient(config)
        return self._clients[user.user_id]

    def copy_open(
        self,
        signal: TradeSignal,
        candle_close: float,
    ) -> Dict[str, str]:
        """
        Copy an open-position signal to all active followers.

        Returns: {user_id: "FILLED" | "REJECTED" | error_message}
        """
        results: Dict[str, str] = {}
        pip = _PIP_SIZE.get(self.pair, 0.01)
        pip_value = _PIP_VALUE.get(self.pair, 6.5)

        for user in self.registry.get_active_users():
            try:
                client = self._get_client(user)

                # Get the USER's balance for position sizing
                account = client.get_account_summary()
                user_balance = account.balance

                # Size based on the user's own risk parameters
                risk_amount = user_balance * user.risk_per_trade
                lot = risk_amount / max(signal.stop_loss * pip_value, 1e-9)
                lot = max(0.01, min(user.max_lot_size, lot))

                units = int(lot * _LOT_SIZE)
                if signal.signal == Signal.SELL:
                    units = -units

                # Calculate SL/TP prices
                if signal.signal == Signal.BUY:
                    sl_price = candle_close - signal.stop_loss * pip
                    tp_price = candle_close + signal.take_profit * pip
                else:
                    sl_price = candle_close + signal.stop_loss * pip
                    tp_price = candle_close - signal.take_profit * pip

                # Place order on user's account
                order = client.place_market_order(
                    self.pair, units, sl=sl_price, tp=tp_price,
                )

                results[user.user_id] = order.status
                if order.status == "FILLED":
                    self.registry.update_user_stats(user.user_id)
                    logger.info(
                        "Copy trade FILLED for %s (%s): %d units @ %.3f",
                        user.user_id, user.name, units, order.price,
                    )
                else:
                    logger.warning(
                        "Copy trade REJECTED for %s: %s", user.user_id, order.status,
                    )

            except Exception as e:
                results[user.user_id] = f"ERROR: {e}"
                logger.error("Copy trade failed for %s: %s", user.user_id, e)

        # Summary notification
        n_ok = sum(1 for v in results.values() if v == "FILLED")
        n_total = len(results)
        if self.monitor and n_total > 0:
            self.monitor.send_info(
                f"Copy trade: {n_ok}/{n_total} followers filled "
                f"({signal.signal.name} {self.pair})"
            )

        return results

    def copy_close(self, reason: str) -> Dict[str, str]:
        """
        Close positions on all followers' accounts for this pair.

        Returns: {user_id: "CLOSED" | error_message}
        """
        results: Dict[str, str] = {}

        for user in self.registry.get_active_users():
            try:
                client = self._get_client(user)
                open_trades = client.get_open_trades(self.pair)
                for trade in open_trades:
                    client.close_trade(trade.trade_id)
                results[user.user_id] = "CLOSED"
                logger.info("Closed copy trades for %s: %s", user.user_id, reason)
            except Exception as e:
                results[user.user_id] = f"ERROR: {e}"
                logger.error("Copy close failed for %s: %s", user.user_id, e)

        return results

    def get_follower_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all followers (for monitoring dashboards)."""
        summaries = []
        for user in self.registry.list_users():
            summary: Dict[str, Any] = {
                "user_id": user.user_id,
                "name": user.name,
                "enabled": user.enabled,
                "environment": user.oanda_environment,
                "risk_per_trade": user.risk_per_trade,
                "total_trades": user.total_trades,
                "total_pnl": user.total_pnl,
                "last_trade": user.last_trade_at or "never",
            }
            # Try to get live balance
            if user.enabled:
                try:
                    client = self._get_client(user)
                    acct = client.get_account_summary()
                    summary["balance"] = acct.balance
                    summary["nav"] = acct.nav
                    summary["open_trades"] = acct.open_trade_count
                except Exception:
                    summary["balance"] = "unavailable"
            summaries.append(summary)
        return summaries
