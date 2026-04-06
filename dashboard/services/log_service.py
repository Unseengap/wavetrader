"""
Unified log service — merges Docker container logs and OANDA transactions
into a single chronological stream filtered to signals and trades only.
"""
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Generator

import requests

import logging

logger = logging.getLogger(__name__)

# Keywords that identify signal/trade log lines from containers
_SIGNAL_TRADE_KEYWORDS = (
    "Evaluating signal",
    "Skipping HOLD",
    "Skipping BUY",
    "Skipping SELL",
    "Already BUY",
    "Already SELL",
    "Opening ",
    "Order filled",
    "Order rejected",
    "Closing positions",
    "trade closed",
    "Trailing SL updated",
    "Trailing stop hit",
    "Signal reversal",
    "Signal received",
    "TRADE_DOESNT_EXIST",
    "FIFO_VIOLATION",
    "INSUFFICIENT_MARGIN",
    "Pyramiding",
    "No margin available",
    "Failed to place",
    "Failed to close",
    "Already have open",
    "Placing order",
    "OANDA [",
    "OANDA API error",
)

# OANDA transaction types we care about
_TRADE_TXN_TYPES = {
    "MARKET_ORDER",
    "ORDER_FILL",
    "ORDER_CANCEL",
    "MARKET_ORDER_REJECT",
    "STOP_LOSS_ORDER",
    "TAKE_PROFIT_ORDER",
    "TRAILING_STOP_LOSS_ORDER",
}


class UnifiedLogService:
    """Tails Docker logs + polls OANDA transactions, merged chronologically."""

    def __init__(self):
        self._buffer: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._last_oanda_txn_id: str = ""

        # OANDA config from env
        self._api_key = (
            os.environ.get("OANDA_DEMO_API_KEY")
            or os.environ.get("OANDA_API_KEY", "")
        )
        self._account_id = (
            os.environ.get("OANDA_DEMO_ACCOUNT_ID")
            or os.environ.get("OANDA_ACCOUNT_ID", "")
        )
        self._oanda_url = "https://api-fxpractice.oanda.com"

    def start(self):
        if self._running:
            return
        self._running = True

        # Tail wavetrader container logs
        t1 = threading.Thread(
            target=self._tail_container, args=("wavetrader",), daemon=True,
        )
        t1.start()
        self._threads.append(t1)

        # Tail dashboard container logs
        t2 = threading.Thread(
            target=self._tail_container, args=("dashboard",), daemon=True,
        )
        t2.start()
        self._threads.append(t2)

        # Poll OANDA transactions
        t3 = threading.Thread(target=self._poll_oanda, daemon=True)
        t3.start()
        self._threads.append(t3)

        logger.info("UnifiedLogService started")

    def stop(self):
        self._running = False
        self._threads.clear()

    def get_recent(self, limit: int = 200) -> list[dict]:
        """Return the most recent log entries."""
        with self._lock:
            entries = list(self._buffer)
        return entries[-limit:]

    def stream(self) -> Generator[str, None, None]:
        """SSE stream of new log entries."""
        seen = len(self._buffer)
        yield f"data: {json.dumps({'type': 'connected', 'count': seen})}\n\n"

        while self._running:
            with self._lock:
                current = list(self._buffer)
            if len(current) > seen:
                for entry in current[seen:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                seen = len(current)
            time.sleep(1)

    # ── Docker log tailing ───────────────────────────────────────────────

    def _tail_container(self, service_name: str):
        """Tail a Docker container's logs via the Docker Engine API over the socket."""
        container = f"wavetrader-{'live' if service_name == 'wavetrader' else service_name}"
        sock_path = "/var/run/docker.sock"

        if not os.path.exists(sock_path):
            logger.warning("Docker socket not found at %s — skipping %s logs", sock_path, service_name)
            return

        import http.client
        import socket as socket_mod
        import struct

        class UnixHTTPConnection(http.client.HTTPConnection):
            def __init__(self):
                super().__init__("localhost")
            def connect(self):
                self.sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
                self.sock.connect(sock_path)
                self.sock.settimeout(30)

        try:
            conn = UnixHTTPConnection()
            conn.request(
                "GET",
                f"/containers/{container}/logs?follow=1&stdout=1&stderr=1&tail=100",
            )
            resp = conn.getresponse()
            if resp.status != 200:
                logger.error("Docker API error for %s: %d", container, resp.status)
                return
        except Exception as e:
            logger.error("Failed to connect to Docker API for %s: %s", container, e)
            return

        logger.info("Tailing container logs: %s", container)

        # The response uses chunked transfer encoding handled by http.client.
        # Docker multiplexed stream: each frame = [stream_type:1][0:3][size:4 big-endian][payload:size]
        # Use resp.read1() to get data as it arrives without blocking for a full amt.
        buf = b""
        while self._running:
            try:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                buf += chunk

                # Process complete frames
                while len(buf) >= 8:
                    payload_size = struct.unpack(">I", buf[4:8])[0]
                    total_frame = 8 + payload_size
                    if len(buf) < total_frame:
                        break  # wait for more data
                    payload = buf[8:total_frame]
                    buf = buf[total_frame:]

                    for raw_line in payload.decode("utf-8", errors="replace").splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if not any(kw in line for kw in _SIGNAL_TRADE_KEYWORDS):
                            continue

                        ts = self._extract_timestamp(line)
                        # Strip the leading "YYYY-MM-DD HH:MM:SS [LEVEL] module: " prefix for cleaner display
                        import re as _re
                        clean = _re.sub(
                            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(?:INFO|WARNING|ERROR)\] [\w.]+: ",
                            "", line,
                        )
                        entry = {
                            "source": f"container:{service_name}",
                            "ts": ts,
                            "message": clean or line,
                            "level": self._extract_level(line),
                        }
                        self._append(entry)
            except Exception:
                break

        try:
            conn.close()
        except Exception:
            pass

    # ── OANDA transaction polling ────────────────────────────────────────

    def _poll_oanda(self):
        """Poll OANDA transaction stream every 5 seconds for new trades."""
        if not self._api_key or not self._account_id:
            logger.warning("OANDA credentials not set — skipping transaction polling")
            return

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Seed the last transaction ID
        try:
            r = requests.get(
                f"{self._oanda_url}/v3/accounts/{self._account_id}/transactions",
                headers=headers,
                params={
                    "from": (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "pageSize": 200,
                },
                timeout=10,
            )
            if r.ok:
                data = r.json()
                self._last_oanda_txn_id = data.get("lastTransactionID", "")
                # Load recent transactions into buffer
                for page_url in data.get("pages", []):
                    try:
                        pr = requests.get(page_url, headers=headers, timeout=10)
                        if pr.ok:
                            for txn in pr.json().get("transactions", []):
                                self._process_oanda_txn(txn)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Failed to seed OANDA transactions: %s", e)

        # Poll loop
        while self._running:
            time.sleep(5)
            if not self._last_oanda_txn_id:
                continue
            try:
                r = requests.get(
                    f"{self._oanda_url}/v3/accounts/{self._account_id}/transactions/sinceid",
                    headers=headers,
                    params={"id": self._last_oanda_txn_id},
                    timeout=10,
                )
                if r.ok:
                    data = r.json()
                    for txn in data.get("transactions", []):
                        self._process_oanda_txn(txn)
                    new_last = data.get("lastTransactionID", self._last_oanda_txn_id)
                    if new_last != self._last_oanda_txn_id:
                        self._last_oanda_txn_id = new_last
            except Exception as e:
                logger.debug("OANDA poll error: %s", e)

    def _process_oanda_txn(self, txn: dict):
        """Convert an OANDA transaction to a log entry."""
        txn_type = txn.get("type", "")
        if txn_type not in _TRADE_TXN_TYPES:
            return

        txn_id = txn.get("id", "")
        ts = txn.get("time", "")[:19].replace("T", " ")
        instrument = txn.get("instrument", "")
        units = txn.get("units", "")
        price = txn.get("price", "")
        reason = txn.get("reason", "")
        pl = txn.get("pl", "")
        reject = txn.get("rejectReason", "")

        trade_opened = txn.get("tradeOpened", {}).get("tradeID", "")
        trades_closed = txn.get("tradesClosed", [])
        trade_closed = trades_closed[0].get("tradeID", "") if trades_closed else ""
        trade_id = trade_opened or trade_closed

        # Build human-readable message
        parts = [f"[{txn_id}]", txn_type]
        if instrument:
            parts.append(instrument)
        if units:
            parts.append(f"units={units}")
        if price:
            parts.append(f"@{price}")
        if pl and pl != "0.0000":
            parts.append(f"PnL={pl}")
        if trade_id:
            parts.append(f"trade={trade_id}")
        if reason:
            parts.append(f"({reason})")
        if reject:
            parts.append(f"REJECTED:{reject}")

        level = "error" if reject or "REJECT" in txn_type else "info"
        if "CANCEL" in txn_type:
            level = "warn"

        entry = {
            "source": "oanda",
            "ts": ts,
            "message": " ".join(parts),
            "level": level,
            "txn_type": txn_type,
        }
        self._append(entry)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _append(self, entry: dict):
        with self._lock:
            self._buffer.append(entry)

    @staticmethod
    def _extract_timestamp(line: str) -> str:
        """Try to pull a timestamp from a log line."""
        # Docker format: "2026-04-06T17:59:41.123Z ..." or
        # App format: "... 2026-04-06 17:59:41 [INFO] ..."
        for fmt_len in (19, 23, 26):
            try:
                candidate = line[:fmt_len].replace("T", " ").rstrip("Z")
                datetime.strptime(candidate[:19], "%Y-%m-%d %H:%M:%S")
                return candidate[:19]
            except (ValueError, IndexError):
                pass
        # Try to find timestamp in the middle of the line
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", line)
        if m:
            return m.group(1).replace("T", " ")
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _extract_level(line: str) -> str:
        if "[ERROR]" in line or "ERROR" in line.upper()[:80]:
            return "error"
        if "[WARN" in line or "WARNING" in line.upper()[:80]:
            return "warn"
        return "info"


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: UnifiedLogService | None = None
_instance_lock = threading.Lock()


def get_log_service() -> UnifiedLogService:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = UnifiedLogService()
                _instance.start()
    return _instance
