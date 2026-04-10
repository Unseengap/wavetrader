"""
OANDA v20 REST + Streaming API client for WaveTrader live trading.

Handles:
  - Account info & balance queries
  - Live price streaming (candle polling via REST — more reliable than WebSocket)
  - Order placement (market orders with SL/TP)
  - Position management (modify SL, close positions)
  - Rate limiting & retry logic

Requires environment variables:
  OANDA_API_KEY      — v20 API token (from OANDA hub)
  OANDA_ACCOUNT_ID   — v20 account ID (e.g. "101-001-12345678-001")
  OANDA_ENVIRONMENT  — "practice" or "live"
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_API_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}

_STREAM_URLS = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live":     "https://stream-fxtrade.oanda.com",
}

# OANDA instrument names (no slash)
_PAIR_MAP = {
    "GBP/JPY": "GBP_JPY",
    "EUR/JPY": "EUR_JPY",
    "GBP/USD": "GBP_USD",
    "USD/JPY": "USD_JPY",
}


@dataclass
class OANDAConfig:
    """OANDA connection configuration."""
    api_key: str = ""
    account_id: str = ""
    environment: str = "practice"   # "practice" or "live"
    max_retries: int = 3
    timeout: int = 30

    @property
    def api_url(self) -> str:
        return _API_URLS[self.environment]

    @property
    def stream_url(self) -> str:
        return _STREAM_URLS[self.environment]

    @classmethod
    def from_env(cls) -> "OANDAConfig":
        """Load configuration from environment variables (legacy single-account)."""
        return cls(
            api_key=os.environ.get("OANDA_API_KEY", ""),
            account_id=os.environ.get("OANDA_ACCOUNT_ID", ""),
            environment=os.environ.get("OANDA_ENVIRONMENT", "practice"),
        )

    @classmethod
    def demo_from_env(cls) -> "OANDAConfig":
        """Load demo (practice) account config from environment variables.

        Falls back to legacy OANDA_API_KEY / OANDA_ACCOUNT_ID if the
        demo-specific variables are not set.
        """
        return cls(
            api_key=(
                os.environ.get("OANDA_DEMO_API_KEY")
                or os.environ.get("OANDA_API_KEY", "")
            ),
            account_id=(
                os.environ.get("OANDA_DEMO_ACCOUNT_ID")
                or os.environ.get("OANDA_ACCOUNT_ID", "")
            ),
            environment="practice",
        )

    @classmethod
    def live_from_env(cls) -> Optional["OANDAConfig"]:
        """Load live account config from environment variables.

        Returns None if live credentials are not configured.
        """
        api_key = os.environ.get("OANDA_LIVE_API_KEY", "")
        account_id = os.environ.get("OANDA_LIVE_ACCOUNT_ID", "")
        if not api_key or not account_id:
            return None
        return cls(
            api_key=api_key,
            account_id=account_id,
            environment="live",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for API responses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AccountSummary:
    """Condensed account state."""
    balance: float
    unrealized_pnl: float
    nav: float
    margin_used: float
    margin_available: float
    open_trade_count: int
    currency: str


@dataclass
class Candle:
    """Single OHLCV candle from OANDA."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


@dataclass
class OrderResponse:
    """Result of an order submission."""
    order_id: str
    trade_id: Optional[str]
    instrument: str
    units: float
    price: float
    status: str  # "FILLED", "PENDING", "REJECTED"


@dataclass
class TradeInfo:
    """Open trade details."""
    trade_id: str
    instrument: str
    units: float
    price: float
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class OANDAClient:
    """
    OANDA v20 REST API client with retry logic and rate-limit awareness.

    Usage:
        client = OANDAClient(OANDAConfig.from_env())
        account = client.get_account_summary()
        candles = client.get_candles("GBP/JPY", "M15", count=100)
        order = client.place_market_order("GBP/JPY", units=10000, sl=189.50, tp=191.00)
    """

    def __init__(self, config: Optional[OANDAConfig] = None) -> None:
        self.config = config or OANDAConfig.from_env()
        if not self.config.api_key:
            raise ValueError(
                "OANDA_API_KEY not set. Get one from https://www.oanda.com/demo-account/tpa/personal_token"
            )
        if not self.config.account_id:
            raise ValueError(
                "OANDA_ACCOUNT_ID not set. Find it in your OANDA account settings."
            )

        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        """Build a requests session with retry logic."""
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        })
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session

    def _api(self, method: str, path: str, **kwargs: Any) -> Dict:
        """Make an API request with error handling."""
        url = f"{self.config.api_url}{path}"
        kwargs.setdefault("timeout", self.config.timeout)

        resp = self._session.request(method, url, **kwargs)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            logger.warning("Rate limited by OANDA, waiting %ds", wait)
            time.sleep(wait)
            resp = self._session.request(method, url, **kwargs)

        if resp.status_code >= 400:
            logger.error("OANDA API error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()

        return resp.json()

    @staticmethod
    def _to_instrument(pair: str) -> str:
        """Convert 'GBP/JPY' to 'GBP_JPY'."""
        if pair in _PAIR_MAP:
            return _PAIR_MAP[pair]
        return pair.replace("/", "_")

    # ── Account ───────────────────────────────────────────────────────────

    def get_account_summary(self) -> AccountSummary:
        """Fetch account balance, margin, and open trade count."""
        data = self._api("GET", f"/v3/accounts/{self.config.account_id}/summary")
        acct = data["account"]
        return AccountSummary(
            balance=float(acct["balance"]),
            unrealized_pnl=float(acct["unrealizedPL"]),
            nav=float(acct["NAV"]),
            margin_used=float(acct["marginUsed"]),
            margin_available=float(acct["marginAvailable"]),
            open_trade_count=int(acct["openTradeCount"]),
            currency=acct["currency"],
        )

    # ── Candles ───────────────────────────────────────────────────────────

    def get_candles(
        self,
        pair: str,
        granularity: str = "M15",
        count: int = 100,
        from_time: Optional[str] = None,
        price: str = "MBA",
    ) -> List[Candle]:
        """
        Fetch historical candles.

        Args:
            pair: e.g. "GBP/JPY"
            granularity: OANDA granularity — M1, M5, M15, M30, H1, H4, D
            count: Number of candles (max 5000)
            from_time: RFC3339 timestamp to fetch from (instead of count)
            price: "M" (mid), "B" (bid), "A" (ask), or combination like "MBA"

        Returns:
            List of Candle objects (only complete candles by default)
        """
        instrument = self._to_instrument(pair)
        params: Dict[str, Any] = {
            "granularity": granularity,
            "price": price,
        }
        if from_time:
            params["from"] = from_time
        else:
            params["count"] = str(min(count, 5000))

        data = self._api(
            "GET",
            f"/v3/instruments/{instrument}/candles",
            params=params,
        )

        candles = []
        for c in data.get("candles", []):
            # Use mid prices (average of bid/ask)
            mid = c.get("mid", c.get("bid", c.get("ask", {})))
            candles.append(Candle(
                timestamp=datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=int(c["volume"]),
                complete=c["complete"],
            ))
        return candles

    def get_latest_candles(
        self,
        pair: str,
        granularity: str = "M15",
        count: int = 100,
    ) -> List[Candle]:
        """Fetch the latest N complete candles (excludes the current forming candle)."""
        candles = self.get_candles(pair, granularity, count + 1)
        return [c for c in candles if c.complete][-count:]

    # ── Pricing ───────────────────────────────────────────────────────────

    def get_price(self, pair: str) -> Dict[str, float]:
        """Get current bid/ask price."""
        instrument = self._to_instrument(pair)
        data = self._api(
            "GET",
            f"/v3/accounts/{self.config.account_id}/pricing",
            params={"instruments": instrument},
        )
        price_data = data["prices"][0]
        return {
            "bid": float(price_data["bids"][0]["price"]),
            "ask": float(price_data["asks"][0]["price"]),
            "spread": float(price_data["asks"][0]["price"]) - float(price_data["bids"][0]["price"]),
            "time": price_data["time"],
        }

    # ── Orders ────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        pair: str,
        units: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResponse:
        """
        Place a market order.

        Args:
            pair: e.g. "GBP/JPY"
            units: Positive for BUY, negative for SELL (in base currency units)
            sl: Stop-loss price (absolute price, not pips)
            tp: Take-profit price (absolute price, not pips)

        Returns:
            OrderResponse with fill details
        """
        instrument = self._to_instrument(pair)
        order_body: Dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",   # Fill-or-kill
        }
        if sl is not None:
            order_body["stopLossOnFill"] = {
                "price": f"{sl:.5f}" if "JPY" not in pair else f"{sl:.3f}",
                "timeInForce": "GTC",
            }
        if tp is not None:
            order_body["takeProfitOnFill"] = {
                "price": f"{tp:.5f}" if "JPY" not in pair else f"{tp:.3f}",
                "timeInForce": "GTC",
            }

        data = self._api(
            "POST",
            f"/v3/accounts/{self.config.account_id}/orders",
            json={"order": order_body},
        )

        # Parse the fill
        if "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]
            trade_id = None
            if "tradeOpened" in fill:
                trade_id = fill["tradeOpened"]["tradeID"]
            return OrderResponse(
                order_id=fill["id"],
                trade_id=trade_id,
                instrument=instrument,
                units=float(fill["units"]),
                price=float(fill["price"]),
                status="FILLED",
            )
        elif "orderCancelTransaction" in data:
            cancel = data["orderCancelTransaction"]
            logger.error("Order rejected: %s", cancel.get("reason", "unknown"))
            return OrderResponse(
                order_id=cancel["id"],
                trade_id=None,
                instrument=instrument,
                units=float(units),
                price=0.0,
                status="REJECTED",
            )
        else:
            logger.error("Unexpected order response: %s", data)
            raise RuntimeError(f"Unexpected OANDA order response: {data}")

    # ── Trade management ──────────────────────────────────────────────────

    def get_pending_orders(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all pending orders, optionally filtered by pair."""
        data = self._api(
            "GET",
            f"/v3/accounts/{self.config.account_id}/pendingOrders",
        )
        orders = []
        for o in data.get("orders", []):
            instrument = o.get("instrument", "")
            if pair and self._to_instrument(pair) != instrument:
                continue
            orders.append({
                "order_id": o.get("id"),
                "type": o.get("type", ""),
                "instrument": instrument,
                "units": o.get("units", "0"),
                "price": o.get("price", "0"),
                "time_in_force": o.get("timeInForce", ""),
                "create_time": o.get("createTime", ""),
            })
        return orders

    def get_open_trades(self, pair: Optional[str] = None) -> List[TradeInfo]:
        """Fetch all open trades, optionally filtered by pair."""
        data = self._api(
            "GET",
            f"/v3/accounts/{self.config.account_id}/openTrades",
        )
        trades = []
        for t in data.get("trades", []):
            instrument = t["instrument"]
            if pair and self._to_instrument(pair) != instrument:
                continue
            trades.append(TradeInfo(
                trade_id=t["id"],
                instrument=instrument,
                units=float(t["currentUnits"]),
                price=float(t["price"]),
                unrealized_pnl=float(t["unrealizedPL"]),
                stop_loss=float(t["stopLossOrder"]["price"]) if "stopLossOrder" in t else None,
                take_profit=float(t["takeProfitOrder"]["price"]) if "takeProfitOrder" in t else None,
            ))
        return trades

    def modify_trade(
        self,
        trade_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Modify stop-loss and/or take-profit on an open trade."""
        body: Dict[str, Any] = {}
        if sl is not None:
            body["stopLoss"] = {"price": f"{sl:.3f}", "timeInForce": "GTC"}
        if tp is not None:
            body["takeProfit"] = {"price": f"{tp:.3f}", "timeInForce": "GTC"}

        if not body:
            return True

        try:
            self._api(
                "PUT",
                f"/v3/accounts/{self.config.account_id}/trades/{trade_id}/orders",
                json=body,
            )
            return True
        except requests.HTTPError as e:
            logger.error("Failed to modify trade %s: %s", trade_id, e)
            return False

    def close_trade(self, trade_id: str, units: Optional[int] = None) -> bool:
        """Close an open trade (partially or fully)."""
        body: Dict[str, Any] = {}
        if units is not None:
            body["units"] = str(units)

        try:
            self._api(
                "PUT",
                f"/v3/accounts/{self.config.account_id}/trades/{trade_id}/close",
                json=body if body else None,
            )
            logger.info("Closed trade %s", trade_id)
            return True
        except requests.HTTPError as e:
            logger.error("Failed to close trade %s: %s", trade_id, e)
            return False

    def close_all_trades(self, pair: Optional[str] = None) -> int:
        """Close all open trades (optionally filtered by pair). Returns count closed."""
        trades = self.get_open_trades(pair)
        closed = 0
        for t in trades:
            if self.close_trade(t.trade_id):
                closed += 1
        return closed

    def get_trade_history(
        self,
        pair: Optional[str] = None,
        state: str = "ALL",
        count: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch trade history (open + closed) from OANDA.

        Args:
            pair: Filter by instrument (e.g. "GBP/JPY"). None = all pairs.
            state: "ALL", "OPEN", or "CLOSED".
            count: Number of trades to return (max 500).

        Returns:
            List of trade dicts with keys: trade_id, instrument, units, price,
            realized_pl, unrealized_pl, open_time, close_time, state, sl, tp.
        """
        params: Dict[str, Any] = {
            "state": state,
            "count": str(min(count, 500)),
        }
        if pair:
            params["instrument"] = self._to_instrument(pair)

        data = self._api(
            "GET",
            f"/v3/accounts/{self.config.account_id}/trades",
            params=params,
        )

        results = []
        for t in data.get("trades", []):
            # Extract close reason from OANDA order states
            reason = ""
            if t.get("state") == "CLOSED":
                sl_order = t.get("stopLossOrder", {})
                tp_order = t.get("takeProfitOrder", {})
                tsl_order = t.get("trailingStopLossOrder", {})

                if sl_order.get("state") == "FILLED":
                    reason = "Stop Loss"
                elif tp_order.get("state") == "FILLED":
                    reason = "Take Profit"
                elif tsl_order.get("state") == "FILLED":
                    reason = "Trailing Stop"
                else:
                    reason = "Manual Close"

            results.append({
                "trade_id": t["id"],
                "instrument": t["instrument"],
                "units": float(t.get("initialUnits", t.get("currentUnits", 0))),
                "price": float(t["price"]),
                "close_price": float(t["averageClosePrice"]) if "averageClosePrice" in t else None,
                "realized_pl": float(t.get("realizedPL", 0)),
                "unrealized_pl": float(t.get("unrealizedPL", 0)),
                "open_time": t.get("openTime", ""),
                "close_time": t.get("closeTime", ""),
                "state": t.get("state", "OPEN"),
                "sl": float(t["stopLossOrder"]["price"]) if "stopLossOrder" in t else None,
                "tp": float(t["takeProfitOrder"]["price"]) if "takeProfitOrder" in t else None,
                "direction": "BUY" if float(t.get("initialUnits", t.get("currentUnits", 0))) > 0 else "SELL",
                "reason": reason,
            })
        return results

    # ── Utility ───────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if forex market is currently open (rough check: Mon-Fri)."""
        now = datetime.now(timezone.utc)
        # Forex closes Friday 22:00 UTC, opens Sunday 22:00 UTC
        if now.weekday() == 5:  # Saturday
            return False
        if now.weekday() == 6 and now.hour < 22:  # Sunday before open
            return False
        if now.weekday() == 4 and now.hour >= 22:  # Friday after close
            return False
        return True

    def ping(self) -> bool:
        """Test API connectivity."""
        try:
            self.get_account_summary()
            return True
        except Exception as e:
            logger.error("OANDA ping failed: %s", e)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Granularity mapping
# ─────────────────────────────────────────────────────────────────────────────

_TF_TO_GRANULARITY = {
    "1min":  "M1",
    "5min":  "M5",
    "15min": "M15",
    "30min": "M30",
    "1h":    "H1",
    "4h":    "H4",
    "1d":    "D",
}


def tf_to_granularity(tf: str) -> str:
    """Convert WaveTrader timeframe string to OANDA granularity."""
    g = _TF_TO_GRANULARITY.get(tf)
    if g is None:
        raise ValueError(f"Unknown timeframe '{tf}'. Use one of: {list(_TF_TO_GRANULARITY)}")
    return g
