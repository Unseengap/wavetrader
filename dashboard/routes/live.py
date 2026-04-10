"""
Live streaming API routes — SSE stream, OANDA candles, account, model signals.
"""
from flask import Blueprint, Response, jsonify, request

from ..services.live_service import get_live_service

live_bp = Blueprint("live", __name__)


# ── SSE Stream ────────────────────────────────────────────────────────────────

@live_bp.route("/stream", methods=["GET"])
def sse_stream():
    """
    Server-Sent Events endpoint.
    Events: candle, price, signal, account, trades, status
    """
    svc = get_live_service()
    return Response(
        svc.sse_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Start / Stop ──────────────────────────────────────────────────────────────

@live_bp.route("/start", methods=["POST"])
def start_stream():
    """Start live streaming for a pair/timeframe."""
    data = request.get_json(force=True, silent=True) or {}
    pair = data.get("pair", "GBP/JPY")
    tf = data.get("timeframe", "15min")

    svc = get_live_service()
    result = svc.start(pair, tf)
    return jsonify(result)


@live_bp.route("/stop", methods=["POST"])
def stop_stream():
    """Stop live streaming."""
    svc = get_live_service()
    result = svc.stop()
    return jsonify(result)


@live_bp.route("/status", methods=["GET"])
def stream_status():
    """Return current streaming status."""
    svc = get_live_service()
    return jsonify(svc.status)


# ── OANDA Data (one-shot, not streamed) ──────────────────────────────────────

@live_bp.route("/candles", methods=["GET"])
def get_live_candles():
    """Fetch candles directly from OANDA (for initial chart load in live mode)."""
    pair = request.args.get("pair", "GBP/JPY")
    tf = request.args.get("tf", "15min")
    count = int(request.args.get("count", 300))

    svc = get_live_service()
    candles = svc.get_live_candles(pair, tf, count)
    return jsonify({"pair": pair, "timeframe": tf, "candles": candles, "source": "oanda"})


@live_bp.route("/account", methods=["GET"])
def get_account():
    """Fetch current OANDA account state."""
    svc = get_live_service()
    return jsonify(svc.get_account())


@live_bp.route("/trades", methods=["GET"])
def get_trades():
    """Fetch open trades from OANDA."""
    svc = get_live_service()
    return jsonify({"trades": svc.get_open_trades()})


@live_bp.route("/orders", methods=["GET"])
def get_orders():
    """Fetch pending orders from OANDA."""
    svc = get_live_service()
    return jsonify({"orders": svc.get_pending_orders()})


# ── Auto-Trade Status (always on) ─────────────────────────────────────────

@live_bp.route("/auto-trade", methods=["GET"])
def get_auto_trade():
    """Return current auto-trade status (always enabled)."""
    svc = get_live_service()
    return jsonify(svc.auto_trade_status)


@live_bp.route("/auto-trade", methods=["POST"])
def set_auto_trade():
    """Auto-trade is always on — returns current status."""
    svc = get_live_service()
    return jsonify(svc.auto_trade_status)


# ── Trade History ─────────────────────────────────────────────────────────

@live_bp.route("/trade-history", methods=["GET"])
def get_trade_history():
    """Fetch trade history from both OANDA demo and live accounts."""
    pair = request.args.get("pair")
    count = int(request.args.get("count", 50))
    svc = get_live_service()
    trades = svc.get_trade_history(pair=pair, count=count)
    return jsonify({"trades": trades})
