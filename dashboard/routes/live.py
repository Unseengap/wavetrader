"""
Live streaming API routes — SSE stream, OANDA candles, account, model signals.
"""
from flask import Blueprint, Response, jsonify, request

from ..services.live_service import get_live_service
from ..services.model_registry import get_model_registry

live_bp = Blueprint("live", __name__)


def _model_id() -> str:
    """Extract the model ID from the query string, defaulting to the registry default."""
    return request.args.get("model", get_model_registry().default_id)


# ── Model Registry ────────────────────────────────────────────────────────────

@live_bp.route("/models", methods=["GET"])
def list_models():
    """Return all configured models (for the dropdown)."""
    reg = get_model_registry()
    return jsonify({
        "models": reg.to_list(),
        "default": reg.default_id,
    })


# ── SSE Stream ────────────────────────────────────────────────────────────────

@live_bp.route("/stream", methods=["GET"])
def sse_stream():
    """
    Server-Sent Events endpoint.
    Events: candle, price, signal, account, trades, status
    Query: ?model=<id>
    """
    svc = get_live_service(_model_id())
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
    model = data.get("model", get_model_registry().default_id)

    svc = get_live_service(model)
    result = svc.start(pair, tf)
    return jsonify(result)


@live_bp.route("/stop", methods=["POST"])
def stop_stream():
    """Stop live streaming."""
    data = request.get_json(force=True, silent=True) or {}
    model = data.get("model", get_model_registry().default_id)
    svc = get_live_service(model)
    result = svc.stop()
    return jsonify(result)


@live_bp.route("/status", methods=["GET"])
def stream_status():
    """Return current streaming status."""
    svc = get_live_service(_model_id())
    return jsonify(svc.status)


# ── OANDA Data (one-shot, not streamed) ──────────────────────────────────────

@live_bp.route("/candles", methods=["GET"])
def get_live_candles():
    """Fetch candles directly from OANDA (for initial chart load in live mode)."""
    pair = request.args.get("pair", "GBP/JPY")
    tf = request.args.get("tf", "15min")
    count = int(request.args.get("count", 300))

    svc = get_live_service(_model_id())
    candles = svc.get_live_candles(pair, tf, count)
    return jsonify({"pair": pair, "timeframe": tf, "candles": candles, "source": "oanda"})


@live_bp.route("/account", methods=["GET"])
def get_account():
    """Fetch current OANDA account state."""
    svc = get_live_service(_model_id())
    return jsonify(svc.get_account())


@live_bp.route("/trades", methods=["GET"])
def get_trades():
    """Fetch open trades from OANDA."""
    svc = get_live_service(_model_id())
    return jsonify({"trades": svc.get_open_trades()})


@live_bp.route("/orders", methods=["GET"])
def get_orders():
    """Fetch pending orders from OANDA."""
    svc = get_live_service(_model_id())
    return jsonify({"orders": svc.get_pending_orders()})


# ── Auto-Trade Status (always on) ─────────────────────────────────────────

@live_bp.route("/auto-trade", methods=["GET"])
def get_auto_trade():
    """Return current auto-trade status (always enabled)."""
    svc = get_live_service(_model_id())
    return jsonify(svc.auto_trade_status)


@live_bp.route("/auto-trade", methods=["POST"])
def set_auto_trade():
    """Auto-trade is always on — returns current status."""
    svc = get_live_service(_model_id())
    return jsonify(svc.auto_trade_status)


# ── Live Config ───────────────────────────────────────────────────────────

@live_bp.route("/config", methods=["GET"])
def get_live_config():
    """Return the current live trading configuration."""
    svc = get_live_service(_model_id())
    return jsonify(svc.live_config)


@live_bp.route("/config", methods=["POST"])
def update_live_config():
    """Update live trading configuration at runtime."""
    svc = get_live_service(_model_id())
    cfg = request.get_json(force=True)
    updated = svc.update_config(cfg)
    return jsonify(updated)


# ── Trade History ─────────────────────────────────────────────────────────

@live_bp.route("/trade-history", methods=["GET"])
def get_trade_history():
    """Fetch trade history from both OANDA demo and live accounts."""
    pair = request.args.get("pair")
    count = int(request.args.get("count", 50))
    svc = get_live_service(_model_id())
    trades = svc.get_trade_history(pair=pair, count=count)
    return jsonify({"trades": trades})


# ── LLM Arbiter ───────────────────────────────────────────────────────────

@live_bp.route("/arbiter/status", methods=["GET"])
def arbiter_status():
    """Return LLM arbiter configuration and statistics."""
    svc = get_live_service(_model_id())
    return jsonify(svc.arbiter_status)


@live_bp.route("/arbiter/config", methods=["POST"])
def update_arbiter_config():
    """Update LLM arbiter configuration at runtime."""
    svc = get_live_service(_model_id())
    cfg = request.get_json(force=True)
    updated = svc.update_arbiter_config(cfg)
    return jsonify(updated)


@live_bp.route("/arbiter/decisions", methods=["GET"])
def get_arbiter_decisions():
    """Return recent LLM arbiter decisions."""
    count = int(request.args.get("count", 50))
    svc = get_live_service(_model_id())
    decisions = svc.get_arbiter_decisions(count)
    return jsonify({"decisions": decisions})


@live_bp.route("/arbiter/stats", methods=["GET"])
def get_arbiter_stats():
    """Return aggregate arbiter statistics (for the trial tracker)."""
    from wavetrader.llm_logger import get_decision_log
    log = get_decision_log()
    return jsonify(log.get_stats())


@live_bp.route("/arbiter/calendar", methods=["GET"])
def get_calendar_events():
    """Return upcoming forex calendar events."""
    pair = request.args.get("pair", "GBP/JPY")
    hours = int(request.args.get("hours", 24))
    from wavetrader.calendar import get_calendar
    cal = get_calendar()
    events = cal.get_upcoming(pair, hours_ahead=hours)
    return jsonify({"events": [e.to_dict() for e in events]})


@live_bp.route("/arbiter/inspect", methods=["POST"])
def run_inspection():
    """Trigger a manual LLM market inspection across all models."""
    svc = get_live_service(_model_id())
    result = svc.run_inspection()
    return jsonify(result)
