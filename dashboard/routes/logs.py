"""
Log routes — unified signal/trade log viewer with SSE streaming.
"""
from flask import Blueprint, Response, jsonify, render_template

from ..services.log_service import get_log_service

logs_bp = Blueprint("logs", __name__)


@logs_bp.route("/view")
def logs_view():
    """Render the live log viewer page."""
    return render_template("logs.html")


@logs_bp.route("/stream")
def logs_stream():
    """SSE endpoint streaming new signal/trade log entries."""
    svc = get_log_service()
    return Response(
        svc.stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@logs_bp.route("/recent")
def logs_recent():
    """Return the most recent log entries as JSON."""
    svc = get_log_service()
    entries = svc.get_recent(limit=200)
    return jsonify(entries)
