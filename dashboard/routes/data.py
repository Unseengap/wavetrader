"""
Data API endpoints — candles, pairs, timeframes for the chart.
"""
from flask import Blueprint, jsonify, request

from ..services.backtest_service import (
    AVAILABLE_PAIRS,
    AVAILABLE_TIMEFRAMES,
    load_candles,
)

data_bp = Blueprint("data", __name__)


@data_bp.route("/candles", methods=["GET"])
def get_candles():
    """Return OHLCV candles for TradingView chart."""
    pair = request.args.get("pair", "GBP/JPY")
    tf = request.args.get("tf", "15min")
    start = request.args.get("start")
    end = request.args.get("end")
    limit = int(request.args.get("limit", 5000))

    candles = load_candles(pair=pair, timeframe=tf, start=start, end=end, limit=limit)
    return jsonify({"pair": pair, "timeframe": tf, "candles": candles})


@data_bp.route("/pairs", methods=["GET"])
def get_pairs():
    return jsonify({"pairs": AVAILABLE_PAIRS})


@data_bp.route("/timeframes", methods=["GET"])
def get_timeframes():
    return jsonify({"timeframes": AVAILABLE_TIMEFRAMES})
