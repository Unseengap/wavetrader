"""
Backtest API endpoints.
"""
from flask import Blueprint, jsonify, request

from ..services.backtest_service import (
    NOTEBOOK_DEFAULTS,
    AVAILABLE_PAIRS,
    AVAILABLE_TIMEFRAMES,
    load_cached_results,
    run_backtest_from_config,
)

backtest_bp = Blueprint("backtest", __name__)


@backtest_bp.route("/defaults", methods=["GET"])
def get_defaults():
    """Return default config (mirrors notebook Cell 10)."""
    return jsonify({
        "config": NOTEBOOK_DEFAULTS,
        "pairs": AVAILABLE_PAIRS,
        "timeframes": AVAILABLE_TIMEFRAMES,
    })


@backtest_bp.route("/run", methods=["POST"])
def run_backtest_endpoint():
    """Run a backtest with the provided config overrides."""
    user_config = request.get_json(force=True, silent=True) or {}
    try:
        results = run_backtest_from_config(user_config)
        if "error" in results:
            return jsonify(results), 400
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@backtest_bp.route("/cached", methods=["GET"])
def get_cached_results():
    """Return cached results from backtest_results/ CSVs if available."""
    results = load_cached_results()
    if results is None:
        return jsonify({"error": "No cached results found"}), 404
    return jsonify(results)
