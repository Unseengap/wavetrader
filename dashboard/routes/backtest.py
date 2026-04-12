"""
Backtest API endpoints.
"""
import logging
import traceback

from flask import Blueprint, jsonify, request

from ..services.backtest_service import (
    NOTEBOOK_DEFAULTS,
    AVAILABLE_PAIRS,
    AVAILABLE_TIMEFRAMES,
    load_cached_results,
    run_backtest_from_config,
    save_backtest_results,
    list_saved_backtests,
    load_saved_backtest,
)

logger = logging.getLogger(__name__)

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
        logger.error("Backtest run failed:\n%s", traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@backtest_bp.route("/cached", methods=["GET"])
def get_cached_results():
    """Return cached results from backtest_results/ CSVs if available."""
    model_id = request.args.get("model", "mtf")
    results = load_cached_results(model_id=model_id)
    if results is None:
        return jsonify({"error": "No cached results found"}), 404
    return jsonify(results)


@backtest_bp.route("/save", methods=["POST"])
def save_backtest():
    """Save a backtest result for later recall."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        entry = save_backtest_results(data)
        return jsonify(entry)
    except Exception as e:
        logger.error("Save backtest failed:\n%s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@backtest_bp.route("/list", methods=["GET"])
def list_backtests():
    """List all saved backtests."""
    return jsonify(list_saved_backtests())


@backtest_bp.route("/load/<run_id>", methods=["GET"])
def load_backtest(run_id):
    """Load a saved backtest by run_id."""
    result = load_saved_backtest(run_id)
    if result is None:
        return jsonify({"error": "Backtest not found"}), 404
    return jsonify(result)
