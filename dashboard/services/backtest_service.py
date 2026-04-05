"""
Backtest service — wraps the wavetrader backtest engine for the dashboard.

Provides JSON-serializable results that can be consumed by the frontend
charts (TradingView LC5 + Plotly.js).  Mirrors the exact defaults from
Validation_And_Backtesting.ipynb Cell 10.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Lazy imports of wavetrader internals ─────────────────────────────────────
# Imported at function call time so the module can be loaded even if torch
# isn't installed (e.g. for frontend-only development).

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_dir(name: str) -> Path:
    """Return the correct path for a data directory, works in Docker and locally."""
    # Docker mounts are at root level: /backtest_results, /data, etc.
    docker_path = Path("/") / name
    local_path = _PROJECT_ROOT / name
    if docker_path.exists() and docker_path.is_dir():
        return docker_path
    return local_path


_RESULTS_DIR = _resolve_dir("backtest_results")


# ─────────────────────────────────────────────────────────────────────────────
# Default config (matches notebook Cell 10 exactly)
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOK_DEFAULTS: Dict[str, Any] = {
    # BacktestConfig
    "initial_balance": 25_000.0,
    "risk_per_trade": 0.01,
    "leverage": 30.0,
    "spread_pips": 3.0,
    "commission_per_lot": 7.0,
    "pip_value": 10.0,
    "min_confidence": 0.55,
    "atr_halt_multiplier": 3.0,
    "drawdown_reduce_threshold": 0.10,
    # Pair / timeframe
    "pair": "GBP/JPY",
    "entry_timeframe": "15min",
    # Friction simulation
    "friction": {
        "slippage_min": 0.5,
        "slippage_max": 3.0,
        "spread_offhours_extra": 2.5,
        "news_spike_prob": 0.05,
        "news_spike_extra": 5.0,
        "lot_cap": 2.0,
        "latency_miss_rate": 0.03,
    },
}

AVAILABLE_PAIRS = ["GBP/JPY", "EUR/JPY", "GBP/USD", "USD/JPY"]
AVAILABLE_TIMEFRAMES = ["15min", "1h", "4h", "1d"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(dt: Any) -> Optional[str]:
    """Convert datetime to ISO string, handling None and pandas Timestamps."""
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        return dt.isoformat()
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _trade_to_dict(t: Any) -> Dict[str, Any]:
    """Serialize a Trade dataclass to a JSON-friendly dict."""
    return {
        "entry_time": _ts(t.entry_time),
        "entry_price": round(float(t.entry_price), 5),
        "direction": t.direction.name,  # "BUY" / "SELL"
        "stop_loss": round(float(t.stop_loss), 5),
        "take_profit": round(float(t.take_profit), 5),
        "trailing_stop_pct": round(float(t.trailing_stop_pct), 4),
        "size": round(float(t.size), 2),
        "exit_time": _ts(t.exit_time),
        "exit_price": round(float(t.exit_price), 5) if t.exit_price else None,
        "pnl": round(float(t.pnl), 2),
        "exit_reason": t.exit_reason,
    }


def _apply_friction(trades: List[Dict], friction_cfg: Dict) -> Dict:
    """
    Apply realistic friction simulation (mirrors notebook Cell 17).
    Returns a dict with theoretical vs realistic equity curves and cost breakdown.
    """
    slippage_min = friction_cfg.get("slippage_min", 0.5)
    slippage_max = friction_cfg.get("slippage_max", 3.0)
    spread_offhours = friction_cfg.get("spread_offhours_extra", 2.5)
    news_prob = friction_cfg.get("news_spike_prob", 0.05)
    news_extra = friction_cfg.get("news_spike_extra", 5.0)
    lot_cap = friction_cfg.get("lot_cap", 2.0)
    pip_value = 10.0  # from config

    total_slippage_cost = 0.0
    total_spread_cost = 0.0
    total_lot_cap_loss = 0.0
    theoretical = []
    realistic = []
    theo_balance = 25_000.0
    real_balance = 25_000.0

    for i, t in enumerate(trades):
        pnl = t["pnl"]
        theo_balance += pnl
        theoretical.append({"trade_num": i + 1, "balance": round(theo_balance, 2)})

        # Slippage
        slippage_pips = random.uniform(slippage_min, slippage_max)
        slippage_cost = slippage_pips * pip_value * t.get("size", 0.1)
        total_slippage_cost += slippage_cost

        # Off-hours spread
        extra_spread_cost = 0.0
        entry_time = t.get("entry_time", "")
        if entry_time:
            try:
                hour = pd.Timestamp(entry_time).hour
                if hour < 8 or hour >= 22:  # Off-hours (Asia)
                    extra_spread_cost = spread_offhours * pip_value * t.get("size", 0.1)
            except Exception:
                pass
        total_spread_cost += extra_spread_cost

        # News spike
        if random.random() < news_prob:
            extra_spread_cost += news_extra * pip_value * t.get("size", 0.1)
            total_spread_cost += news_extra * pip_value * t.get("size", 0.1)

        # Lot cap loss
        lot_cap_loss = 0.0
        if t.get("size", 0) > lot_cap and pnl > 0:
            ratio = lot_cap / t["size"]
            lot_cap_loss = pnl * (1 - ratio)
            total_lot_cap_loss += lot_cap_loss

        adjusted_pnl = pnl - slippage_cost - extra_spread_cost - lot_cap_loss
        real_balance += adjusted_pnl
        realistic.append({"trade_num": i + 1, "balance": round(real_balance, 2)})

    costs = {
        "slippage": round(total_slippage_cost, 2),
        "extra_spread": round(total_spread_cost, 2),
        "lot_cap_loss": round(total_lot_cap_loss, 2),
    }

    return {
        "theoretical": theoretical,
        "realistic": realistic,
        "costs": costs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candle data loader (for the TradingView price chart)
# ─────────────────────────────────────────────────────────────────────────────

def load_candles(
    pair: str = "GBP/JPY",
    timeframe: str = "15min",
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 5000,
) -> List[Dict]:
    """
    Load OHLCV candles for the TradingView Lightweight Chart.
    Returns list of {time, open, high, low, close, volume}.
    """
    from wavetrader.data import load_forex_data

    data_dir = _resolve_dir("data")
    processed_dir = _resolve_dir("processed_data")

    # Try processed_data first (parquet), then raw data/
    df = None
    pair_tag = pair.replace("/", "")
    tf_short = timeframe.replace("min", "m")

    for d in [processed_dir / "test", processed_dir / "val", processed_dir / "train", data_dir]:
        for ext in [".parquet", ".csv"]:
            candidate = d / f"{pair_tag}_{tf_short}{ext}"
            if candidate.exists():
                try:
                    if ext == ".parquet":
                        df = pd.read_parquet(candidate)
                    else:
                        df = pd.read_csv(candidate)
                    break
                except Exception:
                    continue
        if df is not None:
            break

    if df is None:
        try:
            df = load_forex_data(pair=pair, timeframe=timeframe, data_dir=str(data_dir))
        except Exception:
            return []

    # Handle index-based timestamps (parquet files use timestamp index)
    if df.index.name in ("timestamp", "date", "datetime"):
        df = df.reset_index()

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Map mid-price columns (from preprocessed parquet: open_mid, high_mid, etc.)
    col_map = {}
    for base in ("open", "high", "low", "close"):
        if f"{base}_mid" in df.columns and base not in df.columns:
            col_map[f"{base}_mid"] = base
    if col_map:
        df = df.rename(columns=col_map)

    if "date" not in df.columns:
        for alias in ["datetime", "timestamp", "time", "gmt time"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "date"})
                break

    if "date" not in df.columns:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    df = df.tail(limit)

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(row["date"].timestamp()),
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
            "volume": float(row.get("volume", 0)),
        })

    return candles


# ─────────────────────────────────────────────────────────────────────────────
# Run backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_from_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a full backtest with the given config (merged with defaults).
    Returns a JSON-serializable results dict for the frontend.
    """
    import torch
    from wavetrader.config import BacktestConfig, MTFConfig
    from wavetrader.backtest import run_backtest
    from wavetrader.data import load_mtf_data, load_forex_data

    # Merge with defaults
    cfg = {**NOTEBOOK_DEFAULTS, **user_config}
    friction_cfg = {**NOTEBOOK_DEFAULTS["friction"], **cfg.get("friction", {})}

    pair = cfg["pair"]
    entry_tf = cfg["entry_timeframe"]

    # Validate numeric config values
    numeric_keys = [
        "initial_balance", "risk_per_trade", "leverage", "spread_pips",
        "commission_per_lot", "pip_value", "min_confidence",
        "atr_halt_multiplier", "drawdown_reduce_threshold",
    ]
    for key in numeric_keys:
        try:
            float(cfg[key])
        except (ValueError, TypeError):
            return {"error": f"Invalid value for {key}: {cfg[key]}"}

    # Build BacktestConfig
    bt_config = BacktestConfig(
        initial_balance=float(cfg["initial_balance"]),
        risk_per_trade=float(cfg["risk_per_trade"]),
        leverage=float(cfg["leverage"]),
        spread_pips=float(cfg["spread_pips"]),
        commission_per_lot=float(cfg["commission_per_lot"]),
        pip_value=float(cfg["pip_value"]),
        min_confidence=float(cfg["min_confidence"]),
        atr_halt_multiplier=float(cfg["atr_halt_multiplier"]),
        drawdown_reduce_threshold=float(cfg["drawdown_reduce_threshold"]),
    )

    # Build model config
    mtf_config = MTFConfig(pair=pair, entry_timeframe=entry_tf)

    # Load data
    data_dir = _resolve_dir("data")
    processed_dir = _resolve_dir("processed_data") / "test"

    # Try processed test data first
    df_dict = {}
    pair_tag = pair.replace("/", "")
    for tf in mtf_config.timeframes:
        tf_short = tf.replace("min", "m")
        parquet_path = processed_dir / f"{pair_tag}_{tf_short}.parquet"
        if parquet_path.exists():
            try:
                df_dict[tf] = pd.read_parquet(parquet_path)
                logger.info("Loaded %s from %s", tf, parquet_path)
            except Exception as e:
                logger.warning("Failed to read parquet %s: %s", parquet_path, e)

    if not df_dict:
        try:
            df_dict = load_mtf_data(pair=pair, data_dir=str(data_dir))
        except Exception as e:
            logger.error("load_mtf_data failed: %s", traceback.format_exc())
            return {"error": f"Failed to load data for {pair}: {e}"}

    if not df_dict:
        return {"error": f"No data found for {pair}"}

    logger.info("Data loaded: %s", {k: len(v) for k, v in df_dict.items()})

    # Load model from checkpoint
    checkpoint_dir = _resolve_dir("checkpoints")
    model = _load_latest_model(checkpoint_dir, mtf_config)

    if model is None:
        return {"error": "No model checkpoint found. Train a model first."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running backtest: pair=%s, device=%s, model loaded", pair, device)

    # Run backtest
    start_time = time.time()
    try:
        results = run_backtest(model, df_dict, mtf_config, bt_config, device)
    except Exception as e:
        logger.error("run_backtest() crashed:\n%s", traceback.format_exc())
        return {"error": f"Backtest engine error: {e}"}
    elapsed = round(time.time() - start_time, 1)
    logger.info("Backtest completed in %ss — %d trades", elapsed, len(results.trades))

    # Serialize trades
    trades_list = [_trade_to_dict(t) for t in results.trades]

    # Build breakdowns
    breakdowns = _compute_breakdowns(trades_list)

    # Friction simulation
    friction = _apply_friction(trades_list, friction_cfg)

    # Downsample equity curve if too large (>5000 points)
    equity = [round(e, 2) for e in results.equity_curve]
    if len(equity) > 5000:
        step = len(equity) // 5000
        equity = equity[::step]

    run_id = str(uuid.uuid4())[:8]

    return {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "config": cfg,
        "metrics": {
            "total_trades": results.total_trades,
            "winning_trades": results.winning_trades,
            "losing_trades": results.losing_trades,
            "win_rate": round(results.win_rate, 4),
            "total_pnl": round(results.total_pnl, 2),
            "profit_factor": round(results.profit_factor, 2),
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": round(results.max_drawdown, 4),
            "final_balance": round(results.final_balance, 2),
            "return_pct": round(
                (results.final_balance / bt_config.initial_balance - 1) * 100, 1
            ),
        },
        "trades": trades_list,
        "equity_curve": equity,
        "breakdowns": breakdowns,
        "friction": friction,
    }


def load_cached_results() -> Optional[Dict[str, Any]]:
    """
    Load results from existing CSV files in backtest_results/ if available.
    This lets the dashboard display previous backtest results instantly.
    """
    results_dir = _RESULTS_DIR
    if not results_dir.exists():
        return None

    trade_log = results_dir / "trade_log.csv"
    equity_csv = results_dir / "equity_curve.csv"

    if not trade_log.exists():
        return None

    try:
        trade_df = pd.read_csv(trade_log)
        trades_list = []
        for _, row in trade_df.iterrows():
            trades_list.append({
                "entry_time": str(row.get("entry_time", "")),
                "entry_price": round(float(row.get("entry_price", 0)), 5),
                "direction": str(row.get("direction", "HOLD")),
                "stop_loss": round(float(row.get("stop_loss", 0)), 5),
                "take_profit": round(float(row.get("take_profit", 0)), 5),
                "trailing_stop_pct": round(float(row.get("trailing_stop_pct", 0)), 4),
                "size": round(float(row.get("size", row.get("lot_size", 0.01))), 2),
                "exit_time": str(row.get("exit_time", "")),
                "exit_price": round(float(row.get("exit_price", 0)), 5),
                "pnl": round(float(row.get("pnl", 0)), 2),
                "exit_reason": str(row.get("exit_reason", "")),
            })

        # Equity curve
        equity = []
        if equity_csv.exists():
            eq_df = pd.read_csv(equity_csv)
            col = eq_df.columns[-1]  # last column is usually balance
            equity = [round(float(v), 2) for v in eq_df[col].dropna().tolist()]

        # Compute metrics from trades
        winning = [t for t in trades_list if t["pnl"] > 0]
        losing = [t for t in trades_list if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades_list)
        gross_profit = sum(t["pnl"] for t in winning)
        gross_loss = abs(sum(t["pnl"] for t in losing))

        initial = NOTEBOOK_DEFAULTS["initial_balance"]
        final = initial + total_pnl

        breakdowns = _compute_breakdowns(trades_list)
        friction = _apply_friction(trades_list, NOTEBOOK_DEFAULTS["friction"])

        return {
            "run_id": "cached",
            "elapsed_seconds": 0,
            "config": NOTEBOOK_DEFAULTS,
            "metrics": {
                "total_trades": len(trades_list),
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": round(len(winning) / max(len(trades_list), 1), 4),
                "total_pnl": round(total_pnl, 2),
                "profit_factor": round(gross_profit / max(gross_loss, 1e-9), 2),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_balance": round(final, 2),
                "return_pct": round((final / initial - 1) * 100, 1),
            },
            "trades": trades_list,
            "equity_curve": equity if equity else [initial],
            "breakdowns": breakdowns,
            "friction": friction,
        }
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Breakdowns (mirrors notebook Cell 13-15 analytics)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_breakdowns(trades: List[Dict]) -> Dict:
    """Compute monthly, hourly, daily, and exit-reason breakdowns."""
    if not trades:
        return {"monthly": [], "hourly": [], "daily": [], "exit_reasons": [], "durations": []}

    df = pd.DataFrame(trades)

    # Parse timestamps
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["is_winner"] = df["pnl"] > 0

    # Duration in hours
    durations = []
    if df["exit_dt"].notna().any() and df["entry_dt"].notna().any():
        mask = df["entry_dt"].notna() & df["exit_dt"].notna()
        dur = (df.loc[mask, "exit_dt"] - df.loc[mask, "entry_dt"]).dt.total_seconds() / 3600
        df.loc[mask, "duration_hours"] = dur
        win_dur = df.loc[mask & df["is_winner"], "duration_hours"].tolist()
        loss_dur = df.loc[mask & ~df["is_winner"], "duration_hours"].tolist()
        durations = {
            "all": [round(d, 2) for d in dur.tolist()],
            "winners": [round(d, 2) for d in win_dur],
            "losers": [round(d, 2) for d in loss_dur],
        }

    # Monthly
    monthly = []
    if df["entry_dt"].notna().any():
        df["month"] = df["entry_dt"].dt.to_period("M").astype(str)
        mg = df.groupby("month").agg(
            net_pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("is_winner", "mean"),
        ).reset_index()
        monthly = [
            {
                "month": row["month"],
                "net_pnl": round(row["net_pnl"], 2),
                "trades": int(row["trades"]),
                "win_rate": round(row["win_rate"], 4),
            }
            for _, row in mg.iterrows()
        ]

    # Hourly
    hourly = []
    if df["entry_dt"].notna().any():
        df["hour"] = df["entry_dt"].dt.hour
        hg = df.groupby("hour").agg(
            net_pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("is_winner", "mean"),
        ).reset_index()
        hourly = [
            {
                "hour": int(row["hour"]),
                "net_pnl": round(row["net_pnl"], 2),
                "trades": int(row["trades"]),
                "win_rate": round(row["win_rate"], 4),
            }
            for _, row in hg.iterrows()
        ]

    # Day of week
    daily = []
    if df["entry_dt"].notna().any():
        df["dow"] = df["entry_dt"].dt.dayofweek
        dg = df.groupby("dow").agg(
            net_pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("is_winner", "mean"),
        ).reset_index()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = [
            {
                "day": day_names[int(row["dow"])],
                "dow": int(row["dow"]),
                "net_pnl": round(row["net_pnl"], 2),
                "trades": int(row["trades"]),
                "win_rate": round(row["win_rate"], 4),
            }
            for _, row in dg.iterrows()
        ]

    # Exit reasons
    exit_reasons = []
    if "exit_reason" in df.columns:
        eg = df.groupby("exit_reason").agg(
            total_pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("is_winner", "mean"),
        ).reset_index()
        exit_reasons = [
            {
                "reason": row["exit_reason"],
                "total_pnl": round(row["total_pnl"], 2),
                "trades": int(row["trades"]),
                "win_rate": round(row["win_rate"], 4),
            }
            for _, row in eg.iterrows()
        ]

    return {
        "monthly": monthly,
        "hourly": hourly,
        "daily": daily,
        "exit_reasons": exit_reasons,
        "durations": durations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_model(checkpoint_dir: Path, config: Any) -> Any:
    """Find and load the most recent model checkpoint."""
    import torch

    if not checkpoint_dir.exists():
        return None

    # Find latest checkpoint directory
    ckpt_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    for ckpt_dir in ckpt_dirs:
        # Look for model weights
        for pattern in ["best_model.pt", "model.pt", "*.pt"]:
            matches = list(ckpt_dir.glob(pattern))
            if matches:
                try:
                    from wavetrader.model import WaveTraderMTF
                    model = WaveTraderMTF(config)
                    state = torch.load(matches[0], map_location="cpu", weights_only=False)
                    if "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"])
                    else:
                        model.load_state_dict(state)
                    logger.info("Model loaded from %s", matches[0])
                    return model
                except Exception as e:
                    logger.warning("Failed to load model from %s: %s", matches[0], e)
                    continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Saved backtests (persist / recall / compare)
# ─────────────────────────────────────────────────────────────────────────────

_SAVED_DIR = _RESULTS_DIR / "saved"


def save_backtest_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Save a backtest result to disk for later recall."""
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    run_id = data.get("run_id", str(uuid.uuid4())[:8])
    entry = {
        "run_id": run_id,
        "saved_at": datetime.utcnow().isoformat(),
        "config": data.get("config", {}),
        "metrics": data.get("metrics", {}),
        "trades": data.get("trades", []),
        "equity_curve": data.get("equity_curve", []),
        "breakdowns": data.get("breakdowns", {}),
        "friction": data.get("friction", {}),
    }
    path = _SAVED_DIR / f"{run_id}.json"
    path.write_text(json.dumps(entry, default=str))
    logger.info("Saved backtest %s to %s", run_id, path)
    return {"run_id": run_id, "saved_at": entry["saved_at"]}


def list_saved_backtests() -> List[Dict[str, Any]]:
    """List all saved backtests (metadata only, no full trade data)."""
    if not _SAVED_DIR.exists():
        return []
    entries = []
    for f in sorted(_SAVED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            entries.append({
                "run_id": data.get("run_id"),
                "saved_at": data.get("saved_at"),
                "config": data.get("config", {}),
                "metrics": data.get("metrics", {}),
            })
        except Exception:
            continue
    return entries


def load_saved_backtest(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a saved backtest by run_id."""
    # Sanitize: only allow alphanumeric + hyphen
    safe_id = "".join(c for c in run_id if c.isalnum() or c == "-")
    path = _SAVED_DIR / f"{safe_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
