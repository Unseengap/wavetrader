"""Quick backtest optimization for fib_scalper strategy."""
import pandas as pd
from pathlib import Path
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.config import BacktestConfig

# ── Load Data ────────────────────────────────────────────────────────
processed_dir = Path("processed_data/test")
pair_tag = "GBPJPY"
df_dict = {}

# Load 1min data (primary entry timeframe)
p_1m = processed_dir / f"{pair_tag}_1m.parquet"
if p_1m.exists():
    df = pd.read_parquet(p_1m)
    if df.index.name in ("timestamp", "date", "datetime"):
        df = df.reset_index()
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {}
    for base in ("open", "high", "low", "close"):
        if f"{base}_mid" in df.columns and base not in df.columns:
            col_map[f"{base}_mid"] = base
    if col_map:
        df = df.rename(columns=col_map)
    if "date" not in df.columns:
        for alias in ("datetime", "timestamp", "time"):
            if alias in df.columns:
                df = df.rename(columns={alias: "date"})
                break
    df_dict["1min"] = df
    print(f"  1min: {len(df)} bars")
else:
    print(f"  WARNING: {p_1m} not found! Run scripts/process_1min.py first.")

# Load HTF data for filters
for tf in ["15min", "1h", "4h", "1d"]:
    tf_short = tf.replace("min", "m")
    p = processed_dir / f"{pair_tag}_{tf_short}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if df.index.name in ("timestamp", "date", "datetime"):
            df = df.reset_index()
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {}
        for base in ("open", "high", "low", "close"):
            if f"{base}_mid" in df.columns and base not in df.columns:
                col_map[f"{base}_mid"] = base
        if col_map:
            df = df.rename(columns=col_map)
        if "date" not in df.columns:
            for alias in ("datetime", "timestamp", "time"):
                if alias in df.columns:
                    df = df.rename(columns={alias: "date"})
                    break
        df_dict[tf] = df
        print(f"  {tf}: {len(df)} bars")
print()

reg = get_strategy_registry()

# ── Define Parameter Combos ──────────────────────────────────────────
combos = [
    ({}, 3.0, "DEFAULTS"),
    ({"trailing_stop_pct": 0.0}, 3.0, "No Trailing (baseline)"),
    ({"trailing_stop_pct": 0.35, "min_rr_ratio": 1.5}, 2.0, "T=35% R=1.5 A=2"),
    ({"trailing_stop_pct": 0.50, "min_rr_ratio": 1.5}, 3.0, "T=50% R=1.5 A=3"),
    ({"min_rr_ratio": 1.0}, 3.0, "R:R=1.0"),
    ({"min_rr_ratio": 1.5}, 3.0, "R:R=1.5"),
    ({"min_rr_ratio": 2.0}, 3.0, "R:R=2.0"),
    ({"use_ema_200_filter": False, "use_htf_bias": False}, 3.0, "No trend filters"),
    ({"use_ema_200_filter": False}, 3.0, "No EMA filter"),
    ({"require_confirmation": False}, 3.0, "No confirm"),
    ({"require_confirmation": False, "use_ema_200_filter": False, "use_htf_bias": False}, 3.0, "All filters OFF"),
    ({"min_confidence": 0.35}, 3.0, "Low confidence"),
    ({"swing_lookback": 2, "min_impulse_atr": 0.8}, 3.0, "Fast swings"),
    ({"max_swing_age": 200, "max_active_setups": 8}, 3.0, "More setups"),
    ({"require_confirmation": False, "min_rr_ratio": 1.0, "min_confidence": 0.35}, 2.0, "Maximum trades"),
]

# ── Run All Combos ───────────────────────────────────────────────────
print(f"Testing {len(combos)} combos...\n")
for idx, (params, act_r, label) in enumerate(combos):
    cfg = BacktestConfig()
    cfg.initial_balance = 100.0
    cfg.risk_per_trade = 0.10
    cfg.trail_activate_r = act_r

    strat = reg.instantiate('fib_scalper', params=params)
    results = run_strategy_backtest(
        strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False,
    )
    trades = results.trades
    n = len(trades)
    if n > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        total_win = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999
        sl = sum(1 for t in trades if (t.exit_reason or '').strip() == 'Stop Loss')
        tp = sum(1 for t in trades if 'take' in (t.exit_reason or '').lower())
        trail = sum(1 for t in trades if 'trail' in (t.exit_reason or '').lower())
        avg_w = total_win / max(wins, 1)
        avg_l = total_loss / max(n - wins, 1)
        print(f"{idx+1:>2}. [{label:>30}] {n:>4}tr WR={wins/n:.0%} PF={pf:.2f} "
              f"${results.final_balance:>8.2f} SL={sl} TP={tp} Trail={trail} "
              f"AvgW=${avg_w:.2f} AvgL=${avg_l:.2f}")
    else:
        print(f"{idx+1:>2}. [{label:>30}] 0 trades")
print("\nDone!")
