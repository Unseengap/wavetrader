"""Quick backtest optimization for news_catalyst_ob strategy."""
import pandas as pd
from pathlib import Path
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.config import BacktestConfig

# Load data
processed_dir = Path("processed_data/test")
pair_tag = "GBPJPY"
df_dict = {}
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

# ── Final verification: default params + a few variations ────────────
combos = []

# 1. Pure defaults (overlap 13-17, R:R=3, T=50%, engine A=3.0)
combos.append(({}, 3.0, "DEFAULTS (ovl 13-17, R:R=3, T=50%)"))
# 2. Defaults with A=2.0 (trail activates earlier)
combos.append(({}, 2.0, "A=2.0 (rest default)"))
# 3. Compare: wider session 10-20
combos.append(({"session_start_hour": 10, "session_end_hour": 20}, 3.0, "Session 10-20"))
# 4. Compare: full session 7-20
combos.append(({"session_start_hour": 7, "session_end_hour": 20}, 3.0, "Session 7-20"))
# 5. No trailing baseline
combos.append(({"trailing_stop_pct": 0.0}, 3.0, "No Trailing"))
# 6. Higher R:R with trailing
combos.append(({"min_rr_ratio": 5.0}, 3.0, "R:R=5 + trailing"))
# 7. A=3.0 pure trailing (R:R=10, no practical TP)
combos.append(({"min_rr_ratio": 10.0}, 3.0, "Pure trail (R:R=10)"))
# 8. V3 comparison (no confirmation)
combos.append(({"require_confirmation": False, "sl_atr_mult": 1.5, "session_start_hour": 13, "session_end_hour": 17}, 3.0, "V3 no confirm"))

print(f"Testing {len(combos)} combos...\n")

for idx, (params, act_r, label) in enumerate(combos):
    cfg = BacktestConfig()
    cfg.initial_balance = 100.0
    cfg.risk_pct = 0.10
    cfg.trail_activate_r = act_r

    strat = reg.instantiate('news_catalyst_ob', params=params)
    results = run_strategy_backtest(strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)
    trades = results.trades
    n = len(trades)
    if n > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        total_win = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999
        sl = sum(1 for t in trades if (t.exit_reason or '').strip() == 'Stop Loss')
        tp = sum(1 for t in trades if 'take' in (t.exit_reason or '').lower() or 'profit' in (t.exit_reason or '').lower())
        trail = sum(1 for t in trades if 'trail' in (t.exit_reason or '').lower())
        eob = n - sl - tp - trail  # End of backtest exits
        avg_win = total_win / max(wins, 1)
        avg_loss = total_loss / max(n - wins, 1)
        print(f"{idx+1}. [{label:>35}] {n:>3}tr WR={wins/n:.0%} PF={pf:.2f} "
              f"${results.final_balance:>8.2f} SL={sl} TP={tp} Trail={trail} "
              f"AvgW=${avg_win:.2f} AvgL=${avg_loss:.2f}")
    else:
        print(f"{idx+1}. [{label:>35}] 0 trades")

print(f"Testing {len(combos)} combos...\n")

for idx, (params, act_r, label) in enumerate(combos):
    cfg = BacktestConfig()
    cfg.initial_balance = 100.0
    cfg.risk_pct = 0.10
    cfg.trail_activate_r = act_r

    strat = reg.instantiate('news_catalyst_ob', params=params)
    results = run_strategy_backtest(strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)
    trades = results.trades
    n = len(trades)
    if n > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        total_win = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999
        sl = sum(1 for t in trades if (t.exit_reason or '').strip() == 'Stop Loss')
        tp = sum(1 for t in trades if 'take' in (t.exit_reason or '').lower() or 'profit' in (t.exit_reason or '').lower())
        trail = sum(1 for t in trades if 'trail' in (t.exit_reason or '').lower())
        eob = n - sl - tp - trail  # End of backtest exits
        avg_win = total_win / max(wins, 1)
        avg_loss = total_loss / max(n - wins, 1)
        print(f"{idx+1}. [{label:>35}] {n:>3}tr WR={wins/n:.0%} PF={pf:.2f} "
              f"${results.final_balance:>8.2f} SL={sl} TP={tp} Trail={trail} "
              f"AvgW=${avg_win:.2f} AvgL=${avg_loss:.2f}")
    else:
        print(f"{idx+1}. [{label:>35}] 0 trades")

print("\nDone!")
