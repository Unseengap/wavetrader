"""Quick backtest for harmonic_scanner strategy with per-pattern stats."""
import pandas as pd
from pathlib import Path
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.config import BacktestConfig
from wavetrader.strategies.harmonic_scanner import print_pattern_stats

# ── Load data ────────────────────────────────────────────────────────────────
processed_dir = Path("processed_data/test")
pair_tag = "GBPJPY"
pair = "GBP/JPY"

df_dict = {}
for tf in ["1h", "4h", "1d"]:
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

# ── Run backtest ─────────────────────────────────────────────────────────────
reg = get_strategy_registry()
cfg = BacktestConfig()
cfg.initial_balance = 100.0
cfg.risk_per_trade = 0.10

strat = reg.instantiate('harmonic_scanner')
print(f"Strategy: {strat.meta.name} v{strat.meta.version}")
print(f"Params: trailing_stop_pct={strat.params['trailing_stop_pct']}, "
      f"fib_tolerance={strat.params['fib_tolerance']}")
print()

results = run_strategy_backtest(strat, df_dict, bt_config=cfg, pair=pair, verbose=True)
trades = results.trades

# ── Summary ──────────────────────────────────────────────────────────────────
n = len(trades)
if n > 0:
    wins = sum(1 for t in trades if t.pnl > 0)
    total_win = sum(t.pnl for t in trades if t.pnl > 0)
    total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pf = total_win / total_loss if total_loss > 0 else 999
    print(f"\n{'='*60}")
    print(f"TOTAL: {n} trades  WR={wins/n:.0%}  PF={pf:.2f}  "
          f"Final=${results.final_balance:.2f}")
    print(f"{'='*60}")

    # ── Per-pattern breakdown ────────────────────────────────────────────
    print_pattern_stats(trades)

    # ── Exit reason breakdown ────────────────────────────────────────────
    from collections import Counter
    reasons = Counter(t.exit_reason for t in trades)
    print(f"\nExit reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    # ── Context check ────────────────────────────────────────────────────
    has_ctx = sum(1 for t in trades if getattr(t, 'context', {}))
    print(f"\nTrades with context: {has_ctx}/{n}")
    if trades:
        t0 = trades[0]
        ctx = getattr(t0, 'context', {})
        if ctx:
            print(f"Sample context: {ctx}")
else:
    print("\n0 trades — check pattern detection or data availability")

print("\nDone!")
