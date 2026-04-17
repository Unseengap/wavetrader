"""V3 grid search — HTF filters + max_sl_pips optimization."""
import itertools
import pandas as pd
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.strategies.price_action_reversal import PriceActionReversalStrategy
from wavetrader.config import BacktestConfig

# Load data once
df_dict = {}
for tf in ['4h', '1d']:
    path = f'processed_data/test/GBPJPY_{tf}.parquet'
    df = pd.read_parquet(path)
    if df.index.name in ('timestamp', 'date', 'datetime'):
        df = df.reset_index()
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {}
    for base in ('open', 'high', 'low', 'close'):
        if f'{base}_mid' in df.columns and base not in df.columns:
            col_map[f'{base}_mid'] = base
    if col_map:
        df = df.rename(columns=col_map)
    if 'date' not in df.columns:
        for alias in ('datetime', 'timestamp', 'time'):
            if alias in df.columns:
                df = df.rename(columns={alias: 'date'})
                break
    df_dict[tf] = df

bt_config = BacktestConfig(
    initial_balance=100.0, risk_per_trade=0.10,
    multi_tp_levels=((1.0, 0.50), (2.0, 0.25), (3.0, 0.15)),
)

# Grid parameters
grid = {
    "use_daily_ema_filter": [True, False],
    "use_rsi_filter": [True, False],
    "use_adx_filter": [True, False],
    "min_adx": [12.0, 15.0, 20.0],
    "max_sl_pips": [60.0, 80.0, 100.0, 120.0],
    "rsi_lower": [25.0, 30.0],
    "rsi_upper": [70.0, 75.0],
}

# Build combos — skip combos where filter is off but its params vary
combos = []
for ema_on in grid["use_daily_ema_filter"]:
    for rsi_on in grid["use_rsi_filter"]:
        rsi_lowers = grid["rsi_lower"] if rsi_on else [30.0]
        rsi_uppers = grid["rsi_upper"] if rsi_on else [70.0]
        for adx_on in grid["use_adx_filter"]:
            adx_vals = grid["min_adx"] if adx_on else [15.0]
            for max_sl in grid["max_sl_pips"]:
                for rsi_lo in rsi_lowers:
                    for rsi_up in rsi_uppers:
                        for adx_v in adx_vals:
                            combos.append({
                                "use_daily_ema_filter": ema_on,
                                "use_rsi_filter": rsi_on,
                                "use_adx_filter": adx_on,
                                "min_adx": adx_v,
                                "max_sl_pips": max_sl,
                                "rsi_lower": rsi_lo,
                                "rsi_upper": rsi_up,
                            })

print(f"Testing {len(combos)} combinations...")
results = []

for idx, params in enumerate(combos):
    strategy = PriceActionReversalStrategy(params=params)
    r = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt_config, pair='GBP/JPY', verbose=False)
    
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_l = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
    max_w = max((t.pnl for t in r.trades), default=0)
    
    results.append({
        **params,
        "trades": r.total_trades,
        "wr": round(r.win_rate * 100, 1),
        "pf": round(r.profit_factor, 2),
        "bal": round(r.final_balance, 2),
        "avg_w": round(avg_w, 2),
        "avg_l": round(avg_l, 2),
        "max_w": round(max_w, 2),
        "w_l_ratio": round(avg_w / avg_l, 2) if avg_l > 0 else 0,
    })
    
    if (idx + 1) % 50 == 0:
        print(f"  {idx+1}/{len(combos)} done...")

# Sort by PF descending, then balance
results.sort(key=lambda x: (-x["pf"], -x["bal"]))

print(f"\n{'='*120}")
print(f"{'EMA':>4} {'RSI':>4} {'ADX':>4} {'mADX':>5} {'maxSL':>5} {'rLo':>4} {'rUp':>4} | {'Tr':>3} {'WR%':>5} {'PF':>5} {'Bal':>8} {'AvgW':>6} {'AvgL':>6} {'MaxW':>6} {'W/L':>5}")
print(f"{'='*120}")

for r in results[:40]:
    ema = "Y" if r["use_daily_ema_filter"] else "N"
    rsi = "Y" if r["use_rsi_filter"] else "N"
    adx = "Y" if r["use_adx_filter"] else "N"
    print(f"{ema:>4} {rsi:>4} {adx:>4} {r['min_adx']:>5.0f} {r['max_sl_pips']:>5.0f} {r['rsi_lower']:>4.0f} {r['rsi_upper']:>4.0f} | "
          f"{r['trades']:>3} {r['wr']:>5.1f} {r['pf']:>5.2f} ${r['bal']:>7.2f} ${r['avg_w']:>5.2f} ${r['avg_l']:>5.02f} ${r['max_w']:>5.02f} {r['w_l_ratio']:>5.2f}")

# Also show the best by balance (might differ from PF sort)
print(f"\n--- Top 10 by Balance ---")
by_bal = sorted(results, key=lambda x: -x["bal"])
for r in by_bal[:10]:
    ema = "Y" if r["use_daily_ema_filter"] else "N"
    rsi = "Y" if r["use_rsi_filter"] else "N"
    adx = "Y" if r["use_adx_filter"] else "N"
    print(f"{ema:>4} {rsi:>4} {adx:>4} {r['min_adx']:>5.0f} {r['max_sl_pips']:>5.0f} {r['rsi_lower']:>4.0f} {r['rsi_upper']:>4.0f} | "
          f"{r['trades']:>3} {r['wr']:>5.1f} {r['pf']:>5.2f} ${r['bal']:>7.2f} ${r['avg_w']:>5.02f} ${r['avg_l']:>5.02f} ${r['max_w']:>5.02f} {r['w_l_ratio']:>5.02f}")

# Best combo with PF >= 2.0 and trades >= 15
print(f"\n--- Best with PF >= 2.0 and trades >= 15 ---")
filtered = [r for r in results if r["pf"] >= 2.0 and r["trades"] >= 15]
if filtered:
    for r in filtered[:10]:
        ema = "Y" if r["use_daily_ema_filter"] else "N"
        rsi = "Y" if r["use_rsi_filter"] else "N"
        adx = "Y" if r["use_adx_filter"] else "N"
        print(f"{ema:>4} {rsi:>4} {adx:>4} {r['min_adx']:>5.0f} {r['max_sl_pips']:>5.0f} {r['rsi_lower']:>4.0f} {r['rsi_upper']:>4.0f} | "
              f"{r['trades']:>3} {r['wr']:>5.1f} {r['pf']:>5.2f} ${r['bal']:>7.2f} ${r['avg_w']:>5.02f} ${r['avg_l']:>5.02f} ${r['max_w']:>5.02f} {r['w_l_ratio']:>5.02f}")
else:
    print("  No combos met PF >= 2.0 with 15+ trades")
