"""V3 R:R optimization — test all 3 approaches to push R:R higher."""
import pandas as pd
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.strategies.price_action_reversal import PriceActionReversalStrategy
from wavetrader.config import BacktestConfig

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


def run_test(label, tp_levels, params):
    if tp_levels is not None:
        bt = BacktestConfig(initial_balance=100.0, risk_per_trade=0.10, multi_tp_levels=tp_levels)
        params.setdefault("exit_mode", "multi_tp_trail")
    else:
        bt = BacktestConfig(initial_balance=100.0, risk_per_trade=0.10)

    strategy = PriceActionReversalStrategy(params=params)
    r = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt, pair='GBP/JPY', verbose=False)

    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_l = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
    max_w = max((t.pnl for t in r.trades), default=0)
    wl = round(avg_w / avg_l, 2) if avg_l > 0 else 0
    holds = [(t.exit_time - t.entry_time).total_seconds() / 3600
             for t in r.trades if t.exit_time and t.entry_time]
    avg_hold = sum(holds) / len(holds) if holds else 0
    pf = r.profit_factor if r.profit_factor < 999 else 999
    return {
        "label": label, "trades": r.total_trades,
        "wr": r.win_rate * 100, "pf": pf, "bal": r.final_balance,
        "avg_w": avg_w, "avg_l": avg_l, "max_w": max_w,
        "wl": wl, "hold_h": avg_hold,
    }


def print_header(title):
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")
    print(f"{'Config':<35} {'Tr':>3} {'WR%':>5} {'PF':>5} {'Bal':>8} {'AvgW':>7} {'AvgL':>7} {'MaxW':>7} {'R:R':>5} {'HoldH':>5}")
    print(f"{'-'*115}")


def print_row(r):
    pf_s = f"{r['pf']:>5.2f}" if r['pf'] < 100 else "  inf"
    print(f"{r['label']:<35} {r['trades']:>3} {r['wr']:>5.1f} {pf_s} ${r['bal']:>7.2f} "
          f"${r['avg_w']:>6.2f} ${r['avg_l']:>6.2f} ${r['max_w']:>6.2f} {r['wl']:>5.2f} {r['hold_h']:>5.0f}")


# ─── Baseline ────────────────────────────────────────────────────────────────
all_results = []

print_header("BASELINE — current state")
for label, tp, p in [
    ("No filter + 1R50/2R25/3R15", ((1.0,.50),(2.0,.25),(3.0,.15)),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("EMA + 1R50/2R25/3R15", ((1.0,.50),(2.0,.25),(3.0,.15)),
     {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
]:
    r = run_test(label, tp, p)
    print_row(r)
    all_results.append(r)

# ─── APPROACH 1: Geometric trail only ─────────────────────────────────────
print_header("APPROACH 1 — Geometric trail (no multi-TP)")
trail_pcts = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
for tpct in trail_pcts:
    for ema in [False, True]:
        ema_lbl = "+EMA" if ema else ""
        p = {"exit_mode": "geometric_trail", "trailing_stop_pct": tpct,
             "use_daily_ema_filter": ema, "use_rsi_filter": False, "use_adx_filter": False,
             "max_sl_pips": 200.0}
        r = run_test(f"geo trail={tpct}{ema_lbl}", None, p)
        print_row(r)
        all_results.append(r)

# ─── APPROACH 2: Delay first TP to 2R+ ───────────────────────────────────
print_header("APPROACH 2 — Later first TP (bigger wins, lower WR)")
tp_late = [
    ("2R-50%", ((2.0, 0.50),)),
    ("2R-40%/4R-20%", ((2.0, 0.40), (4.0, 0.20))),
    ("2R-30%/3R-20%/5R-10%", ((2.0, 0.30), (3.0, 0.20), (5.0, 0.10))),
    ("3R-50%", ((3.0, 0.50),)),
    ("3R-40%/5R-20%", ((3.0, 0.40), (5.0, 0.20))),
    ("1.5R-40%/2.5R-20%", ((1.5, 0.40), (2.5, 0.20))),
    ("1.5R-30%/3R-30%", ((1.5, 0.30), (3.0, 0.30))),
    ("2R-25%/4R-25%", ((2.0, 0.25), (4.0, 0.25))),
]
for label, tp in tp_late:
    for ema in [False, True]:
        ema_lbl = " +EMA" if ema else ""
        p = {"use_daily_ema_filter": ema, "use_rsi_filter": False, "use_adx_filter": False,
             "max_sl_pips": 200.0}
        r = run_test(f"{label}{ema_lbl}", tp, p)
        print_row(r)
        all_results.append(r)

# ─── APPROACH 3: Smaller TP1 portion (more runner) ───────────────────────
print_header("APPROACH 3 — Small TP1 portion (big runner)")
tp_small = [
    ("1R-20%/2R-20% (60%run)", ((1.0, 0.20), (2.0, 0.20))),
    ("1R-15%/2R-15% (70%run)", ((1.0, 0.15), (2.0, 0.15))),
    ("1R-10%/2R-10%/3R-10%(70%run)", ((1.0, 0.10), (2.0, 0.10), (3.0, 0.10))),
    ("1R-25%/2R-15% (60%run)", ((1.0, 0.25), (2.0, 0.15))),
    ("1R-20%/3R-20% (60%run)", ((1.0, 0.20), (3.0, 0.20))),
    ("1R-15%/2R-15%/4R-10%(60%run)", ((1.0, 0.15), (2.0, 0.15), (4.0, 0.10))),
    ("0.5R-10%/1R-10%/2R-10%(70%run)", ((0.5, 0.10), (1.0, 0.10), (2.0, 0.10))),
    ("1R-30% only (70%run)", ((1.0, 0.30),)),
    ("1R-20% only (80%run)", ((1.0, 0.20),)),
]
for label, tp in tp_small:
    for ema in [False, True]:
        ema_lbl = " +EMA" if ema else ""
        p = {"use_daily_ema_filter": ema, "use_rsi_filter": False, "use_adx_filter": False,
             "max_sl_pips": 200.0}
        r = run_test(f"{label}{ema_lbl}", tp, p)
        print_row(r)
        all_results.append(r)

# ─── Summary: Top 15 by PF (min 15 trades) ──────────────────────────────
print_header("TOP 15 by PF (min 15 trades)")
qual = sorted([r for r in all_results if r["trades"] >= 15], key=lambda x: -x["pf"])
for r in qual[:15]:
    print_row(r)

print_header("TOP 15 by R:R (min 15 trades)")
qual_rr = sorted([r for r in all_results if r["trades"] >= 15], key=lambda x: -x["wl"])
for r in qual_rr[:15]:
    print_row(r)

print_header("TOP 15 by Balance (min 15 trades)")
qual_bal = sorted([r for r in all_results if r["trades"] >= 15], key=lambda x: -x["bal"])
for r in qual_bal[:15]:
    print_row(r)
