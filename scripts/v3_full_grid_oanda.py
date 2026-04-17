"""Full multi-TP + trailing stop grid — OANDA $100, 10% risk, corrected sizing."""
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


def oanda_bt(tp_levels=None):
    kw = dict(initial_balance=100.0, risk_per_trade=0.10, leverage=20.0,
              spread_pips=4.2, commission_per_lot=0.0, pip_value=6.50,
              drawdown_reduce_threshold=0.15, margin_use_limit=0.90)
    if tp_levels is not None:
        kw["multi_tp_levels"] = tp_levels
    return BacktestConfig(**kw)


def run(label, tp_levels, params):
    bt = oanda_bt(tp_levels)
    if tp_levels is not None:
        params["exit_mode"] = "multi_tp_trail"
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
    pf = min(r.profit_factor, 999)
    return {"label": label, "trades": r.total_trades, "wr": r.win_rate * 100,
            "pf": pf, "bal": r.final_balance, "ret": (r.final_balance - 100) / 100 * 100,
            "avg_w": avg_w, "avg_l": avg_l, "max_w": max_w, "wl": wl, "hold_h": avg_hold}


def hdr(title):
    print(f"\n{'='*125}")
    print(f"  {title}  |  OANDA $100, 10% risk, 20:1 lev, 4.2 spread")
    print(f"{'='*125}")
    print(f"{'Config':<35} {'Tr':>3} {'WR%':>5} {'PF':>5} {'EndBal':>8} {'Ret%':>6} {'AvgW':>7} {'AvgL':>7} {'MaxW':>7} {'R:R':>5} {'HldH':>5}")
    print(f"{'-'*125}")


def row(r):
    pf_s = f"{r['pf']:>5.2f}" if r['pf'] < 100 else "  inf"
    print(f"{r['label']:<35} {r['trades']:>3} {r['wr']:>5.1f} {pf_s} ${r['bal']:>7.0f} {r['ret']:>5.0f}% "
          f"${r['avg_w']:>6.0f} ${r['avg_l']:>6.0f} ${r['max_w']:>6.0f} {r['wl']:>5.2f} {r['hold_h']:>5.0f}")


all_results = []
no_filter = {"use_daily_ema_filter": False, "use_rsi_filter": False, "use_adx_filter": False, "max_sl_pips": 200.0}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: GEOMETRIC TRAILING STOP (no multi-TP)
# ══════════════════════════════════════════════════════════════════════════════
hdr("GEOMETRIC TRAILING STOP — no partials, full ride")
for tpct in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    p = {**no_filter, "exit_mode": "geometric_trail", "trailing_stop_pct": tpct}
    r = run(f"geo trail={tpct}", None, p)
    row(r); all_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SINGLE TP + RUNNER (multi_tp_trail with 1 TP level)
# ══════════════════════════════════════════════════════════════════════════════
hdr("SINGLE TP + TRAILING RUNNER")
single_tps = [
    ("1R-20% (80%run)", ((1.0, 0.20),)),
    ("1R-30% (70%run)", ((1.0, 0.30),)),
    ("1R-40% (60%run)", ((1.0, 0.40),)),
    ("1R-50% (50%run)", ((1.0, 0.50),)),
    ("1.5R-30% (70%run)", ((1.5, 0.30),)),
    ("1.5R-50% (50%run)", ((1.5, 0.50),)),
    ("2R-30% (70%run)", ((2.0, 0.30),)),
    ("2R-50% (50%run)", ((2.0, 0.50),)),
    ("3R-30% (70%run)", ((3.0, 0.30),)),
    ("3R-50% (50%run)", ((3.0, 0.50),)),
    ("3R-70% (30%run)", ((3.0, 0.70),)),
    ("4R-50% (50%run)", ((4.0, 0.50),)),
    ("5R-50% (50%run)", ((5.0, 0.50),)),
]
for label, tp in single_tps:
    r = run(label, tp, {**no_filter})
    row(r); all_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TWO TPs + RUNNER
# ══════════════════════════════════════════════════════════════════════════════
hdr("TWO TPs + TRAILING RUNNER")
two_tps = [
    ("1R-25%/2R-25% (50%run)", ((1.0, 0.25), (2.0, 0.25))),
    ("1R-30%/2R-20% (50%run)", ((1.0, 0.30), (2.0, 0.20))),
    ("1R-30%/3R-20% (50%run)", ((1.0, 0.30), (3.0, 0.20))),
    ("1R-20%/3R-30% (50%run)", ((1.0, 0.20), (3.0, 0.30))),
    ("1.5R-30%/3R-20% (50%run)", ((1.5, 0.30), (3.0, 0.20))),
    ("1.5R-40%/3R-20% (40%run)", ((1.5, 0.40), (3.0, 0.20))),
    ("2R-30%/4R-20% (50%run)", ((2.0, 0.30), (4.0, 0.20))),
    ("2R-40%/4R-20% (40%run)", ((2.0, 0.40), (4.0, 0.20))),
    ("2R-25%/5R-25% (50%run)", ((2.0, 0.25), (5.0, 0.25))),
    ("3R-30%/5R-20% (50%run)", ((3.0, 0.30), (5.0, 0.20))),
    ("3R-40%/6R-20% (40%run)", ((3.0, 0.40), (6.0, 0.20))),
]
for label, tp in two_tps:
    r = run(label, tp, {**no_filter})
    row(r); all_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: THREE TPs + RUNNER (classic scale-out)
# ══════════════════════════════════════════════════════════════════════════════
hdr("THREE TPs + TRAILING RUNNER")
three_tps = [
    ("1R50/2R25/3R15 (10%run)", ((1.0, 0.50), (2.0, 0.25), (3.0, 0.15))),
    ("1R30/2R20/3R10 (40%run)", ((1.0, 0.30), (2.0, 0.20), (3.0, 0.10))),
    ("1R20/2R20/3R20 (40%run)", ((1.0, 0.20), (2.0, 0.20), (3.0, 0.20))),
    ("1R20/2R15/4R15 (50%run)", ((1.0, 0.20), (2.0, 0.15), (4.0, 0.15))),
    ("1R25/3R25/5R10 (40%run)", ((1.0, 0.25), (3.0, 0.25), (5.0, 0.10))),
    ("2R20/3R20/5R10 (50%run)", ((2.0, 0.20), (3.0, 0.20), (5.0, 0.10))),
    ("2R30/4R20/6R10 (40%run)", ((2.0, 0.30), (4.0, 0.20), (6.0, 0.10))),
    ("1.5R30/3R20/5R10 (40%run)", ((1.5, 0.30), (3.0, 0.20), (5.0, 0.10))),
    ("1R15/2R15/3R15 (55%run)", ((1.0, 0.15), (2.0, 0.15), (3.0, 0.15))),
    ("1R10/2R10/4R10 (70%run)", ((1.0, 0.10), (2.0, 0.10), (4.0, 0.10))),
]
for label, tp in three_tps:
    r = run(label, tp, {**no_filter})
    row(r); all_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FOUR TPs + RUNNER
# ══════════════════════════════════════════════════════════════════════════════
hdr("FOUR TPs + TRAILING RUNNER")
four_tps = [
    ("1R20/2R15/3R10/5R5 (50%run)", ((1.0,0.20),(2.0,0.15),(3.0,0.10),(5.0,0.05))),
    ("1R15/2R15/3R10/5R10(50%run)", ((1.0,0.15),(2.0,0.15),(3.0,0.10),(5.0,0.10))),
    ("1R25/2R20/4R10/6R5 (40%run)", ((1.0,0.25),(2.0,0.20),(4.0,0.10),(6.0,0.05))),
    ("2R20/3R15/4R10/6R5 (50%run)", ((2.0,0.20),(3.0,0.15),(4.0,0.10),(6.0,0.05))),
    ("1R10/2R10/3R10/5R10(60%run)", ((1.0,0.10),(2.0,0.10),(3.0,0.10),(5.0,0.10))),
]
for label, tp in four_tps:
    r = run(label, tp, {**no_filter})
    row(r); all_results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
qual = [r for r in all_results if r["trades"] >= 15]

hdr("TOP 20 by RETURN % (min 15 trades)")
for r in sorted(qual, key=lambda x: -x["ret"])[:20]:
    row(r)

hdr("TOP 20 by PF (min 15 trades)")
for r in sorted(qual, key=lambda x: -x["pf"])[:20]:
    row(r)

hdr("TOP 20 by R:R (min 15 trades)")
for r in sorted(qual, key=lambda x: -x["wl"])[:20]:
    row(r)

# Expected $ per 10 trades
hdr("TOP 20 by EXPECTED $/10 trades (min 15 trades)")
for r in sorted(qual, key=lambda x: -(x["wr"]/100*x["avg_w"] - (1-x["wr"]/100)*x["avg_l"])*10)[:20]:
    ev = (r["wr"]/100*r["avg_w"] - (1-r["wr"]/100)*r["avg_l"])*10
    pf_s = f"{r['pf']:>5.2f}" if r['pf'] < 100 else "  inf"
    print(f"{r['label']:<35} {r['trades']:>3} {r['wr']:>5.1f} {pf_s} ${r['bal']:>7.0f} {r['ret']:>5.0f}% "
          f"${r['avg_w']:>6.0f} ${r['avg_l']:>6.0f} ${r['max_w']:>6.0f} {r['wl']:>5.2f} EV=${ev:>5.0f}")
