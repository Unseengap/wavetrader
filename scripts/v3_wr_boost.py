"""Win rate optimization — test each lever independently against $646 baseline."""
import pandas as pd
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.strategies.price_action_reversal import PriceActionReversalStrategy
from wavetrader.config import BacktestConfig

df_dict = {}
for tf in ['4h', '1d']:
    df = pd.read_parquet(f'processed_data/test/GBPJPY_{tf}.parquet')
    if df.index.name in ('timestamp','date','datetime'): df = df.reset_index()
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {f'{b}_mid': b for b in ('open','high','low','close') if f'{b}_mid' in df.columns and b not in df.columns}
    if col_map: df = df.rename(columns=col_map)
    if 'date' not in df.columns:
        for a in ('datetime','timestamp','time'):
            if a in df.columns: df = df.rename(columns={a:'date'}); break
    df_dict[tf] = df

def oanda_bt():
    return BacktestConfig(initial_balance=100.0, risk_per_trade=0.10, leverage=20.0,
                          spread_pips=4.2, commission_per_lot=0.0, pip_value=6.50,
                          drawdown_reduce_threshold=0.15, margin_use_limit=0.90,
                          multi_tp_levels=((3.0, 0.70),))

# Base params: filters off, 3R-70%
BASE = {"use_daily_ema_filter": False, "use_rsi_filter": False, "use_adx_filter": False,
        "max_sl_pips": 200.0, "exit_mode": "multi_tp_trail"}

def test(label, overrides):
    params = {**BASE, **overrides}
    s = PriceActionReversalStrategy(params=params)
    r = run_strategy_backtest(strategy=s, candles=df_dict, bt_config=oanda_bt(), pair='GBP/JPY', verbose=False)
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    avg_w = sum(t.pnl for t in wins)/len(wins) if wins else 0
    avg_l = abs(sum(t.pnl for t in losses)/len(losses)) if losses else 0
    max_w = max((t.pnl for t in r.trades), default=0)
    wl = avg_w/avg_l if avg_l > 0 else 0

    # Entry type breakdown
    types = {}
    for t in r.trades:
        et = t.context.get('entry_type', '?') if hasattr(t, 'context') and t.context else '?'
        if et not in types: types[et] = {'w': 0, 'l': 0}
        if t.pnl > 0: types[et]['w'] += 1
        else: types[et]['l'] += 1

    return {"label": label, "trades": r.total_trades, "wr": r.win_rate*100, "pf": min(r.profit_factor, 999),
            "bal": r.final_balance, "avg_w": avg_w, "avg_l": avg_l, "max_w": max_w, "wl": wl, "types": types}

def row(r, show_types=False):
    pf_s = f"{r['pf']:>5.2f}" if r['pf'] < 100 else "  inf"
    delta = "  BASE" if r['label'] == 'BASELINE' else f"  {'+' if r['bal'] >= 646 else ''}{r['bal']-646:+.0f}"
    print(f"{r['label']:<40} {r['trades']:>3} {r['wr']:>5.1f}% {pf_s} ${r['bal']:>7.0f}{delta:>8} "
          f"${r['avg_w']:>6.0f} ${r['avg_l']:>6.0f} {r['wl']:>5.2f}")
    if show_types and r['types']:
        for et, wl in sorted(r['types'].items()):
            tot = wl['w'] + wl['l']
            wr = wl['w']/tot*100 if tot > 0 else 0
            print(f"  └─ {et:<30} {tot:>2}T  {wr:>4.0f}% WR  ({wl['w']}W/{wl['l']}L)")

def hdr(title):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"{'Config':<40} {'Tr':>3} {'WR':>6} {'PF':>5} {'EndBal':>8} {'vs646':>8} {'AvgW':>7} {'AvgL':>7} {'R:R':>5}")
    print(f"{'-'*110}")

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════════════════════
hdr("BASELINE + ENTRY TYPE BREAKDOWN")
bl = test("BASELINE", {})
row(bl, show_types=True)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 1: Confirmation strength (confirm_body_atr_min)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 1: Confirmation Bar Strength")
for v in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    r = test(f"confirm_body_atr_min={v}", {"confirm_body_atr_min": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 2: Wick rejection quality (min_wick_atr_ratio)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 2: Wick Rejection Quality")
for v in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
    r = test(f"min_wick_atr_ratio={v}", {"min_wick_atr_ratio": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 3: Zone age (min_zone_age)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 3: Minimum Zone Age (bars)")
for v in [3, 5, 7, 10, 15, 20]:
    r = test(f"min_zone_age={v}", {"min_zone_age": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 4: Zone must be tested before entry
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 4: Zone Test Count Required")
for v in [0, 1, 2]:
    r = test(f"min_zone_tests={v}", {"min_zone_tests": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 5: Swing lookback (bigger = more major zones)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 5: Swing Lookback")
for v in [5, 7, 9, 11, 14]:
    r = test(f"swing_lookback={v}", {"swing_lookback": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 6: Zone tolerance (how close price must be to zone)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 6: Zone Tolerance %")
for v in [0.001, 0.0015, 0.002, 0.003, 0.004]:
    r = test(f"zone_tol={v}", {"zone_tolerance_pct": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 7: Disable specific entry types (find which hurt WR)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 7: Entry Type Selection")
for off_type, params in [
    ("No reversal", {"enable_reversal": False}),
    ("No break_retest", {"enable_break_retest": False}),
    ("No support_bounce", {"enable_support_bounce": False}),
    ("No resist_reject", {"enable_resistance_reject": False}),
    ("Only support+resist", {"enable_break_retest": False, "enable_reversal": False}),
    ("Only break+retest", {"enable_support_bounce": False, "enable_resistance_reject": False, "enable_reversal": False}),
    ("Only reversal", {"enable_support_bounce": False, "enable_resistance_reject": False, "enable_break_retest": False}),
]:
    r = test(off_type, params)
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 8: Min SL pips (skip setups with tiny SL — usually false signals)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 8: Minimum SL pips")
for v in [10, 15, 20, 25, 30, 40]:
    r = test(f"min_sl_pips={v}", {"min_sl_pips": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 9: Max SL pips (cap worst-case losses)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 9: Max SL Pips Cap")
for v in [60, 80, 100, 120, 150, 200]:
    r = test(f"max_sl_pips={v}", {"max_sl_pips": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 10: Min reversal candles (more extreme = stronger reversal)
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 10: Min Trend Candles for Reversal")
for v in [3, 5, 7, 9, 12]:
    r = test(f"min_trend_candles={v}", {"min_trend_candles_reversal": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 11: Trending requirement for S/R entries
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 11: Trend Alignment")
for v in [0.05, 0.10, 0.15, 0.20, 0.30]:
    r = test(f"trend_bias_thr={v}", {"trend_bias_threshold": v})
    row(r)
r = test(f"no trend required", {"require_trend_alignment": False})
row(r)

# ══════════════════════════════════════════════════════════════════════════════
# LEVER 12: Min confidence threshold
# ══════════════════════════════════════════════════════════════════════════════
hdr("LEVER 12: Min Confidence")
for v in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
    r = test(f"min_confidence={v}", {"min_confidence": v})
    row(r)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY: Collect all results sorted by balance (only if WR > baseline)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  ALL CONFIGS WITH WR > {bl['wr']:.0f}% — sorted by balance")
print(f"{'='*110}")
print(f"{'Config':<40} {'Tr':>3} {'WR':>6} {'PF':>5} {'EndBal':>8} {'vs646':>8} {'AvgW':>7} {'AvgL':>7} {'R:R':>5}")
print(f"{'-'*110}")
