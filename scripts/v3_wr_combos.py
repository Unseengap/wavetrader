"""Combine the WR-boosting levers that individually helped."""
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

def bt():
    return BacktestConfig(initial_balance=100.0, risk_per_trade=0.10, leverage=20.0,
                          spread_pips=4.2, commission_per_lot=0.0, pip_value=6.50,
                          drawdown_reduce_threshold=0.15, margin_use_limit=0.90,
                          multi_tp_levels=((3.0, 0.70),))

BASE = {"use_daily_ema_filter": False, "use_rsi_filter": False, "use_adx_filter": False,
        "max_sl_pips": 200.0, "exit_mode": "multi_tp_trail"}

def test(label, overrides):
    params = {**BASE, **overrides}
    s = PriceActionReversalStrategy(params=params)
    r = run_strategy_backtest(strategy=s, candles=df_dict, bt_config=bt(), pair='GBP/JPY', verbose=False)
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    avg_w = sum(t.pnl for t in wins)/len(wins) if wins else 0
    avg_l = abs(sum(t.pnl for t in losses)/len(losses)) if losses else 0
    max_w = max((t.pnl for t in r.trades), default=0)
    wl = avg_w/avg_l if avg_l > 0 else 0
    return {"label": label, "trades": r.total_trades, "wr": r.win_rate*100, "pf": min(r.profit_factor,999),
            "bal": r.final_balance, "avg_w": avg_w, "avg_l": avg_l, "max_w": max_w, "wl": wl}

def row(r):
    pf_s = f"{r['pf']:>5.2f}" if r['pf'] < 100 else "  inf"
    print(f"{r['label']:<50} {r['trades']:>3} {r['wr']:>5.1f}% {pf_s} ${r['bal']:>7.0f} "
          f"${r['avg_w']:>6.0f} ${r['avg_l']:>6.0f} ${r['max_w']:>6.0f} {r['wl']:>5.2f}")

# Winners from individual tests:
# confirm_body_atr_min=0.3  → 57T, 31.6%, $934 (+$288!)
# max_sl_pips=80            → 47T, 36.2%, $849 (+$203!)
# min_trend_candles=3       → 45T, 35.6%, $768 (+$122!)
# swing_lookback=5          → 45T, 35.6%, $730 (+$84)
# min_wick_atr_ratio=0.4    → 53T, 32.1%, $787 (+$141)
# min_wick_atr_ratio=1.0    → 44T, 34.1%, $645 (+PF 1.84)
# swing_lookback=14         → 25T, 44.0%, $515 (PF 3.05!)

print(f"{'Config':<50} {'Tr':>3} {'WR':>6} {'PF':>5} {'EndBal':>8} {'AvgW':>7} {'AvgL':>7} {'MaxW':>7} {'R:R':>5}")
print("=" * 115)

# Baseline
row(test("BASELINE", {}))

# Individual winners
row(test("A: confirm=0.3", {"confirm_body_atr_min": 0.3}))
row(test("B: maxSL=80", {"max_sl_pips": 80.0}))
row(test("C: min_trend=3", {"min_trend_candles_reversal": 3}))
row(test("D: swing=5", {"swing_lookback": 5}))

# 2-way combos
row(test("A+B: confirm=0.3 maxSL=80", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0}))
row(test("A+C: confirm=0.3 trend=3", {"confirm_body_atr_min": 0.3, "min_trend_candles_reversal": 3}))
row(test("A+D: confirm=0.3 swing=5", {"confirm_body_atr_min": 0.3, "swing_lookback": 5}))
row(test("B+C: maxSL=80 trend=3", {"max_sl_pips": 80.0, "min_trend_candles_reversal": 3}))
row(test("B+D: maxSL=80 swing=5", {"max_sl_pips": 80.0, "swing_lookback": 5}))
row(test("C+D: trend=3 swing=5", {"min_trend_candles_reversal": 3, "swing_lookback": 5}))

# 3-way combos
row(test("A+B+C: conf=0.3 sl=80 trend=3", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0, "min_trend_candles_reversal": 3}))
row(test("A+B+D: conf=0.3 sl=80 swing=5", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0, "swing_lookback": 5}))
row(test("A+C+D: conf=0.3 trend=3 swing=5", {"confirm_body_atr_min": 0.3, "min_trend_candles_reversal": 3, "swing_lookback": 5}))
row(test("B+C+D: sl=80 trend=3 swing=5", {"max_sl_pips": 80.0, "min_trend_candles_reversal": 3, "swing_lookback": 5}))

# 4-way: all winners combined
row(test("A+B+C+D: ALL FOUR", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0, "min_trend_candles_reversal": 3, "swing_lookback": 5}))

# Best combos + wick tweaks
row(test("A+B+C+D + wick=0.5", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0, "min_trend_candles_reversal": 3, "swing_lookback": 5, "min_wick_atr_ratio": 0.5}))
row(test("A+B + wick=0.5", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0, "min_wick_atr_ratio": 0.5}))
row(test("A+B + wick=0.4", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0, "min_wick_atr_ratio": 0.4}))

# Fine-tune confirm around 0.3
row(test("conf=0.25 sl=80", {"confirm_body_atr_min": 0.25, "max_sl_pips": 80.0}))
row(test("conf=0.30 sl=80", {"confirm_body_atr_min": 0.30, "max_sl_pips": 80.0}))
row(test("conf=0.35 sl=80", {"confirm_body_atr_min": 0.35, "max_sl_pips": 80.0}))

# Fine-tune maxSL with confirm=0.3
row(test("conf=0.3 sl=70", {"confirm_body_atr_min": 0.3, "max_sl_pips": 70.0}))
row(test("conf=0.3 sl=75", {"confirm_body_atr_min": 0.3, "max_sl_pips": 75.0}))
row(test("conf=0.3 sl=80", {"confirm_body_atr_min": 0.3, "max_sl_pips": 80.0}))
row(test("conf=0.3 sl=85", {"confirm_body_atr_min": 0.3, "max_sl_pips": 85.0}))
row(test("conf=0.3 sl=90", {"confirm_body_atr_min": 0.3, "max_sl_pips": 90.0}))
