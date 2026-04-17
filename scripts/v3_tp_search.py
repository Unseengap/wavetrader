"""V3 TP-level + filter optimization — find the PF 2.0 combo."""
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

# TP level configs to test: (label, multi_tp_levels, strategy_params)
tp_configs = [
    # Current default
    ("base 1R50/2R25/3R15", ((1.0,0.50),(2.0,0.25),(3.0,0.15)), {}),
    # More runner: close less at early TPs
    ("1R30/2R20 (50%run)", ((1.0,0.30),(2.0,0.20)), {}),
    ("1R25/2R25 (50%run)", ((1.0,0.25),(2.0,0.25)), {}),
    ("1R30/3R20 (50%run)", ((1.0,0.30),(3.0,0.20)), {}),
    # Higher first TP
    ("1.5R40/3R20 (40%run)", ((1.5,0.40),(3.0,0.20)), {}),
    ("2R50 (50%run)", ((2.0,0.50),), {}),
    ("2R40/4R20 (40%run)", ((2.0,0.40),(4.0,0.20)), {}),
    # Conservative: bank early, big runner
    ("0.5R20/1R20 (60%run)", ((0.5,0.20),(1.0,0.20)), {}),
    ("1R20/2R20/3R20 (40%run)", ((1.0,0.20),(2.0,0.20),(3.0,0.20)), {}),
    # With EMA filter
    ("EMA+1R50/2R25/3R15", ((1.0,0.50),(2.0,0.25),(3.0,0.15)), {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
    ("EMA+1R30/2R20 (50%run)", ((1.0,0.30),(2.0,0.20)), {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
    ("EMA+2R50 (50%run)", ((2.0,0.50),), {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
    ("EMA+1.5R40/3R20", ((1.5,0.40),(3.0,0.20)), {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
    ("EMA+2R40/4R20", ((2.0,0.40),(4.0,0.20)), {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
    ("EMA+1R20/2R20/3R20", ((1.0,0.20),(2.0,0.20),(3.0,0.20)), {"use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
    # Baseline no filter no multi-TP (geometric trail only)
    ("geo trail only", None, {"exit_mode": "geometric_trail"}),
    ("EMA+geo trail", None, {"exit_mode": "geometric_trail", "use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False}),
]

print(f"{'Config':<30} {'Tr':>3} {'WR%':>5} {'PF':>5} {'Bal':>8} {'AvgW':>6} {'AvgL':>6} {'MaxW':>6} {'W/L':>5} {'HoldH':>5}")
print("=" * 105)

for label, tp_levels, params in tp_configs:
    if tp_levels is not None:
        bt = BacktestConfig(initial_balance=100.0, risk_per_trade=0.10, multi_tp_levels=tp_levels)
        if "exit_mode" not in params:
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
    holds = [(t.exit_time - t.entry_time).total_seconds()/3600 for t in r.trades if t.exit_time and t.entry_time]
    avg_hold = sum(holds) / len(holds) if holds else 0
    
    pf_str = f"{r.profit_factor:>5.2f}" if r.profit_factor < 100 else " inf"
    print(f"{label:<30} {r.total_trades:>3} {r.win_rate*100:>5.1f} {pf_str} ${r.final_balance:>7.2f} ${avg_w:>5.02f} ${avg_l:>5.02f} ${max_w:>5.02f} {wl:>5.02f} {avg_hold:>5.0f}")
