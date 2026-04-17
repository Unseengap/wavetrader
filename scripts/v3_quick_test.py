"""V3 quick test — current defaults with HTF filters."""
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

bt_config = BacktestConfig(
    initial_balance=100.0, risk_per_trade=0.10,
    multi_tp_levels=((1.0, 0.50), (2.0, 0.25), (3.0, 0.15)),
)

# Test a few key configurations
configs = [
    ("ALL filters ON (default)", {}),
    ("NO filters (baseline)", {"use_daily_ema_filter": False, "use_rsi_filter": False, "use_adx_filter": False, "max_sl_pips": 200.0}),
    ("EMA only", {"use_daily_ema_filter": True, "use_rsi_filter": False, "use_adx_filter": False, "max_sl_pips": 200.0}),
    ("EMA + maxSL100", {"use_daily_ema_filter": True, "use_rsi_filter": False, "use_adx_filter": False, "max_sl_pips": 100.0}),
    ("EMA + ADX only", {"use_daily_ema_filter": True, "use_rsi_filter": False, "use_adx_filter": True, "max_sl_pips": 200.0}),
    ("EMA + RSI only", {"use_daily_ema_filter": True, "use_rsi_filter": True, "use_adx_filter": False, "max_sl_pips": 200.0}),
    ("ALL + maxSL80", {"max_sl_pips": 80.0}),
    ("ALL + maxSL60", {"max_sl_pips": 60.0}),
    ("ALL + ADX20", {"min_adx": 20.0}),
    ("ALL + ADX25", {"min_adx": 25.0}),
    ("ALL + RSI25/75", {"rsi_lower": 25.0, "rsi_upper": 75.0}),
]

print(f"{'Config':<30} {'Tr':>3} {'WR%':>5} {'PF':>5} {'Bal':>8} {'AvgW':>6} {'AvgL':>6} {'MaxW':>6} {'W/L':>5}")
print("=" * 95)

for label, overrides in configs:
    strategy = PriceActionReversalStrategy(params=overrides)
    r = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt_config, pair='GBP/JPY', verbose=False)
    
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_l = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
    max_w = max((t.pnl for t in r.trades), default=0)
    wl = round(avg_w / avg_l, 2) if avg_l > 0 else 0
    
    print(f"{label:<30} {r.total_trades:>3} {r.win_rate*100:>5.1f} {r.profit_factor:>5.2f} ${r.final_balance:>7.2f} ${avg_w:>5.02f} ${avg_l:>5.02f} ${max_w:>5.02f} {wl:>5.02f}")
