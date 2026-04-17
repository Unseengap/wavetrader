"""V3 multi-TP level optimization."""
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

# Different TP level configurations to test
tp_configs = [
    ("geometric (old)", None, 0.3),    # No multi-TP
    # Standard 1/2/3R
    ("1R/2R/3R (25% each)", ((1.0, 0.25), (2.0, 0.25), (3.0, 0.25)), 1.0),
    # Tighter TPs - more frequent partials
    ("0.5R/1R/1.5R (25%)", ((0.5, 0.25), (1.0, 0.25), (1.5, 0.25)), 1.0),
    ("0.5R/1R/2R (25%)", ((0.5, 0.25), (1.0, 0.25), (2.0, 0.25)), 1.0),
    # Wider TPs - bigger moves before partials
    ("2R/4R/6R (25%)", ((2.0, 0.25), (4.0, 0.25), (6.0, 0.25)), 1.5),
    ("1R/3R/5R (25%)", ((1.0, 0.25), (3.0, 0.25), (5.0, 0.25)), 1.5),
    # 2 TPs + bigger runner
    ("1R/2R (33% each)", ((1.0, 0.33), (2.0, 0.33)), 1.0),
    ("1R/3R (33% each)", ((1.0, 0.33), (3.0, 0.33)), 1.5),
    ("2R/4R (33% each)", ((2.0, 0.33), (4.0, 0.33)), 1.5),
    # 1 TP + huge runner
    ("1R only (50%)", ((1.0, 0.50),), 1.0),
    ("2R only (50%)", ((2.0, 0.50),), 1.0),
    ("1R only (25%)", ((1.0, 0.25),), 1.0),
    # Heavy front-load
    ("1R(50%)/2R(25%)/3R(15%)", ((1.0, 0.50), (2.0, 0.25), (3.0, 0.15)), 1.0),
]

print(f"{'Config':30s}  {'T':>3s}  {'WR':>5s}  {'PF':>5s}  {'Bal':>7s}  {'AvgW':>6s}  {'AvgL':>6s}  {'MaxW':>6s}  {'HoldD':>5s}")
print("-" * 95)

for name, tp_levels, trail_pct in tp_configs:
    if tp_levels is None:
        # Geometric trail mode
        strat_params = {"exit_mode": "geometric_trail", "trailing_stop_pct": 0.3}
        bt_config = BacktestConfig(initial_balance=100.0, risk_per_trade=0.10)
    else:
        strat_params = {"exit_mode": "multi_tp_trail", "trailing_stop_pct": trail_pct}
        bt_config = BacktestConfig(initial_balance=100.0, risk_per_trade=0.10, multi_tp_levels=tp_levels)

    strategy = PriceActionReversalStrategy(strat_params)
    result = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt_config, pair='GBP/JPY', verbose=False)
    r = result
    wins = [t for t in r.trades if t.pnl > 0]
    losses = [t for t in r.trades if t.pnl <= 0]
    avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_l = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
    max_w = max((t.pnl for t in r.trades), default=0)

    win_durs = []
    for t in r.trades:
        if t.exit_time and t.entry_time and t.pnl > 0:
            win_durs.append((t.exit_time - t.entry_time).total_seconds() / 86400)
    avg_hold = sum(win_durs) / len(win_durs) if win_durs else 0

    partials = [t.partial_closes for t in r.trades]
    avg_p = sum(partials) / len(partials) if partials else 0
    n_tp = len(tp_levels) if tp_levels else 0

    print(f"{name:30s}  {r.total_trades:3d}  {r.win_rate*100:4.0f}%  {r.profit_factor:5.2f}  ${r.final_balance:6.2f}  ${avg_w:5.2f}  ${avg_l:5.2f}  ${max_w:5.2f}  {avg_hold:4.1f}d")
