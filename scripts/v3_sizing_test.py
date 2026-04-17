"""Test with proper sizing — fix the margin cap problem."""
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

# Top configs to retest with proper sizing
configs = [
    ("geo trail=0.2", None,
     {"exit_mode":"geometric_trail","trailing_stop_pct":0.2,
      "use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("3R-50%", ((3.0,0.50),),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("1R-30% (70%run)", ((1.0,0.30),),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("EMA+geo trail=0.2", None,
     {"exit_mode":"geometric_trail","trailing_stop_pct":0.2,
      "use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("2R-40%/4R-20%", ((2.0,0.40),(4.0,0.20)),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("baseline 1R50/2R25/3R15", ((1.0,.50),(2.0,.25),(3.0,.15)),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
]

# Sizing scenarios
sizing = [
    ("$100 / 30x lev (CURRENT)", 100.0, 30.0),
    ("$100 / 100x lev", 100.0, 100.0),
    ("$100 / 500x lev", 100.0, 500.0),
    ("$1000 / 30x lev", 1000.0, 30.0),
    ("$10K / 30x lev", 10000.0, 30.0),
]

for sz_label, init_bal, lev in sizing:
    print(f"\n{'='*115}")
    print(f"  {sz_label}  —  10% risk per trade")
    print(f"{'='*115}")
    print(f"{'Config':<25} {'Tr':>3} {'WR%':>5} {'PF':>5} {'StartBal':>9} {'EndBal':>9} {'Return':>7} {'AvgW':>8} {'AvgL':>8} {'MaxW':>8} {'R:R':>5}")
    print(f"{'-'*115}")

    for label, tp_levels, params in configs:
        if tp_levels is not None:
            bt = BacktestConfig(initial_balance=init_bal, risk_per_trade=0.10,
                                leverage=lev, multi_tp_levels=tp_levels)
            params["exit_mode"] = "multi_tp_trail"
        else:
            bt = BacktestConfig(initial_balance=init_bal, risk_per_trade=0.10,
                                leverage=lev)
        
        strategy = PriceActionReversalStrategy(params=params)
        r = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt, pair='GBP/JPY', verbose=False)
        
        wins = [t for t in r.trades if t.pnl > 0]
        losses = [t for t in r.trades if t.pnl <= 0]
        avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_l = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
        max_w = max((t.pnl for t in r.trades), default=0)
        wl = round(avg_w / avg_l, 2) if avg_l > 0 else 0
        pf = r.profit_factor if r.profit_factor < 999 else 999
        ret_pct = (r.final_balance - init_bal) / init_bal * 100
        
        print(f"{label:<25} {r.total_trades:>3} {r.win_rate*100:>5.1f} {pf:>5.2f} "
              f"${init_bal:>8.0f} ${r.final_balance:>8.0f} {ret_pct:>6.0f}% "
              f"${avg_w:>7.0f} ${avg_l:>7.0f} ${max_w:>7.0f} {wl:>5.2f}")
