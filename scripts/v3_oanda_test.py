"""Test with OANDA-correct params — leverage 20:1, spread 4.2, no commission."""
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

# Top strategies to test
strats = [
    ("geo trail=0.2", None,
     {"exit_mode":"geometric_trail","trailing_stop_pct":0.2,
      "use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("3R-50%", ((3.0,0.50),),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("1R-30% (70%run)", ((1.0,0.30),),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("2R-40%/4R-20%", ((2.0,0.40),(4.0,0.20)),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("EMA+geo trail=0.2", None,
     {"exit_mode":"geometric_trail","trailing_stop_pct":0.2,
      "use_daily_ema_filter":True,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
    ("baseline 1R50/2R25/3R15", ((1.0,.50),(2.0,.25),(3.0,.15)),
     {"use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}),
]

# OANDA defaults: 20:1 leverage, 4.2 spread, $0 commission, pip_value 6.50
# Test with different starting balances
balances = [100.0, 500.0, 1000.0, 5000.0, 10000.0]

for init_bal in balances:
    print(f"\n{'='*120}")
    print(f"  OANDA | Starting Balance: ${init_bal:,.0f} | 10% risk | 20:1 leverage | 4.2 pip spread | $0 commission")
    print(f"{'='*120}")
    print(f"{'Config':<25} {'Tr':>3} {'WR%':>5} {'PF':>5} {'EndBal':>10} {'Return':>7} {'AvgW':>8} {'AvgL':>8} {'MaxW':>8} {'R:R':>5}")
    print(f"{'-'*120}")

    for label, tp_levels, params in strats:
        if tp_levels is not None:
            bt = BacktestConfig(
                initial_balance=init_bal, risk_per_trade=0.10,
                leverage=20.0, spread_pips=4.2, commission_per_lot=0.0,
                pip_value=6.50, multi_tp_levels=tp_levels,
                drawdown_reduce_threshold=0.15, margin_use_limit=0.90,
            )
            params["exit_mode"] = "multi_tp_trail"
        else:
            bt = BacktestConfig(
                initial_balance=init_bal, risk_per_trade=0.10,
                leverage=20.0, spread_pips=4.2, commission_per_lot=0.0,
                pip_value=6.50,
                drawdown_reduce_threshold=0.15, margin_use_limit=0.90,
            )

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
              f"${r.final_balance:>9,.0f} {ret_pct:>6.0f}% "
              f"${avg_w:>7,.0f} ${avg_l:>7,.0f} ${max_w:>7,.0f} {wl:>5.2f}")

# Detailed trade-level check for $1000 geo=0.2
print(f"\n{'='*120}")
print(f"  DETAILED: $1,000 | geo trail=0.2 | First 10 trades")
print(f"{'='*120}")
bt = BacktestConfig(
    initial_balance=1000.0, risk_per_trade=0.10,
    leverage=20.0, spread_pips=4.2, commission_per_lot=0.0, pip_value=6.50,
    drawdown_reduce_threshold=0.15, margin_use_limit=0.90,
)
params = {"exit_mode":"geometric_trail","trailing_stop_pct":0.2,
          "use_daily_ema_filter":False,"use_rsi_filter":False,"use_adx_filter":False,"max_sl_pips":200.0}
strategy = PriceActionReversalStrategy(params=params)
r = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt, pair='GBP/JPY', verbose=False)

bal = 1000.0
print(f"{'#':>3} {'Dir':>4} {'Bal':>10} {'Lots':>7} {'SLpips':>6} {'Risk$':>7} {'PnL':>8} {'Reason':<20}")
for idx, t in enumerate(r.trades[:15]):
    pip = 0.01
    sl_pips = abs(t.entry_price - t.stop_loss) / pip
    d = "BUY" if t.direction.value == 1 else "SELL"
    risk = t.size * sl_pips * 6.50
    print(f"{idx+1:>3} {d:>4} ${bal:>9,.2f} {t.size:>7.3f} {sl_pips:>6.0f} ${risk:>6.0f} ${t.pnl:>+7.0f} {t.exit_reason:<20}")
    bal += t.pnl
