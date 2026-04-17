"""V3 loss analysis — find what's causing big losses."""
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
strategy = PriceActionReversalStrategy()
result = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt_config, pair='GBP/JPY', verbose=False)

r = result
wins = [t for t in r.trades if t.pnl > 0]
losses = [t for t in r.trades if t.pnl <= 0]

print(f"=== CURRENT STATE ===")
print(f"Trades: {r.total_trades}  WR: {r.win_rate*100:.0f}%  PF: {r.profit_factor:.2f}  Bal: ${r.final_balance:.2f}")
avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
avg_l = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
print(f"AvgW: ${avg_w:.2f}  AvgL: ${avg_l:.2f}  MaxW: ${max(t.pnl for t in r.trades):.2f}")

# SL pip distribution for losses
print(f"\n=== LOSS ANALYSIS ({len(losses)} losses) ===")
loss_sl_pips = []
for t in losses:
    pip = 0.01
    if t.direction.value == 1:  # BUY
        sl_pips = (t.entry_price - t.stop_loss) / pip
    else:
        sl_pips = (t.stop_loss - t.entry_price) / pip
    loss_sl_pips.append(sl_pips)
    
print(f"SL pips: avg={sum(loss_sl_pips)/len(loss_sl_pips):.0f}  min={min(loss_sl_pips):.0f}  max={max(loss_sl_pips):.0f}")

# Loss $ distribution
loss_pnls = sorted([t.pnl for t in losses])
print(f"Loss $: avg=${avg_l:.2f}  worst 5: {['${:.2f}'.format(p) for p in loss_pnls[:5]]}")

# Losses by direction
buy_losses = [t for t in losses if t.direction.value == 1]
sell_losses = [t for t in losses if t.direction.value != 1]
buy_loss_avg = abs(sum(t.pnl for t in buy_losses) / len(buy_losses)) if buy_losses else 0
sell_loss_avg = abs(sum(t.pnl for t in sell_losses) / len(sell_losses)) if sell_losses else 0
print(f"BUY losses: {len(buy_losses)} avg=${buy_loss_avg:.2f}  SELL losses: {len(sell_losses)} avg=${sell_loss_avg:.2f}")

# Wins analysis
print(f"\n=== WIN ANALYSIS ({len(wins)} wins) ===")
win_partials = [t.partial_closes for t in wins]
print(f"Partial TPs hit: avg={sum(win_partials)/len(win_partials):.1f}  max={max(win_partials)}")
tp_dist = {}
for p in win_partials:
    tp_dist[p] = tp_dist.get(p, 0) + 1
print(f"TP distribution: {dict(sorted(tp_dist.items()))}")
win_pnls = sorted([t.pnl for t in wins], reverse=True)
print(f"Top 5 wins: {['${:.2f}'.format(p) for p in win_pnls[:5]]}")

# Hold duration: wins vs losses
win_holds = [(t.exit_time - t.entry_time).total_seconds()/3600 for t in wins if t.exit_time and t.entry_time]
loss_holds = [(t.exit_time - t.entry_time).total_seconds()/3600 for t in losses if t.exit_time and t.entry_time]
print(f"\nWin hold: avg={sum(win_holds)/len(win_holds):.0f}h  Loss hold: avg={sum(loss_holds)/len(loss_holds):.0f}h")

# Show the 10 worst trades
print(f"\n=== 10 WORST TRADES ===")
worst = sorted(r.trades, key=lambda t: t.pnl)[:10]
for t in worst:
    d = "BUY " if t.direction.value == 1 else "SELL"
    pip = 0.01
    if t.direction.value == 1:
        sl_pips = (t.entry_price - t.stop_loss) / pip
    else:
        sl_pips = (t.stop_loss - t.entry_price) / pip
    dur = (t.exit_time - t.entry_time).total_seconds()/3600 if t.exit_time and t.entry_time else 0
    print(f"  {d} @ {t.entry_price:.3f}  SL={sl_pips:.0f}p  PnL=${t.pnl:+.2f}  hold={dur:.0f}h  {t.exit_reason}  partials={t.partial_closes}")
