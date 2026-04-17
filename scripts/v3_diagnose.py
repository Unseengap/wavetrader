"""Diagnose why profits are so small."""
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

# Run with geo trail=0.2, no filters (best PF config)
params = {"exit_mode": "geometric_trail", "trailing_stop_pct": 0.2,
          "use_daily_ema_filter": False, "use_rsi_filter": False, "use_adx_filter": False,
          "max_sl_pips": 200.0}

bt = BacktestConfig(initial_balance=100.0, risk_per_trade=0.10)
strategy = PriceActionReversalStrategy(params=params)
r = run_strategy_backtest(strategy=strategy, candles=df_dict, bt_config=bt, pair='GBP/JPY', verbose=False)

# Show lot sizes, balance at time of trade, and PnL
print(f"{'#':>3} {'Dir':>4} {'Balance':>8} {'LotSize':>8} {'SLpips':>6} {'PnL':>8} {'Reason':<20}")
print("-" * 70)
for idx, t in enumerate(r.trades):
    pip = 0.01
    if t.direction.value == 1:
        sl_pips = (t.entry_price - t.stop_loss) / pip
    else:
        sl_pips = (t.stop_loss - t.entry_price) / pip
    d = "BUY" if t.direction.value == 1 else "SELL"
    print(f"{idx+1:>3} {d:>4} ${t.entry_price:>7.3f} {t.size:>8.2f} {sl_pips:>6.0f} ${t.pnl:>+7.2f} {t.exit_reason:<20}")

# Check what balance was at each trade open
# Reconstruct
print(f"\n=== BALANCE PROGRESSION ===")
bal = 100.0
for idx, t in enumerate(r.trades):
    risk_pct = 0.10
    # Check if drawdown was active
    peak = max(100.0, bal)
    dd = (peak - bal) / peak
    if dd >= 0.05:
        risk_pct = 0.05  # halved!
    risk_amt = bal * risk_pct
    pip = 0.01
    if t.direction.value == 1:
        sl_pips = (t.entry_price - t.stop_loss) / pip
    else:
        sl_pips = (t.stop_loss - t.entry_price) / pip
    ideal_lot = risk_amt / max(sl_pips * 6.5, 0.001)
    # Margin cap
    max_margin_lot = (bal * 0.5 * 30.0) / 100_000
    actual_lot = min(ideal_lot, max_margin_lot)
    actual_lot = max(0.01, min(5.0, actual_lot))
    
    dd_flag = " DD-HALVED" if dd >= 0.05 else ""
    margin_flag = " MARGIN-CAP" if ideal_lot > max_margin_lot else ""
    
    print(f"  T{idx+1:>2}: bal=${bal:>7.2f} risk=${risk_amt:>5.2f} "
          f"ideal_lot={ideal_lot:.4f} margin_max={max_margin_lot:.4f} "
          f"used={t.size:.2f} PnL=${t.pnl:>+7.2f}{dd_flag}{margin_flag}")
    bal += t.pnl

print(f"\nFinal balance: ${bal:.2f}")
print(f"trades where margin capped: counting...")

# Count how many were margin-capped
bal = 100.0
margin_capped = 0
dd_halved = 0
for t in r.trades:
    peak = max(100.0, bal)
    dd = (peak - bal) / peak
    risk_pct = 0.05 if dd >= 0.05 else 0.10
    if dd >= 0.05:
        dd_halved += 1
    risk_amt = bal * risk_pct
    pip = 0.01
    if t.direction.value == 1:
        sl_pips = (t.entry_price - t.stop_loss) / pip
    else:
        sl_pips = (t.stop_loss - t.entry_price) / pip
    ideal_lot = risk_amt / max(sl_pips * 6.5, 0.001)
    max_margin_lot = (bal * 0.5 * 30.0) / 100_000
    if ideal_lot > max_margin_lot:
        margin_capped += 1
    bal += t.pnl

print(f"Margin-capped trades: {margin_capped}/{len(r.trades)}")
print(f"Drawdown-halved trades: {dd_halved}/{len(r.trades)}")
