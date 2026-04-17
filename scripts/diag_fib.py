"""Quick diagnostic for fib_scalper BOS + golden zone detection."""
import pandas as pd, numpy as np
from pathlib import Path

p = Path('processed_data/test/GBPJPY_15m.parquet')
df = pd.read_parquet(p)
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

print(f'Bars: {len(df)}')

from wavetrader.strategies.fib_scalper import FibScalperStrategy
from wavetrader.indicators import calculate_atr

strat = FibScalperStrategy(params={
    'swing_lookback': 3,
    'min_impulse_atr': 1.0,
    'max_swing_age': 120,
    'require_confirmation': False,
    'max_active_setups': 5,
})
strat.reset()

atr_arr = calculate_atr(
    df['high'].values.astype(np.float64),
    df['low'].values.astype(np.float64),
    df['close'].values.astype(np.float64),
)
pip = 0.01
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values

n_bos = 0
n_golden = 0

for i in range(200, min(5000, len(df))):
    close = float(closes[i])
    atr = float(atr_arr[i]) if i < len(atr_arr) else 0
    if np.isnan(atr) or atr <= 0:
        continue
    strat._detect_swings(highs, lows, i)
    prev = strat._prev_close
    strat._prev_close = close
    bos = strat._detect_bos(close, prev, atr, pip, i)
    if bos is not None:
        n_bos += 1
        if len(strat._active_setups) < 5:
            strat._active_setups.append(bos)
        if n_bos <= 8:
            print(f'BOS @{i}: {bos.direction} imp={bos.impulse_pips:.1f}p')
    triggered = strat._check_golden_zone_entry(
        close, float(highs[i]), float(lows[i]),
        float(df.iloc[i]["open"]), i, atr,
    )
    if triggered is not None:
        n_golden += 1
        if n_golden <= 8:
            zone_lo = min(triggered.fib_50, triggered.fib_618)
            zone_hi = max(triggered.fib_50, triggered.fib_618)
            print(f'  GOLDEN @{i}: {triggered.direction} close={close:.3f} zone=[{zone_lo:.3f},{zone_hi:.3f}]')

print(f'\nSwings: {len(strat._swing_highs)}H {len(strat._swing_lows)}L')
print(f'Broken: {len(strat._broken_high_idxs)}H {len(strat._broken_low_idxs)}L')
print(f'BOS: {n_bos} | Golden: {n_golden}')
