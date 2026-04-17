"""Diagnostic: trace where fib_scalper signals are filtered out, bar-by-bar."""
import time, sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

processed_dir = Path("processed_data/test")
pair_tag = "GBPJPY"

# ── Load data ──
print("Loading data...")
t0 = time.time()
df_1m = pd.read_parquet(processed_dir / f"{pair_tag}_1m.parquet")
if df_1m.index.name in ("timestamp", "date", "datetime"):
    df_1m = df_1m.reset_index()
df_1m.columns = [c.strip().lower() for c in df_1m.columns]
col_map = {}
for base in ("open", "high", "low", "close"):
    if f"{base}_mid" in df_1m.columns and base not in df_1m.columns:
        col_map[f"{base}_mid"] = base
if col_map:
    df_1m = df_1m.rename(columns=col_map)
if "date" not in df_1m.columns:
    for alias in ("datetime", "timestamp", "time"):
        if alias in df_1m.columns:
            df_1m = df_1m.rename(columns={alias: "date"})
            break

MAX_BARS = 200_000
if len(df_1m) > MAX_BARS:
    df_1m = df_1m.iloc[:MAX_BARS].copy()
print(f"  1min: {len(df_1m)} bars ({time.time()-t0:.1f}s)")

# Load HTF
df_dict = {"1min": df_1m}
for tf in ["15min", "1h", "4h", "1d"]:
    tf_short = tf.replace("min", "m")
    p = processed_dir / f"{pair_tag}_{tf_short}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if df.index.name in ("timestamp", "date", "datetime"):
            df = df.reset_index()
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {}
        for base in ("open", "high", "low", "close"):
            if f"{base}_mid" in df.columns and base not in df.columns:
                col_map[f"{base}_mid"] = base
        if col_map:
            df = df.rename(columns=col_map)
        if "date" not in df.columns:
            for alias in ("datetime", "timestamp", "time"):
                if alias in df.columns:
                    df = df.rename(columns={alias: "date"})
                    break
        df_dict[tf] = df
        print(f"  {tf}: {len(df)} bars")

print("\nComputing indicators...")
t0 = time.time()
from wavetrader.strategies.indicators import compute_all_indicators, _ema
from wavetrader.indicators import calculate_rsi, calculate_atr

htf_candles = {k: v for k, v in df_dict.items() if k != "1min"}
indicators = compute_all_indicators(htf_candles, entry_tf="15min", pair="GBP/JPY", compute_amd=False)

c1m = df_1m["close"].values.astype(np.float64)
h1m = df_1m["high"].values.astype(np.float64)
l1m = df_1m["low"].values.astype(np.float64)
o1m = df_1m["open"].values.astype(np.float64)

indicators.rsi["1min"] = calculate_rsi(c1m)
indicators.atr["1min"] = calculate_atr(h1m, l1m, c1m)
indicators.ema_20["1min"] = _ema(c1m, 20)
indicators.ema_50["1min"] = _ema(c1m, 50)
indicators.ema_200["1min"] = _ema(c1m, 200)
indicators.entry_tf = "1min"

from wavetrader.amd_features import compute_engulfing_patterns
indicators.engulfing = compute_engulfing_patterns(o1m, h1m, l1m, c1m)
print(f"  Done ({time.time()-t0:.1f}s)")

# ── Diagnostics with patched strategy ──
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.types import Signal

reg = get_strategy_registry()
strat = reg.instantiate('fib_scalper', params={
    "use_ema_200_filter": False,
    "use_htf_bias": False,
    "require_confirmation": False,
    "min_confidence": 0.30,
    "min_rr_ratio": 1.0,
})
strat.reset()

pip = 0.01  # JPY pair

bos_count = Counter()
gz_entries = 0
filter_drops = Counter()
rr_values = []
sl_values = []
tp_values = []
conf_values = []

print("\nScanning 200k bars (All OFF config)...")
t0 = time.time()

for i in range(200, len(df_1m)):
    bar = df_1m.iloc[i]
    close = float(bar["close"])
    high = float(bar["high"])
    low = float(bar["low"])
    o = float(bar["open"])

    atr_a = indicators.atr.get("1min")
    if atr_a is None or i >= len(atr_a):
        continue
    atr = float(atr_a[i])
    if np.isnan(atr) or atr <= 0:
        continue

    atr_pips = atr / pip

    # ATR check
    if atr_pips < strat.params["min_atr_pips"]:
        filter_drops["atr_too_low"] += 1
        # Still run swing detection
        strat._detect_swings(df_1m["high"].values, df_1m["low"].values, i)
        bos = strat._detect_bos(close, high, low, atr, pip, i)
        if bos:
            bos_count[bos.direction.name] += 1
            if len(strat._active_setups) >= strat.params["max_active_setups"]:
                strat._active_setups.pop(0)
            strat._active_setups.append(bos)
        continue

    # Session check
    ts = bar.get("date")
    hour = 12
    if ts is not None:
        hour = pd.Timestamp(ts).hour
        if hour < strat.params["session_start_hour"] or hour >= strat.params["session_end_hour"]:
            filter_drops["outside_session"] += 1
            strat._detect_swings(df_1m["high"].values, df_1m["low"].values, i)
            bos = strat._detect_bos(close, high, low, atr, pip, i)
            if bos:
                bos_count[bos.direction.name] += 1
                if len(strat._active_setups) >= strat.params["max_active_setups"]:
                    strat._active_setups.pop(0)
                strat._active_setups.append(bos)
            continue

    # Swing + BOS
    strat._detect_swings(df_1m["high"].values, df_1m["low"].values, i)
    bos = strat._detect_bos(close, high, low, atr, pip, i)
    if bos:
        bos_count[bos.direction.name] += 1
        if len(strat._active_setups) >= strat.params["max_active_setups"]:
            strat._active_setups.pop(0)
        strat._active_setups.append(bos)

    # Golden zone check
    triggered = strat._check_golden_zone_entry(close, high, low, o, i, atr)
    if triggered is None:
        continue

    gz_entries += 1

    # SL/TP computation
    d = triggered.direction
    if d == Signal.BUY:
        sl_dist = close - triggered.sl_level
        tp_dist = triggered.tp_level - close
    else:
        sl_dist = triggered.sl_level - close
        tp_dist = close - triggered.tp_level

    if sl_dist <= 0 or tp_dist <= 0:
        filter_drops["sl_tp_invalid"] += 1
        continue

    sl_pips = sl_dist / pip
    tp_pips = tp_dist / pip
    sl_pips = max(sl_pips, strat.params["min_sl_pips"])
    sl_pips = min(sl_pips, strat.params["max_sl_pips"])
    rr = tp_pips / sl_pips if sl_pips > 0 else 0

    sl_values.append(sl_pips)
    tp_values.append(tp_pips)
    rr_values.append(rr)

    if rr < strat.params["min_rr_ratio"]:
        filter_drops["rr_too_low"] += 1
        continue

    # Confidence (simplified)
    conf = 0.50
    zone_range = abs(triggered.fib_50 - triggered.fib_618)
    if zone_range > 0:
        if d == Signal.BUY:
            depth = (triggered.fib_50 - close) / zone_range
        else:
            depth = (close - triggered.fib_50) / zone_range
        conf += min(max(depth, 0) * 0.10, 0.10)
    in_overlap = strat.params["overlap_start_hour"] <= hour < strat.params["overlap_end_hour"]
    if in_overlap:
        conf += strat.params["overlap_conf_bonus"]
    if triggered.impulse_pips > atr_pips * 2.0:
        conf += 0.06
    if indicators.engulfing is not None and i < len(indicators.engulfing):
        eng = float(indicators.engulfing[i, 0])
        if (d == Signal.BUY and eng > 0) or (d == Signal.SELL and eng < 0):
            conf += 0.07
    conf = min(conf, 0.95)
    conf_values.append(conf)

    if conf < strat.params["min_confidence"]:
        filter_drops["confidence_too_low"] += 1
        continue

    filter_drops["PASSED"] += 1

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

print(f"\n{'='*60}")
print(f"DIAGNOSTIC RESULTS")
print(f"{'='*60}")
print(f"Total swing highs: {len(strat._swing_highs)}")
print(f"Total swing lows:  {len(strat._swing_lows)}")
print(f"Broken high idxs:  {len(strat._broken_high_idxs)}")
print(f"Broken low idxs:   {len(strat._broken_low_idxs)}")
print(f"\nBOS events by direction:")
for k, v in sorted(bos_count.items()):
    print(f"  {k}: {v}")
print(f"Total BOS: {sum(bos_count.values())}")
print(f"\nGolden zone entries (price entered zone): {gz_entries}")
print(f"\nFilter dropout stage:")
for k, v in sorted(filter_drops.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

if sl_values:
    print(f"\nSL distribution (pips): min={min(sl_values):.1f} median={np.median(sl_values):.1f} max={max(sl_values):.1f} mean={np.mean(sl_values):.1f}")
if tp_values:
    print(f"TP distribution (pips): min={min(tp_values):.1f} median={np.median(tp_values):.1f} max={max(tp_values):.1f} mean={np.mean(tp_values):.1f}")
if rr_values:
    print(f"R:R distribution:       min={min(rr_values):.2f} median={np.median(rr_values):.2f} max={max(rr_values):.2f} mean={np.mean(rr_values):.2f}")
if conf_values:
    print(f"Conf distribution:      min={min(conf_values):.2f} median={np.median(conf_values):.2f} max={max(conf_values):.2f}")

# Show last few active setups
if strat._active_setups:
    print(f"\nActive setups remaining: {len(strat._active_setups)}")
    for s in strat._active_setups[-5:]:
        print(f"  {s.direction.name} created@{s.created_idx} fib50={s.fib_50:.3f} fib618={s.fib_618:.3f} tp={s.tp_level:.3f}")
