"""Quick 1min backtest for fib_scalper — uses small slice first."""
import time
import pandas as pd
import numpy as np
from pathlib import Path

processed_dir = Path("processed_data/test")
pair_tag = "GBPJPY"

# Load 1min
print("Loading 1min data...")
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
print(f"  1min: {len(df_1m)} bars ({time.time()-t0:.1f}s)")

# Use first 200k bars (~140 trading days) for speed
MAX_BARS = 200_000
if len(df_1m) > MAX_BARS:
    df_1m = df_1m.iloc[:MAX_BARS].copy()
    print(f"  Trimmed to {MAX_BARS} bars for speed")

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

print(f"\nComputing indicators (skip 1min structure for speed)...")
t0 = time.time()

# Compute indicators manually, skipping expensive classify_structure on 1min
from wavetrader.strategies.indicators import compute_all_indicators
from wavetrader.indicators import calculate_rsi, calculate_atr, calculate_adx

# Compute HTF indicators normally (fast, small data)
htf_candles = {k: v for k, v in df_dict.items() if k != "1min"}
indicators = compute_all_indicators(htf_candles, entry_tf="15min", pair="GBP/JPY", compute_amd=False)

# Add 1min indicators manually (skip structure + ADX which are slow)
c1m = df_1m["close"].values.astype(np.float64)
h1m = df_1m["high"].values.astype(np.float64)
l1m = df_1m["low"].values.astype(np.float64)
o1m = df_1m["open"].values.astype(np.float64)

from wavetrader.strategies.indicators import _ema
indicators.rsi["1min"] = calculate_rsi(c1m)
indicators.atr["1min"] = calculate_atr(h1m, l1m, c1m)
indicators.ema_20["1min"] = _ema(c1m, 20)
indicators.ema_50["1min"] = _ema(c1m, 50)
indicators.ema_200["1min"] = _ema(c1m, 200)
indicators.entry_tf = "1min"

# Engulfing patterns for the entry TF
from wavetrader.amd_features import compute_engulfing_patterns
indicators.engulfing = compute_engulfing_patterns(o1m, h1m, l1m, c1m)

print(f"  Done ({time.time()-t0:.1f}s)")

# Now run strategy manually (bar-by-bar)
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.backtest import BacktestEngine
from wavetrader.config import BacktestConfig
from wavetrader.types import TradeSignal

print(f"\nRunning backtest...")

combos = [
    # ── Confirmed winners from last round ──
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3}, 3.0, "lb20 ovlap BASE"),
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "lb20 ovlap rr2"),

    # ── Session window variations ──
    ({"swing_lookback": 20, "session_start_hour": 12, "session_end_hour": 18, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "lb20 12-18 rr2"),
    ({"swing_lookback": 20, "session_start_hour": 8, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "lb20 8-17 rr2"),
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 20, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "lb20 13-20 rr2"),

    # ── Add EMA200 trend filter to overlap ──
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "ovlap rr2+EMA"),
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "require_confirmation": False, "min_confidence": 0.3}, 3.0, "ovlap+EMA"),

    # ── Add confirmation candle to overlap ──
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "ovlap rr2+cnf"),
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "min_confidence": 0.3}, 3.0, "ovlap+cnf"),

    # ── EMA + confirmation + overlap ──
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "ovlap rr2 FULL"),
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "min_confidence": 0.3}, 3.0, "ovlap FULL"),

    # ── R:R sweep on overlap ──
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.5}, 3.0, "ovlap rr2.5"),
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 3.0}, 3.0, "ovlap rr3"),

    # ── No trailing on overlap winners ──
    ({"swing_lookback": 20, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0, "trailing_stop_pct": 0.0}, 3.0, "ovlap rr2 noT"),

    # ── lb sweep around 20 with overlap ──
    ({"swing_lookback": 15, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "lb15 ovlap rr2"),
    ({"swing_lookback": 25, "session_start_hour": 13, "session_end_hour": 17, "use_ema_200_filter": False, "require_confirmation": False, "min_confidence": 0.3, "min_rr_ratio": 2.0}, 3.0, "lb25 ovlap rr2"),
]

reg = get_strategy_registry()

for idx, (params, act_r, label) in enumerate(combos):
    cfg = BacktestConfig()
    cfg.initial_balance = 100.0
    cfg.risk_per_trade = 0.10
    cfg.trail_activate_r = act_r

    engine = BacktestEngine(cfg)
    strat = reg.instantiate('fib_scalper', params=params)
    strat.reset()
    
    signals = 0
    for i in range(200, len(df_1m)):
        bar = df_1m.iloc[i]
        engine.record_bar(bar["high"], bar["low"])
        
        # Update open trade
        if engine.open_trade is not None:
            ts = bar.get("date", pd.Timestamp.utcnow())
            closed = engine.update_trade(bar["high"], bar["low"], bar["close"], ts)
            if closed is not None:
                continue
        
        if engine.open_trade is None:
            indicators.current_bar_idx = i
            setup = strat.evaluate(df_dict, indicators, i)
            if setup is not None:
                signals += 1
                ts = bar.get("date", pd.Timestamp.utcnow())
                trade_signal = TradeSignal(
                    signal=setup.direction,
                    confidence=setup.confidence,
                    entry_price=setup.entry_price,
                    stop_loss=setup.sl_pips,
                    take_profit=setup.tp_pips,
                    trailing_stop_pct=setup.trailing_stop_pct,
                    timestamp=ts,
                )
                engine.open_position(trade_signal, bar["close"], ts,
                                   current_high=bar["high"], current_low=bar["low"])
    
    trades = engine.closed_trades
    n = len(trades)
    if n > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        total_win = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999
        sl = sum(1 for t in trades if (t.exit_reason or '').strip() == 'Stop Loss')
        tp = sum(1 for t in trades if 'take' in (t.exit_reason or '').lower())
        trail = sum(1 for t in trades if 'trail' in (t.exit_reason or '').lower())
        print(f"{idx+1:>2}. [{label:>25}] {n:>4}tr WR={wins/n:.0%} PF={pf:.2f} "
              f"${engine.balance:>8.2f} SL={sl} TP={tp} Trail={trail} sigs={signals}")
    else:
        print(f"{idx+1:>2}. [{label:>25}] 0 trades (signals={signals})")

print("\nDone!")
