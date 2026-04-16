---
name: debug-strategy
description: "Debug and fix trading strategy issues in the WaveTrader platform. USE WHEN: a strategy backtest returns bad metrics (low WR, PF < 1.0, 0 trades, 0 trailing exits), the strategy crashes during backtest or live execution, fixing trailing stop not activating, diagnosing why trades are all SL exits, entry signals not firing, indicator errors, or any strategy that was profitable but regressed. Covers: trailing stop diagnosis, entry quality analysis, SL/TP math verification, indicator bundle debugging, session filter issues, OB detection tuning, backtest engine integration bugs."
argument-hint: "Strategy ID and observed symptom (e.g., 'news_catalyst_ob: 0 trailing exits' or 'my_strategy: PF=0.3')"
---

# Debug Trading Strategy — Diagnostic Playbook

Systematic diagnosis and repair of strategy issues.

> **Rule #1**: Always run a no-trailing baseline FIRST. If the strategy is unprofitable with fixed TP, the entry logic is broken — no amount of trailing tuning will fix it.

## When to Use

- Strategy backtest returns PF < 1.0 or win rate < 30%
- Zero trailing stop exits in backtest output
- Strategy produces 0 trades
- Trades all exit via SL (no TP hits)
- Strategy crashes with KeyError, IndexError, or shape mismatch
- Strategy was profitable but regressed after code changes
- Trailing stop appears to tighten SL immediately on entry
- "AI Confirmer" rejects all signals

## Diagnostic Sequence

### Step 1: Quick Baseline Check

Run the strategy with AND without trailing to isolate trailing bugs from entry quality:

```python
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.config import BacktestConfig

reg = get_strategy_registry()

# Baseline: NO trailing, fixed TP
cfg = BacktestConfig()
cfg.initial_balance = 100.0
cfg.risk_pct = 0.10
cfg.trail_activate_r = 3.0  # Won't matter since trailing is off

strat = reg.instantiate('STRATEGY_ID', params={"trailing_stop_pct": 0.0})
results = run_strategy_backtest(strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)
trades = results.trades
n = len(trades)
wins = sum(1 for t in trades if t.pnl > 0)
print(f"NO TRAIL: {n} trades, WR={wins/n:.0%}, ${results.final_balance:.2f}")

# With trailing
strat2 = reg.instantiate('STRATEGY_ID', params={"trailing_stop_pct": 0.50})
results2 = run_strategy_backtest(strat2, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)
trades2 = results2.trades
n2 = len(trades2)
wins2 = sum(1 for t in trades2 if t.pnl > 0)
trail2 = sum(1 for t in trades2 if 'trail' in (t.exit_reason or '').lower())
print(f"TRAIL ON: {n2} trades, WR={wins2/n2:.0%}, Trail={trail2}, ${results2.final_balance:.2f}")
```

### Step 2: Interpret Baseline

| No Trail Result | Trail Result | Diagnosis |
|-----------------|--------------|-----------|
| PF > 1.2, good WR | PF < 1.0, Trail=0 | **Trailing bug** — check `trail_activate_r` |
| PF > 1.2, good WR | PF < 1.0, Trail > 0 but low AvgW | Trail activating too early — raise `trail_activate_r` |
| PF < 1.0 | PF < 1.0 | **Entry logic broken** — skip to Step 4 |
| 0 trades | 0 trades | **No signals firing** — skip to Step 3 |
| > 500 trades, low WR | Same | **Over-trading** — tighten filters |

### Step 3: Zero Trades Diagnosis

Check each filter layer in order:

```python
strat = reg.instantiate('STRATEGY_ID')
p = strat.params

# 1. Does entry TF exist?
print("Entry TF:", strat.meta.entry_timeframe)
print("Available:", list(df_dict.keys()))

# 2. Are bars loaded?
etf = strat.meta.entry_timeframe
print(f"{etf} bars: {len(df_dict.get(etf, []))}")

# 3. Check indicators computed
from wavetrader.strategies.indicators import compute_all_indicators
ind = compute_all_indicators(df_dict, entry_tf=etf, pair='GBP/JPY')
print(f"ATR available: {etf in ind.atr}")
print(f"RSI available: {etf in ind.rsi}")
print(f"EMA20 available: {etf in ind.ema_20}")
print(f"Structure available: {'4h' in ind.structure}")

# 4. Manual bar-by-bar check (first 500 bars after warmup)
strat.reset()
hits = 0
for i in range(200, min(700, len(df_dict[etf]))):
    setup = strat.evaluate(df_dict, ind, i)
    if setup is not None:
        hits += 1
        if hits <= 3:
            print(f"  Bar {i}: {setup.direction} conf={setup.confidence} sl={setup.sl_pips} tp={setup.tp_pips}")
print(f"Signals in first 500 bars: {hits}")
```

**Common causes of 0 trades:**
- `i < 200` check when data has fewer bars
- ATR returns NaN for first 14+ bars
- 4H EMA alignment requires 200 × 16 = 3200 15min bars of history
- Session filter excludes all bars (wrong timezone assumption)
- `min_confidence` set higher than max possible confidence
- Indicators not available for entry TF

### Step 4: Bad Entry Quality (PF < 1.0)

**Test R:R sweep** — if WR scales inversely with R:R as expected, entries have signal:

```python
for rr in [1.0, 1.5, 2.0, 3.0]:
    strat = reg.instantiate('STRATEGY_ID', params={"trailing_stop_pct": 0.0, "min_rr_ratio": rr})
    results = run_strategy_backtest(strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)
    t = results.trades
    n = len(t)
    if n:
        wr = sum(1 for x in t if x.pnl > 0) / n
        tw = sum(x.pnl for x in t if x.pnl > 0)
        tl = abs(sum(x.pnl for x in t if x.pnl <= 0))
        pf = tw / tl if tl else 999
        # Expected Value per trade
        ev = (wr * rr - (1 - wr)) * 1  # normalized
        print(f"R:R={rr}: {n}tr WR={wr:.0%} PF={pf:.2f} EV={ev:.2f}")
```

**Expected pattern for a valid edge:**
- R:R=1.0: WR > 50% (need >50% to be profitable at 1:1)
- R:R=2.0: WR > 35%
- R:R=3.0: WR > 28%

**If WR doesn't scale or all are below random:**
- Entry logic is picking random entries
- Direction logic (macro bias) may be wrong
- OB detection may be creating OBs that don't hold

### Step 5: Trailing Stop Not Activating

Symptoms: Trail > 0% configured, but 0 trailing exits in backtest.

```python
# Check: do trades ever reach the activation threshold?
for t in trades:
    initial_risk = abs(t.entry_price - t.stop_loss)
    if t.direction.name == 'BUY':
        max_r = (t.highest_price - t.entry_price) / max(initial_risk, 1e-9)
    else:
        max_r = (t.entry_price - t.lowest_price) / max(initial_risk, 1e-9)
    if max_r >= 1.0:
        print(f"Trade reached {max_r:.1f}R (activate_r={cfg.trail_activate_r})")
```

**Fix table:**

| Issue | Fix |
|-------|-----|
| No trades reach activate_r threshold | Lower `trail_activate_r` (e.g., 2.0 instead of 3.0) |
| Trades reach threshold but no trail exits | Bug in `backtest.py` update_trade — check `unrealised_r` calc |
| Trail exits but all at ~breakeven | `trailing_stop_pct` too low (trail too tight) — raise to 0.50+ |
| All trades exit at identical price levels | SL/TP math using wrong pip size (0.01 vs 0.0001) |

### Step 6: SL/TP Pip Size Verification

**CRITICAL**: JPY pairs use `pip = 0.01`, USD pairs use `pip = 0.0001`.

```python
# In backtest.py open_position():
pip = 0.01  # GBP/JPY convention
# BUY: sl = entry - signal.stop_loss * pip
# SELL: sl = entry + signal.stop_loss * pip
```

Verify the strategy returns SL in PIPS (not price distance):
```python
# If SL should be 30 pips for GBP/JPY (e.g., 190.00 → 189.70):
sl_pips = 30.0  # ← what strategy returns
# Engine does: sl_price = 190.00 - 30.0 * 0.01 = 189.70 ✓
```

If you see SL prices like `189.99` for a 30-pip intended stop, the strategy is returning price distance instead of pips.

### Step 7: AI Confirmer Issues

If AI confirmer rejects all signals:

```python
# Test without AI first
results_no_ai = run_strategy_backtest(strat, df_dict, bt_config=cfg, ai_confirmer=None)
# Then with AI
from wavetrader.strategies.ai_confirmer import AIConfirmer
ai = AIConfirmer()
results_ai = run_strategy_backtest(strat, df_dict, bt_config=cfg, ai_confirmer=ai)
```

**Known AI confirmer bug** (fixed April 2026): Shape mismatch in `load_state_dict` crashes on model load. Fix: filter incompatible keys before loading:
```python
# In ai_confirmer.py
filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
```

If `ai_confirmer.model is None` (no checkpoint found), `confirm()` now passes through signals unchanged.

### Step 8: Common Indicator Pitfalls

| Indicator | Pitfall | Fix |
|-----------|---------|-----|
| `indicators.atr[tf]` | Returns NaN for first 14 bars | Check `i >= 14` or handle NaN |
| `indicators.structure["4h"]` | 4H index = `i // 16` | Clamp: `min(i // 16, len(arr) - 1)` |
| `indicators.ema_200[tf]` | NaN for first ~200 bars | Strategy skips `i < 200` |
| `indicators.rsi[tf]` | Range 0-100, can be NaN | Always check `not np.isnan(rsi)` |
| `indicators.engulfing` | Entry TF only, shape (n, 3) | Check `i < len(indicators.engulfing)` |

### Step 9: Session Filter Debugging

All timestamps in the data are UTC. Common mistakes:
- London open = 08:00 UTC (not local time)
- NY open = 13:00 UTC (not EST)
- London-NY overlap = 13:00–17:00 UTC
- Asian session = 00:00–08:00 UTC
- Weekend data may exist — filter Saturday/Sunday if needed

```python
# Check hour distribution of entry bars
ts = df_dict['15min']['date']
hours = pd.to_datetime(ts).dt.hour
print("Bar count by hour:")
print(hours.value_counts().sort_index())
```

## Quick Reference: Strategy File Locations

| File | What to Check |
|------|---------------|
| `wavetrader/strategies/{id}.py` | Entry logic, SL/TP calculation, filter chain |
| `wavetrader/strategies/registry.py` | Is strategy registered? Correct import path? |
| `wavetrader/backtest.py` | Trailing stop logic, SL/TP price conversion |
| `wavetrader/config.py` | `BacktestConfig.trail_activate_r`, pip_value |
| `wavetrader/strategy_backtest.py` | Bar loop, indicator computation, signal flow |
| `wavetrader/strategies/indicators.py` | `compute_all_indicators()`, available indicators |
| `wavetrader/strategies/ai_confirmer.py` | Model loading, shape filtering, confirm() |
| `wavetrader/streaming.py` | Live trailing stop (must match backtest.py logic) |
| `dashboard/services/live_service.py` | Dashboard trailing stop (must match backtest.py logic) |

## Trailing Stop Verification Checklist

After ANY change to trailing stop logic, verify ALL THREE files match:
1. `wavetrader/backtest.py` — `update_trade()` method
2. `wavetrader/streaming.py` — `_update_trailing_stop()` method
3. `dashboard/services/live_service.py` — `_update_trailing_stops()` method

All three MUST use:
- Same `trail_activate_r` activation threshold (or 1R minimum)
- Same formula: `trail_distance = initial_risk * (1 - trailing_stop_pct)`
- Same floor: `min_trail = initial_risk * 0.5`
- NO reference to `DEFAULT_RISK_SCALING.min_trail_pips` (that's the old bug)
