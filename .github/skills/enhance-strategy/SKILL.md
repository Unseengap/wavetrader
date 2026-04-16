---
name: enhance-strategy
description: "Improve an existing trading strategy's win rate, profit factor, and risk-adjusted returns. USE WHEN: a strategy is functional and PF > 1.0 but needs higher win rate, better profit factor, more trades, or reduced drawdown. Covers: entry confirmation techniques, multi-timeframe alignment, session optimization, trailing stop tuning, R:R optimization, filter stacking, next-bar confirmation, structure-based SL, volume/momentum filters, and systematic parameter grid search."
argument-hint: "Strategy ID and current metrics (e.g., 'news_catalyst_ob: WR=44% PF=2.0, want higher WR')"
---

# Enhance Trading Strategy — Optimization Playbook

Systematically improve an already-profitable strategy's performance.

> **Prerequisite**: Strategy must already be profitable (PF > 1.0) before enhancing. If PF < 1.0, use the **debug-strategy** skill first to fix fundamental issues.

## When to Use

- Strategy is profitable but win rate is below 50%
- Want higher profit factor (PF > 2.0 target)
- Need more trades without sacrificing win rate
- Reducing max drawdown for safer compounding
- Tuning trailing stop to capture more of winning moves
- Adding filters to eliminate low-quality entries

## Enhancement Hierarchy (Most to Least Impact)

Work through these in order. Each level builds on the previous.

### Level 1: Session Optimization (Highest Impact)

**The market has clear statistical edges during certain hours.** Testing the news_catalyst_ob strategy across session windows showed:

| Session | Trades | WR | PF | Impact |
|---------|--------|----|----|--------|
| Full 7-20 UTC | 60 | 22% | 1.01 | Break-even |
| 10-20 UTC | 40 | 25% | 1.07 | Marginal |
| **13-17 UTC (overlap)** | **16** | **44%** | **2.03** | **Best** |

**Action**: Test your strategy in 3 session windows:
```python
sessions = [
    (7, 20, "Full session"),
    (10, 20, "London + NY"),
    (13, 17, "Overlap only"),
    (8, 12, "London only"),
    (13, 20, "NY session"),
]
for sh, eh, label in sessions:
    strat = reg.instantiate('STRATEGY_ID', params={
        "session_start_hour": sh, "session_end_hour": eh,
        "trailing_stop_pct": 0.0,  # Test without trailing first
    })
    results = run_strategy_backtest(strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)
    # ... print metrics
```

**Why overlap (13-17 UTC) wins**: Both London and NY institutional desks are active simultaneously, creating the highest liquidity and most decisive moves. OB retests during this window have institutional backing.

### Level 2: Next-Bar Confirmation

**Instead of entering on the signal bar, wait one bar for confirmation.** This was the single biggest win for NCOB strategy (23% → 44% WR).

**Pattern**:
```
Bar N:   Signal fires (e.g., price retests OB zone)  → STORE as pending
Bar N+1: Confirmation bar closes in direction          → ENTER
```

**Implementation in strategy class**:
```python
def __init__(self, params):
    super().__init__(params)
    self._pending = None  # Store pending signal

def evaluate(self, candles, indicators, current_bar_idx):
    # ... setup ...

    # Check pending confirmation FIRST
    if self._pending is not None:
        pend = self._pending
        self._pending = None
        if pend.bar_idx == i - 1 and pend.direction == current_direction:
            # Confirmation: bar closes in our direction
            if direction == Signal.BUY and close > open_price:
                return self._build_setup(...)  # ENTER
            elif direction == Signal.SELL and close < open_price:
                return self._build_setup(...)

    # Find new signal
    signal = self._find_entry(...)
    if signal is not None:
        if self.params["require_confirmation"]:
            self._pending = PendingSignal(signal, direction, i, high, low)
            return None  # Don't enter yet
        else:
            return self._build_setup(...)  # Direct entry (fallback)
```

**Confirmation quality checks** (descending strictness):
1. **Strong**: Confirm bar body > 0.3 × ATR AND closes in direction
2. **Moderate**: Confirm bar closes in direction (any size)
3. **Loose**: Confirm bar closes above/below the signal level

### Level 3: Structure-Based Stop Loss

**ATR-based SL is good. Structure-based SL is better.**

Instead of `SL = N × ATR`, place SL at a structural level:

```python
# BUY trade: SL below the retest bar's low + buffer
sl_price = retest_bar_low - (atr * buffer_mult)
sl_pips = (entry_price - sl_price) / pip

# SELL trade: SL above the retest bar's high + buffer
sl_price = retest_bar_high + (atr * buffer_mult)
sl_pips = (sl_price - entry_price) / pip
```

**Why this works**: ATR-based SL may be too wide or too tight for the actual structure. A retest bar's extreme is a meaningful invalidation level — if price closes through it, the setup is genuinely invalid.

**Buffer recommendations**:
- `buffer_mult = 0.1`: Tight — less room but better R:R
- `buffer_mult = 0.3`: Default — good balance
- `buffer_mult = 0.5`: Wide — more room for noise

### Level 4: R:R and Trailing Stop Optimization

**The key insight**: R:R and trailing stop interact. They should work together, not compete.

#### Strategy A: TP + Late Trail (recommended for OB strategies)
- `min_rr_ratio = 3.0` — fixed TP at 3R
- `trail_activate_r = 3.0` — trail starts at TP level
- TP handles clean moves, trail catches runners past TP
- **Best for**: Reversal entries (OB retest, engulfing, etc.)

#### Strategy B: Pure Trailing (no practical TP)
- `min_rr_ratio = 10.0` — TP so far it never hits
- `trail_activate_r = 1.5–2.0` — trail starts early
- Every winning trade exits via trail
- **Best for**: Trend-following strategies where you want to ride the move

#### Strategy C: Tight TP + No Trail
- `min_rr_ratio = 1.5` — quick TP
- `trailing_stop_pct = 0.0` — no trail
- Simple, high WR, but misses runners
- **Best for**: High-frequency scalpers

**Grid search template for R:R × Trail**:
```python
for rr in [1.5, 2.0, 3.0, 5.0]:
    for trail_pct in [0.0, 0.35, 0.50]:
        for act_r in [2.0, 3.0]:
            if trail_pct == 0 and act_r > 2.0:
                continue  # Skip redundant combos
            cfg.trail_activate_r = act_r
            params = {"min_rr_ratio": rr, "trailing_stop_pct": trail_pct}
            # ... run and collect metrics
```

### Level 5: Multi-Timeframe Alignment

**Higher timeframe confirms direction, lower timeframe times entry.**

#### 4H EMA Stack (Macro Bias)
```python
# In strategy evaluate():
if "4h" in candles:
    ix_4h = min(i // 16, len(ema_20_4h) - 1)
    e20, e50, e200 = ema_20_4h[ix_4h], ema_50_4h[ix_4h], ema_200_4h[ix_4h]
    if e20 > e50 > e200:
        direction = Signal.BUY
    elif e20 < e50 < e200:
        direction = Signal.SELL
    else:
        return None  # No clear trend — skip
```

**Statistics from GBP/JPY data**:
- 4H EMA aligned: 71% of bars (53% bullish, 18% bearish)
- Mixed (no trade): 29% of bars
- Trading only aligned bars filters out the worst setups

#### 4H Structure Bias
```python
if "4h" in indicators.structure:
    ix = min(i // 16, len(indicators.structure["4h"]) - 1)
    bias = float(indicators.structure["4h"][ix, 7])  # Column 7 = trend bias
    if direction == Signal.BUY and bias < 0.1:
        return None  # Not enough bullish structure
    if direction == Signal.SELL and bias > -0.1:
        return None
```

### Level 6: Momentum and Volatility Filters

#### RSI Filter (avoid overbought/oversold entries)
```python
rsi = indicators.rsi[etf][i]
if direction == Signal.BUY and rsi > 65:
    return None  # Already overbought
if direction == Signal.SELL and rsi < 35:
    return None  # Already oversold
```

**Tuning**: Wider thresholds (70/30) = more trades. Tighter (60/40) = fewer but higher quality.

#### ADX Filter (require trending market)
```python
adx = indicators.adx[etf][i]
if adx < 12:
    return None  # Choppy market, skip
```

**Tuning**: ADX > 20 = strong trend only. ADX > 12 = moderate trend. ADX > 8 = any trend.

#### Engulfing Pattern Bonus
```python
if indicators.engulfing is not None and i < len(indicators.engulfing):
    eng = float(indicators.engulfing[i, 0])
    if (direction == Signal.BUY and eng > 0) or (direction == Signal.SELL and eng < 0):
        confidence += 0.08  # Boost confidence for engulfing patterns
```

### Level 7: Trade Count Optimization

If strategy has good WR and PF but too few trades:

| Lever | Direction | Impact |
|-------|-----------|--------|
| Widen session hours | More trades | May lower WR |
| Lower `impulse_atr_mult` | More OBs detected | More entries |
| Raise `ob_max_age_bars` | OBs stay valid longer | More retests |
| Lower `min_adx` | Trade in choppy markets | Riskier entries |
| Lower `min_confidence` | Accept weaker signals | Lower quality |
| Raise `ob_retest_tolerance` | Wider retest zone | More false retests |
| Add pairs | 3× opportunities | Different pair dynamics |

**Strategy**: Loosen ONE filter at a time. After each change, re-run the full combo sweep to check WR and PF didn't drop below acceptable levels.

### Level 8: Walk-Forward Validation

After optimization, validate that results aren't overfit:

```python
# Split data: 70% train, 30% test
split_idx = int(len(df_dict['15min']) * 0.7)
train_dict = {tf: df.iloc[:int(len(df) * 0.7)] for tf, df in df_dict.items()}
test_dict = {tf: df.iloc[int(len(df) * 0.7):] for tf, df in df_dict.items()}

# Run on train
strat = reg.instantiate('STRATEGY_ID')
train_results = run_strategy_backtest(strat, train_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)

# Run on test (unseen data)
strat.reset()  # IMPORTANT: reset internal state
test_results = run_strategy_backtest(strat, test_dict, bt_config=cfg, pair='GBP/JPY', verbose=False)

print(f"Train: {len(train_results.trades)}tr PF={train_pf:.2f}")
print(f"Test:  {len(test_results.trades)}tr PF={test_pf:.2f}")
```

**Healthy**: Test PF > 80% of Train PF
**Overfit**: Test PF < 50% of Train PF (or negative)

## Enhancement Checklist

Before marking a strategy enhancement as complete, verify:

- [ ] No trailing baseline tested (PF > 1.0 confirms entry edge)
- [ ] Session window optimized (overlap hours tested)
- [ ] R:R × trailing grid search done (at least 8 combos)
- [ ] Exit mix is healthy: SL > 0, TP > 0, Trail > 0
- [ ] AvgWin > AvgLoss (profit factor verification)
- [ ] Trade count is reasonable (>10 for overlap, >40 for full session)
- [ ] Defaults updated in strategy file's `default_params()`
- [ ] Walk-forward validation shows no overfitting

## Proven Optimization Results (Reference)

### News Catalyst OB V4 Journey

| Version | Change | WR | PF | Trades |
|---------|--------|----|----|--------|
| V1 | Vol spike entry | 42% | 1.05 | 12 |
| V2 | Reversal candle | 39% | 0.85 | 44 |
| V3 | ATR SL + strong close | 36% | 0.65 | 45 |
| V3 fix | Disabled trailing (bug) | 44% | 1.94 | 16 |
| **V4** | **Next-bar confirm + fixed trail** | **44%** | **2.03** | **16** |

**Key takeaway**: The single biggest improvement came from fixing the trailing stop activation bug, not from entry logic changes. Always verify infrastructure before tweaking entries.

### What Made V4 Win

1. **Next-bar confirmation** (+15% WR): Eliminated false retests
2. **Overlap-only hours** (+20% WR vs full session): Best liquidity window
3. **Fixed trailing stop** (PF 0.65 → 2.03): 1R activation was killing trades
4. **Structure-based SL**: Retest bar's low + buffer instead of pure ATR
5. **Trail at TP level** (activate_r=3.0): TP handles clean wins, trail catches runners
