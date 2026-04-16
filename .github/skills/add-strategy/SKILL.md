---
name: add-strategy
description: "Create a new trading strategy for the WaveTrader platform end-to-end. USE WHEN: building a new rule-based strategy, testing/optimizing it via backtest, deploying it live to OANDA with its own account, wiring it into the dashboard dropdown, adding its docker-compose service, or fixing strategy-related backtest/live errors. Covers: strategy class, param schema, registry wiring, backtest validation, parameter optimization, trailing stop configuration, docker-compose service, OANDA account binding, frontend dropdown auto-discovery."
argument-hint: "Strategy name, category (scalper/swing/trend/mean-reversion), and trading concept"
---

# Add New Strategy — Full Pipeline

Create, validate, optimize, and deploy a new rule-based trading strategy.

> **The goal is aggressive compounding: $100 → $10,000 in 6 months using 10% risk per trade with trailing stops on every position.** Every strategy must be independently profitable before deployment.

## When to Use

- Creating a new rule-based trading strategy from scratch
- Optimizing strategy parameters for better backtest returns
- Deploying a validated strategy to OANDA with its own account
- Wiring a strategy into the dashboard dropdown
- Fixing strategy backtest or live execution errors
- Adding trailing stop logic to an existing strategy

## Key Constraints

1. **Every strategy MUST use trailing stops** — `trailing_stop_pct` > 0 in every `StrategySetup`
2. **$100 initial balance, 10% risk per trade** — configured in `NOTEBOOK_DEFAULTS`
3. **Each strategy gets its own OANDA sub-account** — separate demo + live accounts
4. **Available data timeframes**: `15min`, `1h`, `4h`, `1d` (NO 5min data exists)
5. **Available pairs**: `GBP/JPY`, `EUR/JPY`, `GBP/USD`
6. **Backtest data**: ~43,000 bars (15min) from 2015-2026 in `processed_data/test/`
7. **Pip size**: 0.01 for JPY pairs, 0.0001 for USD pairs

## Target Performance Metrics (per strategy)

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Win Rate | > 40% | > 50% | > 60% |
| Profit Factor | > 1.2 | > 1.8 | > 2.5 |
| Sharpe Ratio | > 0.5 | > 1.5 | > 2.5 |
| Max Drawdown | < 40% | < 25% | < 15% |
| Total Return ($100→) | > $200 | > $1,000 | > $5,000+ |
| Trades (over test set) | > 50 | > 150 | > 300 |

## Procedure

### Phase 1: Strategy Design

1. **Choose identifiers**:
   - `STRATEGY_ID`: lowercase snake_case (e.g., `amd_session`, `ema_crossover`)
   - `STRATEGY_NAME`: display name (e.g., `AMD Session Scalper`)
   - `CLASS_NAME`: PascalCase class (e.g., `AMDSessionStrategy`)
   - `CATEGORY`: `scalper` | `swing` | `trend` | `mean_reversion`
   - `AUTHOR`: `Dectrick McGee`

2. **Choose entry timeframe**: `15min` (finest available — do NOT use 5min)

3. **Define the trading concept** — every strategy needs:
   - **Entry logic**: What conditions trigger BUY/SELL?
   - **Direction logic**: How do we determine BUY vs SELL?
   - **Stop loss**: Structure-based (swing high/low ± ATR buffer)
   - **Take profit**: Risk-multiple based (SL × R:R ratio)
   - **Trailing stop**: MANDATORY — percentage of initial risk (0.3–0.6 typical)
   - **Filters**: Session time, volatility, trend alignment, etc.

### Phase 2: Create Strategy File

Create `wavetrader/strategies/{STRATEGY_ID}.py`:

```python
"""
{STRATEGY_NAME} — by {AUTHOR}

{One paragraph describing the trading concept, when it trades, what it looks for.}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec
from ..types import Signal


class {CLASS_NAME}(BaseStrategy):

    meta = StrategyMeta(
        id="{STRATEGY_ID}",
        name="{STRATEGY_NAME}",
        author="{AUTHOR}",
        version="1.0.0",
        description="{One-liner description}",
        category="{CATEGORY}",
        timeframes=["15min", "1h", "4h"],
        pairs=["GBP/JPY", "EUR/JPY", "GBP/USD"],
        entry_timeframe="15min",
    )

    def default_params(self) -> Dict[str, Any]:
        return {
            # ── Core entry params ────────────────────
            # ... strategy-specific params here ...
            # ── Risk management (MUST HAVE) ──────────
            "min_rr_ratio": 2.0,            # Minimum reward:risk ratio
            "trailing_stop_pct": 0.4,       # Trail 40% of initial risk
            "sl_atr_mult": 1.5,             # SL = N × ATR
            "min_sl_pips": 8.0,             # Floor SL to avoid micro-stops
            "max_sl_pips": 80.0,            # Cap SL to limit risk per trade
            # ── Filters ──────────────────────────────
            "min_confidence": 0.5,          # Minimum confidence to take trade
            "min_atr_pips": 3.0,            # Skip low-volatility bars
        }

    def param_schema(self) -> List[ParamSpec]:
        return [
            # Expose tunable params for dashboard config panel
            # ParamSpec(name, label, type, default, min, max, step, description)
            ParamSpec("min_rr_ratio", "Min R:R Ratio", "float", 2.0, 1.0, 5.0, 0.5,
                      "Minimum reward-to-risk ratio for entry"),
            ParamSpec("trailing_stop_pct", "Trailing Stop %", "float", 0.4, 0.1, 0.8, 0.05,
                      "Trail distance as fraction of initial risk (lower = tighter)"),
            ParamSpec("sl_atr_mult", "SL ATR Multiplier", "float", 1.5, 0.5, 4.0, 0.25,
                      "Stop loss distance in ATR multiples"),
            # ... strategy-specific params ...
        ]

    def evaluate(
        self,
        candles: Dict[str, pd.DataFrame],
        indicators: IndicatorBundle,
        current_bar_idx: int,
    ) -> Optional[StrategySetup]:
        """Evaluate on current bar. Return StrategySetup or None."""
        p = self.params
        etf = self.meta.entry_timeframe
        if etf not in candles:
            return None

        df = candles[etf]
        i = current_bar_idx
        if i < 200 or i >= len(df):
            return None

        bar = df.iloc[i]
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        pip_size = 0.01  # JPY pairs; use 0.0001 for USD pairs

        # ── Get indicators ────────────────────────────────────────────
        atr_val = indicators.atr.get(etf, np.array([]))[i] if etf in indicators.atr else 0.0
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        atr_pips = atr_val / pip_size

        # Skip low-volatility bars
        if atr_pips < p["min_atr_pips"]:
            return None

        # ══════════════════════════════════════════════════════════════
        # YOUR ENTRY LOGIC HERE
        # Must determine: direction (Signal.BUY or Signal.SELL)
        # Must determine: confidence (0.0–1.0)
        # Return None if no valid setup
        # ══════════════════════════════════════════════════════════════

        direction = ...  # Signal.BUY or Signal.SELL
        confidence = ...  # 0.0–1.0

        if confidence < p["min_confidence"]:
            return None

        # ── Compute SL/TP (MANDATORY PATTERN) ────────────────────────
        sl_pips = atr_pips * p["sl_atr_mult"]
        sl_pips = max(sl_pips, p["min_sl_pips"])
        sl_pips = min(sl_pips, p["max_sl_pips"])
        tp_pips = sl_pips * p["min_rr_ratio"]

        return StrategySetup(
            direction=direction,
            entry_price=close,
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            confidence=round(confidence, 3),
            trailing_stop_pct=p["trailing_stop_pct"],  # ← MANDATORY
            reason=f"...",  # Human-readable reason
        )
```

### Phase 3: Register the Strategy

Add to `wavetrader/strategies/registry.py` in `_BUILTIN_STRATEGIES`:

```python
_BUILTIN_STRATEGIES: Dict[str, str] = {
    # ... existing strategies ...
    "{STRATEGY_ID}": "wavetrader.strategies.{STRATEGY_ID}.{CLASS_NAME}",
}
```

Set `default_id` if this is the first strategy:
```python
self.default_id: str = "{STRATEGY_ID}"
```

### Phase 4: Validate — Quick Smoke Test

Run from terminal (venv must be active):

```bash
python -c "
from wavetrader.strategies.registry import get_strategy_registry
reg = get_strategy_registry()
strat = reg.instantiate('{STRATEGY_ID}')
print(f'OK: {strat.meta.name} has {len(strat.param_schema())} params')
print(f'Trailing stop: {strat.params[\"trailing_stop_pct\"]}')
assert strat.params['trailing_stop_pct'] > 0, 'TRAILING STOP MUST BE > 0'
"
```

### Phase 5: Backtest via API

Start the dashboard and run a backtest:

```bash
# Start server
python dashboard/run.py &

# Run backtest (AI off for speed)
curl -s -X POST http://127.0.0.1:5000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "{STRATEGY_ID}",
    "pair": "GBP/JPY",
    "initial_balance": 100,
    "risk_per_trade": 0.10,
    "ai_confirm": false
  }' | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'error' in d: print('ERROR:', d['error']); exit(1)
m = d.get('metrics', {})
t = d.get('trades', [])
print(f'Trades:   {len(t)}')
print(f'Win Rate: {m.get(\"win_rate\", 0):.1%}')
print(f'PF:       {m.get(\"profit_factor\", 0):.2f}')
print(f'Balance:  \${m.get(\"final_balance\", 100):.2f}')
print(f'Return:   {((m.get(\"final_balance\",100)-100)/100)*100:.1f}%')
print(f'Sharpe:   {m.get(\"sharpe_ratio\", 0):.2f}')
print(f'Max DD:   {m.get(\"max_drawdown_pct\", 0):.1f}%')
# Check trailing stops
trail_exits = sum(1 for t_ in t if 'trail' in t_.get('exit_reason','').lower())
print(f'Trailing exits: {trail_exits}/{len(t)}')
"
```

### Phase 6: Parameter Optimization

If backtest results are poor, adjust parameters. Key levers:

| Problem | Fix |
|---------|-----|
| Too few trades (<50) | Lower thresholds, widen filters, reduce `min_confidence` |
| Too many trades (>1000) | Tighten filters, require multi-TF confirmation |
| Low win rate (<35%) | Improve entry logic, add trend filter, require engulfing |
| Low profit factor (<1.0) | Increase `min_rr_ratio`, tighten `trailing_stop_pct` |
| Huge drawdown (>40%) | Lower `sl_atr_mult`, add `max_sl_pips` cap, reduce risk |
| All SL exits, no trails | Lower `trailing_stop_pct` (e.g., 0.3), or increase TP range |

**Optimization loop** — test parameter combos via API:

```bash
for rr in 1.5 2.0 2.5 3.0; do
  for trail in 0.2 0.3 0.4 0.5; do
    echo -n "RR=$rr Trail=$trail: "
    curl -s -X POST http://127.0.0.1:5000/api/backtest/run \
      -H "Content-Type: application/json" \
      -d "{
        \"strategy\": \"{STRATEGY_ID}\",
        \"strategy_params\": {\"min_rr_ratio\": $rr, \"trailing_stop_pct\": $trail},
        \"pair\": \"GBP/JPY\",
        \"initial_balance\": 100,
        \"risk_per_trade\": 0.10,
        \"ai_confirm\": false
      }" | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d.get('metrics', {})
t = d.get('trades', [])
bal = m.get('final_balance', 100)
wr = m.get('win_rate', 0)
pf = m.get('profit_factor', 0)
print(f'{len(t)} trades  WR={wr:.0%}  PF={pf:.2f}  \${bal:.0f}')
"
  done
done
```

### Phase 7: Test with AI Confirmation

Once the strategy is profitable without AI, test with AI confirm ON:

```bash
curl -s -X POST http://127.0.0.1:5000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "{STRATEGY_ID}",
    "pair": "GBP/JPY",
    "initial_balance": 100,
    "risk_per_trade": 0.10,
    "ai_confirm": true
  }' | python3 -c "..."  # Same parsing as above
```

AI confirmation should **improve win rate** (fewer trades but higher quality). If it makes things worse, the strategy is fine without it.

### Phase 8: Remove from "Coming Soon" List

The strategy auto-removes from the "Coming Soon" dropdown in both:
- `dashboard/static/js/backtest-init.js` → `COMING_SOON_STRATEGIES` array
- `dashboard/static/js/live-init.js` → `COMING_SOON` array

This happens automatically — the JS filters out any strategy name that matches a live strategy from the API. No code change needed.

### Phase 9: Docker & OANDA Deployment

#### 9A. Create OANDA sub-account

1. Log into OANDA → Account → Create Sub-Account
2. Name it: `WaveTrader-{STRATEGY_NAME}`
3. Note the new account ID
4. **CRITICAL**: Revoke and regenerate API key (affects ALL accounts temporarily)

#### 9B. Add env vars to `.env`

```env
# {STRATEGY_NAME}
{PREFIX}_OANDA_DEMO_API_KEY=your_regenerated_key
{PREFIX}_OANDA_DEMO_ACCOUNT_ID=101-001-XXXXXXXX-XXX
{PREFIX}_OANDA_LIVE_API_KEY=
{PREFIX}_OANDA_LIVE_ACCOUNT_ID=
```

#### 9C. Add docker-compose service

```yaml
  strategy-{STRATEGY_ID}:
    build: .
    image: wavetrader:latest
    container_name: strategy-{STRATEGY_ID}
    restart: unless-stopped
    volumes:
      - ./data:/data:rw
      - ./checkpoints:/checkpoints:ro
      - ./logs:/app/logs:rw
    env_file:
      - .env
    environment:
      - STRATEGY_ID={STRATEGY_ID}
      - AI_CHECKPOINT_PATH=/checkpoints
      - PAIR=GBP/JPY
      - OANDA_API_KEY=${{{PREFIX}_OANDA_DEMO_API_KEY:-${{OANDA_DEMO_API_KEY}}}}
      - OANDA_ACCOUNT_ID=${{{PREFIX}_OANDA_DEMO_ACCOUNT_ID:-${{OANDA_DEMO_ACCOUNT_ID}}}}
      - OANDA_ENVIRONMENT=practice
      - GEMINI_API_KEY=${{GEMINI_API_KEY:-}}
      - LLM_ARBITER_ENABLED=${{LLM_ARBITER_ENABLED:-true}}
    depends_on:
      - redis
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
```

### Phase 10: Live Verification

```bash
# Deploy
docker-compose up -d strategy-{STRATEGY_ID}

# Check logs
docker logs -f strategy-{STRATEGY_ID}

# Verify in dashboard at http://localhost:5000/live
# Select the strategy from dropdown → should see live candles + signals
```

---

## Available Indicators (from IndicatorBundle)

Every strategy receives these pre-computed indicators:

| Indicator | Key | Shape | Notes |
|-----------|-----|-------|-------|
| RSI | `indicators.rsi[tf]` | (n,) | 0–100, per timeframe |
| ATR | `indicators.atr[tf]` | (n,) | Average True Range, per TF |
| ADX | `indicators.adx[tf]` | (n,) | Trend strength 0–100 |
| Market Structure | `indicators.structure[tf]` | (n,8) | Swing highs/lows/trend |
| EMA 20 | `indicators.ema_20[tf]` | (n,) | |
| EMA 50 | `indicators.ema_50[tf]` | (n,) | |
| EMA 200 | `indicators.ema_200[tf]` | (n,) | |
| Bollinger Upper | `indicators.bollinger_upper[tf]` | (n,) | |
| Bollinger Lower | `indicators.bollinger_lower[tf]` | (n,) | |
| Bollinger Mid | `indicators.bollinger_mid[tf]` | (n,) | |
| Asian Range | `indicators.asian_range` | (n,5) | Entry TF only |
| London Sweep | `indicators.london_sweep` | (n,4) | Entry TF only |
| Engulfing | `indicators.engulfing` | (n,3) | Entry TF only |
| FVG (Fair Value Gaps) | `indicators.fair_value_gaps` | (n,4) | Entry TF only |
| S/R Zones | `indicators.sr_zones` | (n,3) | Entry TF only |
| ORB Features | `indicators.orb_features` | (n,4) | Entry TF only |

## Trailing Stop Mechanics

**CRITICAL**: The trailing stop system has a 3-phase lifecycle. Getting this wrong was the #1 cause of unprofitable strategies before April 2026.

### How It Actually Works (BacktestEngine + streaming.py + live_service.py)

```
Phase 1: DORMANT (price < activate_r × R from entry)
  → SL stays at original level — no tightening, full breathing room

Phase 2: ACTIVATED (price reaches activate_r × R in our favor)
  → SL starts trailing: new_SL = peak_price - trail_distance
  → trail_distance = initial_risk × (1 - trailing_stop_pct)
  → Floor: trail_distance >= initial_risk × 0.5

Phase 3: EXIT (price pulls back to trailing SL)
  → Closed at trailing SL with reason "Trailing Stop"
  → Profit = trailing SL - entry (guaranteed profit since trail only activates in profit)
```

### Config Parameter: `trail_activate_r`

Set in `BacktestConfig.trail_activate_r` (default: `3.0`). Controls when trailing begins.

| `trail_activate_r` | Behavior | Best For |
|---------------------|----------|----------|
| 1.0 | Trail at breakeven — tight, exits winners early | Scalpers needing quick locks |
| 1.5 | Trail at +50% of risk — moderate | |
| 2.0 | Trail at 2R — lets winners develop | Swing trades |
| **3.0** (default) | Trail at TP level — TP handles clean wins, trail handles runners | **Best for OB strategies** |
| 5.0+ | Very late trailing — mostly TP exits | Pure TP strategies |

### Strategy-Side Setup (in StrategySetup)

```python
return StrategySetup(
    direction=direction,
    entry_price=close,
    sl_pips=round(sl_pips, 1),
    tp_pips=round(tp_pips, 1),
    confidence=round(confidence, 3),
    trailing_stop_pct=p["trailing_stop_pct"],  # ← MANDATORY
    reason=f"...",
)
```

### Known Bug History (DO NOT REINTRODUCE)

**OLD BUG** (pre-April 2026): A global `min_trail_pips = 20.0` floor in `DEFAULT_RISK_SCALING` immediately tightened SL from 30→20 pips on entry. This killed 100% of trades before they could run. Symptoms:
- 0 trailing stop exits in all backtests
- Win rate below random (12-25%)
- All exits are "Stop Loss", never "Trailing Stop"

**FIX**: Activation threshold (`trail_activate_r`) + scaled floor (`initial_risk * 0.5`). The global `min_trail_pips` in `RiskScaling` is now IGNORED by the engine.

### Recommended Defaults by Strategy Type

| Strategy Type | `trailing_stop_pct` | `trail_activate_r` | `min_rr_ratio` |
|--------------|---------------------|---------------------|----------------|
| Scalper (quick) | 0.35–0.50 | 1.5–2.0 | 1.5–2.0 |
| OB / Reversal | 0.50 | 3.0 | 3.0 |
| Trend follower | 0.60–0.70 | 2.0 | 2.0–3.0 |
| Swing | 0.50–0.60 | 2.5 | 2.5–3.0 |

### Exit Reason Reference

| Exit Reason | Meaning |
|-------------|---------|
| `"Stop Loss"` | Hit original SL (trail never activated) |
| `"Take Profit"` | Hit TP target |
| `"Trailing Stop"` | Trail activated and price pulled back — PROFIT LOCKED |
| `"End of Backtest"` | Position still open at end of data |

### Verification: Checking Trail Health in Backtest Output

After backtesting, always check the exit mix:
```python
sl = sum(1 for t in trades if t.exit_reason.strip() == 'Stop Loss')
tp = sum(1 for t in trades if 'take' in t.exit_reason.lower() or 'profit' in t.exit_reason.lower())
trail = sum(1 for t in trades if 'trail' in t.exit_reason.lower())
print(f"SL={sl} TP={tp} Trail={trail}")
```

**Healthy exit mix**: SL > 0, TP > 0, Trail > 0. If Trail = 0, check `trail_activate_r` isn't too high for the strategy's moves.


### Phase 5b: Rapid Parameter Optimization (Script Method)

For fast iteration WITHOUT the dashboard, write a test script at `scripts/test_{STRATEGY_ID}.py`:

```python
"""Quick backtest optimization for {STRATEGY_ID} strategy."""
import pandas as pd
from pathlib import Path
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.config import BacktestConfig

# ── Load Data ────────────────────────────────────────────────────────
processed_dir = Path("processed_data/test")
pair_tag = "GBPJPY"  # or EURJPY, GBPUSD
df_dict = {}
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
print()

reg = get_strategy_registry()

# ── Define Parameter Combos ──────────────────────────────────────────
# Format: (strategy_params_dict, trail_activate_r, label)
combos = [
    ({}, 3.0, "DEFAULTS"),
    ({"trailing_stop_pct": 0.0}, 3.0, "No Trailing (baseline)"),
    ({"trailing_stop_pct": 0.35}, 3.0, "T=35%"),
    ({"trailing_stop_pct": 0.50}, 3.0, "T=50%"),
    ({"min_rr_ratio": 2.0}, 3.0, "R:R=2"),
    ({"min_rr_ratio": 3.0}, 3.0, "R:R=3"),
    ({"min_rr_ratio": 3.0, "trailing_stop_pct": 0.50}, 2.0, "R:R=3 T=50% A=2"),
    ({"min_rr_ratio": 3.0, "trailing_stop_pct": 0.50}, 3.0, "R:R=3 T=50% A=3"),
]

# ── Run All Combos ───────────────────────────────────────────────────
print(f"Testing {len(combos)} combos...\n")
for idx, (params, act_r, label) in enumerate(combos):
    cfg = BacktestConfig()
    cfg.initial_balance = 100.0
    cfg.risk_pct = 0.10
    cfg.trail_activate_r = act_r

    strat = reg.instantiate('{STRATEGY_ID}', params=params)
    results = run_strategy_backtest(
        strat, df_dict, bt_config=cfg, pair='GBP/JPY', verbose=False,
    )
    trades = results.trades
    n = len(trades)
    if n > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        total_win = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999
        sl = sum(1 for t in trades if (t.exit_reason or '').strip() == 'Stop Loss')
        tp = sum(1 for t in trades if 'take' in (t.exit_reason or '').lower())
        trail = sum(1 for t in trades if 'trail' in (t.exit_reason or '').lower())
        avg_w = total_win / max(wins, 1)
        avg_l = total_loss / max(n - wins, 1)
        print(f"{idx+1}. [{label:>30}] {n:>3}tr WR={wins/n:.0%} PF={pf:.2f} "
              f"${results.final_balance:>8.2f} SL={sl} TP={tp} Trail={trail} "
              f"AvgW=${avg_w:.2f} AvgL=${avg_l:.2f}")
    else:
        print(f"{idx+1}. [{label:>30}] 0 trades")
print("\nDone!")
```

Run with: `source .venv/bin/activate && PYTHONPATH=. python scripts/test_{STRATEGY_ID}.py`

### Interpreting Test Results

| Pattern | Diagnosis |
|---------|-----------|
| All Trail exits, low AvgW | `trail_activate_r` too low — trail chokes winners |
| 0 Trail exits, all SL+TP | `trail_activate_r` too high — trail never activates |
| High WR with no trail + low WR with trail | Trail tightening SL prematurely — raise `trail_activate_r` |
| Good WR, PF < 1.0 | AvgW < AvgL — raise `min_rr_ratio` or lower `sl_atr_mult` |
| Good PF, too few trades | Widen session hours, lower `impulse_atr_mult`, lower `min_adx` |

### Case Study: News Catalyst OB (44% WR, PF=2.03)

The optimal configuration found through testing:
- `trail_activate_r = 3.0` — trail starts right at TP level (3R)
- `trailing_stop_pct = 0.50` — tightens to 50% of initial risk
- `min_rr_ratio = 3.0` — 3R fixed TP for clean moves
- `session_start_hour = 13, session_end_hour = 17` — overlap only

Exit mix: 9 SL + 2 TP + 5 Trail = 16 total trades
- TP handles clean 3R moves (2/16)
- Trail locks profit on runners past 3R (5/16)
- SL exits losers at original level — no premature tightening (9/16)

## Math: $100 → $10,000 in 6 Months

With 10% risk per trade and compounding:
- Need ~48.3× return (100 → 10,000)
- At 10% risk, each win = ~10% × R:R of account
- With 2:1 R:R and 50% win rate: expected value per trade = (0.5 × 20%) - (0.5 × 10%) = +5% per trade
- 48.3× requires ~78 consecutive +5% compounds: `1.05^78 = 44.2×`
- Over 6 months (~130 trading days), need ~78 net-positive trades
- That's roughly 0.6 net wins per trading day across ALL strategies combined
- With multiple strategies running in parallel, this is achievable if each is independently profitable

## File Locations Reference

| File | Purpose |
|------|---------|
| `wavetrader/strategies/{STRATEGY_ID}.py` | Strategy class |
| `wavetrader/strategies/registry.py` | `_BUILTIN_STRATEGIES` dict |
| `wavetrader/strategies/base.py` | BaseStrategy, StrategySetup, ParamSpec |
| `wavetrader/strategies/indicators.py` | compute_all_indicators() |
| `wavetrader/strategies/ai_confirmer.py` | AI confirmation wrapper |
| `wavetrader/strategy_backtest.py` | Strategy backtest engine |
| `wavetrader/backtest.py` | Core BacktestEngine (trailing stops) |
| `dashboard/services/backtest_service.py` | `run_strategy_backtest_from_config()` |
| `dashboard/routes/backtest.py` | `/api/backtest/run`, `/api/backtest/strategies` |
| `dashboard/routes/live.py` | `/api/live/strategies`, `/api/live/start` |
| `dashboard/static/js/backtest-init.js` | Strategy dropdown + coming soon |
| `dashboard/static/js/live-init.js` | Live strategy dropdown |
| `dashboard/static/js/config-panel.js` | runBacktest(), collectConfig() |
| `docker-compose.yml` | Strategy service definitions |
