# WaveTrader Strategy Architecture — Plan & Status

> **Author:** Dectrick McGee  
> **Last Updated:** April 16, 2026  
> **Goal:** Replace AI models with rule-based trading strategies confirmed by the WaveTrader MTF model

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  STRATEGY LAYER (replaces model inference)                │
│  wavetrader/strategies/                                   │
│  ├─ BaseStrategy.evaluate(candles, indicators) → Setup    │
│  ├─ amd_session.py       ┐                               │
│  ├─ supply_demand.py     │                               │
│  ├─ ict_smc.py           │ 7 strategies                  │
│  ├─ orb_breakout.py      │ (extensible)                  │
│  ├─ ema_crossover.py     │                               │
│  ├─ mean_reversion.py    │                               │
│  └─ structure_break.py   ┘                               │
└─────────────┬────────────────────────────────────────────┘
              │ StrategySetup (direction, sl, tp, reason, context)
              ▼
┌──────────────────────────────────────────────────────────┐
│  AI CONFIRMATION LAYER (WaveTrader MTF — only model kept)│
│  Check: signal_logits agrees with strategy direction?     │
│  Check: confidence > threshold? alignment > threshold?    │
│  Result: CONFIRMED (with AI confidence) or REJECTED       │
└─────────────┬────────────────────────────────────────────┘
              │ ConfirmedSignal (TradeSignal + strategy_reason)
              ▼
┌──────────────────────────────────────────────────────────┐
│  LLM ARBITER — STRATEGY VOICE & ANALYST (Gemini)         │
│  Narrates what strategy saw, why it fired, risk notes     │
│  Modes: ADVISORY / VETO / OVERRIDE                        │
│  SSE → dashboard signals tab + Telegram                   │
└─────────────┬────────────────────────────────────────────┘
              │ ArbiterDecision (approved signal + narrative)
              ▼
┌──────────────────────────────────────────────────────────┐
│  EXECUTION (BacktestEngine / OANDA live)                  │
│  Same TradeSignal, position sizing, trailing stops        │
└──────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- Strategies own SL/TP (structure-based), AI only confirms direction
- Each strategy gets its own OANDA account for isolation
- LLM Arbiter repurposed as "strategy voice" — narrates what strategy detected
- Registry auto-discovers strategies + supports env var JSON override

---

## 7 Strategies

| # | Strategy | File | Category | Entry Logic | SL | TP |
|---|----------|------|----------|-------------|----|----|
| 1 | **AMD Session Scalper** | `amd_session.py` | scalper | Asian range → London sweep → NY reversal entry (engulfing at S&R flip zone) | Below demand/supply zone + ATR buffer | Next opposing zone |
| 2 | **Supply & Demand** | `supply_demand.py` | swing | Price enters S&D zone → engulfing/pin bar confirm → enter with HTF trend | Below zone low/above zone high + buffer | 1:2 or 1:3 RR to opposing zone |
| 3 | **ICT/SMC** | `ict_smc.py` | scalper | Liquidity sweep (stop hunt) → order block entry → FVG confluence | Below OB low (buys) / above OB high (sells) | Opposing FVG or liquidity pool |
| 4 | **ORB Breakout** | `orb_breakout.py` | breakout | NY ORB forms → breakout → pullback to ORB edge → enter retest | Below/above ORB range | 1x–1.5x ORB range projection |
| 5 | **EMA Crossover** | `ema_crossover.py` | trend | EMA 20×50 cross → pullback to EMA 20 → enter with 200 EMA trend filter | Below recent swing low/high | Next structure level or 1:2 RR |
| 6 | **Mean Reversion** | `mean_reversion.py` | mean_reversion | Price at ±2σ Bollinger → RSI extreme → ADX < 25 (ranging) | Beyond band extreme + ATR buffer | Bollinger midline (20 SMA) |
| 7 | **Structure Break** | `structure_break.py` | swing | HH/HL→LL/LH break → retest of broken level → rejection candle confirm | Below/above retested level | Next major structure level |

All by **Dectrick McGee** · v1.0.0 · Pairs: GBP/JPY, EUR/JPY, GBP/USD

---

## Implementation Status

### ✅ Phase 1 — Strategy Framework Core (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `wavetrader/strategies/__init__.py` | ✅ Done | Package init, re-exports |
| `wavetrader/strategies/base.py` | ✅ Done | BaseStrategy ABC, StrategySetup, StrategyMeta, IndicatorBundle, ParamSpec |
| `wavetrader/strategies/indicators.py` | ✅ Done | `compute_all_indicators()` wrapping indicators.py + amd_features.py |
| `wavetrader/strategies/registry.py` | ✅ Done | StrategyRegistry, StrategyEntry, auto-discovery + env var JSON |
| `wavetrader/strategies/amd_session.py` | ✅ Done | AMD Session Scalper — Asian accumulation → London sweep → NY reversal |
| `wavetrader/strategies/supply_demand.py` | ✅ Done | S&D zone reversal with engulfing/pin bar + HTF trend filter |
| `wavetrader/strategies/ict_smc.py` | ✅ Done | ICT/SMC — liquidity sweep → order block → FVG confluence |
| `wavetrader/strategies/orb_breakout.py` | ✅ Done | NY Open Range Breakout with pullback retracement |
| `wavetrader/strategies/ema_crossover.py` | ✅ Done | EMA 20/50 crossover + pullback + 200 EMA trend filter |
| `wavetrader/strategies/mean_reversion.py` | ✅ Done | Bollinger ±2σ + RSI extreme + ADX < 25 range filter |
| `wavetrader/strategies/structure_break.py` | ✅ Done | Structure shift (HH/HL→LL/LH) + retest + rejection confirm |

### ✅ Phase 2 — AI Confirmation Layer (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `wavetrader/strategies/ai_confirmer.py` | ✅ Done | AIConfirmer wraps WaveTrader MTF. Checks direction match, confidence > 0.55, alignment > 0.40. Returns ConfirmedSignal or None. |

### ✅ Phase 3 — LLM Arbiter → Strategy Voice (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `wavetrader/llm_arbiter.py` | ✅ Modified | Added strategy_id/name/author/reason/context to ArbiterContext, `narrative` to ArbiterDecision, strategy-voice system prompt |

### ✅ Phase 4 — Strategy Registry (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `wavetrader/strategies/registry.py` | ✅ Done | StrategyRegistry with 7 builtin strategies, `instantiate()`, `to_list()` for API |

### ✅ Phase 5 — Backtest Pipeline (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `wavetrader/strategy_backtest.py` | ✅ Done | `run_strategy_backtest()` — bar-by-bar using existing BacktestEngine |
| `dashboard/services/backtest_service.py` | ✅ Modified | Added `run_strategy_backtest_from_config()`, `_find_latest_checkpoint()`, `_build_replay_candles()` |
| `dashboard/routes/backtest.py` | ✅ Modified | Added `/strategies`, `/strategies/<id>/params` endpoints. `/run` auto-routes to strategy backtest if `strategy` key present |

### ✅ Phase 6 — Live Streaming (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `wavetrader/strategy_streaming.py` | ✅ Done | StrategyStreamingEngine — strategy-driven live trading loop |
| `dashboard/routes/live.py` | ✅ Modified | Added `/strategies` endpoint for dropdown |

### ✅ Phase 7 — Dashboard Frontend (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `dashboard/templates/index.html` | ✅ Modified | Added `#nav-strategy-select` dropdown + `#strategy-params-section` dynamic container |
| `dashboard/static/js/live-init.js` | ✅ Modified | Added `loadStrategySelector()`, `switchStrategy()`, `currentStrategy` |
| `dashboard/static/js/backtest-init.js` | ✅ Modified | Added `populateStrategyDropdown()`, fetches + renders strategy params on change |
| `dashboard/static/js/config-panel.js` | ✅ Modified | `renderStrategyParams()`, `collectStrategyParams()`, sends `strategy` + `strategy_params` in POST |
| `dashboard/static/js/live-panel.js` | ✅ Modified | `appendSignalLog()` shows strategy badge + reason when present |
| `dashboard/static/js/arbiter-panel.js` | ✅ Modified | Decision rows show narrative, strategy badge, risk notes; detail modal has strategy + narrative sections |
| `dashboard/templates/live.html` | ✅ Modified | Arbiter detail modal: added strategy info + narrative sections |
| `dashboard/static/css/dashboard.css` | ✅ Modified | `.wt-strategy-badge`, `.wt-narrative-text`, `.wt-signal-reason`, strategy params styles |

### ✅ Phase 8 — Docker & Deployment (COMPLETE)

| File | Status | Description |
|------|--------|-------------|
| `docker-compose.yml` | ✅ Modified | 7 strategy services (strategy-amd-session, etc.) with per-strategy OANDA accounts. Old model services commented out. Dashboard env vars updated with per-strategy OANDA creds. |

### ✅ Phase 9 — Cleanup (COMPLETE)

#### 9.1 — Files DELETED

| File | Status | Notes |
|------|--------|-------|
| `wavetrader/amd_scalper.py` | ✅ Deleted | Model class removed (features stay in `amd_features.py`) |
| `wavetrader/wave_follower.py` | ✅ Deleted | WaveFollower model class |
| `wavetrader/mean_reversion.py` | ✅ Deleted | MeanReversion model class (NOT the strategy) |
| `wavetrader/train_amd_scalper.py` | ✅ Deleted | Training code for removed model |
| `wavetrader/train_wave_follower.py` | ✅ Deleted | Training code for removed model |
| `wavetrader/train_mean_reversion.py` | ✅ Deleted | Training code for removed model |
| `wavetrader/wave_follower_backtest.py` | ✅ Deleted | Pyramid backtest — replaced by strategy_backtest.py |

#### 9.2 — Files MODIFIED (deprecated references removed)

| File | What was removed | Status |
|------|-----------------|--------|
| `wavetrader/config.py` | Removed `AMDScalperConfig`, `MeanRevConfig` | ✅ Done |
| `wavetrader/__init__.py` | Removed imports of MeanReversion, AMDScalper, MeanRevLoss, AMDScalperLoss, AMDScalperConfig, MeanRevConfig | ✅ Done |
| `dashboard/services/backtest_service.py` | Removed model_type branching for wavefollower/meanrev/amd_scalper in `run_backtest_from_config()` and `_load_latest_model()` | ✅ Done |
| `dashboard/services/live_service.py` | Removed model_type branching in `_load_model()` — now only loads WaveTrader MTF | ✅ Done |
| `wavetrader/streaming.py` | Removed WaveFollower branching in `__main__` — now only WaveTrader MTF | ✅ Done |

#### 9.3 — Notebooks ARCHIVED (moved to `archive/`)

| Notebook | Status |
|----------|--------|
| `AMDScalper_Training.ipynb` | ✅ Archived |
| `AMDScalper_Validation.ipynb` | ✅ Archived |
| `MeanReversion_Training.ipynb` | ✅ Archived |
| `MeanReversion_Validation.ipynb` | ✅ Archived |
| `WaveFollower_Training.ipynb` | ✅ Archived |
| `WaveFollower_Validation.ipynb` | ✅ Archived |

**Kept:** `WaveTrader_Colab.ipynb` (MTF training — still needed)

#### 9.4 — Strategy Backtest notebook

| Notebook | Status |
|----------|--------|
| `Strategy_Backtest.ipynb` | ⏭ Deferred (run strategy backtests via dashboard instead) |

---

## Summary

| Phase | Status | Items |
|-------|--------|-------|
| Phase 1: Strategy Framework | ✅ Complete | 11 new files |
| Phase 2: AI Confirmation | ✅ Complete | 1 new file |
| Phase 3: LLM Arbiter | ✅ Complete | 1 modified file |
| Phase 4: Strategy Registry | ✅ Complete | (part of Phase 1) |
| Phase 5: Backtest Pipeline | ✅ Complete | 1 new + 2 modified files |
| Phase 6: Live Streaming | ✅ Complete | 1 new + 1 modified file |
| Phase 7: Dashboard Frontend | ✅ Complete | 8 modified files (strategy params + narrative cards) |
| Phase 8: Docker & Deploy | ✅ Complete | 1 modified file |
| Phase 9: Cleanup | ✅ Complete | 7 deletions, 5 modifications, 6 archives |

**Totals:**  
- ✅ Done: 14 new files created, 22 files modified, 7 files deleted, 6 notebooks archived  
- ❌ Remaining: None — all phases complete  
- **Strategy architecture fully deployed**

---

## Key Specs

### StrategySetup (returned by every strategy)
```python
@dataclass
class StrategySetup:
    direction: Signal          # BUY or SELL (never HOLD)
    entry_price: float         # Current price at detection
    sl_pips: float             # Structure-based stop loss
    tp_pips: float             # Structure-based take profit
    confidence: float          # Strategy confidence 0.0–1.0
    reason: str                # "London swept Asian low → bullish reversal at demand zone"
    trailing_stop_pct: float   # 0 = no trailing
    timestamp: datetime
    context: dict              # Extra data for LLM narrative
```

### ConfirmedSignal (returned by AI Confirmer)
```python
@dataclass
class ConfirmedSignal:
    setup: StrategySetup
    ai_direction_agrees: bool
    ai_confidence: float       # Model confidence (0-1)
    ai_alignment: float        # Multi-TF alignment (0-1)
    combined_confidence: float  # 0.6 × strategy + 0.4 × AI
    trade_signal: TradeSignal   # SL/TP from strategy, not model
```

### ArbiterDecision (LLM output)
```python
@dataclass
class ArbiterDecision:
    action: str                # APPROVE / VETO / OVERRIDE
    narrative: str             # "AMD Session Scalper detected London sweep of Asian low..."
    reasoning: str             # Decision reasoning
    risk_notes: str            # Calendar/portfolio warnings
    strategy_meta: dict        # Strategy name, author, version
```

### API Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/live/strategies` | GET | List all strategies for dropdown |
| `/api/backtest/strategies` | GET | List all strategies for dropdown |
| `/api/backtest/strategies/<id>/params` | GET | Get strategy param schema + meta |
| `/api/backtest/run` | POST | Run backtest (strategy key → strategy path, else model path) |
| `/api/live/stream?strategy=X` | GET | SSE stream for strategy |

### Docker Services

| Service | Strategy | OANDA Env Vars |
|---------|----------|---------------|
| `strategy-amd-session` | AMD Session Scalper | `AMD_OANDA_DEMO_*` |
| `strategy-supply-demand` | Supply & Demand | `SND_OANDA_DEMO_*` |
| `strategy-ict-smc` | ICT/SMC | `ICT_OANDA_DEMO_*` |
| `strategy-orb-breakout` | ORB Breakout | `ORB_OANDA_DEMO_*` |
| `strategy-ema-crossover` | EMA Crossover | `EMA_OANDA_DEMO_*` |
| `strategy-mean-reversion` | Mean Reversion | `MR_OANDA_DEMO_*` |
| `strategy-structure-break` | Structure Break | `SB_OANDA_DEMO_*` |

All share `AI_CHECKPOINT_PATH=/checkpoints` (read-only WaveTrader MTF weights).
