# WaveTrader LLM Signal Arbiter — Complete Reference

> **Version:** 1.0 — April 2026
> **Provider:** Google Gemini 2.5 (Flash / Pro)
> **SDK:** `google-genai` v1.72.0+
> **Source:** `wavetrader/llm_arbiter.py`, `wavetrader/calendar.py`, `wavetrader/llm_logger.py`

---

## 1. Architecture Overview

The LLM Arbiter is a meta-decision layer that sits **between model inference and trade execution**. Every time an AI model (WaveTrader MTF or WaveFollower) produces a BUY/SELL signal, the arbiter receives the signal plus rich market context and decides whether to:

```
Model signal  ──►  LLM Arbiter  ──►  OANDA execution
                       │
                  APPROVE / VETO / OVERRIDE
```

### Data Flow

```
1. Model inference produces: signal, confidence, alignment, SL/TP pips
2. LiveService builds ArbiterContext:
   ├── Signal details (direction, confidence, alignment, SL, TP, entry price)
   ├── Last 30 OHLCV bars from entry timeframe
   ├── Portfolio state (balance, unrealised P&L, drawdown, win rate)
   ├── Open positions from OANDA
   ├── Last 10 closed trades with P&L
   ├── Upcoming economic calendar events (Forex Factory, 4-hour window)
   └── Current trading session (Tokyo / London / New York / Off-hours)
3. LLMArbiter._build_prompt() serialises context into Markdown
4. Gemini API call (system instruction + user prompt)
5. JSON response parsed → ArbiterDecision
6. Authority mode enforced (advisory strips all power, veto strips modifications)
7. Decision applied to signal dict → passed to _execute_signal()
8. Decision logged to JSONL file + broadcast via SSE to dashboard
```

---

## 2. Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google AI Studio API key |
| `LLM_ARBITER_ENABLED` | `false` | Master enable switch |
| `LLM_AUTHORITY_MODE` | `advisory` | `advisory` / `veto` / `override` |
| `LLM_MODEL` | `gemini-2.5-flash` | Primary model |

### LLMArbiterConfig (Python dataclass)

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Master toggle |
| `authority_mode` | str | `"advisory"` | Authority level |
| `model` | str | `"gemini-2.5-flash"` | Primary Gemini model |
| `escalation_model` | str | `"gemini-2.5-pro"` | Used for high-impact news events |
| `api_key_env` | str | `"GEMINI_API_KEY"` | Env var name holding the API key |
| `timeout` | int | `15` | API timeout in seconds |
| `temperature` | float | `0.2` | Low = deterministic, high = creative |
| `escalate_on_high_impact` | bool | `True` | Auto-escalate to Pro model for high-impact events |
| `max_retries` | int | `2` | API retry count before fallback |
| `recent_bars_count` | int | `30` | Number of OHLCV bars included in prompt |
| `recent_trades_count` | int | `10` | Number of recent trades included in prompt |

### API Call Configuration

```python
config = types.GenerateContentConfig(
    system_instruction=system_instruction,   # see Section 3
    temperature=0.2,                         # low for consistency
    max_output_tokens=1024,                  # JSON response is ~200-400 tokens
)
```

---

## 3. System Instruction (System Role)

This is set as the `system_instruction` parameter in the Gemini API call. It defines the LLM's identity, decision rules, and output format:

```
You are a forex trade signal arbiter for an automated trading platform.
You receive signals from AI models and must decide whether to APPROVE, VETO, or OVERRIDE them.

RULES:
1. APPROVE — the signal looks valid given market context. No changes needed.
2. VETO — block the trade. Reasons: upcoming high-impact news, overextended drawdown,
   conflicting higher-timeframe trend, choppy/ranging market, or concentrated risk.
3. OVERRIDE — modify the signal. Only when you have HIGH confidence the adjustment is better.
   You can: flip direction, adjust SL/TP, or adjust confidence.

GUIDELINES:
- Be conservative with VETO and OVERRIDE. When in doubt, APPROVE.
- ALWAYS veto within 30 minutes of high-impact news events for the relevant currencies.
- Consider the model's confidence and alignment scores — high alignment means multiple
  timeframes agree.
- Check trade history: if recent trades are mostly losses, be more cautious.
- Consider session: signals during off-hours (Sydney session) with wide spreads deserve scrutiny.

Respond ONLY with valid JSON matching this schema:
{"action": "APPROVE"|"VETO"|"OVERRIDE", "reasoning": "string (2-4 sentences)",
"confidence_adjustment": float (-0.3 to 0.3),
"modified_signal": null|"BUY"|"SELL"|"HOLD",
"modified_sl_pips": null|float, "modified_tp_pips": null|float,
"risk_notes": "string (calendar/portfolio warnings)"}
```

---

## 4. User Prompt (Dynamic — Built Per Signal)

The user prompt is constructed by `_build_prompt(ctx: ArbiterContext)`. Below is the full template with field substitution:

```markdown
## Signal to Evaluate
Model: {ctx.model_id} | Pair: {ctx.pair} | Timeframe: {ctx.timeframe}
Signal: **{ctx.signal}** | Confidence: {ctx.confidence:.4f} | Alignment: {ctx.alignment:.4f}
Entry: {ctx.entry_price:.5f} | SL: {ctx.sl_pips:.1f} pips | TP: {ctx.tp_pips:.1f} pips

## Market Context
Session: {ctx.current_session}

## Last {N} Bars (newest last)
Time | Open | High | Low | Close | Volume
---|---|---|---|---|---
{bar.time} | {bar.open:.5f} | {bar.high:.5f} | {bar.low:.5f} | {bar.close:.5f} | {bar.volume}
... (up to recent_bars_count rows, default 30)

## Portfolio State
Balance: ${ctx.balance:,.2f} | Unrealised P&L: ${ctx.unrealized_pnl:,.2f}
Max Drawdown: {ctx.max_drawdown:.2%} | Win Rate: {ctx.win_rate:.1%} | Total Trades: {ctx.total_trades}
Open Positions: {count}
  - {direction} {instrument} @ {price:.5f} P&L: ${unrealized_pnl:.2f}

## Last {N} Trades
  {direction} | {open_time} → {close_time} | P&L: ${pnl:+.2f} ({WIN/LOSS})
... (up to recent_trades_count rows, default 10)

## Upcoming Economic Events
  [{impact.UPPER}] {time} — {currency}: {event} (Prev: {previous}, Forecast: {forecast})
... (all events within 4 hours for currencies in the pair)

  (or: "None within the next 4 hours." if no events)

## Decision Required
Based on ALL the above, what is your decision? Respond with JSON only.
```

### Fully Populated Example

```markdown
## Signal to Evaluate
Model: mtf | Pair: GBP/JPY | Timeframe: 15min
Signal: **BUY** | Confidence: 0.8742 | Alignment: 0.6521
Entry: 191.38500 | SL: 25.0 pips | TP: 50.0 pips

## Market Context
Session: London

## Last 30 Bars (newest last)
Time | Open | High | Low | Close | Volume
---|---|---|---|---|---
2026-04-12T09:00:00Z | 190.89200 | 191.02300 | 190.85100 | 190.98700 | 1832
2026-04-12T09:15:00Z | 190.98700 | 191.15400 | 190.94600 | 191.12100 | 1654
2026-04-12T09:30:00Z | 191.12100 | 191.23800 | 191.08900 | 191.20500 | 1423
2026-04-12T09:45:00Z | 191.20500 | 191.31200 | 191.17800 | 191.28900 | 1298
2026-04-12T10:00:00Z | 191.28900 | 191.42100 | 191.25600 | 191.38500 | 1187
...

## Portfolio State
Balance: $25,142.50 | Unrealised P&L: $0.00
Max Drawdown: 3.20% | Win Rate: 62.5% | Total Trades: 24
Open Positions: 0

## Last 10 Trades
  BUY  | 2026-04-12T06:00Z → 2026-04-12T08:30Z | P&L: +$85.20 (WIN)
  SELL | 2026-04-11T14:00Z → 2026-04-11T15:45Z | P&L: -$32.10 (LOSS)
  BUY  | 2026-04-11T10:15Z → 2026-04-11T12:00Z | P&L: +$62.40 (WIN)
  BUY  | 2026-04-11T07:30Z → 2026-04-11T08:15Z | P&L: +$41.80 (WIN)
  SELL | 2026-04-10T19:00Z → 2026-04-10T20:30Z | P&L: -$28.90 (LOSS)
  BUY  | 2026-04-10T14:00Z → 2026-04-10T16:15Z | P&L: +$97.30 (WIN)
  SELL | 2026-04-10T11:00Z → 2026-04-10T12:45Z | P&L: -$15.60 (LOSS)
  BUY  | 2026-04-10T08:30Z → 2026-04-10T10:00Z | P&L: +$53.70 (WIN)
  BUY  | 2026-04-09T14:00Z → 2026-04-09T16:15Z | P&L: +$44.20 (WIN)
  SELL | 2026-04-09T10:00Z → 2026-04-09T11:30Z | P&L: -$21.40 (LOSS)

## Upcoming Economic Events
  [HIGH] 2026-04-12 13:30 — GBP: GDP Monthly (Prev: 0.1%, Forecast: 0.2%)
  [MEDIUM] 2026-04-12 15:00 — JPY: Tankan Large Manufacturing (Prev: 12, Forecast: 14)

## Decision Required
Based on ALL the above, what is your decision? Respond with JSON only.
```

---

## 5. Expected JSON Response Format

The LLM must respond with **only** valid JSON matching this schema:

```json
{
  "action": "APPROVE" | "VETO" | "OVERRIDE",
  "reasoning": "string — 2 to 4 sentences explaining the decision",
  "confidence_adjustment": 0.0,
  "modified_signal": null,
  "modified_sl_pips": null,
  "modified_tp_pips": null,
  "risk_notes": "string — calendar or portfolio risk warnings"
}
```

### Field Definitions

| Field | Type | Constraints | Description |
|---|---|---|---|
| `action` | string | `APPROVE` / `VETO` / `OVERRIDE` | The arbiter's decision |
| `reasoning` | string | 2-4 sentences | Explanation of why |
| `confidence_adjustment` | float | -0.3 to +0.3 | Additive adjustment to model confidence. Only applied in `override` mode |
| `modified_signal` | string or null | `BUY` / `SELL` / `HOLD` / `null` | Replacement signal. Only applied in `override` mode |
| `modified_sl_pips` | float or null | positive | New stop-loss in pips. Only applied in `override` mode |
| `modified_tp_pips` | float or null | positive | New take-profit in pips. Only applied in `override` mode |
| `risk_notes` | string | free text | Warnings about news, drawdown, or position risk |

### Example Responses

**APPROVE — clean signal, no concerns:**
```json
{
  "action": "APPROVE",
  "reasoning": "Strong BUY signal with 87% confidence and 65% alignment across timeframes. Price is trending up with clean higher highs and higher lows over the last 30 bars. London session provides good liquidity.",
  "confidence_adjustment": 0.0,
  "modified_signal": null,
  "modified_sl_pips": null,
  "modified_tp_pips": null,
  "risk_notes": "GBP GDP release in 3.5 hours — monitor but not imminent."
}
```

**VETO — upcoming high-impact news:**
```json
{
  "action": "VETO",
  "reasoning": "High-impact GBP GDP Monthly release at 13:30 UTC, only 22 minutes away. Even though the BUY signal has strong confidence, entering now exposes the position to a news-driven spike that could blow through the 25-pip stop loss instantly.",
  "confidence_adjustment": 0.0,
  "modified_signal": null,
  "modified_sl_pips": null,
  "modified_tp_pips": null,
  "risk_notes": "GBP GDP Monthly in 22min — HIGH impact. Avoid all GBP trades until at least 30 minutes post-release."
}
```

**VETO — losing streak / drawdown:**
```json
{
  "action": "VETO",
  "reasoning": "The last 4 out of 5 trades have been losses, suggesting the current market regime may not suit the model's strategy. Win rate has dropped to 40%. Recommend waiting for a higher-confidence signal or regime confirmation before re-entering.",
  "confidence_adjustment": 0.0,
  "modified_signal": null,
  "modified_sl_pips": null,
  "modified_tp_pips": null,
  "risk_notes": "Win rate at 40% (below historical average of 62%). Consider pausing trading until conditions improve."
}
```

**OVERRIDE — widen stop-loss:**
```json
{
  "action": "OVERRIDE",
  "reasoning": "The BUY signal looks valid, but the 25-pip SL is too tight given recent volatility — the last 10 bars show an average range of 18 pips. Widening SL to 35 pips and TP to 70 pips to maintain 1:2 R:R while avoiding a volatility stop-out.",
  "confidence_adjustment": -0.05,
  "modified_signal": null,
  "modified_sl_pips": 35.0,
  "modified_tp_pips": 70.0,
  "risk_notes": "Elevated intraday volatility — ATR is 1.3x normal for this session."
}
```

**OVERRIDE — flip direction:**
```json
{
  "action": "OVERRIDE",
  "reasoning": "Model says BUY but the last 30 bars show a clear lower-high lower-low structure. The recent bounce is likely a retest of broken support (now resistance) at 191.40. Flipping to SELL with tighter SL above the retest level.",
  "confidence_adjustment": -0.15,
  "modified_signal": "SELL",
  "modified_sl_pips": 20.0,
  "modified_tp_pips": 40.0,
  "risk_notes": "Contrarian override — only valid if 191.45 holds as resistance."
}
```

---

## 6. Response Parsing

The parser (`_parse_response()`) handles:

1. **Clean JSON** — parsed directly
2. **Markdown-wrapped JSON** — strips ` ```json ... ``` ` code fences
3. **JSON embedded in prose** — extracts first `{...}` block via regex
4. **Parse failure** — defaults to `APPROVE` with "No reasoning provided"

### Validation & Clamping

| Field | Validation |
|---|---|
| `action` | Must be `APPROVE`, `VETO`, or `OVERRIDE`. Invalid → `APPROVE` |
| `confidence_adjustment` | Clamped to `[-0.3, +0.3]`. Non-numeric → `0.0` |
| `modified_signal` | Must be `BUY`, `SELL`, or `HOLD`. Invalid → `null` |
| `modified_sl_pips` | Must be numeric. Invalid → `null` |
| `modified_tp_pips` | Must be numeric. Invalid → `null` |

---

## 7. Authority Mode Enforcement

After parsing the LLM response, authority mode constraints are applied **server-side** (the LLM's output is never trusted at face value):

### Advisory Mode
```
action        → forced to APPROVE (always)
modified_*    → all set to null
conf_adj      → forced to 0.0
reasoning     → preserved (logged for review)
risk_notes    → preserved (displayed in dashboard)
```
Trades always execute as the model intended. The LLM's reasoning is logged and displayed but has zero execution power.

### Veto Mode
```
action        → APPROVE or VETO (OVERRIDE demoted to VETO)
modified_*    → all set to null
conf_adj      → forced to 0.0
```
Can block trades (signal forced to HOLD) but cannot modify signals, SL/TP, or confidence.

### Override Mode
```
action        → APPROVE, VETO, or OVERRIDE (no restrictions)
modified_*    → applied if present
conf_adj      → applied (clamped to ±0.3)
```
Full power: can block trades, flip direction, change SL/TP, adjust confidence.

---

## 8. Signal Application

After authority enforcement, `apply_decision()` modifies the signal dict:

| Action | Effect on signal dict |
|---|---|
| `APPROVE` | No changes. Signal executes as-is |
| `VETO` | `signal` → `"HOLD"`, `_arbiter_vetoed` → `True`. Trade does not execute |
| `OVERRIDE` | Applies: `modified_signal`, `confidence + adjustment`, `modified_sl_pips`, `modified_tp_pips`. Sets `_arbiter_overridden` → `True` |

---

## 9. Model Escalation

When `escalate_on_high_impact` is `True` (default) and the `ArbiterContext.has_high_impact_event` flag is `True`:

```
Default model:    gemini-2.5-flash   (~1-2s, $0.15/1M input tokens)
Escalated model:  gemini-2.5-pro     (~3-5s, $1.25/1M input tokens)
```

The escalation is automatic and transparent. The `model_used` field in the decision records which model actually ran.

---

## 10. ArbiterContext — Full Field Reference

All fields populated by `_evaluate_with_arbiter()` in `live_service.py`:

### Signal Fields
| Field | Type | Source |
|---|---|---|
| `signal` | str (`BUY`/`SELL`/`HOLD`) | Model inference |
| `confidence` | float (0-1) | Model inference |
| `alignment` | float (0-1) | Multi-timeframe agreement score |
| `sl_pips` | float | Model inference |
| `tp_pips` | float | Model inference |
| `entry_price` | float | Current candle close price |
| `model_id` | str | Registry ID (e.g. `"mtf"`, `"wavefollower"`) |
| `pair` | str | Trading pair (e.g. `"GBP/JPY"`) |
| `timeframe` | str | Entry timeframe (e.g. `"15min"`) |

### Market Data
| Field | Type | Source |
|---|---|---|
| `recent_bars` | List[dict] | Last 30 OHLCV bars from entry TF (keys: `time`, `open`, `high`, `low`, `close`, `volume`) |
| `current_session` | str | UTC hour mapping: 0-9→Tokyo, 7-16→London, 12-21→New York, else→Off-hours |

### Portfolio State
| Field | Type | Source |
|---|---|---|
| `balance` | float | OANDA account summary |
| `unrealized_pnl` | float | OANDA account summary |
| `open_positions` | List[dict] | OANDA open trades (keys: `trade_id`, `instrument`, `units`, `price`, `unrealized_pnl`, `direction`, `stop_loss`, `take_profit`) |
| `max_drawdown` | float | Currently `0.0` (reserved for future calculation) |
| `win_rate` | float | Computed from `_trade_log`: wins/total |
| `total_trades` | int | Length of `_trade_log` |

### Trade History
| Field | Type | Source |
|---|---|---|
| `recent_trades` | List[dict] | Last 10 entries from `_trade_log` (keys: `signal`, `pair`, `entry_price`, `units`, `sl`, `tp`, `confidence`, `timestamp`, `account`, `trade_id`, `status`, `realized_pl`) |

### Calendar
| Field | Type | Source |
|---|---|---|
| `calendar_events` | List[dict] | Forex Factory events within 4 hours for currencies in pair (keys: `time`, `currency`, `impact`, `event`, `forecast`, `previous`, `actual`) |
| `has_high_impact_event` | bool | `True` if any event has `impact == "high"` |

---

## 11. Economic Calendar

**Source:** Forex Factory via `nfs.faireconomy.media/ff_calendar_thisweek.json` (primary) with HTML scraping fallback.

### Caching
- Events cached in memory for **4 hours**
- Refreshed on first call after cache expiry

### Pair Currency Filtering
Events are filtered to only include currencies in the traded pair:
```
GBP/JPY → shows GBP and JPY events
EUR/USD → shows EUR and USD events
```

### Impact Levels
| Level | Examples |
|---|---|
| `high` | NFP, CPI, GDP, Rate Decision, Employment Change |
| `medium` | PMI, Trade Balance, Retail Sales |
| `low` | Building Permits, Consumer Confidence preliminary |

---

## 12. Decision Logging (JSONL)

All decisions are logged to `logs/llm_decisions.jsonl` in append-only format.

### Log Entry Schema

```json
{
  "decision_id": "a1b2c3d4e5f6",
  "timestamp": "2026-04-12T10:15:32.451Z",
  "model_id": "mtf",
  "pair": "GBP/JPY",
  "original_signal": "BUY",
  "original_confidence": 0.8742,
  "action": "APPROVE",
  "reasoning": "Strong uptrend with multi-TF alignment...",
  "confidence_adjustment": 0.0,
  "modified_signal": null,
  "modified_sl_pips": null,
  "modified_tp_pips": null,
  "risk_notes": "GBP GDP in 3.5 hours — monitor.",
  "model_used": "gemini-2.5-flash",
  "latency_ms": 1342.5,
  "entry_price": 191.385,
  "trade_placed": true,
  "context": {},
  "outcome": null
}
```

### Outcome Backfill

When a trade closes, `log_outcome(decision_id, outcome)` rewrites the log line:

```json
{
  "outcome": {
    "pnl": 85.20,
    "pips": 42.5,
    "result": "win",
    "exit_reason": "TP hit",
    "duration_bars": 12
  }
}
```

### Aggregate Statistics

`get_stats()` computes across all logged decisions:

| Stat | Description |
|---|---|
| `total_decisions` | Total signals evaluated |
| `approvals` | Signals approved |
| `vetoes` | Signals vetoed (blocked) |
| `overrides` | Signals modified |
| `with_outcome` | Decisions with trade outcome recorded |
| `veto_would_have_won` | Vetoed signals that would have been profitable (missed opportunity) |
| `veto_would_have_lost` | Vetoed signals that would have lost money (saved by veto) |
| `override_won` | Overridden signals that were profitable |
| `override_lost` | Overridden signals that lost money |
| `avg_latency_ms` | Average Gemini API response time |

---

## 13. Graceful Degradation

The arbiter **never blocks trading** due to its own failures:

| Failure | Behaviour |
|---|---|
| `GEMINI_API_KEY` not set | Returns `APPROVE` — "LLM arbiter unavailable" |
| `google-genai` not installed | Returns `APPROVE` — "google-genai not installed" |
| API timeout / network error | Retries up to `max_retries` (2), then returns `APPROVE` |
| Invalid JSON response | Defaults to `APPROVE` with "No reasoning provided" |
| Unknown `action` value | Forced to `APPROVE` |
| Arbiter disabled | Returns `APPROVE` — "LLM arbiter disabled" |
| Any uncaught exception | Returns `APPROVE` — logged as error |

**Design principle:** The LLM is an advisor, not a gatekeeper. If it fails, trading continues normally.

---

## 14. Dashboard Integration

### SSE Events

When a decision is made, the following SSE event is pushed to all connected dashboard clients:

```
event: arbiter
data: {"decision_id":"a1b2c3","action":"VETO","reasoning":"...","original_signal":"BUY","trade_placed":false,...}
```

### REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/live/arbiter/status?model=mtf` | Current config + aggregate stats |
| `POST` | `/api/live/arbiter/config?model=mtf` | Update enabled/mode/model at runtime |
| `GET` | `/api/live/arbiter/decisions?model=mtf&count=50` | Recent decisions list |
| `GET` | `/api/live/arbiter/stats?model=mtf` | Aggregate statistics only |
| `GET` | `/api/live/arbiter/calendar?pair=GBP/JPY&hours=24` | Upcoming calendar events |

### Dashboard Tab

The "LLM Arbiter" tab displays:
- **Header:** Status badge (mode), authority mode dropdown, enable/disable toggle
- **Stats bar:** Total decisions, approved, vetoed, overrides, veto-saved, veto-missed, avg latency
- **Decision list:** Scrollable list of all decisions, colour-coded by action (green=approve, red=veto, yellow=override)
- **Detail modal:** Click any decision to see full reasoning, risk notes, signal details grid, outcome tracker, calendar context

---

## 15. Cost Estimates

| Scenario | Calls/Day | Model | Est. Cost/Day |
|---|---|---|---|
| 15min bars, 1 pair, no news | ~96 | Flash | ~$0.01 |
| 15min bars, 1 pair, with news escalation | ~94 Flash + ~2 Pro | Mixed | ~$0.02 |
| 2 models × 15min bars | ~192 | Flash | ~$0.02 |
| All above + Trade Manager (60s) @ 1 trade | ~1,440 + ~192 | Flash | ~$0.16 |

Token estimates per call: ~2,000 input tokens (prompt), ~200 output tokens (JSON response).

---

## 16. Security

- API key is read from environment variable, never hardcoded or logged
- The LLM cannot execute trades directly — it only returns a JSON recommendation
- Authority mode enforcement is server-side (`_enforce_authority()`) — the LLM's response is clamped regardless of what it says
- Confidence adjustment is clamped to ±0.3 to prevent a single LLM call from dramatically altering position sizing
- All decisions are logged for audit trail
- Graceful degradation ensures the LLM can never halt trading by failing
