# WaveTrader Training Gap Management & Live Warm-Up Protocol

## 1. The Temporal Memory Problem
**Why You Cannot Start Cold**
WaveTrader is stateful, not stateless like most ML models:
*   **CWC Hidden State:** 100-bar rolling window (25 hours of market structure memory)
*   **Resonance Buffer:** Recent "experiences" for few-shot retrieval
*   **Regime Detector:** Rolling volatility statistics (ATR percentiles)

Starting without warm-up = Trading blind for 25 hours. The model generates invalid signals until the lookback window fills.

---

## 2. The 100-Bar Rule
**Minimum Warm-Up Requirements**

| Timeframe | Bars Needed | Hours of History | Data Points |
| --------- | ----------- | ---------------- | ----------- |
| 15m (Entry)| 100         | 25 hours         | 1,600       |
| 1h (Confirm)| 100        | 100 hours        | 400         |
| 4h (Trend) | 50          | 200 hours        | 200         |

**Constraint:** The 15m lookback is the bottleneck. You need 25 continuous hours of recent data before the first valid signal.

---

## 3. Gap Scenarios & Solutions

### Scenario A: Training End → Live Start (Normal Deployment)
*   **Training data ends:** April 4, 2026 (Friday)
*   **Live trading starts:** April 6, 2026 (Sunday market open)
*   **Action:**
    1.  Download April 4-5 data (weekend = minimal).
    2.  Run "silent mode" (inference without execution) on this historical data.
    3.  Save checkpoint after processing 100+ bars.
    4.  Resume from checkpoint on Sunday 22:00 with live WebSocket.

### Scenario B: Extended Downtime (Crash/Maintenance)
*   **Last checkpoint:** Monday 10:00 UTC
*   **Recovery:** Wednesday 14:00 UTC (52-hour gap)
*   **Action:**
    1.  CRITICAL: Do NOT trade immediately.
    2.  Download missing 52 hours of historical data from OANDA.
    3.  Replay through model at 2x speed (batch inference) silently.
    4.  Verify state consistency and save. Resume live.

### Scenario C: New Pair Addition (e.g., Adding USDJPY Back)
*   **Action:**
    1.  Download USDJPY historical data from training end date to now.
    2.  Run offline synchronization (process USDJPY history).
    3.  Merge USDJPY state into live model (or restart from checkpoint with 4-pair model and warm up all pairs together).

---

## 4. Backfill Data Sources
**Primary: OANDA Historical API**
```python
# Fetch missing bars
params = {
    'from': last_checkpoint_time.isoformat(),
    'to': now().isoformat(),
    'granularity': 'M15',
    'price': 'MBA'  # Mid, Bid, Ask
}
response = requests.get(f"{OANDA_API}/v3/instruments/{pair}/candles", 
                       params=params, headers=auth)
```
*   **Rate limit:** 100 requests/second.
*   **Max Limit:** 500 candles per request (paginate for large gaps).

---

## 5. State Synchronization Protocol

**Step-by-Step Warm-Up Procedure**

1.  **Data Verification (Manual Check)**
    *   Find how stale the checkpoint is using `torch.load()`. If >25 hours old, proceed to backfill.
2.  **Historical Replay (Automated loop)**
    ```python
    engine = StreamingEngine(model, crystal_store)
    engine.load_checkpoint('checkpoint.pt')
    gap_data = download_oanda_history(from_time=ckpt['timestamp'], to_time=now())
    
    # Process silently (no trading)
    for bar in gap_data: 
        engine.ingest_bar(bar, execute=False)
    
    engine.save_checkpoint('checkpoint_warm.pt')
    ```
3.  **Validation (Manual Check)**
    *   Verify Resonance Buffer contains recent events.
    *   Verify CWC state shape is normal (no NaNs).
4.  **Live Transition**
    ```python
    live_engine = StreamingEngine(model, crystal_store)
    live_engine.load_checkpoint('checkpoint_warm.pt')
    live_engine.connect_websocket(oanda_credentials)  # execute=True
    ```

---

## 6. Edge Cases & Failure Modes

### Sub-Bar Interruptions (WebSocket Drops)
*   **The Issue:** High volatility triggers brief WebSocket lag/reconnects (10-30 seconds). An incomplete OHLC tick stream can distort the 15-minute bar.
*   **Action Logic:**
    1.  Hold partial bar in-memory.
    2.  On reconnect, perform a rapid REST check against the current 15m candle.
    3.  If `open` or `high`/`low` differs significantly, fast-fetch the entire candle from REST to patch the missing sub-bar ticks. Do **not** process a partial block as the final candle close.

### Timezone/DST Transitions
*   **The Issue:** Handling DST conversions introduces length offsets during missing gaps.
*   **Action Logic:**
    1.  All data is standard UTC (conversion logic via `zoneinfo` explicitly shifts EET timestamps to naive UTC natively).
    2.  Instead of checking specific offset dates, simply verify that your 15m resampling logic **did not output any 30-minute gaps** when reconstructing bars near missing/outage days.
    
**Golden Rule:** If the model has been asleep for more than 24 hours, it must dream (backfill) before it can trade.