# WaveTrader Live Deployment Specification

## 1. Infrastructure Architecture
**Recommended Stack:** VPS + Containerization
*   **Primary Recommendation:** Hetzner Cloud (CPX31) or DigitalOcean (Premium AMD)
*   **Specs:** 4 vCPU, 8GB RAM, 160GB NVMe SSD
*   **Location:** Frankfurt, Germany (latency to OANDA London: ~15ms)
*   **Cost:** €12.50/month vs $50+/month for Heroku
*   **OS:** Ubuntu 22.04 LTS

**Why Not Heroku/Railway/Render:**
*   Dyno sleeping (Heroku): Missed bars during "wake up"
*   Ephemeral filesystem: CWC state and Resonance Buffer lost on restart
*   Cost: 2-4x more expensive for 24/5 operation
*   No GPU option if you later scale to tick-level inference

---

## 2. Deployment Components
**Docker Configuration**

`Dockerfile` (production):
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# Pin PyTorch version to avoid breaking changes in future rebuilds
RUN pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY wavetrader/ ./wavetrader/
COPY config/ ./config/
VOLUME ["/data"]  # Persistent state storage
CMD ["python", "-m", "wavetrader.streaming"]
```

`docker-compose.yml`:
```yaml
version: '3.8'
services:
  wavetrader:
    image: wavetrader:latest
    restart: unless-stopped
    volumes:
      - ./data:/data:rw
      - ./checkpoints:/checkpoints:rw
    environment:
      - OANDA_API_KEY=${OANDA_API_KEY}
      - CHECKPOINT_INTERVAL=100
      - PAIRS=GBPJPY,EURJPY,GBPUSD
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
    # Redis is only used for caching the structural Resonance Buffer
    command: redis-server --appendonly yes --save 60 1000
```

---

## 3. State Persistence Strategy (Critical)
**The Problem:**
If the process restarts cold, it needs 25 hours to "catch up" before producing valid signals.
*   **CWC (Causal Wave Chainer) State:** Holds 100-bar hidden state (~25 hours of memory) and must be saved to persistent disk.
*   **Resonance Buffer:** Contains "memories" of recent high-volatility events, typically stored in Redis.

**Solution: Continuous Checkpointing** (Every 100 bars)
Do not rely on Redis for the main neural network state. Use explicit `torch.save` to dump the exact structure directly to your persistent `/data/checkpoints` volume:
```python
# Pseudo-code for deployment
checkpoint = {
    'timestamp': last_bar.isoformat(),
    'cwc_hidden': model.cwc.get_states(),  # [num_layers, batch, hidden_dim]
    'resonance_buffer': crystal_store.serialize(), # Backup from Redis occasionally too
    'last_predictions': prediction_buffer,
    'equity_state': backtest_engine.get_state(),
    'model_version': '2.0.1',
}
torch.save(checkpoint, f'/data/checkpoints/checkpoint_{int(time.time())}.pt')
```

---

## 4. Monitoring & Alerting
**Health Metrics:**
*   **Inference latency:** <50ms per bar 
*   **OANDA connection:** WebSocket heartbeat
*   **Signal generation:** Count of BUY/SELL/HOLD per hour
*   **Drawdown:** Current equity vs peak

**Alert Channels (Tier 1 - Critical):**
*   Circuit breaker triggered (halt trading)
*   Position size mismatch (risk management failure)
*   Checkpoint save failure (state loss imminent)