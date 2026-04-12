---
name: add-model
description: "Add a new AI model to the WaveTrader platform end-to-end. USE WHEN: creating a new trading model, setting up training/validation notebooks with GPU optimizations, deploying model to OANDA with its own account, updating the dashboard dropdown, regenerating API keys after adding accounts, fixing 403 Forbidden errors from OANDA. Covers: training notebook, validation+backtest notebook, checkpoint saving, OANDA account binding, MODEL_REGISTRY, docker-compose service, frontend dropdown, API key lifecycle."
argument-hint: "Model name, type (mtf/wavefollower/custom), and target pair"
---

# Add New Model — Full Deployment Pipeline

Add a new AI model to the WaveTrader platform: training notebook → validation notebook → OANDA account → Docker service → dashboard dropdown → live trading.

**Every model MUST have exactly 2 notebooks** (never combined):
1. `{ModelName}_Training.ipynb` — GPU-optimized training only
2. `{ModelName}_Validation.ipynb` — holdout eval + backtest + friction sim only

Existing examples:
| Model | Training Notebook | Validation Notebook |
|-------|------------------|-------------------|
| WaveTrader MTF | `WaveTrader_Colab.ipynb` | (included in same notebook — legacy) |
| WaveFollower | `WaveFollower_Training.ipynb` | `WaveFollower_Validation.ipynb` |

**WaveFollower_Training.ipynb is the gold standard** — all new training notebooks MUST match its GPU optimizations. See [optimization reference](./references/gpu-optimizations.md) for the full comparison.

## When to Use

- Training a new trading model from scratch
- Setting up GPU-optimized Colab/local training notebooks
- Creating a validation & backtesting notebook for a new model
- Deploying a trained model to OANDA with its own account
- Fixing 403 Forbidden errors after adding OANDA sub-accounts
- Updating the dashboard model selector dropdown
- Regenerating API keys after account changes

## Key Constraint: OANDA API Key Lifecycle

> **CRITICAL**: Every new OANDA sub-account under the same user requires the API key to be **revoked and regenerated**. Until the new key is active, ALL models return **403 Forbidden**. Plan for downtime.

## Procedure

### Phase 1: Naming & Configuration

1. **Choose model identifiers** — used everywhere:
   - `MODEL_TAG`: short lowercase id (e.g., `wavebreaker`, `scalper_v2`)
   - `MODEL_NAME`: display name (e.g., `WaveBreaker`, `Scalper V2`)
   - `MODEL_TYPE`: architecture class — `mtf`, `wavefollower`, or new class name
   - `TARGET_PAIR`: e.g., `GBP/JPY`
   - `PAIR_TAG`: e.g., `GBPJPY` (no slash, used in filenames)

2. **Choose env var prefix** — UPPERCASE, unique per model:
   - Pattern: `{PREFIX}_OANDA_DEMO_ACCOUNT_ID`, `{PREFIX}_OANDA_DEMO_API_KEY`
   - Examples: `WB_`, `SV2_`, `SC_`
   - Existing prefixes: `MTF_` (WaveTrader MTF), `WF_` (WaveFollower)

### Phase 2: Create Training Notebook

Create `{ModelName}_Training.ipynb` — modeled after `WaveFollower_Training.ipynb` (the fastest training setup). See [training template](./templates/training-notebook.md).

**MANDATORY GPU Optimizations** (all from WaveFollower — do NOT skip any):

| Optimization | Where | Impact |
|---|---|---|
| `cudnn.benchmark = True` | Cell 1 | Auto-tunes conv kernels |
| `allow_tf32 = True` (cuda + cudnn) | Cell 1 | 2-3x matmul speedup on Ampere+ |
| `torch.compile(model, mode="reduce-overhead")` | After model init | 15-30% kernel fusion |
| `autocast("cuda")` + `GradScaler` | Training loop | FP16 mixed precision, halves VRAM |
| `scaler.unscale_()` before `clip_grad_norm_` | Training loop | Correct gradient clipping order |
| `optimizer.zero_grad(set_to_none=True)` | Training loop | Avoids memset overhead |
| `AdamW(fused=True)` | Optimizer init | Fewer kernel launches |
| `PreCachedMTFDataset` | Dataset build | Eliminates pandas overhead per batch |
| `pin_memory=True, persistent_workers=True` | DataLoader | Zero-copy GPU transfers |
| `prefetch_factor=4, num_workers=4` | DataLoader | Overlaps CPU/GPU work |
| `.to(device, non_blocking=True)` | Batch transfer | Async H2D copies |
| `drop_last=True` on train loader | DataLoader | Avoids small last-batch slowdown |
| Val batch = 2x train batch | DataLoader | No grads = spare VRAM |
| Batch size 640 (T4/16GB) | Config | Fill GPU utilization |
| LR scaled: `lr *= batch_size / 128` | Config | Linear scaling rule |

```python
# Cell 1: Performance flags — ALWAYS set these
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Training loop MUST include** (inline, NOT delegated to `train_model()`):
- Custom `train_one_epoch()` + `validate()` functions with full AMP support
- `torch.amp.autocast("cuda")` + `GradScaler` for mixed-precision (FP16)
- `GradScaler.unscale_()` before `clip_grad_norm_` for safe gradient clipping
- `torch.compile(model)` after model creation (PyTorch 2.0+, ~15-30% speedup)
- Batch size scaling: start at 128, scale up to GPU VRAM limit (640 for T4/16GB)
- Learning rate scaling: linear rule — `lr *= batch_size / 128`
- `pin_memory=True` and `num_workers=4` on DataLoader
- Cosine annealing scheduler: `CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)`
- Early stopping on validation accuracy (patience=10)
- Per-epoch timing and progress logging

**Training notebook cell structure** (13 cells — matches WaveFollower_Training.ipynb):
1. Header (markdown) — model description, OANDA account
2. Imports & GPU Check (performance flags + VRAM report)
3. Configuration & Data Paths (batch size, LR, checkpoint dir)
4. Check processed data availability
5. Load data from processed parquets (or raw CSV fallback)
6. Verify data format compatibility (feature columns check)
7. Build `PreCachedMTFDataset` + DataLoaders (GPU-optimized settings)
8. Instantiate model + `torch.compile` + forward pass sanity check
9. Loss, optimizer (fused AdamW), scheduler, mixed precision setup
10. Training Loop — custom inline with AMP, per-epoch val, best checkpoint save
11. Training curves (Plotly: loss, accuracy, LR, overfit gap)
12. Test set evaluation (confusion matrix, per-class accuracy)
13. Save final checkpoint + training metadata + SHA-256 + history + Google Drive copy

**Checkpoint save pattern** (Cell 8):

```python
import hashlib, json
from datetime import datetime

ckpt_name = f"{MODEL_TAG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ckpt_dir = CHECKPOINT_DIR / ckpt_name
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Save weights
weights_path = ckpt_dir / "model_weights.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "training": {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "epochs_trained": epoch + 1,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    },
}, str(weights_path))

# Save config
with open(ckpt_dir / "config.json", "w") as f:
    json.dump(config.__dict__ if hasattr(config, '__dict__') else
              {k: getattr(config, k) for k in config.__dataclass_fields__}, f, indent=2, default=str)

# SHA-256 integrity hash
h = hashlib.sha256()
with open(weights_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        h.update(chunk)
(ckpt_dir / "weights.sha256").write_text(f"{h.hexdigest()}  model_weights.pt\n")

# Training history
with open(ckpt_dir / "history.json", "w") as f:
    json.dump({"train_loss": train_losses, "val_loss": val_losses,
               "val_accuracy": val_accs}, f)

print(f"Checkpoint saved: {ckpt_dir}")
```

### Phase 3: Create Validation & Backtesting Notebook (SEPARATE file)

Create `{ModelName}_Validation.ipynb` — a **separate notebook** from training. Modeled after `WaveFollower_Validation.ipynb`. See [validation template](./templates/validation-notebook.md).

> **Why separate?** Training runs on Colab with GPU. Validation loads a checkpoint and can run locally or on Colab. Keeping them separate means you can re-validate without re-training, and share validation results without exposing training code.

**Validation notebook cell structure** (16 cells — matches WaveFollower_Validation.ipynb):
1. Header (markdown) — model name, OANDA account, backtest parameters table
2. Setup & Mount Drive / set paths
3. Clone repo & install dependencies
4. Load model from checkpoint (SHA-256 verify, `_orig_mod.` stripping)
5. Load test data for all target pairs
6. Holdout evaluation — accuracy, loss, progress logging
7. Detailed metrics — F1, confusion matrix, confidence distribution, calibration curve
8. Per-pair backtest with OANDA-realistic config
9. Primary pair deep dive — period breakdown (daily/weekly/monthly/yearly)
10. Full week-by-week table
11. Timeline visualizations (4 subplots: PnL+balance, monthly bars, rolling WR, exit reasons)
12. Session & time-of-day analysis (hourly trades, PnL by hour, day-of-week, session table)
13. Friction simulation (slippage, spread widening, news spikes, lot caps, latency misses)
14. Equity curve overlay + sample trades table
15. Save all results to Drive / `backtest_results/`

**OANDA-realistic backtest config**:

```python
BacktestConfig(
    initial_balance=100.0,        # Small account start
    risk_per_trade=0.10,          # 10% for growth
    min_confidence=0.55,          # Only confident trades
    spread_pips=OANDA_SPREADS[pair],  # Pair-specific
    pip_value=PIP_VALUES[pair],
    commission_per_lot=0.0,       # OANDA Standard
    leverage=30.0,                # ESMA retail
    atr_halt_multiplier=3.0,
    drawdown_reduce_threshold=0.15,
)
```

### Phase 4: OANDA Account Setup

> **⚠️ This causes downtime for ALL models until the new API key is active.**

1. **Create sub-account on OANDA**:
   - Log in to [OANDA Hub](https://hub.oanda.com/) (demo: fxpractice hub)
   - Navigate to Account Management → Create Sub-Account
   - Note the new account ID (format: `101-001-XXXXXXXX-NNN`)

2. **Revoke and regenerate API key**:
   - Go to Manage API Access → Revoke existing token
   - Generate new token — this token now covers ALL sub-accounts
   - **ALL existing models will 403 until you update the key everywhere**

3. **Update `.env` on VPS** (`/opt/wavetrader/.env`):

   ```bash
   # New model credentials
   {PREFIX}_OANDA_DEMO_API_KEY=<new-regenerated-key>
   {PREFIX}_OANDA_DEMO_ACCOUNT_ID=101-001-XXXXXXXX-NNN
   
   # UPDATE ALL EXISTING MODELS with the new key too!
   OANDA_DEMO_API_KEY=<new-regenerated-key>       # MTF
   WF_OANDA_DEMO_API_KEY=<new-regenerated-key>    # WaveFollower
   ```

4. **Verify 403 is resolved**:
   ```bash
   curl -s -H "Authorization: Bearer <new-key>" \
     "https://api-fxpractice.oanda.com/v3/accounts/<new-account-id>/summary" | jq .
   ```
   - If still 403: wait 1-2 minutes for OANDA propagation
   - If persistent: verify token is for correct user, account is linked

### Phase 5: Register in Model Registry

1. **Update `MODEL_REGISTRY` in `.env`**:

   ```json
   MODEL_REGISTRY=[
     {"id":"mtf","name":"WaveTrader MTF","pair":"GBP/JPY","model_type":"mtf",
      "demo_api_key_env":"OANDA_DEMO_API_KEY","demo_account_id_env":"OANDA_DEMO_ACCOUNT_ID",
      "checkpoint_dir":"checkpoints/wavetrader_mtf_GBPJPY_20260404_235854"},
     {"id":"wavefollower","name":"WaveFollower","pair":"GBP/JPY","model_type":"wavefollower",
      "description":"Trend-following with pullback pyramiding",
      "demo_api_key_env":"WF_OANDA_DEMO_API_KEY","demo_account_id_env":"WF_OANDA_DEMO_ACCOUNT_ID",
      "checkpoint_dir":"checkpoints/wavefollower_20260412_023912",
      "results_dir":"wavefollower_20260412_023912"},
     {"id":"{MODEL_TAG}","name":"{MODEL_NAME}","pair":"{TARGET_PAIR}",
      "model_type":"{MODEL_TYPE}",
      "description":"<brief description>",
      "demo_api_key_env":"{PREFIX}_OANDA_DEMO_API_KEY",
      "demo_account_id_env":"{PREFIX}_OANDA_DEMO_ACCOUNT_ID",
      "checkpoint_dir":"checkpoints/{ckpt_name}",
      "results_dir":"{ckpt_name}"}
   ]
   ```

2. **Fields reference** (from `dashboard/services/model_registry.py`):
   - `id`: unique slug, used in API `?model=` param
   - `name`: display name in dropdown
   - `pair`: trading pair
   - `model_type`: `mtf` | `wavefollower` | custom
   - `description`: shown in UI tooltip
   - `demo_api_key_env` / `demo_account_id_env`: env var names (NOT raw keys)
   - `checkpoint_dir`: relative path to checkpoint folder
   - `results_dir`: subfolder under `backtest_results/`

### Phase 6: Docker Compose Service

Add a new service block to `docker-compose.yml`:

```yaml
  # ── Model C: {MODEL_NAME} ──────────────────────────────────────
  wavetrader-{MODEL_TAG}:
    build: .
    image: wavetrader:latest
    container_name: wavetrader-{MODEL_TAG}
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "kill -0 1"]
      interval: 30s
      timeout: 5s
      retries: 3
    volumes:
      - ./data:/data:rw
      - ./checkpoints:/checkpoints:rw
    env_file:
      - .env
    environment:
      - CHECKPOINT_DIR=/data/checkpoints
      - CHECKPOINT_INTERVAL=100
      - MODEL_ID={MODEL_TAG}
      - OANDA_API_KEY=${{{PREFIX}_OANDA_DEMO_API_KEY}}
      - OANDA_ACCOUNT_ID=${{{PREFIX}_OANDA_DEMO_ACCOUNT_ID}}
      - OANDA_DEMO_API_KEY=${{{PREFIX}_OANDA_DEMO_API_KEY}}
      - OANDA_DEMO_ACCOUNT_ID=${{{PREFIX}_OANDA_DEMO_ACCOUNT_ID}}
      - OANDA_ENVIRONMENT=practice
      - CHECKPOINT_PATH=/checkpoints/{ckpt_name}/model_weights.pt
      - PAIR={TARGET_PAIR}
      - MIN_CONFIDENCE=0.55
    depends_on:
      - redis
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
```

**Also update the dashboard service** — pass the new model's env vars:

```yaml
  dashboard:
    environment:
      # ... existing vars ...
      # Model C ({MODEL_NAME}) — OANDA credentials
      - {PREFIX}_OANDA_DEMO_API_KEY=${{{PREFIX}_OANDA_DEMO_API_KEY:-}}
      - {PREFIX}_OANDA_DEMO_ACCOUNT_ID=${{{PREFIX}_OANDA_DEMO_ACCOUNT_ID:-}}
```

### Phase 7: Frontend Dropdown

The frontend auto-populates from `GET /api/live/models` → `model_registry.to_list()`. No JS changes needed if MODEL_REGISTRY is set correctly.

**Verify**:
1. Dashboard loads → model dropdown shows new model
2. Switching to new model queries its OANDA account
3. SSE stream reconnects with `?model={MODEL_TAG}`

Files involved (read-only verification):
- `dashboard/static/js/live-init.js` — `loadModelSelector()`, `switchModel()`
- `dashboard/routes/live.py` — `GET /api/live/models`
- `dashboard/templates/index.html` — `#nav-model-select`

### Phase 8: Deploy

```bash
# On VPS (104.207.143.54)
cd /opt/wavetrader

# 1. Pull latest code
git pull origin main

# 2. Upload checkpoint (from local/colab)
scp -r checkpoints/{ckpt_name} root@104.207.143.54:/opt/wavetrader/checkpoints/

# 3. Update .env (already done in Phase 4)

# 4. Rebuild & restart all services
docker compose build
docker compose up -d

# 5. Verify all containers are healthy
docker compose ps
docker logs wavetrader-{MODEL_TAG} --tail 50

# 6. Verify dashboard sees the new model
curl -s http://localhost:5000/api/live/models | jq .

# 7. Check OANDA connectivity per model
curl -s http://localhost:5000/api/live/account?model={MODEL_TAG} | jq .
```

## Troubleshooting

### 403 Forbidden from OANDA
**Cause**: API key was revoked during sub-account creation but not yet regenerated/updated.  
**Fix**:
1. Regenerate API key in OANDA Hub
2. Update ALL `*_OANDA_DEMO_API_KEY` vars in `.env` with the new key
3. `docker compose restart`
4. Wait 1-2 min for OANDA propagation

### Model not in dropdown
**Cause**: `MODEL_REGISTRY` env var not set or JSON is malformed.  
**Fix**:
1. Verify JSON: `echo $MODEL_REGISTRY | python -m json.tool`
2. Check dashboard logs: `docker logs wavetrader-dashboard | grep -i registry`
3. Test endpoint: `curl localhost:5000/api/live/models`

### Checkpoint load fails (`_orig_mod.` prefix)
**Cause**: Model was saved after `torch.compile()`.  
**Fix**: Strip prefix during load:
```python
cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(cleaned, strict=False)
```

### Shape mismatch on `load_state_dict`
**Cause**: Architecture evolved between training and deployment.  
**Fix**: Filter mismatched keys:
```python
model_sd = model.state_dict()
filtered = {k: v for k, v in state_dict.items()
            if k in model_sd and v.shape == model_sd[k].shape}
model.load_state_dict(filtered, strict=False)
```

### Container exits immediately
**Cause**: Missing checkpoint file or OANDA credentials.  
**Fix**: Check `docker logs wavetrader-{MODEL_TAG} --tail 100` for the specific error.

## Checklist

- [ ] Model trained to >55% accuracy (5-fold CV or holdout)
- [ ] Checkpoint saved: `model_weights.pt` + `config.json` + `weights.sha256` + `history.json`
- [ ] Validation notebook passes: holdout eval + backtest + friction sim
- [ ] OANDA sub-account created
- [ ] API key regenerated and updated in ALL `.env` entries
- [ ] 403 resolved — `curl` returns account summary
- [ ] `MODEL_REGISTRY` JSON updated with new model entry
- [ ] `docker-compose.yml` has new service + dashboard env vars
- [ ] `docker compose up -d` — all containers healthy
- [ ] Dashboard dropdown shows new model
- [ ] Model's OANDA account queries work in dashboard
- [ ] Paper-trade 50+ bars — check signals in logs
- [ ] Backtest results saved to `backtest_results/{ckpt_name}/`
