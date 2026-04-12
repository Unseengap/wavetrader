---
name: add-model
description: "Add a new AI model to the WaveTrader platform end-to-end. USE WHEN: creating a new trading model, setting up training/validation notebooks with GPU optimizations, deploying model to OANDA with its own account, updating the dashboard dropdown, regenerating API keys after adding accounts, fixing 403 Forbidden errors from OANDA. Covers: model class, config, loss, training code, backtest code, dashboard integration, training notebook, validation+backtest notebook, checkpoint saving, OANDA account binding, MODEL_REGISTRY, docker-compose service, frontend dropdown, API key lifecycle."
argument-hint: "Model name, type (mtf/wavefollower/custom), and target pair"
---

# Add New Model — Full Deployment Pipeline

Add a new AI model to the WaveTrader platform end-to-end.

> **Phase order is strict**: model code → training code → backtest code → dashboard integration → notebooks → OANDA → docker → deploy. ALL Python code must exist and import correctly BEFORE any notebook is created.

**Every model MUST have exactly 2 notebooks** (never combined):
1. `{ModelName}_Training.ipynb` — GPU-optimized training only
2. `{ModelName}_Validation.ipynb` — holdout eval + backtest + friction sim only

Existing examples:
| Model | Training Notebook | Validation Notebook |
|-------|------------------|-------------------|
| WaveTrader MTF | `WaveTrader_Colab.ipynb` | (included in same notebook — legacy) |
| WaveFollower | `WaveFollower_Training.ipynb` | `WaveFollower_Validation.ipynb` |

**WaveFollower_Training.ipynb is the gold standard** — all new training notebooks MUST match its GPU optimizations. See [optimization reference](./references/gpu-optimizations.md).

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

---

### Phase 2: Model Architecture Code (BEFORE any notebook)

All code files MUST be created and importable BEFORE creating notebooks. The notebooks just call this code.

#### 2A. Config dataclass — `wavetrader/config.py`

Add a new config dataclass. Must include standard fields:

```python
@dataclass
class {ModelName}Config:
    pair: str = "GBP/JPY"
    timeframes: list = field(default_factory=lambda: ["15min", "1h", "4h", "1d"])
    lookbacks: dict = field(default_factory=lambda: {"15min": 100, "1h": 100, "4h": 100, "1d": 50})
    # Architecture
    tf_wave_dim: int = 256
    fused_dim: int = 512
    predictor_layers: int = 4
    predictor_heads: int = 8
    dropout: float = 0.15
    n_signal_classes: int = 3  # BUY/SELL/HOLD
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 640
    epochs: int = 60

    @property
    def output_wave_dim(self) -> int:
        return self.fused_dim
```

#### 2B. Model class — `wavetrader/{model_tag}.py` (new file)

Follow the output contract — all models must produce:

```python
class {ModelName}(nn.Module):
    def __init__(self, config: {ModelName}Config):
        ...

    def forward(self, batch: dict) -> dict:
        # REQUIRED outputs:
        return {
            "signal_logits": tensor [B, 3],   # BUY/SELL/HOLD
            "confidence": tensor [B, 1],       # 0-1 calibrated
            "risk_params": tensor [B, 3],      # SL mult, TP mult, trailing pct
        }
        # OPTIONAL (model-specific):
        # "trend_logits": tensor [B, 3],      # UP/DOWN/NEUTRAL
        # "add_score": tensor [B, 1],         # pyramiding score

    def predict(self, batch: dict) -> TradeSignal:
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

Reuse existing encoders from `wavetrader/encoders.py` where possible (PriceWaveEncoder, StructureWaveEncoder, RSIWaveEncoder, VolumeWaveEncoder, RegimeEncoder).

#### 2C. Export in `wavetrader/__init__.py`

```python
from .{model_tag} import {ModelName}, {ModelName}Config
```

### Phase 3: Training Code (BEFORE notebooks)

#### 3A. Loss function — `wavetrader/train_{model_tag}.py` (new file)

If the model has extra outputs beyond signal/confidence (e.g. trend, add_score), create a custom loss. Otherwise reuse `SignalLoss` from `wavetrader/training.py`.

```python
class {ModelName}Loss(nn.Module):
    def __init__(self, signal_weight=1.0, conf_weight=0.1, ...):
        ...

    def forward(self, output: dict, labels: torch.Tensor, ...) -> dict:
        # Must return dict with "total" key + component keys
        return {
            "total": combined_loss,
            "signal_loss": signal_ce,
            "conf_loss": conf_mse,
            ...
        }
```

#### 3B. Export training code in `wavetrader/__init__.py`

### Phase 4: Backtest Code (BEFORE notebooks)

#### 4A. Custom backtest (OPTIONAL) — `wavetrader/{model_tag}_backtest.py`

Only needed if the model has custom exit logic (e.g., WaveFollower's structure-break exits and pyramiding). Otherwise `run_backtest()` from `wavetrader/backtest.py` auto-dispatches based on `hasattr(config, 'timeframes')`.

#### 4B. Export backtest code in `wavetrader/__init__.py` (if custom backtest)

### Phase 5: Dashboard Integration (BEFORE notebooks)

The dashboard has **hardcoded model-type if/else chains** in 2 service files. These MUST be updated for a new model type to work in both live and backtest modes.

#### 5A. Live service — `dashboard/services/live_service.py`

**Two locations** with model-type if/else chains:

**Location 1: Checkpoint prefix (~line 742)**
```python
# Add new model type:
if model_type == "wavefollower":
    prefix = "wavefollower_"
elif model_type == "{MODEL_TYPE}":
    prefix = "{MODEL_TAG}_"
else:
    prefix = "wavetrader_mtf_"
```

**Location 2: Model instantiation (~line 754)**
```python
# Add elif BEFORE the else clause:
elif model_type == "{MODEL_TYPE}":
    from wavetrader.{model_tag} import {ModelName}, {ModelName}Config
    self._model_config = {ModelName}Config(pair=self._pair)
    self._model = {ModelName}(self._model_config)
```

#### 5B. Backtest service — `dashboard/services/backtest_service.py`

**Three locations** with model-type checks:

**Location 1: Config building (~line 335)**
```python
elif model_type == "{MODEL_TYPE}":
    from wavetrader.{model_tag} import {ModelName}Config
    model_config = {ModelName}Config(pair=pair)
```

**Location 2: Checkpoint prefix (~line 732)**
```python
elif model_type == "{MODEL_TYPE}":
    prefix = "{MODEL_TAG}_"
```

**Location 3: Model loading (~line 753)**
```python
elif model_type == "{MODEL_TYPE}":
    from wavetrader.{model_tag} import {ModelName}
    model = {ModelName}(config)
```

#### 5C. Frontend — NO changes needed

Both live and backtest dropdowns are **fully dynamic**:
- **Live mode**: `live-init.js` → `loadModelSelector()` fetches `GET /api/live/models`
- **Backtest mode**: `backtest-init.js` → `currentBacktestModel` fetches same endpoint, `config-panel.js` sends `config.model` to backend

The dropdown populates automatically from `MODEL_REGISTRY`. No JS changes required.

#### 5D. API routes — NO changes needed

Both `dashboard/routes/live.py` and `dashboard/routes/backtest.py` are fully generic.

#### 5E. Verify imports work

```bash
python -c "from wavetrader.{model_tag} import {ModelName}, {ModelName}Config; print('OK')"
```

---

### Phase 6: Create Training Notebook

> **Only create AFTER Phases 2-5 are complete** — all code must import cleanly.

Create `{ModelName}_Training.ipynb` — modeled after `WaveFollower_Training.ipynb`. See [training template](./templates/training-notebook.md).

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

**Training loop MUST be inline** (NOT delegated to `train_model()` / `train_mtf_model()`).

**Cell structure** (13 cells — matches WaveFollower_Training.ipynb):
1. Header (markdown) — model description, OANDA account
2. Imports & GPU Check (performance flags + VRAM report)
3. Configuration & Data Paths
4. Check processed data availability
5. Load data from processed parquets
6. Verify data format compatibility
7. Build `PreCachedMTFDataset` + DataLoaders
8. Instantiate model + `torch.compile` + sanity check
9. Loss, optimizer (fused AdamW), scheduler, AMP setup
10. Training Loop — inline with AMP, per-epoch val, best checkpoint save
11. Training curves (Plotly)
12. Test set evaluation
13. Save checkpoint + SHA-256 + history + Google Drive copy

### Phase 7: Create Validation & Backtesting Notebook (SEPARATE file)

Create `{ModelName}_Validation.ipynb` — modeled after `WaveFollower_Validation.ipynb`. See [validation template](./templates/validation-notebook.md).

**Cell structure** (16 cells — matches WaveFollower_Validation.ipynb):
1. Header (markdown) — OANDA params table
2. Setup & Mount Drive
3. Clone repo & install deps
4. Load model from checkpoint (SHA-256 verify, `_orig_mod.` strip)
5. Load test data for all target pairs
6. Holdout evaluation — accuracy, loss
7. Detailed metrics — F1, confusion matrix, calibration
8. Per-pair backtest with OANDA-realistic config
9. Primary pair deep dive — period breakdowns
10. Full week-by-week table
11. Timeline visualizations (4 subplots)
12. Session & time-of-day analysis
13. Friction simulation
14. Equity curve + sample trades
15. Save all results to Drive

---

### Phase 8: OANDA Account Setup

> **⚠️ This causes downtime for ALL models until the new API key is active.**

1. Create sub-account on OANDA Hub
2. Revoke and regenerate API key
3. Update `.env` on VPS — **ALL models** get the new key
4. Verify: `curl` each account → 200 OK

See [OANDA reference](./references/oanda-accounts.md) for detailed steps.

### Phase 9: Model Registry + Docker Compose

1. Update `MODEL_REGISTRY` JSON in `.env` with new model entry
2. Add `wavetrader-{MODEL_TAG}` service in `docker-compose.yml`
3. Add env vars to dashboard service for new model's credentials

### Phase 10: Deploy

```bash
cd /opt/wavetrader
git pull origin main
docker compose build && docker compose up -d
docker compose ps
curl -s http://localhost:5000/api/live/models | jq .
curl -s http://localhost:5000/api/live/account?model={MODEL_TAG} | jq .
```

## Code Files Changed Per New Model

| File | Change Type | Required? |
|------|------------|-----------|
| `wavetrader/config.py` | Add `{ModelName}Config` dataclass | **Yes** |
| `wavetrader/{model_tag}.py` | NEW — model class | **Yes** |
| `wavetrader/train_{model_tag}.py` | NEW — custom loss (if extra outputs) | If needed |
| `wavetrader/{model_tag}_backtest.py` | NEW — custom backtest (if custom exits) | If needed |
| `wavetrader/__init__.py` | Export new classes | **Yes** |
| `dashboard/services/live_service.py` | Add `elif` at 2 locations | **Yes** |
| `dashboard/services/backtest_service.py` | Add `elif` at 3 locations | **Yes** |
| `dashboard/routes/*.py` | No changes (generic) | — |
| `dashboard/static/js/*.js` | No changes (dynamic from registry) | — |
| `.env` / `.env.template` | Add OANDA creds + update MODEL_REGISTRY | **Yes** |
| `docker-compose.yml` | Add service block + dashboard env vars | **Yes** |
| `{ModelName}_Training.ipynb` | NEW — training notebook | **Yes** |
| `{ModelName}_Validation.ipynb` | NEW — validation notebook | **Yes** |

## Troubleshooting

### 403 Forbidden from OANDA
**Cause**: API key revoked but not regenerated/updated.
**Fix**: Regenerate key → update ALL `*_OANDA_DEMO_API_KEY` vars → `docker compose restart` → wait 1-2 min.

### Model not in dropdown (live or backtest)
**Cause**: `MODEL_REGISTRY` env var not set or JSON malformed.
**Fix**: `echo $MODEL_REGISTRY | python -m json.tool` → fix → restart dashboard.

### Backtest returns wrong model results
**Cause**: Missing `elif` in `backtest_service.py` — falls through to MTF.
**Fix**: Add checks at all 3 locations (Phase 5B).

### Live mode shows no signals for new model
**Cause**: Missing `elif` in `live_service.py` — loads wrong model class.
**Fix**: Add checks at both locations (Phase 5A).

### Checkpoint load fails (`_orig_mod.` prefix)
**Fix**: `cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}`

## Checklist

**Code (MUST complete before notebooks)**:
- [ ] `{ModelName}Config` added to `wavetrader/config.py`
- [ ] `wavetrader/{model_tag}.py` created with forward() → signal_logits + confidence
- [ ] Custom loss in `wavetrader/train_{model_tag}.py` (if needed)
- [ ] Custom backtest in `wavetrader/{model_tag}_backtest.py` (if needed)
- [ ] All new classes exported in `wavetrader/__init__.py`
- [ ] `live_service.py` updated — 2 elif branches
- [ ] `backtest_service.py` updated — 3 elif branches
- [ ] `python -c "from wavetrader.{model_tag} import {ModelName}"` succeeds

**Notebooks (AFTER code is importable)**:
- [ ] `{ModelName}_Training.ipynb` with all GPU optimizations
- [ ] `{ModelName}_Validation.ipynb` (SEPARATE file)
- [ ] Model trained to >55% accuracy
- [ ] Checkpoint: `model_weights.pt` + `config.json` + `weights.sha256` + `history.json`
- [ ] Validation passes: holdout eval + backtest + friction sim

**Deployment**:
- [ ] OANDA sub-account created
- [ ] API key regenerated → ALL `.env` entries updated
- [ ] 403 resolved for all accounts
- [ ] `MODEL_REGISTRY` JSON updated
- [ ] `docker-compose.yml` has new service
- [ ] `docker compose up -d` — all containers healthy
- [ ] Dashboard live dropdown shows + switches to new model
- [ ] Dashboard backtest dropdown shows + runs new model
- [ ] Paper-trade 50+ bars
- [ ] Results saved to `backtest_results/{ckpt_name}/`
