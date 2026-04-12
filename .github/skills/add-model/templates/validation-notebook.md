# Validation & Backtesting Notebook Template

Reference template for `{ModelName}_Validation.ipynb`. Produces complete performance analysis.

## Cell Structure

### Cell 1: Header (Markdown)

```markdown
# {ModelName} — Validation & Backtesting
### Load Checkpoint → Evaluate → Realistic Backtest ($100 start, 10% risk, OANDA conditions)

| OANDA Reality | Value |
|---------------|-------|
| Starting Balance | $100 |
| Risk per Trade | 10% ($10 at start) |
| Leverage | 30:1 (ESMA retail) |
| Spread | Pair-specific (OANDA avg) |
| Commission | $0 (OANDA Standard) |
| Min lot | 0.01 (1,000 units) |
| OANDA Account | `{account_id}` |
```

### Cell 2: Setup

```python
from google.colab import drive  # Remove for local
import os, pathlib, sys

# Colab paths
DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/phase_lm")
DRIVE_CKPT = DRIVE_ROOT / "checkpoints"
DRIVE_PROC = DRIVE_ROOT / "processed_data"

# Show available checkpoints
ckpts = sorted(DRIVE_CKPT.glob("{model_tag}_*"), key=lambda p: p.name, reverse=True)
print(f"Checkpoints found: {len(ckpts)}")
```

### Cell 3: Load Model

```python
import torch, json, hashlib
from wavetrader.{model_module} import {ModelClass}, {ModelConfig}

CKPT_DIR = ckpts[0]  # Latest

# SHA-256 verify
weights_path = CKPT_DIR / "model_weights.pt"
sha_path = CKPT_DIR / "weights.sha256"
if sha_path.exists():
    expected = sha_path.read_text().split()[0]
    h = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    assert expected == h.hexdigest(), "Checksum MISMATCH!"

# Load config
config_path = CKPT_DIR / "config.json"
with open(config_path) as f:
    cfg_dict = json.load(f)
clean = {k: v for k, v in cfg_dict.items()
         if not k.startswith("_") and k in {ModelConfig}.__dataclass_fields__}
config = {ModelConfig}(**clean)

# Load model
model = {ModelClass}(config).to(device)
checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint
cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(cleaned)
model.eval()
```

### Cell 4: Load Test Data

```python
PAIRS = ["GBPJPY", "EURJPY", "USDJPY", "GBPUSD"]
PAIR_NAMES = {"GBPJPY": "GBP/JPY", "EURJPY": "EUR/JPY",
              "USDJPY": "USD/JPY", "GBPUSD": "GBP/USD"}

# OANDA typical spreads per pair (pips)
OANDA_SPREADS = {"GBPJPY": 2.5, "EURJPY": 2.0, "USDJPY": 1.4, "GBPUSD": 1.6}
PIP_VALUES = {"GBPJPY": 6.5, "EURJPY": 6.5, "USDJPY": 6.5, "GBPUSD": 10.0}
```

### Cell 5: Holdout Evaluation

Must produce:
- Overall accuracy
- Classification report (per-class precision/recall/F1)
- Directional F1 (BUY+SELL avg)
- Confusion matrix plot
- Confidence distribution histogram
- Calibration curve

### Cell 6: Per-Pair Backtest

```python
from wavetrader.backtest import run_backtest
from wavetrader.config import BacktestConfig

INITIAL_BALANCE = 100.0
RISK_PER_TRADE = 0.10

for pair_tag in test_data:
    bt_config = BacktestConfig(
        initial_balance=INITIAL_BALANCE,
        risk_per_trade=RISK_PER_TRADE,
        min_confidence=0.55,
        spread_pips=OANDA_SPREADS[pair_tag],
        pip_value=PIP_VALUES[pair_tag],
        commission_per_lot=0.0,
        leverage=30.0,
        atr_halt_multiplier=3.0,
        drawdown_reduce_threshold=0.15,
    )
    results = run_backtest(model, test_data[pair_tag], model_config, bt_config, device)
```

### Cell 7: Deep Dive (Primary Pair)

Produce:
- Daily/weekly/monthly/yearly PnL breakdown
- Week-by-week table
- Direction breakdown (BUY vs SELL)
- Streak analysis
- Timeline visualizations (4 subplots)
- Duration distribution

### Cell 8: Session Analysis

Produce:
- Trades by hour (with Asia/London/NY bands)
- PnL & win rate by hour
- Day-of-week PnL
- Session breakdown table

### Cell 9: Friction Simulation

```python
# OANDA-specific friction parameters
SLIPPAGE_RANGE = (0.3, 2.0)
SPREAD_OFFHOURS = 1.5
SPREAD_NEWS_PROB = 0.05
SPREAD_NEWS_EXTRA = 4.0
LOT_CAP = 0.5
LATENCY_MISS_RATE = 0.02
```

Produce:
- Theoretical vs realistic comparison table
- Annualized ROI projection
- Compound projections (1/2/3/5 year)
- Equity curve overlay (linear + log scale)

### Cell 10: Save Results

Save to `backtest_results/{ckpt_name}/`:
- `trade_log.csv`
- `daily_breakdown.csv`
- `weekly_breakdown.csv`
- `monthly_breakdown.csv`
- `yearly_breakdown.csv`
- `equity_curve.csv`
- `session_breakdown.csv`
- `friction_simulation.csv`
- `per_pair_summary.csv`

## Required Metrics (Aggregate Table)

```
Pair | Trades | Win Rate | PF | Final $ | ROI | MaxDD | Sharpe
```

All results must be reproducible — seed RNG for friction simulation.
