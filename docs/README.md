# WaveTrader Documentation

> Wave-Based Neural Trading Signal Model for Forex — from training to live deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Data Pipeline](#data-pipeline)
5. [Training](#training)
6. [Backtesting](#backtesting)
7. [Live Trading](#live-trading)
8. [Copy Trading & Signal Broadcast](#copy-trading--signal-broadcast)
9. [Configuration Reference](#configuration-reference)
10. [CLI Reference](#cli-reference)
11. [Deployment](#deployment)
12. [API Reference](#api-reference)
13. [Accounts & Services Required](#accounts--services-required)

---

## Overview

WaveTrader is a multi-timeframe transformer-based trading system for Forex. It generates BUY/SELL/HOLD signals with confidence scores, stop-loss, take-profit, and trailing stop parameters — all learned end-to-end.

**Core pipeline:**

```
Historical Data → Training (Walk-Forward CV) → Backtesting → Live Trading (OANDA)
                                                                ↓
                                                   Copy Trading → Followers' Accounts
                                                   Signal Channel → Telegram Subscribers
```

**Supported pairs:** GBP/JPY, EUR/JPY, GBP/USD, USD/JPY

**Timeframes:** 15min (entry), 1H (confirmation), 4H (trend), Daily (bias)

---

## Architecture

### Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `FluxSignal` | Single-timeframe wave model | Prototyping, single-TF pairs |
| `WaveTraderMTF` | Multi-timeframe (4 TFs) | **Recommended for production** |
| `FluxSignalFabric` | Multi-pair + cross-pair attention + FiLM regime gating | Multi-pair portfolio |

### Signal Flow

```
Per Timeframe:
  OHLCV → PriceWaveEncoder  ──┐
  MarketStructure → StructureWaveEncoder ──┤
  RSI → RSIWaveEncoder ──┤  → WaveFusion → CausalWaveChainer → WavePredictor
  Volume → VolumeWaveEncoder ──┘
 
Multi-Timeframe:
  15min encoder ──┐
  1H encoder    ──┤ → MultiTimeframeFusion (cross-attention) → CWC → Predictor → SignalHead
  4H encoder    ──┤
  Daily encoder ──┘

SignalHead outputs:
  signal_logits  [B, 3]   →  BUY / SELL / HOLD
  confidence     [B, 1]   →  calibrated probability (scaled by TF alignment)
  risk_params    [B, 3]   →  stop_loss / take_profit / trailing_stop_pct
```

### Key Components

- **CausalWaveChainer (CWC):** Stateful module with ~25 hours of temporal memory via positional encoding + causal self-attention. Critical state that must be persisted for live trading.
- **ResonanceBuffer:** Episodic memory — stores the top-100 high-salience wave states (large PnL moves). Retrieved via L2 similarity at inference for "what happened last time the market looked like this?"
- **Synaptic Intelligence (SI):** Continual learning regulariser that prevents catastrophic forgetting during online fine-tuning.

---

## Quick Start

### Train a model

```bash
# 1. Place data files in data/
#    (Dukascopy, HistData, or MT4 CSV format — see Data Pipeline section)

# 2. Run data quality pipeline
python3 cli.py --mode preprocess --data data/

# 3. Train multi-timeframe model (recommended)
python3 cli.py --mode mtf --pair "GBP/JPY" --epochs 30

# 4. Model checkpoint saved to checkpoints/
```

### Backtest

```bash
python3 cli.py --mode backtest --checkpoint checkpoints/wavetrader_mtf_GBPJPY_*/model_weights.pt --balance 10000
```

### Go live (paper trading)

```bash
# Set up OANDA credentials
cp .env.template .env
# Edit .env with your OANDA_API_KEY and OANDA_ACCOUNT_ID

python3 cli.py --mode live --pair "GBP/JPY" --balance 10000
```

---

## Data Pipeline

### Supported Data Sources (free)

| Source | Format | URL |
|--------|--------|-----|
| **Dukascopy** | `GBPJPY_Candlestick_15_m_BID_*.csv` | [dukascopy.com/marketwatch/historic](https://www.dukascopy.com/swiss/english/marketwatch/historic/) |
| **HistData** | `DAT_ASCII_GBPJPY_M15_2023.csv` (semicolon-separated) | [histdata.com](https://www.histdata.com/download-free-forex-data/) |
| **MetaTrader 4/5** | Tab-separated export from History Centre | Your MT4/5 terminal |
| **Generic CSV** | Any file with date/open/high/low/close/volume columns | — |

### Load Priority

When calling `load_forex_data(pair, timeframe, data_dir)`:

1. Local Dukascopy CSV
2. Local HistData CSV
3. Local MT4/MT5 CSV
4. Generic auto-detect CSV
5. yfinance (limited to ~60 days for intraday)
6. Synthetic GBM+GARCH fallback (demo only — **not for training**)

### Data Quality Pipeline

```bash
python3 cli.py --mode preprocess --data data/
```

This runs on all 4 pairs × 4 timeframes:

1. Load raw CSVs from `data/`
2. `filter_flash_crashes()` — remove extreme-move / ghost-tick bars
3. `detect_gaps()` — annotate bars preceded by intraday gaps
4. `verify_session_alignment()` — check cross-pair timestamp alignment
5. Save cleaned DataFrames as `<PAIR>_<TF>_clean.parquet`

### Required File Layout

```
data/
  GBPJPY_1 Min_Bid_*.csv
  GBPJPY_Hourly_Bid_*.csv
  GBPJPY_4 Hours_Bid_*.csv
  GBPJPY_Daily_Bid_*.csv
  EURJPY_1 Min_Bid_*.csv
  ...
```

The loader auto-detects format from filenames and column headers.

---

## Training

### Single-Timeframe

```bash
python3 cli.py --mode train --pair "GBP/JPY" --timeframe 15min --epochs 50
# → Saves checkpoint: flux_signal_best.pt
```

### Multi-Timeframe (recommended)

```bash
python3 cli.py --mode mtf --pair "GBP/JPY" --epochs 30
# → Saves checkpoint: wavetrader_mtf_best.pt + timestamped dir in checkpoints/
```

### Training Features

- **Walk-forward cross-validation:** Purged time-series CV with expanding window — no future leakage.
- **Synaptic Intelligence:** Continual learning prevents catastrophic forgetting when retraining on new data.
- **Label generation:** BUY/SELL/HOLD labels from future price movement within a lookahead window using ATR-scaled pip thresholds.
- **Data split:** Chronological train/val/test (never shuffled).

---

## Backtesting

### Run a backtest

```bash
python3 cli.py --mode backtest --checkpoint model.pt --balance 10000
```

### BacktestEngine Features

| Feature | Description |
|---------|-------------|
| **Spread simulation** | Configurable spread (default 2.0 pips for GBP/JPY) |
| **Commission** | $7/lot round-trip |
| **Position sizing** | Fixed-fractional: `lot = (balance × risk%) / (SL_pips × pip_value)` |
| **Trailing stops** | Dynamic trailing as price moves in favour |
| **Volatility halt** | Skip trade if current range > 3× rolling 20-bar mean |
| **Drawdown step-down** | Halve risk when drawdown exceeds 5% |
| **Margin check** | Position capped at 50% of available margin |

### Walk-Forward Backtest

```python
import wavetrader as wt
results = wt.walk_forward_backtest(model, data, config, bt_config, device)
```

Expanding-window evaluation — retrains on growing data, tests on unseen forward period.

---

## Live Trading

### Prerequisites

1. **OANDA account** (practice or live) — see [Accounts section](#accounts--services-required)
2. **Trained model checkpoint** in `checkpoints/`
3. **Environment file** — copy `.env.template` to `.env` and fill in credentials

### Start Paper Trading

```bash
# Paper trading (default — no real orders)
python3 cli.py --mode live --pair "GBP/JPY" --balance 10000
```

### Start Live Trading

```bash
# Real money (requires confirmation prompt)
python3 cli.py --mode live --pair "GBP/JPY" --balance 10000 --live-trading
```

### How the Streaming Engine Works

Every ~30 seconds, the engine:

1. **Polls** OANDA for the latest complete 15m candle
2. **Refreshes** higher timeframes (1H every 4 bars, 4H every 16, Daily every 96)
3. **Builds features** using the same `prepare_features()` pipeline as training
4. **Runs inference** through `WaveTraderMTF` → `TradeSignal`
5. **Executes** — opens/closes positions via OANDA REST API
6. **Updates** trailing stops on open positions
7. **Checkpoints** state every 100 bars (configurable)
8. **Monitors** — sends Telegram alerts, tracks metrics

### Circuit Breakers (Live)

- **Volatility halt:** Skips signal if current bar range > 3× the 20-bar rolling mean
- **Drawdown step-down:** Halves risk when drawdown exceeds 5%
- **Confidence filter:** Skips signals below 60% confidence (configurable)
- **Market hours:** Pauses during weekend (Fri 22:00 — Sun 22:00 UTC)

### State Persistence

CWC requires ~25 hours of bars to build valid state. Without checkpointing, a cold restart means 25 hours of warm-up before valid signals.

Checkpoints save (every 100 bars):
- Model weights
- CWC hidden state
- Resonance Buffer contents
- Equity/position/trade tracking
- Recent bar ranges (circuit breaker state)

Storage: atomic write to `<checkpoint_dir>/latest.pt` + rolling history (last 5).

### Running with Docker

```bash
cp .env.template .env
# Edit .env with credentials

docker-compose up -d
# Logs: docker-compose logs -f wavetrader
# Stop: docker-compose down
```

---

## Copy Trading & Signal Broadcast

Two methods for letting other users trade with the bot.

### Method 1: Copy Trading (automatic execution)

The bot copies every trade to followers' OANDA accounts. Position sizing is per-user based on their balance and risk settings.

```bash
# Register a follower (interactive)
python3 cli.py --mode add-user

# Prompts for:
#   User ID, Name, OANDA API Key, Account ID,
#   Environment (practice/live), Risk per trade, Max lot size,
#   Telegram chat ID (optional)

# Credentials are verified against OANDA before saving.
# API keys are obfuscated at rest (XOR + base64).

# Manage followers
python3 cli.py --mode list-users       # Table of all followers + stats
python3 cli.py --mode user-status      # Live balances from OANDA
python3 cli.py --mode remove-user      # Remove a follower
```

**How it works:**
1. Bot generates a signal (single inference — no extra compute per follower)
2. `CopyTradeManager` iterates all active followers
3. For each follower: fetches their balance, sizes position to their risk settings, places order on their OANDA account
4. On close: closes all follower positions for the pair

**What each follower needs:** An OANDA practice or live account → generate an API key → give it to the bot operator.

**User data storage:** `data/users.json` — JSON file with obfuscated API keys.

### Method 2: Signal Broadcast (manual execution)

Bot posts signals to a public Telegram channel. Subscribers execute manually on their own broker.

```bash
# Setup:
# 1. Create a Telegram channel (e.g. @WaveTraderSignals)
# 2. Add your bot as a channel admin
# 3. In .env:
TELEGRAM_CHANNEL_ID=@WaveTraderSignals
```

**Signal format posted to channel:**

```
🟢 WaveTrader Signal

BUY GBP/JPY
Entry: 191.250
Stop Loss: 190.850 (40 pips)
Take Profit: 192.050 (80 pips)
Confidence: 72.3% ⭐⭐⭐

⏰ 2026-04-05 14:30 UTC
Risk management: never risk more than 1-2% per trade
```

**What each subscriber needs:** Just join the Telegram channel — no accounts required.

---

## Configuration Reference

### SignalConfig (Single-Timeframe)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `price_wave_dim` | 128 | Price encoder output dimension |
| `structure_wave_dim` | 64 | Structure encoder output dimension |
| `rsi_wave_dim` | 32 | RSI encoder output dimension |
| `volume_wave_dim` | 32 | Volume encoder output dimension |
| `fused_wave_dim` | 432 | Fused wave dimension after concat |
| `causal_dim` | 176 | CWC causal feature dimension |
| `predictor_hidden` | 512 | Transformer hidden dimension |
| `predictor_heads` | 8 | Attention heads |
| `predictor_layers` | 4 | Transformer layers |
| `predictor_ff_dim` | 2048 | Feed-forward dimension |
| `lookback` | 100 | Bars of history per sample |
| `dropout` | 0.2 | Dropout rate |
| `learning_rate` | 1e-4 | Adam LR |
| `batch_size` | 32 | Training batch size |
| `epochs` | 50 | Training epochs |
| `pair` | "GBP/JPY" | Target pair |
| `timeframe` | "15min" | Entry timeframe |

### MTFConfig (Multi-Timeframe)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeframes` | ["15min", "1h", "4h", "1d"] | Timeframe hierarchy |
| `lookbacks` | {15min: 100, 1h: 100, 4h: 100, 1d: 50} | Bars per TF |
| `tf_wave_dim` | 256 | Per-timeframe encoder output |
| `fused_wave_dim` | 512 | Post-fusion dimension |
| `batch_size` | 16 | Smaller (4× data per sample) |

### BacktestConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_balance` | 10,000 | Starting capital (USD) |
| `risk_per_trade` | 0.02 | 2% of balance per trade |
| `leverage` | 30.0 | Standard retail forex |
| `spread_pips` | 2.0 | GBP/JPY typical spread |
| `commission_per_lot` | 7.0 | USD round-trip per standard lot |
| `pip_value` | 6.5 | USD per pip per standard lot |
| `min_confidence` | 0.60 | Signal confidence threshold |
| `atr_halt_multiplier` | 3.0 | Volatility circuit breaker |
| `drawdown_reduce_threshold` | 0.05 | Halve risk at 5% drawdown |

### ResonanceConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity` | 100 | Max stored wave states |
| `wave_dim` | 608 | Must match `output_wave_dim` |
| `salience_threshold` | 2.0 | Std devs above mean for storage |
| `top_k` | 5 | States retrieved per inference |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OANDA_API_KEY` | **Yes** (live mode) | — | OANDA v20 API token |
| `OANDA_ACCOUNT_ID` | **Yes** (live mode) | — | OANDA account ID |
| `OANDA_ENVIRONMENT` | No | `practice` | `practice` or `live` |
| `PAIR` | No | `GBP/JPY` | Trading pair |
| `PAPER_TRADING` | No | `true` | `true` or `false` |
| `INITIAL_BALANCE` | No | `10000` | Starting capital for paper mode |
| `CHECKPOINT_DIR` | No | `/data/checkpoints` | Checkpoint storage directory |
| `CHECKPOINT_INTERVAL` | No | `100` | Bars between checkpoints |
| `CHECKPOINT_PATH` | No | — | Path to trained model weights |
| `TELEGRAM_BOT_TOKEN` | No | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | No | — | Admin notification chat ID |
| `TELEGRAM_CHANNEL_ID` | No | — | Signal broadcast channel ID |
| `DATA_DIR` | No | `data` | User registry location |

---

## CLI Reference

```
python3 cli.py --mode <MODE> [OPTIONS]
```

### Modes

| Mode | Description |
|------|-------------|
| `demo` | Single-TF demo with synthetic data fallback |
| `train` | Train a new single-TF model |
| `mtf` | Multi-timeframe training (recommended) |
| `backtest` | Backtest a saved checkpoint on test data |
| `preprocess` | Data quality pipeline (clean, filter, align) |
| `live` | Start live trading engine (paper or real) |
| `add-user` | Register a new copy-trade follower |
| `remove-user` | Remove a registered follower |
| `list-users` | List all followers with stats |
| `user-status` | Show live OANDA balances for all followers |

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pair` | GBP/JPY | Forex pair |
| `--timeframe` | 15min | Entry timeframe (single-TF modes) |
| `--epochs` | 30 | Training epochs |
| `--balance` | 10000 | Initial balance (USD) |
| `--data` | data | Data directory |
| `--checkpoint` | None | Path to .pt checkpoint |
| `--paper` | True | Paper trading (default) |
| `--live-trading` | False | Enable real order execution |

### Examples

```bash
# Train multi-timeframe model
python3 cli.py --mode mtf --pair "GBP/JPY" --epochs 30

# Backtest with custom balance
python3 cli.py --mode backtest --balance 25000 --checkpoint checkpoints/*/model_weights.pt

# Live paper trading
python3 cli.py --mode live --pair "GBP/JPY"

# Live real trading (asks for confirmation)
python3 cli.py --mode live --pair "GBP/JPY" --live-trading

# Manage followers
python3 cli.py --mode add-user
python3 cli.py --mode list-users
python3 cli.py --mode user-status --pair "GBP/JPY"
python3 cli.py --mode remove-user
```

---

## Deployment

### Recommended Infrastructure

| Component | Recommendation | Cost |
|-----------|---------------|------|
| **VPS** | Hetzner CPX31 (4 vCPU, 8GB RAM, Frankfurt) | ~€12.50/mo |
| **OS** | Ubuntu 22.04 LTS | — |
| **Container** | Docker + docker-compose | — |
| **Cache** | Redis 7 (for Resonance Buffer backup) | Included |

### Deploy to VPS

```bash
# On the VPS:
git clone <your-repo> && cd phase_lm

# Configure
cp .env.template .env
nano .env   # Fill in OANDA + Telegram credentials

# Copy model checkpoint
scp -r checkpoints/ user@vps:phase_lm/checkpoints/

# Start
docker-compose up -d

# Monitor
docker-compose logs -f wavetrader

# Stop
docker-compose down
```

### Why Not Heroku/Railway/Render

- **Dyno sleeping:** Missed bars during "wake up"
- **Ephemeral filesystem:** CWC state (25 hours of memory) lost on restart
- **Cost:** 2-4× more expensive for 24/5 continuous operation
- **No GPU:** If you scale to tick-level inference later

---

## API Reference

### Models

```python
import wavetrader as wt

# Single-timeframe
model = wt.FluxSignal(wt.SignalConfig(pair="GBP/JPY"))
signal = model.predict(batch)  # → TradeSignal

# Multi-timeframe (recommended)
model = wt.WaveTraderMTF(wt.MTFConfig(pair="GBP/JPY"))
signal = model.predict(mtf_batch)  # → TradeSignal

# Multi-pair fabric
model = wt.FluxSignalFabric(wt.SignalConfig(), peer_pairs=["USD/JPY", "EUR/JPY", "GBP/USD"])
signal = model.predict(multi_pair_batch)  # → TradeSignal
```

### Data Loading

```python
# Auto-detect format
df = wt.load_forex_data("GBP/JPY", "15min", data_dir="data/")

# Specific formats
df = wt.load_dukascopy_csv("data/GBPJPY_Candlestick_15_m_BID_*.csv")
df = wt.load_histdata_csv("data/DAT_ASCII_GBPJPY_M15_2023.csv")

# Multi-timeframe
mtf_data = wt.load_mtf_data("GBP/JPY", data_dir="data/")
# Returns: {"15min": df, "1h": df, "4h": df, "1d": df}

# Data cleaning
clean_df = wt.preprocess_pipeline(df, pair="GBP/JPY", timeframe="15min")
```

### Training

```python
# Datasets
train_ds = wt.MTFForexDataset(train_data, config, pair="GBP/JPY")
loader = DataLoader(train_ds, batch_size=16, collate_fn=wt.mtf_collate_fn)

# Training loop
history = wt.train_mtf_model(model, train_loader, val_loader, config, device)

# Walk-forward splits
splits = wt.walk_forward_splits(df, n_splits=5, purge_bars=100)
```

### Backtesting

```python
# Simple backtest
results = wt.run_backtest(model, test_df, config, bt_config, device)
print(f"PnL: ${results.total_pnl:.2f}, Win rate: {results.win_rate:.1%}")
wt.print_equity_chart(results.equity_curve)

# Walk-forward backtest
results = wt.walk_forward_backtest(model, data, config, bt_config, device)
```

### Live Trading (programmatic)

```python
from wavetrader.oanda import OANDAClient
from wavetrader.streaming import StreamingEngine
from wavetrader.copytrade import CopyTradeManager, UserRegistry

# OANDA client
client = OANDAClient()  # Reads from env vars
account = client.get_account_summary()
candles = client.get_latest_candles("GBP/JPY", "M15", count=100)
order = client.place_market_order("GBP/JPY", units=10000, sl=189.50, tp=191.00)

# Copy trading
registry = UserRegistry("data/")
mgr = CopyTradeManager(registry, pair="GBP/JPY")
results = mgr.copy_open(signal, candle_close=190.25)

# Streaming engine
engine = StreamingEngine(model, client, pair="GBP/JPY", paper_trading=True)
engine.warmup()
engine.run()
```

### Types

```python
from wavetrader.types import Signal, TradeSignal, Trade, BacktestResults

Signal.BUY    # 0
Signal.SELL   # 1
Signal.HOLD   # 2

# TradeSignal fields: signal, confidence, entry_price, stop_loss (pips),
#                     take_profit (pips), trailing_stop_pct, timestamp
```

---

## Accounts & Services Required

| Account | Purpose | Cost | Required? |
|---------|---------|------|-----------|
| **OANDA Practice** | Paper trading + API access | Free | Yes (for live mode) |
| **OANDA Live** | Real money trading | Free (deposit capital) | Only for real trading |
| **Telegram Bot** | Trade alerts & signal broadcast | Free | Optional |
| **VPS (Hetzner)** | 24/5 server for the bot | ~€12.50/mo | For production deployment |
| **Dukascopy/HistData** | Historical data download | Free | For training |

### OANDA Setup

1. Register at [oanda.com/demo-account](https://www.oanda.com/demo-account/)
2. Go to **Manage API Access** → generate a Personal Access Token
3. Note your Account ID (format: `101-001-12345678-001`)
4. Set `OANDA_API_KEY` and `OANDA_ACCOUNT_ID` in `.env`
5. Start with `OANDA_ENVIRONMENT=practice` — **always paper-trade first**

### Telegram Setup

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → get token
2. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your `chat_id`
3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
4. For signal channel: create a channel, add bot as admin, set `TELEGRAM_CHANNEL_ID`

---

## Project Structure

```
phase_lm/
├── cli.py                    # Main entry point — all modes
├── Dockerfile                # Production container
├── docker-compose.yml        # Full stack (bot + Redis)
├── requirements.txt          # Python dependencies
├── .env.template             # Environment config template
│
├── wavetrader/               # Core package
│   ├── __init__.py           # Public API exports
│   ├── __main__.py           # python -m wavetrader.streaming
│   ├── model.py              # FluxSignal, WaveTraderMTF, FluxSignalFabric
│   ├── encoders.py           # Wave encoders, CausalWaveChainer, RegimeGatedLayer
│   ├── config.py             # SignalConfig, MTFConfig, BacktestConfig, etc.
│   ├── types.py              # Signal, TradeSignal, Trade, BacktestResults
│   ├── data.py               # Data loaders (Dukascopy, HistData, MT4, yfinance)
│   ├── dataset.py            # ForexDataset, MTFForexDataset, ResonanceBuffer
│   ├── indicators.py         # RSI, ATR, swing points, market structure, ADX, Hurst
│   ├── training.py           # Training loops, SignalLoss, SynapticIntelligence
│   ├── backtest.py           # BacktestEngine, run_backtest, walk_forward_backtest
│   ├── utils.py              # Splits, ASCII charts, calendar walk-forward
│   ├── oanda.py              # OANDA v20 REST client
│   ├── streaming.py          # StreamingEngine — live trading loop
│   ├── state.py              # StateManager — checkpoint save/load
│   ├── monitor.py            # Telegram alerts + signal broadcast + metrics
│   └── copytrade.py          # UserRegistry, CopyTradeManager
│
├── data/                     # Raw & cleaned data files
├── checkpoints/              # Trained model checkpoints
├── backtest_results/         # CSV outputs from backtests
├── processed_data/           # Walk-forward split Parquets
├── scripts/                  # Data download & preprocessing scripts
└── docs/                     # Deployment guide, warmup protocol
```
