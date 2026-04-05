# WaveTrader — Project Progress

**Last updated:** April 5, 2026

---

## Current Status: DEPLOYED & LIVE

| Component | Status | Location |
|-----------|--------|----------|
| Dashboard | Running (healthy) | http://104.207.143.54:5000 |
| Live Streaming Engine | Standby (market closed Saturday) | Vultr VPS Chicago |
| Redis State Store | Running | Vultr VPS Chicago |
| Model Checkpoint | Loaded (77 MB) | `wavetrader_mtf_GBPJPY_20260404_235854` |
| OANDA Account | Connected (practice, $122.25) | Practice environment |
| GitHub Repo | Up to date | github.com/Unseengap/wavetrader |

---

## Infrastructure

| Resource | Details |
|----------|---------|
| VPS | Vultr Cloud Compute, Chicago |
| Specs | 2 vCPU, 4 GB RAM, 80 GB SSD |
| OS | Ubuntu 24.04 LTS |
| IP | 104.207.143.54 |
| Docker | v29.3.1, Compose v5.1.1 |
| Containers | 3 (dashboard, wavetrader-live, redis) |
| Credits | $250 Vultr (~10 months at $24/mo) |

---

## Model Performance (Test Holdout: Jul 2024 — Apr 2026)

| Metric | Value |
|--------|-------|
| Architecture | WaveTraderMTF (multi-timeframe) |
| Pair | GBP/JPY |
| Timeframes | 15m, 1h, 4h, 1d |
| Parameters | ~20M |
| Training Data | Jan 2015 — Dec 2022 |
| Validation Data | Jan 2023 — Jun 2024 |
| Test Data | Jul 2024 — Apr 2026 (never seen) |

### Backtest Results ($100 starting capital)

| Metric | Value |
|--------|-------|
| Final Balance | $688,180.53 |
| ROI | +688,080% |
| Total Trades | 1,866 |
| Win Rate | 44.1% |
| Profit Factor | 2.34 |
| Sharpe Ratio | 3.87 |
| Max Drawdown | 27.3% |
| Winning / Losing | 823 / 1,043 |

---

## Features Completed

### Training Pipeline (Google Colab)
- [x] JForex CSV preprocessing (EET → UTC, mid-price, spread filters)
- [x] 1m → 15m/1h/4h/1d resampling with walk-forward splits
- [x] FluxSignal single-TF model
- [x] WaveTraderMTF multi-TF model (15m/1h/4h/1d fusion)
- [x] Training with early stopping, loss curves, progress tracking
- [x] SHA-256 checkpoint verification
- [x] Google Drive checkpoint persistence

### Dashboard
- [x] Flask app with 4 blueprint modules (pages, backtest, data, live)
- [x] TradingView Lightweight Charts v4 candlestick charting
- [x] Plotly.js analytics (equity curve, PnL breakdowns, sessions, duration)
- [x] Config panel (capital, risk, spread, circuit breakers, friction)
- [x] Backtest runner from dashboard UI
- [x] Cached backtest result loading from CSVs
- [x] Multi-pair support (GBP/JPY, EUR/JPY, GBP/USD, USD/JPY)
- [x] Multi-timeframe chart switching (15m, 1h, 4h, 1d)

### Live Trading
- [x] OANDA v20 REST API integration (candles, prices, account, trades)
- [x] SSE (Server-Sent Events) real-time streaming to dashboard
- [x] Live toggle switch in navbar (backtest ↔ live mode)
- [x] Live panel: account info, bid/ask/spread, model signal, open positions
- [x] Auto-loads latest model checkpoint for inference
- [x] Background polling (5s interval) with thread-safe SSE broadcast
- [x] Streaming engine with 100-bar warmup protocol

### Deployment
- [x] Docker multi-container setup (app + redis + dashboard)
- [x] Production Dockerfile (PyTorch CPU, healthcheck)
- [x] docker-compose with volume mounts and env injection
- [x] .dockerignore for optimized builds
- [x] VPS setup script (scripts/setup-vps.sh)
- [x] One-command deploy updates (scripts/deploy-update.sh)
- [x] UFW firewall (SSH + port 5000)
- [x] 4 GB swap for OOM protection
- [x] Deployed to Vultr VPS Chicago

---

## Git History

```
7001f51 fix: remove lzma from requirements (builtin), drop deprecated version key
e832d8d Production Docker config: add dashboard to image, .dockerignore, VPS setup script
84ab5cb feat: Implement live trading features with OANDA integration
5d09df1 feat: Add Chart Manager and Config Panel for WaveTrader Dashboard
0784d6f Load environment variables from .env file and optimize model loading in live trading
5e2e30d Implement state persistence and live trading engine for WaveTrader
3c6bd9e Add utility cell to pull latest changes from GitHub in WaveTrader Colab notebook
50bfd23 Refactor backtest functionality to support multi-timeframe datasets
b19e993 Refactor trading pairs and preprocessing scripts
cf1ad9f Refactor preprocessing logging and training epoch display
6561fb8 Update Colab notebook to clone from wavetrader repo
1bc963a Initial commit
```

---

## Architecture

```
Mac (development)
  └── VS Code + Copilot
  └── .env (credentials — gitignored)
  └── checkpoints/ (model weights — gitignored)
  └── data/ (JForex CSVs — gitignored)

Google Colab (training)
  └── GPU runtime (T4/L4)
  └── Google Drive persistence
  └── WaveTrader_Colab.ipynb

Vultr VPS — 104.207.143.54 (production)
  ├── wavetrader-dashboard  :5000  (Flask + TradingView charts + live SSE)
  ├── wavetrader-live              (StreamingEngine + OANDA polling + model inference)
  └── wavetrader-redis      :6379  (state persistence, resonance buffer)
```

---

## What's Next

- [ ] Market opens Sunday 5 PM CDT — verify live signals flow end-to-end
- [ ] Monitor first week of live paper trading
- [ ] Set up Telegram alerts for trade signals
- [ ] Add SSL/HTTPS (Let's Encrypt + nginx reverse proxy)
- [ ] Custom domain for dashboard
- [ ] Expand to additional pairs (EUR/JPY, GBP/USD)
- [ ] Fund OANDA live account after paper-trade validation
- [ ] Continual learning (SI) — retrain on new data monthly
