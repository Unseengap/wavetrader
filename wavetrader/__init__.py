"""
wavetrader — Wave-Based Neural Trading Signal Model for Forex.

Public API
──────────
Models
    FluxSignal          Single-timeframe model
    WaveTraderMTF       Multi-timeframe model
    FluxSignalFabric    Multi-pair + regime-gated model (Wave Fabric)
    CrossPairAttention  Cross-pair attention layer

Configs
    SignalConfig    Single-TF hyper-parameters
    MTFConfig       Multi-TF hyper-parameters
    BacktestConfig  Backtesting parameters
    ResonanceConfig Episodic memory buffer configuration
    SIConfig        Synaptic Intelligence continual learning config

Data
    load_forex_data          Smart loader (local CSV/Parquet > yfinance > synthetic)
    load_dukascopy_csv       Parse Dukascopy OHLCV export
    load_histdata_csv        Parse HistData.com ASCII export
    load_mt4_csv             Parse MetaTrader 4/5 export
    load_mtf_data            Load / resample all timeframes
    preprocess_pipeline      Full data-quality pipeline (gaps + flash-crash filter)
    filter_flash_crashes     Remove extreme-move / ghost-tick bars
    detect_gaps              Annotate bars preceded by intraday gaps
    verify_session_alignment Cross-pair timestamp alignment check
    generate_synthetic_forex Synthetic GBM+GARCH data (demo only)

Datasets
    ForexDataset             Single-TF PyTorch Dataset
    MTFForexDataset          Multi-TF PyTorch Dataset
    mtf_collate_fn           DataLoader collate function for MTF
    ResonanceBuffer          Episodic memory: rolling window of salient wave states

Training
    train_model              FluxSignal training loop
    train_mtf_model          WaveTraderMTF training loop
    walk_forward_splits      Purged time-series CV splits
    SynapticIntelligence     Online continual learning (SI)

Backtest
    BacktestEngine           Bar-by-bar simulation engine (with circuit breakers)
    run_backtest             Convenience single-split runner
    walk_forward_backtest    Expanding-window walk-forward evaluation

Indicators
    calculate_adx            Average Directional Index (trend strength)
    calculate_hurst          Hurst exponent (mean-reverting vs trending)

Utils
    print_equity_chart             ASCII equity curve
    chronological_split            Chronological train/val/test split
    chronological_split_mtf        Same for multi-TF dicts
    walk_forward_splits_calendar   Calendar-aware purged walk-forward CV
"""

from .backtest import BacktestEngine, run_backtest, walk_forward_backtest
from .config import BacktestConfig, MeanRevConfig, MTFConfig, ResonanceConfig, SIConfig, SignalConfig
from .data import (
    detect_gaps,
    filter_flash_crashes,
    generate_synthetic_forex,
    generate_synthetic_mtf_data,
    load_dukascopy_csv,
    load_forex_data,
    load_generic_csv,
    load_histdata_csv,
    load_mt4_csv,
    load_mtf_data,
    preprocess_pipeline,
    verify_session_alignment,
)
from .dataset import ForexDataset, MTFForexDataset, ResonanceBuffer, mtf_collate_fn
from .encoders import RegimeGatedLayer
from .indicators import calculate_adx, calculate_hurst
from .model import CrossPairAttention, FluxSignal, FluxSignalFabric, WaveTraderMTF
from .training import (
    FocalLoss,
    SignalLoss,
    SynapticIntelligence,
    train_model,
    train_mtf_model,
    walk_forward_splits,
)
from .types import BacktestResults, Signal, StructureType, Trade, TradeSignal
from .mean_reversion import MeanReversion
from .train_mean_reversion import MeanRevLoss
from .utils import (
    chronological_split,
    chronological_split_mtf,
    print_equity_chart,
    walk_forward_splits_calendar,
)

# Live trading (lazy — only imported when used)
from .copytrade import CopyTradeManager, UserAccount, UserRegistry
from .monitor import Monitor, MonitorConfig
from .oanda import OANDAClient, OANDAConfig
from .state import LiveState, StateManager
from .streaming import StreamingEngine
