"""
Shared enums and dataclasses for WaveTrader.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class Signal(Enum):
    BUY  = 0
    SELL = 1
    HOLD = 2


class StructureType(Enum):
    HH   = 0   # Higher High
    HL   = 1   # Higher Low
    LL   = 2   # Lower Low
    LH   = 3   # Lower High
    NONE = 4


@dataclass
class TradeSignal:
    """Model output: direction + risk management parameters."""
    signal: Signal
    confidence: float
    entry_price: float
    stop_loss: float          # pips
    take_profit: float        # pips
    trailing_stop_pct: float  # fraction of move to trail
    timestamp: datetime


@dataclass
class Trade:
    """Single trade lifecycle: open → active → closed."""
    entry_time: datetime
    entry_price: float
    direction: Signal
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    size: float               # Lot size (standard lots)

    # Mutable state — updated each bar while active
    current_sl: float = 0.0
    highest_price: float = 0.0   # Peak price for long trailing
    lowest_price:  float = 0.0   # Trough price for short trailing

    # Filled on close
    exit_time:   Optional[datetime] = None
    exit_price:  Optional[float]    = None
    pnl:         float = 0.0
    exit_reason: str   = ""

    def __post_init__(self) -> None:
        self.current_sl    = self.stop_loss
        self.highest_price = self.entry_price
        self.lowest_price  = self.entry_price


@dataclass
class BacktestResults:
    """Aggregate statistics from a completed backtest run."""
    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    total_pnl:      float = 0.0
    max_drawdown:   float = 0.0
    win_rate:       float = 0.0
    profit_factor:  float = 0.0
    sharpe_ratio:   float = 0.0
    final_balance:  float = 0.0
    trades:         List[Trade] = field(default_factory=list)
    equity_curve:   List[float] = field(default_factory=list)
