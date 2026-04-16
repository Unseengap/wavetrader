"""
WaveTrader Strategy Framework — rule-based strategies confirmed by AI.

All strategies produce the same StrategySetup interface, enabling uniform
backtesting, live execution, and dashboard integration.
"""
from .base import BaseStrategy, StrategySetup, StrategyMeta, IndicatorBundle  # noqa: F401
from .registry import StrategyRegistry, get_strategy_registry  # noqa: F401
from .ai_confirmer import AIConfirmer, ConfirmedSignal  # noqa: F401
