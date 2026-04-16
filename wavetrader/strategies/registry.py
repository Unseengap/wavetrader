"""
Strategy registry — discovers, loads, and manages trading strategies.

Replaces the model registry for the strategy-led architecture.
Strategies can be registered via:
  1. STRATEGY_REGISTRY env var (JSON array) — for Docker/production
  2. Auto-discovery from this package — for development
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from .base import BaseStrategy, StrategyMeta

logger = logging.getLogger(__name__)


@dataclass
class StrategyEntry:
    """A single strategy definition in the registry."""
    id: str
    name: str
    author: str = "Dectrick McGee"
    category: str = "swing"
    description: str = ""
    strategy_class: str = ""         # "wavetrader.strategies.amd_session.AMDSessionStrategy"
    pairs: List[str] = field(default_factory=lambda: ["GBP/JPY", "EUR/JPY", "GBP/USD"])
    timeframes: List[str] = field(default_factory=lambda: ["15min", "1h", "4h", "1d"])
    entry_timeframe: str = "15min"
    # OANDA per-strategy
    demo_api_key_env: str = "OANDA_DEMO_API_KEY"
    demo_account_id_env: str = "OANDA_DEMO_ACCOUNT_ID"
    live_api_key_env: str = "OANDA_LIVE_API_KEY"
    live_account_id_env: str = "OANDA_LIVE_ACCOUNT_ID"
    # Defaults
    default_params: Dict = field(default_factory=dict)
    enabled: bool = True

    @property
    def demo_api_key(self) -> str:
        return os.environ.get(self.demo_api_key_env, "")

    @property
    def demo_account_id(self) -> str:
        return os.environ.get(self.demo_account_id_env, "")

    @property
    def live_api_key(self) -> str:
        return os.environ.get(self.live_api_key_env, "")

    @property
    def live_account_id(self) -> str:
        return os.environ.get(self.live_account_id_env, "")

    def to_dict(self) -> dict:
        """Public-safe dict (no API keys)."""
        return {
            "id": self.id,
            "name": self.name,
            "author": self.author,
            "category": self.category,
            "description": self.description,
            "pairs": self.pairs,
            "timeframes": self.timeframes,
            "entry_timeframe": self.entry_timeframe,
            "enabled": self.enabled,
            "demo_connected": bool(self.demo_api_key and self.demo_account_id),
            "live_connected": bool(self.live_api_key and self.live_account_id),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Built-in strategy map — auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_STRATEGIES: Dict[str, str] = {
    "news_catalyst_ob": "wavetrader.strategies.news_catalyst_ob.NewsCatalystOBStrategy",
    "opening_break_retest": "wavetrader.strategies.opening_break_retest.OpeningBreakRetestStrategy",
    "price_action_reversal": "wavetrader.strategies.price_action_reversal.PriceActionReversalStrategy",
}


class StrategyRegistry:
    """Loads and provides access to all configured strategy definitions."""

    def __init__(self) -> None:
        self._entries: Dict[str, StrategyEntry] = {}
        self._classes: Dict[str, Type[BaseStrategy]] = {}
        self.default_id: str = "news_catalyst_ob"
        self._load()

    def _load(self) -> None:
        """Load strategies from env var or auto-discover builtins."""
        raw = os.environ.get("STRATEGY_REGISTRY", "").strip()
        if raw:
            try:
                items = json.loads(raw)
                for item in items:
                    entry = StrategyEntry(**{
                        k: v for k, v in item.items()
                        if k in StrategyEntry.__dataclass_fields__
                    })
                    self._entries[entry.id] = entry
                logger.info("Loaded %d strategies from STRATEGY_REGISTRY env var", len(self._entries))
            except (json.JSONDecodeError, TypeError) as e:
                logger.error("Failed to parse STRATEGY_REGISTRY: %s", e)

        # Auto-discover builtins not already defined by env
        for strat_id, class_path in _BUILTIN_STRATEGIES.items():
            if strat_id not in self._entries:
                try:
                    cls = self._import_class(class_path)
                    meta: StrategyMeta = cls.meta
                    self._entries[strat_id] = StrategyEntry(
                        id=meta.id,
                        name=meta.name,
                        author=meta.author,
                        category=meta.category,
                        description=meta.description,
                        strategy_class=class_path,
                        pairs=meta.pairs,
                        timeframes=meta.timeframes,
                        entry_timeframe=meta.entry_timeframe,
                    )
                    self._classes[strat_id] = cls
                except Exception as e:
                    logger.debug("Could not auto-discover strategy '%s': %s", strat_id, e)

        if not self._entries:
            logger.warning("No strategies loaded — registry is empty")

        if self.default_id not in self._entries and self._entries:
            self.default_id = next(iter(self._entries))

    def _import_class(self, class_path: str) -> Type[BaseStrategy]:
        """Dynamically import a strategy class from dotted path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get(self, strategy_id: str) -> Optional[StrategyEntry]:
        """Look up a strategy entry by ID."""
        return self._entries.get(strategy_id)

    def list_strategies(self) -> List[StrategyEntry]:
        """Return all registered strategies."""
        return list(self._entries.values())

    def list_enabled(self) -> List[StrategyEntry]:
        """Return only enabled strategies."""
        return [e for e in self._entries.values() if e.enabled]

    def instantiate(
        self, strategy_id: str, params: Optional[Dict] = None
    ) -> BaseStrategy:
        """Create a strategy instance by ID."""
        entry = self._entries.get(strategy_id)
        if entry is None:
            raise KeyError(f"Unknown strategy: {strategy_id}")

        if strategy_id in self._classes:
            cls = self._classes[strategy_id]
        else:
            class_path = entry.strategy_class or _BUILTIN_STRATEGIES.get(strategy_id)
            if not class_path:
                raise ValueError(f"No class path for strategy: {strategy_id}")
            cls = self._import_class(class_path)
            self._classes[strategy_id] = cls

        merged_params = {**entry.default_params}
        if params:
            merged_params.update(params)
        return cls(params=merged_params)

    def to_list(self) -> List[dict]:
        """Return public-safe dicts for API responses."""
        return [e.to_dict() for e in self._entries.values()]


# ─────────────────────────────────────────────────────────────────────────────
# Module singleton
# ─────────────────────────────────────────────────────────────────────────────

_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """Get or create the global strategy registry singleton."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry
