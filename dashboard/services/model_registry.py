"""
Model registry — defines available AI models and their associated OANDA accounts.

Each model runs in its own wavetrader container and trades independently.
The dashboard uses this registry to know which OANDA accounts to query
when the user switches the model selector dropdown.

Configuration is loaded from environment variables:
  MODEL_REGISTRY — JSON array of model definitions, e.g.:
    [
      {"id": "mtf", "name": "WaveTrader MTF", "pair": "GBP/JPY",
       "demo_api_key_env": "OANDA_DEMO_API_KEY",
       "demo_account_id_env": "OANDA_DEMO_ACCOUNT_ID",
       "live_api_key_env": "OANDA_LIVE_API_KEY",
       "live_account_id_env": "OANDA_LIVE_ACCOUNT_ID",
       "checkpoint_dir": "checkpoints/wavetrader_mtf_GBPJPY_20260404_235854"}
    ]

Falls back to a single model using the existing env vars if MODEL_REGISTRY is
not set, keeping full backward compatibility.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A single model definition in the registry."""
    id: str
    name: str
    pair: str = "GBP/JPY"
    description: str = ""
    model_type: str = "mtf"  # "mtf" = WaveTraderMTF, "wavefollower" = WaveFollower
    # Env var names that hold the OANDA credentials for this model
    demo_api_key_env: str = "OANDA_DEMO_API_KEY"
    demo_account_id_env: str = "OANDA_DEMO_ACCOUNT_ID"
    live_api_key_env: str = "OANDA_LIVE_API_KEY"
    live_account_id_env: str = "OANDA_LIVE_ACCOUNT_ID"
    checkpoint_dir: str = ""

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
            "pair": self.pair,
            "description": self.description,
            "model_type": self.model_type,
            "demo_connected": bool(self.demo_api_key and self.demo_account_id),
            "live_connected": bool(self.live_api_key and self.live_account_id),
        }


class ModelRegistry:
    """Loads and provides access to all configured model definitions."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelEntry] = {}
        self._default_id: str = ""
        self._load()

    def _load(self) -> None:
        raw = os.environ.get("MODEL_REGISTRY", "").strip()
        if raw:
            try:
                entries = json.loads(raw)
                for entry in entries:
                    m = ModelEntry(**entry)
                    self._models[m.id] = m
                if entries:
                    self._default_id = entries[0]["id"]
                logger.info("Loaded %d models from MODEL_REGISTRY", len(self._models))
                return
            except Exception as e:
                logger.error("Failed to parse MODEL_REGISTRY: %s — falling back to default", e)

        # Fallback: single model from existing env vars (backward compat)
        self._models["mtf"] = ModelEntry(
            id="mtf",
            name="WaveTrader MTF",
            pair="GBP/JPY",
            description="Multi-timeframe wave-based model",
            demo_api_key_env="OANDA_DEMO_API_KEY",
            demo_account_id_env="OANDA_DEMO_ACCOUNT_ID",
            live_api_key_env="OANDA_LIVE_API_KEY",
            live_account_id_env="OANDA_LIVE_ACCOUNT_ID",
        )
        self._default_id = "mtf"
        logger.info("Using default single-model registry (mtf)")

    @property
    def default_id(self) -> str:
        return self._default_id

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def list_models(self) -> List[ModelEntry]:
        return list(self._models.values())

    def list_ids(self) -> List[str]:
        return list(self._models.keys())

    def to_list(self) -> List[dict]:
        """Return public-safe list for the API."""
        return [m.to_dict() for m in self._models.values()]


# Module-level singleton
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
