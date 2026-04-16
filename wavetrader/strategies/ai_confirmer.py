"""
AI Confirmation Layer — wraps WaveTrader MTF as a directional confirmation engine.

The strategy proposes entries (direction + SL/TP).  The AI confirmer checks:
  1. Model's signal_logits agree with the strategy's direction
  2. Model confidence exceeds threshold
  3. Multi-TF alignment exceeds threshold

If all pass, returns ConfirmedSignal.  Otherwise returns None (rejected).

Speed doesn't matter here — this runs on historical lookback data after the
strategy has already determined timing.  Accuracy is the only goal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from .base import StrategySetup
from ..types import Signal, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ConfirmedSignal:
    """A strategy signal that has been confirmed by the AI model."""
    setup: StrategySetup                  # Original strategy signal
    ai_direction_agrees: bool             # signal_logits matches strategy direction
    ai_confidence: float                  # Model confidence (0-1)
    ai_alignment: float                   # Multi-TF alignment score (0-1)
    combined_confidence: float            # Weighted blend
    trade_signal: TradeSignal             # Final signal for execution
    ai_raw: Dict[str, Any] = None        # Raw model output (for logging)


class AIConfirmer:
    """Wraps WaveTrader MTF model as a confirmation engine for strategy setups.

    Loads the model once, then confirms/rejects strategy setups on demand.
    SL/TP come from the strategy (structure-based), not from model risk_params.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[torch.device] = None,
        min_ai_confidence: float = 0.55,
        min_alignment: float = 0.40,
        strategy_weight: float = 0.6,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_ai_confidence = min_ai_confidence
        self.min_alignment = min_alignment
        self.strategy_weight = strategy_weight  # Weight for combined confidence
        self.model = None
        self.config = None
        self._loaded = False

    def _load_model(self) -> None:
        """Lazy-load the WaveTrader MTF model from checkpoint."""
        if self._loaded:
            return

        import json
        from ..config import MTFConfig
        from ..model import WaveTraderMTF

        config_path = self.checkpoint_dir / "config.json"
        weights_path = self.checkpoint_dir / "model_weights.pt"

        if not weights_path.exists():
            raise FileNotFoundError(f"No model weights at {weights_path}")

        # Load config
        if config_path.exists():
            with open(config_path) as f:
                cfg_dict = json.load(f)
            clean = {k: v for k, v in cfg_dict.items()
                     if not k.startswith("_") and k in MTFConfig.__dataclass_fields__}
            self.config = MTFConfig(**clean)
        else:
            self.config = MTFConfig()

        # Load model
        self.model = WaveTraderMTF(self.config).to(self.device)
        state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        # Handle torch.compile prefix
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        try:
            self.model.load_state_dict(cleaned)
        except RuntimeError as e:
            # Checkpoint incompatible (architecture changed) — load what we can
            logger.warning("AI Confirmer strict load failed, trying non-strict: %s", e)
            try:
                # Filter out keys with shape mismatches before loading
                model_state = self.model.state_dict()
                compatible = {}
                skipped = 0
                for k, v in cleaned.items():
                    if k in model_state and model_state[k].shape == v.shape:
                        compatible[k] = v
                    else:
                        skipped += 1
                incompatible = self.model.load_state_dict(compatible, strict=False)
                logger.warning("AI Confirmer loaded %d/%d params (skipped %d shape mismatches). "
                               "Predictions may be unreliable.",
                               len(compatible), len(cleaned), skipped)
            except Exception as e2:
                logger.warning("AI Confirmer could not load at all: %s — will passthrough", e2)
                self._loaded = False
                self.model = None
                return
        self.model.eval()
        self._loaded = True
        logger.info("AI Confirmer loaded: %s on %s", self.checkpoint_dir.name, self.device)

    def confirm(
        self,
        setup: StrategySetup,
        candles: Dict[str, pd.DataFrame],
    ) -> Optional[ConfirmedSignal]:
        """Confirm or reject a strategy setup using the AI model.

        Args:
            setup: The strategy's proposed entry.
            candles: Dict of TF → DataFrame for model input preparation.

        Returns:
            ConfirmedSignal if the model agrees, None if it rejects.
        """
        try:
            self._load_model()
        except Exception as e:
            logger.warning("AI Confirmer load failed: %s — passing through", e)
            return self._passthrough(setup)

        if self.model is None:
            return self._passthrough(setup)

        try:
            # Prepare model input from candles
            model_input = self._prepare_input(candles)
            if model_input is None:
                # Can't prepare input — pass through without AI confirmation
                return self._passthrough(setup)

            # Run inference
            with torch.no_grad():
                out = self.model(model_input)

            signal_logits = out["signal_logits"][0]  # [3]
            probs = torch.softmax(signal_logits, dim=-1)
            pred_idx = signal_logits.argmax(-1).item()
            ai_confidence = out["confidence"].item()
            ai_alignment = out.get("alignment", torch.tensor(0.5)).item()

            # Map strategy direction to model index
            expected_idx = setup.direction.value  # BUY=0, SELL=1

            ai_direction_agrees = (pred_idx == expected_idx)
            direction_prob = probs[expected_idx].item()

        except Exception as e:
            logger.warning("AI inference failed: %s — passing through", e)
            return self._passthrough(setup)

        # ── Confirmation checks ──────────────────────────────────────────
        if not ai_direction_agrees:
            logger.debug(
                "AI REJECTED: strategy=%s, model=%s (conf=%.3f)",
                setup.direction.name, Signal(pred_idx).name, ai_confidence,
            )
            return None

        if ai_confidence < self.min_ai_confidence:
            logger.debug("AI REJECTED: confidence %.3f < %.3f", ai_confidence, self.min_ai_confidence)
            return None

        if ai_alignment < self.min_alignment:
            logger.debug("AI REJECTED: alignment %.3f < %.3f", ai_alignment, self.min_alignment)
            return None

        # ── Compute combined confidence ──────────────────────────────────
        combined = (
            self.strategy_weight * setup.confidence
            + (1 - self.strategy_weight) * ai_confidence
        )

        # ── Build TradeSignal (SL/TP from strategy, not model) ───────────
        trade_signal = TradeSignal(
            signal=setup.direction,
            confidence=combined,
            entry_price=setup.entry_price,
            stop_loss=setup.sl_pips,
            take_profit=setup.tp_pips,
            trailing_stop_pct=setup.trailing_stop_pct,
            timestamp=setup.timestamp,
        )

        return ConfirmedSignal(
            setup=setup,
            ai_direction_agrees=True,
            ai_confidence=ai_confidence,
            ai_alignment=ai_alignment,
            combined_confidence=round(combined, 4),
            trade_signal=trade_signal,
            ai_raw={
                "probs": [round(p, 4) for p in probs.tolist()],
                "direction_prob": round(direction_prob, 4),
                "pred_signal": Signal(pred_idx).name,
            },
        )

    def _passthrough(self, setup: StrategySetup) -> ConfirmedSignal:
        """Create a confirmed signal without AI validation (fallback)."""
        trade_signal = TradeSignal(
            signal=setup.direction,
            confidence=setup.confidence,
            entry_price=setup.entry_price,
            stop_loss=setup.sl_pips,
            take_profit=setup.tp_pips,
            trailing_stop_pct=setup.trailing_stop_pct,
            timestamp=setup.timestamp,
        )
        return ConfirmedSignal(
            setup=setup,
            ai_direction_agrees=True,
            ai_confidence=0.0,
            ai_alignment=0.0,
            combined_confidence=setup.confidence,
            trade_signal=trade_signal,
        )

    def _prepare_input(self, candles: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Prepare model input tensors from candle DataFrames.

        Uses the dataset pipeline's prepare_features() to compute features,
        then wraps them into the batch format WaveTraderMTF expects.
        """
        if self.config is None:
            return None

        try:
            from ..dataset import prepare_features, MTFForexDataset

            # Prepare features for each timeframe
            prepared = {}
            for tf in self.config.timeframes:
                if tf not in candles or len(candles[tf]) < 50:
                    return None
                prepared[tf] = prepare_features(candles[tf])

            # Create a dataset to get properly formatted tensors
            dataset = MTFForexDataset(
                candles, self.config, lookahead=1,
                pair=self.config.pair,
            )
            if len(dataset) == 0:
                return None

            # Use the last available sample
            sample = dataset[-1]
            model_input = {}
            for k, v in sample.items():
                if k == "label":
                    continue
                if isinstance(v, dict):
                    model_input[k] = {
                        feat: tensor.unsqueeze(0).to(self.device)
                        for feat, tensor in v.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                elif isinstance(v, torch.Tensor):
                    model_input[k] = v.unsqueeze(0).to(self.device)

            return model_input

        except Exception as e:
            logger.warning("Failed to prepare model input: %s", e)
            return None
