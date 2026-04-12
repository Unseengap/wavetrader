"""
State persistence for WaveTrader live deployment.

Handles saving and loading of:
  - Model weights
  - CWC hidden state (critical — 25hr warmup if lost)
  - Resonance Buffer contents
  - Equity / position tracking state
  - Bar history for feature computation

Checkpoints are saved to persistent disk every N bars to survive restarts.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class LiveState:
    """Serialisable snapshot of the streaming engine's mutable state."""
    timestamp: str                    # ISO-8601
    bar_count: int                    # Total bars processed since start
    balance: float
    equity: float
    peak_equity: float
    max_drawdown: float
    open_trade_id: Optional[str]      # OANDA trade ID if position is open
    open_trade_direction: Optional[str]  # "BUY" or "SELL"
    open_trade_entry: Optional[float]
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    model_version: str


class StateManager:
    """
    Manages checkpoint save/load for live trading state.

    Directory layout:
        checkpoint_dir/
            latest.pt          – most recent full checkpoint (torch)
            latest_meta.json   – human-readable metadata
            history/
                checkpoint_<unix>.pt   – rolling checkpoint history (keep last 5)
    """

    def __init__(
        self,
        checkpoint_dir: str = "/data/checkpoints",
        max_history: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history_dir = self.checkpoint_dir / "history"
        self.max_history = max_history

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        live_state: LiveState,
        resonance_waves: Optional[List[Tensor]] = None,
        resonance_outcomes: Optional[List[float]] = None,
        bar_history: Optional[Dict[str, Any]] = None,
        recent_ranges: Optional[List[float]] = None,
    ) -> Path:
        """
        Save a full checkpoint to disk.

        This is atomic: write to temp file, then rename, to avoid corruption
        if the process is killed mid-write.
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "live_state": asdict(live_state),
            "resonance_waves": [w.cpu() for w in resonance_waves] if resonance_waves else [],
            "resonance_outcomes": resonance_outcomes or [],
            "bar_history": bar_history or {},
            "recent_ranges": recent_ranges or [],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Atomic write: save to temp, then rename
        latest_path = self.checkpoint_dir / "latest.pt"
        tmp_path = self.checkpoint_dir / "latest.pt.tmp"

        torch.save(checkpoint, tmp_path)
        tmp_path.rename(latest_path)

        # Save human-readable metadata
        meta = {
            "timestamp": live_state.timestamp,
            "bar_count": live_state.bar_count,
            "balance": live_state.balance,
            "equity": live_state.equity,
            "total_trades": live_state.total_trades,
            "win_rate": (
                live_state.winning_trades / max(live_state.total_trades, 1)
            ),
            "max_drawdown": live_state.max_drawdown,
            "total_pnl": live_state.total_pnl,
        }
        meta_path = self.checkpoint_dir / "latest_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Archive to history (rolling)
        ts = int(time.time())
        history_path = self.history_dir / f"checkpoint_{ts}.pt"
        torch.save(checkpoint, history_path)
        self._prune_history()

        logger.info(
            "Checkpoint saved: bar=%d balance=%.2f equity=%.2f",
            live_state.bar_count, live_state.balance, live_state.equity,
        )
        return latest_path

    def load_checkpoint(
        self, path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint (or a specific one).

        Returns None if no checkpoint exists (cold start).
        """
        if path:
            load_path = Path(path)
        else:
            load_path = self.checkpoint_dir / "latest.pt"

        if not load_path.exists():
            logger.info("No checkpoint found at %s — cold start", load_path)
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(load_path, weights_only=False, map_location=device)
        logger.info(
            "Loaded checkpoint from %s (bar_count=%d)",
            load_path,
            checkpoint.get("live_state", {}).get("bar_count", 0),
        )
        return checkpoint

    def restore_model(
        self, model: torch.nn.Module, checkpoint: Dict[str, Any]
    ) -> None:
        """Restore model weights from checkpoint."""
        if "model_state_dict" in checkpoint:
            raw_sd = checkpoint["model_state_dict"]
        else:
            raw_sd = checkpoint
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        # Filter out keys with shape mismatches (architecture evolution)
        model_sd = model.state_dict()
        compatible = {
            k: v for k, v in cleaned.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if missing:
            logger.warning("Checkpoint partial load: %d/%d params (%d missing/reshaped)",
                           len(compatible), len(model_sd), len(missing))
        logger.info("Model weights restored")

    def restore_resonance_buffer(
        self, buffer: Any, checkpoint: Dict[str, Any]
    ) -> None:
        """Restore Resonance Buffer contents from checkpoint."""
        waves = checkpoint.get("resonance_waves", [])
        outcomes = checkpoint.get("resonance_outcomes", [])
        for wave, outcome in zip(waves, outcomes):
            buffer._waves.append(wave)
            buffer._outcomes.append(outcome)
        logger.info("Resonance buffer restored: %d entries", len(waves))

    def _prune_history(self) -> None:
        """Keep only the most recent N checkpoints in history/."""
        history_files = sorted(
            self.history_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(history_files) > self.max_history:
            oldest = history_files.pop(0)
            oldest.unlink()
            logger.debug("Pruned old checkpoint: %s", oldest.name)

    def get_latest_meta(self) -> Optional[Dict]:
        """Read the latest checkpoint metadata without loading the full checkpoint."""
        meta_path = self.checkpoint_dir / "latest_meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)
