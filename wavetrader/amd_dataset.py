"""
AMD Scalper Dataset — extends MTFForexDataset with AMD-specific features.

Adds 23 AMD features (Asian range, London sweep, engulfing, FVG, S&R, ORB)
and phase labels (ACCUM/MANIP/DIST/INVALID) to each timeframe's tensor dict.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .amd_features import build_amd_features, compute_amd_phase_labels
from .config import AMDScalperConfig
from .dataset import MTFForexDataset, prepare_features, _get_label, _pip_size

# AMD feature column names (23 total) — must match build_amd_features() output
AMD_FEATURE_COLS: List[str] = [
    # Asian range (5)
    "asian_high_rel", "asian_low_rel", "asian_range_norm",
    "price_vs_asian_mid", "asian_valid",
    # London sweep (4)
    "sweep_direction", "sweep_magnitude", "sweep_aggression", "sweep_valid",
    # Engulfing (3)
    "engulfing_type", "engulfing_strength", "engulfing_vol_confirm",
    # FVG (4)
    "fvg_type", "fvg_size", "fvg_midpoint", "fvg_filled",
    # S&R (3)
    "sr_proximity", "sr_strength", "sr_is_flip",
    # ORB (4)
    "orb_high_dist", "orb_low_dist", "orb_breakout", "orb_valid",
]


class AMDForexDataset(Dataset):
    """
    Multi-timeframe dataset with AMD-specific features.

    Same structure as MTFForexDataset, but each timeframe's tensor dict
    includes an extra ``amd_feats`` key [T, 23] and the sample includes
    a ``phase_label`` alongside the signal ``label``.
    """

    _TF_RATIOS: Dict[str, int] = {
        "1min": 1, "5min": 5, "15min": 15, "30min": 30,
        "1h": 60, "4h": 240, "1d": 1440,
    }

    def __init__(
        self,
        dataframes: Dict[str, pd.DataFrame],
        config: Optional[AMDScalperConfig] = None,
        lookahead: int = 6,          # 6 × 5min = 30min (scalping horizon)
        threshold_pips: float = 15.0, # Tighter for scalping
        pair: str = "GBP/JPY",
    ) -> None:
        self.config = config or AMDScalperConfig()
        self.lookahead = lookahead
        self.threshold_pips = threshold_pips
        self.pair = pair

        # Prepare standard features, then ADD AMD features on top
        self.prepared: Dict[str, pd.DataFrame] = {}
        for tf, df in dataframes.items():
            prepped = prepare_features(df.copy(), lookahead=lookahead, pair=pair)
            prepped = build_amd_features(prepped)
            self.prepared[tf] = prepped

        entry_tf = self.config.entry_timeframe
        entry_df = self.prepared[entry_tf]
        lookback = self.config.lookbacks[entry_tf]
        self.valid_indices = list(range(lookback, len(entry_df) - lookahead))
        self.entry_tf = entry_tf

    def _get_tf_slice(self, tf: str, entry_idx: int) -> pd.DataFrame:
        entry_df = self.prepared[self.entry_tf]
        tf_df = self.prepared[tf]
        lookback = self.config.lookbacks[tf]

        if tf == self.entry_tf:
            start = max(0, entry_idx - lookback)
            return tf_df.iloc[start:entry_idx]

        if "date" in entry_df.columns and "date" in tf_df.columns:
            entry_time = entry_df.iloc[entry_idx]["date"]
            mask = tf_df["date"] <= entry_time
            tf_idx = int(mask.sum()) - 1 if mask.any() else 0
        else:
            entry_mins = self._TF_RATIOS.get(self.entry_tf, 5)
            tf_mins = self._TF_RATIOS.get(tf, 60)
            tf_idx = min(int(entry_idx * entry_mins / tf_mins), len(tf_df) - 1)

        start = max(0, tf_idx - lookback)
        return tf_df.iloc[start:tf_idx + 1]

    @staticmethod
    def _to_tensors(sl: pd.DataFrame, lookback: int) -> Dict[str, Tensor]:
        if len(sl) < lookback:
            pad = pd.concat([sl.iloc[:1]] * (lookback - len(sl)) + [sl])
        else:
            pad = sl
        pad = pad.iloc[-lookback:]

        tensors = {
            "ohlcv": torch.tensor(
                pad[["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]].values,
                dtype=torch.float32,
            ),
            "structure": torch.tensor(
                pad[[f"structure_{i}" for i in range(8)]].values,
                dtype=torch.float32,
            ),
            "rsi": torch.tensor(
                pad[["rsi_norm", "rsi_delta_norm", "rsi_accel_norm"]].values,
                dtype=torch.float32,
            ),
            "volume": torch.tensor(
                pad[["volume_norm", "volume_ratio", "volume_delta"]].values,
                dtype=torch.float32,
            ),
        }

        # AMD features (23-dim)
        amd_cols_present = [c for c in AMD_FEATURE_COLS if c in pad.columns]
        if len(amd_cols_present) == len(AMD_FEATURE_COLS):
            tensors["amd_feats"] = torch.tensor(
                pad[AMD_FEATURE_COLS].values, dtype=torch.float32,
            )
        else:
            # Fallback: zeros (e.g. higher TFs without session data)
            tensors["amd_feats"] = torch.zeros(lookback, len(AMD_FEATURE_COLS), dtype=torch.float32)

        # Regime context: session flags + ATR percentile
        regime_cols = ["session_tokyo", "session_london", "session_newyork", "atr_pct"]
        if all(c in pad.columns for c in regime_cols):
            tensors["regime"] = torch.tensor(
                pad[regime_cols].values, dtype=torch.float32,
            )
        else:
            tensors["regime"] = torch.zeros(lookback, 4, dtype=torch.float32)

        return tensors

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        actual = self.valid_indices[idx]
        result: Dict = {}

        for tf in self.config.timeframes:
            sl = self._get_tf_slice(tf, actual)
            lookback = self.config.lookbacks[tf]
            result[tf] = self._to_tensors(sl, lookback)

        # Signal label (BUY/SELL/HOLD)
        entry_row = self.prepared[self.entry_tf].iloc[actual]
        label = _get_label(entry_row, self.threshold_pips, self.pair)
        result["label"] = torch.tensor(label, dtype=torch.long)

        # Phase label (from AMD phase column)
        if "amd_phase" in self.prepared[self.entry_tf].columns:
            phase = int(self.prepared[self.entry_tf].iloc[actual]["amd_phase"])
            result["phase_label"] = torch.tensor(phase, dtype=torch.long)

        return result


def amd_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader with AMDForexDataset items."""
    skip_keys = {"label", "phase_label"}
    timeframes = [k for k in batch[0] if k not in skip_keys]

    result: Dict = {
        tf: {
            feat: torch.stack([b[tf][feat] for b in batch])
            for feat in batch[0][tf]
        }
        for tf in timeframes
    }
    result["label"] = torch.stack([b["label"] for b in batch])
    if "phase_label" in batch[0]:
        result["phase_label"] = torch.stack([b["phase_label"] for b in batch])
    return result
