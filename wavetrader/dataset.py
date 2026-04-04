"""
PyTorch Dataset classes for single-timeframe and multi-timeframe training.

Label generation note
─────────────────────
Labels are derived from FUTURE price movement (lookahead bars ahead of the
current bar).  This is intentional for supervised training and does NOT cause
inference-time leakage because:
  1. Future columns (future_high / future_low) are never included in the
     tensors passed to the model.
  2. At inference time no labels are computed at all — the model generates
     signals from the encoders alone.

Walk-forward / validation split
────────────────────────────────
Always split data chronologically (earlier bars → train, later bars → val/test).
Never shuffle the full dataset before splitting; that would contaminate future
bars into the training window.  See utils.walk_forward_splits() for proper
purged time-series cross-validation.
"""
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import MTFConfig, SignalConfig
from .indicators import calculate_atr, calculate_rsi, classify_structure, session_features
from .types import Signal

_DEFAULT_SINGLE_CONFIG = SignalConfig()
_DEFAULT_MTF_CONFIG    = MTFConfig()

# Pip size per pair — used for threshold conversion
_PIP_SIZE: Dict[str, float] = {
    "GBP/JPY": 0.01,
    "EUR/USD": 0.0001,
    "USD/JPY": 0.01,
    "GBP/USD": 0.0001,
    "EUR/JPY": 0.01,
    "default": 0.0001,
}


def _pip_size(pair: str) -> float:
    return _PIP_SIZE.get(pair, _PIP_SIZE["default"])


# ─────────────────────────────────────────────────────────────────────────────
# Shared feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(
    df:       pd.DataFrame,
    lookahead: int = 10,
    pair:     str  = "GBP/JPY",
) -> pd.DataFrame:
    """
    Add all derived features in-place to a copy of `df`.
    Expects columns: open, high, low, close, volume
    Optionally: date (datetime-like) for session features.

    Returns the enriched DataFrame.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    # ── RSI features ──────────────────────────────────────────────────────────
    df["rsi"]          = calculate_rsi(df["close"].values)
    df["rsi_delta"]    = df["rsi"].diff().fillna(0.0)
    df["rsi_accel"]    = df["rsi_delta"].diff().fillna(0.0)
    df["rsi_norm"]     = df["rsi"] / 100.0
    df["rsi_delta_norm"] = (df["rsi_delta"] / 20.0).clip(-1.0, 1.0)
    df["rsi_accel_norm"] = (df["rsi_accel"] / 10.0).clip(-1.0, 1.0)

    # ── Volume features ───────────────────────────────────────────────────────
    vol_ma              = df["volume"].rolling(20).mean().fillna(df["volume"])
    df["volume_ma"]     = vol_ma
    df["volume_ratio"]  = (df["volume"] / vol_ma).clip(0.0, 5.0) / 5.0
    df["volume_delta"]  = df["volume"].pct_change().fillna(0.0).clip(-1.0, 1.0)
    vol_max             = df["volume"].rolling(100).max().fillna(df["volume"])
    df["volume_norm"]   = (df["volume"] / vol_max).fillna(0.5).clip(0.0, 1.0)

    # ── Market structure ──────────────────────────────────────────────────────
    structure = classify_structure(df["high"].values, df["low"].values)
    for i in range(8):
        df[f"structure_{i}"] = structure[:, i]

    # ── Normalised OHLC (z-score over rolling 100-bar window) ─────────────────
    roll_mean = df["close"].rolling(100).mean()
    roll_std  = df["close"].rolling(100).std().replace(0.0, np.nan)
    for col in ("open", "high", "low", "close"):
        normed = (df[col] - roll_mean) / roll_std
        df[f"{col}_norm"] = normed.fillna(0.0).clip(-3.0, 3.0) / 3.0

    # ── ATR-based regime context ──────────────────────────────────────────────
    atr_arr  = calculate_atr(df["high"].values, df["low"].values, df["close"].values)
    atr_pct  = pd.Series(atr_arr).rolling(100).rank(pct=True).fillna(0.5).values
    df["atr_pct"] = atr_pct

    # ── Session flags (Tokyo / London / NY) ───────────────────────────────────
    if "date" in df.columns:
        sess = session_features(pd.DatetimeIndex(df["date"]))
        df["session_tokyo"]   = sess[:, 0]
        df["session_london"]  = sess[:, 1]
        df["session_newyork"] = sess[:, 2]
    else:
        df["session_tokyo"] = df["session_london"] = df["session_newyork"] = 0.5

    # ── Labels (future high / low within lookahead bars) ──────────────────────
    # future_high[i] = max(high[i+1 .. i+lookahead])
    # future_low[i]  = min(low[i+1  .. i+lookahead])
    df["future_high"] = (
        df["high"].shift(-lookahead).rolling(lookahead).max()
    )
    df["future_low"] = (
        df["low"].shift(-lookahead).rolling(lookahead).min()
    )

    return df.ffill().fillna(0.0)


def _get_label(row: pd.Series, threshold_pips: float, pair: str) -> int:
    entry       = row["close"]
    threshold   = threshold_pips * _pip_size(pair)
    max_up      = row["future_high"] - entry
    max_down    = entry - row["future_low"]

    if max_up >= threshold and max_up > max_down:
        return Signal.BUY.value
    if max_down >= threshold and max_down > max_up:
        return Signal.SELL.value
    return Signal.HOLD.value


# ─────────────────────────────────────────────────────────────────────────────
# Single-timeframe dataset
# ─────────────────────────────────────────────────────────────────────────────

class ForexDataset(Dataset):
    """
    Single-timeframe dataset for FluxSignal training.

    Each sample covers a [lookback] bar window ending just before `actual_idx`.
    Features are strictly causal — no future data leaks into model inputs.
    """

    def __init__(
        self,
        df:              pd.DataFrame,
        lookback:        int   = 100,
        lookahead:       int   = 10,
        threshold_pips:  float = 20.0,
        pair:            str   = "GBP/JPY",
    ) -> None:
        self.lookback       = lookback
        self.lookahead      = lookahead
        self.threshold_pips = threshold_pips
        self.pair           = pair

        self.df           = prepare_features(df, lookahead=lookahead, pair=pair)
        self.valid_indices = list(range(lookback, len(self.df) - lookahead))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        actual    = self.valid_indices[idx]
        sl        = self.df.iloc[actual - self.lookback : actual]

        ohlcv = torch.tensor(
            sl[["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]].values,
            dtype=torch.float32,
        )
        structure = torch.tensor(
            sl[[f"structure_{i}" for i in range(8)]].values,
            dtype=torch.float32,
        )
        rsi = torch.tensor(
            sl[["rsi_norm", "rsi_delta_norm", "rsi_accel_norm"]].values,
            dtype=torch.float32,
        )
        volume = torch.tensor(
            sl[["volume_norm", "volume_ratio", "volume_delta"]].values,
            dtype=torch.float32,
        )
        regime = torch.tensor(
            sl[["session_tokyo", "session_london", "session_newyork", "atr_pct"]].values,
            dtype=torch.float32,
        )
        label = _get_label(self.df.iloc[actual], self.threshold_pips, self.pair)

        return {
            "ohlcv":     ohlcv,
            "structure": structure,
            "rsi":       rsi,
            "volume":    volume,
            "regime":    regime,
            "label":     torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe dataset
# ─────────────────────────────────────────────────────────────────────────────

class MTFForexDataset(Dataset):
    """
    Multi-timeframe dataset for WaveTraderMTF training.

    Aligns higher-TF bars to the entry-TF timestamp using the 'date' column.
    Falls back to a fixed ratio when timestamps are unavailable.
    """

    _TF_RATIOS: Dict[str, int] = {
        "1min": 1, "5min": 5, "15min": 15, "30min": 30,
        "1h": 60,  "4h": 240, "1d": 1440,
    }

    def __init__(
        self,
        dataframes:      Dict[str, pd.DataFrame],
        config:          Optional[MTFConfig] = None,
        lookahead:       int   = 10,
        threshold_pips:  float = 20.0,
        pair:            str   = "GBP/JPY",
    ) -> None:
        self.config         = config or _DEFAULT_MTF_CONFIG
        self.lookahead      = lookahead
        self.threshold_pips = threshold_pips
        self.pair           = pair

        self.prepared: Dict[str, pd.DataFrame] = {
            tf: prepare_features(df.copy(), lookahead=lookahead, pair=pair)
            for tf, df in dataframes.items()
        }

        entry_tf      = self.config.entry_timeframe
        entry_df      = self.prepared[entry_tf]
        lookback      = self.config.lookbacks[entry_tf]
        self.valid_indices = list(range(lookback, len(entry_df) - lookahead))
        self.entry_tf = entry_tf

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_tf_slice(self, tf: str, entry_idx: int) -> pd.DataFrame:
        entry_df = self.prepared[self.entry_tf]
        tf_df    = self.prepared[tf]
        lookback = self.config.lookbacks[tf]

        if tf == self.entry_tf:
            start = max(0, entry_idx - lookback)
            return tf_df.iloc[start:entry_idx]

        if "date" in entry_df.columns and "date" in tf_df.columns:
            entry_time = entry_df.iloc[entry_idx]["date"]
            mask       = tf_df["date"] <= entry_time
            tf_idx     = int(mask.sum()) - 1 if mask.any() else 0
        else:
            entry_mins = self._TF_RATIOS.get(self.entry_tf, 15)
            tf_mins    = self._TF_RATIOS.get(tf, 60)
            tf_idx     = min(int(entry_idx * entry_mins / tf_mins), len(tf_df) - 1)

        start = max(0, tf_idx - lookback)
        return tf_df.iloc[start : tf_idx + 1]

    @staticmethod
    def _to_tensors(sl: pd.DataFrame, lookback: int) -> Dict[str, Tensor]:
        if len(sl) < lookback:
            pad = pd.concat([sl.iloc[:1]] * (lookback - len(sl)) + [sl])
        else:
            pad = sl
        pad = pad.iloc[-lookback:]

        return {
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

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Tensor]]:
        actual  = self.valid_indices[idx]
        result: Dict = {}

        for tf in self.config.timeframes:
            sl       = self._get_tf_slice(tf, actual)
            lookback = self.config.lookbacks[tf]
            result[tf] = self._to_tensors(sl, lookback)

        label = _get_label(
            self.prepared[self.entry_tf].iloc[actual],
            self.threshold_pips,
            self.pair,
        )
        result["label"] = torch.tensor(label, dtype=torch.long)
        return result


def mtf_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader with MTFForexDataset items."""
    timeframes = [k for k in batch[0] if k != "label"]
    result: Dict = {
        tf: {
            feat: torch.stack([b[tf][feat] for b in batch])
            for feat in batch[0][tf]
        }
        for tf in timeframes
    }
    result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Resonance Buffer — episodic memory for “last N significant events”
# ─────────────────────────────────────────────────────────────────────────────

class ResonanceBuffer:
    """
    Sliding window of high-salience wave states (large PnL moves or strong signals).

    Provides lightweight episodic context:  "what happened last time the market
    wave looked like this?"  Replaces Crystal Memory with a ~100-element
    rolling buffer — trivial compute, no FAISS, no pre-trained embedding space.

    Note on when to use:
        Similarity is raw L2 distance in *model wave space*, which is only
        meaningful after the model has learned a useful wave representation.
        Use ResonanceBuffer during fine-tuning / online adaptation phases, not
        during initial random-weight pretraining.
    """

    def __init__(
        self,
        capacity:            int   = 100,
        wave_dim:            int   = 608,
        salience_threshold:  float = 2.0,
    ) -> None:
        self.capacity           = capacity
        self.wave_dim           = wave_dim
        self.salience_threshold = salience_threshold

        self._waves:    deque = deque(maxlen=capacity)
        self._outcomes: deque = deque(maxlen=capacity)   # float PnL values

    # ── Storage ───────────────────────────────────────────────────────────

    def store(self, wave: Tensor, outcome: float) -> bool:
        """
        Conditionally store a wave state and its associated trade outcome.

        A state is stored only when |outcome| is more than `salience_threshold`
        standard deviations above the recent mean.  Once the buffer has fewer
        than 10 entries, every non-zero outcome is stored.

        Returns True if the state was stored.
        """
        if not self.is_salient(outcome):
            return False
        self._waves.append(wave.detach().cpu())
        self._outcomes.append(float(outcome))
        return True

    def is_salient(self, outcome: float) -> bool:
        """True if this outcome qualifies as a memorable event."""
        n = len(self._outcomes)
        if n < 10:
            return abs(outcome) > 0.0
        hist = np.array(list(self._outcomes)[-50:])
        std  = hist.std()
        if std < 1e-8:
            return False
        return abs(outcome - hist.mean()) > self.salience_threshold * std

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query_wave: Tensor, k: int = 5) -> Optional[Tensor]:
        """
        Return the top-k most similar stored waves as [k, wave_dim], or None
        when the buffer has fewer than k entries.

        Similarity metric: negative L2 distance in wave space.
        """
        if len(self._waves) < k:
            return None

        stored = torch.stack(list(self._waves))     # [N, wave_dim]
        q = query_wave.detach().cpu()
        if q.dim() == 2:
            q = q[-1]    # take last time-step if sequence

        dists    = torch.cdist(q.unsqueeze(0).float(), stored.float()).squeeze(0)   # [N]
        topk_idx = dists.topk(k, largest=False).indices
        return stored[topk_idx]   # [k, wave_dim]

    def retrieve_with_outcomes(
        self, query_wave: Tensor, k: int = 5
    ) -> Optional[Tuple[Tensor, List[float]]]:
        """
        Same as retrieve(), but also returns the k associated PnL outcomes.
        Useful for conditioning: did similar past regimes tend to win or lose?
        """
        if len(self._waves) < k:
            return None

        stored   = torch.stack(list(self._waves))
        outcomes = list(self._outcomes)
        q        = query_wave.detach().cpu()
        if q.dim() == 2:
            q = q[-1]

        dists    = torch.cdist(q.unsqueeze(0).float(), stored.float()).squeeze(0)
        topk_idx = dists.topk(k, largest=False).indices.tolist()
        return stored[topk_idx], [outcomes[i] for i in topk_idx]

    # ── Introspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._waves)

    def mean_outcome(self) -> float:
        """Mean PnL of stored events (useful for sanity-checking salience filter)."""
        if not self._outcomes:
            return 0.0
        return float(np.mean(list(self._outcomes)))
