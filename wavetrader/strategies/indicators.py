"""
Unified indicator computation for all strategies.

Wraps wavetrader.indicators + wavetrader.amd_features into a single
compute_all_indicators() call that returns an IndicatorBundle.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import IndicatorBundle
from ..indicators import calculate_rsi, calculate_atr, calculate_adx, classify_structure
from ..amd_features import (
    compute_asian_range,
    compute_london_sweep,
    compute_engulfing_patterns,
    compute_fair_value_gaps,
    compute_sr_zones,
    compute_orb_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMA helper
# ─────────────────────────────────────────────────────────────────────────────

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — no look-ahead."""
    out = np.full_like(prices, np.nan, dtype=np.float64)
    if len(prices) < period:
        return out
    out[period - 1] = prices[:period].mean()
    alpha = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Bands helper
# ─────────────────────────────────────────────────────────────────────────────

def _bollinger(closes: np.ndarray, window: int = 20, num_std: float = 2.0):
    """Returns (mid, upper, lower) arrays — no look-ahead."""
    n = len(closes)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = closes[i - window + 1: i + 1]
        m = segment.mean()
        s = segment.std(ddof=0)
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
    return mid, upper, lower


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_indicators(
    candles: Dict[str, pd.DataFrame],
    entry_tf: str = "15min",
    pair: str = "GBP/JPY",
    compute_amd: bool = True,
) -> IndicatorBundle:
    """Compute all indicators across all timeframes.

    Args:
        candles: Dict of TF → DataFrame with [date, open, high, low, close, volume].
        entry_tf: Which timeframe is the entry timeframe.
        pair: Currency pair (for metadata).
        compute_amd: Whether to compute AMD-specific features (Asian range,
                     London sweep, etc.). Set False for non-session strategies.

    Returns:
        Fully populated IndicatorBundle.
    """
    bundle = IndicatorBundle(entry_tf=entry_tf, pair=pair)

    for tf, df in candles.items():
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        opens = df["open"].values.astype(np.float64)

        # Core indicators for every timeframe
        bundle.rsi[tf] = calculate_rsi(closes)
        bundle.atr[tf] = calculate_atr(highs, lows, closes)
        bundle.adx[tf] = calculate_adx(highs, lows, closes)
        bundle.structure[tf] = classify_structure(highs, lows)

        # EMAs
        bundle.ema_20[tf] = _ema(closes, 20)
        bundle.ema_50[tf] = _ema(closes, 50)
        bundle.ema_200[tf] = _ema(closes, 200)

        # Bollinger Bands
        mid, upper, lower = _bollinger(closes)
        bundle.bollinger_mid[tf] = mid
        bundle.bollinger_upper[tf] = upper
        bundle.bollinger_lower[tf] = lower

    # ── AMD-specific features (entry TF only) ────────────────────────────
    if compute_amd and entry_tf in candles:
        edf = candles[entry_tf]
        if "date" in edf.columns:
            timestamps = pd.DatetimeIndex(edf["date"])
        else:
            timestamps = edf.index

        h = edf["high"].values.astype(np.float64)
        l = edf["low"].values.astype(np.float64)
        c = edf["close"].values.astype(np.float64)
        o = edf["open"].values.astype(np.float64)

        bundle.asian_range = compute_asian_range(timestamps, h, l, c)
        bundle.london_sweep = compute_london_sweep(timestamps, h, l, c)
        bundle.engulfing = compute_engulfing_patterns(o, h, l, c)
        bundle.fair_value_gaps = compute_fair_value_gaps(h, l, c)
        bundle.sr_zones = compute_sr_zones(h, l, c)
        bundle.orb_features = compute_orb_features(timestamps, h, l, c)

    return bundle
