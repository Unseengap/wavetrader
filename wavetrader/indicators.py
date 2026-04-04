"""
Technical indicator calculations — pure NumPy, no look-ahead.
"""
from typing import List, Tuple

import numpy as np

from .types import StructureType


# ─────────────────────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder-smoothed RSI.  All values at index < period are set to 50 (neutral).
    No look-ahead: each value only uses prices[0..i].
    """
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas,  0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    if len(gains) >= period:
        avg_gain[period] = gains[:period].mean()
        avg_loss[period] = losses[:period].mean()

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)

    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return np.nan_to_num(rsi, nan=50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Swing-point detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_swing_points(
    highs: np.ndarray,
    lows:  np.ndarray,
    left:  int = 5,
    right: int = 5,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Pivot-point algorithm: a bar is a swing high (low) when its high (low)
    is the maximum (minimum) of the surrounding [left + right + 1] bar window.

    NOTE: requires `right` future bars — used only during dataset preparation,
    not during live inference.
    """
    n = len(highs)
    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []

    for i in range(left, n - right):
        if highs[i] == highs[i - left : i + right + 1].max():
            swing_highs.append((i, highs[i]))
        if lows[i]  == lows[i  - left : i + right + 1].min():
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


# ─────────────────────────────────────────────────────────────────────────────
# Market-structure classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_structure(
    highs:       np.ndarray,
    lows:        np.ndarray,
    swing_left:  int = 5,
    swing_right: int = 5,
) -> np.ndarray:
    """
    Returns a (n, 8) float32 array per bar:
      [0:5]  one-hot of StructureType (HH / HL / LL / LH / NONE)
      [5]    normalised bars since last swing (0-1, capped at 50 bars)
      [6]    recent swing activity rate (swings / 20-bar window)
      [7]    trend bias: (bullish_swings - bearish_swings) / total  ∈ [-1, 1]
    """
    n = len(highs)
    structure = np.full(n, StructureType.NONE.value, dtype=np.int64)

    swing_highs, swing_lows = detect_swing_points(
        highs, lows, swing_left, swing_right
    )

    for i, (idx, price) in enumerate(swing_highs):
        if i > 0:
            prev = swing_highs[i - 1][1]
            structure[idx] = (
                StructureType.HH.value if price > prev else StructureType.LH.value
            )

    for i, (idx, price) in enumerate(swing_lows):
        if i > 0:
            prev = swing_lows[i - 1][1]
            structure[idx] = (
                StructureType.HL.value if price > prev else StructureType.LL.value
            )

    filled: np.ndarray = np.zeros((n, 8), dtype=np.float32)
    last_struct = StructureType.NONE.value
    last_swing  = 0
    window      = 20

    for i in range(n):
        if structure[i] != StructureType.NONE.value:
            last_struct = int(structure[i])
            last_swing  = i

        filled[i, last_struct] = 1.0
        filled[i, 5] = min((i - last_swing) / 50.0, 1.0)

        recent = structure[max(0, i - window) : i]
        filled[i, 6] = (recent != StructureType.NONE.value).sum() / window

        bullish = int(
            ((recent == StructureType.HH.value) | (recent == StructureType.HL.value)).sum()
        )
        bearish = int(
            ((recent == StructureType.LL.value) | (recent == StructureType.LH.value)).sum()
        )
        total = bullish + bearish
        filled[i, 7] = (bullish - bearish) / total if total > 0 else 0.0

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# ATR — used for regime / volatility context
# ─────────────────────────────────────────────────────────────────────────────

def calculate_atr(
    highs:  np.ndarray,
    lows:   np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Wilder ATR.  Returns same-length array; first `period` bars = NaN."""
    n  = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr[i] = max(
            highs[i]  - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )

    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ─────────────────────────────────────────────────────────────────────────────
# ADX — trend strength
# ─────────────────────────────────────────────────────────────────────────────

def calculate_adx(
    highs:  np.ndarray,
    lows:   np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Wilder-smoothed ADX.  Returns same-length array; values before 2*period = NaN.

    ADX measures trend *strength* (0-100), not direction:
      • > 25: trending market
      • < 20: ranging / choppy

    No look-ahead: each bar uses only prior highs/lows/closes.
    """
    n  = len(closes)
    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)
    tr       = np.zeros(n)

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        up_move   = highs[i]  - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        dm_plus[i]  = up_move   if up_move   > down_move and up_move   > 0 else 0.0
        dm_minus[i] = down_move if down_move > up_move   and down_move > 0 else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )

    # Wilder smoothing
    atr_w    = np.full(n, np.nan)
    dmp_w    = np.full(n, np.nan)
    dmm_w    = np.full(n, np.nan)

    if n >= period:
        atr_w[period - 1]  = tr[:period].mean()
        dmp_w[period - 1]  = dm_plus[:period].mean()
        dmm_w[period - 1]  = dm_minus[:period].mean()
        for i in range(period, n):
            atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
            dmp_w[i] = dmp_w[i - 1] - dmp_w[i - 1] / period + dm_plus[i]
            dmm_w[i] = dmm_w[i - 1] - dmm_w[i - 1] / period + dm_minus[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus  = 100.0 * np.where(atr_w > 1e-10, dmp_w / atr_w, 0.0)
        di_minus = 100.0 * np.where(atr_w > 1e-10, dmm_w / atr_w, 0.0)
        dx       = 100.0 * np.abs(di_plus - di_minus) / np.where(
            (di_plus + di_minus) > 1e-10, di_plus + di_minus, 1.0
        )

    adx = np.full(n, np.nan)
    start = 2 * period - 1
    if n > start:
        valid_dx = np.where(np.isnan(dx[period - 1:]), 0.0, dx[period - 1:])
        adx[start] = valid_dx[:period].mean()
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return np.nan_to_num(adx, nan=20.0)  # 20 = neutral (no trend)


# ─────────────────────────────────────────────────────────────────────────────
# Hurst Exponent — mean-reverting vs trending
# ─────────────────────────────────────────────────────────────────────────────

def calculate_hurst(prices: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
    """
    Estimate the Hurst exponent via rescaled-range (R/S) analysis over the
    full price series.

    Returns a single scalar:
      H < 0.45  → mean-reverting
      H ≈ 0.50  → random walk
      H > 0.55  → trending / persistent

    Requires at least 2 * max_lag data points; returns 0.5 (neutral) otherwise.
    """
    n = len(prices)
    if n < 2 * max_lag:
        return 0.5

    lags = range(min_lag, min(max_lag, n // 2) + 1)
    log_lags: List[float] = []
    log_rs:   List[float] = []

    for lag in lags:
        sub_returns = np.diff(np.log(np.maximum(prices[:lag + 1], 1e-10)))
        if len(sub_returns) < 2:
            continue
        mean_r = sub_returns.mean()
        deviation = np.cumsum(sub_returns - mean_r)
        r = deviation.max() - deviation.min()
        s = sub_returns.std(ddof=1)
        if s < 1e-10:
            continue
        log_lags.append(np.log(lag))
        log_rs.append(np.log(r / s))

    if len(log_lags) < 2:
        return 0.5

    hurst, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(hurst, 0.0, 1.0))


def session_features(timestamps) -> np.ndarray:
    """
    Return (n, 3) float32: [tokyo_active, london_active, newyork_active].
    Based on UTC hour of each timestamp.
    Helps the model distinguish session behaviour (liquidity / volatility regimes).
    """
    import pandas as pd
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)

    hours = timestamps.hour.to_numpy()
    tokyo   = ((hours >= 0)  & (hours < 9)).astype(np.float32)
    london  = ((hours >= 7)  & (hours < 16)).astype(np.float32)
    newyork = ((hours >= 13) & (hours < 22)).astype(np.float32)

    return np.stack([tokyo, london, newyork], axis=1)
