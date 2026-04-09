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


# ─────────────────────────────────────────────────────────────────────────────
# 4H Reversal Pattern Detection (v3)
# ─────────────────────────────────────────────────────────────────────────────

def detect_reversal_pattern(
    opens:  np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Detect 3-candle reversal patterns on a given timeframe (designed for 4H).

    Pattern (Bullish Reversal — BUY signal):
      1. Candle 1: Bearish (close < open) — part of existing downtrend
      2. Candle 2: Closes HIGHER than Candle 1's close — first reversal sign
      3. Candle 3: Closes >= Candle 2's close (even if it dipped intra-bar) — CONFIRMATION
         The candle 3 can spike down but must close at or above candle 2's close.

    Pattern (Bearish Reversal — SELL signal): Mirror image.

    Returns (n, 8) float32 array per bar:
      [0] reversal_type:       0=none, 1=bullish_reversal, 2=bearish_reversal
      [1] pattern_strength:    0-1, magnitude of reversal relative to pattern range
      [2] pattern_low:         lowest low of the 3-candle window (normalised relative to close)
      [3] pattern_high:        highest high of the 3-candle window (normalised relative to close)
      [4] candle2_recovery:    (c2_close - c1_close) / pattern_range — how strong the reversal candle is
      [5] candle3_confirmation: 1.0 if c3 confirms, 0.0 otherwise
      [6] candle3_dip_recovery: how much c3 dipped then recovered (0=no dip, 1=full recovery from dip)
      [7] trend_reversal_score: composite score 0-1 combining all sub-signals

    All values at index < 2 are zero (need 3 bars of history).
    No look-ahead: each value only uses bars[0..i].
    """
    n = len(closes)
    result = np.zeros((n, 8), dtype=np.float32)

    for i in range(2, n):
        c1_open, c1_high, c1_low, c1_close = opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]
        c2_open, c2_high, c2_low, c2_close = opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]
        c3_open, c3_high, c3_low, c3_close = opens[i], highs[i], lows[i], closes[i]

        pattern_low  = min(c1_low, c2_low, c3_low)
        pattern_high = max(c1_high, c2_high, c3_high)
        pattern_range = pattern_high - pattern_low
        if pattern_range < 1e-10:
            continue

        # Normalise pattern bounds relative to current close
        result[i, 2] = (c3_close - pattern_low) / pattern_range    # pattern_low (normalised)
        result[i, 3] = (pattern_high - c3_close) / pattern_range   # pattern_high (normalised)

        c1_bearish = c1_close < c1_open
        c1_bullish = c1_close > c1_open

        # ── Bullish reversal check ──
        if c1_bearish and c2_close > c1_close:
            c3_confirms = c3_close >= c2_close
            recovery = (c2_close - c1_close) / pattern_range
            c3_dip = max(0.0, c2_close - c3_low) / pattern_range if c3_low < c2_close else 0.0
            c3_dip_recovery = 0.0
            if c3_dip > 0.01:
                # Dipped below c2 but recovered — strong confirmation
                c3_dip_recovery = min(1.0, (c3_close - c3_low) / max(c2_close - c3_low, 1e-10))

            if c3_confirms:
                strength = min(1.0, recovery + 0.3 * c3_dip_recovery)
                result[i, 0] = 1.0  # bullish reversal
                result[i, 1] = strength
                result[i, 4] = min(1.0, recovery)
                result[i, 5] = 1.0  # confirmed
                result[i, 6] = c3_dip_recovery
                result[i, 7] = min(1.0, 0.4 * recovery + 0.3 * 1.0 + 0.3 * c3_dip_recovery)
            elif c3_close > c1_close:
                # Partial — c3 didn't fully confirm but still above c1
                result[i, 0] = 0.0  # not confirmed enough
                result[i, 4] = min(1.0, recovery)
                result[i, 5] = 0.0

        # ── Bearish reversal check ──
        elif c1_bullish and c2_close < c1_close:
            c3_confirms = c3_close <= c2_close
            recovery = (c1_close - c2_close) / pattern_range
            c3_spike = max(0.0, c3_high - c2_close) / pattern_range if c3_high > c2_close else 0.0
            c3_spike_recovery = 0.0
            if c3_spike > 0.01:
                c3_spike_recovery = min(1.0, (c3_high - c3_close) / max(c3_high - c2_close, 1e-10))

            if c3_confirms:
                strength = min(1.0, recovery + 0.3 * c3_spike_recovery)
                result[i, 0] = 2.0  # bearish reversal
                result[i, 1] = strength
                result[i, 4] = min(1.0, recovery)
                result[i, 5] = 1.0  # confirmed
                result[i, 6] = c3_spike_recovery
                result[i, 7] = min(1.0, 0.4 * recovery + 0.3 * 1.0 + 0.3 * c3_spike_recovery)
            elif c3_close < c1_close:
                result[i, 0] = 0.0
                result[i, 4] = min(1.0, recovery)
                result[i, 5] = 0.0

    return result


def detect_trend_direction(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    lookback: int = 10,
) -> np.ndarray:
    """
    Determine trend direction from daily (or any TF) price data.
    Uses a combination of:
      1. Linear regression slope of closes over `lookback` bars
      2. Higher-highs / lower-lows count (structure-based)

    Returns (n, 2) float32 array per bar:
      [0] trend_direction: -1.0 (bearish) to +1.0 (bullish), 0.0 = neutral
      [1] trend_strength:  0.0 (no trend) to 1.0 (strong trend)

    No look-ahead: each value uses only bars[0..i].
    Values at index < lookback are set to (0.0, 0.0).
    """
    n = len(closes)
    result = np.zeros((n, 2), dtype=np.float32)

    for i in range(lookback, n):
        window_close = closes[i - lookback + 1 : i + 1]
        window_high  = highs[i - lookback + 1 : i + 1]
        window_low   = lows[i - lookback + 1 : i + 1]
        k = len(window_close)

        # 1. Slope via linear regression of closes
        x = np.arange(k, dtype=np.float64)
        x_mean = x.mean()
        y_mean = window_close.mean()
        slope = np.sum((x - x_mean) * (window_close - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-10)
        # Normalise slope by average price level
        norm_slope = slope / max(abs(y_mean), 1e-10) * k

        # 2. Count higher highs / lower lows
        hh_count, ll_count = 0, 0
        for j in range(1, k):
            if window_high[j] > window_high[j - 1]:
                hh_count += 1
            if window_low[j] < window_low[j - 1]:
                ll_count += 1

        structure_bias = (hh_count - ll_count) / max(k - 1, 1)

        # Combine: 60% slope + 40% structure
        combined = 0.6 * np.clip(norm_slope * 10, -1.0, 1.0) + 0.4 * np.clip(structure_bias, -1.0, 1.0)
        direction = float(np.clip(combined, -1.0, 1.0))
        strength = float(min(1.0, abs(combined)))

        result[i, 0] = direction
        result[i, 1] = strength

    return result
