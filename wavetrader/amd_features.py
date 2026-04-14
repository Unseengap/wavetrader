"""
AMD Scalper feature engineering — Asian range, London sweep, engulfing patterns,
Fair Value Gaps, and ORB features.

All functions are pure NumPy, no look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Session boundary helpers (UTC)
# ─────────────────────────────────────────────────────────────────────────────

ASIAN_START_HOUR = 0    # 00:00 UTC (Tokyo open)
ASIAN_END_HOUR = 9      # 09:00 UTC (London pre-open)
LONDON_START_HOUR = 7   # 07:00 UTC
LONDON_END_HOUR = 13    # 13:00 UTC (NY open)
NY_START_HOUR = 13      # 13:00 UTC
NY_END_HOUR = 22        # 22:00 UTC
NY_ORB_START_HOUR = 13  # 9:30 AM ET ≈ 13:30 UTC (approximation)
NY_ORB_END_HOUR = 14    # first 15 min ends ~13:45 UTC


def _get_trading_date(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Assign each bar a trading date (Asian session starts the day)."""
    # Bars before ASIAN_END belong to "today", bars after belong to "today"
    return timestamps.normalize()


# ─────────────────────────────────────────────────────────────────────────────
# Asian Range Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_asian_range(
    timestamps: pd.DatetimeIndex,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Per-bar Asian session range features (no look-ahead).

    For each bar, once the Asian session for that day is complete (hour >= ASIAN_END),
    we know the Asian high/low. Before that, we use the running Asian high/low.

    Returns (n, 5) float32:
      [0] asian_high  (normalised relative to current close)
      [1] asian_low   (normalised relative to current close)
      [2] asian_range  (high - low, normalised by close)
      [3] price_vs_asian_mid  (current price relative to Asian midpoint, -1 to 1)
      [4] asian_range_valid   (1.0 if Asian range is established, 0.0 otherwise)
    """
    n = len(timestamps)
    result = np.zeros((n, 5), dtype=np.float32)

    hours = timestamps.hour.to_numpy()
    dates = timestamps.date

    current_asian_high = -np.inf
    current_asian_low = np.inf
    current_date = None
    asian_established = False

    for i in range(n):
        bar_date = dates[i]
        bar_hour = hours[i]

        # New trading day — reset
        if bar_date != current_date:
            current_date = bar_date
            current_asian_high = -np.inf
            current_asian_low = np.inf
            asian_established = False

        # During Asian session, track running high/low
        if bar_hour < ASIAN_END_HOUR:
            current_asian_high = max(current_asian_high, highs[i])
            current_asian_low = min(current_asian_low, lows[i])

        # After Asian session ends, range is established
        if bar_hour >= ASIAN_END_HOUR and current_asian_high > -np.inf:
            asian_established = True

        if asian_established and closes[i] > 0:
            c = closes[i]
            ah = current_asian_high
            al = current_asian_low
            amid = (ah + al) / 2.0
            arange = ah - al

            result[i, 0] = (ah - c) / c       # asian_high relative
            result[i, 1] = (al - c) / c       # asian_low relative
            result[i, 2] = arange / c          # asian_range normalised
            result[i, 3] = np.clip((c - amid) / max(arange / 2.0, 1e-8), -1.0, 1.0)
            result[i, 4] = 1.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# London Sweep Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_london_sweep(
    timestamps: pd.DatetimeIndex,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Detects London session sweep of the Asian range (no look-ahead).

    A sweep occurs when London price pushes above/below the Asian high/low.

    Returns (n, 4) float32:
      [0] sweep_direction  (-1 = swept low / bullish reversal expected,
                             +1 = swept high / bearish reversal expected,
                              0 = no sweep)
      [1] sweep_magnitude  (how far past the Asian extreme, normalised)
      [2] sweep_aggression (velocity — magnitude / bars since London open)
      [3] sweep_valid      (1.0 if a clear sweep has occurred, 0.0 otherwise)
    """
    n = len(timestamps)
    result = np.zeros((n, 4), dtype=np.float32)
    hours = timestamps.hour.to_numpy()
    dates = timestamps.date

    current_asian_high = -np.inf
    current_asian_low = np.inf
    current_date = None
    swept_high = False
    swept_low = False
    sweep_dir = 0.0
    sweep_mag = 0.0
    london_bars = 0

    for i in range(n):
        bar_date = dates[i]
        bar_hour = hours[i]

        if bar_date != current_date:
            current_date = bar_date
            current_asian_high = -np.inf
            current_asian_low = np.inf
            swept_high = False
            swept_low = False
            sweep_dir = 0.0
            sweep_mag = 0.0
            london_bars = 0

        # Track Asian range
        if bar_hour < ASIAN_END_HOUR:
            current_asian_high = max(current_asian_high, highs[i])
            current_asian_low = min(current_asian_low, lows[i])

        # London session — detect sweep
        if LONDON_START_HOUR <= bar_hour < LONDON_END_HOUR and current_asian_high > -np.inf:
            london_bars += 1
            arange = max(current_asian_high - current_asian_low, 1e-8)

            if highs[i] > current_asian_high and not swept_high:
                swept_high = True
                sweep_dir = 1.0   # swept high → expect bearish reversal
                sweep_mag = (highs[i] - current_asian_high) / arange

            if lows[i] < current_asian_low and not swept_low:
                swept_low = True
                sweep_dir = -1.0  # swept low → expect bullish reversal
                sweep_mag = (current_asian_low - lows[i]) / arange

            # If both swept, take the most recent / strongest
            if swept_high and swept_low:
                high_mag = (highs[i] - current_asian_high) / arange if highs[i] > current_asian_high else 0
                low_mag = (current_asian_low - lows[i]) / arange if lows[i] < current_asian_low else 0
                if low_mag > high_mag:
                    sweep_dir = -1.0
                    sweep_mag = low_mag
                else:
                    sweep_dir = 1.0
                    sweep_mag = high_mag

        # Propagate sweep info to all bars after detection
        if swept_high or swept_low:
            result[i, 0] = sweep_dir
            result[i, 1] = min(sweep_mag, 3.0) / 3.0  # normalise to [0, 1]
            result[i, 2] = min(sweep_mag / max(london_bars, 1), 1.0)  # aggression
            result[i, 3] = 1.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Engulfing Pattern Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_engulfing_patterns(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Detect bullish and bearish engulfing candle patterns.

    Bullish engulfing: green candle body fully overlaps preceding red candle body.
    Bearish engulfing: red candle body fully overlaps preceding green candle body.

    Returns (n, 3) float32:
      [0] engulfing_type     (-1 = bearish, 0 = none, +1 = bullish)
      [1] engulfing_strength (body overlap ratio, 0-1)
      [2] engulfing_volume_confirm (placeholder, set externally if volume available)
    """
    n = len(opens)
    result = np.zeros((n, 3), dtype=np.float32)

    for i in range(1, n):
        prev_open, prev_close = opens[i - 1], closes[i - 1]
        curr_open, curr_close = opens[i], closes[i]

        prev_body_top = max(prev_open, prev_close)
        prev_body_bot = min(prev_open, prev_close)
        curr_body_top = max(curr_open, curr_close)
        curr_body_bot = min(curr_open, curr_close)

        prev_body = prev_body_top - prev_body_bot
        curr_body = curr_body_top - curr_body_bot

        if prev_body < 1e-8 or curr_body < 1e-8:
            continue

        # Bullish engulfing: prev is red, curr is green, curr body engulfs prev
        prev_red = prev_close < prev_open
        curr_green = curr_close > curr_open

        if prev_red and curr_green:
            if curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top:
                strength = min(curr_body / prev_body, 3.0) / 3.0
                result[i, 0] = 1.0
                result[i, 1] = strength

        # Bearish engulfing: prev is green, curr is red, curr body engulfs prev
        prev_green = prev_close > prev_open
        curr_red = curr_close < curr_open

        if prev_green and curr_red:
            if curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top:
                strength = min(curr_body / prev_body, 3.0) / 3.0
                result[i, 0] = -1.0
                result[i, 1] = strength

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fair Value Gap (FVG) Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_fair_value_gaps(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Detect Fair Value Gaps (imbalance zones from 3-candle formations).

    Bullish FVG: candle[i-2].high < candle[i].low (gap between candle 1 high and candle 3 low)
    Bearish FVG: candle[i-2].low > candle[i].high

    Returns (n, 4) float32:
      [0] fvg_type       (-1 = bearish, 0 = none, +1 = bullish)
      [1] fvg_size        (gap size normalised by close)
      [2] fvg_midpoint    (gap midpoint relative to close)
      [3] fvg_filled      (1 if price has returned to fill the gap, 0 otherwise)
    """
    n = len(highs)
    result = np.zeros((n, 4), dtype=np.float32)

    # Track active FVGs for fill detection
    active_bullish_fvgs = []  # (fvg_low, fvg_high)
    active_bearish_fvgs = []

    for i in range(2, n):
        c = closes[i] if closes[i] > 0 else 1.0

        # Bullish FVG: gap up (candle 1 high < candle 3 low)
        if highs[i - 2] < lows[i]:
            gap_low = highs[i - 2]
            gap_high = lows[i]
            gap_size = gap_high - gap_low
            result[i, 0] = 1.0
            result[i, 1] = min(gap_size / c, 0.01) / 0.01  # normalise
            result[i, 2] = ((gap_low + gap_high) / 2 - c) / c
            active_bullish_fvgs.append((gap_low, gap_high))

        # Bearish FVG: gap down (candle 1 low > candle 3 high)
        elif lows[i - 2] > highs[i]:
            gap_low = highs[i]
            gap_high = lows[i - 2]
            gap_size = gap_high - gap_low
            result[i, 0] = -1.0
            result[i, 1] = min(gap_size / c, 0.01) / 0.01
            result[i, 2] = ((gap_low + gap_high) / 2 - c) / c
            active_bearish_fvgs.append((gap_low, gap_high))

        # Check if price has filled any active FVGs
        filled_count = 0
        new_bullish = []
        for fvg_low, fvg_high in active_bullish_fvgs:
            if lows[i] <= fvg_low:
                filled_count += 1
            else:
                new_bullish.append((fvg_low, fvg_high))
        active_bullish_fvgs = new_bullish[-20:]  # keep last 20

        new_bearish = []
        for fvg_low, fvg_high in active_bearish_fvgs:
            if highs[i] >= fvg_high:
                filled_count += 1
            else:
                new_bearish.append((fvg_low, fvg_high))
        active_bearish_fvgs = new_bearish[-20:]

        if filled_count > 0:
            result[i, 3] = 1.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Support & Resistance Zone Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_sr_zones(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 50,
    tolerance_pct: float = 0.002,
) -> np.ndarray:
    """
    Detect horizontal zones that have acted as both support and resistance (flip zones).

    Returns (n, 3) float32:
      [0] sr_proximity   (distance to nearest S/R zone, normalised, 0 = at zone)
      [1] sr_strength    (how many times the zone has been tested, normalised)
      [2] sr_is_flip     (1.0 if zone acted as both support and resistance)
    """
    n = len(highs)
    result = np.zeros((n, 3), dtype=np.float32)

    for i in range(lookback, n):
        c = closes[i]
        if c <= 0:
            continue

        # Find local swing highs and lows in lookback window
        window_highs = highs[i - lookback:i]
        window_lows = lows[i - lookback:i]
        window_closes = closes[i - lookback:i]

        # Cluster swing levels within tolerance
        levels = []
        for j in range(2, len(window_highs) - 1):
            # Swing high
            if window_highs[j] > window_highs[j - 1] and window_highs[j] > window_highs[j + 1]:
                levels.append(("R", window_highs[j]))
            # Swing low
            if window_lows[j] < window_lows[j - 1] and window_lows[j] < window_lows[j + 1]:
                levels.append(("S", window_lows[j]))

        if not levels:
            continue

        # Cluster levels within tolerance
        price_levels = sorted(set(lv[1] for lv in levels))
        clusters = []
        for pl in price_levels:
            merged = False
            for cluster in clusters:
                if abs(pl - cluster["price"]) / c < tolerance_pct:
                    cluster["count"] += 1
                    cluster["price"] = (cluster["price"] + pl) / 2
                    # Track if it was S and R
                    for lt, lp in levels:
                        if abs(lp - pl) / c < tolerance_pct:
                            cluster["types"].add(lt)
                    merged = True
                    break
            if not merged:
                types = set()
                for lt, lp in levels:
                    if abs(lp - pl) / c < tolerance_pct:
                        types.add(lt)
                clusters.append({"price": pl, "count": 1, "types": types})

        if not clusters:
            continue

        # Find nearest cluster to current price
        nearest = min(clusters, key=lambda cl: abs(cl["price"] - c))
        dist = abs(nearest["price"] - c) / c
        result[i, 0] = 1.0 - min(dist / 0.01, 1.0)  # 0 = far, 1 = at zone
        result[i, 1] = min(nearest["count"] / 5.0, 1.0)  # strength
        result[i, 2] = 1.0 if len(nearest["types"]) >= 2 else 0.0  # flip zone

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Open Range Breakout (ORB) Features
# ─────────────────────────────────────────────────────────────────────────────

def compute_orb_features(
    timestamps: pd.DatetimeIndex,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Compute ORB (Open Range Breakout) features for the first 15 minutes of NY session.

    Returns (n, 4) float32:
      [0] orb_high_dist   (current price distance from ORB high, normalised)
      [1] orb_low_dist    (current price distance from ORB low, normalised)
      [2] orb_breakout    (+1 = broke above, -1 = broke below, 0 = within range)
      [3] orb_valid       (1 if ORB range is established for this day)
    """
    n = len(timestamps)
    result = np.zeros((n, 4), dtype=np.float32)
    hours = timestamps.hour.to_numpy()
    minutes = timestamps.minute.to_numpy()
    dates = timestamps.date

    current_date = None
    orb_high = -np.inf
    orb_low = np.inf
    orb_established = False

    for i in range(n):
        bar_date = dates[i]
        bar_hour = hours[i]
        bar_min = minutes[i]

        if bar_date != current_date:
            current_date = bar_date
            orb_high = -np.inf
            orb_low = np.inf
            orb_established = False

        # During ORB window (first 15 min of NY session: ~13:30-13:45 UTC)
        # Approximation: hours 13-14 on 5min bars gives us the opening range bars
        if bar_hour == NY_ORB_START_HOUR and bar_min < 45:
            orb_high = max(orb_high, highs[i])
            orb_low = min(orb_low, lows[i])
        elif bar_hour >= NY_ORB_START_HOUR and orb_high > -np.inf:
            orb_established = True

        if orb_established and closes[i] > 0:
            c = closes[i]
            orb_range = max(orb_high - orb_low, 1e-8)

            result[i, 0] = np.clip((c - orb_high) / orb_range, -3.0, 3.0) / 3.0
            result[i, 1] = np.clip((c - orb_low) / orb_range, -3.0, 3.0) / 3.0
            result[i, 3] = 1.0

            # Breakout detection (close above/below ORB)
            if closes[i] > orb_high:
                result[i, 2] = 1.0
            elif closes[i] < orb_low:
                result[i, 2] = -1.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# AMD Phase Labels (for supervised training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_amd_phase_labels(
    timestamps: pd.DatetimeIndex,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    asian_range_max_pips: float = 80.0,
    london_sweep_min_pips: float = 15.0,
    pip_size: float = 0.01,
) -> np.ndarray:
    """
    Assign AMD phase labels to each bar for supervised training.

    Labels:
      0 = ACCUMULATION (Asian session — tight range building)
      1 = MANIPULATION (London session — sweeping Asian range)
      2 = DISTRIBUTION (NY session — reversal entry zone)
      3 = INVALID (filters failed — skip this bar for signal training)

    Returns (n,) int64 array.
    """
    n = len(timestamps)
    labels = np.full(n, 3, dtype=np.int64)  # default INVALID
    hours = timestamps.hour.to_numpy()
    dates = timestamps.date

    current_date = None
    asian_high = -np.inf
    asian_low = np.inf
    day_valid = False
    london_swept = False

    for i in range(n):
        bar_date = dates[i]
        bar_hour = hours[i]

        if bar_date != current_date:
            current_date = bar_date
            asian_high = -np.inf
            asian_low = np.inf
            day_valid = True
            london_swept = False

        # Asian session
        if bar_hour < ASIAN_END_HOUR:
            asian_high = max(asian_high, highs[i])
            asian_low = min(asian_low, lows[i])
            asian_range_pips = (asian_high - asian_low) / pip_size

            if asian_range_pips <= asian_range_max_pips:
                labels[i] = 0  # ACCUMULATION
            else:
                day_valid = False
                labels[i] = 3  # INVALID — range too wide

        # London session
        elif LONDON_START_HOUR <= bar_hour < LONDON_END_HOUR:
            if not day_valid:
                labels[i] = 3
                continue

            # Check for sweep
            if asian_high > -np.inf:
                high_sweep = (highs[i] - asian_high) / pip_size
                low_sweep = (asian_low - lows[i]) / pip_size
                if high_sweep >= london_sweep_min_pips or low_sweep >= london_sweep_min_pips:
                    london_swept = True
                    labels[i] = 1  # MANIPULATION
                else:
                    labels[i] = 1  # Still London, but weak manipulation

        # NY session
        elif NY_START_HOUR <= bar_hour < NY_END_HOUR:
            if day_valid and london_swept:
                labels[i] = 2  # DISTRIBUTION
            else:
                labels[i] = 3  # INVALID — setup didn't form

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Combined AMD feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_amd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all AMD-specific features to a prepared DataFrame.
    Expects columns: open, high, low, close, volume, date
    (plus standard features from prepare_features already applied).

    Returns the DataFrame with new AMD feature columns added.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    timestamps = pd.DatetimeIndex(df["date"]) if "date" in df.columns else None
    if timestamps is None:
        raise ValueError("AMD features require 'date' column with timestamps")

    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values

    # Asian range
    asian = compute_asian_range(timestamps, h, lo, c)
    for i, name in enumerate(["asian_high_rel", "asian_low_rel", "asian_range_norm",
                               "price_vs_asian_mid", "asian_valid"]):
        df[name] = asian[:, i]

    # London sweep
    sweep = compute_london_sweep(timestamps, h, lo, c)
    for i, name in enumerate(["sweep_direction", "sweep_magnitude",
                               "sweep_aggression", "sweep_valid"]):
        df[name] = sweep[:, i]

    # Engulfing patterns
    engulf = compute_engulfing_patterns(o, h, lo, c)
    for i, name in enumerate(["engulfing_type", "engulfing_strength",
                               "engulfing_vol_confirm"]):
        df[name] = engulf[:, i]

    # Fair Value Gaps
    fvg = compute_fair_value_gaps(h, lo, c)
    for i, name in enumerate(["fvg_type", "fvg_size", "fvg_midpoint", "fvg_filled"]):
        df[name] = fvg[:, i]

    # S&R zones
    sr = compute_sr_zones(h, lo, c)
    for i, name in enumerate(["sr_proximity", "sr_strength", "sr_is_flip"]):
        df[name] = sr[:, i]

    # ORB features
    orb = compute_orb_features(timestamps, h, lo, c)
    for i, name in enumerate(["orb_high_dist", "orb_low_dist", "orb_breakout", "orb_valid"]):
        df[name] = orb[:, i]

    # AMD phase labels
    labels = compute_amd_phase_labels(timestamps, h, lo, c)
    df["amd_phase"] = labels

    return df
