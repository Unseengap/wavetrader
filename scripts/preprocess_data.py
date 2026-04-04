#!/usr/bin/env python3
"""
JForex Data Preprocessing Pipeline
====================================
Processes raw JForex CSV files (4 pairs × 4 TFs × Bid/Ask) into
training-ready Parquet files under processed_data/.

Data format expected
--------------------
  Columns : Time (EET),Open,High,Low,Close,Volume
  Timestamp: dd.MM.yyyy HH:mm:ss  (EET = Europe/Athens, UTC+2/+3 DST)

Output structure
----------------
  processed_data/
  ├── train/  PAIR_15m.parquet  PAIR_1h.parquet  PAIR_4h.parquet  PAIR_1d.parquet
  ├── val/
  └── test/

Run
---
  cd /path/to/phase_lm
  python scripts/preprocess_data.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "processed_data"

# Add project root so we can import wavetrader
sys.path.insert(0, str(ROOT_DIR))

# ── Constants ─────────────────────────────────────────────────────────────────
PAIRS = ["GBPJPY", "EURJPY", "GBPUSD"]  # USDJPY temporarily removed due to missing 2023-2026 data

# JForex directory name fragment for each timeframe
TF_DIR_NAME: Dict[str, str] = {
    "1m": "1 Min",
    "1h": "Hourly",
    "4h": "4 Hours",
    "1d": "Daily",
}

# How long each higher-TF bar lasts (used for no-lookahead index shift)
TF_DURATION: Dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "1d": "24h",   # daily bars open at 22:00 UTC; shift 24h to 22:00 next day
}

# Pip size (1 pip in price units)
PIP_SIZE: Dict[str, float] = {
    "GBPJPY": 0.01,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPUSD": 0.0001,
}

# Maximum allowed spread in pips (illiquidity filter)
SPREAD_THRESHOLD_PIPS: Dict[str, float] = {
    "GBPJPY": 20.0,
    "USDJPY": 20.0,
    "EURJPY": 20.0,
    "GBPUSD":  5.0,
}

# Walk-forward split boundaries (compare against UTC DatetimeIndex)
TRAIN_END   = pd.Timestamp("2022-12-14 23:59:00")   # purge starts 2022-12-15
PURGE_START = pd.Timestamp("2022-12-15 00:00:00")
VAL_START   = pd.Timestamp("2023-01-01 00:00:00")
VAL_END     = pd.Timestamp("2024-06-30 23:59:00")
TEST_START  = pd.Timestamp("2024-07-01 00:00:00")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── File discovery ─────────────────────────────────────────────────────────────

def _find_csv(pair: str, tf_key: str, side: str) -> Path:
    """
    Locate the CSV file for (pair, timeframe, side='Bid'|'Ask').
    Files ending in '(1).csv' are exact duplicates — skip them.
    """
    fragment = TF_DIR_NAME[tf_key]
    candidates = [
        f for f in DATA_DIR.glob(f"{pair}_{fragment}_{side}_*.csv")
        if "(1)" not in f.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV for pair={pair} tf={tf_key} side={side} "
            f"(searched {DATA_DIR} with pattern '{pair}_{fragment}_{side}_*.csv')"
        )
    return sorted(candidates)[0]


# ── CSV loading ───────────────────────────────────────────────────────────────

def _load_jforex_csv(path: Path) -> pd.DataFrame:
    """
    Read a JForex OHLCV CSV and return a DataFrame indexed by UTC timestamp.

    JForex exports timestamps in EET (Europe/Athens: UTC+2 standard / UTC+3 DST).
    We convert to UTC and strip timezone info so the index is tz-naive UTC.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # Parse EET timestamp
    raw_ts = pd.to_datetime(df["Time (EET)"], format="%d.%m.%Y %H:%M:%S")

    # zoneinfo is stdlib since Python 3.9; fall back to pytz, then fixed offset
    try:
        from zoneinfo import ZoneInfo
        eet = ZoneInfo("Europe/Athens")
        ts_utc = (
            raw_ts.dt.tz_localize(eet, ambiguous="infer", nonexistent="shift_forward")
                  .dt.tz_convert("UTC")
                  .dt.tz_localize(None)
        )
    except ImportError:
        try:
            import pytz
            eet = pytz.timezone("Europe/Athens")
            ts_utc = (
                raw_ts.dt.tz_localize(eet, ambiguous="infer", nonexistent="shift_forward")
                      .dt.tz_convert("UTC")
                      .dt.tz_localize(None)
            )
        except ImportError:
            warnings.warn(
                "Neither zoneinfo nor pytz available — using fixed UTC+2 offset "
                "(DST not corrected). Install Python ≥3.9 or 'pip install pytz'.",
                stacklevel=2,
            )
            ts_utc = raw_ts - pd.Timedelta(hours=2)

    df = df.drop(columns=["Time (EET)"])
    df.columns = [c.strip().lower() for c in df.columns]
    df.insert(0, "timestamp", ts_utc)

    # De-duplicate on timestamp (keep first occurrence)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    return df


# ── Mid-price computation ─────────────────────────────────────────────────────

def _load_mid(pair: str, tf_key: str) -> pd.DataFrame:
    """
    Load bid + ask CSVs, inner-join on timestamp, compute mid-prices and spread.

    Returns DataFrame with columns:
        open_mid, high_mid, low_mid, close_mid, volume, spread_avg
    Index: UTC naive datetime.
    """
    bid = _load_jforex_csv(_find_csv(pair, tf_key, "Bid"))
    ask = _load_jforex_csv(_find_csv(pair, tf_key, "Ask"))

    bid = bid.rename(columns={
        "open": "bid_open", "high": "bid_high",
        "low":  "bid_low",  "close": "bid_close",
        "volume": "volume",
    })
    ask = ask.rename(columns={
        "open": "ask_open", "high": "ask_high",
        "low":  "ask_low",  "close": "ask_close",
        "volume": "ask_volume",
    })

    df = bid.join(ask, how="inner")

    df["open_mid"]  = (df["bid_open"]  + df["ask_open"])  / 2.0
    df["high_mid"]  = (df["bid_high"]  + df["ask_high"])  / 2.0
    df["low_mid"]   = (df["bid_low"]   + df["ask_low"])   / 2.0
    df["close_mid"] = (df["bid_close"] + df["ask_close"]) / 2.0

    # Spread in pips (average of open and close spread for the bar)
    pip = PIP_SIZE[pair]
    open_spread  = (df["ask_open"]  - df["bid_open"])  / pip
    close_spread = (df["ask_close"] - df["bid_close"]) / pip
    df["spread_avg"] = (open_spread + close_spread) / 2.0

    return df[["open_mid", "high_mid", "low_mid", "close_mid", "volume", "spread_avg"]]


# ── 1m → 15m resampling ───────────────────────────────────────────────────────

def _resample_1m_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute mid-price bars to 15-minute bars.
    Bars with all-NaN OHLC (missing minutes) are dropped.
    """
    agg = df.resample("15min").agg(
        open_mid  =("open_mid",  "first"),
        high_mid  =("high_mid",  "max"),
        low_mid   =("low_mid",   "min"),
        close_mid =("close_mid", "last"),
        volume    =("volume",    "sum"),
        spread_avg=("spread_avg","mean"),
    )
    return agg.dropna(subset=["open_mid", "close_mid"])


# ── Quality filters ───────────────────────────────────────────────────────────

def _apply_filters_15m(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Quality filters for 15-minute entry bars:
    1. |close_mid pct-change| > 5%  → flash crash / bad tick
    2. spread_avg > threshold pips  → illiquid bar
    3. NaN in any mid-price column  → incomplete bar
    """
    n0 = len(df)

    pct_chg = df["close_mid"].pct_change().abs()
    df = df[pct_chg <= 0.05]
    n1 = len(df)

    thr = SPREAD_THRESHOLD_PIPS[pair]
    df = df[df["spread_avg"] <= thr]
    n2 = len(df)

    mid_cols = ["open_mid", "high_mid", "low_mid", "close_mid"]
    df = df.dropna(subset=mid_cols)
    n3 = len(df)

    log.info(
        "    [%s 15m] filters: %d → %d (flash crash) → %d (spread>%g pip) → %d (NaN)",
        pair, n0, n1, n2, thr, n3,
    )
    return df


def _clean_htf(df: pd.DataFrame, pair: str, tf_label: str) -> pd.DataFrame:
    """
    Minimal cleanup for higher-TF bars (1h / 4h / 1d).
    The spec says keep these 'as-is'; we only strip obvious bad rows:
    - NaN in mid-price columns
    - Extreme spread outliers (>5× the 15m threshold) — e.g. 100-pip daily spreads
    """
    mid_cols = ["open_mid", "high_mid", "low_mid", "close_mid"]
    n0 = len(df)
    df = df.dropna(subset=mid_cols)
    # Only remove extreme outliers (5× threshold), not routine wide-spread bars
    extreme_thr = SPREAD_THRESHOLD_PIPS[pair] * 5
    df = df[df["spread_avg"] <= extreme_thr]
    log.info(
        "    [%s %s] cleanup: %d → %d (NaN/extreme spread>%g pip)",
        pair, tf_label, n0, len(df), extreme_thr,
    )
    return df


# ── Gap detection ─────────────────────────────────────────────────────────────

def _count_market_hour_gaps(idx: pd.DatetimeIndex, gap_threshold_min: int = 30) -> int:
    """
    Count gaps > gap_threshold_min minutes that occur during forex market hours.

    Forex closed: Friday 22:00 UTC → Sunday 22:00 UTC.
    Weekend gaps (and roll-over periods) are NOT counted.
    """
    ts = pd.Series(idx)
    diffs = ts.diff().dropna()

    big_gap_mask = diffs > pd.Timedelta(minutes=gap_threshold_min)
    if not big_gap_mask.any():
        return 0

    # Evaluate the timestamp of the bar BEFORE each gap
    # (i.e., the bar at index i-1 for gap between i-1 and i)
    gap_positions = diffs.index[big_gap_mask]
    prev_ts = ts.iloc[gap_positions - 1]
    prev_wd = prev_ts.dt.weekday   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    prev_hr = prev_ts.dt.hour

    # Weekend gap: previous bar is on Fri ≥22:00, Sat, or Sun <22:00
    is_weekend = (
        ((prev_wd == 4) & (prev_hr >= 22)) |  # Fri 22:00+
        (prev_wd == 5) |                       # Saturday
        ((prev_wd == 6) & (prev_hr < 22))      # Sun before 22:00
    )
    return int((~is_weekend).sum())


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-calculate features on the 15m DataFrame:
    - log_return     : ln(close_mid[t] / close_mid[t-1])
    - rsi_14         : Wilder RSI(14) on close_mid
    - atr_14         : Wilder ATR(14) using mid OHLC
    - session_tokyo  : 1 if Tokyo session active (00-09 UTC)
    - session_london : 1 if London session active (07-16 UTC)
    - session_newyork: 1 if New York session active (13-22 UTC)
    """
    from wavetrader.indicators import calculate_atr, calculate_rsi, session_features

    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["close_mid"] / df["close_mid"].shift(1))

    # RSI(14)
    df["rsi_14"] = calculate_rsi(df["close_mid"].values, period=14)

    # ATR(14)
    df["atr_14"] = calculate_atr(
        df["high_mid"].values,
        df["low_mid"].values,
        df["close_mid"].values,
        period=14,
    )

    # Session flags (uses UTC hour)
    sess = session_features(df.index)   # (n, 3) float32
    df["session_tokyo"]    = sess[:, 0]
    df["session_london"]   = sess[:, 1]
    df["session_newyork"]  = sess[:, 2]

    return df


# ── No-lookahead multi-TF alignment (informational, not saved separately) ─────

def align_htf_to_15m(
    df_15m: pd.DataFrame,
    df_htf: pd.DataFrame,
    tf_key: str,
) -> pd.DataFrame:
    """
    Return df_htf reindexed to df_15m.index with strict no-lookahead guarantee.

    A higher-TF bar with open-time T covers [T, T+duration).
    It becomes available at T+duration (when it closes).
    We shift the HTF index by +duration, then forward-fill to 15m grid.
    """
    duration = TF_DURATION[tf_key]
    df_avail = df_htf.copy()
    df_avail.index = df_avail.index + pd.Timedelta(duration)

    # Build a combined index covering both grids, ffill, then select 15m only
    combined_idx = df_15m.index.union(df_avail.index).sort_values()
    aligned = df_avail.reindex(combined_idx).ffill().reindex(df_15m.index)
    return aligned


# ── Train / val / test split ──────────────────────────────────────────────────

def _split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Chronological split with purge gap:
      Train : 2015-01-01 … 2022-12-14  (strictly before purge start)
      [Purge: 2022-12-15 … 2022-12-31 — excluded from all splits]
      Val   : 2023-01-01 … 2024-06-30
      Test  : 2024-07-01 … 2026-04-04
    """
    return {
        "train": df[df.index <= TRAIN_END],
        "val":   df[(df.index >= VAL_START) & (df.index <= VAL_END)],
        "test":  df[df.index >= TEST_START],
    }


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_splits(df: pd.DataFrame, pair: str, tf_label: str) -> Dict[str, int]:
    """Split df, save as Parquet, return bar counts per split."""
    counts = {}
    for split_name, subset in _split(df).items():
        out_dir = OUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{pair}_{tf_label}.parquet"
        subset.to_parquet(path, index=True)
        counts[split_name] = len(subset)
        log.info("    → %s/%s_%s.parquet  (%d bars)", split_name, pair, tf_label, len(subset))
    return counts


# ── Per-pair pipeline ─────────────────────────────────────────────────────────

def _process_pair(pair: str) -> Dict:
    """Full pipeline for one currency pair. Returns stats dict."""
    log.info("")
    log.info("=" * 60)
    log.info("  Processing %s", pair)
    log.info("=" * 60)

    stats: Dict = {"pair": pair}

    # ── 1. Load 1m → resample to 15m ──────────────────────────────────────
    log.info("  [1/5] Loading 1m data …")
    df_1m = _load_mid(pair, "1m")
    log.info("        1m bars loaded: %d", len(df_1m))

    log.info("  [2/5] Resampling 1m → 15m …")
    df_15m = _resample_1m_to_15m(df_1m)
    del df_1m
    log.info("        15m raw: %d bars", len(df_15m))

    # ── 2. Quality filters on 15m ─────────────────────────────────────────
    log.info("  [3/5] Applying quality filters …")
    df_15m = _apply_filters_15m(df_15m, pair)
    stats["15m_bars"] = len(df_15m)

    # Gap count
    n_gaps = _count_market_hour_gaps(df_15m.index)
    gap_pct = 100.0 * n_gaps / max(len(df_15m), 1)
    stats["15m_gaps"] = n_gaps
    stats["15m_gap_pct"] = gap_pct
    log.info(
        "        Market-hour gaps >30min: %d  (%.3f%% of bars)%s",
        n_gaps, gap_pct, "  ⚠ EXCEEDS 0.1%" if gap_pct > 0.1 else "",
    )

    # ── 3. Add features to 15m ────────────────────────────────────────────
    log.info("  [4/5] Adding RSI/ATR/session features to 15m …")
    df_15m = _add_features(df_15m)

    # ── 4. Save 15m splits ────────────────────────────────────────────────
    log.info("  [5/5] Saving 15m Parquet files …")
    stats["15m_split_bars"] = _save_splits(df_15m, pair, "15m")

    # ── 5. Higher timeframes ──────────────────────────────────────────────
    for tf_key, tf_label in [("1h", "1h"), ("4h", "4h"), ("1d", "1d")]:
        log.info("  Loading %s …", tf_key)
        df_htf = _load_mid(pair, tf_key)
        log.info("        raw: %d bars", len(df_htf))
        df_htf = _clean_htf(df_htf, pair, tf_label)
        stats[f"{tf_label}_bars"] = len(df_htf)
        _save_splits(df_htf, pair, tf_label)

    return stats


# ── Validation report ─────────────────────────────────────────────────────────

def _validation_report(all_stats: list, close_series: Dict[str, pd.Series]) -> None:
    print()
    print("=" * 70)
    print("  VALIDATION REPORT")
    print("=" * 70)

    # ── Per-pair bar counts and gap stats ──────────────────────────────────
    hdr = f"{'Pair':<10} {'15m':>9} {'1h':>8} {'4h':>8} {'1d':>8}  {'Gaps':>6} {'Gap%':>7}"
    print()
    print(hdr)
    print("-" * 70)
    for s in all_stats:
        p      = s["pair"]
        n15    = s.get("15m_bars", 0)
        n1h    = s.get("1h_bars",  0)
        n4h    = s.get("4h_bars",  0)
        n1d    = s.get("1d_bars",  0)
        gaps   = s.get("15m_gaps", 0)
        gp     = s.get("15m_gap_pct", 0.0)
        flag   = " ⚠" if gp > 0.1 else ""
        print(
            f"{p:<10} {n15:>9,} {n1h:>8,} {n4h:>8,} {n1d:>8,}  "
            f"{gaps:>6}  {gp:>6.3f}%{flag}"
        )
    print()

    # ── Train/val/test bar counts ──────────────────────────────────────────
    print("15m split bar counts:")
    hdr2 = f"{'Pair':<10} {'train':>10} {'val':>10} {'test':>10}"
    print(hdr2)
    print("-" * 45)
    for s in all_stats:
        split_bars = s.get("15m_split_bars", {})
        print(
            f"{s['pair']:<10} "
            f"{split_bars.get('train', 0):>10,} "
            f"{split_bars.get('val',   0):>10,} "
            f"{split_bars.get('test',  0):>10,}"
        )
    print()

    # ── Cross-pair log-return correlation ─────────────────────────────────
    if close_series:
        log_returns: Dict[str, pd.Series] = {
            p: np.log(s / s.shift(1)).dropna()
            for p, s in close_series.items()
        }
        corr = pd.DataFrame(log_returns).corr()
        print("15m close_mid log-return correlation matrix:")
        print(corr.round(4).to_string())
        print()

    # ── NaN / zero-volume check ────────────────────────────────────────────
    print("Integrity checks (15m train split):")
    any_issue = False
    for s in all_stats:
        pair = s["pair"]
        path = OUT_DIR / "train" / f"{pair}_15m.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        mid_cols = ["open_mid", "high_mid", "low_mid", "close_mid"]
        nan_count  = df[mid_cols].isna().sum().sum()
        zero_vol   = (df["volume"] == 0).sum()
        status     = "OK" if (nan_count == 0 and zero_vol == 0) else "ISSUES FOUND"
        if status != "OK":
            any_issue = True
        print(f"  {pair}: NaN in mid-prices={nan_count}, zero-volume bars={zero_vol}  [{status}]")
    if not any_issue:
        print("  All checks passed.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train").mkdir(exist_ok=True)
    (OUT_DIR / "val").mkdir(exist_ok=True)
    (OUT_DIR / "test").mkdir(exist_ok=True)

    all_stats: list = []
    succeeded: list[str] = []

    for pair in PAIRS:
        try:
            stats = _process_pair(pair)
            all_stats.append(stats)
            succeeded.append(pair)
        except FileNotFoundError as exc:
            log.error("SKIP %s: %s", pair, exc)
        except Exception as exc:
            log.exception("ERROR processing %s:", pair)

    if not succeeded:
        log.error("No pairs were processed successfully. Exiting.")
        sys.exit(1)

    # ── Collect close prices for correlation ─────────────────────────────────
    log.info("")
    log.info("Computing cross-pair correlation …")
    close_series: Dict[str, pd.Series] = {}
    for pair in succeeded:
        frames = []
        for split in ("train", "val", "test"):
            p = OUT_DIR / split / f"{pair}_15m.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p, columns=["close_mid"]))
        if frames:
            close_series[pair] = pd.concat(frames)["close_mid"].sort_index()

    _validation_report(all_stats, close_series)

    log.info("Done!  Output written to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
