"""
Data loading module — real historical data first, then graceful fallbacks.

Priority order when calling load_forex_data():
  1. Local Dukascopy CSV  (data/<PAIR>_<TF>.csv)
  2. Local HistData CSV   (data/DAT_ASCII_<PAIR>_M<N>_<YEAR>.csv)
  3. Local MT4/MT5 CSV
  4. Generic auto-detect CSV
  5. yfinance             (limited to ~60 days for 15m; no real tick volume)
  6. Synthetic fallback   (demo / CI only — NOT for model training)

─────────────────────────────────────────────────────────────────
REAL DATA SOURCES (free):
  • Dukascopy: https://www.dukascopy.com/swiss/english/marketwatch/historic/
    → Download "OHLCV" candles, any pair/timeframe, CSV format
    → Filename example: GBPJPY_Candlestick_15_m_BID_01.01.2020-01.01.2024.csv
    → Column header: Gmt time,Open,High,Low,Close,Volume

  • HistData.com: https://www.histdata.com/download-free-forex-data/
    → Download "ASCII" format, any M1/M5/M15/M30/H1/H4/D1
    → Filename example: DAT_ASCII_GBPJPY_M15_2023.csv
    → Format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume (semicolon-separated)

  • MetaTrader 4/5: use "Export" in History Centre
    → Tab-separated, header: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<VOL>

To use a local file, place it in a 'data/' folder next to wavetrader.py:
  data/GBPJPY_15m.csv      ← will be auto-detected for pair="GBP/JPY" tf="15min"
─────────────────────────────────────────────────────────────────

Synthetic data quality note
────────────────────────────
The synthetic generator uses a GBM + GARCH-style volatility model with
Student-t noise to approximate fat tails.  Still not suitable for training
a production model — use it only for smoke-tests and architecture validation.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV = ["date", "open", "high", "low", "close", "volume"]

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, sort by date, reset index."""
    df = df.copy()
    # Handle timestamp-indexed parquet (processed_data format)
    if df.index.name in ("timestamp", "date", "datetime"):
        df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Map mid-price columns from preprocessed parquets
    _mid_map = {"open_mid": "open", "high_mid": "high", "low_mid": "low", "close_mid": "close"}
    df = df.rename(columns={k: v for k, v in _mid_map.items() if k in df.columns and v not in df.columns})
    # Find date column
    if "date" not in df.columns:
        for alias in ("timestamp", "datetime", "time"):
            if alias in df.columns:
                df = df.rename(columns={alias: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"DataFrame missing 'date' column. Got: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[_OHLCV].dropna()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a date-indexed OHLCV DataFrame to a higher timeframe."""
    df = df.set_index("date")
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna().reset_index()
    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_dukascopy_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a Dukascopy OHLCV candle export.

    Expected header (comma-separated):
        Gmt time,Open,High,Low,Close,Volume
    Date format: 01.01.2020 00:00:00.000
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the timestamp column (may be "gmt time" or "local time")
    time_col = next(
        (c for c in df.columns if "time" in c or "date" in c), None
    )
    if time_col is None:
        raise ValueError("Cannot find timestamp column in Dukascopy CSV.")

    df["date"] = pd.to_datetime(
        df[time_col].str.strip(),
        format="%d.%m.%Y %H:%M:%S.%f",
        errors="coerce",
    )
    df = df.drop(columns=[time_col])
    return _normalise_df(df)


def load_histdata_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a HistData.com ASCII CSV export.

    Format (semicolon-separated, no header):
        20230102 150000;157.498;157.512;157.490;157.502;36
    """
    df = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        names=["_dt", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["_dt"], format="%Y%m%d %H%M%S", errors="coerce")
    df = df.drop(columns=["_dt"])
    return _normalise_df(df)


def load_mt4_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a MetaTrader 4/5 CSV or TXT history export.

    Typical format (tab or comma separated):
        <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <VOL>
    or  2023.01.02,00:00,157.498,157.512,157.490,157.502,36
    """
    # Sniff separator
    sample = Path(filepath).read_text(errors="ignore")[:2000]
    sep = "\t" if "\t" in sample else ","

    df = pd.read_csv(filepath, sep=sep, header=None, skipinitialspace=True)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try to combine first two columns as date+time
    if df.shape[1] >= 7:
        df["date"] = pd.to_datetime(
            df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str),
            errors="coerce",
        )
        df = df.rename(columns={
            df.columns[2]: "open",
            df.columns[3]: "high",
            df.columns[4]: "low",
            df.columns[5]: "close",
            df.columns[6]: "volume",
        })
        df = df[["date", "open", "high", "low", "close", "volume"]]
    else:
        raise ValueError(
            f"MT4 CSV must have ≥7 columns (date, time, O, H, L, C, V). "
            f"Found {df.shape[1]}: {list(df.columns)}"
        )

    return _normalise_df(df)


def load_generic_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Auto-detect delimiter and column names for any OHLCV CSV.
    Tries common variations before giving up.
    """
    path = Path(filepath)
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep, skipinitialspace=True, nrows=5)
            if df.shape[1] >= 5:
                break
        except Exception:
            continue

    df = pd.read_csv(path, sep=sep, skipinitialspace=True)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map common column name aliases
    aliases = {
        "datetime": "date", "timestamp": "date", "time": "date",
        "o": "open",  "h": "high",  "l": "low",  "c": "close",
        "vol": "volume", "tick_volume": "volume", "tickvolume": "volume",
    }
    df = df.rename(columns=aliases)

    if "date" not in df.columns:
        raise ValueError(
            "Cannot identify a date/time column. "
            "Rename it to 'date' or 'datetime'."
        )

    return _normalise_df(df)


# ─────────────────────────────────────────────────────────────────────────────
# Smart loader
# ─────────────────────────────────────────────────────────────────────────────

_TF_TO_INTERVAL = {
    "1min": "1m",  "5min": "5m",  "15min": "15m",
    "30min": "30m", "1h": "1h",   "4h": "4h",    "1d": "1d",
}

_TF_PERIOD = {
    "1m": "7d", "5m": "7d", "15m": "60d", "30m": "60d",
    "1h": "730d", "4h": "730d", "1d": "2000d",
}

_PAIR_TO_YFSYMBOL = {
    "GBP/JPY": "GBPJPY=X",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
}


def _pair_to_filestem(pair: str, timeframe: str) -> str:
    """GBPJPY_15m"""
    return f"{pair.replace('/', '')}_{ timeframe.lower()}"


def load_forex_data(
    pair:       str  = "GBP/JPY",
    timeframe:  str  = "15min",
    days:       int  = 365,
    data_dir:   Union[str, Path] = "data",
) -> pd.DataFrame:
    """
    Smart loader. Priority:
      1. Local file in `data_dir`  (Dukascopy, HistData, MT4, or generic CSV)
      2. yfinance                  (limited; no real tick volume)
      3. Synthetic fallback        (demo / CI only)

    For best results supply real tick-data CSV files.
    See module docstring for download instructions.
    """
    data_dir = Path(data_dir)

    # ── 1. Scan for a local file ──────────────────────────────────────────────
    stem     = _pair_to_filestem(pair, timeframe)        # e.g. "GBPJPY_15min"
    pair_tag = pair.replace("/", "")                     # e.g. "GBPJPY"
    tf_tag   = timeframe.replace("min", "").replace("h", "H").replace("d", "D")
    # Short form used by download_dukascopy.py  ("15m", "1h", "4h", "1d")
    tf_short = timeframe.replace("min", "m")

    candidates = [
        # Parquet files (from download_dukascopy.py) — fastest, checked first
        data_dir / f"{pair_tag}_{tf_short}.parquet",
        data_dir / f"{stem}.parquet",
        *sorted(data_dir.glob(f"{pair_tag}*{tf_short}*.parquet")),
        # CSV formats
        data_dir / f"{stem}.csv",
        *data_dir.glob(f"{pair_tag}*{timeframe}*.csv"),
        *data_dir.glob(f"{pair_tag}*Candlestick*{tf_tag}*.csv"),     # Dukascopy
        *data_dir.glob(f"DAT_ASCII_{pair_tag}_M{tf_tag}*.csv"),       # HistData min
        *data_dir.glob(f"DAT_ASCII_{pair_tag}_H{tf_tag}*.csv"),       # HistData hour
    ]

    for path in candidates:
        if not path.exists():
            continue
        logger.info("Loading local file: %s", path)
        try:
            return _detect_and_load(path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path, exc)

    # ── 2. yfinance fallback ──────────────────────────────────────────────────
    try:
        return _load_yfinance(pair, timeframe, days)
    except Exception as exc:
        logger.warning("yfinance failed: %s", exc)

    # ── 3. Synthetic last resort ──────────────────────────────────────────────
    logger.warning(
        "No real data found for %s %s — using SYNTHETIC data. "
        "Do NOT use this for model training.", pair, timeframe,
    )
    n_bars = days * 96  # ~96 15-min bars per day
    return generate_synthetic_forex(n_bars, pair)


def _detect_and_load(path: Path) -> pd.DataFrame:
    """Try each known format in order."""
    # Parquet: fastest loader, no format sniffing needed
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return _normalise_df(df)

    name = path.name.lower()  # noqa: F841  (kept for future format guards)

    # Dukascopy header signature
    header = path.read_text(errors="ignore")[:200]
    if "gmt time" in header.lower():
        return load_dukascopy_csv(path)

    # HistData: semicolon-delimited, no header, starts with YYYYMMDD
    if re.match(r"^\d{8} \d{6};", header.strip()):
        return load_histdata_csv(path)

    # MT4 signature: date in YYYY.MM.DD format
    if re.search(r"\d{4}\.\d{2}\.\d{2}", header):
        return load_mt4_csv(path)

    return load_generic_csv(path)


def _load_yfinance(pair: str, timeframe: str, days: int) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    symbol   = _PAIR_TO_YFSYMBOL.get(pair, f"{pair.replace('/', '')}=X")
    interval = _TF_TO_INTERVAL.get(timeframe, "15m")
    period   = _TF_PERIOD.get(interval, f"{days}d")

    logger.info("Fetching %s %s from yfinance (period=%s)…", pair, timeframe, period)
    df = yf.Ticker(symbol).history(period=period, interval=interval).reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    elif "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
    logger.info("yfinance returned %d bars", len(df))
    return _normalise_df(df)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_mtf_data(
    pair:      str  = "GBP/JPY",
    timeframes = ("15min", "1h", "4h", "1d"),
    data_dir:  Union[str, Path] = "data",
    days:      int  = 730,
) -> Dict[str, pd.DataFrame]:
    """
    Load each timeframe independently from `data_dir`.
    If only the base timeframe is available, higher TFs are derived by
    resampling (less accurate but workable).
    """
    _TF_RULE = {
        "15min": "15min", "30min": "30min",
        "1h": "1h", "4h": "4h", "1d": "1D",
    }

    result: Dict[str, pd.DataFrame] = {}
    base_df: Optional[pd.DataFrame] = None

    for tf in timeframes:
        try:
            df = load_forex_data(pair=pair, timeframe=tf, data_dir=data_dir, days=days)
            result[tf] = df
            if base_df is None or len(df) > len(base_df):
                base_df = df
        except Exception as exc:
            logger.warning("Could not load %s %s: %s", pair, tf, exc)

    # If some TFs are missing, derive from the finest available TF
    if base_df is not None:
        finest_len = max(len(v) for v in result.values())
        base_df    = max(result.values(), key=len)  # use longest as base

        for tf in timeframes:
            if tf not in result:
                rule = _TF_RULE.get(tf)
                if rule:
                    logger.info("Resampling %s → %s for %s", base_df.shape, rule, tf)
                    result[tf] = _resample_ohlcv(base_df, rule)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline  (run before building datasets)
# ─────────────────────────────────────────────────────────────────────────────

_PIP_VALUE: Dict[str, float] = {
    "GBP/JPY": 0.01,
    "USD/JPY": 0.01,
    "EUR/JPY": 0.01,
    "GBP/USD": 0.0001,
    "EUR/USD": 0.0001,
    "AUD/USD": 0.0001,
    "USD/CHF": 0.0001,
}

# Maximum high-low BAR RANGE (pips) beyond which the bar is a flash-crash candidate.
# These are 15-minute thresholds; scale up ×2 for 1h, ×4 for 4h.
# Normal GBP/JPY 15m ATR ≈ 30–150 pips; genuine flash crashes > 600 pips.
_MAX_BAR_RANGE_PIPS: Dict[str, float] = {
    "GBP/JPY": 800.0,
    "USD/JPY": 500.0,
    "EUR/JPY": 600.0,
    "GBP/USD": 400.0,
    "EUR/USD": 300.0,
}

_TF_MINUTES: Dict[str, int] = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str = "15min",
    max_gap_minutes: int = 30,
) -> pd.DataFrame:
    """
    Annotate bars preceded by an unexpected intraday gap.

    Adds a boolean column ``gap_before`` that is True when the time delta
    from the previous bar exceeds *max_gap_minutes* AND the gap does NOT
    span the weekend (Fri ~22:00 UTC → Sun ~22:00 UTC).

    Gaps during official market-closed periods (Christmas, New Year's Day,
    Good Friday) are not flagged — they appear as weekend-like spans.

    Returns the input DataFrame with the added column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    bar_minutes = _TF_MINUTES.get(timeframe, 15)
    max_gap     = pd.Timedelta(minutes=max(max_gap_minutes, bar_minutes * 2))
    dates       = df["date"].tolist()
    gap_flags   = [False]  # first row never has a predecessor

    for i in range(1, len(dates)):
        diff = dates[i] - dates[i - 1]
        if diff <= max_gap:
            gap_flags.append(False)
            continue
        # Determine whether the gap straddles a Saturday or Sunday
        mid_point  = dates[i - 1] + diff / 2
        is_weekend = mid_point.weekday() in (5, 6)  # Sat=5, Sun=6
        gap_flags.append(not is_weekend)

    df["gap_before"] = gap_flags
    n_gaps = sum(gap_flags)
    if n_gaps > 0:
        logger.warning(
            "detect_gaps [%s %s]: %d intraday gaps > %dm found "
            "(DST transitions or feed outages — verify before training)",
            timeframe, df["date"].iloc[0].date() if len(df) else "?",
            n_gaps, max_gap_minutes,
        )
    return df


def filter_flash_crashes(
    df: pd.DataFrame,
    pair: str = "GBP/JPY",
    max_bar_move_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Remove bars that are untradeable data artefacts or flash-crash spikes.

    Filters applied (any one removes the bar):
      1. |close − open| / open  >  *max_bar_move_pct* %
         Catches: 2016-10-07 GBP flash crash (−6%), 2019-01-03 JPY flash crash (−4%).
      2. (high − low) / pip_size  >  max_bar_range_pips
         Catches bars with an implausibly wide 15-minute price range.
         NOTE: (high − low) is the BAR RANGE, not the bid-ask spread. Defaults
         in _MAX_BAR_RANGE_PIPS are calibrated for 15-minute OHLCV bars:
         normal GBP/JPY 15m ATR ≈ 30–150 pips; threshold = 800 pips.
      3. volume == 0  AND  |close − open| > 0
         Catches ghost ticks present in some broker/historical feeds.

    Returns a clean copy with reset index.
    """
    df  = df.copy()
    pre = len(df)

    pip           = _PIP_VALUE.get(pair, 0.0001)
    max_bar_range = _MAX_BAR_RANGE_PIPS.get(pair, 800.0)

    bar_move   = (df["close"] - df["open"]).abs() / df["open"].abs()
    bar_range  = (df["high"] - df["low"]) / pip
    ghost_tick = (df["volume"] == 0) & ((df["close"] - df["open"]).abs() > 0)

    bad = (
        (bar_move  > max_bar_move_pct / 100.0)
        | (bar_range > max_bar_range)
        | ghost_tick
    )
    df = df[~bad].reset_index(drop=True)

    removed = pre - len(df)
    if removed > 0:
        logger.info(
            "filter_flash_crashes [%s]: removed %d/%d bars "
            "(bar-move>%.1f%% | range>%.0f pips | ghost-tick)",
            pair, removed, pre, max_bar_move_pct, max_bar_range,
        )
    return df


def verify_session_alignment(
    dataframes: Dict[str, "pd.DataFrame"],
    timeframe: str = "15min",
    tolerance_minutes: int = 2,
) -> bool:
    """
    Check that all pairs share the same 15-minute timestamp grid.

    Cross-pair attention learns the equation
        GBP/JPY ≈ f(USD/JPY, EUR/JPY, GBP/USD) + noise.
    If GBP/JPY 14:00 corresponds to USD/JPY 14:03 due to different broker
    feeds or DST handling, the model learns feed-specific drift instead.

    Args:
        dataframes         Dict of {pair → DataFrame}.  Each DataFrame must
                           have a 'date' column.
        timeframe          Which timeframe to compare (default '15min').
        tolerance_minutes  Timestamps within this tolerance are considered
                           aligned (default 2 minutes).

    Returns:
        True if all pairs are aligned, False otherwise (with warning logged).
    """
    if len(dataframes) < 2:
        return True

    pairs   = list(dataframes.keys())
    ref_key = pairs[0]
    ref_ts  = set(
        pd.to_datetime(dataframes[ref_key]["date"]).dt.floor(f"{tolerance_minutes}min")
    )

    all_ok = True
    for key in pairs[1:]:
        other_ts = set(
            pd.to_datetime(dataframes[key]["date"]).dt.floor(f"{tolerance_minutes}min")
        )
        missing = ref_ts - other_ts
        extra   = other_ts - ref_ts
        if missing or extra:
            logger.warning(
                "verify_session_alignment: '%s' vs '%s' at %s — "
                "%d timestamps only in %s, %d only in %s.  "
                "Fix DST drift before training.",
                ref_key, key, timeframe,
                len(missing), ref_key,
                len(extra),   key,
            )
            all_ok = False
        else:
            logger.debug("Session alignment OK: %s ↔ %s", ref_key, key)

    if all_ok:
        logger.info(
            "verify_session_alignment: all %d pairs aligned at %s.",
            len(dataframes), timeframe,
        )
    return all_ok


def preprocess_pipeline(
    df: pd.DataFrame,
    pair: str = "GBP/JPY",
    timeframe: str = "15min",
    max_bar_move_pct: float = 5.0,
    max_gap_minutes: int = 30,
) -> pd.DataFrame:
    """
    Full data-quality pipeline.  Apply to every DataFrame before training:

      1. filter_flash_crashes  — remove untradeable artefact bars
      2. detect_gaps           — annotate (but keep) intraday gap bars;
                                 gaps are logged so the user can decide
                                 whether to drop or forward-fill them

    Returns a clean DataFrame.  The ``gap_before`` column is included so
    downstream code can optionally drop bars that follow a feed gap.

    Example::

        df = load_forex_data("GBP/JPY", "15min", data_dir="data/")
        df = preprocess_pipeline(df, pair="GBP/JPY", timeframe="15min")
        # Optionally drop bars that follow a data gap:
        # df = df[~df["gap_before"]].drop(columns=["gap_before"])
    """
    df = filter_flash_crashes(df, pair=pair, max_bar_move_pct=max_bar_move_pct)
    df = detect_gaps(df, timeframe=timeframe, max_gap_minutes=max_gap_minutes)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data  (demo / architecture smoke-tests only)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_forex(n_bars: int = 10_000, pair: str = "GBP/JPY") -> pd.DataFrame:
    """
    GBM + GARCH-inspired vol clustering + Student-t shocks.

    Compared to plain Gaussian noise this better mimics:
      • fat-tailed return distributions
      • volatility clustering (calm → calm, spike → spike)
      • intraday mean-reverting drift

    Still NOT suitable for production training — use real tick data.
    """
    rng = np.random.default_rng(42)
    base_price = 190.0 if "JPY" in pair else 1.25

    # GARCH(1,1)-like vol process
    omega, alpha, beta = 1e-6, 0.08, 0.90
    vol = np.zeros(n_bars)
    vol[0] = 5e-4
    shocks = rng.standard_t(df=5, size=n_bars) * vol[0]
    for i in range(1, n_bars):
        vol[i] = np.sqrt(omega + alpha * shocks[i - 1] ** 2 + beta * vol[i - 1] ** 2)
        shocks[i] = rng.standard_t(df=5) * vol[i]

    returns = shocks

    prices = [base_price]
    for r in returns[1:]:
        mean_rev = (base_price - prices[-1]) * 0.0005
        prices.append(max(prices[-1] * (1.0 + r + mean_rev), base_price * 0.5))
    prices = np.array(prices)

    spread = vol * prices * 0.5 + 0.02
    high   = prices + rng.uniform(0.1, 1.0, n_bars) * spread
    low    = prices - rng.uniform(0.1, 1.0, n_bars) * spread
    open_  = np.concatenate([[prices[0]], prices[:-1]])

    df = pd.DataFrame({
        "open":   open_,
        "high":   np.maximum.reduce([open_, high, prices]),
        "low":    np.minimum.reduce([open_, low, prices]),
        "close":  prices,
    })
    # Tick volume: higher when volatility is high
    df["volume"] = (rng.lognormal(10, 0.5, n_bars) * (1 + vol / vol.mean())).astype(int)

    start = datetime.utcnow() - timedelta(minutes=15 * n_bars)
    df["date"] = pd.date_range(start=start, periods=n_bars, freq="15min")

    return df[_OHLCV]


def generate_synthetic_mtf_data(
    base_bars: int = 35_000, pair: str = "GBP/JPY"
) -> Dict[str, pd.DataFrame]:
    """Derive multi-TF data by resampling synthetic 15-min base."""
    df_15m = generate_synthetic_forex(base_bars, pair)
    result = {
        "15min": df_15m,
        "1h":    _resample_ohlcv(df_15m, "1h"),
        "4h":    _resample_ohlcv(df_15m, "4h"),
        "1d":    _resample_ohlcv(df_15m, "1D"),
    }
    for tf, df in result.items():
        logger.info("  %s: %d bars", tf, len(df))
    return result
