#!/usr/bin/env python3
"""
Dukascopy Bulk Historical Data Downloader
==========================================
Downloads 1-minute BID candles from Dukascopy's public datafeed CDN,
resamples to 15m / 1h / 4h / 1d, and saves as Parquet files in data/.

Usage
─────
  cd /path/to/phase_lm
  python scripts/download_dukascopy.py \\
      --pairs GBPJPY USDJPY EURJPY GBPUSD \\
      --start 2015-01-01 --end 2024-12-31 \\
      --out data/

  # Resume / cache raw BI5 bytes to avoid re-downloading:
  python scripts/download_dukascopy.py ... --cache data/.bi5_cache

  # Verify one day's data vs web UI (sanity-check field order):
  python scripts/download_dukascopy.py --verify GBPJPY 2022-03-16

Requirements
────────────
  pip install requests pandas pyarrow

Dukascopy datafeed endpoint (no auth required)
───────────────────────────────────────────────
  https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YEAR}/{MM}/{DD}/BID_candles_min_1.bi5
  Note: MM is 0-indexed  (January = 00, December = 11).

BI5 binary format  (after LZMA decompression)
──────────────────────────────────────────────
  Each record = 24 bytes, big-endian:
    uint32   timestamp_ms   ms from midnight UTC of that calendar day
    uint32   open           price × SCALE_FACTOR
    uint32   close          (Dukascopy's own non-standard OCLHV ordering)
    uint32   low
    uint32   high
    float32  volume         tick volume

  Scale factors:
    JPY pairs  (GBPJPY, USDJPY, EURJPY, …)  →  1_000   (3 decimal places)
    Non-JPY    (GBPUSD, EURUSD, AUDUSD, …)  →  100_000 (5 decimal places)

  Field-order verification
  ────────────────────────
  Run:  python scripts/download_dukascopy.py --verify GBPJPY 2022-03-16
  Then compare the printed bars against the Dukascopy web chart for that day.
  If open/close look reversed, swap the _CANDLE_FORMAT field order below.

Instrument codes (Dukascopy naming — no slash)
───────────────────────────────────────────────
  GBPJPY  USDJPY  EURJPY  GBPUSD  EURUSD
  AUDUSD  USDCHF  USDCAD  EURGBP  CADJPY

Output
──────
  data/GBPJPY_1m.parquet
  data/GBPJPY_15m.parquet
  data/GBPJPY_1h.parquet
  data/GBPJPY_4h.parquet
  data/GBPJPY_1d.parquet
  … (one file per pair × timeframe)

  Then run:
    python cli.py --mode preprocess --data data/
    python cli.py --mode train      --data data/
"""
from __future__ import annotations

import argparse
import logging
import lzma
import struct
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Price scale factors per instrument  (prices stored as integer × scale)
_SCALE: Dict[str, int] = {
    # JPY crosses → 3 decimal places
    "GBPJPY": 1_000,
    "USDJPY": 1_000,
    "EURJPY": 1_000,
    "AUDJPY": 1_000,
    "CADJPY": 1_000,
    "CHFJPY": 1_000,
    "NZDJPY": 1_000,
    # Non-JPY → 5 decimal places
    "GBPUSD": 100_000,
    "EURUSD": 100_000,
    "AUDUSD": 100_000,
    "NZDUSD": 100_000,
    "USDCHF": 100_000,
    "USDCAD": 100_000,
    "EURGBP": 100_000,
    "GBPCHF": 100_000,
    "EURAUD": 100_000,
}

# BI5 candle record: 24 bytes, big-endian
# Fields: timestamp_ms, open, close, low, high, volume
# NOTE: Dukascopy uses non-standard OCLHV field ordering (not OHLCV).
# If open/close look swapped vs web UI, flip _O and _C constants below.
_CANDLE_STRUCT = struct.Struct(">IIIIIf")
_TS, _O, _C, _L, _H, _V = 0, 1, 2, 3, 4, 5   # field indices in unpacked tuple

# Pandas resample rules for each output timeframe
_TF_RULES: Dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

_OHLCV = ["date", "open", "high", "low", "close", "volume"]

# Polite download rate  (Dukascopy CDN; no auth but be respectful)
_SLEEP_BETWEEN_REQUESTS = 0.4   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_url(instrument: str, day: date) -> str:
    """Build the Dukascopy CDN URL for a single day of 1m candles."""
    # Months are 0-indexed in the URL
    return (
        f"{_BASE_URL}/{instrument}"
        f"/{day.year}/{day.month - 1:02d}/{day.day:02d}"
        f"/BID_candles_min_1.bi5"
    )


def _fetch_bi5(url: str, retries: int = 3, backoff: float = 1.5) -> Optional[bytes]:
    """
    Download a BI5 file; return raw bytes or None.

    Returns None for:
      • HTTP 404  (weekend / holiday — expected, not an error)
      • All retries exhausted
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.exceptions.HTTPError as exc:
            logger.debug("HTTP %s for %s (attempt %d)", exc.response.status_code, url, attempt + 1)
            if attempt == retries - 1:
                return None
        except requests.RequestException as exc:
            logger.warning("Request failed for %s (attempt %d): %s", url, attempt + 1, exc)
            if attempt == retries - 1:
                return None
        time.sleep(backoff * (attempt + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BI5 decoding
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bi5(data: bytes, day: date, scale: int) -> pd.DataFrame:
    """
    Decompress and decode one day of 1-minute candle BI5 data.

    Returns a DataFrame with columns [date, open, high, low, close, volume],
    UTC timestamps. Empty DataFrame on failure or empty file.
    """
    if not data:
        return pd.DataFrame(columns=_OHLCV)
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError as exc:
        logger.debug("LZMA decompress failed for %s: %s", day, exc)
        return pd.DataFrame(columns=_OHLCV)

    n_records = len(raw) // _CANDLE_STRUCT.size
    if n_records == 0:
        return pd.DataFrame(columns=_OHLCV)

    midnight_utc = datetime(day.year, day.month, day.day, 0, 0, 0)
    rows: List[Tuple] = []

    for i in range(n_records):
        chunk = raw[i * _CANDLE_STRUCT.size : (i + 1) * _CANDLE_STRUCT.size]
        fields = _CANDLE_STRUCT.unpack(chunk)

        ts = midnight_utc + timedelta(milliseconds=int(fields[_TS]))
        o  = fields[_O] / scale
        c  = fields[_C] / scale
        lo = fields[_L] / scale
        hi = fields[_H] / scale
        v  = float(fields[_V])

        rows.append((ts, o, hi, lo, c, v))

    return pd.DataFrame(rows, columns=_OHLCV)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair download
# ─────────────────────────────────────────────────────────────────────────────

def _iter_dates(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_pair_1m(
    instrument:  str,
    start:       date,
    end:         date,
    cache_dir:   Optional[Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Download 1-minute BID candles for *instrument* from *start* to *end*.

    Args:
        instrument    Dukascopy code, e.g. 'GBPJPY' (no slash).
        start / end   Inclusive date range.
        cache_dir     If set, raw BI5 bytes are cached here; subsequent calls
                      with the same pair/date range skip re-downloading.
        show_progress Print a progress line every 30 days.

    Returns:
        Chronologically sorted DataFrame[date, open, high, low, close, volume].
        Empty DataFrame if no data is available.
    """
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    scale       = _SCALE.get(instrument, 100_000)
    frames: List[pd.DataFrame] = []
    n_downloaded = 0
    total_days   = (end - start).days + 1

    for idx, day in enumerate(_iter_dates(start, end)):
        raw: Optional[bytes] = None

        # Try cache first
        if cache_dir:
            cache_path = cache_dir / f"{instrument}_{day.isoformat()}.bi5"
            if cache_path.exists():
                raw = cache_path.read_bytes()

        if raw is None:
            url = _build_url(instrument, day)
            raw = _fetch_bi5(url)
            if raw and cache_dir:
                cache_path = cache_dir / f"{instrument}_{day.isoformat()}.bi5"
                cache_path.write_bytes(raw)
            time.sleep(_SLEEP_BETWEEN_REQUESTS)

        if raw:
            df = _parse_bi5(raw, day, scale)
            if not df.empty:
                frames.append(df)
                n_downloaded += 1

        if show_progress and (idx + 1) % 30 == 0:
            pct = (idx + 1) / total_days * 100
            logger.info(
                "  %s  %s  [%d/%d days, %.0f%% complete]",
                instrument, day.isoformat(), idx + 1, total_days, pct,
            )

    logger.info(
        "%s: %d/%d calendar days returned data  (%d skipped — weekends/holidays)",
        instrument, n_downloaded, total_days, total_days - n_downloaded,
    )

    if not frames:
        return pd.DataFrame(columns=_OHLCV)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a *date*-column OHLCV DataFrame to a higher timeframe."""
    indexed = df.set_index("date")
    out = indexed.resample(rule, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"]).reset_index()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────────────────

def verify_day(instrument: str, verify_date: str) -> None:
    """
    Download and print a single day's bars so you can compare against
    the Dukascopy web UI to confirm field ordering is correct.

    Usage:
        python scripts/download_dukascopy.py --verify GBPJPY 2022-03-16
    """
    day   = date.fromisoformat(verify_date)
    scale = _SCALE.get(instrument, 100_000)
    url   = _build_url(instrument, day)

    print(f"\nVerification: {instrument}  {verify_date}")
    print(f"URL: {url}\n")

    raw = _fetch_bi5(url)
    if raw is None:
        print("No data returned (404 or network error).")
        return

    df = _parse_bi5(raw, day, scale)
    if df.empty:
        print("Data decoded but DataFrame is empty.")
        return

    print(f"Decoded {len(df)} bars.  First 10:")
    print(df.head(10).to_string(index=False))
    print(f"\nLast 5:")
    print(df.tail(5).to_string(index=False))
    print(
        "\nCompare open/close/high/low against the Dukascopy web chart for this day.\n"
        "URL: https://www.dukascopy.com/swiss/english/marketwatch/historic/\n"
        "If open and close appear swapped, edit _O and _C indices at the top\n"
        "of this script (_O, _C = 2, 1 instead of 1, 2).\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Dukascopy 1-minute candles and resample to multiple timeframes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ── download sub-command (default when no sub-command given) ──────────────
    dl = sub.add_parser("download", help="Download historical data (default)")
    dl.add_argument(
        "--pairs", nargs="+", metavar="PAIR",
        default=["GBPJPY", "USDJPY", "EURJPY", "GBPUSD"],
        help="Dukascopy instrument codes (no slash)",
    )
    dl.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    dl.add_argument("--end",   default="2024-12-31", help="End date YYYY-MM-DD")
    dl.add_argument("--out",   default="data",       help="Output directory for Parquet files")
    dl.add_argument(
        "--timeframes", nargs="+", metavar="TF",
        default=["1m", "15m", "1h", "4h", "1d"],
        help="Timeframes to save (1m always used as base for resampling)",
    )
    dl.add_argument(
        "--cache", default=None,
        help="Cache directory for raw BI5 files (omit to disable caching)",
    )
    dl.add_argument("--keep-1m", action="store_true",
                    help="Keep 1m Parquet even when not in --timeframes")
    dl.add_argument("--verbose", action="store_true")

    # ── verify sub-command ────────────────────────────────────────────────────
    vf = sub.add_parser("verify", help="Print a single day's bars for field-order verification")
    vf.add_argument("instrument", help="e.g. GBPJPY")
    vf.add_argument("date",       help="YYYY-MM-DD")
    vf.add_argument("--verbose",  action="store_true")

    # Backwards-compat: allow flags at top level (no sub-command)
    p.add_argument("--pairs", nargs="+", metavar="PAIR",
                   default=["GBPJPY", "USDJPY", "EURJPY", "GBPUSD"])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end",   default="2024-12-31")
    p.add_argument("--out",   default="data")
    p.add_argument("--timeframes", nargs="+", metavar="TF",
                   default=["1m", "15m", "1h", "4h", "1d"])
    p.add_argument("--cache", default=None)
    p.add_argument("--keep-1m", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verify", nargs=2, metavar=("INSTRUMENT", "DATE"),
                   help="Verify field order for one day (e.g. --verify GBPJPY 2022-03-16)")

    return p.parse_args()


def run_download(args: argparse.Namespace) -> None:
    pairs      = getattr(args, "pairs", ["GBPJPY", "USDJPY", "EURJPY", "GBPUSD"])
    start      = date.fromisoformat(getattr(args, "start", "2015-01-01"))
    end        = date.fromisoformat(getattr(args, "end",   "2024-12-31"))
    out_dir    = Path(getattr(args, "out",   "data"))
    timeframes = getattr(args, "timeframes", ["1m", "15m", "1h", "4h", "1d"])
    cache_arg  = getattr(args, "cache", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_arg) if cache_arg else (out_dir / ".bi5_cache")

    total_days = (end - start).days + 1
    print(f"\nDukascopy downloader")
    print(f"Pairs       : {', '.join(pairs)}")
    print(f"Date range  : {start} → {end}  ({total_days} calendar days)")
    print(f"Timeframes  : {', '.join(timeframes)}")
    print(f"Output      : {out_dir.resolve()}")
    print(f"BI5 cache   : {cache_dir.resolve()}")
    print()

    for instrument in pairs:
        print(f"{'─' * 64}")
        print(f"  {instrument}  ({start} → {end})")
        print(f"{'─' * 64}")

        df_1m = download_pair_1m(
            instrument, start, end,
            cache_dir=cache_dir / instrument,
        )

        if df_1m.empty:
            print(f"  [WARN] No data returned for {instrument} — skipping.\n")
            continue

        print(f"  1m bars: {len(df_1m):,}  "
              f"({df_1m['date'].min().date()} → {df_1m['date'].max().date()})")

        # Save 1m base
        if "1m" in timeframes or getattr(args, "keep_1m", False):
            path_1m = out_dir / f"{instrument}_1m.parquet"
            df_1m.to_parquet(path_1m, index=False)
            print(f"  Saved  1m  → {path_1m.name}")

        # Resample and save each requested TF
        for tf in timeframes:
            if tf == "1m":
                continue
            rule = _TF_RULES.get(tf)
            if rule is None:
                logger.warning("Unknown timeframe '%s' — skipped", tf)
                continue
            df_tf   = resample_ohlcv(df_1m, rule)
            tf_path = out_dir / f"{instrument}_{tf}.parquet"
            df_tf.to_parquet(tf_path, index=False)
            print(f"  Saved {tf:>3}  → {tf_path.name}  ({len(df_tf):,} bars)")

        print()

    print("=" * 64)
    print(f"Done.  Parquet files are in:  {out_dir.resolve()}")
    print()
    print("Next steps:")
    print("  Verify field ordering (compare one day vs Dukascopy web UI):")
    print(f"    python scripts/download_dukascopy.py verify {pairs[0]} 2022-03-16")
    print()
    print("  Run preprocessing pipeline (gap check + flash-crash filter):")
    print("    python cli.py --mode preprocess --data data/")
    print()
    print("  Train the model:")
    print("    python cli.py --mode train --data data/")


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle --verify shortcut
    if getattr(args, "verify", None):
        instrument, verify_date = args.verify
        verify_day(instrument, verify_date)
        return

    # Handle 'verify' sub-command
    if getattr(args, "command", None) == "verify":
        verify_day(args.instrument, args.date)
        return

    run_download(args)


if __name__ == "__main__":
    main()
