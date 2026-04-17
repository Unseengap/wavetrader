"""Process raw 1-min JForex CSVs into test-split parquet for fib_scalper."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "processed_data"
sys.path.insert(0, str(ROOT))

PAIRS = ["GBPJPY"]  # Start with GBPJPY
TEST_START = pd.Timestamp("2024-07-01 00:00:00")
PIP_SIZE = {"GBPJPY": 0.01, "EURJPY": 0.01, "GBPUSD": 0.0001}

TF_DIR_NAME = {"1m": "1 Min"}

def _find_csv(pair, tf_key, side):
    fragment = TF_DIR_NAME[tf_key]
    candidates = [
        f for f in DATA_DIR.glob(f"{pair}_{fragment}_{side}_*.csv")
        if "(1)" not in f.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV for {pair} {tf_key} {side}")
    return sorted(candidates)[0]

def _load_jforex_csv(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    raw_ts = pd.to_datetime(df["Time (EET)"], format="%d.%m.%Y %H:%M:%S")
    from zoneinfo import ZoneInfo
    eet = ZoneInfo("Europe/Athens")
    ts_utc = (
        raw_ts.dt.tz_localize(eet, ambiguous="infer", nonexistent="shift_forward")
              .dt.tz_convert("UTC")
              .dt.tz_localize(None)
    )
    df = df.drop(columns=["Time (EET)"])
    df.columns = [c.strip().lower() for c in df.columns]
    df.insert(0, "timestamp", ts_utc)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

def _load_mid(pair, tf_key):
    bid = _load_jforex_csv(_find_csv(pair, tf_key, "Bid"))
    ask = _load_jforex_csv(_find_csv(pair, tf_key, "Ask"))
    bid = bid.rename(columns={"open": "bid_open", "high": "bid_high", "low": "bid_low", "close": "bid_close", "volume": "volume"})
    ask = ask.rename(columns={"open": "ask_open", "high": "ask_high", "low": "ask_low", "close": "ask_close", "volume": "ask_volume"})
    df = bid.join(ask, how="inner")
    df["open_mid"]  = (df["bid_open"]  + df["ask_open"])  / 2.0
    df["high_mid"]  = (df["bid_high"]  + df["ask_high"])  / 2.0
    df["low_mid"]   = (df["bid_low"]   + df["ask_low"])   / 2.0
    df["close_mid"] = (df["bid_close"] + df["ask_close"]) / 2.0
    pip = PIP_SIZE[pair]
    open_spread  = (df["ask_open"]  - df["bid_open"])  / pip
    close_spread = (df["ask_close"] - df["bid_close"]) / pip
    df["spread_avg"] = (open_spread + close_spread) / 2.0
    return df[["open_mid", "high_mid", "low_mid", "close_mid", "volume", "spread_avg"]]

for pair in PAIRS:
    print(f"Loading {pair} 1m data...")
    df_1m = _load_mid(pair, "1m")
    print(f"  Total 1m bars: {len(df_1m)}")

    # Quality filters
    pct_chg = df_1m["close_mid"].pct_change().abs()
    df_1m = df_1m[pct_chg <= 0.05]
    df_1m = df_1m[df_1m["spread_avg"] <= 20.0]
    df_1m = df_1m.dropna(subset=["open_mid", "high_mid", "low_mid", "close_mid"])
    print(f"  After filters: {len(df_1m)}")

    # Add basic features
    from wavetrader.indicators import calculate_atr, calculate_rsi
    df_1m["log_return"] = np.log(df_1m["close_mid"] / df_1m["close_mid"].shift(1))
    print(f"  Features added")

    # Test split only (save memory)
    df_test = df_1m[df_1m.index >= TEST_START]
    print(f"  Test split: {len(df_test)} bars ({df_test.index[0]} to {df_test.index[-1]})")

    out_dir = OUT_DIR / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pair}_1m.parquet"
    df_test.to_parquet(path, index=True)
    print(f"  Saved: {path} ({len(df_test)} bars)")

print("\nDone!")
