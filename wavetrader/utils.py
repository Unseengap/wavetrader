"""
Miscellaneous utilities: ASCII equity chart, chronological data splitting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# ASCII equity chart
# ─────────────────────────────────────────────────────────────────────────────

def print_equity_chart(equity_curve: List[float], width: int = 60) -> None:
    """Render a terminal-width ASCII equity curve."""
    if len(equity_curve) < 2:
        return

    step    = max(1, len(equity_curve) // width)
    sampled = equity_curve[::step][:width]

    min_eq  = min(sampled)
    max_eq  = max(sampled)
    rng     = max_eq - min_eq or 1.0
    height  = 15

    chart = [[" "] * len(sampled) for _ in range(height)]
    for x, eq in enumerate(sampled):
        y = int((eq - min_eq) / rng * (height - 1))
        chart[height - 1 - y][x] = "█"

    print("\nEquity Curve:")
    print(f"${max_eq:,.0f} ┤{''.join(chart[0])}")
    for row in chart[1:-1]:
        print(f"{'':>8} │{''.join(row)}")
    print(f"${min_eq:,.0f} ┤{''.join(chart[-1])}")
    print(f"{'':>9}└{'─' * len(sampled)}")


# ─────────────────────────────────────────────────────────────────────────────
# Chronological data splits
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct:   float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split `df` in time order into train / val / test.
    Never shuffle: shuffling would leak future prices into the training window.
    """
    n          = len(df)
    train_end  = int(n * train_pct)
    val_end    = train_end + int(n * val_pct)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def chronological_split_mtf(
    dataframes: Dict[str, pd.DataFrame],
    ref_timeframe: str = "15min",
    train_pct: float   = 0.70,
    val_pct:   float   = 0.15,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split all timeframe DataFrames consistently based on the reference TF.
    Higher TFs are split by their own length × the same fractional positions.
    """
    ref = dataframes[ref_timeframe]
    n   = len(ref)

    train_data: Dict[str, pd.DataFrame] = {}
    val_data:   Dict[str, pd.DataFrame] = {}
    test_data:  Dict[str, pd.DataFrame] = {}

    for tf, df in dataframes.items():
        ratio  = len(df) / n
        t_end  = int(int(n * train_pct) * ratio)
        v_end  = t_end + int(int(n * val_pct) * ratio)
        train_data[tf] = df.iloc[:t_end].reset_index(drop=True)
        val_data[tf]   = df.iloc[t_end:v_end].reset_index(drop=True)
        test_data[tf]  = df.iloc[v_end:].reset_index(drop=True)

    return train_data, val_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# Calendar-aware purged walk-forward CV
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits_calendar(
    df: pd.DataFrame,
    min_train_date:   str           = "2015-01-01",
    first_test_date:  str           = "2019-01-01",
    test_months:      int           = 3,
    purge_days:       int           = 14,
    holdout_date:     Optional[str] = "2024-01-01",
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calendar-aware purged walk-forward CV for time-series data.

    Implements the spec from the data preparation guide:
      • Expanding train window anchored at *min_train_date*
      • Quarterly (or custom) test folds starting from *first_test_date*
      • Mandatory purge gap between train end and test start to prevent
        label-lookahead leakage from overlapping prediction windows
      • Optional holdout: data from *holdout_date* onward is excluded from
        all folds — reserve this for the final production validation

    With default settings the splits are:
        Fold  1: Train 2015–2018-Q4,  Test 2019-Q1  (purge: 2 weeks before)
        Fold  2: Train 2015–2019-Q1,  Test 2019-Q2
        ...
        Fold 16: Train 2015–2022-Q4,  Test 2023-Q1
        ...     (ends just before 2024-01-01 holdout)

    Args:
        df               DataFrame with a ``date`` column, sorted ascending.
        min_train_date   Earliest date included in any training fold.
        first_test_date  Start of the first test fold.
        test_months      Length of each test fold in months (3 = quarterly).
        purge_days       Days between train end and test start (default 14 =
                         2-week purge prevents lookahead from 10-bar windows).
        holdout_date     If given, rows on or after this date are excluded from
                         all folds.  Pass ``None`` to disable.

    Returns:
        List of ``(train_df, test_df)`` tuples, one per fold.

    Example::

        folds = walk_forward_splits_calendar(df, n_folds_hint=16)
        for i, (train, test) in enumerate(folds, 1):
            model.fit(train)
            print(f"Fold {i:02d}  test={test['date'].min().date()} "
                  f"..{test['date'].max().date()}  n={len(test):,}")

    See also: walk_forward_splits() in training.py for an index-based version.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    anchor   = pd.Timestamp(min_train_date)
    fold_cur = pd.Timestamp(first_test_date)
    cutoff   = pd.Timestamp(holdout_date) if holdout_date else df["date"].max() + pd.Timedelta(days=1)

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

    while fold_cur < cutoff:
        fold_end   = fold_cur + pd.DateOffset(months=test_months)
        if fold_end > cutoff:
            fold_end = cutoff

        purge_end   = fold_cur                          # test window opens here
        purge_start = fold_cur - pd.Timedelta(days=purge_days)

        train_mask = (df["date"] >= anchor)    & (df["date"] <  purge_start)
        test_mask  = (df["date"] >= purge_end) & (df["date"] <  fold_end)

        train_fold = df[train_mask].reset_index(drop=True)
        test_fold  = df[test_mask].reset_index(drop=True)

        if len(train_fold) > 0 and len(test_fold) > 0:
            splits.append((train_fold, test_fold))

        fold_cur = fold_end

    return splits
