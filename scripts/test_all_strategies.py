"""
Test ALL strategies with OANDA config ($100, 10% risk, 20:1 leverage).
Same setup used for price_action_reversal optimization.
"""
import pandas as pd
from pathlib import Path
from wavetrader.strategies.registry import get_strategy_registry
from wavetrader.strategy_backtest import run_strategy_backtest
from wavetrader.config import BacktestConfig

# ── Data loader ──────────────────────────────────────────────────────────────
processed_dir = Path("processed_data/test")

PAIRS = {
    "GBP/JPY": "GBPJPY",
    "EUR/JPY": "EURJPY",
    "GBP/USD": "GBPUSD",
}

ALL_TFS = ["15min", "1h", "4h", "1d"]
TF_FILE_MAP = {"15min": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}


def load_pair_data(pair_tag: str) -> dict:
    """Load all available timeframes for a pair."""
    df_dict = {}
    for tf in ALL_TFS:
        tf_short = TF_FILE_MAP[tf]
        p = processed_dir / f"{pair_tag}_{tf_short}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.name in ("timestamp", "date", "datetime"):
                df = df.reset_index()
            df.columns = [c.strip().lower() for c in df.columns]
            col_map = {}
            for base in ("open", "high", "low", "close"):
                if f"{base}_mid" in df.columns and base not in df.columns:
                    col_map[f"{base}_mid"] = base
            if col_map:
                df = df.rename(columns=col_map)
            if "date" not in df.columns:
                for alias in ("datetime", "timestamp", "time"):
                    if alias in df.columns:
                        df = df.rename(columns={alias: "date"})
                        break
            df_dict[tf] = df
    return df_dict


# ── Config (OANDA $100, 10% risk) ───────────────────────────────────────────
def make_config(pair: str) -> BacktestConfig:
    cfg = BacktestConfig()
    cfg.initial_balance = 100.0
    cfg.risk_pct = 0.10
    # OANDA settings already in defaults: leverage 20, spread 4.2, commission 0
    # Adjust pip_value for non-JPY pairs
    if "JPY" not in pair:
        cfg.pip_value = 10.0       # USD-denominated majors
        cfg.pip_size = 0.0001
        cfg.spread_pips = 1.5      # GBP/USD spread
    return cfg


# ── Strategies to test ───────────────────────────────────────────────────────
STRATEGIES = [
    "news_catalyst_ob",
    "opening_break_retest",
    "price_action_reversal",
    "harmonic_scanner",
]

reg = get_strategy_registry()

# ── Run ──────────────────────────────────────────────────────────────────────
print("=" * 90)
print("ALL STRATEGIES BACKTEST — OANDA $100, 10% risk, 20:1 leverage")
print("=" * 90)

all_results = []

for strat_id in STRATEGIES:
    print(f"\n{'─' * 90}")
    print(f"STRATEGY: {strat_id}")
    print(f"{'─' * 90}")

    for pair_name, pair_tag in PAIRS.items():
        df_dict = load_pair_data(pair_tag)
        if not df_dict:
            print(f"  {pair_name}: NO DATA")
            continue

        tfs_loaded = {tf: len(df) for tf, df in df_dict.items()}

        cfg = make_config(pair_name)

        try:
            strat = reg.instantiate(strat_id)
            results = run_strategy_backtest(
                strat, df_dict, bt_config=cfg, pair=pair_name, verbose=False
            )
        except Exception as e:
            print(f"  {pair_name}: ERROR — {e}")
            all_results.append({
                "strategy": strat_id, "pair": pair_name,
                "trades": 0, "error": str(e),
            })
            continue

        trades = results.trades
        n = len(trades)

        if n == 0:
            print(f"  {pair_name}: 0 trades (TFs: {tfs_loaded})")
            all_results.append({
                "strategy": strat_id, "pair": pair_name,
                "trades": 0, "wr": 0, "pf": 0,
                "balance": cfg.initial_balance, "pnl_pct": 0,
            })
            continue

        wins = sum(1 for t in trades if t.pnl > 0)
        total_win = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = total_win / total_loss if total_loss > 0 else 999.0
        wr = wins / n

        # Exit reason breakdown
        sl = sum(1 for t in trades if (t.exit_reason or '').strip() == 'Stop Loss')
        tp = sum(1 for t in trades if 'take' in (t.exit_reason or '').lower() or 'profit' in (t.exit_reason or '').lower())
        trail = sum(1 for t in trades if 'trail' in (t.exit_reason or '').lower())
        partial = sum(1 for t in trades if 'partial' in (t.exit_reason or '').lower())
        other = n - sl - tp - trail - partial

        avg_w = total_win / max(wins, 1)
        avg_l = total_loss / max(n - wins, 1)
        rr = avg_w / avg_l if avg_l > 0 else 999
        max_w = max((t.pnl for t in trades), default=0)
        max_l = min((t.pnl for t in trades), default=0)

        pnl_pct = (results.final_balance - cfg.initial_balance) / cfg.initial_balance * 100

        # Entry types
        entry_types = {}
        for t in trades:
            et = (t.context or {}).get("entry_type", "unknown")
            entry_types[et] = entry_types.get(et, 0) + 1

        print(f"  {pair_name}: {n:>3} trades | WR={wr:.1%} | PF={pf:.2f} | "
              f"${cfg.initial_balance:.0f}→${results.final_balance:.0f} ({pnl_pct:+.0f}%) | "
              f"AvgW=${avg_w:.1f} AvgL=${avg_l:.1f} R:R={rr:.2f}")
        print(f"           Exits: SL={sl} TP={tp} Trail={trail} Partial={partial} Other={other}")
        if entry_types:
            et_str = ", ".join(f"{k}={v}" for k, v in sorted(entry_types.items()))
            print(f"           Entry types: {et_str}")

        all_results.append({
            "strategy": strat_id, "pair": pair_name,
            "trades": n, "wins": wins, "wr": wr, "pf": pf,
            "balance": results.final_balance, "pnl_pct": pnl_pct,
            "avg_w": avg_w, "avg_l": avg_l, "rr": rr,
            "max_w": max_w, "max_l": max_l,
            "sl": sl, "tp": tp, "trail": trail,
        })


# ── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("SUMMARY")
print(f"{'=' * 90}")
print(f"{'Strategy':<25} {'Pair':<10} {'Trades':>6} {'WR':>6} {'PF':>6} {'Balance':>10} {'PnL%':>8} {'R:R':>6}")
print(f"{'─' * 25} {'─' * 10} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 6}")
for r in all_results:
    if r["trades"] == 0:
        err = r.get("error", "0 trades")
        print(f"{r['strategy']:<25} {r['pair']:<10} {'—':>6} {'—':>6} {'—':>6} {'—':>10} {'—':>8} {err}")
    else:
        print(f"{r['strategy']:<25} {r['pair']:<10} {r['trades']:>6} {r['wr']:>5.1%} {r['pf']:>6.2f} "
              f"${r['balance']:>9.0f} {r['pnl_pct']:>+7.0f}% {r['rr']:>6.2f}")

# ── Best per strategy ────────────────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("BEST PAIR PER STRATEGY")
print(f"{'=' * 90}")
for strat_id in STRATEGIES:
    strat_results = [r for r in all_results if r["strategy"] == strat_id and r["trades"] > 0]
    if strat_results:
        best = max(strat_results, key=lambda r: r["pnl_pct"])
        print(f"  {strat_id:<25} → {best['pair']} | {best['trades']} trades | "
              f"WR={best['wr']:.1%} PF={best['pf']:.2f} | ${best['balance']:.0f} ({best['pnl_pct']:+.0f}%)")
    else:
        print(f"  {strat_id:<25} → No trades on any pair")
