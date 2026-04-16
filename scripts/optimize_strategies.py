#!/usr/bin/env python3
"""
Strategy parameter optimizer — tests parameter grids for each strategy
and reports the best configurations.

Usage:
    python scripts/optimize_strategies.py
"""
import itertools
import json
import sys
import time
import requests

BASE = "http://127.0.0.1:5000"
BASE_CONFIG = {
    "pair": "GBP/JPY",
    "initial_balance": 100.0,
    "risk_per_trade": 0.10,
    "leverage": 20.0,
    "spread_pips": 3.0,
    "pip_value": 6.5,
    "min_confidence": 0.40,
}


def run_backtest(strategy_id, strategy_params=None, ai_confirm=False):
    """Run a single backtest and return key metrics."""
    config = {
        **BASE_CONFIG,
        "strategy": strategy_id,
        "ai_confirm": ai_confirm,
    }
    if strategy_params:
        config["strategy_params"] = strategy_params

    try:
        resp = requests.post(f"{BASE}/api/backtest/run", json=config, timeout=300)
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    if "error" in data:
        return {"error": data["error"]}

    m = data.get("metrics", {})
    trades = data.get("trades", [])
    return {
        "trades": m.get("total_trades", len(trades)),
        "win_rate": m.get("win_rate", 0),
        "final_balance": m.get("final_balance", 100),
        "total_pnl": m.get("total_pnl", 0),
        "profit_factor": m.get("profit_factor", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "max_dd": m.get("max_drawdown_pct", 0),
        "return_pct": ((m.get("final_balance", 100) - 100) / 100) * 100,
    }


def score(result):
    """Composite score: profit factor * sqrt(trades) * (1 - max_dd/100)."""
    if "error" in result or result["trades"] < 5:
        return -999
    pf = result.get("profit_factor", 0)
    wr = result.get("win_rate", 0)
    trades = result["trades"]
    ret = result["return_pct"]
    dd = abs(result.get("max_dd", 100))
    # Favor: high return, high PF, low drawdown, decent trade count
    import math
    return ret * (1 - dd / 200) * min(pf, 3.0) * math.log10(max(trades, 1))


# ══════════════════════════════════════════════════════════════════════════
# Parameter grids for each strategy
# ══════════════════════════════════════════════════════════════════════════

STRATEGY_GRIDS = {
    "amd_session": [
        {"asian_range_max_pips": ar, "london_sweep_min_mag": sw, "sr_proximity_min": sr,
         "min_engulfing_strength": eng, "min_rr_ratio": rr, "min_confidence": 0.4}
        for ar in [100, 150, 200]
        for sw in [0.05, 0.08, 0.12]
        for sr in [0.1, 0.2, 0.3]
        for eng in [0.05, 0.1]
        for rr in [1.0, 1.5, 2.0]
    ],
    "ema_crossover": [
        {"ema_fast": fast, "ema_slow": slow, "trend_ema": trend,
         "min_adx": adx, "min_rr_ratio": rr, "min_confidence": 0.4}
        for fast in [8, 12, 20]
        for slow in [30, 50, 100]
        for trend in [150, 200]
        for adx in [15, 20, 25]
        for rr in [1.0, 1.5, 2.0]
    ],
    "orb_breakout": [
        {"min_orb_range_pips": mn, "max_orb_range_pips": mx,
         "pullback_tolerance_pct": pb, "tp_range_mult": tp, "min_rr_ratio": rr}
        for mn in [5, 8, 12]
        for mx in [40, 60, 80]
        for pb in [0.2, 0.4, 0.6]
        for tp in [1.5, 2.0, 2.5]
        for rr in [1.0, 1.5]
    ],
    "supply_demand": [
        {"zone_atr_scale": z, "min_zone_touches": t, "entry_type": e,
         "min_rr_ratio": rr, "min_confidence": mc}
        for z in [1.0, 1.5, 2.0]
        for t in [1, 2, 3]
        for e in ["aggressive", "conservative"]
        for rr in [1.5, 2.0, 2.5]
        for mc in [0.4, 0.5, 0.6]
    ],
    "ict_smc": [
        {"min_displacement_atr": d, "fvg_fill_pct": f, "min_ob_strength": ob,
         "min_rr_ratio": rr, "session_filter": sf}
        for d in [1.0, 1.5, 2.0]
        for f in [0.3, 0.5, 0.7]
        for ob in [0.3, 0.5, 0.7]
        for rr in [1.5, 2.0, 2.5]
        for sf in [True, False]
    ],
    "structure_break": [
        {"min_break_strength": b, "retest_tolerance_pct": rt, "use_fvg": fvg,
         "min_rr_ratio": rr, "min_confidence": mc}
        for b in [0.3, 0.5, 0.8]
        for rt in [0.2, 0.4, 0.6]
        for fvg in [True, False]
        for rr in [1.5, 2.0, 2.5]
        for mc in [0.4, 0.5]
    ],
    "mean_reversion": [
        {"bb_period": bp, "bb_std": bs, "rsi_oversold": ro, "rsi_overbought": rb,
         "min_rr_ratio": rr}
        for bp in [15, 20, 30]
        for bs in [1.5, 2.0, 2.5]
        for ro in [25, 30, 35]
        for rb in [65, 70, 75]
        for rr in [1.0, 1.5, 2.0]
    ],
}


def optimize_strategy(strategy_id, grid, top_n=3):
    """Test all parameter combos and return top N results."""
    print(f"\n{'='*70}")
    print(f"  OPTIMIZING: {strategy_id}  ({len(grid)} combinations)")
    print(f"{'='*70}")

    results = []
    total = len(grid)

    for idx, params in enumerate(grid):
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        sys.stdout.write(f"\r  [{idx+1}/{total}] Testing... ")
        sys.stdout.flush()

        r = run_backtest(strategy_id, params, ai_confirm=False)
        if "error" not in r:
            s = score(r)
            results.append((s, params, r))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n\n  TOP {top_n} CONFIGURATIONS:")
    print(f"  {'-'*66}")
    for rank, (s, params, r) in enumerate(results[:top_n], 1):
        print(f"\n  #{rank} (score: {s:.1f})")
        print(f"    Trades: {r['trades']}  WR: {r['win_rate']:.1%}  "
              f"PF: {r['profit_factor']:.2f}  Return: {r['return_pct']:+.1f}%  "
              f"MaxDD: {r.get('max_dd', '?')}%  Sharpe: {r.get('sharpe', '?')}")
        print(f"    Params: {json.dumps(params, indent=None)}")

    # Test best config with AI on
    if results:
        best_params = results[0][1]
        print(f"\n  BEST CONFIG + AI CONFIRM ON:")
        r_ai = run_backtest(strategy_id, best_params, ai_confirm=True)
        if "error" not in r_ai:
            print(f"    Trades: {r_ai['trades']}  WR: {r_ai['win_rate']:.1%}  "
                  f"PF: {r_ai['profit_factor']:.2f}  Return: {r_ai['return_pct']:+.1f}%  "
                  f"MaxDD: {r_ai.get('max_dd', '?')}%")

    return results[:top_n] if results else []


if __name__ == "__main__":
    # Check server
    try:
        r = requests.get(f"{BASE}/api/backtest/strategies", timeout=5)
        strategies = r.json()
        print(f"Connected to server — {len(strategies)} strategies available")
    except Exception:
        print("ERROR: Server not running at", BASE)
        sys.exit(1)

    all_best = {}
    for strat_id in STRATEGY_GRIDS:
        grid = STRATEGY_GRIDS[strat_id]
        top = optimize_strategy(strat_id, grid)
        if top:
            all_best[strat_id] = {
                "best_params": top[0][1],
                "best_result": top[0][2],
                "score": top[0][0],
            }

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY — ALL STRATEGIES")
    print(f"{'='*70}")
    for sid, info in sorted(all_best.items(), key=lambda x: x[1]["score"], reverse=True):
        r = info["best_result"]
        print(f"\n  {sid:20s}  Return: {r['return_pct']:+7.1f}%  "
              f"WR: {r['win_rate']:.1%}  PF: {r['profit_factor']:.2f}  "
              f"Trades: {r['trades']:4d}  Score: {info['score']:.1f}")
        print(f"    → {json.dumps(info['best_params'])}")

    # Save results
    with open("scripts/optimization_results.json", "w") as f:
        json.dump(all_best, f, indent=2, default=str)
    print(f"\nResults saved to scripts/optimization_results.json")
