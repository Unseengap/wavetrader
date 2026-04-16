#!/usr/bin/env python3
"""
Focused strategy parameter optimizer.
Tests ~10-15 configs per strategy with $100 balance, 10% risk.
"""
import json, sys, time, math, requests

BASE = "http://127.0.0.1:5000"
CFG = {
    "pair": "GBP/JPY",
    "initial_balance": 100.0,
    "risk_per_trade": 0.10,
    "leverage": 20.0,
    "spread_pips": 3.0,
    "pip_value": 6.5,
    "min_confidence": 0.40,
}

def bt(sid, sp=None, ai=False):
    c = {**CFG, "strategy": sid, "ai_confirm": ai}
    if sp: c["strategy_params"] = sp
    try:
        r = requests.post(f"{BASE}/api/backtest/run", json=c, timeout=300).json()
    except Exception as e:
        return {"error": str(e)}
    if "error" in r: return {"error": r["error"]}
    m = r.get("metrics", {})
    t = r.get("trades", [])
    fb = m.get("final_balance", 100)
    return {
        "trades": m.get("total_trades", len(t)),
        "wr": m.get("win_rate", 0),
        "fb": fb,
        "pnl": m.get("total_pnl", 0),
        "pf": m.get("profit_factor", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "dd": m.get("max_drawdown_pct", 0),
        "ret": ((fb - 100) / 100) * 100,
    }

def sc(r):
    if "error" in r or r["trades"] < 3: return -999
    return r["ret"] * (1 - abs(r.get("dd", 100)) / 200) * min(r["pf"], 3) * math.log10(max(r["trades"], 1))

# ═══ STRATEGY GRIDS (correct param names, small focused grid) ═══

GRIDS = {
    "amd_session": [
        {"asian_range_max_pips": ar, "london_sweep_min_mag": sw, "sr_proximity_min": sr,
         "min_engulfing_strength": eng, "min_rr_ratio": rr}
        for ar, sw, sr, eng, rr in [
            (150, 0.05, 0.1, 0.05, 1.0),   # very loose
            (150, 0.05, 0.1, 0.05, 1.5),
            (200, 0.05, 0.05, 0.05, 1.0),  # widest
            (120, 0.08, 0.2, 0.1, 1.5),    # current default
            (120, 0.06, 0.15, 0.08, 1.0),
            (180, 0.04, 0.0, 0.05, 1.0),   # no S&R filter
            (150, 0.06, 0.1, 0.05, 2.0),   # tight RR
            (200, 0.03, 0.0, 0.05, 1.5),   # very loose
            (100, 0.1, 0.15, 0.1, 1.5),    # tighter
            (250, 0.02, 0.0, 0.05, 1.0),   # maximum loose
        ]
    ],
    "ema_crossover": [
        {"fast_ema": f, "slow_ema": s, "trend_ema": t, "pullback_max_bars": pb, "min_rr_ratio": rr}
        for f, s, t, pb, rr in [
            (8, 21, 200, 8, 1.5),
            (12, 50, 200, 10, 1.5),
            (20, 50, 200, 10, 2.0),   # current default
            (8, 30, 150, 12, 1.0),
            (12, 30, 200, 15, 1.5),
            (5, 21, 100, 8, 1.0),
            (10, 40, 150, 10, 1.5),
            (8, 21, 100, 15, 1.0),    # fast + wide pullback
            (15, 50, 200, 8, 2.0),
            (8, 30, 200, 20, 1.0),    # very wide pullback
            (12, 26, 100, 10, 1.5),   # classic EMA
        ]
    ],
    "orb_breakout": [
        {"min_orb_range_pips": mn, "max_orb_range_pips": mx, "pullback_tolerance_pct": pb,
         "tp_range_mult": tp, "min_rr_ratio": rr}
        for mn, mx, pb, tp, rr in [
            (5, 80, 0.4, 2.0, 1.0),
            (5, 60, 0.3, 1.5, 1.0),
            (8, 60, 0.3, 1.5, 1.5),   # current default
            (3, 100, 0.5, 2.0, 1.0),  # widest
            (5, 80, 0.6, 2.5, 1.0),
            (10, 50, 0.2, 1.5, 1.5),
            (3, 80, 0.4, 2.0, 1.5),
            (5, 60, 0.5, 2.5, 1.0),   # wide pullback + high TP
            (8, 100, 0.3, 2.0, 1.0),
            (3, 60, 0.6, 3.0, 1.0),   # max TP
        ]
    ],
    "supply_demand": [
        {"sr_proximity_min": sr, "min_engulfing_strength": eng, "require_htf_trend": htf,
         "min_rr_ratio": rr, "zone_lookback": zl}
        for sr, eng, htf, rr, zl in [
            (0.4, 0.1, True, 1.5, 30),
            (0.3, 0.1, False, 1.5, 30),
            (0.6, 0.15, True, 2.0, 30),  # current default
            (0.5, 0.1, True, 2.0, 50),
            (0.3, 0.05, False, 1.5, 50),
            (0.4, 0.1, True, 2.5, 40),
            (0.2, 0.05, False, 1.0, 30),  # loosest
            (0.5, 0.15, True, 2.0, 20),
            (0.3, 0.1, True, 2.0, 60),
            (0.4, 0.08, False, 1.5, 40),
        ]
    ],
    "ict_smc": [
        {"swing_lookback": sl, "sweep_min_pips": sp, "ob_max_age_bars": oa,
         "fvg_confluence": fvg, "min_rr_ratio": rr}
        for sl, sp, oa, fvg, rr in [
            (10, 3, 20, True, 1.5),
            (15, 3, 20, False, 1.5),
            (20, 5, 15, True, 2.0),   # current default
            (10, 2, 25, False, 1.0),  # very loose
            (15, 5, 15, True, 1.5),
            (10, 3, 25, True, 1.0),
            (30, 3, 10, False, 2.0),
            (20, 3, 20, False, 1.5),
            (10, 2, 30, False, 1.0),  # widest
            (15, 4, 20, True, 1.5),
        ]
    ],
    "structure_break": [
        {"structure_lookback": sl, "retest_tolerance_pips": rt, "require_rejection": rj,
         "htf_structure_confirm": htf, "min_rr_ratio": rr}
        for sl, rt, rj, htf, rr in [
            (20, 15, True, True, 1.5),
            (20, 15, False, False, 1.5),
            (30, 10, True, True, 2.0),   # current default
            (15, 20, False, False, 1.0),  # loosest
            (20, 10, True, False, 1.5),
            (40, 8, True, True, 2.0),
            (15, 15, False, True, 1.5),
            (25, 12, True, False, 1.5),
            (10, 25, False, False, 1.0),  # very loose
            (30, 15, False, False, 1.5),
        ]
    ],
    "mean_reversion": [
        {"bb_window": bw, "bb_std": bs, "rsi_oversold": ro, "rsi_overbought": rb,
         "adx_max": adx, "min_rr_ratio": rr}
        for bw, bs, ro, rb, adx, rr in [
            (15, 1.5, 35, 65, 30, 1.0),   # loose BB + loose RSI
            (20, 2.0, 30, 70, 25, 1.5),   # current default
            (20, 1.8, 35, 65, 30, 1.0),
            (15, 2.0, 30, 70, 30, 1.0),
            (30, 2.5, 25, 75, 20, 2.0),   # tight BB/RSI
            (20, 1.5, 35, 65, 35, 1.0),   # loose everything
            (15, 2.0, 30, 70, 35, 1.5),
            (20, 2.0, 35, 65, 30, 1.0),   # widened RSI
            (25, 2.0, 30, 70, 25, 1.5),
            (15, 1.8, 30, 70, 30, 1.0),
        ]
    ],
}

def optimize(sid, grid):
    print(f"\n{'='*70}")
    print(f"  {sid.upper()}  ({len(grid)} combos)")
    print(f"{'='*70}")
    results = []
    for i, params in enumerate(grid):
        sys.stdout.write(f"\r  [{i+1}/{len(grid)}] Testing...")
        sys.stdout.flush()
        r = bt(sid, params)
        if "error" not in r:
            results.append((sc(r), params, r))
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n\n  {'Rank':>4} {'Trades':>6} {'WinRate':>7} {'Return':>8} {'PF':>6} {'MaxDD':>7} {'Sharpe':>7}")
    print(f"  {'-'*54}")
    for rank, (s, p, r) in enumerate(results[:5], 1):
        print(f"  #{rank:>3} {r['trades']:>6} {r['wr']:>6.1%} {r['ret']:>+7.1f}% {r['pf']:>5.2f} {r.get('dd','?'):>6}% {r.get('sharpe','?'):>6}")
        if rank <= 3:
            kv = ", ".join(f"{k}={v}" for k, v in p.items())
            print(f"       → {kv}")

    # Test #1 with AI
    if results and results[0][0] > -999:
        best = results[0]
        r_ai = bt(sid, best[1], ai=True)
        if "error" not in r_ai:
            print(f"\n  BEST + AI ON:  trades={r_ai['trades']}  WR={r_ai['wr']:.1%}  "
                  f"ret={r_ai['ret']:+.1f}%  PF={r_ai['pf']:.2f}")

    return results

if __name__ == "__main__":
    try:
        r = requests.get(f"{BASE}/api/backtest/strategies", timeout=5)
        print(f"Server connected — {len(r.json())} strategies")
    except:
        print("ERROR: Server not running"); sys.exit(1)

    ALL = {}
    for sid, grid in GRIDS.items():
        res = optimize(sid, grid)
        if res and res[0][0] > -999:
            ALL[sid] = {"params": res[0][1], "result": res[0][2], "score": res[0][0]}

    print(f"\n\n{'='*70}")
    print(f"  FINAL RANKINGS")
    print(f"{'='*70}")
    for sid, info in sorted(ALL.items(), key=lambda x: x[1]["score"], reverse=True):
        r = info["result"]
        print(f"\n  {sid:20s}  ret={r['ret']:+7.1f}%  WR={r['wr']:.1%}  PF={r['pf']:.2f}  "
              f"trades={r['trades']}  score={info['score']:.0f}")
        print(f"    → {json.dumps(info['params'])}")

    with open("scripts/optimization_results.json", "w") as f:
        json.dump(ALL, f, indent=2, default=str)
    print(f"\nSaved to scripts/optimization_results.json")
