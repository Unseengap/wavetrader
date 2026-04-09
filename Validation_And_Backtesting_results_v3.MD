Backtesting
Final phase is simulated real-world trading using run_backtest(model, dataset).


[10]
2m
# ── Perform Trading Simulation ────────────────────────────────────────────────
from wavetrader.backtest import run_backtest
from wavetrader.config import BacktestConfig

bt_config = BacktestConfig(
    initial_balance        = 100.0,
    risk_per_trade         = 0.10,
    min_confidence         = 0.70,
    spread_pips            = 3.0,
    pip_value              = 10.0,
    leverage               = 30,
    atr_halt_multiplier    = 3.0,
    drawdown_reduce_threshold = 0.10,
)

print(f"Executing Backtesting Engine (Balance: ${bt_config.initial_balance:,.2f})...")
if is_v3:
    print(f"  V3 Mode: opposite-signal exit, trailing SL at {config.default_trailing_pct:.0%}, no TP")

# run_backtest expects raw dataframes (dict for MTF, single df for single-TF)
backtest_data = test_mtf if (is_v3 or is_mtf or is_v2) else test_df
results = run_backtest(model, backtest_data, config, bt_config, device)

# ── Summary ───────────────────────────────────────────────────────────────────
roi = (results.final_balance / bt_config.initial_balance - 1) * 100
print()
print("=" * 60)
print(f"  BACKTEST SUMMARY  ({arch_name})")
print("=" * 60)
print(f"  Starting Capital : ${bt_config.initial_balance:>12,.2f}")
print(f"  Ending Capital   : ${results.final_balance:>12,.2f}")
print(f"  Net P&L          : ${results.total_pnl:>12,.2f}")
print(f"  ROI              : {roi:>+.2f}%")
print(f"  Max Drawdown     : {results.max_drawdown:.2%}")
print(f"  Win Rate         : {results.win_rate:.2%}")
print(f"  Profit Factor    : {results.profit_factor:.2f}")
print(f"  Sharpe Ratio     : {results.sharpe_ratio:.2f}")
print(f"  Total Trades     : {results.total_trades}")
print(f"  Winning / Losing : {results.winning_trades} / {results.losing_trades}")

# V3-specific: exit reason breakdown
if is_v3 and results.trades:
    reasons = {}
    for t in results.trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print()
    print(f"  EXIT REASON BREAKDOWN:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / len(results.trades) * 100
        print(f"    {reason:20s}: {count:>5}  ({pct:.1f}%)")

print("=" * 60)
Executing Backtesting Engine (Balance: $100.00)...
  V3 Mode: opposite-signal exit, trailing SL at 40%, no TP
  V3 Label Distribution: BUY=200 (1.8%)  SELL=369 (3.4%)  HOLD=10272 (94.8%)

======================================================================
BACKTEST: GBP/JPY  ['1h', '4h', '1d']
Initial Balance : $100.00
Risk per Trade  : 10.0%
Exit Mode       : opposite_signal (no TP, trailing SL only)
======================================================================
  Bar   1000/10841  Trades:   36  Balance: $     69.70
  Bar   2000/10841  Trades:  114  Balance: $     96.48
  Bar   3000/10841  Trades:  155  Balance: $     95.70
  Bar   4000/10841  Trades:  198  Balance: $    167.66
  Bar   5000/10841  Trades:  241  Balance: $    200.15
  Bar   6000/10841  Trades:  294  Balance: $    305.87
  Bar   7000/10841  Trades:  321  Balance: $    337.96
  Bar   8000/10841  Trades:  344  Balance: $    365.06
  Bar   9000/10841  Trades:  391  Balance: $    416.74
  Bar  10000/10841  Trades:  452  Balance: $    457.23

======================================================================
BACKTEST RESULTS
======================================================================
Total Trades   : 476
Winning        : 258
Losing         : 218
Win Rate       : 54.2%
----------------------------------------
Total P&L      : $436.83
Final Balance  : $442.54
Return         : 342.5%
----------------------------------------
Max Drawdown   : 46.6%
Profit Factor  : 1.56
Sharpe Ratio   : 2.37
======================================================================

Last 10 Trades:
----------------------------------------------------------------------
  SELL @ 212.634 → 212.636  Max Hold      -$0.17
  SELL @ 212.621 → 212.712  Max Hold      -$6.37
  SELL @ 211.361 → 211.412  Max Hold      -$3.09
  SELL @ 211.397 → 211.210  Max Hold      +$11.22
  SELL @ 211.195 → 211.204  Max Hold      -$0.54
  SELL @ 211.189 → 211.379  Max Hold      -$11.37
  BUY  @ 210.923 → 210.791  Max Hold      -$7.98
  BUY  @ 210.805 → 210.873  Max Hold      +$4.02
  BUY  @ 210.887 → 210.868  Max Hold      -$1.17
  BUY  @ 210.883 → 211.164  Max Hold      +$16.86

============================================================
  BACKTEST SUMMARY  (WaveTraderMTFv3)
============================================================
  Starting Capital : $      100.00
  Ending Capital   : $      442.54
  Net P&L          : $      436.83
  ROI              : +342.54%
  Max Drawdown     : 46.64%
  Win Rate         : 54.20%
  Profit Factor    : 1.56
  Sharpe Ratio     : 2.37
  Total Trades     : 476
  Winning / Losing : 258 / 218

  EXIT REASON BREAKDOWN:
    Max Hold            :   476  (100.0%)
============================================================
7. Full Trade Breakdown — Timeline, Win/Loss Ratios, Period Analysis
Build a complete trade log DataFrame from the backtest results and break down performance by day, week, month, and year. Save everything to CSV and Google Drive.


[11]
0s
# ── Build Complete Trade Log ───────────────────────────────────────────────────
import pandas as pd
import numpy as np

trades = results.trades
print(f"Total closed trades: {len(trades)}")

# Build DataFrame from Trade objects
rows = []
for t in trades:
    direction = "BUY" if t.direction.value == 0 else "SELL"
    row = {
        "entry_time":   t.entry_time,
        "exit_time":    t.exit_time,
        "direction":    direction,
        "entry_price":  t.entry_price,
        "exit_price":   t.exit_price,
        "stop_loss":    t.stop_loss,
        "take_profit":  t.take_profit,
        "size_lots":    t.size,
        "pnl":          t.pnl,
        "exit_reason":  t.exit_reason,
        "is_winner":    t.pnl > 0,
    }
    # V3-specific fields
    if hasattr(t, 'exit_mode'):
        row["exit_mode"] = t.exit_mode
    rows.append(row)

trade_df = pd.DataFrame(rows)
trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"])
trade_df["exit_time"]  = pd.to_datetime(trade_df["exit_time"])
trade_df["cumulative_pnl"] = trade_df["pnl"].cumsum()
trade_df["trade_num"] = range(1, len(trade_df) + 1)

# Duration
trade_df["duration"] = trade_df["exit_time"] - trade_df["entry_time"]

# Running balance
trade_df["balance_after"] = bt_config.initial_balance + trade_df["cumulative_pnl"]

# Running win rate
trade_df["running_win_rate"] = trade_df["is_winner"].expanding().mean()

print(f"\nTrade log built: {len(trade_df)} trades from {trade_df['entry_time'].min()} to {trade_df['exit_time'].max()}")
print(f"Date span: {(trade_df['exit_time'].max() - trade_df['entry_time'].min()).days} days")

if is_v3:
    print(f"\nV3 Exit Modes:")
    print(f"  Opposite Signal exits: {(trade_df['exit_reason'] == 'Opposite Signal').sum()}")
    print(f"  Trailing SL exits    : {(trade_df['exit_reason'].str.contains('Trail', case=False, na=False)).sum()}")
    print(f"  Stop Loss exits      : {(trade_df['exit_reason'].str.contains('Stop Loss', case=False, na=False) & ~trade_df['exit_reason'].str.contains('Trail', case=False, na=False)).sum()}")
    avg_dur = trade_df["duration"].mean()
    print(f"  Avg hold duration    : {avg_dur}")

print()

# Show first and last 5 trades
print("── First 5 Trades ──")
display(trade_df.head())
print("\n── Last 5 Trades ──")
display(trade_df.tail())


[12]
0s
# ── Period Breakdown: Daily / Weekly / Monthly / Yearly ────────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Helper to compute stats for a group
def period_stats(group):
    wins   = group["is_winner"].sum()
    losses = len(group) - wins
    gross_profit = group.loc[group["pnl"] > 0, "pnl"].sum()
    gross_loss   = abs(group.loc[group["pnl"] <= 0, "pnl"].sum())
    return pd.Series({
        "trades":        len(group),
        "wins":          int(wins),
        "losses":        int(losses),
        "win_rate":      wins / max(len(group), 1),
        "gross_profit":  gross_profit,
        "gross_loss":    gross_loss,
        "net_pnl":       group["pnl"].sum(),
        "avg_pnl":       group["pnl"].mean(),
        "max_win":       group["pnl"].max(),
        "max_loss":      group["pnl"].min(),
        "profit_factor": gross_profit / max(gross_loss, 1e-9),
        "avg_duration":  group["duration"].mean(),
    })

# ── DAILY ──
trade_df["exit_date"] = trade_df["exit_time"].dt.date
daily = trade_df.groupby("exit_date").apply(period_stats).reset_index()
daily["exit_date"] = pd.to_datetime(daily["exit_date"])
daily["cumulative_pnl"] = daily["net_pnl"].cumsum()

# ── WEEKLY ──
trade_df["exit_week"] = trade_df["exit_time"].dt.to_period("W")
weekly = trade_df.groupby("exit_week").apply(period_stats).reset_index()
weekly["cumulative_pnl"] = weekly["net_pnl"].cumsum()

# ── MONTHLY ──
trade_df["exit_month"] = trade_df["exit_time"].dt.to_period("M")
monthly = trade_df.groupby("exit_month").apply(period_stats).reset_index()
monthly["cumulative_pnl"] = monthly["net_pnl"].cumsum()

# ── YEARLY ──
trade_df["exit_year"] = trade_df["exit_time"].dt.year
yearly = trade_df.groupby("exit_year").apply(period_stats).reset_index()
yearly["cumulative_pnl"] = yearly["net_pnl"].cumsum()

# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  YEARLY BREAKDOWN")
print("=" * 80)
for _, r in yearly.iterrows():
    print(f"  {int(r['exit_year'])}  |  Trades: {int(r['trades']):>5}  |  "
          f"W/L: {int(r['wins'])}/{int(r['losses'])}  |  WR: {r['win_rate']:.1%}  |  "
          f"PnL: ${r['net_pnl']:>+12,.2f}  |  PF: {r['profit_factor']:.2f}")

print()
print("=" * 80)
print("  MONTHLY BREAKDOWN")
print("=" * 80)
for _, r in monthly.iterrows():
    bar = "█" * max(1, int(abs(r['net_pnl']) / monthly['net_pnl'].abs().max() * 20))
    sign = "+" if r['net_pnl'] >= 0 else "-"
    color_code = "\033[92m" if r['net_pnl'] >= 0 else "\033[91m"
    print(f"  {str(r['exit_month']):>7}  |  Trades: {int(r['trades']):>4}  |  "
          f"W/L: {int(r['wins']):>3}/{int(r['losses']):<3}  |  WR: {r['win_rate']:.0%}  |  "
          f"PnL: ${r['net_pnl']:>+10,.2f}  |  PF: {r['profit_factor']:>5.2f}")

print()
print("=" * 80)
print(f"  WEEKLY SUMMARY ({len(weekly)} weeks)")
print("=" * 80)
print(f"  Profitable weeks : {(weekly['net_pnl'] > 0).sum()} / {len(weekly)}  "
      f"({(weekly['net_pnl'] > 0).mean():.1%})")
print(f"  Best week PnL    : ${weekly['net_pnl'].max():>+12,.2f}")
print(f"  Worst week PnL   : ${weekly['net_pnl'].min():>+12,.2f}")
print(f"  Avg week PnL     : ${weekly['net_pnl'].mean():>+12,.2f}")
print(f"  Median week PnL  : ${weekly['net_pnl'].median():>+12,.2f}")

print()
print("=" * 80)
print(f"  DAILY SUMMARY ({len(daily)} trading days)")
print("=" * 80)
print(f"  Profitable days  : {(daily['net_pnl'] > 0).sum()} / {len(daily)}  "
      f"({(daily['net_pnl'] > 0).mean():.1%})")
print(f"  Best day PnL     : ${daily['net_pnl'].max():>+12,.2f}")
print(f"  Worst day PnL    : ${daily['net_pnl'].min():>+12,.2f}")
print(f"  Avg day PnL      : ${daily['net_pnl'].mean():>+12,.2f}")
print(f"  Avg trades/day   : {daily['trades'].mean():.1f}")
/tmp/ipykernel_27396/500510850.py:28: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  daily = trade_df.groupby("exit_date").apply(period_stats).reset_index()
/tmp/ipykernel_27396/500510850.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  weekly = trade_df.groupby("exit_week").apply(period_stats).reset_index()
================================================================================
  YEARLY BREAKDOWN
================================================================================
  2024  |  Trades:   159  |  W/L: 81/78  |  WR: 50.9%  |  PnL: $      +24.93  |  PF: 1.16
  2025  |  Trades:   244  |  W/L: 141/103  |  WR: 57.8%  |  PnL: $     +366.79  |  PF: 1.94
  2026  |  Trades:    73  |  W/L: 36/37  |  WR: 49.3%  |  PnL: $      +45.12  |  PF: 1.18

================================================================================
  MONTHLY BREAKDOWN
================================================================================
  2024-07  |  Trades:   18  |  W/L:   9/9    |  WR: 50%  |  PnL: $    +10.87  |  PF:  1.82
  2024-08  |  Trades:   18  |  W/L:   7/11   |  WR: 39%  |  PnL: $    -38.29  |  PF:  0.28
  2024-09  |  Trades:   36  |  W/L:  20/16   |  WR: 56%  |  PnL: $    +12.30  |  PF:  1.43
  2024-10  |  Trades:   42  |  W/L:  23/19   |  WR: 55%  |  PnL: $    +19.94  |  PF:  1.93
  2024-11  |  Trades:   19  |  W/L:   9/10   |  WR: 47%  |  PnL: $     +1.13  |  PF:  1.06
  2024-12  |  Trades:   26  |  W/L:  13/13   |  WR: 50%  |  PnL: $    +18.99  |  PF:  2.07
  2025-01  |  Trades:   24  |  W/L:  15/9    |  WR: 62%  |  PnL: $    +21.48  |  PF:  2.04
  2025-02  |  Trades:   15  |  W/L:  11/4    |  WR: 73%  |  PnL: $    +37.91  |  PF:  6.23
  2025-03  |  Trades:   27  |  W/L:  16/11   |  WR: 59%  |  PnL: $    +19.11  |  PF:  1.71
  2025-04  |  Trades:   27  |  W/L:  15/12   |  WR: 56%  |  PnL: $    +24.91  |  PF:  1.43
  2025-05  |  Trades:   36  |  W/L:  21/15   |  WR: 58%  |  PnL: $    +95.26  |  PF:  3.35
  2025-06  |  Trades:    9  |  W/L:   6/3    |  WR: 67%  |  PnL: $    +16.27  |  PF:  2.23
  2025-07  |  Trades:   12  |  W/L:   8/4    |  WR: 67%  |  PnL: $    +21.99  |  PF:  1.81
  2025-08  |  Trades:   12  |  W/L:   7/5    |  WR: 58%  |  PnL: $    +17.56  |  PF:  1.72
  2025-09  |  Trades:    4  |  W/L:   3/1    |  WR: 75%  |  PnL: $     +0.01  |  PF:  1.00
  2025-10  |  Trades:   23  |  W/L:  13/10   |  WR: 57%  |  PnL: $    +45.49  |  PF:  2.09
  2025-11  |  Trades:   26  |  W/L:  12/14   |  WR: 46%  |  PnL: $    +40.47  |  PF:  1.70
  2025-12  |  Trades:   29  |  W/L:  14/15   |  WR: 48%  |  PnL: $    +26.34  |  PF:  1.41
  2026-01  |  Trades:   29  |  W/L:  16/13   |  WR: 55%  |  PnL: $    +29.60  |  PF:  1.38
  2026-02  |  Trades:   28  |  W/L:  14/14   |  WR: 50%  |  PnL: $    +23.77  |  PF:  1.22
  2026-03  |  Trades:   12  |  W/L:   4/8    |  WR: 33%  |  PnL: $    -19.98  |  PF:  0.60
  2026-04  |  Trades:    4  |  W/L:   2/2    |  WR: 50%  |  PnL: $    +11.73  |  PF:  2.28

================================================================================
  WEEKLY SUMMARY (66 weeks)
================================================================================
  Profitable weeks : 50 / 66  (75.8%)
  Best week PnL    : $      +38.98
  Worst week PnL   : $      -32.92
  Avg week PnL     : $       +6.62
  Median week PnL  : $       +4.63

================================================================================
  DAILY SUMMARY (140 trading days)
================================================================================
  Profitable days  : 84 / 140  (60.0%)
  Best day PnL     : $      +51.72
  Worst day PnL    : $      -22.90
  Avg day PnL      : $       +3.12
  Avg trades/day   : 3.4
/tmp/ipykernel_27396/500510850.py:39: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  monthly = trade_df.groupby("exit_month").apply(period_stats).reset_index()
/tmp/ipykernel_27396/500510850.py:44: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  yearly = trade_df.groupby("exit_year").apply(period_stats).reset_index()

[13]
1s
# ── Timeline Visualizations ────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axes = plt.subplots(4, 1, figsize=(16, 20))
fig.suptitle(f"Trade Analysis — {config.pair}", fontsize=16, fontweight="bold", y=0.98)

# ── 1. Cumulative PnL over time (by trade) ──
ax = axes[0]
colors = ["#4CAF50" if p > 0 else "#F44336" for p in trade_df["pnl"]]
ax.bar(trade_df["trade_num"], trade_df["pnl"], color=colors, width=1.0, alpha=0.5, label="Per-trade PnL")
ax2 = ax.twinx()
ax2.plot(trade_df["trade_num"], trade_df["cumulative_pnl"], color="#2196F3", lw=1.5, label="Cumulative PnL")
ax2.set_ylabel("Cumulative PnL ($)")
ax.set_xlabel("Trade #")
ax.set_ylabel("Individual PnL ($)")
ax.set_title("Per-Trade PnL & Cumulative Growth")
ax.grid(alpha=0.2)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# ── 2. Monthly Net PnL bar chart ──
ax = axes[1]
month_labels = [str(m) for m in monthly["exit_month"]]
monthly_colors = ["#4CAF50" if p >= 0 else "#F44336" for p in monthly["net_pnl"]]
bars = ax.bar(range(len(monthly)), monthly["net_pnl"], color=monthly_colors, edgecolor="white", lw=0.5)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("Net PnL ($)")
ax.set_title("Monthly Net PnL")
ax.grid(alpha=0.2, axis="y")

# ── 3. Rolling win rate (50-trade window) ──
ax = axes[2]
window = min(50, len(trade_df) // 5) or 10
rolling_wr = trade_df["is_winner"].rolling(window).mean() * 100
ax.plot(trade_df["trade_num"], rolling_wr, color="#FF9800", lw=1.2, label=f"{window}-trade rolling WR")
ax.axhline(50, linestyle="--", color="gray", alpha=0.6, label="50% baseline")
ax.fill_between(trade_df["trade_num"], 50, rolling_wr,
                where=rolling_wr >= 50, color="#4CAF50", alpha=0.15)
ax.fill_between(trade_df["trade_num"], 50, rolling_wr,
                where=rolling_wr < 50, color="#F44336", alpha=0.15)
ax.set_xlabel("Trade #")
ax.set_ylabel("Win Rate (%)")
ax.set_title(f"Rolling Win Rate ({window}-trade window)")
ax.legend()
ax.grid(alpha=0.2)
ax.set_ylim(0, 100)

# ── 4. Win/Loss by exit reason ──
ax = axes[3]
reason_stats = trade_df.groupby("exit_reason").agg(
    count=("pnl", "size"),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
    win_rate=("is_winner", "mean"),
).sort_values("count", ascending=True)

y_pos = range(len(reason_stats))
bar_colors = ["#4CAF50" if p >= 0 else "#F44336" for p in reason_stats["total_pnl"]]
ax.barh(y_pos, reason_stats["total_pnl"], color=bar_colors, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r}\n({int(c)} trades, {wr:.0%} WR)"
                    for r, c, wr in zip(reason_stats.index, reason_stats["count"], reason_stats["win_rate"])])
ax.set_xlabel("Total PnL ($)")
ax.set_title("PnL by Exit Reason")
ax.axvline(0, color="gray", lw=0.8)
ax.grid(alpha=0.2, axis="x")

plt.tight_layout()
plt.show()

# ── 5. Trade duration distribution ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

durations_hours = trade_df["duration"].dt.total_seconds() / 3600
axes[0].hist(durations_hours, bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
axes[0].set_xlabel("Duration (hours)")
axes[0].set_ylabel("Count")
axes[0].set_title("Trade Duration Distribution")
axes[0].grid(alpha=0.2)

# Win vs loss duration comparison
win_dur = durations_hours[trade_df["is_winner"]]
loss_dur = durations_hours[~trade_df["is_winner"]]
axes[1].hist(win_dur, bins=40, alpha=0.6, color="#4CAF50", label=f"Winners (avg {win_dur.mean():.1f}h)")
axes[1].hist(loss_dur, bins=40, alpha=0.6, color="#F44336", label=f"Losers (avg {loss_dur.mean():.1f}h)")
axes[1].set_xlabel("Duration (hours)")
axes[1].set_ylabel("Count")
axes[1].set_title("Duration: Winners vs Losers")
axes[1].legend()
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()

# ── 6. Buy vs Sell performance ──
print("\n" + "=" * 70)
print("  DIRECTION BREAKDOWN")
print("=" * 70)
for direction in ["BUY", "SELL"]:
    d = trade_df[trade_df["direction"] == direction]
    if len(d) == 0:
        continue
    wins = d["is_winner"].sum()
    gp = d.loc[d["pnl"] > 0, "pnl"].sum()
    gl = abs(d.loc[d["pnl"] <= 0, "pnl"].sum())
    print(f"  {direction:>4}  |  Trades: {len(d):>5}  |  W/L: {int(wins)}/{len(d)-int(wins)}  |  "
          f"WR: {wins/len(d):.1%}  |  Net: ${d['pnl'].sum():>+12,.2f}  |  "
          f"Avg: ${d['pnl'].mean():>+8,.2f}  |  PF: {gp/max(gl,1e-9):.2f}")
print("=" * 70)

# ── 7. Streak analysis ──
streaks = []
current_streak = 0
for w in trade_df["is_winner"]:
    if w:
        current_streak = max(0, current_streak) + 1
    else:
        current_streak = min(0, current_streak) - 1
    streaks.append(current_streak)

trade_df["streak"] = streaks
max_win_streak  = max(streaks)
max_loss_streak = min(streaks)

print(f"\n  Max winning streak : {max_win_streak} trades")
print(f"  Max losing streak  : {abs(max_loss_streak)} trades")
print(f"  Avg winning streak : {trade_df[trade_df['streak'] > 0]['streak'].mean():.1f}")
print(f"  Avg losing streak  : {abs(trade_df[trade_df['streak'] < 0]['streak'].mean()):.1f}")


[14]
1s
# ── Trading Sessions & Time-of-Day / Day-of-Week Analysis ─────────────────────
import matplotlib.pyplot as plt
import numpy as np

# ── Derive variables from config ──────────────────────────────────────────────
trade_pair = getattr(config, "pair", "GBP/JPY")
entry_tf   = getattr(config, "timeframes", ["15min"])[0] if hasattr(config, "timeframes") else getattr(config, "timeframe", "15min")
all_tfs    = getattr(config, "timeframes", [entry_tf])
lookbacks  = getattr(config, "lookback", 100)

# ── Compute hour/dow from entry_time ──────────────────────────────────────────
trade_df["entry_hour"] = trade_df["entry_time"].dt.hour
trade_df["entry_dow"]  = trade_df["entry_time"].dt.day_name()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"When Does the Model Trade Best?  —  {trade_pair} ({entry_tf} entries)",
             fontsize=14, fontweight="bold")

# ── 1. Trades by Hour of Day ──
ax = axes[0, 0]
hourly = trade_df.groupby("entry_hour").agg(
    trades=("pnl", "size"),
    net_pnl=("pnl", "sum"),
    win_rate=("is_winner", "mean"),
).reindex(range(24), fill_value=0)
bar_colors = ["#4CAF50" if p >= 0 else "#F44336" for p in hourly["net_pnl"]]
ax.bar(hourly.index, hourly["trades"], color=bar_colors, edgecolor="white", alpha=0.8)
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Number of Trades")
ax.set_title("Trade Count by Hour")
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.2, axis="y")
# Session overlays
ax.axvspan(0, 8, alpha=0.05, color="blue", label="Asia")
ax.axvspan(7, 16, alpha=0.05, color="green", label="London")
ax.axvspan(13, 22, alpha=0.05, color="orange", label="New York")
ax.legend(fontsize=8)

# ── 2. PnL by Hour of Day ──
ax = axes[0, 1]
bar_colors_pnl = ["#4CAF50" if p >= 0 else "#F44336" for p in hourly["net_pnl"]]
ax.bar(hourly.index, hourly["net_pnl"], color=bar_colors_pnl, edgecolor="white")
ax2 = ax.twinx()
ax2.plot(hourly.index, hourly["win_rate"] * 100, "o-", color="#FF9800", markersize=4, lw=1.5, label="Win Rate")
ax2.set_ylabel("Win Rate (%)")
ax2.set_ylim(0, 100)
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Net PnL ($)")
ax.set_title("Net PnL & Win Rate by Hour")
ax.set_xticks(range(0, 24, 2))
ax.axhline(0, color="gray", lw=0.8)
ax.grid(alpha=0.2)
ax2.legend(loc="upper right", fontsize=8)

# ── 3. Day-of-Week breakdown ──
ax = axes[1, 0]
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = trade_df.groupby("entry_dow").agg(
    trades=("pnl", "size"),
    net_pnl=("pnl", "sum"),
    win_rate=("is_winner", "mean"),
    avg_pnl=("pnl", "mean"),
).reindex(dow_order).dropna()

bar_colors_dow = ["#4CAF50" if p >= 0 else "#F44336" for p in dow["net_pnl"]]
x = range(len(dow))
ax.bar(x, dow["net_pnl"], color=bar_colors_dow, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([d[:3] for d in dow.index])
ax.set_ylabel("Net PnL ($)")
ax.set_title("Net PnL by Day of Week")
ax.axhline(0, color="gray", lw=0.8)
ax.grid(alpha=0.2, axis="y")

# ── 4. Win Rate by Day-of-Week ──
ax = axes[1, 1]
ax.bar(x, dow["win_rate"] * 100, color="#2196F3", edgecolor="white", alpha=0.8)
ax.axhline(50, linestyle="--", color="gray", alpha=0.6)
for i, (wr, n) in enumerate(zip(dow["win_rate"], dow["trades"])):
    ax.text(i, wr * 100 + 1.5, f"{int(n)}t", ha="center", fontsize=9, color="gray")
ax.set_xticks(x)
ax.set_xticklabels([d[:3] for d in dow.index])
ax.set_ylabel("Win Rate (%)")
ax.set_title("Win Rate by Day of Week (trade count shown)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.show()

# ── Print session breakdown ──
def _session(hour):
    if 0 <= hour < 8:   return "Asia (00-08)"
    elif 8 <= hour < 13: return "London (08-13)"
    elif 13 <= hour < 17: return "Overlap (13-17)"
    else:                return "New York (17-22)"

trade_df["session"] = trade_df["entry_hour"].apply(_session)
session_stats = trade_df.groupby("session").agg(
    trades=("pnl", "size"),
    net_pnl=("pnl", "sum"),
    win_rate=("is_winner", "mean"),
    avg_pnl=("pnl", "mean"),
)

print("\n" + "=" * 70)
print(f"  TRADING SESSION BREAKDOWN — {trade_pair}")
print("=" * 70)
for sess, r in session_stats.iterrows():
    print(f"  {sess:<20}  |  Trades: {int(r['trades']):>5}  |  "
          f"WR: {r['win_rate']:.1%}  |  Net: ${r['net_pnl']:>+12,.2f}  |  "
          f"Avg: ${r['avg_pnl']:>+8,.2f}")
print("=" * 70)

print(f"\n  Pair           : {trade_pair}")
print(f"  Entry TF       : {entry_tf}")
print(f"  Input TFs      : {' → '.join(all_tfs)}")
print(f"  Lookback Bars  : {lookbacks}")


[ ]
1m
# ── Save Everything to Google Drive ────────────────────────────────────────────
import os

SAVE_DIR = DRIVE_ROOT / "backtest_results" / CKPT_NAME
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Full trade log (includes pair, timeframes, session, hour, dow)
trade_export = trade_df.drop(columns=["exit_date", "exit_week", "exit_month", "exit_year", "streak"], errors="ignore")
trade_export.to_csv(SAVE_DIR / "trade_log.csv", index=False)
print(f"Saved trade_log.csv           ({len(trade_export)} trades)")

# 2. Daily breakdown
daily.to_csv(SAVE_DIR / "daily_breakdown.csv", index=False)
print(f"Saved daily_breakdown.csv     ({len(daily)} days)")

# 3. Weekly breakdown
weekly_export = weekly.copy()
weekly_export["exit_week"] = weekly_export["exit_week"].astype(str)
weekly_export.to_csv(SAVE_DIR / "weekly_breakdown.csv", index=False)
print(f"Saved weekly_breakdown.csv    ({len(weekly_export)} weeks)")

# 4. Monthly breakdown
monthly_export = monthly.copy()
monthly_export["exit_month"] = monthly_export["exit_month"].astype(str)
monthly_export.to_csv(SAVE_DIR / "monthly_breakdown.csv", index=False)
print(f"Saved monthly_breakdown.csv   ({len(monthly_export)} months)")

# 5. Yearly breakdown
yearly.to_csv(SAVE_DIR / "yearly_breakdown.csv", index=False)
print(f"Saved yearly_breakdown.csv    ({len(yearly)} years)")

# 6. Equity curve
eq_df = pd.DataFrame({"equity": results.equity_curve})
eq_df.to_csv(SAVE_DIR / "equity_curve.csv", index=False)
print(f"Saved equity_curve.csv        ({len(eq_df)} points)")

# 7. Session stats
session_stats.to_csv(SAVE_DIR / "session_breakdown.csv")
print(f"Saved session_breakdown.csv   ({len(session_stats)} sessions)")

print(f"\nAll results saved to: {SAVE_DIR}")
print(f"\nFiles:")
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(SAVE_DIR / f)
    print(f"  {f:<30s}  {size / 1024:>8.1f} KB")
8. Realistic Friction Simulation — What Would This Actually Return?
Apply real-world market friction to the trade log: random slippage (1-5 pips), dynamic spread widening during off-hours, and lot size caps at higher balances. Compare theoretical ceiling vs realistic returns.


[15]
1s
# ── Realistic Friction Simulation ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Friction parameters (tweak these to model your broker) ────────────────────
SLIPPAGE_RANGE     = (0.5, 3.0)   # Random slippage in pips per trade
SPREAD_BASE        = 3.0          # Base spread already in backtest (pips)
SPREAD_OFFHOURS    = 2.5          # Extra spread pips during Asia/low-liq hours
SPREAD_NEWS_PROB   = 0.05         # 5% chance any trade hits a news spike
SPREAD_NEWS_EXTRA  = 5.0          # Extra pips during news spikes
LOT_CAP            = 2.0          # Max lots regardless of balance (realistic retail)
LATENCY_MISS_RATE  = 0.03         # 3% of trades missed entirely (requotes, latency)
PIP_VALUE          = bt_config.pip_value  # USD per pip per lot

print("=" * 70)
print("  REALISTIC FRICTION SIMULATION")
print("=" * 70)
print(f"  Slippage       : {SLIPPAGE_RANGE[0]}-{SLIPPAGE_RANGE[1]} pips random")
print(f"  Off-hours spread: +{SPREAD_OFFHOURS} pips (Asia session)")
print(f"  News spike prob : {SPREAD_NEWS_PROB:.0%} at +{SPREAD_NEWS_EXTRA} pips")
print(f"  Lot cap         : {LOT_CAP} lots max")
print(f"  Latency misses  : {LATENCY_MISS_RATE:.0%} of trades skipped")
print("=" * 70)

# ── Apply friction to each trade ──────────────────────────────────────────────
realistic_rows = []
balance_theoretical = bt_config.initial_balance
balance_realistic   = bt_config.initial_balance

for _, t in trade_df.iterrows():
    # Theoretical: original PnL compounded
    balance_theoretical += t["pnl"]

    # ── Skip trade? (latency miss) ──
    if np.random.random() < LATENCY_MISS_RATE:
        realistic_rows.append({
            "trade_num":       t["trade_num"],
            "original_pnl":    t["pnl"],
            "slippage_cost":   0,
            "spread_cost":     0,
            "lot_penalty":     0,
            "adjusted_pnl":    0,
            "skipped":         True,
            "balance_theo":    balance_theoretical,
            "balance_real":    balance_realistic,
        })
        continue

    # ── Slippage (always against you) ──
    slippage_pips = np.random.uniform(*SLIPPAGE_RANGE)

    # ── Dynamic spread ──
    extra_spread = 0.0
    hour = t["entry_hour"]
    if hour < 7 or hour >= 21:  # Asia / low liquidity
        extra_spread += SPREAD_OFFHOURS
    if np.random.random() < SPREAD_NEWS_PROB:  # News spike
        extra_spread += SPREAD_NEWS_EXTRA

    # ── Lot cap penalty ──
    lot_ratio = min(1.0, LOT_CAP / max(t["size_lots"], 0.01))

    # ── Compute adjusted PnL ──
    friction_pips = slippage_pips + extra_spread
    friction_cost = friction_pips * PIP_VALUE * min(t["size_lots"], LOT_CAP)
    adjusted_pnl  = (t["pnl"] * lot_ratio) - friction_cost

    balance_realistic += adjusted_pnl

    realistic_rows.append({
        "trade_num":       t["trade_num"],
        "original_pnl":    t["pnl"],
        "slippage_cost":   slippage_pips * PIP_VALUE * min(t["size_lots"], LOT_CAP),
        "spread_cost":     extra_spread * PIP_VALUE * min(t["size_lots"], LOT_CAP),
        "lot_penalty":     t["pnl"] * (1 - lot_ratio),
        "adjusted_pnl":    adjusted_pnl,
        "skipped":         False,
        "balance_theo":    balance_theoretical,
        "balance_real":    balance_realistic,
    })

rf = pd.DataFrame(realistic_rows)
active = rf[~rf["skipped"]]

# ── Summary comparison ────────────────────────────────────────────────────────
theo_roi  = (balance_theoretical / bt_config.initial_balance - 1) * 100
real_roi  = (balance_realistic / bt_config.initial_balance - 1) * 100
real_annual = ((balance_realistic / bt_config.initial_balance) ** (12 / 21) - 1) * 100  # ~21 months

# Realistic win rate (adjusted PnL)
real_wins  = (active["adjusted_pnl"] > 0).sum()
real_total = len(active)
real_wr    = real_wins / max(real_total, 1)
real_gp    = active.loc[active["adjusted_pnl"] > 0, "adjusted_pnl"].sum()
real_gl    = abs(active.loc[active["adjusted_pnl"] <= 0, "adjusted_pnl"].sum())
real_pf    = real_gp / max(real_gl, 1e-9)

print()
print("=" * 70)
print(f"  {'METRIC':<25} {'THEORETICAL':>15} {'REALISTIC':>15}")
print("=" * 70)
print(f"  {'Trades Executed':<25} {len(trade_df):>15,} {real_total:>15,}")
print(f"  {'Trades Skipped':<25} {'0':>15} {rf['skipped'].sum():>15,}")
print(f"  {'Final Balance':<25} ${balance_theoretical:>14,.2f} ${balance_realistic:>14,.2f}")
print(f"  {'Total ROI':<25} {theo_roi:>+14.1f}% {real_roi:>+14.1f}%")
print(f"  {'Annualized ROI':<25} {'—':>15} {real_annual:>+14.1f}%")
print(f"  {'Win Rate':<25} {results.win_rate:>14.1%} {real_wr:>14.1%}")
print(f"  {'Profit Factor':<25} {results.profit_factor:>15.2f} {real_pf:>15.2f}")
print(f"  {'Avg Slippage Cost':<25} {'$0':>15} ${active['slippage_cost'].mean():>14,.2f}")
print(f"  {'Avg Spread Cost':<25} {'$0':>15} ${active['spread_cost'].mean():>14,.2f}")
print(f"  {'Total Friction Cost':<25} {'$0':>15} ${(active['slippage_cost'].sum() + active['spread_cost'].sum()):>14,.2f}")
print("=" * 70)

# ── Can you still make big gains? ──
print()
print("  ┌─────────────────────────────────────────────────────────┐")
if real_roi > 500:
    verdict = "YES — even with full friction this is a high-performance system"
elif real_roi > 100:
    verdict = "YES — strong returns that beat most professional funds"
elif real_roi > 30:
    verdict = "SOLID — competitive with top-tier prop traders"
else:
    verdict = "MODEST — edge exists but friction eats most of the profit"
print(f"  │  {verdict:^55} │")
print(f"  │  Realistic {real_annual:+.0f}% annualized on ${bt_config.initial_balance:,.0f} start{'':>10} │")
print(f"  │  = ${bt_config.initial_balance * (1 + real_annual/100):>12,.0f} after 1 year{'':>19} │")
print(f"  │  = ${bt_config.initial_balance * (1 + real_annual/100)**2:>12,.0f} after 2 years{'':>18} │")
print(f"  │  = ${bt_config.initial_balance * (1 + real_annual/100)**3:>12,.0f} after 3 years{'':>18} │")
print("  └─────────────────────────────────────────────────────────┘")

# ── Plot: Theoretical vs Realistic equity curves ──────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

ax = axes[0]
ax.plot(rf["trade_num"], rf["balance_theo"], lw=1.2, color="#2196F3", alpha=0.6, label="Theoretical (no friction)")
ax.plot(rf["trade_num"], rf["balance_real"], lw=1.5, color="#4CAF50", label="Realistic (with friction)")
ax.axhline(bt_config.initial_balance, ls="--", color="gray", alpha=0.5)
ax.set_xlabel("Trade #")
ax.set_ylabel("Balance ($)")
ax.set_title("Theoretical vs Realistic Equity Curve")
ax.legend()
ax.grid(alpha=0.2)

# Log scale version
ax = axes[1]
ax.plot(rf["trade_num"], rf["balance_theo"], lw=1.2, color="#2196F3", alpha=0.6, label="Theoretical")
ax.plot(rf["trade_num"], rf["balance_real"], lw=1.5, color="#4CAF50", label="Realistic")
ax.axhline(bt_config.initial_balance, ls="--", color="gray", alpha=0.5)
ax.set_yscale("log")
ax.set_xlabel("Trade #")
ax.set_ylabel("Balance ($) — log scale")
ax.set_title("Log Scale — Shows True Growth Rate (Straight = Consistent Compounding)")
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# ── Friction breakdown pie chart ──
fig, ax = plt.subplots(figsize=(6, 6))
friction_totals = {
    "Slippage":       active["slippage_cost"].sum(),
    "Extra Spread":   active["spread_cost"].sum(),
    "Lot Cap Loss":   active["lot_penalty"].sum(),
}
labels = [f"{k}\n${v:,.0f}" for k, v in friction_totals.items() if v > 0]
values = [v for v in friction_totals.values() if v > 0]
ax.pie(values, labels=labels, autopct="%1.1f%%", colors=["#FF9800", "#F44336", "#9C27B0"])
ax.set_title("Where Friction Costs Come From")
plt.show()


[16]
0s
# ── Plot Equity Curve & Sample Predictions ───────────────────────────────────
import matplotlib.pyplot as plt

# 1. Equity Curve
fig, ax = plt.subplots(figsize=(14, 4))
equity_curve = results.equity_curve

ax.plot(equity_curve, lw=1.2, color="#4CAF50")
ax.axhline(bt_config.initial_balance, linestyle="--", color="gray", alpha=0.7, label="Starting Capital")

ax.fill_between(range(len(equity_curve)), bt_config.initial_balance, equity_curve,
                where=[e > bt_config.initial_balance for e in equity_curve],
                color="#4CAF50", alpha=0.15)
ax.fill_between(range(len(equity_curve)), bt_config.initial_balance, equity_curve,
                where=[e < bt_config.initial_balance for e in equity_curve],
                color="#F44336", alpha=0.15)

ax.set_title("Equity Curve — Holdout Backtest")
ax.set_xlabel("Bars Passed")
ax.set_ylabel("Balance (USD)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Sample Trades Log (from backtest trade_df)
print("\nSample Trades (first 15 executed trades):")
print(f"  {'Dir':>5}  {'Entry':>10}  {'Exit':>10}  {'PnL':>10}  {'Reason':>14}  {'Win':>4}")
print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*4}")
for _, row in trade_df.head(15).iterrows():
    win = "✓" if row["is_winner"] else "✗"
    print(f"  {row['direction']:>5}  {row['entry_price']:>10.3f}  {row['exit_price']:>10.3f}  "
          f"${row['pnl']:>+9.2f}  {row['exit_reason']:>14}  {win:>4}")

Colab paid products - Cancel contracts here
