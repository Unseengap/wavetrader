6. Backtesting
Final phase is simulated real-world trading using run_backtest(model, dataset).


[7]
6m
# ── Perform Trading Simulation ────────────────────────────────────────────────
from wavetrader.backtest import run_backtest
from wavetrader.config import BacktestConfig

bt_config = BacktestConfig(
    initial_balance        = 100.0,
    risk_per_trade         = 0.100,
    min_confidence         = 0.65,
    spread_pips            = 3.0,
    pip_value              = 10.0,
    leverage               = 30,
    atr_halt_multiplier    = 3.0,
    drawdown_reduce_threshold = 0.10,
)

print(f"Executing Backtesting Engine (Balance: ${bt_config.initial_balance:,.2f})...")

# run_backtest expects raw dataframes (dict for MTF, single df for single-TF)
backtest_data = test_mtf if (is_mtf or is_v2) else test_df
results = run_backtest(model, backtest_data, config, bt_config, device)

# ── Summary ───────────────────────────────────────────────────────────────────
roi = (results.final_balance / bt_config.initial_balance - 1) * 100
print()
print("=" * 60)
print(f"  BACKTEST SUMMARY")
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
print("=" * 60)
Executing Backtesting Engine (Balance: $100.00)...

======================================================================
BACKTEST: GBP/JPY  ['15min', '1h', '4h', '1d']
Initial Balance : $100.00
Risk per Trade  : 10.0%
======================================================================
  Bar   1000/42958  Trades:   67  Balance: $    442.59
  Bar   2000/42958  Trades:  321  Balance: $    475.46
  Bar   3000/42958  Trades:  818  Balance: $ 28,375.04
  Bar   4000/42958  Trades: 1004  Balance: $ 44,351.74
  Bar   5000/42958  Trades: 1258  Balance: $ 83,885.01
  Bar   6000/42958  Trades: 1488  Balance: $125,346.76
  Bar   7000/42958  Trades: 1703  Balance: $157,124.32
  Bar   8000/42958  Trades: 1833  Balance: $188,015.94
  Bar   9000/42958  Trades: 2030  Balance: $202,552.51
  Bar  10000/42958  Trades: 2246  Balance: $234,391.08
  Bar  11000/42958  Trades: 2409  Balance: $259,690.87
  Bar  12000/42958  Trades: 2546  Balance: $278,534.73
  Bar  13000/42958  Trades: 2736  Balance: $285,984.30
  Bar  14000/42958  Trades: 2872  Balance: $309,370.84
  Bar  15000/42958  Trades: 3041  Balance: $347,181.33
  Bar  16000/42958  Trades: 3171  Balance: $371,978.12
  Bar  17000/42958  Trades: 3482  Balance: $374,781.93
  Bar  18000/42958  Trades: 3583  Balance: $395,750.18
  Bar  19000/42958  Trades: 4016  Balance: $372,366.19
  Bar  20000/42958  Trades: 4189  Balance: $376,205.46
  Bar  21000/42958  Trades: 4279  Balance: $407,062.44
  Bar  22000/42958  Trades: 4369  Balance: $443,175.59
  Bar  23000/42958  Trades: 4469  Balance: $473,334.47
  Bar  24000/42958  Trades: 4570  Balance: $508,117.41
  Bar  25000/42958  Trades: 4625  Balance: $548,372.90
  Bar  26000/42958  Trades: 4680  Balance: $583,507.40
  Bar  27000/42958  Trades: 4767  Balance: $615,497.80
  Bar  28000/42958  Trades: 4818  Balance: $645,351.14
  Bar  29000/42958  Trades: 4877  Balance: $659,812.42
  Bar  30000/42958  Trades: 4915  Balance: $682,006.20
  Bar  31000/42958  Trades: 4971  Balance: $716,147.09
  Bar  32000/42958  Trades: 5098  Balance: $733,622.00
  Bar  33000/42958  Trades: 5208  Balance: $742,877.00
  Bar  34000/42958  Trades: 5297  Balance: $769,789.83
  Bar  35000/42958  Trades: 5382  Balance: $790,226.54
  Bar  36000/42958  Trades: 5433  Balance: $826,553.53
  Bar  37000/42958  Trades: 5470  Balance: $850,501.96
  Bar  38000/42958  Trades: 5537  Balance: $880,504.48
  Bar  39000/42958  Trades: 5660  Balance: $934,732.80
  Bar  40000/42958  Trades: 5800  Balance: $983,577.97
  Bar  41000/42958  Trades: 5899  Balance: $1,007,519.78
  Bar  42000/42958  Trades: 5959  Balance: $1,039,208.70

======================================================================
BACKTEST RESULTS
======================================================================
Total Trades   : 6033
Winning        : 3067
Losing         : 2966
Win Rate       : 50.8%
----------------------------------------
Total P&L      : $1,250,158.92
Final Balance  : $1,061,886.19
Return         : 1061786.2%
----------------------------------------
Max Drawdown   : 36.2%
Profit Factor  : 1.79
Sharpe Ratio   : 2.89
======================================================================

Last 10 Trades:
----------------------------------------------------------------------
  SELL @ 211.012 → 211.223  Stop Loss     -$1050.77
  SELL @ 211.088 → 211.106  Stop Loss     -$92.49
  SELL @ 210.997 → 211.085  Stop Loss     -$440.76
  SELL @ 211.088 → 211.167  Stop Loss     -$395.97
  SELL @ 210.966 → 210.613  Stop Loss     +$1768.36
  BUY  @ 210.428 → 210.743  Stop Loss     +$1571.62
  BUY  @ 210.755 → 210.608  Stop Loss     -$730.22
  BUY  @ 210.610 → 210.659  Stop Loss     +$244.79
  BUY  @ 211.035 → 210.949  Stop Loss     -$434.94
  SELL @ 210.869 → 210.832  Stop Loss     +$183.92

============================================================
  BACKTEST SUMMARY
============================================================
  Starting Capital : $      100.00
  Ending Capital   : $1,061,886.19
  Net P&L          : $1,250,158.92
  ROI              : +1061786.19%
  Max Drawdown     : 36.21%
  Win Rate         : 50.84%
  Profit Factor    : 1.79
  Sharpe Ratio     : 2.89
  Total Trades     : 6033
  Winning / Losing : 3067 / 2966
============================================================
7. Full Trade Breakdown — Timeline, Win/Loss Ratios, Period Analysis
Build a complete trade log DataFrame from the backtest results and break down performance by day, week, month, and year. Save everything to CSV and Google Drive.


[8]
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
    rows.append({
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
    })

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
print()

# Show first and last 5 trades
print("── First 5 Trades ──")
display(trade_df.head())
print("\n── Last 5 Trades ──")
display(trade_df.tail())


[9]
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
================================================================================
  YEARLY BREAKDOWN
================================================================================
  2024  |  Trades:  2573  |  W/L: 1277/1296  |  WR: 49.6%  |  PnL: $ +353,987.98  |  PF: 1.61
  2025  |  Trades:  2886  |  W/L: 1474/1412  |  WR: 51.1%  |  PnL: $ +652,885.27  |  PF: 1.77
  2026  |  Trades:   574  |  W/L: 316/258  |  WR: 55.1%  |  PnL: $ +243,285.66  |  PF: 2.58

================================================================================
  MONTHLY BREAKDOWN
================================================================================
  2024-07  |  Trades:  378  |  W/L: 184/194  |  WR: 49%  |  PnL: $   +928.76  |  PF:  1.58
  2024-08  |  Trades:  649  |  W/L: 336/313  |  WR: 52%  |  PnL: $+62,856.55  |  PF:  1.51
  2024-09  |  Trades:  513  |  W/L: 237/276  |  WR: 46%  |  PnL: $+85,709.53  |  PF:  1.58
  2024-10  |  Trades:  364  |  W/L: 181/183  |  WR: 50%  |  PnL: $+87,792.67  |  PF:  1.83
  2024-11  |  Trades:  400  |  W/L: 196/204  |  WR: 49%  |  PnL: $+54,452.15  |  PF:  1.47
  2024-12  |  Trades:  269  |  W/L: 143/126  |  WR: 53%  |  PnL: $+62,248.32  |  PF:  1.75
  2025-01  |  Trades:  347  |  W/L: 168/179  |  WR: 48%  |  PnL: $+43,229.25  |  PF:  1.37
  2025-02  |  Trades:  311  |  W/L: 162/149  |  WR: 52%  |  PnL: $+68,999.01  |  PF:  1.86
  2025-03  |  Trades:  411  |  W/L: 201/210  |  WR: 49%  |  PnL: $+32,477.77  |  PF:  1.23
  2025-04  |  Trades:  567  |  W/L: 247/320  |  WR: 44%  |  PnL: $ +7,695.90  |  PF:  1.04
  2025-05  |  Trades:  214  |  W/L: 119/95   |  WR: 56%  |  PnL: $+79,392.53  |  PF:  2.47
  2025-06  |  Trades:  164  |  W/L:  86/78   |  WR: 52%  |  PnL: $+66,781.19  |  PF:  2.64
  2025-07  |  Trades:  131  |  W/L:  86/45   |  WR: 66%  |  PnL: $+89,363.69  |  PF:  4.84
  2025-08  |  Trades:  122  |  W/L:  71/51   |  WR: 58%  |  PnL: $+50,081.80  |  PF:  2.63
  2025-09  |  Trades:  101  |  W/L:  57/44   |  WR: 56%  |  PnL: $+39,811.88  |  PF:  2.82
  2025-10  |  Trades:  228  |  W/L: 119/109  |  WR: 52%  |  PnL: $+67,617.03  |  PF:  2.01
  2025-11  |  Trades:  195  |  W/L:  94/101  |  WR: 48%  |  PnL: $+46,839.06  |  PF:  1.86
  2025-12  |  Trades:   95  |  W/L:  64/31   |  WR: 67%  |  PnL: $+60,596.17  |  PF:  4.05
  2026-01  |  Trades:  182  |  W/L:  95/87   |  WR: 52%  |  PnL: $+88,402.84  |  PF:  2.76
  2026-02  |  Trades:  205  |  W/L: 113/92   |  WR: 55%  |  PnL: $+84,008.69  |  PF:  2.46
  2026-03  |  Trades:  169  |  W/L:  97/72   |  WR: 57%  |  PnL: $+67,520.46  |  PF:  2.58
  2026-04  |  Trades:   18  |  W/L:  11/7    |  WR: 61%  |  PnL: $ +3,353.67  |  PF:  2.05

================================================================================
  WEEKLY SUMMARY (92 weeks)
================================================================================
  Profitable weeks : 88 / 92  (95.7%)
  Best week PnL    : $  +41,175.72
  Worst week PnL   : $  -17,090.37
  Avg week PnL     : $  +13,588.68
  Median week PnL  : $  +13,748.93

================================================================================
  DAILY SUMMARY (504 trading days)
================================================================================
  Profitable days  : 398 / 504  (79.0%)
  Best day PnL     : $  +21,935.28
  Worst day PnL    : $  -14,696.58
  Avg day PnL      : $   +2,480.47
  Avg trades/day   : 12.0
/tmp/ipykernel_23683/500510850.py:28: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  daily = trade_df.groupby("exit_date").apply(period_stats).reset_index()
/tmp/ipykernel_23683/500510850.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  weekly = trade_df.groupby("exit_week").apply(period_stats).reset_index()
/tmp/ipykernel_23683/500510850.py:39: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  monthly = trade_df.groupby("exit_month").apply(period_stats).reset_index()
/tmp/ipykernel_23683/500510850.py:44: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  yearly = trade_df.groupby("exit_year").apply(period_stats).reset_index()

[10]
7s
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


[11]
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


[12]
3s
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
Saved trade_log.csv           (6033 trades)
Saved daily_breakdown.csv     (504 days)
Saved weekly_breakdown.csv    (92 weeks)
Saved monthly_breakdown.csv   (22 months)
Saved yearly_breakdown.csv    (3 years)
Saved equity_curve.csv        (33539 points)
Saved session_breakdown.csv   (4 sessions)

All results saved to: /content/drive/MyDrive/phase_lm/backtest_results/wavetrader_mtf_GBPJPY_20260404_235854

Files:
  daily_breakdown.csv                 95.2 KB
  equity_curve.csv                   597.1 KB
  monthly_breakdown.csv                4.7 KB
  session_breakdown.csv                0.3 KB
  trade_log.csv                     1501.2 KB
  weekly_breakdown.csv                20.0 KB
  yearly_breakdown.csv                 0.8 KB
8. Realistic Friction Simulation — What Would This Actually Return?
Apply real-world market friction to the trade log: random slippage (1-5 pips), dynamic spread widening during off-hours, and lot size caps at higher balances. Compare theoretical ceiling vs realistic returns.


[13]
2s
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


[14]
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

