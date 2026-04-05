"""
WaveTrader CLI

Usage
─────
# Single-timeframe demo (synthetic data):
  python cli.py --mode demo

# Single-timeframe with real data in data/:
  python cli.py --mode demo --data data/

# Train only:
  python cli.py --mode train --epochs 50

# Backtest a saved checkpoint:
  python cli.py --mode backtest --balance 25000

# Multi-timeframe (recommended):
  python cli.py --mode mtf --pair "GBP/JPY" --epochs 30

Real data instructions
──────────────────────
Drop any of the following file formats into a folder (default: data/):
  • Dukascopy CSV  – e.g. GBPJPY_Candlestick_15_m_BID_01.01.2020-01.01.2024.csv
  • HistData CSV   – e.g. DAT_ASCII_GBPJPY_M15_2023.csv
  • MT4/MT5 CSV    – exported from History Centre
  • Any OHLCV CSV with date/open/high/low/close/volume columns

Then run:
  python cli.py --mode demo --data data/

See wavetrader/data.py for full loader documentation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Load .env file if present (before any imports that read env vars)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

import torch
import torch.utils.data as tud

import wavetrader as wt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="WaveTrader — Wave-Based Neural Trading Signal Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["train", "backtest", "demo", "mtf", "preprocess", "live",
                 "add-user", "remove-user", "list-users", "user-status"],
        default="demo",
        help="train | backtest | demo | mtf | preprocess | live | add-user | remove-user | list-users | user-status",
    )
    p.add_argument("--pair",      default="GBP/JPY",  help="Forex pair")
    p.add_argument("--timeframe", default="15min",    help="Entry timeframe")
    p.add_argument("--epochs",    type=int,   default=30, help="Training epochs")
    p.add_argument("--balance",   type=float, default=10_000, help="Initial balance (USD)")
    p.add_argument(
        "--data", default="data",
        help="Directory containing real CSV files (Dukascopy / HistData / MT4)",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a saved .pt checkpoint to load before backtest",
    )
    p.add_argument(
        "--paper", action="store_true", default=True,
        help="Paper trading mode (no real orders) — default for safety",
    )
    p.add_argument(
        "--live-trading", action="store_true", default=False,
        help="Enable real order execution (requires OANDA live account)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing / data-QC mode
# ─────────────────────────────────────────────────────────────────────────────

_PAIRS = ["GBP/JPY", "USD/JPY", "EUR/JPY", "GBP/USD"]
_TFS   = ["15min", "1h", "4h", "1d"]


def run_preprocess(args: argparse.Namespace) -> None:
    """
    Data quality pipeline:
      1. Load all four pairs × four timeframes from data_dir
      2. Apply filter_flash_crashes + detect_gaps to each 15m DataFrame
      3. Verify cross-pair timestamp alignment
      4. Save cleaned DataFrames as <PAIR>_<TF>_clean.parquet
      5. Print a QC summary table
    """
    import pandas as pd
    from pathlib import Path

    data_dir = Path(args.data)
    print("\n" + "=" * 70)
    print("WAVETRADER  DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Data directory : {data_dir.resolve()}")
    print()

    all_15m: dict = {}            # pair → cleaned 15m df (for alignment check)
    qc_rows: list = []            # summary table rows

    for pair in _PAIRS:
        pair_tag = pair.replace("/", "")
        print(f"{'─' * 64}")
        print(f"  {pair}")
        for tf in _TFS:
            raw = wt.load_forex_data(pair, tf, data_dir=data_dir)
            n_raw = len(raw)

            if n_raw == 0:
                print(f"    {tf:>5}  [SKIP] no data found")
                qc_rows.append({"pair": pair, "tf": tf, "raw": 0, "removed": 0, "gaps": 0})
                continue

            clean = wt.preprocess_pipeline(raw, pair=pair, timeframe=tf)
            n_removed = n_raw - len(clean[~clean.get("gap_before", False) if "gap_before" in clean else clean.index])
            n_cleaned = len(clean)
            n_flash   = n_raw - n_cleaned + (clean["gap_before"].sum() if "gap_before" in clean.columns else 0)
            n_gaps    = int(clean["gap_before"].sum()) if "gap_before" in clean.columns else 0
            n_flash   = n_raw - n_cleaned

            date_range = (
                f"{raw['date'].min().date()} → {raw['date'].max().date()}"
                if n_raw > 0 else "N/A"
            )
            print(
                f"    {tf:>5}  {n_raw:>8,} bars  [{date_range}]"
                f"  flash-removed={n_flash}  gaps={n_gaps}"
            )

            # Save cleaned parquet
            out_path = data_dir / f"{pair_tag}_{tf.replace('min','m')}_clean.parquet"
            # Drop the helper column before saving
            save_df = clean.drop(columns=["gap_before"], errors="ignore")
            save_df.to_parquet(out_path, index=False)

            if tf == "15min":
                all_15m[pair] = clean

            qc_rows.append({
                "pair": pair, "tf": tf,
                "raw": n_raw, "removed": n_flash, "gaps": n_gaps,
            })

    # Cross-pair alignment check
    print()
    print("Cross-pair timestamp alignment (15min):")
    if len(all_15m) >= 2:
        aligned = wt.verify_session_alignment(all_15m, timeframe="15min")
        status  = "OK \u2713" if aligned else "MISALIGNED \u2717  (fix DST before training)"
        print(f"  {status}")
    else:
        print("  (fewer than 2 pairs loaded — skipped)")

    # QC summary table
    print()
    print("QC Summary")
    print(f"  {'Pair':<10} {'TF':>5}  {'Raw':>10}  {'Flash-rm':>10}  {'Gaps':>6}")
    print(f"  {'─'*10} {'─'*5}  {'─'*10}  {'─'*10}  {'─'*6}")
    for r in qc_rows:
        print(
            f"  {r['pair']:<10} {r['tf']:>5}  "
            f"{r['raw']:>10,}  {r['removed']:>10,}  {r['gaps']:>6,}"
        )

    print()
    print("Cleaned Parquet files written to:", data_dir.resolve())
    print("(suffix: _clean.parquet)")
    print()
    print("Next step:")
    print("  python cli.py --mode train --data", args.data)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe mode
# ─────────────────────────────────────────────────────────────────────────────

def run_mtf(args: argparse.Namespace, device: torch.device) -> None:
    print("\n" + "=" * 70)
    print("WAVETRADER  MULTI-TIMEFRAME")
    print("=" * 70)

    config = wt.MTFConfig(pair=args.pair, epochs=args.epochs)

    print(f"\nLoading multi-timeframe data for {args.pair} from '{args.data}'...")
    mtf_data = wt.load_mtf_data(args.pair, data_dir=args.data)
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df):,} bars")

    train_data, val_data, test_data = wt.chronological_split_mtf(mtf_data)

    model = wt.WaveTraderMTF(config)
    print(f"\nModel parameters: {model.count_parameters():,}")

    ckpt = args.checkpoint or "wavetrader_mtf_best.pt"

    if args.mode in ("train", "mtf"):
        train_ds = wt.MTFForexDataset(train_data, config, pair=args.pair)
        val_ds   = wt.MTFForexDataset(val_data,   config, pair=args.pair)
        train_loader = tud.DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, collate_fn=wt.mtf_collate_fn,
        )
        val_loader = tud.DataLoader(
            val_ds, batch_size=config.batch_size,
            collate_fn=wt.mtf_collate_fn,
        )
        history = wt.train_mtf_model(
            model, train_loader, val_loader, config, device, checkpoint=ckpt
        )
        print(f"\nBest val accuracy: {max(history['val_accuracy']):.2%}")

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-timeframe mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args: argparse.Namespace, device: torch.device) -> None:
    config = wt.SignalConfig(
        pair=args.pair,
        timeframe=args.timeframe,
        epochs=args.epochs,
    )

    print(f"\nLoading {args.pair} {args.timeframe} data from '{args.data}'...")
    df = wt.load_forex_data(args.pair, args.timeframe, data_dir=args.data)
    print(f"Total bars: {len(df):,}")

    if len(df) < config.lookback + 100:
        print("Not enough real data — falling back to synthetic (demo only).")
        df = wt.generate_synthetic_forex(35_000, args.pair)
        print(f"Synthetic bars generated: {len(df):,}")

    train_df, val_df, test_df = wt.chronological_split(df)
    print(f"Split  → Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    model = wt.FluxSignal(config)
    print(f"\nModel parameters: {model.count_parameters():,}")

    ckpt = args.checkpoint or "flux_signal_best.pt"

    if args.mode in ("train", "demo"):
        train_ds = wt.ForexDataset(train_df, config.lookback, pair=args.pair)
        val_ds   = wt.ForexDataset(val_df,   config.lookback, pair=args.pair)
        train_loader = tud.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True
        )
        val_loader = tud.DataLoader(val_ds, batch_size=config.batch_size)

        history = wt.train_model(
            model, train_loader, val_loader, config, device, checkpoint=ckpt
        )
        print(f"\nBest val accuracy: {max(history['val_accuracy']):.2%}")

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")

    if args.mode in ("backtest", "demo"):
        bt_config = wt.BacktestConfig(initial_balance=args.balance)
        results   = wt.run_backtest(model, test_df, config, bt_config, device)
        wt.print_equity_chart(results.equity_curve)

        print("\n" + "=" * 70)
        roi = (results.final_balance / args.balance - 1) * 100
        print(f"Starting Capital : ${args.balance:,.2f}")
        print(f"Ending Capital   : ${results.final_balance:,.2f}")
        print(f"Net P&L          : ${results.total_pnl:,.2f}")
        print(f"ROI              : {roi:.1f}%")
        print(f"Max Drawdown     : {results.max_drawdown:.1%}")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Live trading mode
# ─────────────────────────────────────────────────────────────────────────────

def run_live(args: argparse.Namespace, device: torch.device) -> None:
    """Launch the live streaming engine."""
    import logging
    from wavetrader.oanda import OANDAClient, OANDAConfig
    from wavetrader.streaming import StreamingEngine
    from wavetrader.monitor import Monitor, MonitorConfig
    from wavetrader.state import StateManager

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    paper = not args.live_trading
    print("\n" + "=" * 70)
    print("WAVETRADER  LIVE TRADING")
    print("=" * 70)
    print(f"  Pair:    {args.pair}")
    print(f"  Mode:    {'PAPER' if paper else '*** LIVE ***'}")
    print(f"  Balance: ${args.balance:,.2f}")
    print(f"  Device:  {device}")

    if not paper:
        print("\n  ⚠️  LIVE TRADING — real money at risk!")
        confirm = input("  Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("  Aborted.")
            return

    # Load model
    config = wt.MTFConfig(pair=args.pair)
    model = wt.WaveTraderMTF(config)
    print(f"\n  Model parameters: {model.count_parameters():,}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, weights_only=False, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        # Look for the latest checkpoint in checkpoints/
        ckpt_dir = "checkpoints"
        if os.path.isdir(ckpt_dir):
            subdirs = sorted(os.listdir(ckpt_dir))
            for d in reversed(subdirs):
                weights = os.path.join(ckpt_dir, d, "model_weights.pt")
                if os.path.exists(weights):
                    state = torch.load(weights, weights_only=False, map_location=device)
                    if "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"])
                    else:
                        model.load_state_dict(state)
                    print(f"  Auto-loaded: {weights}")
                    break

    # OANDA
    oanda = OANDAClient()
    account = oanda.get_account_summary()
    print(f"\n  OANDA Account: {oanda.config.account_id}")
    print(f"  Balance: {account.currency} {account.balance:,.2f}")
    print(f"  NAV:     {account.currency} {account.nav:,.2f}")

    # Monitor
    monitor_config = MonitorConfig.from_env()
    monitor = Monitor(monitor_config) if monitor_config.telegram_token else None

    # Backtest config for risk params
    bt_config = wt.BacktestConfig(initial_balance=args.balance)

    # Engine
    engine = StreamingEngine(
        model=model,
        oanda=oanda,
        pair=args.pair,
        config=config,
        bt_config=bt_config,
        checkpoint_dir=os.environ.get("CHECKPOINT_DIR", "data/live_checkpoints"),
        checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "100")),
        monitor=monitor,
        paper_trading=paper,
    )

    # Set up copy trading if users are registered
    data_dir = os.environ.get("DATA_DIR", "data")
    from wavetrader.copytrade import CopyTradeManager, UserRegistry
    registry = UserRegistry(data_dir)
    active_users = registry.get_active_users()
    if active_users:
        engine.copy_trade_mgr = CopyTradeManager(registry, pair=args.pair, monitor=monitor)
        print(f"\n  Copy trading: {len(active_users)} followers connected")
    else:
        print("\n  Copy trading: no followers (use --mode add-user to register)")

    print("\n  Warming up (fetching historical candles)...")
    engine.warmup()
    print("  Warmup complete — entering main loop\n")
    engine.run()


# ─────────────────────────────────────────────────────────────────────────────
# User management
# ─────────────────────────────────────────────────────────────────────────────

def run_add_user(args: argparse.Namespace) -> None:
    """Interactive: register a new copy-trade follower."""
    from wavetrader.copytrade import UserAccount, UserRegistry

    data_dir = os.environ.get("DATA_DIR", "data")
    registry = UserRegistry(data_dir)

    print("\n" + "=" * 70)
    print("WAVETRADER  ADD FOLLOWER")
    print("=" * 70)

    user_id = input("  User ID (e.g. user_001): ").strip()
    if not user_id:
        print("  Aborted — user ID required.")
        return

    name = input("  Display name: ").strip() or user_id
    api_key = input("  OANDA API key: ").strip()
    if not api_key:
        print("  Aborted — API key required.")
        return

    account_id = input("  OANDA Account ID (e.g. 101-001-12345678-001): ").strip()
    if not account_id:
        print("  Aborted — account ID required.")
        return

    env = input("  Environment [practice/live] (default: practice): ").strip() or "practice"
    risk = input("  Risk per trade [0.01 = 1%] (default: 0.01): ").strip()
    risk_val = float(risk) if risk else 0.01
    max_lot = input("  Max lot size (default: 1.0): ").strip()
    max_lot_val = float(max_lot) if max_lot else 1.0
    chat_id = input("  Telegram chat ID for notifications (optional): ").strip()

    user = UserAccount(
        user_id=user_id,
        name=name,
        oanda_api_key=api_key,
        oanda_account_id=account_id,
        oanda_environment=env,
        risk_per_trade=risk_val,
        max_lot_size=max_lot_val,
        telegram_chat_id=chat_id,
    )

    # Verify the credentials work
    from wavetrader.oanda import OANDAClient, OANDAConfig
    print("\n  Verifying OANDA credentials...")
    try:
        test_config = OANDAConfig(api_key=api_key, account_id=account_id, environment=env)
        test_client = OANDAClient(test_config)
        acct = test_client.get_account_summary()
        print(f"  ✓ Connected: {acct.currency} {acct.balance:,.2f} balance")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        confirm = input("  Add anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return

    try:
        registry.add_user(user)
        print(f"\n  ✓ User '{user_id}' ({name}) registered successfully!")
        print(f"    Environment: {env}")
        print(f"    Risk: {risk_val:.1%} per trade, max {max_lot_val} lots")
    except ValueError as e:
        print(f"\n  ✗ {e}")


def run_remove_user(args: argparse.Namespace) -> None:
    """Remove a registered follower."""
    from wavetrader.copytrade import UserRegistry

    data_dir = os.environ.get("DATA_DIR", "data")
    registry = UserRegistry(data_dir)

    users = registry.list_users()
    if not users:
        print("\n  No users registered.")
        return

    print("\n  Registered users:")
    for u in users:
        status = "✓ active" if u.enabled else "✗ paused"
        print(f"    {u.user_id} ({u.name}) — {status}")

    user_id = input("\n  User ID to remove: ").strip()
    if not user_id:
        return

    confirm = input(f"  Remove '{user_id}'? This cannot be undone. [y/N]: ").strip().lower()
    if confirm == "y":
        if registry.remove_user(user_id):
            print(f"  ✓ Removed '{user_id}'")
        else:
            print(f"  ✗ User '{user_id}' not found")


def run_list_users(args: argparse.Namespace) -> None:
    """List all registered followers."""
    from wavetrader.copytrade import UserRegistry

    data_dir = os.environ.get("DATA_DIR", "data")
    registry = UserRegistry(data_dir)

    users = registry.list_users()
    if not users:
        print("\n  No users registered.")
        print("  Use: python cli.py --mode add-user")
        return

    print("\n" + "=" * 70)
    print("WAVETRADER  REGISTERED FOLLOWERS")
    print("=" * 70)
    print(f"  {'ID':<15} {'Name':<15} {'Env':<10} {'Risk':>6} {'Trades':>7} {'PnL':>10} {'Status'}")
    print(f"  {'─'*15} {'─'*15} {'─'*10} {'─'*6} {'─'*7} {'─'*10} {'─'*8}")
    for u in users:
        status = "active" if u.enabled else "PAUSED"
        print(
            f"  {u.user_id:<15} {u.name:<15} {u.oanda_environment:<10} "
            f"{u.risk_per_trade:>5.1%} {u.total_trades:>7} "
            f"${u.total_pnl:>9.2f} {status}"
        )
    print(f"\n  Total: {len(users)} users ({sum(1 for u in users if u.enabled)} active)")


def run_user_status(args: argparse.Namespace) -> None:
    """Show live balance/status for all followers."""
    from wavetrader.copytrade import CopyTradeManager, UserRegistry

    data_dir = os.environ.get("DATA_DIR", "data")
    registry = UserRegistry(data_dir)
    mgr = CopyTradeManager(registry, pair=args.pair)

    users = registry.list_users()
    if not users:
        print("\n  No users registered.")
        return

    print("\n" + "=" * 70)
    print("WAVETRADER  FOLLOWER STATUS (live)")
    print("=" * 70)

    summaries = mgr.get_follower_summary()
    for s in summaries:
        status = "✓" if s["enabled"] else "✗"
        balance = f"${s['balance']:,.2f}" if isinstance(s.get("balance"), (int, float)) else s.get("balance", "N/A")
        print(f"  {status} {s['user_id']} ({s['name']})")
        print(f"    Balance: {balance}  |  Trades: {s['total_trades']}  |  PnL: ${s['total_pnl']:.2f}")
        if isinstance(s.get("open_trades"), int) and s["open_trades"] > 0:
            print(f"    Open trades: {s['open_trades']}")
        print()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if args.mode == "preprocess":
        run_preprocess(args)
    elif args.mode == "mtf":
        run_mtf(args, device)
    elif args.mode == "live":
        run_live(args, device)
    elif args.mode == "add-user":
        run_add_user(args)
    elif args.mode == "remove-user":
        run_remove_user(args)
    elif args.mode == "list-users":
        run_list_users(args)
    elif args.mode == "user-status":
        run_user_status(args)
    else:
        run_single(args, device)


if __name__ == "__main__":
    main()
