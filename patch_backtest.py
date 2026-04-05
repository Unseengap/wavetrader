import re

with open("wavetrader/backtest.py", "r") as f:
    text = f.read()

# Replace run_backtest to gracefully handle MTF datasets
old = """def run_backtest(
    model:     Any,
    df:        pd.DataFrame,
    config:    SignalConfig,
    bt_config: Optional[BacktestConfig] = None,
    device:    Optional[torch.device]   = None,
) -> BacktestResults:
    \"\"\"
    Run a full bar-by-bar backtest on `df` using `model` for signal generation.

    The model is run in inference mode (no gradients).  One position at a time.
    Each bar: update open trade → generate signal if flat → open if above threshold.
    \"\"\"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bt_config = bt_config or BacktestConfig()
    engine    = BacktestEngine(bt_config)
    dataset   = ForexDataset(df, lookback=config.lookback, lookahead=10, pair=config.pair)

    model = model.to(device)
    model.eval()

    print("\\n" + "=" * 70)
    print(f"BACKTEST: {config.pair}  {config.timeframe}")
    print(f"Initial Balance : ${bt_config.initial_balance:,.2f}")
    print(f"Risk per Trade  : {bt_config.risk_per_trade:.1%}")
    print("=" * 70)

    with torch.no_grad():
        for i in range(len(dataset)):
            actual      = dataset.valid_indices[i]
            current_bar = dataset.df.iloc[actual]
            timestamp   = (
                current_bar["date"]
                if "date" in current_bar.index
                else datetime.utcnow()
            )

            # Register bar range for volatility circuit breaker every bar
            engine.record_bar(current_bar["high"], current_bar["low"])

            if engine.open_trade:
                engine.update_trade(
                    current_bar["high"],
                    current_bar["low"],
                    current_bar["close"],
                    timestamp,
                )

            if engine.open_trade is None:
                sample = dataset[i]
                model_input = {
                    k: v.unsqueeze(0).to(device)
                    for k, v in sample.items()
                    if k != "label"
                }
                out      = model(model_input)
                sig_idx  = out["signal_logits"].argmax(-1).item()
                conf     = out["confidence"].item()
                risk     = out["risk_params"][0]

                if sig_idx != Signal.HOLD.value and conf >= bt_config.min_confidence:
                    trade_signal = TradeSignal(
                        signal=Signal(sig_idx),
                        confidence=conf,
                        entry_price=current_bar["close"],
                        stop_loss=float(risk[0].item() * 30 + 15),
                        take_profit=float(risk[1].item() * 60 + 30),
                        trailing_stop_pct=float(risk[2].item() * 0.3),
                        timestamp=timestamp,
                    )
                    engine.open_position(
                        trade_signal,
                        current_bar["close"],
                        timestamp,
                        current_high=current_bar["high"],
                        current_low=current_bar["low"],
                    )

            if (i + 1) % 1000 == 0:
                print(
                    f"  Bar {i+1:>6}/{len(dataset)}  "
                    f"Trades: {len(engine.closed_trades):>4}  "
                    f"Balance: ${engine.balance:>10,.2f}"
                )

    if engine.open_trade:
        last = dataset.df.iloc[-1]
        engine.close_position(
            last["close"],
            last["date"] if "date" in last.index else datetime.utcnow(),
            "End of Backtest",
        )

    results = engine.get_results()
    _print_results(results, bt_config.initial_balance)
    return results"""

new = """def run_backtest(
    model:     Any,
    df:        Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    config:    Union[SignalConfig, 'MTFConfig'],
    bt_config: Optional[BacktestConfig] = None,
    device:    Optional[torch.device]   = None,
) -> BacktestResults:
    \"\"\"
    Run a full bar-by-bar backtest on `df` using `model` for signal generation.
    Supports both single-timeframe and multi-timeframe backtesting.
    \"\"\"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bt_config = bt_config or BacktestConfig()
    engine    = BacktestEngine(bt_config)
    
    if hasattr(config, 'timeframes'):
        from .dataset import MTFForexDataset
        dataset = MTFForexDataset(df, config, lookahead=10, pair=config.pair)
        base_df = dataset.prepared[dataset.entry_tf]
        is_mtf = True
    else:
        dataset = ForexDataset(df, lookback=config.lookback, lookahead=10, pair=config.pair)
        base_df = dataset.df
        is_mtf = False

    model = model.to(device)
    model.eval()

    print("\\n" + "=" * 70)
    print(f"BACKTEST: {config.pair}  {getattr(config, 'timeframe', getattr(config, 'timeframes', 'MTF'))}")
    print(f"Initial Balance : ${bt_config.initial_balance:,.2f}")
    print(f"Risk per Trade  : {bt_config.risk_per_trade:.1%}")
    print("=" * 70)

    with torch.no_grad():
        for i in range(len(dataset)):
            actual      = dataset.valid_indices[i]
            current_bar = base_df.iloc[actual]
            timestamp   = (
                current_bar["date"]
                if "date" in current_bar.index
                else datetime.utcnow()
            )

            # Register bar range for volatility circuit breaker every bar
            engine.record_bar(current_bar["high"], current_bar["low"])

            if engine.open_trade:
                engine.update_trade(
                    current_bar["high"],
                    current_bar["low"],
                    current_bar["close"],
                    timestamp,
                )

            if engine.open_trade is None:
                sample = dataset[i]
                if is_mtf:
                    model_input = {
                        k: {feat: v.unsqueeze(0).to(device) for feat, v in val.items() if isinstance(v, torch.Tensor)} if isinstance(val, dict) else val.to(device) if isinstance(val, torch.Tensor) else val
                        for k, val in sample.items() if k != "label"
                    }
                else:
                    model_input = {
                        k: v.unsqueeze(0).to(device)
                        for k, v in sample.items()
                        if k != "label"
                    }
                
                out      = model(model_input)
                sig_idx  = out["signal_logits"].argmax(-1).item()
                conf     = out["confidence"].item()
                risk     = out["risk_params"][0]

                if sig_idx != Signal.HOLD.value and conf >= bt_config.min_confidence:
                    trade_signal = TradeSignal(
                        signal=Signal(sig_idx),
                        confidence=conf,
                        entry_price=current_bar["close"],
                        stop_loss=float(risk[0].item() * 30 + 15),
                        take_profit=float(risk[1].item() * 60 + 30),
                        trailing_stop_pct=float(risk[2].item() * 0.3),
                        timestamp=timestamp,
                    )
                    engine.open_position(
                        trade_signal,
                        current_bar["close"],
                        timestamp,
                        current_high=current_bar["high"],
                        current_low=current_bar["low"],
                    )

            if (i + 1) % 1000 == 0:
                print(
                    f"  Bar {i+1:>6}/{len(dataset)}  "
                    f"Trades: {len(engine.closed_trades):>4}  "
                    f"Balance: ${engine.balance:>10,.2f}"
                )

    if engine.open_trade:
        last = base_df.iloc[-1]
        engine.close_position(
            last["close"],
            last["date"] if "date" in last.index else datetime.utcnow(),
            "End of Backtest",
        )

    results = engine.get_results()
    _print_results(results, bt_config.initial_balance)
    return results"""

with open("wavetrader/backtest.py", "w") as f:
    text = text.replace(old, new)
    
    # We also need to import Dict and Union at the top since we used them
    if "from typing import" in text:
        text = text.replace("from typing import Any, Optional, Tuple, List, Callable", "from typing import Any, Optional, Tuple, List, Callable, Dict, Union")
    f.write(text)

