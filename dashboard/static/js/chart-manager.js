/**
 * WaveTrader Chart Manager
 * Wraps TradingView Lightweight Charts for candlestick display
 * with trade markers, SL/TP price lines, and timeframe switching.
 */
class ChartManager {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.chart = null;
        this.candleSeries = null;
        this.volumeSeries = null;
        this.markers = [];
        this.priceLines = [];
        this.currentPair = 'GBP/JPY';
        this.currentTF = '1h';
        this._trades = [];
        this._tradeTimeIndex = {};
        this._init();
    }

    _init() {
        this.chart = LightweightCharts.createChart(this.container, {
            layout: {
                background: { type: 'solid', color: '#0d1117' },
                textColor: '#8b949e',
                fontSize: 11,
            },
            grid: {
                vertLines: { color: 'rgba(48, 54, 61, 0.4)' },
                horzLines: { color: 'rgba(48, 54, 61, 0.4)' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: 'rgba(88, 166, 255, 0.3)', width: 1 },
                horzLine: { color: 'rgba(88, 166, 255, 0.3)', width: 1 },
            },
            rightPriceScale: {
                borderColor: '#30363d',
                scaleMargins: { top: 0.1, bottom: 0.25 },
            },
            timeScale: {
                borderColor: '#30363d',
                timeVisible: true,
                secondsVisible: false,
            },
            handleScroll: { vertTouchDrag: false },
        });

        // Candlestick series
        this.candleSeries = this.chart.addCandlestickSeries({
            upColor: '#3fb950',
            downColor: '#f85149',
            borderUpColor: '#3fb950',
            borderDownColor: '#f85149',
            wickUpColor: '#3fb950',
            wickDownColor: '#f85149',
        });

        // Volume series (histogram pane)
        this.volumeSeries = this.chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });

        this.chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        // Responsive resize
        const ro = new ResizeObserver(() => {
            if (this.container.clientWidth > 0 && this.container.clientHeight > 0) {
                this.chart.applyOptions({
                    width: this.container.clientWidth,
                    height: this.container.clientHeight,
                });
            }
        });
        ro.observe(this.container);

        // Tooltip
        this._setupTooltip();
    }

    async loadCandles(pair, timeframe) {
        this.currentPair = pair;
        this.currentTF = timeframe;

        try {
            const params = new URLSearchParams({ pair, tf: timeframe, limit: '5000' });
            const resp = await fetch(`/api/data/candles?${params}`);
            const data = await resp.json();

            if (!data.candles || data.candles.length === 0) {
                return;
            }

            const candles = data.candles.map(c => ({
                time: c.time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
            }));

            const volumes = data.candles.map(c => ({
                time: c.time,
                value: c.volume,
                color: c.close >= c.open
                    ? 'rgba(63, 185, 80, 0.3)'
                    : 'rgba(248, 81, 73, 0.3)',
            }));

            this.candleSeries.setData(candles);
            this.volumeSeries.setData(volumes);
            this.chart.timeScale().fitContent();
        } catch (err) {
            console.error('Failed to load candles:', err);
        }
    }

    setTradeMarkers(trades) {
        if (!trades || trades.length === 0) {
            this.candleSeries.setMarkers([]);
            this._trades = [];
            this._tradeTimeIndex = {};
            return;
        }

        this._trades = trades;
        this._buildTradeTimeIndex(trades);

        const markers = [];

        trades.forEach((t, i) => {
            const isBuy = t.direction === 'BUY';
            const isWin = t.pnl > 0;

            // Entry marker
            if (t.entry_time) {
                const entryTs = Math.floor(new Date(t.entry_time).getTime() / 1000);
                markers.push({
                    time: entryTs,
                    position: isBuy ? 'belowBar' : 'aboveBar',
                    color: isBuy ? '#3fb950' : '#f85149',
                    shape: isBuy ? 'arrowUp' : 'arrowDown',
                    text: `${isBuy ? 'B' : 'S'}${i + 1}`,
                    size: 1,
                });
            }

            // Exit marker
            if (t.exit_time) {
                const exitTs = Math.floor(new Date(t.exit_time).getTime() / 1000);
                markers.push({
                    time: exitTs,
                    position: isBuy ? 'aboveBar' : 'belowBar',
                    color: isWin ? '#58a6ff' : '#d29922',
                    shape: 'circle',
                    text: `${isWin ? '+' : ''}$${t.pnl.toFixed(0)}`,
                    size: 1,
                });
            }
        });

        // Sort by time (required by LC)
        markers.sort((a, b) => a.time - b.time);
        this.candleSeries.setMarkers(markers);
    }

    setPriceLines(trades) {
        // Remove existing price lines
        this.priceLines.forEach(pl => {
            try { this.candleSeries.removePriceLine(pl); } catch (e) {}
        });
        this.priceLines = [];

        if (!trades || trades.length === 0) return;

        // Show SL/TP for the last few trades (max 5 to avoid clutter)
        const recent = trades.slice(-5);

        recent.forEach(t => {
            if (t.stop_loss) {
                const sl = this.candleSeries.createPriceLine({
                    price: t.stop_loss,
                    color: '#d29922',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'SL',
                });
                this.priceLines.push(sl);
            }

            if (t.take_profit) {
                const tp = this.candleSeries.createPriceLine({
                    price: t.take_profit,
                    color: '#58a6ff',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'TP',
                });
                this.priceLines.push(tp);
            }
        });
    }

    scrollToTrade(trade) {
        if (!trade || !trade.entry_time) return;
        const ts = Math.floor(new Date(trade.entry_time).getTime() / 1000);
        this.chart.timeScale().scrollToPosition(-10, false);
        // Use setVisibleRange to center on trade
        const range = 3600 * 24; // 1 day window
        this.chart.timeScale().setVisibleRange({
            from: ts - range,
            to: ts + range,
        });
    }

    _setupTooltip() {
        const tooltip = document.createElement('div');
        tooltip.className = 'wt-chart-tooltip';
        tooltip.style.cssText = `
            position: absolute; display: none; z-index: 50;
            background: #1c2128; border: 1px solid #30363d; border-radius: 6px;
            padding: 8px 10px; font-size: 11px; color: #e6edf3;
            pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        `;
        this.container.style.position = 'relative';
        this.container.appendChild(tooltip);

        this.chart.subscribeCrosshairMove(param => {
            if (!param.time || !param.seriesData) {
                tooltip.style.display = 'none';
                return;
            }

            const bar = param.seriesData.get(this.candleSeries);
            if (!bar) {
                tooltip.style.display = 'none';
                return;
            }

            const change = bar.close - bar.open;
            const changePct = ((change / bar.open) * 100).toFixed(3);
            const color = change >= 0 ? '#3fb950' : '#f85149';

            // Check if any trade active at this time
            let tradeInfo = '';
            const ts = typeof param.time === 'object' ? new Date(param.time.year, param.time.month - 1, param.time.day).getTime() / 1000 : param.time;
            const tradesAtTime = this._findTradesAtTime(ts);
            if (tradesAtTime.length > 0) {
                const t = tradesAtTime[0];
                const dirColor = t.trade.direction === 'BUY' ? '#3fb950' : '#f85149';
                const pnlColor = t.trade.pnl >= 0 ? '#3fb950' : '#f85149';
                tradeInfo = `
                    <div style="margin-top:4px;padding-top:4px;border-top:1px solid #30363d">
                        <span style="color:${dirColor};font-weight:700">${t.trade.direction}</span>
                        #${t.index + 1} ·
                        P&L: <b style="color:${pnlColor}">${t.trade.pnl >= 0 ? '+' : ''}$${t.trade.pnl.toFixed(2)}</b>
                    </div>
                `;
            }

            tooltip.innerHTML = `
                <div style="margin-bottom:4px;color:#8b949e">
                    ${this.currentPair} · ${this.currentTF}
                </div>
                <div>O <b>${bar.open.toFixed(3)}</b>  H <b>${bar.high.toFixed(3)}</b></div>
                <div>L <b>${bar.low.toFixed(3)}</b>  C <b style="color:${color}">${bar.close.toFixed(3)}</b></div>
                <div style="color:${color};margin-top:2px">${change >= 0 ? '+' : ''}${change.toFixed(3)} (${changePct}%)</div>
                ${tradeInfo}
            `;

            tooltip.style.display = 'block';
            const x = param.point.x;
            tooltip.style.left = (x > this.container.clientWidth / 2)
                ? `${x - 160}px`
                : `${x + 20}px`;
            tooltip.style.top = '10px';
        });

        // Click handler for chart → trade navigation
        this.chart.subscribeClick(param => {
            if (!param.time) return;
            const ts = typeof param.time === 'object' ? new Date(param.time.year, param.time.month - 1, param.time.day).getTime() / 1000 : param.time;
            const tradesAtTime = this._findTradesAtTime(ts);

            if (tradesAtTime.length > 0) {
                const t = tradesAtTime[0];
                this._showTradeTooltip(t, param.point);
            } else {
                this._hideTradeTooltip();
            }
        });
    }

    _buildTradeTimeIndex(trades) {
        this._tradeTimeIndex = {};
        trades.forEach((trade, i) => {
            if (trade.entry_time) {
                const entryTs = Math.floor(new Date(trade.entry_time).getTime() / 1000);
                if (!this._tradeTimeIndex[entryTs]) this._tradeTimeIndex[entryTs] = [];
                this._tradeTimeIndex[entryTs].push({ index: i, trade, type: 'entry' });
            }
            if (trade.exit_time) {
                const exitTs = Math.floor(new Date(trade.exit_time).getTime() / 1000);
                if (!this._tradeTimeIndex[exitTs]) this._tradeTimeIndex[exitTs] = [];
                this._tradeTimeIndex[exitTs].push({ index: i, trade, type: 'exit' });
            }
        });
    }

    _findTradesAtTime(timestamp) {
        // Look for exact match first, then within a range
        if (this._tradeTimeIndex[timestamp]) {
            return this._tradeTimeIndex[timestamp];
        }
        // Check within ±1 candle depending on timeframe
        const tfSeconds = { '15min': 900, '1h': 3600, '4h': 14400, '1d': 86400 };
        const range = tfSeconds[this.currentTF] || 3600;
        const results = [];
        for (let t = timestamp - range; t <= timestamp + range; t++) {
            if (this._tradeTimeIndex[t]) {
                results.push(...this._tradeTimeIndex[t]);
            }
        }
        return results;
    }

    _showTradeTooltip(tradeInfo, point) {
        const tooltip = document.getElementById('chart-trade-tooltip');
        if (!tooltip) return;

        const t = tradeInfo.trade;
        const dirClass = t.direction === 'BUY' ? 'buy' : 'sell';
        const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';

        tooltip.innerHTML = `
            <div class="tooltip-dir ${dirClass}">${t.direction} #${tradeInfo.index + 1}</div>
            <div>Entry: ${t.entry_price ? t.entry_price.toFixed(3) : '—'} → Exit: ${t.exit_price ? t.exit_price.toFixed(3) : '—'}</div>
            <div>SL: ${t.stop_loss ? t.stop_loss.toFixed(3) : '—'} · TP: ${t.take_profit ? t.take_profit.toFixed(3) : '—'}</div>
            <div class="tooltip-pnl ${pnlClass}">P&L: ${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}</div>
            <div style="font-size:0.7rem;color:var(--wt-text-muted);margin-top:0.2rem">
                ${t.exit_reason || ''} · Click to view in Trade Log
            </div>
        `;

        // Position near the chart point
        const chartRect = this.container.getBoundingClientRect();
        tooltip.style.display = 'block';
        tooltip.style.left = `${chartRect.left + point.x + 15}px`;
        tooltip.style.top = `${chartRect.top + point.y - 20}px`;

        // Click tooltip → navigate to trade in Trade Log
        tooltip.onclick = () => {
            if (typeof scrollToTradeRow === 'function') {
                scrollToTradeRow(tradeInfo.index);
            }
            this._hideTradeTooltip();
        };

        // Auto-hide after 5s
        clearTimeout(this._tooltipTimer);
        this._tooltipTimer = setTimeout(() => this._hideTradeTooltip(), 5000);
    }

    _hideTradeTooltip() {
        const tooltip = document.getElementById('chart-trade-tooltip');
        if (tooltip) tooltip.style.display = 'none';
    }

    destroy() {
        if (this.chart) {
            this.chart.remove();
            this.chart = null;
        }
    }
}

// Export for use
window.ChartManager = ChartManager;
