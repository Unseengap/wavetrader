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
            return;
        }

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

            tooltip.innerHTML = `
                <div style="margin-bottom:4px;color:#8b949e">
                    ${this.currentPair} · ${this.currentTF}
                </div>
                <div>O <b>${bar.open.toFixed(3)}</b>  H <b>${bar.high.toFixed(3)}</b></div>
                <div>L <b>${bar.low.toFixed(3)}</b>  C <b style="color:${color}">${bar.close.toFixed(3)}</b></div>
                <div style="color:${color};margin-top:2px">${change >= 0 ? '+' : ''}${change.toFixed(3)} (${changePct}%)</div>
            `;

            tooltip.style.display = 'block';
            const x = param.point.x;
            tooltip.style.left = (x > this.container.clientWidth / 2)
                ? `${x - 160}px`
                : `${x + 20}px`;
            tooltip.style.top = '10px';
        });
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
