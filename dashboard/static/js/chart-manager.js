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
        this._tradeRangeSeries = [];  // SL/TP shaded area series
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

        // Click handler for chart → navigate to trade in Trade History
        this.chart.subscribeClick(param => {
            if (!param.time) return;
            const ts = typeof param.time === 'object' ? new Date(param.time.year, param.time.month - 1, param.time.day).getTime() / 1000 : param.time;
            const tradesAtTime = this._findTradesAtTime(ts);

            if (tradesAtTime.length > 0) {
                const t = tradesAtTime[0];
                // Show SL/TP range zones on chart
                this.showTradeRangeZones(t.trade, ts);
                // Navigate to trade in Trade History tab
                if (typeof navigateToTradeHistory === 'function') {
                    navigateToTradeHistory(t.index);
                }
            } else {
                this.clearTradeRangeZones();
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

    /**
     * Show SL/TP as colored range zones (like the trading image: green TP zone, red SL zone)
     * with an entry price line. Uses line series with shaded areas.
     */
    showTradeRangeZones(trade, clickTime) {
        this.clearTradeRangeZones();

        const isBuy = trade.direction === 'BUY';
        const entry = trade.entry_price || trade.price || 0;
        const sl = trade.stop_loss || trade.sl || null;
        const tp = trade.take_profit || trade.tp || null;

        if (!entry) return;

        // Determine time range for the zones
        const entryTs = trade.entry_time
            ? Math.floor(new Date(trade.entry_time).getTime() / 1000)
            : (trade.open_time ? Math.floor(new Date(trade.open_time).getTime() / 1000) : clickTime);
        const exitTs = trade.exit_time
            ? Math.floor(new Date(trade.exit_time).getTime() / 1000)
            : (trade.close_time ? Math.floor(new Date(trade.close_time).getTime() / 1000) : entryTs + 86400);

        // Ensure we have at least 2 data points
        const t0 = entryTs;
        const t1 = Math.max(exitTs, entryTs + 3600);

        // Entry price line
        const entryLine = this.candleSeries.createPriceLine({
            price: entry,
            color: '#58a6ff',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: `Entry ${entry.toFixed(3)}`,
        });
        this.priceLines.push(entryLine);

        // TP zone (green shaded area between entry and TP)
        if (tp) {
            const tpTop = Math.max(entry, tp);
            const tpBot = Math.min(entry, tp);

            const tpTopSeries = this.chart.addLineSeries({
                color: 'transparent',
                lineWidth: 0,
                lastValueVisible: false,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
            });
            const tpBotSeries = this.chart.addLineSeries({
                color: 'transparent',
                lineWidth: 0,
                lastValueVisible: false,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
            });

            tpTopSeries.setData([{ time: t0, value: tpTop }, { time: t1, value: tpTop }]);
            tpBotSeries.setData([{ time: t0, value: tpBot }, { time: t1, value: tpBot }]);

            // TP label
            const tpLine = this.candleSeries.createPriceLine({
                price: tp,
                color: '#3fb950',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: `TP ${tp.toFixed(3)}`,
            });
            this.priceLines.push(tpLine);
            this._tradeRangeSeries.push(tpTopSeries, tpBotSeries);
        }

        // SL zone (red shaded area between entry and SL)
        if (sl) {
            const slTop = Math.max(entry, sl);
            const slBot = Math.min(entry, sl);

            const slTopSeries = this.chart.addLineSeries({
                color: 'transparent',
                lineWidth: 0,
                lastValueVisible: false,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
            });
            const slBotSeries = this.chart.addLineSeries({
                color: 'transparent',
                lineWidth: 0,
                lastValueVisible: false,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
            });

            slTopSeries.setData([{ time: t0, value: slTop }, { time: t1, value: slTop }]);
            slBotSeries.setData([{ time: t0, value: slBot }, { time: t1, value: slBot }]);

            // SL label
            const slLine = this.candleSeries.createPriceLine({
                price: sl,
                color: '#f85149',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: `SL ${sl.toFixed(3)}`,
            });
            this.priceLines.push(slLine);
            this._tradeRangeSeries.push(slTopSeries, slBotSeries);
        }

        // Draw semi-transparent rectangles via custom plugin overlay
        this._drawTradeZoneOverlay(entry, sl, tp, t0, t1, isBuy);
    }

    /**
     * Draw colored overlay rectangles on the chart for TP/SL zones.
     */
    _drawTradeZoneOverlay(entry, sl, tp, t0, t1, isBuy) {
        // Remove old overlay if exists
        const old = this.container.querySelector('.wt-trade-zone-overlay');
        if (old) old.remove();

        const overlay = document.createElement('canvas');
        overlay.className = 'wt-trade-zone-overlay';
        overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;';
        this.container.appendChild(overlay);
        this._zoneOverlay = overlay;

        const draw = () => {
            const w = this.container.clientWidth;
            const h = this.container.clientHeight;
            overlay.width = w;
            overlay.height = h;
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, w, h);

            const ts = this.chart.timeScale();
            const ps = this.chart.priceScale('right');

            // Convert time to x coordinates
            const x0 = ts.timeToCoordinate(t0);
            const x1 = ts.timeToCoordinate(t1);
            if (x0 === null || x1 === null) return;

            const xLeft = Math.max(0, Math.min(x0, x1));
            const xRight = Math.min(w, Math.max(x0, x1));
            const zoneWidth = xRight - xLeft;
            if (zoneWidth <= 0) return;

            // Convert prices to y coordinates
            const yEntry = this.candleSeries.priceToCoordinate(entry);
            if (yEntry === null) return;

            // TP zone (green)
            if (tp) {
                const yTp = this.candleSeries.priceToCoordinate(tp);
                if (yTp !== null) {
                    const yTop = Math.min(yEntry, yTp);
                    const yBot = Math.max(yEntry, yTp);
                    ctx.fillStyle = 'rgba(63, 185, 80, 0.12)';
                    ctx.fillRect(xLeft, yTop, zoneWidth, yBot - yTop);
                    // Border
                    ctx.strokeStyle = 'rgba(63, 185, 80, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(xLeft, yTop, zoneWidth, yBot - yTop);
                }
            }

            // SL zone (red)
            if (sl) {
                const ySl = this.candleSeries.priceToCoordinate(sl);
                if (ySl !== null) {
                    const yTop = Math.min(yEntry, ySl);
                    const yBot = Math.max(yEntry, ySl);
                    ctx.fillStyle = 'rgba(248, 81, 73, 0.12)';
                    ctx.fillRect(xLeft, yTop, zoneWidth, yBot - yTop);
                    // Border
                    ctx.strokeStyle = 'rgba(248, 81, 73, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(xLeft, yTop, zoneWidth, yBot - yTop);
                }
            }

            // Entry line highlight
            ctx.strokeStyle = 'rgba(88, 166, 255, 0.6)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.beginPath();
            ctx.moveTo(xLeft, yEntry);
            ctx.lineTo(xRight, yEntry);
            ctx.stroke();
            ctx.setLineDash([]);
        };

        draw();
        // Redraw on chart changes
        this._zoneRedraw = draw;
        this.chart.timeScale().subscribeVisibleLogicalRangeChange(draw);
        this.chart.subscribeCrosshairMove(() => draw());
    }

    /**
     * Clear all trade range zone overlays from the chart.
     */
    clearTradeRangeZones() {
        // Remove canvas overlay
        if (this._zoneOverlay) {
            this._zoneOverlay.remove();
            this._zoneOverlay = null;
        }
        if (this._zoneRedraw) {
            try {
                this.chart.timeScale().unsubscribeVisibleLogicalRangeChange(this._zoneRedraw);
            } catch (e) {}
            this._zoneRedraw = null;
        }
        // Remove helper line series
        this._tradeRangeSeries.forEach(s => {
            try { this.chart.removeSeries(s); } catch (e) {}
        });
        this._tradeRangeSeries = [];
        // Remove price lines
        this.priceLines.forEach(pl => {
            try { this.candleSeries.removePriceLine(pl); } catch (e) {}
        });
        this.priceLines = [];
    }

    /**
     * Show SL/TP price lines for a single trade (legacy — kept for compatibility).
     */
    showTradeLines(trade) {
        this.showTradeRangeZones(trade);
    }

    /**
     * Remove all SL/TP price lines from the chart.
     */
    clearPriceLines() {
        this.clearTradeRangeZones();
    }

    /**
     * Add a live trade entry marker incrementally (for live mode).
     */
    addLiveTradeMarker(trade) {
        const isBuy = trade.signal === 'BUY';
        const ts = trade.timestamp
            ? Math.floor(new Date(trade.timestamp).getTime() / 1000)
            : Math.floor(Date.now() / 1000);

        this.markers.push({
            time: ts,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: isBuy ? '#3fb950' : '#f85149',
            shape: isBuy ? 'arrowUp' : 'arrowDown',
            text: isBuy ? 'BUY' : 'SELL',
            size: 1,
        });

        this.markers.sort((a, b) => a.time - b.time);
        this.candleSeries.setMarkers(this.markers);
    }

    /**
     * Add a live trade exit marker incrementally (for live mode).
     */
    addLiveExitMarker(trade) {
        const ts = trade.timestamp
            ? Math.floor(new Date(trade.timestamp).getTime() / 1000)
            : Math.floor(Date.now() / 1000);

        this.markers.push({
            time: ts,
            position: 'aboveBar',
            color: '#58a6ff',
            shape: 'circle',
            text: trade.reason || 'EXIT',
            size: 1,
        });

        this.markers.sort((a, b) => a.time - b.time);
        this.candleSeries.setMarkers(this.markers);
    }

    /**
     * Add a signal indicator (diamond) on the chart for BUY/SELL signals.
     */
    addSignalMarker(signal) {
        if (signal.signal === 'HOLD') return;

        const isBuy = signal.signal === 'BUY';
        const ts = signal.timestamp
            ? Math.floor(new Date(signal.timestamp).getTime() / 1000)
            : Math.floor(Date.now() / 1000);

        this.markers.push({
            time: ts,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: isBuy ? 'rgba(63, 185, 80, 0.5)' : 'rgba(248, 81, 73, 0.5)',
            shape: 'square',
            text: `${isBuy ? '▲' : '▼'} ${(signal.confidence * 100).toFixed(0)}%`,
            size: 0,
        });

        this.markers.sort((a, b) => a.time - b.time);
        this.candleSeries.setMarkers(this.markers);
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
