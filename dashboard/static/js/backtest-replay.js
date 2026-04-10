/**
 * WaveTrader Backtest Replay Controller
 * Animates backtest results on the TradingView chart — candles appear
 * one-by-one with trade markers injected at entry/exit times.
 *
 * Controls: play/pause, rewind, fast-forward, speed (0.5x–50x),
 * progress scrubber, step forward/back.
 */
class BacktestReplayController {
    /**
     * @param {ChartManager} chartManager
     * @param {Array} candles  — [{time, open, high, low, close, volume}, ...]
     * @param {Array} trades   — [{entry_time, exit_time, direction, pnl, ...}, ...]
     * @param {number} initialBalance
     */
    constructor(chartManager, candles, trades, initialBalance = 25000) {
        this.cm = chartManager;
        this.allCandles = candles || [];
        this.allTrades = trades || [];
        this.initialBalance = initialBalance;

        // Playback state
        this.currentIndex = 0;
        this.isPlaying = false;
        this.speed = 1;           // multiplier
        this._timerId = null;
        this._baseInterval = 80;  // ms between candles at 1x speed

        // Pre-process trade timestamps for fast lookup
        this._tradeEntryMap = new Map(); // candleTime → [trade, ...]
        this._tradeExitMap = new Map();
        this._activeMarkers = [];
        this._closedTradeCount = 0;
        this._runningBalance = initialBalance;
        this._activeTrade = null;  // currently open position

        this._buildTradeMaps();

        // DOM elements (populated by bindUI)
        this.ui = {};
    }

    // ── Trade timestamp mapping ────────────────────────────────────────

    _buildTradeMaps() {
        this._tradeEntryMap.clear();
        this._tradeExitMap.clear();

        const candleTimes = new Set(this.allCandles.map(c => c.time));

        this.allTrades.forEach((trade, idx) => {
            trade._replayIdx = idx;

            if (trade.entry_time) {
                const ts = this._tradeTimeToCandle(trade.entry_time);
                if (!this._tradeEntryMap.has(ts)) this._tradeEntryMap.set(ts, []);
                this._tradeEntryMap.get(ts).push(trade);
            }
            if (trade.exit_time) {
                const ts = this._tradeTimeToCandle(trade.exit_time);
                if (!this._tradeExitMap.has(ts)) this._tradeExitMap.set(ts, []);
                this._tradeExitMap.get(ts).push(trade);
            }
        });
    }

    /** Snap a trade ISO timestamp to the nearest candle time. */
    _tradeTimeToCandle(isoString) {
        const tradeSec = Math.floor(new Date(isoString).getTime() / 1000);
        // Binary search for nearest candle
        let best = this.allCandles[0]?.time || tradeSec;
        let bestDiff = Math.abs(tradeSec - best);
        let lo = 0, hi = this.allCandles.length - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const diff = Math.abs(this.allCandles[mid].time - tradeSec);
            if (diff < bestDiff) { best = this.allCandles[mid].time; bestDiff = diff; }
            if (this.allCandles[mid].time < tradeSec) lo = mid + 1;
            else hi = mid - 1;
        }
        return best;
    }

    // ── UI binding ─────────────────────────────────────────────────────

    bindUI() {
        this.ui = {
            btnPlay:       document.getElementById('replay-play'),
            btnStepBack:   document.getElementById('replay-step-back'),
            btnStepFwd:    document.getElementById('replay-step-fwd'),
            btnStart:      document.getElementById('replay-start'),
            btnEnd:        document.getElementById('replay-end'),
            scrubber:      document.getElementById('replay-scrubber'),
            speedBtns:     document.querySelectorAll('.replay-speed-btn'),
            timeLabel:     document.getElementById('replay-time'),
            tradeCounter:  document.getElementById('replay-trade-counter'),
            equityLabel:   document.getElementById('replay-equity'),
            pnlLabel:      document.getElementById('replay-pnl'),
            bar:           document.getElementById('replay-bar'),
        };

        if (!this.ui.btnPlay) return; // controls not in the DOM

        // Play / Pause
        this.ui.btnPlay.addEventListener('click', () => this.togglePlay());

        // Step buttons
        this.ui.btnStepBack?.addEventListener('click', () => { this.pause(); this.seekTo(this.currentIndex - 1); });
        this.ui.btnStepFwd?.addEventListener('click', () => { this.pause(); this.seekTo(this.currentIndex + 1); });
        this.ui.btnStart?.addEventListener('click', () => { this.pause(); this.seekTo(0); });
        this.ui.btnEnd?.addEventListener('click', () => { this.pause(); this.seekTo(this.allCandles.length - 1); });

        // Scrubber
        if (this.ui.scrubber) {
            this.ui.scrubber.max = Math.max(0, this.allCandles.length - 1);
            this.ui.scrubber.value = 0;
            this.ui.scrubber.addEventListener('input', () => {
                this.pause();
                this.seekTo(parseInt(this.ui.scrubber.value, 10));
            });
        }

        // Speed buttons
        this.ui.speedBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const s = parseFloat(btn.dataset.speed);
                this.setSpeed(s);
                this.ui.speedBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Keyboard shortcuts
        this._keyHandler = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === ' ') { e.preventDefault(); this.togglePlay(); }
            if (e.key === 'ArrowLeft')  { e.preventDefault(); this.pause(); this.seekTo(this.currentIndex - 1); }
            if (e.key === 'ArrowRight') { e.preventDefault(); this.pause(); this.seekTo(this.currentIndex + 1); }
            if (e.key === 'Home')       { e.preventDefault(); this.pause(); this.seekTo(0); }
            if (e.key === 'End')        { e.preventDefault(); this.pause(); this.seekTo(this.allCandles.length - 1); }
        };
        document.addEventListener('keydown', this._keyHandler);
    }

    // ── Playback controls ──────────────────────────────────────────────

    play() {
        if (this.isPlaying) return;
        if (this.currentIndex >= this.allCandles.length - 1) {
            this.seekTo(0);  // restart if at end
        }
        this.isPlaying = true;
        this._updatePlayButton();
        this._scheduleNext();
    }

    pause() {
        this.isPlaying = false;
        if (this._timerId) { clearTimeout(this._timerId); this._timerId = null; }
        this._updatePlayButton();
    }

    togglePlay() {
        this.isPlaying ? this.pause() : this.play();
    }

    setSpeed(multiplier) {
        this.speed = multiplier;
        // If currently playing, reschedule with new speed
        if (this.isPlaying) {
            if (this._timerId) clearTimeout(this._timerId);
            this._scheduleNext();
        }
    }

    seekTo(index) {
        index = Math.max(0, Math.min(index, this.allCandles.length - 1));
        this.currentIndex = index;
        this._renderAtIndex(index);
        this._updateUI();
    }

    // ── Core animation loop ────────────────────────────────────────────

    _scheduleNext() {
        if (!this.isPlaying) return;
        if (this.currentIndex >= this.allCandles.length - 1) {
            this.pause();
            return;
        }

        const interval = Math.max(1, Math.round(this._baseInterval / this.speed));

        // At very high speeds (>=10x), batch multiple candles per tick for smoothness
        const batchSize = this.speed >= 50 ? 20 : this.speed >= 10 ? 5 : this.speed >= 5 ? 2 : 1;

        this._timerId = setTimeout(() => {
            const target = Math.min(this.currentIndex + batchSize, this.allCandles.length - 1);
            this.currentIndex = target;
            this._renderAtIndex(target);
            this._updateUI();
            this._scheduleNext();
        }, interval);
    }

    // ── Rendering ──────────────────────────────────────────────────────

    _renderAtIndex(index) {
        const sliceEnd = index + 1;
        const candleSlice = this.allCandles.slice(0, sliceEnd);
        const volumeSlice = candleSlice.map(c => ({
            time: c.time,
            value: c.volume,
            color: c.close >= c.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
        }));

        // Set candle data
        this.cm.candleSeries.setData(candleSlice);
        this.cm.volumeSeries.setData(volumeSlice);

        // Build markers up to current time
        const currentTime = this.allCandles[index].time;
        this._rebuildMarkers(currentTime);

        // Compute running balance & trade count
        this._computeState(currentTime);

        // SL/TP lines for active trade
        this._showActiveTradePriceLines(currentTime);

        // Auto-scroll: keep the latest candle visible
        if (this.isPlaying) {
            this.cm.chart.timeScale().scrollToRealTime();
        }
    }

    _rebuildMarkers(upToTime) {
        const markers = [];
        let tradeIdx = 0;

        for (const trade of this.allTrades) {
            const entryTs = trade.entry_time ? Math.floor(new Date(trade.entry_time).getTime() / 1000) : null;
            const exitTs = trade.exit_time ? Math.floor(new Date(trade.exit_time).getTime() / 1000) : null;

            const isBuy = trade.direction === 'BUY';
            const isWin = trade.pnl > 0;

            // Entry marker
            if (entryTs && entryTs <= upToTime) {
                markers.push({
                    time: entryTs,
                    position: isBuy ? 'belowBar' : 'aboveBar',
                    color: isBuy ? '#3fb950' : '#f85149',
                    shape: isBuy ? 'arrowUp' : 'arrowDown',
                    text: `${isBuy ? 'B' : 'S'}${tradeIdx + 1}`,
                    size: 1,
                });
            }

            // Exit marker (only if exit has happened)
            if (exitTs && exitTs <= upToTime) {
                markers.push({
                    time: exitTs,
                    position: isBuy ? 'aboveBar' : 'belowBar',
                    color: isWin ? '#58a6ff' : '#d29922',
                    shape: 'circle',
                    text: `${isWin ? '+' : ''}$${trade.pnl.toFixed(0)}`,
                    size: 1,
                });
            }

            tradeIdx++;
        }

        markers.sort((a, b) => a.time - b.time);
        this.cm.candleSeries.setMarkers(markers);
    }

    _computeState(upToTime) {
        let balance = this.initialBalance;
        let closedCount = 0;
        let activeTrade = null;

        for (const trade of this.allTrades) {
            const entryTs = trade.entry_time ? Math.floor(new Date(trade.entry_time).getTime() / 1000) : null;
            const exitTs = trade.exit_time ? Math.floor(new Date(trade.exit_time).getTime() / 1000) : null;

            if (exitTs && exitTs <= upToTime) {
                balance += trade.pnl;
                closedCount++;
            } else if (entryTs && entryTs <= upToTime && (!exitTs || exitTs > upToTime)) {
                activeTrade = trade;
            }
        }

        this._closedTradeCount = closedCount;
        this._runningBalance = balance;
        this._activeTrade = activeTrade;
    }

    _showActiveTradePriceLines(currentTime) {
        // Clear previous replay price lines
        this.cm.priceLines.forEach(pl => {
            try { this.cm.candleSeries.removePriceLine(pl); } catch (e) {}
        });
        this.cm.priceLines = [];

        if (!this._activeTrade) return;

        const t = this._activeTrade;
        // Entry line
        this.cm.priceLines.push(this.cm.candleSeries.createPriceLine({
            price: t.entry_price,
            color: '#58a6ff',
            lineWidth: 1,
            lineStyle: 2, // Dashed
            axisLabelVisible: true,
            title: `Entry ${t.entry_price.toFixed(3)}`,
        }));
        // SL line
        if (t.stop_loss) {
            this.cm.priceLines.push(this.cm.candleSeries.createPriceLine({
                price: t.stop_loss,
                color: '#d29922',
                lineWidth: 1,
                lineStyle: 2,
                axisLabelVisible: true,
                title: 'SL',
            }));
        }
        // TP line
        if (t.take_profit) {
            this.cm.priceLines.push(this.cm.candleSeries.createPriceLine({
                price: t.take_profit,
                color: '#3fb950',
                lineWidth: 1,
                lineStyle: 2,
                axisLabelVisible: true,
                title: 'TP',
            }));
        }
    }

    // ── UI updates ─────────────────────────────────────────────────────

    _updateUI() {
        if (!this.ui.scrubber) return;

        // Progress scrubber
        this.ui.scrubber.value = this.currentIndex;

        // Time label
        if (this.ui.timeLabel && this.allCandles[this.currentIndex]) {
            const ts = this.allCandles[this.currentIndex].time;
            const d = new Date(ts * 1000);
            this.ui.timeLabel.textContent = d.toISOString().replace('T', ' ').slice(0, 16);
        }

        // Progress bar fill
        if (this.ui.bar) {
            const pct = this.allCandles.length > 1
                ? (this.currentIndex / (this.allCandles.length - 1)) * 100
                : 0;
            this.ui.bar.style.width = `${pct}%`;
        }

        // Trade counter
        if (this.ui.tradeCounter) {
            const active = this._activeTrade ? ' (1 open)' : '';
            this.ui.tradeCounter.textContent = `${this._closedTradeCount} / ${this.allTrades.length} trades${active}`;
        }

        // Equity
        if (this.ui.equityLabel) {
            this.ui.equityLabel.textContent = `$${this._runningBalance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        }

        // P&L
        if (this.ui.pnlLabel) {
            const pnl = this._runningBalance - this.initialBalance;
            const sign = pnl >= 0 ? '+' : '';
            this.ui.pnlLabel.textContent = `${sign}$${pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            this.ui.pnlLabel.className = `replay-pnl-value ${pnl >= 0 ? 'positive' : 'negative'}`;
        }

        // Candle count
        const barLabel = document.getElementById('replay-bar-counter');
        if (barLabel) {
            barLabel.textContent = `Bar ${this.currentIndex + 1} / ${this.allCandles.length}`;
        }
    }

    _updatePlayButton() {
        if (!this.ui.btnPlay) return;
        const icon = this.ui.btnPlay.querySelector('i');
        if (icon) {
            icon.className = this.isPlaying ? 'bi bi-pause-fill' : 'bi bi-play-fill';
        }
        this.ui.btnPlay.title = this.isPlaying ? 'Pause (Space)' : 'Play (Space)';
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /** Show the toolbar and initialize to the start. */
    show() {
        const toolbar = document.getElementById('replay-toolbar');
        if (toolbar) toolbar.style.display = '';
        this.seekTo(0);
    }

    /** Hide the toolbar and stop playback. */
    hide() {
        this.pause();
        const toolbar = document.getElementById('replay-toolbar');
        if (toolbar) toolbar.style.display = 'none';
    }

    /** Full cleanup. */
    destroy() {
        this.pause();
        if (this._keyHandler) {
            document.removeEventListener('keydown', this._keyHandler);
        }
    }
}

// Export
window.BacktestReplayController = BacktestReplayController;
