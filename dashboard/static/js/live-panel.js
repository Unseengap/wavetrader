/**
 * WaveTrader Live Panel
 * Handles live OANDA data streaming via SSE, chart updates,
 * account display, and model signal rendering.
 */

let liveMode = false;
let eventSource = null;
let liveReconnectTimer = null;

// ── Toggle Handler ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('live-toggle');
    if (toggle) {
        toggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                enableLiveMode();
            } else {
                disableLiveMode();
            }
        });
    }
});

async function enableLiveMode() {
    liveMode = true;

    // Update UI
    document.getElementById('live-label').textContent = 'Live';
    document.getElementById('live-label').style.color = 'var(--wt-green)';
    document.getElementById('live-panel').style.display = 'block';
    document.getElementById('backtest-panel').style.display = 'none';
    document.getElementById('connection-status').innerHTML =
        '<i class="bi bi-circle-fill" style="color:var(--wt-yellow);font-size:6px;vertical-align:middle"></i> Connecting…';

    // Switch to Live Trading hub tab
    if (typeof switchToTopTab === 'function') {
        switchToTopTab('live-hub');
    }

    const pair = document.getElementById('nav-pair-select').value || 'GBP/JPY';
    const tf = document.getElementById('nav-tf-select').value || '15min';

    // 1. Start the server-side stream
    try {
        await fetch('/api/live/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pair, timeframe: '15min' }),
        });
    } catch (err) {
        showToast('Failed to start live stream: ' + err.message, 'error');
    }

    // 2. Load initial candles from OANDA for the chart
    await loadLiveCandles(pair, tf);

    // 3. Connect SSE
    connectSSE();

    // 4. Load account
    await refreshAccount();

    showToast('Live mode enabled — streaming from OANDA', 'success');
}

function disableLiveMode() {
    liveMode = false;

    // Disconnect SSE
    disconnectSSE();

    // Update UI
    document.getElementById('live-label').textContent = 'Historical';
    document.getElementById('live-label').style.color = 'var(--wt-text-muted)';
    document.getElementById('live-panel').style.display = 'none';
    document.getElementById('backtest-panel').style.display = 'block';
    document.getElementById('connection-status').innerHTML =
        '<i class="bi bi-circle-fill" style="color:var(--wt-green);font-size:6px;vertical-align:middle"></i> Dashboard Active';

    // Switch back to Backtest hub tab
    if (typeof switchToTopTab === 'function') {
        switchToTopTab('backtest-hub');
    }

    // Reload historical candles
    const pair = document.getElementById('nav-pair-select').value || 'GBP/JPY';
    const tf = document.getElementById('nav-tf-select').value || '1h';
    if (chartManager) {
        chartManager.loadCandles(pair, tf);
    }

    // Stop server-side stream
    fetch('/api/live/stop', { method: 'POST' }).catch(() => {});

    showToast('Switched to historical mode', 'info');
}

// ── Load OANDA Candles ──────────────────────────────────────────────────────

async function loadLiveCandles(pair, tf) {
    if (!chartManager) return;

    try {
        const params = new URLSearchParams({ pair, tf, count: '500' });
        const resp = await fetch(`/api/live/candles?${params}`);
        const data = await resp.json();

        if (!data.candles || data.candles.length === 0) {
            showToast('No live candles returned from OANDA', 'error');
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

        chartManager.candleSeries.setData(candles);
        chartManager.volumeSeries.setData(volumes);
        chartManager.chart.timeScale().scrollToRealTime();
    } catch (err) {
        showToast('Failed to load live candles: ' + err.message, 'error');
    }
}

// ── SSE Connection ──────────────────────────────────────────────────────────

function connectSSE() {
    disconnectSSE();

    eventSource = new EventSource('/api/live/stream');

    eventSource.onopen = () => {
        document.getElementById('connection-status').innerHTML =
            '<i class="bi bi-circle-fill" style="color:var(--wt-green);font-size:6px;vertical-align:middle"></i> Live Connected';
        document.getElementById('live-status-dot').className = 'wt-status-dot online';
    };

    eventSource.onerror = () => {
        document.getElementById('connection-status').innerHTML =
            '<i class="bi bi-circle-fill" style="color:var(--wt-red);font-size:6px;vertical-align:middle"></i> Reconnecting…';
        document.getElementById('live-status-dot').className = 'wt-status-dot offline';
    };

    // ── Candle updates ────────────────────────────────────────────────
    eventSource.addEventListener('candle', (e) => {
        if (!chartManager || !liveMode) return;
        try {
            const c = JSON.parse(e.data);
            const t = typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000);
            if (!t || isNaN(t)) return;
            chartManager.candleSeries.update({
                time: t,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
            });
            chartManager.volumeSeries.update({
                time: t,
                value: c.volume,
                color: c.close >= c.open
                    ? 'rgba(63, 185, 80, 0.3)'
                    : 'rgba(248, 81, 73, 0.3)',
            });
        } catch (err) {
            console.error('candle parse error:', err);
        }
    });

    // ── Price updates ─────────────────────────────────────────────────
    eventSource.addEventListener('price', (e) => {
        if (!liveMode) return;
        try {
            const p = JSON.parse(e.data);
            setText('live-bid', p.bid.toFixed(3));
            setText('live-ask', p.ask.toFixed(3));

            const spreadPips = (p.spread / 0.01).toFixed(1); // JPY pair
            setText('live-spread', spreadPips + ' pips');
        } catch (err) {
            console.error('price parse error:', err);
        }
    });

    // ── Model signal ──────────────────────────────────────────────────
    eventSource.addEventListener('signal', (e) => {
        if (!liveMode) return;
        try {
            const s = JSON.parse(e.data);
            const badge = document.getElementById('live-signal-badge');
            badge.textContent = s.signal;
            badge.className = 'wt-signal-badge ' + s.signal.toLowerCase();

            setText('live-confidence', (s.confidence * 100).toFixed(1) + '%');
            setText('live-alignment', (s.alignment * 100).toFixed(1) + '%');
            setText('live-sl', s.sl_pips.toFixed(1));
            setText('live-tp', s.tp_pips.toFixed(1));

            const ts = s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—';
            setText('live-signal-time', 'Last signal: ' + ts);

            // Flash the badge
            badge.style.animation = 'none';
            badge.offsetHeight; // reflow
            badge.style.animation = 'signal-flash 0.6s ease';

            showToast(`Signal: ${s.signal} (${(s.confidence * 100).toFixed(0)}% conf)`,
                s.signal === 'BUY' ? 'success' : s.signal === 'SELL' ? 'error' : 'info');
        } catch (err) {
            console.error('signal parse error:', err);
        }
    });

    // ── Account updates ───────────────────────────────────────────────
    eventSource.addEventListener('account', (e) => {
        if (!liveMode) return;
        try {
            const a = JSON.parse(e.data);
            if (a.error) return;
            setText('live-balance', '$' + a.balance.toLocaleString('en-US', { minimumFractionDigits: 2 }));
            setText('live-nav', '$' + a.nav.toLocaleString('en-US', { minimumFractionDigits: 2 }));

            const pnlEl = document.getElementById('live-unreal-pnl');
            if (pnlEl) {
                const pnl = a.unrealized_pnl;
                pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
                pnlEl.style.color = pnl >= 0 ? 'var(--wt-green)' : 'var(--wt-red)';
            }

            setText('live-open-trades', a.open_trades);
            const marketEl = document.getElementById('live-market-status');
            if (marketEl) {
                marketEl.textContent = a.market_open ? 'Open' : 'Closed';
                marketEl.style.color = a.market_open ? 'var(--wt-green)' : 'var(--wt-text-muted)';
            }
        } catch (err) {
            console.error('account parse error:', err);
        }
    });

    // ── Open trades ───────────────────────────────────────────────────
    eventSource.addEventListener('trades', (e) => {
        if (!liveMode) return;
        try {
            const data = JSON.parse(e.data);
            renderOpenPositions(data.trades || []);
        } catch (err) {
            console.error('trades parse error:', err);
        }
    });

    // ── Status ────────────────────────────────────────────────────────
    eventSource.addEventListener('status', (e) => {
        try {
            const s = JSON.parse(e.data);
            if (s.status === 'error') {
                showToast('Stream error: ' + s.error, 'error');
            }
        } catch (err) {}
    });
}

function disconnectSSE() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    if (liveReconnectTimer) {
        clearTimeout(liveReconnectTimer);
        liveReconnectTimer = null;
    }
}

// ── Refresh Account (one-shot) ──────────────────────────────────────────────

async function refreshAccount() {
    try {
        const resp = await fetch('/api/live/account');
        const a = await resp.json();
        if (a.error) {
            showToast('OANDA: ' + a.error, 'error');
            return;
        }
        setText('live-balance', '$' + a.balance.toLocaleString('en-US', { minimumFractionDigits: 2 }));
        setText('live-nav', '$' + a.nav.toLocaleString('en-US', { minimumFractionDigits: 2 }));
        setText('live-open-trades', a.open_trades);

        const marketEl = document.getElementById('live-market-status');
        if (marketEl) {
            marketEl.textContent = a.market_open ? 'Open' : 'Closed';
            marketEl.style.color = a.market_open ? 'var(--wt-green)' : 'var(--wt-text-muted)';
        }
    } catch (err) {
        showToast('Failed to load account: ' + err.message, 'error');
    }
}

// ── Render Open Positions ───────────────────────────────────────────────────

function renderOpenPositions(trades) {
    const container = document.getElementById('live-positions-list');
    if (!container) return;

    if (!trades || trades.length === 0) {
        container.innerHTML = '<div class="wt-empty-state" style="padding:0.5rem">No open positions</div>';
        return;
    }

    let html = '';
    trades.forEach(t => {
        const dir = t.direction;
        const color = dir === 'BUY' ? 'var(--wt-green)' : 'var(--wt-red)';
        const pnlColor = t.unrealized_pnl >= 0 ? 'var(--wt-green)' : 'var(--wt-red)';
        html += `
            <div style="padding:0.35rem 0.5rem;border-bottom:1px solid var(--wt-border)">
                <div style="display:flex;justify-content:space-between">
                    <span style="color:${color};font-weight:600">${dir}</span>
                    <span style="color:${pnlColor}">${t.unrealized_pnl >= 0 ? '+' : ''}$${t.unrealized_pnl.toFixed(2)}</span>
                </div>
                <div style="color:var(--wt-text-muted);font-size:0.68rem">
                    Entry: ${t.price.toFixed(3)} | SL: ${t.stop_loss ? t.stop_loss.toFixed(3) : '—'} | TP: ${t.take_profit ? t.take_profit.toFixed(3) : '—'}
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

// ── Utility ─────────────────────────────────────────────────────────────────

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
