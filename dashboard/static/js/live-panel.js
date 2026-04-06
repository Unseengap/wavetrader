/**
 * WaveTrader Live Panel
 * Handles live OANDA data streaming via SSE, chart updates,
 * account display, and model signal rendering.
 */

let liveMode = false;
let eventSource = null;
let liveReconnectTimer = null;
let lastCandleTime = 0;  // Guard against out-of-order candle updates

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
    document.getElementById('connection-status').innerHTML =
        '<i class="bi bi-circle-fill" style="color:var(--wt-yellow);font-size:6px;vertical-align:middle"></i> Connecting…';

    // Show split sidebar: performance (top) + live feed (bottom)
    document.getElementById('live-panel').style.display = 'block';
    document.getElementById('sidebar-drag-handle').style.display = 'block';
    document.getElementById('backtest-panel').style.display = 'block';
    activateSidebarSplit();

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

    // 5. Load trade history from broker
    await loadTradeHistory();

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
    document.getElementById('sidebar-drag-handle').style.display = 'none';
    document.getElementById('backtest-panel').style.display = 'block';
    deactivateSidebarSplit();
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
        // Reset candle guard to the latest candle time
        if (candles.length > 0) {
            lastCandleTime = candles[candles.length - 1].time;
        }
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
            // Skip candles older than the last one we rendered
            if (t < lastCandleTime) return;
            lastCandleTime = t;
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

            // Mirror to sidebar
            setText('live-bid-sidebar', p.bid.toFixed(3));
            setText('live-ask-sidebar', p.ask.toFixed(3));
            setText('live-spread-sidebar', spreadPips + ' pips');
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

            // Mirror to sidebar
            const sidebarBadge = document.getElementById('live-signal-badge-sidebar');
            if (sidebarBadge) {
                sidebarBadge.textContent = s.signal;
                sidebarBadge.className = 'wt-signal-badge ' + s.signal.toLowerCase();
            }
            setText('live-confidence-sidebar', (s.confidence * 100).toFixed(1) + '%');
            setText('live-alignment-sidebar', (s.alignment * 100).toFixed(1) + '%');

            showToast(`Signal: ${s.signal} (${(s.confidence * 100).toFixed(0)}% conf)`,
                s.signal === 'BUY' ? 'success' : s.signal === 'SELL' ? 'error' : 'info');

            // Log signal to Signals tab
            appendSignalLog(s);
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

            // Mirror to sidebar
            setText('live-balance-sidebar', '$' + a.balance.toLocaleString('en-US', { minimumFractionDigits: 2 }));
            setText('live-nav-sidebar', '$' + a.nav.toLocaleString('en-US', { minimumFractionDigits: 2 }));
            const sidebarPnl = document.getElementById('live-pnl-sidebar');
            if (sidebarPnl) {
                const pnl = a.unrealized_pnl;
                sidebarPnl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
                sidebarPnl.style.color = pnl >= 0 ? 'var(--wt-green)' : 'var(--wt-red)';
            }
            setText('live-trades-sidebar', a.open_trades);
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

    // ── Trade executed (auto-trade) ───────────────────────────────────
    eventSource.addEventListener('trade_executed', (e) => {
        if (!liveMode) return;
        try {
            const t = JSON.parse(e.data);
            appendTradeLog(t);
            const accountLabel = t.account === 'live' ? 'LIVE' : 'Demo';
            showToast(`Trade ${t.signal} ${t.pair} @ ${t.entry_price} (${accountLabel})`,
                t.signal === 'BUY' ? 'success' : 'error');
            refreshAccount();
            loadTradeHistory();
        } catch (err) {
            console.error('trade_executed parse error:', err);
        }
    });

    // ── Trade closed ──────────────────────────────────────────────────
    eventSource.addEventListener('trade_closed', (e) => {
        if (!liveMode) return;
        try {
            const t = JSON.parse(e.data);
            const accountLabel = t.account === 'live' ? 'LIVE' : 'Demo';
            showToast(`Position closed (${accountLabel}): ${t.reason}`, 'info');
            refreshAccount();
            loadTradeHistory();
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

// ── Auto-Trade Status (always on — no toggle) ──────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Fetch auto-trade status to show demo/live badges
    fetch('/api/live/auto-trade')
        .then(r => r.json())
        .then(data => {
            const liveBadge = document.getElementById('auto-trade-live-badge');
            if (liveBadge && data.live_active) {
                liveBadge.style.display = 'inline';
            }
        })
        .catch(() => {});

    // Trade history filter buttons
    document.querySelectorAll('.wt-history-filter').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.wt-history-filter').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const filter = btn.dataset.filter;
            document.querySelectorAll('.wt-history-entry').forEach(row => {
                if (filter === 'all' || row.dataset.account === filter) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    });
});

// ── Signal & Trade Logging ──────────────────────────────────────────────────

function appendSignalLog(signal) {
    const log = document.getElementById('live-signals-log');
    if (!log) return;

    // Clear empty state
    const empty = log.querySelector('.wt-empty-state');
    if (empty) empty.remove();

    const color = signal.signal === 'BUY' ? 'var(--wt-green)' : signal.signal === 'SELL' ? 'var(--wt-red)' : 'var(--wt-text-muted)';
    const ts = signal.timestamp ? new Date(signal.timestamp).toLocaleTimeString() : '—';
    const html = `
        <div class="wt-signal-log-entry" style="padding:0.35rem 0.5rem;border-bottom:1px solid var(--wt-border);font-size:0.78rem">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:${color};font-weight:600">${signal.signal}</span>
                <span style="color:var(--wt-text-muted);font-size:0.68rem">${ts}</span>
            </div>
            <div style="color:var(--wt-text-muted);font-size:0.68rem">
                Conf: ${(signal.confidence * 100).toFixed(1)}% | Align: ${(signal.alignment * 100).toFixed(1)}% | SL: ${signal.sl_pips.toFixed(1)} | TP: ${signal.tp_pips.toFixed(1)}
            </div>
        </div>
    `;
    log.insertAdjacentHTML('afterbegin', html);

    // Keep only last 50 entries
    const entries = log.querySelectorAll('.wt-signal-log-entry');
    if (entries.length > 50) {
        for (let i = 50; i < entries.length; i++) entries[i].remove();
    }
}

function appendTradeLog(trade) {
    const log = document.getElementById('live-signals-log');
    if (!log) return;

    const color = trade.signal === 'BUY' ? 'var(--wt-green)' : 'var(--wt-red)';
    const ts = trade.timestamp ? new Date(trade.timestamp).toLocaleTimeString() : '—';
    const accountLabel = trade.account === 'live'
        ? '<span style="color:var(--wt-green);font-weight:600">LIVE</span>'
        : '<span style="color:#58a6ff">Demo</span>';
    const html = `
        <div class="wt-signal-log-entry" style="padding:0.35rem 0.5rem;border-bottom:1px solid var(--wt-border);font-size:0.78rem;background:rgba(88,166,255,0.06)">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span><i class="bi bi-arrow-left-right"></i> <strong style="color:${color}">TRADE ${trade.signal}</strong> ${accountLabel}</span>
                <span style="font-size:0.68rem;color:var(--wt-text-muted)">${ts}</span>
            </div>
            <div style="color:var(--wt-text-muted);font-size:0.68rem">
                @ ${trade.entry_price} | SL: ${trade.sl} | TP: ${trade.tp} | ${trade.status}
            </div>
        </div>
    `;
    log.insertAdjacentHTML('afterbegin', html);
}

// ── Trade History from Broker ────────────────────────────────────────────────

async function loadTradeHistory() {
    const list = document.getElementById('trade-history-list');
    if (!list) return;

    try {
        const resp = await fetch('/api/live/trade-history?count=50');
        const data = await resp.json();
        const trades = data.trades || [];

        if (trades.length === 0) {
            list.innerHTML = '<div class="wt-empty-state" style="padding:1rem"><p>No trade history yet</p></div>';
            return;
        }

        let html = '';
        for (const t of trades) {
            const dir = parseFloat(t.units) > 0 ? 'BUY' : 'SELL';
            const dirColor = dir === 'BUY' ? 'var(--wt-green)' : 'var(--wt-red)';
            const pl = parseFloat(t.realized_pl || 0);
            const plColor = pl >= 0 ? 'var(--wt-green)' : 'var(--wt-red)';
            const plStr = pl >= 0 ? `+${pl.toFixed(2)}` : pl.toFixed(2);
            const accountBg = t.account === 'live' ? 'rgba(63,185,80,0.12)' : 'rgba(88,166,255,0.08)';
            const accountColor = t.account === 'live' ? 'var(--wt-green)' : '#58a6ff';
            const state = t.state === 'OPEN' ? '<span style="color:var(--wt-yellow)">OPEN</span>' : '<span style="color:var(--wt-text-muted)">CLOSED</span>';
            const openTime = t.open_time ? new Date(t.open_time).toLocaleString() : '—';

            html += `
                <div class="wt-history-entry" data-account="${t.account}" style="padding:0.4rem 0.5rem;border-bottom:1px solid var(--wt-border);font-size:0.75rem;background:${accountBg}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span>
                            <strong style="color:${dirColor}">${dir}</strong>
                            <span style="color:${accountColor};font-size:0.65rem;margin-left:4px">${t.account.toUpperCase()}</span>
                        </span>
                        <span style="color:${plColor};font-weight:600">${plStr}</span>
                    </div>
                    <div style="color:var(--wt-text-muted);font-size:0.68rem;margin-top:2px">
                        ${t.instrument} @ ${parseFloat(t.price).toFixed(3)} | ${state} | ${openTime}
                    </div>
                </div>
            `;
        }
        list.innerHTML = html;
    } catch (err) {
        console.error('loadTradeHistory error:', err);
    }
}

// ── Sidebar Split (Performance + Live Feed) ─────────────────────────────────

let sidebarDragging = false;

function activateSidebarSplit() {
    const sidebar = document.getElementById('sidebar-right');
    const top = document.getElementById('backtest-panel');
    const handle = document.getElementById('sidebar-drag-handle');
    const bottom = document.getElementById('live-panel');
    if (!sidebar || !top || !handle || !bottom) return;

    sidebar.classList.add('split-active');

    // Restore saved ratio or default to 50%
    const saved = localStorage.getItem('wt-sidebar-ratio');
    const ratio = saved ? parseFloat(saved) : 0.5;
    applySidebarRatio(ratio);
}

function deactivateSidebarSplit() {
    const sidebar = document.getElementById('sidebar-right');
    if (sidebar) sidebar.classList.remove('split-active');
    const top = document.getElementById('backtest-panel');
    if (top) { top.style.flex = ''; top.style.overflow = ''; }
    const bottom = document.getElementById('live-panel');
    if (bottom) { bottom.style.flex = ''; bottom.style.overflow = ''; }
}

function applySidebarRatio(ratio) {
    const top = document.getElementById('backtest-panel');
    const bottom = document.getElementById('live-panel');
    if (!top || !bottom) return;
    ratio = Math.max(0.2, Math.min(0.8, ratio));
    top.style.flex = `0 0 ${ratio * 100}%`;
    top.style.overflow = 'auto';
    bottom.style.flex = `0 0 ${(1 - ratio) * 100 - 2}%`; // 2% for handle
    bottom.style.overflow = 'auto';
}

document.addEventListener('DOMContentLoaded', () => {
    const handle = document.getElementById('sidebar-drag-handle');
    if (!handle) return;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        sidebarDragging = true;
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';

        const sidebar = document.getElementById('sidebar-right');
        const startY = e.clientY;
        const sidebarRect = sidebar.getBoundingClientRect();
        const sidebarH = sidebarRect.height;
        const top = document.getElementById('backtest-panel');
        const startTopH = top.getBoundingClientRect().height;

        function onMove(ev) {
            if (!sidebarDragging) return;
            const dy = ev.clientY - startY;
            const newRatio = (startTopH + dy) / sidebarH;
            applySidebarRatio(newRatio);
        }

        function onUp() {
            sidebarDragging = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            // Save ratio
            const top = document.getElementById('backtest-panel');
            const sidebar = document.getElementById('sidebar-right');
            if (top && sidebar) {
                const ratio = top.getBoundingClientRect().height / sidebar.getBoundingClientRect().height;
                localStorage.setItem('wt-sidebar-ratio', ratio.toFixed(3));
            }
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    // Double-click to reset to 50/50
    handle.addEventListener('dblclick', () => {
        applySidebarRatio(0.5);
        localStorage.setItem('wt-sidebar-ratio', '0.5');
    });
});
