/**
 * WaveTrader Live Panel
 * Handles live OANDA data streaming via SSE, chart updates,
 * account display, model signal rendering, trade markers, and
 * table-based trade history.
 *
 * Used by the standalone live page (live.html + live-init.js).
 */

let liveMode = true;  // Always true on the live page
let eventSource = null;
let liveReconnectTimer = null;
let lastCandleTime = 0;  // Guard against out-of-order candle updates
let _liveTradeHistory = [];  // Cached for filtering

// ── Load OANDA Candles ──────────────────────────────────────────────────────

async function loadLiveCandles(pair, tf) {
    if (!chartManager) return;

    try {
        // First load historical data from local CSV files to backfill the chart
        const histParams = new URLSearchParams({ pair, tf, limit: '5000' });
        let allCandles = [];
        let allVolumes = [];

        try {
            const histResp = await fetch(`/api/data/candles?${histParams}`);
            const histData = await histResp.json();
            if (histData.candles && histData.candles.length > 0) {
                allCandles = histData.candles.map(c => ({
                    time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
                }));
                allVolumes = histData.candles.map(c => ({
                    time: c.time, value: c.volume,
                    color: c.close >= c.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
                }));
            }
        } catch (histErr) {
            console.warn('Could not load historical backfill:', histErr);
        }

        // Then load recent live candles from OANDA (up to 5000)
        const params = new URLSearchParams({ pair, tf, count: '5000', model: (typeof currentModel !== 'undefined' ? currentModel : 'mtf') });
        const resp = await fetch(`/api/live/candles?${params}`);
        const data = await resp.json();

        if (data.candles && data.candles.length > 0) {
            const liveCandles = data.candles.map(c => ({
                time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
            }));
            const liveVolumes = data.candles.map(c => ({
                time: c.time, value: c.volume,
                color: c.close >= c.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
            }));

            if (allCandles.length > 0) {
                // Merge: use historical data up to where live data starts, then live data
                const liveStart = liveCandles[0].time;
                const histOnly = allCandles.filter(c => c.time < liveStart);
                const histVolOnly = allVolumes.filter(v => v.time < liveStart);
                allCandles = [...histOnly, ...liveCandles];
                allVolumes = [...histVolOnly, ...liveVolumes];
            } else {
                allCandles = liveCandles;
                allVolumes = liveVolumes;
            }
        }

        if (allCandles.length === 0) {
            if (typeof showToast === 'function') showToast('No candles available', 'error');
            return;
        }

        chartManager.candleSeries.setData(allCandles);
        chartManager.volumeSeries.setData(allVolumes);
        chartManager.chart.timeScale().scrollToRealTime();
        // Reset markers for fresh candle data
        chartManager.markers = [];
        // Reset candle guard to the latest candle time
        lastCandleTime = allCandles[allCandles.length - 1].time;
    } catch (err) {
        if (typeof showToast === 'function') showToast('Failed to load live candles: ' + err.message, 'error');
    }
}

// ── SSE Connection ──────────────────────────────────────────────────────────

function connectSSE() {
    disconnectSSE();

    const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
    eventSource = new EventSource(`/api/live/stream?model=${encodeURIComponent(model)}`);

    eventSource.onopen = () => {
        setText('connection-status',  '');
        const el = document.getElementById('connection-status');
        if (el) el.innerHTML = '<i class="bi bi-circle-fill" style="color:var(--wt-green);font-size:6px;vertical-align:middle"></i> Live Connected';
        const dot = document.getElementById('live-status-dot');
        if (dot) dot.className = 'wt-status-dot online';
    };

    eventSource.onerror = () => {
        const el = document.getElementById('connection-status');
        if (el) el.innerHTML = '<i class="bi bi-circle-fill" style="color:var(--wt-red);font-size:6px;vertical-align:middle"></i> Reconnecting…';
        const dot = document.getElementById('live-status-dot');
        if (dot) dot.className = 'wt-status-dot offline';
    };

    // ── Candle updates ────────────────────────────────────────────────
    eventSource.addEventListener('candle', (e) => {
        if (!chartManager) return;
        try {
            const c = JSON.parse(e.data);
            const t = typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000);
            if (!t || isNaN(t)) return;
            if (t < lastCandleTime) return;
            lastCandleTime = t;
            chartManager.candleSeries.update({
                time: t, open: c.open, high: c.high, low: c.low, close: c.close,
            });
            chartManager.volumeSeries.update({
                time: t, value: c.volume,
                color: c.close >= c.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
            });
        } catch (err) {
            console.error('candle parse error:', err);
        }
    });

    // ── Price updates ─────────────────────────────────────────────────
    eventSource.addEventListener('price', (e) => {
        try {
            const p = JSON.parse(e.data);
            setText('live-bid', p.bid.toFixed(3));
            setText('live-ask', p.ask.toFixed(3));
            const spreadPips = (p.spread / 0.01).toFixed(1);
            setText('live-spread', spreadPips + ' pips');
            setText('live-bid-sidebar', p.bid.toFixed(3));
            setText('live-ask-sidebar', p.ask.toFixed(3));
            setText('live-spread-sidebar', spreadPips + ' pips');
        } catch (err) {
            console.error('price parse error:', err);
        }
    });

    // ── Model signal ──────────────────────────────────────────────────
    eventSource.addEventListener('signal', (e) => {
        try {
            const s = JSON.parse(e.data);
            const badge = document.getElementById('live-signal-badge');
            if (badge) {
                badge.textContent = s.signal;
                badge.className = 'wt-signal-badge ' + s.signal.toLowerCase();
                badge.style.animation = 'none';
                badge.offsetHeight;
                badge.style.animation = 'signal-flash 0.6s ease';
            }

            setText('live-confidence', (s.confidence * 100).toFixed(1) + '%');
            setText('live-alignment', (s.alignment * 100).toFixed(1) + '%');
            setText('live-sl', s.sl_pips.toFixed(1));
            setText('live-tp', s.tp_pips.toFixed(1));
            const ts = s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—';
            setText('live-signal-time', 'Last signal: ' + ts);

            // Mirror to sidebar
            const sidebarBadge = document.getElementById('live-signal-badge-sidebar');
            if (sidebarBadge) {
                sidebarBadge.textContent = s.signal;
                sidebarBadge.className = 'wt-signal-badge ' + s.signal.toLowerCase();
            }
            setText('live-confidence-sidebar', (s.confidence * 100).toFixed(1) + '%');
            setText('live-alignment-sidebar', (s.alignment * 100).toFixed(1) + '%');

            if (typeof showToast === 'function') {
                showToast(`Signal: ${s.signal} (${(s.confidence * 100).toFixed(0)}% conf)`,
                    s.signal === 'BUY' ? 'success' : s.signal === 'SELL' ? 'error' : 'info');
            }

            // Add signal marker on chart
            if (chartManager && s.signal !== 'HOLD') {
                chartManager.addSignalMarker(s);
            }

            appendSignalLog(s);
        } catch (err) {
            console.error('signal parse error:', err);
        }
    });

    // ── Account updates ───────────────────────────────────────────────
    eventSource.addEventListener('account', (e) => {
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
            if (s.status === 'error' && typeof showToast === 'function') {
                showToast('Stream error: ' + s.error, 'error');
            }
        } catch (err) {}
    });

    // ── Trade executed (auto-trade) ───────────────────────────────────
    eventSource.addEventListener('trade_executed', (e) => {
        try {
            const t = JSON.parse(e.data);
            // Add entry marker on chart
            if (chartManager) {
                chartManager.addLiveTradeMarker(t);
            }
            appendSignalTradeLog(t);
            const accountLabel = t.account === 'live' ? 'LIVE' : 'Demo';
            if (typeof showToast === 'function') {
                showToast(`Trade ${t.signal} ${t.pair} @ ${t.entry_price} (${accountLabel})`,
                    t.signal === 'BUY' ? 'success' : 'error');
            }
            refreshAccount();
            loadTradeHistory();
        } catch (err) {
            console.error('trade_executed parse error:', err);
        }
    });

    // ── Trade closed ──────────────────────────────────────────────────
    eventSource.addEventListener('trade_closed', (e) => {
        try {
            const t = JSON.parse(e.data);
            // Add exit marker on chart
            if (chartManager) {
                chartManager.addLiveExitMarker(t);
            }
            const accountLabel = t.account === 'live' ? 'LIVE' : 'Demo';
            if (typeof showToast === 'function') {
                showToast(`Position closed (${accountLabel}): ${t.reason}`, 'info');
            }
            refreshAccount();
            loadTradeHistory();
        } catch (err) {}
    });

    // ── LLM Arbiter decision ────────────────────────────────────────────
    eventSource.addEventListener('arbiter', (e) => {
        try {
            const data = JSON.parse(e.data);
            if (typeof handleArbiterSSE === 'function') {
                handleArbiterSSE(data);
            }
            // Show toast for veto/override
            if (data.action === 'VETO') {
                if (typeof showToast === 'function') {
                    showToast(`LLM Arbiter VETOED ${data.original_signal}: ${(data.reasoning || '').substring(0, 60)}`, 'warning');
                }
            } else if (data.action === 'OVERRIDE') {
                if (typeof showToast === 'function') {
                    showToast(`LLM Arbiter OVERRIDE: ${data.original_signal} → ${data.modified_signal}`, 'warning');
                }
            }
        } catch (err) {}
    });

    // ── LLM Market Inspection result ────────────────────────────────────
    eventSource.addEventListener('inspection', (e) => {
        try {
            const data = JSON.parse(e.data);
            if (typeof displayInspectionResult === 'function') {
                displayInspectionResult(data);
            }
            if (data.trade_action && typeof showToast === 'function') {
                showToast(`LLM Inspection: ${data.trade_action.signal} recommended via ${data.trade_action.model_id}`, 'info');
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
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/account?model=${encodeURIComponent(model)}`);
        const a = await resp.json();
        if (a.error) {
            if (typeof showToast === 'function') showToast('OANDA: ' + a.error, 'error');
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
        if (typeof showToast === 'function') showToast('Failed to load account: ' + err.message, 'error');
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
        const accountColor = t.account === 'live' ? 'var(--wt-green)' : '#58a6ff';
        const initialSl = t.initial_stop_loss ? t.initial_stop_loss.toFixed(3) : '—';
        const currentSl = t.stop_loss ? t.stop_loss.toFixed(3) : '—';
        const tp = t.take_profit ? t.take_profit.toFixed(3) : '—';
        const tsl = t.trailing_stop_loss ? t.trailing_stop_loss.toFixed(3) : null;

        // Check if TSL has moved from initial SL
        const slMoved = t.initial_stop_loss && t.stop_loss &&
            Math.abs(t.stop_loss - t.initial_stop_loss) > 0.001;
        const slLabel = slMoved ? 'TSL' : 'SL';
        const slHighlight = slMoved ? 'color:#ff9800;font-weight:600' : '';

        html += `
            <div style="padding:0.5rem;border-bottom:1px solid var(--wt-border)">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                    <span>
                        <span style="color:${color};font-weight:700;font-size:0.85rem">${dir}</span>
                        <span style="color:var(--wt-text);font-size:0.78rem;margin-left:6px">${t.instrument || ''}</span>
                        <span style="color:${accountColor};font-size:0.65rem;margin-left:6px">${t.account ? t.account.toUpperCase() : ''}</span>
                    </span>
                    <span style="color:${pnlColor};font-weight:700;font-size:0.85rem">${t.unrealized_pnl >= 0 ? '+' : ''}$${t.unrealized_pnl.toFixed(2)}</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:0.72rem;color:var(--wt-text-muted)">
                    <span>Entry: <span style="color:var(--wt-text);font-weight:500">${t.price.toFixed(3)}</span></span>
                    <span>TP: <span style="color:var(--wt-green);font-weight:500">${tp}</span></span>
                    <span>Initial SL: <span style="color:var(--wt-red);font-weight:500">${initialSl}</span></span>
                    <span style="${slHighlight}">${slLabel}: <span style="font-weight:500">${currentSl}</span></span>
                </div>
                ${slMoved ? `<div style="font-size:0.68rem;color:#ff9800;margin-top:3px"><i class="bi bi-shield-check"></i> TSL active — SL moved from ${initialSl} to ${currentSl}</div>` : ''}
            </div>
        `;
    });
    container.innerHTML = html;
}

// ── Load & Render Orders ────────────────────────────────────────────────────

async function loadOrders() {
    const container = document.getElementById('live-orders-list');
    if (!container) return;

    try {
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/orders?model=${encodeURIComponent(model)}`);
        const data = await resp.json();
        const orders = data.orders || [];

        if (orders.length === 0) {
            container.innerHTML = '<div class="wt-empty-state" style="padding:0.5rem">No pending orders</div>';
            return;
        }

        let html = '';
        orders.forEach(o => {
            const typeColor = o.type === 'LIMIT' ? '#58a6ff' : o.type === 'STOP' ? '#d29922' : 'var(--wt-text)';
            html += `
                <div style="padding:0.35rem 0.5rem;border-bottom:1px solid var(--wt-border)">
                    <div style="display:flex;justify-content:space-between">
                        <span style="color:${typeColor};font-weight:600">${o.type}</span>
                        <span style="color:var(--wt-text-muted);font-size:0.68rem">${o.instrument || ''}</span>
                    </div>
                    <div style="color:var(--wt-text-muted);font-size:0.68rem">
                        Units: ${o.units} | Price: ${parseFloat(o.price || 0).toFixed(3)} | ${o.account ? o.account.toUpperCase() : ''}
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    } catch (err) {
        console.error('loadOrders error:', err);
    }
}

// ── Trade History (table-based, like backtest trade log) ────────────────────

async function loadTradeHistory() {
    const body = document.getElementById('live-trade-log-body');
    if (!body) return;

    try {
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/trade-history?count=100&model=${encodeURIComponent(model)}`);
        const data = await resp.json();
        const trades = data.trades || [];
        _liveTradeHistory = trades;

        renderLiveTradeHistoryTable(trades);

        // Plot historical trades as markers on the chart
        if (chartManager && trades.length > 0) {
            plotTradeHistoryMarkers(trades);
        }
    } catch (err) {
        console.error('loadTradeHistory error:', err);
    }
}

/**
 * Plot entry/exit markers for all historical trades on the live chart.
 */
function plotTradeHistoryMarkers(trades) {
    if (!chartManager) return;

    trades.forEach(t => {
        const isBuy = t.direction === 'BUY';

        // Entry marker
        if (t.open_time) {
            const entryTs = Math.floor(new Date(t.open_time).getTime() / 1000);
            chartManager.markers.push({
                time: entryTs,
                position: isBuy ? 'belowBar' : 'aboveBar',
                color: isBuy ? '#3fb950' : '#f85149',
                shape: isBuy ? 'arrowUp' : 'arrowDown',
                text: isBuy ? 'BUY' : 'SELL',
                size: 1,
            });
        }

        // Exit marker (only for closed trades)
        if (t.close_time && t.state === 'CLOSED') {
            const exitTs = Math.floor(new Date(t.close_time).getTime() / 1000);
            const reason = t.reason || 'EXIT';
            chartManager.markers.push({
                time: exitTs,
                position: 'aboveBar',
                color: '#58a6ff',
                shape: 'circle',
                text: reason,
                size: 1,
            });
        }
    });

    chartManager.markers.sort((a, b) => a.time - b.time);
    chartManager.candleSeries.setMarkers(chartManager.markers);
}

function renderLiveTradeHistoryTable(trades) {
    const body = document.getElementById('live-trade-log-body');
    const countEl = document.getElementById('live-trade-log-count');
    if (!body) return;

    if (!trades || trades.length === 0) {
        body.innerHTML = '<tr><td colspan="16" style="text-align:center;color:var(--wt-text-muted);padding:2rem">No trade history yet</td></tr>';
        if (countEl) countEl.textContent = '0 trades';
        return;
    }

    if (countEl) countEl.textContent = `${trades.length} trades`;

    // Compute running balance (trades arrive newest-first, process in chronological order)
    const chronological = [...trades].reverse();
    const balanceMap = new Map();
    let runningBalance = 0;
    chronological.forEach(t => {
        const pl = parseFloat(t.realized_pl || 0);
        runningBalance += pl;
        balanceMap.set(t.trade_id, runningBalance);
    });

    let html = '';
    trades.forEach((t, i) => {
        const dir = t.direction || (parseFloat(t.units) > 0 ? 'BUY' : 'SELL');
        const dirColor = dir === 'BUY' ? 'var(--wt-green)' : 'var(--wt-red)';
        const pl = parseFloat(t.realized_pl || 0);
        const plColor = pl >= 0 ? 'var(--wt-green)' : 'var(--wt-red)';
        const plStr = pl >= 0 ? `+${pl.toFixed(2)}` : pl.toFixed(2);
        const openTime = t.open_time ? new Date(t.open_time).toLocaleString() : '—';
        const closeTime = t.close_time ? new Date(t.close_time).toLocaleString() : '—';
        const entryPrice = parseFloat(t.price || 0).toFixed(3);
        const exitPrice = t.close_price ? parseFloat(t.close_price).toFixed(3) : '—';
        const initialSl = t.initial_sl != null ? parseFloat(t.initial_sl).toFixed(3) : '—';
        const sl = t.sl != null ? parseFloat(t.sl).toFixed(3) : '—';
        const tsl = t.tsl != null ? parseFloat(t.tsl).toFixed(3) : '—';
        const tp = t.tp != null ? parseFloat(t.tp).toFixed(3) : '—';
        const units = Math.abs(parseFloat(t.units || 0));
        const state = t.state === 'OPEN'
            ? '<span style="color:var(--wt-yellow)">OPEN</span>'
            : '<span style="color:var(--wt-text-muted)">CLOSED</span>';
        const reason = t.reason || '—';
        const bal = balanceMap.has(t.trade_id) ? balanceMap.get(t.trade_id).toFixed(2) : '—';
        const accountColor = t.account === 'live' ? 'var(--wt-green)' : '#58a6ff';
        const rowBg = pl > 0 ? 'rgba(63,185,80,0.04)' : pl < 0 ? 'rgba(248,81,73,0.04)' : '';

        // Highlight TSL if it moved from initial SL (trailing stop is working)
        const slMoved = t.initial_sl != null && t.sl != null && Math.abs(parseFloat(t.sl) - parseFloat(t.initial_sl)) > 0.001;
        const slColor = slMoved ? 'color:var(--wt-yellow);font-weight:600' : '';
        const tslColor = tsl !== '—' ? 'color:#ff9800;font-weight:600' : '';

        html += `<tr data-account="${t.account || ''}" style="background:${rowBg}">
            <td>${i + 1}</td>
            <td>${openTime}</td>
            <td>${closeTime}</td>
            <td style="color:${dirColor};font-weight:600">${dir}</td>
            <td>${entryPrice}</td>
            <td>${exitPrice}</td>
            <td style="color:var(--wt-text-muted)">${initialSl}</td>
            <td style="${slColor}">${sl}</td>
            <td style="${tslColor}">${tsl}</td>
            <td>${tp}</td>
            <td>${units}</td>
            <td style="color:${plColor};font-weight:600">${plStr}</td>
            <td>${reason}</td>
            <td>${bal}</td>
            <td>${state}</td>
            <td style="color:${accountColor}">${(t.account || '').toUpperCase()}</td>
        </tr>`;
    });
    body.innerHTML = html;
}

function filterLiveTradeHistory(filter) {
    if (filter === 'all') {
        renderLiveTradeHistoryTable(_liveTradeHistory);
    } else {
        const filtered = _liveTradeHistory.filter(t => t.account === filter);
        renderLiveTradeHistoryTable(filtered);
    }
}

// ── Signal & Trade Logging (Signals tab) ────────────────────────────────────

function appendSignalLog(signal) {
    const log = document.getElementById('live-signals-log');
    if (!log) return;

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

    const entries = log.querySelectorAll('.wt-signal-log-entry');
    if (entries.length > 50) {
        for (let i = 50; i < entries.length; i++) entries[i].remove();
    }
}

function appendSignalTradeLog(trade) {
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

// ── Utility ─────────────────────────────────────────────────────────────────

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ── Navigate to Trade in Trade History ──────────────────────────────────

/**
 * Called when a trade marker is clicked on the chart.
 * Switches to the Trade History tab and highlights the row.
 */
function navigateToTradeHistory(tradeIndex) {
    // Switch to Trade History tab
    const histTab = document.querySelector('[data-live-tab="live-history"]');
    if (histTab) {
        document.querySelectorAll('.wt-live-tabs .wt-tab').forEach(t => t.classList.remove('active'));
        histTab.classList.add('active');
        document.querySelectorAll('.wt-live-tab-pane').forEach(p => p.classList.remove('active'));
        const pane = document.getElementById('tab-live-history');
        if (pane) pane.classList.add('active');
    }

    // Highlight the trade row
    const body = document.getElementById('live-trade-log-body');
    if (!body) return;
    const rows = body.querySelectorAll('tr');
    rows.forEach(r => r.classList.remove('wt-trade-row-highlight'));

    if (tradeIndex >= 0 && tradeIndex < rows.length) {
        const row = rows[tradeIndex];
        row.classList.add('wt-trade-row-highlight');
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Flash effect
        setTimeout(() => row.classList.remove('wt-trade-row-highlight'), 3000);
    }
}

// ── Footer Log Terminal ─────────────────────────────────────────────────
// (Shared implementation in log-terminal.js — initLogTerminal() called from live-init.js)
