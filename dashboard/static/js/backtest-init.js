/**
 * WaveTrader Backtest Page Initializer
 * Entry point for the backtest page — creates the chart,
 * loads defaults, cached results, and wires up event listeners.
 */

// Current model (persisted in localStorage)
let currentBacktestModel = localStorage.getItem('wt-backtest-model') || 'mtf';

document.addEventListener('DOMContentLoaded', async () => {
    // Init TradingView chart
    chartManager = new ChartManager('priceChart');

    // Restore layout state
    restoreLayoutState();

    // Init footer log terminal
    initLogTerminal();

    // Populate model dropdown from API
    await populateModelDropdown();

    // Load defaults into form
    await loadDefaults();

    // Load candles for initial pair/tf
    const pair = document.getElementById('cfg-pair').value;
    const tf = document.getElementById('nav-tf-select').value || '1h';
    await chartManager.loadCandles(pair, tf);

    // Try loading cached results for the selected model
    await loadCachedResults();

    // Set up event listeners
    setupBacktestEventListeners();
    setupSidebarToggles();
    setupDragHandle();
    setupKeyboardShortcuts();
});

async function populateModelDropdown() {
    const select = document.getElementById('nav-model-select');
    if (!select) return;
    try {
        const resp = await fetch('/api/live/models');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.models && data.models.length) {
            select.innerHTML = data.models.map(m =>
                `<option value="${m.id}" ${m.id === currentBacktestModel ? 'selected' : ''}>${m.name}</option>`
            ).join('');
        }
    } catch (err) { /* keep default option */ }

    select.addEventListener('change', async (e) => {
        currentBacktestModel = e.target.value;
        localStorage.setItem('wt-backtest-model', currentBacktestModel);
        // Reload cached results for the newly selected model
        currentResults = null;
        await loadCachedResults();
    });
}

function setupBacktestEventListeners() {
    // Run backtest button
    document.getElementById('btn-run-backtest').addEventListener('click', runBacktest);

    // Reset to defaults
    document.getElementById('btn-reset-defaults').addEventListener('click', loadDefaults);

    // Timeframe buttons
    document.querySelectorAll('.wt-tf-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            document.querySelectorAll('.wt-tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tf = btn.dataset.tf;
            const pair = document.getElementById('nav-pair-select').value;
            document.getElementById('nav-tf-select').value = tf;

            await chartManager.loadCandles(pair, tf);

            // Re-apply trade markers if we have results
            if (currentResults && currentResults.trades) {
                chartManager.setTradeMarkers(currentResults.trades);
            }
        });
    });

    // Nav pair change
    document.getElementById('nav-pair-select').addEventListener('change', async (e) => {
        const pair = e.target.value;
        document.getElementById('cfg-pair').value = pair;
        const tf = document.getElementById('nav-tf-select').value || '1h';
        await chartManager.loadCandles(pair, tf);
    });

    // Config pair change → sync nav
    document.getElementById('cfg-pair').addEventListener('change', (e) => {
        document.getElementById('nav-pair-select').value = e.target.value;
    });

    // Analytics sub-tabs
    document.querySelectorAll('.wt-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.wt-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.wt-tab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${target}`).classList.add('active');

            // Trigger Plotly resize for newly visible charts
            setTimeout(() => {
                document.querySelectorAll(`#tab-${target} .plotly-chart`).forEach(el => {
                    if (el.querySelector('.js-plotly-plot')) {
                        Plotly.Plots.resize(el.querySelector('.js-plotly-plot'));
                    }
                });
            }, 50);
        });
    });

    // Range input live displays
    document.querySelectorAll('.wt-field input[type="range"]').forEach(input => {
        input.addEventListener('input', updateRangeDisplays);
    });
}

// ── Live Data Overlay for Backtest Chart ─────────────────────────────────

let backtestSSE = null;

/**
 * Connect SSE to stream live candles onto the backtest chart.
 * This appends real-time candle data after the historical data ends.
 */
function connectBacktestLiveFeed() {
    if (backtestSSE) {
        backtestSSE.close();
        backtestSSE = null;
    }

    // Start the live stream server-side first
    const pair = document.getElementById('cfg-pair')?.value || 'GBP/JPY';
    fetch('/api/live/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pair, timeframe: '15min' }),
    }).catch(() => {});

    backtestSSE = new EventSource('/api/live/stream');

    backtestSSE.addEventListener('candle', (e) => {
        if (!chartManager) return;
        try {
            const c = JSON.parse(e.data);
            const t = typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000);
            if (!t || isNaN(t)) return;
            chartManager.candleSeries.update({
                time: t, open: c.open, high: c.high, low: c.low, close: c.close,
            });
            chartManager.volumeSeries.update({
                time: t, value: c.volume,
                color: c.close >= c.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
            });
        } catch (err) {
            console.error('backtest live candle parse error:', err);
        }
    });

    backtestSSE.onerror = () => {
        // Silent reconnect (EventSource auto-reconnects)
    };
}

// Auto-connect live feed after page loads
setTimeout(() => {
    connectBacktestLiveFeed();
}, 2000);

// ── Navigate to trade from chart click (backtest version) ───────────────

/**
 * Called by ChartManager when a trade marker is clicked.
 * Uses the existing scrollToTradeRow from trade-log.js.
 */
function navigateToTradeHistory(tradeIndex) {
    if (typeof scrollToTradeRow === 'function') {
        scrollToTradeRow(tradeIndex);
    }
}
