/**
 * WaveTrader Backtest Page Initializer
 * Entry point for the backtest page — creates the chart,
 * loads defaults, cached results, and wires up event listeners.
 */

// Current strategy (persisted in localStorage)
let currentBacktestStrategy = localStorage.getItem('wt-backtest-strategy') || '';

document.addEventListener('DOMContentLoaded', async () => {
    // Init TradingView chart
    chartManager = new ChartManager('priceChart');

    // Restore layout state
    restoreLayoutState();

    // Init footer log terminal
    initLogTerminal();

    // Populate strategy dropdown from API
    await populateStrategyDropdown();

    // Load defaults into form
    await loadDefaults();

    // Load candles for initial pair/tf
    const pair = document.getElementById('cfg-pair').value;
    const tf = document.getElementById('nav-tf-select').value || '1h';
    await chartManager.loadCandles(pair, tf);

    // Set up event listeners
    setupBacktestEventListeners();
    setupSidebarToggles();
    setupDragHandle();
    setupKeyboardShortcuts();
});

// Strategies announced but not yet implemented
const COMING_SOON_STRATEGIES = [
    { name: 'AMD Session Scalper', category: 'scalper' },
    { name: 'Supply & Demand Zones', category: 'swing' },
    { name: 'ICT / Smart Money Concepts', category: 'swing' },
    { name: 'ORB Breakout + Pullback', category: 'scalper' },
    { name: 'EMA Crossover + Trend', category: 'trend' },
    { name: 'Mean Reversion (Bollinger/RSI)', category: 'mean-reversion' },
    { name: 'Structure Break & Retest', category: 'swing' },
];

function _appendComingSoon(select, liveIds) {
    const announced = COMING_SOON_STRATEGIES.filter(s => !liveIds.has(s.name));
    if (!announced.length) return;
    const sep = document.createElement('option');
    sep.disabled = true;
    sep.textContent = '── Coming Soon ──────────';
    sep.style.color = '#666';
    select.appendChild(sep);
    announced.forEach(s => {
        const opt = document.createElement('option');
        opt.disabled = true;
        opt.value = '';
        opt.textContent = `🔒 ${s.name}`;
        opt.style.color = '#555';
        opt.title = `${s.category} — under development`;
        select.appendChild(opt);
    });
}

async function populateStrategyDropdown() {
    const select = document.getElementById('nav-strategy-select');
    if (!select) return;
    const liveIds = new Set();
    try {
        const resp = await fetch('/api/backtest/strategies');
        if (!resp.ok) return;
        const strategies = await resp.json();
        if (strategies && strategies.length) {
            select.innerHTML = strategies.map(s => {
                liveIds.add(s.name);
                return `<option value="${s.id}" ${s.id === currentBacktestStrategy ? 'selected' : ''}>${s.name} — ${s.author}</option>`;
            }).join('');
            if (!currentBacktestStrategy && strategies.length) {
                currentBacktestStrategy = strategies[0].id;
            }
            select.value = currentBacktestStrategy;
        } else {
            select.innerHTML = '<option value="" disabled selected>No strategies yet</option>';
        }
    } catch (err) {
        console.warn('Could not load strategy list:', err);
        select.innerHTML = '<option value="" disabled selected>No strategies yet</option>';
    }
    _appendComingSoon(select, liveIds);

    select.addEventListener('change', (e) => {
        currentBacktestStrategy = e.target.value;
        localStorage.setItem('wt-backtest-strategy', currentBacktestStrategy);
        // Fetch and render strategy-specific params
        if (typeof loadStrategyParams === 'function') {
            loadStrategyParams(currentBacktestStrategy);
        }
    });

    // Load params for the initially selected strategy
    if (currentBacktestStrategy && typeof loadStrategyParams === 'function') {
        loadStrategyParams(currentBacktestStrategy);
    }
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
