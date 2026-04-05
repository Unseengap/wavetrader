/**
 * WaveTrader Config Panel & Dashboard Controller
 * Handles config form, backtest execution, and state management.
 */

let chartManager = null;
let currentResults = null;

// ── Initialization ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    // Init TradingView chart
    chartManager = new ChartManager('priceChart');

    // Load defaults into form
    await loadDefaults();

    // Load candles for initial pair/tf
    const pair = document.getElementById('cfg-pair').value;
    const tf = document.getElementById('nav-tf-select').value || '1h';
    await chartManager.loadCandles(pair, tf);

    // Try loading cached results
    await loadCachedResults();

    // Set up event listeners
    setupEventListeners();
});


// ── Load defaults from API ──────────────────────────────────────────────────

async function loadDefaults() {
    try {
        const resp = await fetch('/api/backtest/defaults');
        const data = await resp.json();
        const cfg = data.config;

        // Populate form fields
        setField('cfg-initial-balance', cfg.initial_balance);
        setField('cfg-risk-per-trade', cfg.risk_per_trade * 100);
        setField('cfg-leverage', cfg.leverage);
        setField('cfg-spread-pips', cfg.spread_pips);
        setField('cfg-commission', cfg.commission_per_lot);
        setField('cfg-pip-value', cfg.pip_value);
        setField('cfg-min-confidence', cfg.min_confidence * 100);
        setField('cfg-atr-halt', cfg.atr_halt_multiplier);
        setField('cfg-dd-threshold', cfg.drawdown_reduce_threshold * 100);

        // Friction
        if (cfg.friction) {
            setField('cfg-slip-min', cfg.friction.slippage_min);
            setField('cfg-slip-max', cfg.friction.slippage_max);
            setField('cfg-spread-offhours', cfg.friction.spread_offhours_extra);
            setField('cfg-news-prob', cfg.friction.news_spike_prob * 100);
            setField('cfg-news-extra', cfg.friction.news_spike_extra);
            setField('cfg-lot-cap', cfg.friction.lot_cap);
        }

        // Pair dropdown
        const pairSelect = document.getElementById('cfg-pair');
        if (data.pairs) {
            pairSelect.innerHTML = data.pairs.map(p =>
                `<option value="${p}" ${p === cfg.pair ? 'selected' : ''}>${p}</option>`
            ).join('');
        }

        // Also populate nav pair selector
        const navPair = document.getElementById('nav-pair-select');
        if (navPair && data.pairs) {
            navPair.innerHTML = data.pairs.map(p =>
                `<option value="${p}" ${p === cfg.pair ? 'selected' : ''}>${p}</option>`
            ).join('');
        }

        // Update range displays
        updateRangeDisplays();
    } catch (err) {
        showToast('Failed to load defaults', 'error');
    }
}


// ── Load cached results ─────────────────────────────────────────────────────

async function loadCachedResults() {
    try {
        const resp = await fetch('/api/backtest/cached');
        if (!resp.ok) return;

        const results = await resp.json();
        if (results && !results.error) {
            currentResults = results;
            updateDashboard(results);
            showToast('Loaded cached backtest results', 'info');
        }
    } catch (err) {
        // No cached results — that's fine
    }
}


// ── Run Backtest ────────────────────────────────────────────────────────────

async function runBacktest() {
    const btn = document.getElementById('btn-run-backtest');
    btn.classList.add('loading');
    btn.disabled = true;

    // Collect config from form
    const config = collectConfig();

    try {
        showToast('Running backtest… This may take a few minutes.', 'info');

        const resp = await fetch('/api/backtest/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        const results = await resp.json();

        if (results.error) {
            showToast(`Backtest error: ${results.error}`, 'error');
            return;
        }

        currentResults = results;
        updateDashboard(results);
        showToast(`Backtest complete — ${results.metrics.total_trades} trades in ${results.elapsed_seconds}s`, 'success');
    } catch (err) {
        showToast(`Failed: ${err.message}`, 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}


// ── Collect Config from Form ────────────────────────────────────────────────

function collectConfig() {
    return {
        initial_balance: parseFloat(getField('cfg-initial-balance')),
        risk_per_trade: parseFloat(getField('cfg-risk-per-trade')) / 100,
        leverage: parseFloat(getField('cfg-leverage')),
        spread_pips: parseFloat(getField('cfg-spread-pips')),
        commission_per_lot: parseFloat(getField('cfg-commission')),
        pip_value: parseFloat(getField('cfg-pip-value')),
        min_confidence: parseFloat(getField('cfg-min-confidence')) / 100,
        atr_halt_multiplier: parseFloat(getField('cfg-atr-halt')),
        drawdown_reduce_threshold: parseFloat(getField('cfg-dd-threshold')) / 100,
        pair: document.getElementById('cfg-pair').value,
        entry_timeframe: '15min',
        friction: {
            slippage_min: parseFloat(getField('cfg-slip-min')),
            slippage_max: parseFloat(getField('cfg-slip-max')),
            spread_offhours_extra: parseFloat(getField('cfg-spread-offhours')),
            news_spike_prob: parseFloat(getField('cfg-news-prob')) / 100,
            news_spike_extra: parseFloat(getField('cfg-news-extra')),
            lot_cap: parseFloat(getField('cfg-lot-cap')),
        },
    };
}


// ── Update Dashboard ────────────────────────────────────────────────────────

function updateDashboard(results) {
    // Update metrics sidebar
    updateMetrics(results.metrics);

    // Update TradingView chart with trade markers
    if (chartManager && results.trades) {
        chartManager.setTradeMarkers(results.trades);
        chartManager.setPriceLines(results.trades);
    }

    // Update analytics charts
    renderAllAnalytics(results);

    // Show the analytics area
    document.querySelectorAll('.wt-tab-pane').forEach(p => p.classList.remove('active'));
    document.querySelector('.wt-tab-pane').classList.add('active');
    document.querySelectorAll('.wt-tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.wt-tab').classList.add('active');
}


function updateMetrics(m) {
    const set = (id, val, cls) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = val;
            if (cls) el.className = `wt-metric-value ${cls}`;
            else el.className = 'wt-metric-value';
        }
    };

    const setS = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    // Big stat cards
    setS('stat-total-trades', m.total_trades);
    setS('stat-win-rate', `${(m.win_rate * 100).toFixed(1)}%`);
    setS('stat-profit-factor', m.profit_factor.toFixed(2));
    setS('stat-sharpe', m.sharpe_ratio.toFixed(2));

    // Detail metrics
    set('metric-final-balance', `$${m.final_balance.toLocaleString('en-US', {minimumFractionDigits: 2})}`,
        m.total_pnl >= 0 ? 'positive' : 'negative');
    set('metric-total-pnl', `${m.total_pnl >= 0 ? '+' : ''}$${m.total_pnl.toLocaleString('en-US', {minimumFractionDigits: 2})}`,
        m.total_pnl >= 0 ? 'positive' : 'negative');
    set('metric-return', `${m.return_pct >= 0 ? '+' : ''}${m.return_pct.toFixed(1)}%`,
        m.return_pct >= 0 ? 'positive' : 'negative');
    set('metric-max-dd', `${(m.max_drawdown * 100).toFixed(1)}%`, 'negative');
    set('metric-winning', m.winning_trades);
    set('metric-losing', m.losing_trades);
}


// ── Event Listeners ─────────────────────────────────────────────────────────

function setupEventListeners() {
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

            if (typeof liveMode !== 'undefined' && liveMode) {
                await loadLiveCandles(pair, tf);
            } else {
                await chartManager.loadCandles(pair, tf);

                // Re-apply trade markers if we have results
                if (currentResults && currentResults.trades) {
                    chartManager.setTradeMarkers(currentResults.trades);
                    chartManager.setPriceLines(currentResults.trades);
                }
            }
        });
    });

    // Nav pair change
    document.getElementById('nav-pair-select').addEventListener('change', async (e) => {
        const pair = e.target.value;
        document.getElementById('cfg-pair').value = pair;
        const tf = document.getElementById('nav-tf-select').value || '1h';

        if (typeof liveMode !== 'undefined' && liveMode) {
            await loadLiveCandles(pair, tf);
        } else {
            await chartManager.loadCandles(pair, tf);
        }
    });

    // Config pair change → sync nav
    document.getElementById('cfg-pair').addEventListener('change', (e) => {
        document.getElementById('nav-pair-select').value = e.target.value;
    });

    // Analytics tabs
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


// ── Utility ─────────────────────────────────────────────────────────────────

function setField(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
}

function getField(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
}

function updateRangeDisplays() {
    document.querySelectorAll('.wt-range-value').forEach(span => {
        const inputId = span.dataset.for;
        const input = document.getElementById(inputId);
        if (input) {
            const suffix = span.dataset.suffix || '';
            span.textContent = input.value + suffix;
        }
    });
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `wt-toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
