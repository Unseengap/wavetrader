/**
 * WaveTrader Config Panel & Dashboard Controller
 * Handles config form, backtest execution, sidebar toggles,
 * drag handle, and state management.
 * Shared between backtest page and backtest-init.js.
 */

let chartManager = null;
let currentResults = null;
let replayController = null;

// ── Initialization is handled by backtest-init.js ───────────────────────────


// ── Load defaults from API ──────────────────────────────────────────────────

async function loadDefaults() {
    try {
        const resp = await fetch('/api/backtest/defaults');
        const data = await resp.json();
        const cfg = data.config;

        // Populate form fields (skip OANDA-locked fields)
        if (!isLocked('cfg-initial-balance')) setField('cfg-initial-balance', cfg.initial_balance);
        setField('cfg-risk-per-trade', cfg.risk_per_trade * 100);
        if (!isLocked('cfg-leverage')) setField('cfg-leverage', cfg.leverage);
        if (!isLocked('cfg-spread-pips')) setField('cfg-spread-pips', cfg.spread_pips);
        if (!isLocked('cfg-commission')) setField('cfg-commission', cfg.commission_per_lot);
        if (!isLocked('cfg-pip-value')) setField('cfg-pip-value', cfg.pip_value);
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


// ── Run Backtest ────────────────────────────────────────────────────────────

async function runBacktest() {
    const btn = document.getElementById('btn-run-backtest');
    btn.classList.add('loading');
    btn.disabled = true;

    // Collect config from form
    const config = collectConfig();
    // Include selected strategy — this triggers the strategy backtest path
    if (typeof currentBacktestStrategy !== 'undefined' && currentBacktestStrategy) {
        config.strategy = currentBacktestStrategy;
    }

    // AI confirmation toggle
    const aiCheckbox = document.getElementById('cfg-ai-confirm');
    if (aiCheckbox) {
        config.ai_confirm = aiCheckbox.checked;
    }

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
    const cfg = {
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

    // Include strategy params if a strategy is selected
    const strategyParams = collectStrategyParams();
    if (strategyParams && Object.keys(strategyParams).length > 0) {
        cfg.strategy_params = strategyParams;
    }

    return cfg;
}


// ── Strategy Parameters (dynamic per-strategy form) ─────────────────────────

/**
 * Render strategy-specific parameter inputs from the /params API.
 * @param {Array} params — Array of {name, label, type, default, min, max, step, description}
 * @param {Object} meta — Strategy metadata {name, author, category}
 */
function renderStrategyParams(params, meta) {
    const section = document.getElementById('strategy-params-section');
    const container = document.getElementById('strategy-params-fields');
    if (!section || !container) return;

    if (!params || params.length === 0) {
        section.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    section.style.display = '';
    let html = '';

    if (meta && meta.name) {
        html += `<div class="wt-strategy-meta-badge">
            <span class="wt-strategy-badge">${meta.category || 'strategy'}</span>
            <span style="font-size:0.72rem;color:var(--wt-text-muted)">${meta.name}</span>
        </div>`;
    }

    for (const p of params) {
        const id = `cfg-strat-${p.name}`;
        const tooltip = p.description ? ` title="${p.description}"` : '';

        if (p.type === 'bool') {
            html += `
            <div class="wt-field wt-field-checkbox"${tooltip}>
                <label for="${id}">
                    <input type="checkbox" id="${id}" ${p.default ? 'checked' : ''}>
                    ${p.label}
                </label>
            </div>`;
        } else if (p.min !== null && p.max !== null && p.type === 'float') {
            // Range slider for bounded floats
            const step = p.step || (p.type === 'int' ? 1 : 0.1);
            const suffix = p.label.includes('pips') ? ' pips' : p.label.includes('%') ? '%' : '';
            html += `
            <div class="wt-field"${tooltip}>
                <label for="${id}">
                    ${p.label}
                    <span class="wt-range-value" data-for="${id}" data-suffix="${suffix}">${p.default}${suffix}</span>
                </label>
                <input type="range" id="${id}" min="${p.min}" max="${p.max}" step="${step}" value="${p.default}">
            </div>`;
        } else {
            // Number input
            const step = p.step || (p.type === 'int' ? 1 : 0.1);
            const minAttr = p.min !== null ? ` min="${p.min}"` : '';
            const maxAttr = p.max !== null ? ` max="${p.max}"` : '';
            html += `
            <div class="wt-field"${tooltip}>
                <label for="${id}">${p.label}</label>
                <input type="number" id="${id}" value="${p.default}" step="${step}"${minAttr}${maxAttr}>
            </div>`;
        }
    }

    container.innerHTML = html;

    // Wire up range display updates for new range inputs
    container.querySelectorAll('input[type="range"]').forEach(input => {
        const display = container.querySelector(`.wt-range-value[data-for="${input.id}"]`);
        if (display) {
            const suffix = display.dataset.suffix || '';
            input.addEventListener('input', () => {
                display.textContent = input.value + suffix;
            });
        }
    });
}

/**
 * Collect strategy-specific parameter values from the dynamic form.
 * @returns {Object} — Dict of param_name → value (typed correctly)
 */
function collectStrategyParams() {
    const container = document.getElementById('strategy-params-fields');
    if (!container) return {};

    const params = {};
    container.querySelectorAll('[id^="cfg-strat-"]').forEach(input => {
        const name = input.id.replace('cfg-strat-', '');
        if (input.type === 'checkbox') {
            params[name] = input.checked;
        } else if (input.type === 'range' || input.type === 'number') {
            params[name] = parseFloat(input.value);
        }
    });
    return params;
}

/**
 * Fetch and render strategy params for the given strategy id.
 */
async function loadStrategyParams(strategyId) {
    if (!strategyId) {
        renderStrategyParams([], null);
        return;
    }
    try {
        const resp = await fetch(`/api/backtest/strategies/${encodeURIComponent(strategyId)}/params`);
        if (!resp.ok) { renderStrategyParams([], null); return; }
        const data = await resp.json();
        renderStrategyParams(data.params || [], data.meta || null);
    } catch (err) {
        console.warn('Failed to load strategy params:', err);
        renderStrategyParams([], null);
    }
}


// ── Update Dashboard ────────────────────────────────────────────────────────

function updateDashboard(results) {
    // Update metrics sidebar
    updateMetrics(results.metrics);

    // Update TradingView chart with trade markers (SL/TP shown on click only)
    if (chartManager && results.trades) {
        chartManager.setTradeMarkers(results.trades);
    }

    // Update analytics charts
    renderAllAnalytics(results);

    // Update trade log
    if (typeof renderTradeLog === 'function') {
        renderTradeLog(results.trades);
    }

    // Initialize backtest replay controller if candle data is available
    initReplayController(results);

    // Show the analytics area
    document.querySelectorAll('.wt-tab-pane').forEach(p => p.classList.remove('active'));
    const firstPane = document.querySelector('.wt-tab-pane');
    if (firstPane) firstPane.classList.add('active');
    document.querySelectorAll('.wt-tab').forEach(t => t.classList.remove('active'));
    const firstTab = document.querySelector('.wt-tab');
    if (firstTab) firstTab.classList.add('active');
}


// ── Replay Controller ───────────────────────────────────────────────────────

function initReplayController(results) {
    // Destroy previous replay if exists
    if (replayController) {
        replayController.destroy();
        replayController = null;
    }

    const candles = results.candles;
    const trades = results.trades;
    if (!candles || candles.length === 0 || !chartManager) {
        // Hide toolbar if no candle data
        const toolbar = document.getElementById('replay-toolbar');
        if (toolbar) toolbar.style.display = 'none';
        return;
    }

    const initialBalance = results.config?.initial_balance || 25000;
    replayController = new BacktestReplayController(chartManager, candles, trades, initialBalance);
    replayController.bindUI();

    // Show the toolbar (hidden by default, user clicks Replay to start)
    const toolbar = document.getElementById('replay-toolbar');
    if (toolbar) toolbar.style.display = '';

    // The replay-play button already exists in the HTML toolbar.
    // Show a toast hinting at the feature.
    showToast('Replay ready — press ▶ below the chart to animate the backtest', 'info');
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


// ── Event listeners are set up by backtest-init.js (setupBacktestEventListeners) ──


// ── Utility ─────────────────────────────────────────────────────────────────

function setField(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
}

function getField(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
}

function isLocked(id) {
    const el = document.getElementById(id);
    return el && el.disabled;
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


// ═══════════════════════════════════════════════════════════════════════════
// Sidebar Toggles
// ═══════════════════════════════════════════════════════════════════════════

function setupSidebarToggles() {
    const layout = document.getElementById('wt-layout');
    const leftBtn = document.getElementById('toggle-left-sidebar');
    const rightBtn = document.getElementById('toggle-right-sidebar');

    leftBtn.addEventListener('click', () => {
        layout.classList.toggle('left-collapsed');
        leftBtn.classList.toggle('active', layout.classList.contains('left-collapsed'));
        saveLayoutState();
        triggerChartResize();
    });

    rightBtn.addEventListener('click', () => {
        layout.classList.toggle('right-collapsed');
        rightBtn.classList.toggle('active', layout.classList.contains('right-collapsed'));
        saveLayoutState();
        triggerChartResize();
    });
}

function triggerChartResize() {
    // Allow CSS transition to complete
    setTimeout(() => {
        if (chartManager && chartManager.chart) {
            const container = chartManager.container;
            chartManager.chart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        }
        // Resize Plotly charts too
        document.querySelectorAll('.plotly-chart .js-plotly-plot').forEach(el => {
            Plotly.Plots.resize(el);
        });
    }, 320);
}

function saveLayoutState() {
    const layout = document.getElementById('wt-layout');
    const state = {
        leftCollapsed: layout.classList.contains('left-collapsed'),
        rightCollapsed: layout.classList.contains('right-collapsed'),
        chartRatio: parseFloat(localStorage.getItem('wt-chart-ratio') || '0.5'),
    };
    localStorage.setItem('wt-layout-state', JSON.stringify(state));
}

function restoreLayoutState() {
    try {
        const raw = localStorage.getItem('wt-layout-state');
        if (!raw) return;
        const state = JSON.parse(raw);
        const layout = document.getElementById('wt-layout');

        if (state.leftCollapsed) {
            layout.classList.add('left-collapsed');
            document.getElementById('toggle-left-sidebar').classList.add('active');
        }
        if (state.rightCollapsed) {
            layout.classList.add('right-collapsed');
            document.getElementById('toggle-right-sidebar').classList.add('active');
        }
        if (state.chartRatio) {
            applyChartRatio(state.chartRatio);
        }
    } catch (e) {
        // Ignore corrupt state
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Drag Handle (Chart / Central Panel resizer)
// ═══════════════════════════════════════════════════════════════════════════

function setupDragHandle() {
    const handle = document.getElementById('drag-handle');
    const chartArea = document.getElementById('chart-area');
    const centralPanel = document.getElementById('central-panel');
    const main = document.getElementById('wt-main');

    let dragging = false;
    let startY = 0;
    let startChartH = 0;
    let startPanelH = 0;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        startY = e.clientY;
        startChartH = chartArea.getBoundingClientRect().height;
        startPanelH = centralPanel.getBoundingClientRect().height;
        handle.classList.add('dragging');
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const dy = e.clientY - startY;
        const mainH = main.getBoundingClientRect().height;
        // Subtract legend (approx 32px) + handle (8px) from available space
        const available = mainH - 40;
        const newChartH = Math.max(200, Math.min(available - 150, startChartH + dy));
        const ratio = newChartH / available;
        applyChartRatio(ratio);
    });

    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        handle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        // Persist ratio
        const chartArea = document.getElementById('chart-area');
        const main = document.getElementById('wt-main');
        const available = main.getBoundingClientRect().height - 40;
        const ratio = chartArea.getBoundingClientRect().height / available;
        localStorage.setItem('wt-chart-ratio', ratio.toFixed(3));
        saveLayoutState();
        triggerChartResize();
    });

    // Double-click to toggle panel collapsed / 50-50
    handle.addEventListener('dblclick', () => {
        const panel = document.getElementById('central-panel');
        if (panel.classList.contains('collapsed')) {
            panel.classList.remove('collapsed');
            applyChartRatio(0.5);
        } else {
            panel.classList.add('collapsed');
            applyChartRatio(1.0);
        }
        localStorage.setItem('wt-chart-ratio', panel.classList.contains('collapsed') ? '1.0' : '0.5');
        saveLayoutState();
        setTimeout(triggerChartResize, 50);
    });
}

function applyChartRatio(ratio) {
    const chartArea = document.getElementById('chart-area');
    const centralPanel = document.getElementById('central-panel');
    ratio = Math.max(0.15, Math.min(0.95, ratio));
    chartArea.style.flex = `${ratio} 1 0`;
    centralPanel.style.flex = `${1 - ratio} 1 0`;
    if (ratio >= 0.95) {
        centralPanel.classList.add('collapsed');
    } else {
        centralPanel.classList.remove('collapsed');
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Top-level tabs removed — Live and Backtest are now separate pages.


// ═══════════════════════════════════════════════════════════════════════════
// Keyboard Shortcuts
// ═══════════════════════════════════════════════════════════════════════════

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't trigger when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === '[') {
            e.preventDefault();
            document.getElementById('toggle-left-sidebar').click();
        } else if (e.key === ']') {
            e.preventDefault();
            document.getElementById('toggle-right-sidebar').click();
        }
    });
}
