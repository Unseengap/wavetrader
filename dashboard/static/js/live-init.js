/**
 * WaveTrader Live Page Initializer
 * Entry point for the live trading page — creates the chart,
 * auto-starts SSE, loads OANDA candles, and wires live event listeners.
 */

let chartManager = null;
let currentModel = localStorage.getItem('wt-selected-model') || 'mtf';

/**
 * Return the current model query parameter string, e.g. "&model=mtf".
 * Callers can append this to any URL that needs the model parameter.
 */
function modelParam(prefix = '&') {
    return `${prefix}model=${encodeURIComponent(currentModel)}`;
}

document.addEventListener('DOMContentLoaded', async () => {
    // Init TradingView chart
    chartManager = new ChartManager('priceChart');

    // Restore layout state
    restoreLayoutState();

    // Load available models into the dropdown, then start
    await loadModelSelector();

    // Wire up event listeners
    setupLiveEventListeners();
    setupLiveSidebarToggle();
    setupDragHandle();
    setupLiveSubtabs();
    setupKeyboardShortcuts();

    // Init live config panel
    setupLiveConfigPanel();
    await loadLiveConfig();

    // Init footer log terminal
    initLogTerminal();

    // Init LLM Arbiter panel (load status, decisions, wire dropdown)
    if (typeof initArbiterPanel === 'function') initArbiterPanel();

    // Auto-start live mode on page load
    await startLiveMode();
});

async function loadModelSelector() {
    const select = document.getElementById('nav-model-select');
    if (!select) return;

    try {
        const resp = await fetch('/api/live/models');
        const data = await resp.json();
        const models = data.models || [];
        const defaultId = data.default || 'mtf';

        if (models.length > 0) {
            select.innerHTML = '';
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.name;
                if (m.description) opt.title = m.description;
                select.appendChild(opt);
            });
        }

        // Restore previously selected model (if still valid)
        const stored = localStorage.getItem('wt-selected-model');
        if (stored && models.some(m => m.id === stored)) {
            currentModel = stored;
        } else {
            currentModel = defaultId;
        }
        select.value = currentModel;

        // On change: switch model view
        select.addEventListener('change', async (e) => {
            currentModel = e.target.value;
            localStorage.setItem('wt-selected-model', currentModel);
            showToast(`Switching to ${select.options[select.selectedIndex].text}…`, 'info');
            await switchModel();
        });
    } catch (err) {
        console.warn('Could not load model list:', err);
    }
}

async function switchModel() {
    // Disconnect current SSE, restart with new model
    disconnectSSE();
    await startLiveMode();
    // Refresh arbiter panel for the new model
    if (typeof initArbiterPanel === 'function') initArbiterPanel();
}

async function startLiveMode() {
    const pair = document.getElementById('nav-pair-select').value || 'GBP/JPY';
    const tf = document.getElementById('nav-tf-select').value || '15min';

    // 1. Start the server-side stream for the selected model
    try {
        await fetch('/api/live/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pair, timeframe: '15min', model: currentModel }),
        });
    } catch (err) {
        showToast('Failed to start live stream: ' + err.message, 'error');
    }

    // 2. Load initial candles from OANDA
    await loadLiveCandles(pair, tf);

    // 3. Connect SSE
    connectSSE();

    // 4. Load account
    await refreshAccount();

    // 5. Load trade history
    await loadTradeHistory();

    // 6. Load orders
    await loadOrders();

    const modelSelect = document.getElementById('nav-model-select');
    const modelName = modelSelect ? modelSelect.options[modelSelect.selectedIndex].text : currentModel;
    showToast(`Live mode active — ${modelName} streaming from OANDA`, 'success');
}

function setupLiveEventListeners() {
    // Timeframe buttons
    document.querySelectorAll('.wt-tf-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            document.querySelectorAll('.wt-tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tf = btn.dataset.tf;
            const pair = document.getElementById('nav-pair-select').value;
            document.getElementById('nav-tf-select').value = tf;
            await loadLiveCandles(pair, tf);
        });
    });

    // Nav pair change
    document.getElementById('nav-pair-select').addEventListener('change', async (e) => {
        const pair = e.target.value;
        const tf = document.getElementById('nav-tf-select').value || '15min';
        await loadLiveCandles(pair, tf);
    });

    // Live hub sub-tabs
    document.querySelectorAll('.wt-live-tabs .wt-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.liveTab;
            document.querySelectorAll('.wt-live-tabs .wt-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.wt-live-tab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${target}`).classList.add('active');
        });
    });

    // Trade history filter buttons
    document.querySelectorAll('.wt-history-filter').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.wt-history-filter').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterLiveTradeHistory(btn.dataset.filter);
        });
    });

    // Fetch auto-trade status to show badges
    fetch(`/api/live/auto-trade?model=${encodeURIComponent(currentModel)}`)
        .then(r => r.json())
        .then(data => {
            const liveBadge = document.getElementById('auto-trade-live-badge');
            if (liveBadge && data.live_active) {
                liveBadge.style.display = 'inline';
            }
        })
        .catch(() => {});
}

function setupLiveSubtabs() {
    // Trades tab sub-tabs (Positions / Orders)
    document.querySelectorAll('.wt-subtab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.subtab;
            document.querySelectorAll('.wt-subtab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.wt-subtab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(`subtab-${target}`).classList.add('active');
        });
    });
}

function setupLiveSidebarToggle() {
    const layout = document.getElementById('wt-layout');

    // Left sidebar toggle
    const leftBtn = document.getElementById('toggle-left-sidebar');
    if (leftBtn) {
        leftBtn.addEventListener('click', () => {
            layout.classList.toggle('left-collapsed');
            leftBtn.classList.toggle('active', layout.classList.contains('left-collapsed'));
            saveLayoutState();
            triggerChartResize();
        });
    }

    // Right sidebar toggle
    const rightBtn = document.getElementById('toggle-right-sidebar');
    if (rightBtn) {
        rightBtn.addEventListener('click', () => {
            layout.classList.toggle('right-collapsed');
            rightBtn.classList.toggle('active', layout.classList.contains('right-collapsed'));
            saveLayoutState();
            triggerChartResize();
        });
    }
}

// ── Shared helpers (replicated for standalone live page) ────────────────

function restoreLayoutState() {
    try {
        const raw = localStorage.getItem('wt-layout-state');
        if (!raw) return;
        const state = JSON.parse(raw);
        const layout = document.getElementById('wt-layout');

        if (state.leftCollapsed) {
            layout.classList.add('left-collapsed');
            const btn = document.getElementById('toggle-left-sidebar');
            if (btn) btn.classList.add('active');
        }
        if (state.rightCollapsed) {
            layout.classList.add('right-collapsed');
            const btn = document.getElementById('toggle-right-sidebar');
            if (btn) btn.classList.add('active');
        }
        if (state.chartRatio) {
            applyChartRatio(state.chartRatio);
        }
    } catch (e) {}
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

function triggerChartResize() {
    setTimeout(() => {
        if (chartManager && chartManager.chart) {
            const container = chartManager.container;
            chartManager.chart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        }
    }, 320);
}

function applyChartRatio(ratio) {
    const chartArea = document.getElementById('chart-area');
    const centralPanel = document.getElementById('central-panel');
    if (!chartArea || !centralPanel) return;
    ratio = Math.max(0.15, Math.min(0.95, ratio));
    chartArea.style.flex = `${ratio} 1 0`;
    centralPanel.style.flex = `${1 - ratio} 1 0`;
    if (ratio >= 0.95) {
        centralPanel.classList.add('collapsed');
    } else {
        centralPanel.classList.remove('collapsed');
    }
}

function setupDragHandle() {
    const handle = document.getElementById('drag-handle');
    const chartArea = document.getElementById('chart-area');
    const centralPanel = document.getElementById('central-panel');
    const main = document.getElementById('wt-main');
    if (!handle || !chartArea || !centralPanel || !main) return;

    let dragging = false;
    let startY = 0;
    let startChartH = 0;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        startY = e.clientY;
        startChartH = chartArea.getBoundingClientRect().height;
        handle.classList.add('dragging');
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const dy = e.clientY - startY;
        const mainH = main.getBoundingClientRect().height;
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
        const available = main.getBoundingClientRect().height - 40;
        const ratio = chartArea.getBoundingClientRect().height / available;
        localStorage.setItem('wt-chart-ratio', ratio.toFixed(3));
        saveLayoutState();
        triggerChartResize();
    });

    handle.addEventListener('dblclick', () => {
        if (centralPanel.classList.contains('collapsed')) {
            centralPanel.classList.remove('collapsed');
            applyChartRatio(0.5);
        } else {
            centralPanel.classList.add('collapsed');
            applyChartRatio(1.0);
        }
        localStorage.setItem('wt-chart-ratio', centralPanel.classList.contains('collapsed') ? '1.0' : '0.5');
        saveLayoutState();
        setTimeout(triggerChartResize, 50);
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === '[') {
            e.preventDefault();
            const btn = document.getElementById('toggle-left-sidebar');
            if (btn) btn.click();
        }
        if (e.key === ']') {
            e.preventDefault();
            const btn = document.getElementById('toggle-right-sidebar');
            if (btn) btn.click();
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

// ── Live Config Panel ──────────────────────────────────────────────────────

const LIVE_CONFIG_DEFAULTS = {
    min_confidence: 0.52,
    risk_per_trade: 0.10,
    atr_halt_multiplier: 3.0,
    drawdown_reduce_threshold: 0.10,
    friction: {
        slippage_min: 0.5,
        slippage_max: 3.0,
        spread_offhours_extra: 2.5,
        news_spike_prob: 0.05,
        news_spike_extra: 5.0,
        lot_cap: 2.0,
    },
};

function _liveField(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
}

function collectLiveConfig() {
    return {
        min_confidence: parseFloat(_liveField('live-cfg-min-confidence')) / 100,
        risk_per_trade: parseFloat(_liveField('live-cfg-risk-per-trade')) / 100,
        atr_halt_multiplier: parseFloat(_liveField('live-cfg-atr-halt')),
        drawdown_reduce_threshold: parseFloat(_liveField('live-cfg-dd-threshold')) / 100,
        friction: {
            slippage_min: parseFloat(_liveField('live-cfg-slip-min')),
            slippage_max: parseFloat(_liveField('live-cfg-slip-max')),
            spread_offhours_extra: parseFloat(_liveField('live-cfg-spread-offhours')),
            news_spike_prob: parseFloat(_liveField('live-cfg-news-prob')) / 100,
            news_spike_extra: parseFloat(_liveField('live-cfg-news-extra')),
            lot_cap: parseFloat(_liveField('live-cfg-lot-cap')),
        },
    };
}

function populateLiveConfig(cfg) {
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    };
    setVal('live-cfg-risk-per-trade', Math.round((cfg.risk_per_trade || 0.10) * 100 * 4) / 4);
    setVal('live-cfg-min-confidence', Math.round((cfg.min_confidence || 0.52) * 100));
    setVal('live-cfg-atr-halt', cfg.atr_halt_multiplier || 3.0);
    setVal('live-cfg-dd-threshold', Math.round((cfg.drawdown_reduce_threshold || 0.10) * 100));
    const f = cfg.friction || {};
    setVal('live-cfg-slip-min', f.slippage_min ?? 0.5);
    setVal('live-cfg-slip-max', f.slippage_max ?? 3.0);
    setVal('live-cfg-spread-offhours', f.spread_offhours_extra ?? 2.5);
    setVal('live-cfg-news-prob', Math.round((f.news_spike_prob ?? 0.05) * 100));
    setVal('live-cfg-news-extra', f.news_spike_extra ?? 5.0);
    setVal('live-cfg-lot-cap', f.lot_cap ?? 2.0);
    updateLiveRangeDisplays();
}

function updateLiveRangeDisplays() {
    document.querySelectorAll('#sidebar-left .wt-range-value[data-for]').forEach(span => {
        const input = document.getElementById(span.dataset.for);
        if (!input) return;
        const suffix = span.dataset.suffix || '';
        span.textContent = input.value + suffix;
    });
}

async function loadLiveConfig() {
    try {
        const resp = await fetch(`/api/live/config?model=${encodeURIComponent(currentModel)}`);
        if (resp.ok) {
            const cfg = await resp.json();
            populateLiveConfig(cfg);
        }
    } catch (err) {
        console.warn('Could not load live config:', err);
    }
}

async function applyLiveConfig() {
    const btn = document.getElementById('btn-apply-live-config');
    if (!btn) return;
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.wt-spinner');
    btn.disabled = true;
    if (btnText) btnText.style.display = 'none';
    if (spinner) spinner.style.display = 'inline-block';

    try {
        const cfg = collectLiveConfig();
        const resp = await fetch(`/api/live/config?model=${encodeURIComponent(currentModel)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg),
        });
        if (resp.ok) {
            const updated = await resp.json();
            populateLiveConfig(updated);
            showToast('Live configuration applied', 'success');
        } else {
            showToast('Failed to apply live config', 'error');
        }
    } catch (err) {
        showToast('Error: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        if (btnText) btnText.style.display = '';
        if (spinner) spinner.style.display = 'none';
    }
}

function setupLiveConfigPanel() {
    // Apply button
    const applyBtn = document.getElementById('btn-apply-live-config');
    if (applyBtn) applyBtn.addEventListener('click', applyLiveConfig);

    // Reset button
    const resetBtn = document.getElementById('btn-reset-live-defaults');
    if (resetBtn) {
        resetBtn.addEventListener('click', async () => {
            populateLiveConfig(LIVE_CONFIG_DEFAULTS);
            await applyLiveConfig();
        });
    }

    // Range slider live display updates
    document.querySelectorAll('#sidebar-left input[type="range"]').forEach(input => {
        input.addEventListener('input', updateLiveRangeDisplays);
    });

    // Sync currency pair dropdown with nav pair selector
    const cfgPair = document.getElementById('live-cfg-pair');
    const navPair = document.getElementById('nav-pair-select');
    if (cfgPair && navPair) {
        cfgPair.value = navPair.value;
        cfgPair.addEventListener('change', (e) => {
            navPair.value = e.target.value;
            navPair.dispatchEvent(new Event('change'));
        });
        navPair.addEventListener('change', () => {
            cfgPair.value = navPair.value;
        });
    }
}
