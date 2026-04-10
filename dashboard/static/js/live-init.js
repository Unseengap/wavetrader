/**
 * WaveTrader Live Page Initializer
 * Entry point for the live trading page — creates the chart,
 * auto-starts SSE, loads OANDA candles, and wires live event listeners.
 */

let chartManager = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Init TradingView chart
    chartManager = new ChartManager('priceChart');

    // Restore layout state
    restoreLayoutState();

    // Wire up event listeners
    setupLiveEventListeners();
    setupLiveSidebarToggle();
    setupDragHandle();
    setupLiveSubtabs();
    setupKeyboardShortcuts();

    // Init footer log terminal
    initLogTerminal();

    // Auto-start live mode on page load
    await startLiveMode();
});

async function startLiveMode() {
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

    showToast('Live mode active — streaming from OANDA', 'success');
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
    fetch('/api/live/auto-trade')
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
