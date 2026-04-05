/**
 * WaveTrader Backtest History
 * Save, load, list, and compare backtest results.
 */

// ── Save Current Backtest ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const saveBtn = document.getElementById('btn-save-backtest');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveCurrentBacktest);
    }
    // Load existing saved backtests
    loadSavedBacktestsList();
});

async function saveCurrentBacktest() {
    if (!currentResults) {
        showToast('No backtest results to save', 'error');
        return;
    }

    try {
        const resp = await fetch('/api/backtest/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResults),
        });
        const data = await resp.json();
        if (data.error) {
            showToast(`Save failed: ${data.error}`, 'error');
            return;
        }
        showToast(`Backtest saved: ${data.run_id}`, 'success');
        loadSavedBacktestsList();
    } catch (err) {
        showToast(`Save failed: ${err.message}`, 'error');
    }
}


// ── Load Saved Backtests List ───────────────────────────────────────────────

async function loadSavedBacktestsList() {
    const container = document.getElementById('saved-backtests-list');
    if (!container) return;

    try {
        const resp = await fetch('/api/backtest/list');
        const entries = await resp.json();

        if (!entries || entries.length === 0) {
            container.innerHTML = '<div class="wt-empty-state" style="padding:1rem"><p>No saved backtests yet</p></div>';
            return;
        }

        container.innerHTML = entries.map(entry => {
            const m = entry.metrics || {};
            const cfg = entry.config || {};
            const date = entry.saved_at ? new Date(entry.saved_at).toLocaleString('en-GB', {
                day: '2-digit', month: 'short', year: '2-digit',
                hour: '2-digit', minute: '2-digit',
            }) : '—';

            return `
                <div class="wt-saved-entry" data-run-id="${entry.run_id}">
                    <div class="wt-saved-entry-header">
                        <span class="wt-saved-entry-id">${entry.run_id}</span>
                        <span class="wt-saved-entry-date">${date}</span>
                    </div>
                    <div class="wt-saved-entry-metrics">
                        <span>Pair: <b>${cfg.pair || '—'}</b></span>
                        <span>Trades: <b>${m.total_trades || 0}</b></span>
                        <span>WR: <b>${m.win_rate ? (m.win_rate * 100).toFixed(1) + '%' : '—'}</b></span>
                        <span>PF: <b>${m.profit_factor ? m.profit_factor.toFixed(2) : '—'}</b></span>
                        <span>Return: <b style="color:${(m.return_pct || 0) >= 0 ? 'var(--wt-green)' : 'var(--wt-red)'}">${m.return_pct ? m.return_pct.toFixed(1) + '%' : '—'}</b></span>
                    </div>
                    <div class="wt-saved-entry-actions">
                        <button class="wt-btn-small" onclick="loadSavedBacktest('${entry.run_id}')">
                            <i class="bi bi-box-arrow-in-up-right"></i> Load
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = '<div class="wt-empty-state" style="padding:1rem"><p>Failed to load saved backtests</p></div>';
    }
}


// ── Load a Saved Backtest ───────────────────────────────────────────────────

async function loadSavedBacktest(runId) {
    try {
        const resp = await fetch(`/api/backtest/load/${runId}`);
        const results = await resp.json();

        if (results.error) {
            showToast(`Load failed: ${results.error}`, 'error');
            return;
        }

        currentResults = results;
        updateDashboard(results);

        // Restore config panel
        if (results.config) {
            const cfg = results.config;
            setField('cfg-initial-balance', cfg.initial_balance);
            setField('cfg-risk-per-trade', (cfg.risk_per_trade || 0.01) * 100);
            setField('cfg-leverage', cfg.leverage);
            setField('cfg-spread-pips', cfg.spread_pips);
            setField('cfg-commission', cfg.commission_per_lot);
            setField('cfg-pip-value', cfg.pip_value);
            setField('cfg-min-confidence', (cfg.min_confidence || 0.55) * 100);
            setField('cfg-atr-halt', cfg.atr_halt_multiplier);
            setField('cfg-dd-threshold', (cfg.drawdown_reduce_threshold || 0.10) * 100);
            if (cfg.friction) {
                setField('cfg-slip-min', cfg.friction.slippage_min);
                setField('cfg-slip-max', cfg.friction.slippage_max);
                setField('cfg-spread-offhours', cfg.friction.spread_offhours_extra);
                setField('cfg-news-prob', (cfg.friction.news_spike_prob || 0.05) * 100);
                setField('cfg-news-extra', cfg.friction.news_spike_extra);
                setField('cfg-lot-cap', cfg.friction.lot_cap);
            }
            updateRangeDisplays();
        }

        // Switch to equity tab
        switchToTopTab('backtest-hub');
        showToast(`Loaded backtest: ${runId}`, 'success');
    } catch (err) {
        showToast(`Load failed: ${err.message}`, 'error');
    }
}

// Export
window.loadSavedBacktest = loadSavedBacktest;
window.loadSavedBacktestsList = loadSavedBacktestsList;
