/**
 * WaveTrader Trade Log
 * Full trade table with sorting, filtering, row-click navigation,
 * and two-way binding to the chart.
 */

let _allTrades = [];
let _sortField = 'index';
let _sortAsc = true;
let _filterText = '';

// ── Render Trade Log ────────────────────────────────────────────────────────

function renderTradeLog(trades) {
    if (!trades || trades.length === 0) {
        _allTrades = [];
        document.getElementById('trade-log-count').textContent = '0 trades';
        document.getElementById('trade-log-body').innerHTML =
            '<tr><td colspan=\"13\" style=\"text-align:center;color:var(--wt-text-muted);padding:2rem\">No trades to display</td></tr>';
        return;
    }

    // Pre-compute index and duration
    _allTrades = trades.map((t, i) => {
        const entry = t.entry_time ? new Date(t.entry_time) : null;
        const exit = t.exit_time ? new Date(t.exit_time) : null;
        let duration = '';
        if (entry && exit) {
            const hours = (exit - entry) / 3600000;
            if (hours < 1) duration = `${Math.round(hours * 60)}m`;
            else if (hours < 24) duration = `${hours.toFixed(1)}h`;
            else duration = `${(hours / 24).toFixed(1)}d`;
        }
        return { ...t, _index: i + 1, _duration: duration, _durationHours: entry && exit ? (exit - entry) / 3600000 : 0 };
    });

    _sortField = 'index';
    _sortAsc = true;
    _filterText = '';
    document.getElementById('trade-log-search').value = '';
    _renderRows();
}

function _renderRows() {
    let filtered = _allTrades;

    // Apply filter
    if (_filterText) {
        const q = _filterText.toLowerCase();
        filtered = filtered.filter(t =>
            (t.direction || '').toLowerCase().includes(q) ||
            (t.exit_reason || '').toLowerCase().includes(q) ||
            (t.entry_time || '').toLowerCase().includes(q) ||
            (t.exit_time || '').toLowerCase().includes(q) ||
            String(t.pnl).includes(q)
        );
    }

    // Sort
    const sorted = [...filtered].sort((a, b) => {
        let va, vb;
        switch (_sortField) {
            case 'index': va = a._index; vb = b._index; break;
            case 'entry_time': va = a.entry_time || ''; vb = b.entry_time || ''; break;
            case 'exit_time': va = a.exit_time || ''; vb = b.exit_time || ''; break;
            case 'direction': va = a.direction || ''; vb = b.direction || ''; break;
            case 'entry_price': va = a.entry_price || 0; vb = b.entry_price || 0; break;
            case 'exit_price': va = a.exit_price || 0; vb = b.exit_price || 0; break;
            case 'stop_loss': va = a.stop_loss || 0; vb = b.stop_loss || 0; break;
            case 'take_profit': va = a.take_profit || 0; vb = b.take_profit || 0; break;
            case 'size': va = a.size || 0; vb = b.size || 0; break;
            case 'pnl': va = a.pnl || 0; vb = b.pnl || 0; break;
            case 'balance': va = a.balance || 0; vb = b.balance || 0; break;
            case 'exit_reason': va = a.exit_reason || ''; vb = b.exit_reason || ''; break;
            case 'duration': va = a._durationHours; vb = b._durationHours; break;
            default: va = a._index; vb = b._index;
        }
        if (va < vb) return _sortAsc ? -1 : 1;
        if (va > vb) return _sortAsc ? 1 : -1;
        return 0;
    });

    document.getElementById('trade-log-count').textContent = `${sorted.length} trade${sorted.length !== 1 ? 's' : ''}`;

    // Update sort indicators
    document.querySelectorAll('.wt-trade-log-table th').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
        if (th.dataset.sort === _sortField) {
            th.classList.add(_sortAsc ? 'sorted-asc' : 'sorted-desc');
        }
    });

    // Render rows (virtual: only render visible + buffer)
    const body = document.getElementById('trade-log-body');
    const fragment = document.createDocumentFragment();

    sorted.forEach((t) => {
        const tr = document.createElement('tr');
        tr.className = t.pnl > 0 ? 'winner' : 'loser';
        tr.dataset.tradeIndex = t._index - 1;

        const fmtTime = (ts) => {
            if (!ts) return '—';
            const d = new Date(ts);
            return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit' }) +
                ' ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
        };

        tr.innerHTML = `
            <td>${t._index}</td>
            <td>${fmtTime(t.entry_time)}</td>
            <td>${fmtTime(t.exit_time)}</td>
            <td class="dir-${t.direction === 'BUY' ? 'buy' : 'sell'}">${t.direction}</td>
            <td>${t.entry_price ? t.entry_price.toFixed(3) : '—'}</td>
            <td>${t.exit_price ? t.exit_price.toFixed(3) : '—'}</td>
            <td>${t.stop_loss ? t.stop_loss.toFixed(3) : '—'}</td>
            <td>${t.take_profit ? t.take_profit.toFixed(3) : '—'}</td>
            <td>${t.size}</td>
            <td>${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</td>
            <td>${t.balance != null ? '$' + t.balance.toLocaleString('en-US', {minimumFractionDigits: 2}) : '—'}</td>
            <td>${t.exit_reason || '—'}</td>
            <td>${t._duration || '—'}</td>
        `;

        // Click row → scroll chart to trade
        tr.addEventListener('click', () => {
            navigateToTrade(t._index - 1);
        });

        fragment.appendChild(tr);
    });

    body.innerHTML = '';
    body.appendChild(fragment);
}


// ── Navigation ──────────────────────────────────────────────────────────────

function navigateToTrade(index) {
    const trade = _allTrades[index];
    if (!trade) return;

    // Scroll chart to trade
    if (chartManager) {
        chartManager.scrollToTrade(trade);
    }

    // Highlight the row
    highlightTradeRow(index);
}

function highlightTradeRow(index) {
    // Remove previous highlight
    document.querySelectorAll('.wt-trade-log-table tbody tr.highlighted').forEach(
        r => r.classList.remove('highlighted')
    );
    // Add highlight to matching row
    const row = document.querySelector(`.wt-trade-log-table tbody tr[data-trade-index="${index}"]`);
    if (row) {
        row.classList.add('highlighted');
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function scrollToTradeRow(index) {
    // Switch to trade log tab
    const tab = document.querySelector('.wt-tab[data-tab="trade-log"]');
    if (tab) tab.click();

    // Highlight
    setTimeout(() => highlightTradeRow(index), 100);
}


// ── Event Listeners (sorting, filtering) ────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Column sort headers
    document.querySelectorAll('.wt-trade-log-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const field = th.dataset.sort;
            if (_sortField === field) {
                _sortAsc = !_sortAsc;
            } else {
                _sortField = field;
                _sortAsc = true;
            }
            _renderRows();
        });
    });

    // Filter input
    const searchInput = document.getElementById('trade-log-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            _filterText = e.target.value.trim();
            _renderRows();
        });
    }
});

// Export
window.renderTradeLog = renderTradeLog;
window.navigateToTrade = navigateToTrade;
window.highlightTradeRow = highlightTradeRow;
window.scrollToTradeRow = scrollToTradeRow;
