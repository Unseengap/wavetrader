/**
 * WaveTrader Footer Log Terminal
 * Shared between live and backtest pages.
 * Provides an embedded terminal-style log viewer that slides up from footer.
 */

let _logTerminalSSE = null;
let _logAutoScroll = true;
let _logEntryCount = 0;

function initLogTerminal() {
    const toggle = document.getElementById('log-terminal-toggle');
    const panel = document.getElementById('log-terminal-panel');
    const closeBtn = document.getElementById('log-terminal-close');
    const fullscreenBtn = document.getElementById('log-terminal-fullscreen');
    const clearBtn = document.getElementById('log-terminal-clear');
    const container = document.getElementById('log-terminal-content');
    if (!toggle || !panel || !container) return;

    // Toggle slide up
    toggle.addEventListener('click', () => {
        panel.classList.toggle('open');
        if (panel.classList.contains('open') && !_logTerminalSSE) {
            _loadRecentAndConnect();
        }
    });

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            panel.classList.remove('open');
            panel.classList.remove('fullscreen');
        });
    }

    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', () => {
            panel.classList.toggle('fullscreen');
            if (container && _logAutoScroll) {
                container.scrollTop = container.scrollHeight;
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            container.innerHTML = '';
            _logEntryCount = 0;
            const countEl = document.getElementById('log-terminal-count');
            if (countEl) countEl.textContent = '0';
        });
    }

    // Detect scroll to toggle auto-scroll
    container.addEventListener('scroll', () => {
        const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
        _logAutoScroll = atBottom;
    });
}

function _loadRecentAndConnect() {
    fetch('/api/logs/recent')
        .then(r => r.json())
        .then(entries => {
            entries.forEach(entry => appendLogEntry(entry));
            _connectLogSSE();
        })
        .catch(() => _connectLogSSE());
}

function _connectLogSSE() {
    if (_logTerminalSSE) return;

    _logTerminalSSE = new EventSource('/api/logs/stream');

    _logTerminalSSE.onopen = () => {
        const dots = document.querySelectorAll('.wt-log-status-dot');
        dots.forEach(d => d.className = 'wt-log-status-dot connected');
    };

    _logTerminalSSE.onmessage = (ev) => {
        try {
            const data = JSON.parse(ev.data);
            if (data.type === 'connected') return;
            appendLogEntry(data);
        } catch (e) {}
    };

    _logTerminalSSE.onerror = () => {
        const dots = document.querySelectorAll('.wt-log-status-dot');
        dots.forEach(d => d.className = 'wt-log-status-dot disconnected');
        _logTerminalSSE.close();
        _logTerminalSSE = null;
        setTimeout(_connectLogSSE, 3000);
    };
}

function appendLogEntry(entry) {
    const container = document.getElementById('log-terminal-content');
    if (!container) return;

    const level = entry.level || 'info';
    const source = entry.source || '';
    const srcCls = source.includes('oanda') ? 'oanda'
        : source.includes('wavetrader') ? 'engine'
        : 'dash';
    const srcLabel = source.includes('oanda') ? 'OANDA'
        : source.includes('wavetrader') ? 'ENGINE'
        : 'DASH';

    const msg = (entry.message || '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\bBUY\b/g, '<span style="color:#3fb950;font-weight:700">BUY</span>')
        .replace(/\bSELL\b/g, '<span style="color:#f85149;font-weight:700">SELL</span>')
        .replace(/\bFILLED\b/g, '<span style="color:#3fb950">FILLED</span>')
        .replace(/\bREJECTED\b/g, '<span style="color:#f85149;font-weight:700">REJECTED</span>')
        .replace(/\bHOLD\b/g, '<span style="color:#8b949e">HOLD</span>');

    const levelColor = level === 'error' ? '#f85149'
        : level === 'warn' || level === 'warning' ? '#d29922'
        : '#8b949e';

    const srcColor = srcCls === 'oanda' ? '#f0883e'
        : srcCls === 'engine' ? '#a5d6ff'
        : '#d2a8ff';

    const div = document.createElement('div');
    div.className = 'wt-log-line';
    div.innerHTML = `<span class="wt-log-ts">${entry.ts || ''}</span>`
        + `<span class="wt-log-level" style="color:${levelColor}">${level.toUpperCase()}</span>`
        + `<span class="wt-log-src" style="color:${srcColor}">${srcLabel}</span>`
        + `<span class="wt-log-msg">${msg}</span>`;

    container.appendChild(div);
    _logEntryCount++;

    const countEl = document.getElementById('log-terminal-count');
    if (countEl) countEl.textContent = _logEntryCount;

    // Keep max 500 entries
    while (container.children.length > 500) {
        container.removeChild(container.firstChild);
    }

    if (_logAutoScroll) {
        container.scrollTop = container.scrollHeight;
    }
}

window.initLogTerminal = initLogTerminal;
window.appendLogEntry = appendLogEntry;
