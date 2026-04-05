/**
 * WaveTrader Analytics Charts
 * Plotly.js chart builders — replicates all 15 notebook charts.
 */
const WT_COLORS = {
    green: '#3fb950',
    red: '#f85149',
    blue: '#58a6ff',
    orange: '#d29922',
    purple: '#bc8cff',
    cyan: '#39d2c0',
    bg: '#0d1117',
    card: '#1c2128',
    grid: 'rgba(48, 54, 61, 0.4)',
    text: '#8b949e',
    textBright: '#e6edf3',
};

const WT_LAYOUT_BASE = {
    paper_bgcolor: WT_COLORS.card,
    plot_bgcolor: WT_COLORS.card,
    font: { color: WT_COLORS.text, size: 11, family: '-apple-system, sans-serif' },
    margin: { l: 50, r: 20, t: 35, b: 40 },
    xaxis: { gridcolor: WT_COLORS.grid, zerolinecolor: WT_COLORS.grid },
    yaxis: { gridcolor: WT_COLORS.grid, zerolinecolor: WT_COLORS.grid },
    modebar: { bgcolor: 'transparent', color: WT_COLORS.text },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
};

function mergLayout(overrides) {
    return { ...WT_LAYOUT_BASE, ...overrides };
}


// ═══════════════════════════════════════════════════════════════════════════
// 1. Equity Curve
// ═══════════════════════════════════════════════════════════════════════════

function renderEquityCurve(containerId, equityCurve, initialBalance) {
    const x = equityCurve.map((_, i) => i);
    const profitY = equityCurve.map(v => v >= initialBalance ? v : null);
    const lossY = equityCurve.map(v => v < initialBalance ? v : null);

    const traces = [
        {
            x, y: profitY, type: 'scatter', mode: 'lines',
            fill: 'tozeroy', fillcolor: 'rgba(63,185,80,0.08)',
            line: { color: WT_COLORS.green, width: 1.5 },
            name: 'Profit',
        },
        {
            x, y: lossY, type: 'scatter', mode: 'lines',
            fill: 'tozeroy', fillcolor: 'rgba(248,81,73,0.08)',
            line: { color: WT_COLORS.red, width: 1.5 },
            name: 'Loss',
        },
        {
            x: [0, equityCurve.length - 1],
            y: [initialBalance, initialBalance],
            type: 'scatter', mode: 'lines',
            line: { color: WT_COLORS.text, width: 1, dash: 'dash' },
            name: 'Starting Capital',
        },
    ];

    Plotly.newPlot(containerId, traces, mergLayout({
        title: { text: 'Equity Curve', font: { size: 13, color: WT_COLORS.textBright } },
        yaxis: { title: 'Balance (USD)', gridcolor: WT_COLORS.grid },
        xaxis: { title: 'Bars', gridcolor: WT_COLORS.grid },
        showlegend: false,
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 2. Per-Trade PnL + Cumulative Line
// ═══════════════════════════════════════════════════════════════════════════

function renderTradePnL(containerId, trades) {
    const nums = trades.map((_, i) => i + 1);
    const pnls = trades.map(t => t.pnl);
    const colors = pnls.map(p => p >= 0 ? WT_COLORS.green : WT_COLORS.red);

    let cum = 0;
    const cumPnl = pnls.map(p => { cum += p; return Math.round(cum * 100) / 100; });

    const traces = [
        {
            x: nums, y: pnls, type: 'bar',
            marker: { color: colors },
            name: 'Per-Trade PnL',
            yaxis: 'y1',
        },
        {
            x: nums, y: cumPnl, type: 'scatter', mode: 'lines',
            line: { color: WT_COLORS.blue, width: 2 },
            name: 'Cumulative PnL',
            yaxis: 'y2',
        },
    ];

    Plotly.newPlot(containerId, traces, mergLayout({
        title: { text: 'Per-Trade PnL', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Trade #', gridcolor: WT_COLORS.grid },
        yaxis: { title: 'PnL ($)', gridcolor: WT_COLORS.grid, side: 'left' },
        yaxis2: {
            title: 'Cumulative ($)', overlaying: 'y', side: 'right',
            gridcolor: 'transparent', showgrid: false,
        },
        barmode: 'relative',
        showlegend: true,
        legend: { x: 0, y: 1.12, orientation: 'h', font: { size: 10 } },
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 3. Monthly Net PnL
// ═══════════════════════════════════════════════════════════════════════════

function renderMonthlyPnL(containerId, monthly) {
    const months = monthly.map(m => m.month);
    const pnls = monthly.map(m => m.net_pnl);
    const colors = pnls.map(p => p >= 0 ? WT_COLORS.green : WT_COLORS.red);

    Plotly.newPlot(containerId, [{
        x: months, y: pnls, type: 'bar',
        marker: { color: colors },
        hovertemplate: '%{x}<br>PnL: $%{y:.2f}<extra></extra>',
    }], mergLayout({
        title: { text: 'Monthly Net PnL', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Month', gridcolor: WT_COLORS.grid, tickangle: -45 },
        yaxis: { title: 'Net PnL ($)', gridcolor: WT_COLORS.grid },
        showlegend: false,
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 4. Rolling Win Rate (50-trade window)
// ═══════════════════════════════════════════════════════════════════════════

function renderRollingWinRate(containerId, trades) {
    const window = 50;
    const nums = [];
    const rates = [];

    for (let i = window - 1; i < trades.length; i++) {
        const slice = trades.slice(i - window + 1, i + 1);
        const wins = slice.filter(t => t.pnl > 0).length;
        nums.push(i + 1);
        rates.push((wins / window) * 100);
    }

    Plotly.newPlot(containerId, [
        {
            x: nums, y: rates, type: 'scatter', mode: 'lines',
            line: { color: WT_COLORS.orange, width: 1.5 },
            fill: 'tozeroy', fillcolor: 'rgba(210,153,34,0.08)',
            name: 'Win Rate',
        },
        {
            x: [nums[0], nums[nums.length - 1]], y: [50, 50],
            type: 'scatter', mode: 'lines',
            line: { color: WT_COLORS.text, width: 1, dash: 'dash' },
            name: '50% Baseline',
        },
    ], mergLayout({
        title: { text: `Rolling Win Rate (${window}-trade)`, font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Trade #', gridcolor: WT_COLORS.grid },
        yaxis: { title: 'Win Rate (%)', gridcolor: WT_COLORS.grid, range: [0, 100] },
        showlegend: false,
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 5. Exit Reason Breakdown (horizontal bar)
// ═══════════════════════════════════════════════════════════════════════════

function renderExitReasons(containerId, exitReasons) {
    const reasons = exitReasons.map(e => e.reason);
    const pnls = exitReasons.map(e => e.total_pnl);
    const colors = pnls.map(p => p >= 0 ? WT_COLORS.green : WT_COLORS.red);
    const text = exitReasons.map(e => `${e.trades} trades · ${(e.win_rate * 100).toFixed(1)}% WR`);

    Plotly.newPlot(containerId, [{
        x: pnls, y: reasons, type: 'bar', orientation: 'h',
        marker: { color: colors },
        text: text, textposition: 'auto',
        textfont: { size: 10, color: WT_COLORS.textBright },
    }], mergLayout({
        title: { text: 'PnL by Exit Reason', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Total PnL ($)', gridcolor: WT_COLORS.grid },
        yaxis: { gridcolor: WT_COLORS.grid, automargin: true },
        showlegend: false,
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 6. Trade Duration Histograms
// ═══════════════════════════════════════════════════════════════════════════

function renderDurationHistogram(containerId, durations) {
    const traces = [];

    if (durations.winners && durations.winners.length > 0) {
        traces.push({
            x: durations.winners, type: 'histogram', nbinsx: 40,
            marker: { color: 'rgba(63,185,80,0.5)' },
            name: 'Winners',
            opacity: 0.7,
        });
    }
    if (durations.losers && durations.losers.length > 0) {
        traces.push({
            x: durations.losers, type: 'histogram', nbinsx: 40,
            marker: { color: 'rgba(248,81,73,0.5)' },
            name: 'Losers',
            opacity: 0.7,
        });
    }

    Plotly.newPlot(containerId, traces, mergLayout({
        title: { text: 'Trade Duration Distribution', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Duration (hours)', gridcolor: WT_COLORS.grid },
        yaxis: { title: 'Count', gridcolor: WT_COLORS.grid },
        barmode: 'overlay',
        legend: { x: 0.7, y: 0.95, font: { size: 10 } },
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 7. Trades by Hour of Day
// ═══════════════════════════════════════════════════════════════════════════

function renderHourlyTrades(containerId, hourly) {
    const hours = hourly.map(h => h.hour);
    const counts = hourly.map(h => h.trades);

    // Color by session
    const colors = hours.map(h => {
        if (h >= 0 && h < 8) return 'rgba(188,140,255,0.6)';  // Asia/Tokyo
        if (h >= 8 && h < 16) return 'rgba(63,185,80,0.6)';   // London
        return 'rgba(210,153,34,0.6)';                          // NY
    });

    Plotly.newPlot(containerId, [{
        x: hours, y: counts, type: 'bar',
        marker: { color: colors },
        hovertemplate: 'Hour %{x}:00<br>Trades: %{y}<extra></extra>',
    }], mergLayout({
        title: { text: 'Trades by Hour of Day', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: {
            title: 'Hour (UTC)', gridcolor: WT_COLORS.grid,
            tickmode: 'linear', dtick: 1, range: [-0.5, 23.5],
        },
        yaxis: { title: 'Trade Count', gridcolor: WT_COLORS.grid },
        showlegend: false,
        annotations: [
            { x: 4, y: 1.05, xref: 'x', yref: 'paper', text: '🟣 Asia', showarrow: false, font: { size: 9 } },
            { x: 12, y: 1.05, xref: 'x', yref: 'paper', text: '🟢 London', showarrow: false, font: { size: 9 } },
            { x: 19, y: 1.05, xref: 'x', yref: 'paper', text: '🟠 New York', showarrow: false, font: { size: 9 } },
        ],
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 8. PnL + Win Rate by Hour (dual axis)
// ═══════════════════════════════════════════════════════════════════════════

function renderHourlyPnL(containerId, hourly) {
    const hours = hourly.map(h => h.hour);
    const pnls = hourly.map(h => h.net_pnl);
    const wrs = hourly.map(h => h.win_rate * 100);
    const pnlColors = pnls.map(p => p >= 0 ? WT_COLORS.green : WT_COLORS.red);

    Plotly.newPlot(containerId, [
        {
            x: hours, y: pnls, type: 'bar',
            marker: { color: pnlColors },
            name: 'Net PnL', yaxis: 'y1',
        },
        {
            x: hours, y: wrs, type: 'scatter', mode: 'lines+markers',
            line: { color: WT_COLORS.orange, width: 2 },
            marker: { size: 5, color: WT_COLORS.orange },
            name: 'Win Rate', yaxis: 'y2',
        },
    ], mergLayout({
        title: { text: 'Hourly PnL & Win Rate', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Hour (UTC)', gridcolor: WT_COLORS.grid, tickmode: 'linear', dtick: 2 },
        yaxis: { title: 'Net PnL ($)', gridcolor: WT_COLORS.grid },
        yaxis2: {
            title: 'Win Rate (%)', overlaying: 'y', side: 'right',
            range: [0, 100], showgrid: false,
        },
        legend: { x: 0, y: 1.12, orientation: 'h', font: { size: 10 } },
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 9. Day-of-Week PnL
// ═══════════════════════════════════════════════════════════════════════════

function renderDailyPnL(containerId, daily) {
    const days = daily.map(d => d.day);
    const pnls = daily.map(d => d.net_pnl);
    const colors = pnls.map(p => p >= 0 ? WT_COLORS.green : WT_COLORS.red);

    Plotly.newPlot(containerId, [{
        x: days, y: pnls, type: 'bar',
        marker: { color: colors },
        hovertemplate: '%{x}<br>PnL: $%{y:.2f}<extra></extra>',
    }], mergLayout({
        title: { text: 'Day-of-Week Net PnL', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { gridcolor: WT_COLORS.grid },
        yaxis: { title: 'Net PnL ($)', gridcolor: WT_COLORS.grid },
        showlegend: false,
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 10. Day-of-Week Win Rate
// ═══════════════════════════════════════════════════════════════════════════

function renderDailyWinRate(containerId, daily) {
    const days = daily.map(d => d.day);
    const wrs = daily.map(d => d.win_rate * 100);
    const text = daily.map(d => `${d.trades} trades`);

    Plotly.newPlot(containerId, [{
        x: days, y: wrs, type: 'bar',
        marker: { color: WT_COLORS.blue },
        text: text, textposition: 'outside',
        textfont: { size: 9, color: WT_COLORS.text },
        hovertemplate: '%{x}<br>Win Rate: %{y:.1f}%<extra></extra>',
    }], mergLayout({
        title: { text: 'Day-of-Week Win Rate', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { gridcolor: WT_COLORS.grid },
        yaxis: { title: 'Win Rate (%)', gridcolor: WT_COLORS.grid, range: [0, 100] },
        showlegend: false,
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 11. Theoretical vs Realistic Equity (Friction)
// ═══════════════════════════════════════════════════════════════════════════

function renderFrictionEquity(containerId, friction) {
    const theoX = friction.theoretical.map(t => t.trade_num);
    const theoY = friction.theoretical.map(t => t.balance);
    const realX = friction.realistic.map(t => t.trade_num);
    const realY = friction.realistic.map(t => t.balance);

    Plotly.newPlot(containerId, [
        {
            x: theoX, y: theoY, type: 'scatter', mode: 'lines',
            line: { color: WT_COLORS.blue, width: 1.5 },
            name: 'Theoretical',
        },
        {
            x: realX, y: realY, type: 'scatter', mode: 'lines',
            line: { color: WT_COLORS.green, width: 2 },
            name: 'Realistic',
        },
    ], mergLayout({
        title: { text: 'Theoretical vs Realistic Equity', font: { size: 13, color: WT_COLORS.textBright } },
        xaxis: { title: 'Trade #', gridcolor: WT_COLORS.grid },
        yaxis: { title: 'Balance ($)', gridcolor: WT_COLORS.grid },
        legend: { x: 0, y: 1.12, orientation: 'h', font: { size: 10 } },
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// 12. Friction Breakdown Pie
// ═══════════════════════════════════════════════════════════════════════════

function renderFrictionPie(containerId, costs) {
    const labels = ['Slippage', 'Extra Spread', 'Lot Cap Loss'];
    const values = [costs.slippage, costs.extra_spread, costs.lot_cap_loss];

    Plotly.newPlot(containerId, [{
        labels, values, type: 'pie',
        marker: {
            colors: [WT_COLORS.orange, WT_COLORS.purple, WT_COLORS.red],
        },
        textinfo: 'label+percent',
        textfont: { size: 11, color: WT_COLORS.textBright },
        hole: 0.4,
    }], mergLayout({
        title: { text: 'Friction Cost Breakdown', font: { size: 13, color: WT_COLORS.textBright } },
        showlegend: true,
        legend: { font: { size: 10, color: WT_COLORS.text } },
    }), PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════════════
// Master render function — updates all analytics charts from results
// ═══════════════════════════════════════════════════════════════════════════

function renderAllAnalytics(results) {
    const { trades, equity_curve, breakdowns, friction, config } = results;
    const initial = config?.initial_balance || 25000;

    // Equity tab
    renderEquityCurve('chart-equity-curve', equity_curve, initial);
    renderTradePnL('chart-trade-pnl', trades);
    renderRollingWinRate('chart-rolling-wr', trades);

    // PnL tab
    renderMonthlyPnL('chart-monthly-pnl', breakdowns.monthly || []);
    renderExitReasons('chart-exit-reasons', breakdowns.exit_reasons || []);

    // Sessions tab
    renderHourlyTrades('chart-hourly-trades', breakdowns.hourly || []);
    renderHourlyPnL('chart-hourly-pnl', breakdowns.hourly || []);
    renderDailyPnL('chart-daily-pnl', breakdowns.daily || []);
    renderDailyWinRate('chart-daily-wr', breakdowns.daily || []);

    // Duration tab
    if (breakdowns.durations && (breakdowns.durations.all || []).length > 0) {
        renderDurationHistogram('chart-duration', breakdowns.durations);
    }

    // Friction tab
    if (friction) {
        renderFrictionEquity('chart-friction-equity', friction);
        if (friction.costs) {
            renderFrictionPie('chart-friction-pie', friction.costs);
        }
    }
}

// Export
window.renderAllAnalytics = renderAllAnalytics;
window.renderEquityCurve = renderEquityCurve;
