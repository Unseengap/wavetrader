/**
 * WaveTrader LLM Arbiter Panel
 * Handles arbiter decisions display, detail modal, SSE events,
 * configuration toggles, and trial stats tracking.
 */

let _arbiterDecisions = [];

// ── Initialization ──────────────────────────────────────────────────────────

function initArbiterPanel() {
    // Load arbiter status
    loadArbiterStatus();
    // Load existing decisions
    loadArbiterDecisions();

    // Close detail modal
    const closeBtn = document.getElementById('arbiter-detail-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeArbiterDetail);
    }
}

// ── Load Status ─────────────────────────────────────────────────────────────

async function loadArbiterStatus() {
    try {
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/arbiter/status?model=${encodeURIComponent(model)}`);
        const data = await resp.json();

        // Update stats
        if (data.stats) updateArbiterStats(data.stats);
    } catch (err) {
        console.warn('Could not load arbiter status:', err);
    }
}

// ── Load Decisions ──────────────────────────────────────────────────────────

async function loadArbiterDecisions() {
    try {
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/arbiter/decisions?model=${encodeURIComponent(model)}&count=50`);
        const data = await resp.json();
        _arbiterDecisions = data.decisions || [];
        renderArbiterDecisions();
    } catch (err) {
        console.warn('Could not load arbiter decisions:', err);
    }
}

// ── (Config update removed — mode locked to override) ───────────────────────

// ── Render Decision List ────────────────────────────────────────────────────

function renderArbiterDecisions() {
    const container = document.getElementById('arbiter-decisions-list');
    if (!container) return;

    if (_arbiterDecisions.length === 0) {
        container.innerHTML = `
            <div class="wt-empty-state" style="padding:2rem">
                <i class="bi bi-robot" style="font-size:2rem;color:var(--wt-text-muted);opacity:0.3"></i>
                <p style="margin-top:0.5rem">No arbiter decisions yet. Enable the arbiter and wait for signals.</p>
            </div>`;
        return;
    }

    let html = '';
    for (const dec of _arbiterDecisions) {
        const action = dec.action || 'APPROVE';
        const actionClass = action === 'APPROVE' ? 'approve' : action === 'VETO' ? 'veto' : 'override';
        const signal = dec.original_signal || '?';
        const conf = dec.original_confidence ? (dec.original_confidence * 100).toFixed(1) : '?';
        const pair = dec.pair || '';
        const time = dec.timestamp ? new Date(dec.timestamp).toLocaleTimeString() : '';
        const reasoning = dec.reasoning || '';
        const shortReason = reasoning.length > 80 ? reasoning.substring(0, 80) + '…' : reasoning;
        const tradePlaced = dec.trade_placed;
        const latency = dec.latency_ms ? Math.round(dec.latency_ms) : 0;

        // Strategy metadata
        const stratName = dec.strategy_name || '';
        const stratBadge = stratName
            ? `<span class="wt-strategy-badge">${stratName}</span>`
            : '';

        // Narrative (LLM strategy voice)
        const narrative = dec.narrative || '';
        const shortNarrative = narrative.length > 120 ? narrative.substring(0, 120) + '…' : narrative;
        const narrativeHtml = shortNarrative
            ? `<div class="wt-narrative-text">${shortNarrative}</div>`
            : '';

        // Risk notes indicator
        const riskNotes = dec.risk_notes || '';
        const riskIcon = riskNotes
            ? '<i class="bi bi-exclamation-triangle-fill" style="color:var(--wt-yellow,#d29922);font-size:0.68rem" title="Risk notes available"></i>'
            : '';

        // Outcome status
        const outcome = dec.outcome;
        let outcomeHtml = '';
        if (outcome && outcome.pnl !== undefined) {
            const pnl = parseFloat(outcome.pnl);
            const cls = pnl >= 0 ? 'wt-pnl-positive' : 'wt-pnl-negative';
            outcomeHtml = `<span class="${cls}">$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>`;
        } else if (tradePlaced) {
            outcomeHtml = '<span style="color:var(--wt-text-muted);font-size:0.68rem">Pending…</span>';
        } else if (action === 'VETO') {
            outcomeHtml = '<span style="color:var(--wt-text-muted);font-size:0.68rem">Simulating…</span>';
        }

        html += `
        <div class="wt-arbiter-decision-row ${actionClass}" data-decision-id="${dec.decision_id || ''}" onclick="openArbiterDetail(this.dataset.decisionId)">
            <div class="wt-arbiter-decision-main">
                <span class="wt-arbiter-action-badge ${actionClass}">${action}</span>
                <span class="wt-arbiter-signal-badge ${signal.toLowerCase()}">${signal}</span>
                ${stratBadge}
                <span style="font-size:0.72rem;color:var(--wt-text-muted)">${pair}</span>
                <span style="font-size:0.72rem;color:var(--wt-text-muted)">Conf: ${conf}%</span>
                ${riskIcon}
                <span style="flex:1;font-size:0.72rem;color:var(--wt-text);opacity:0.8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${shortReason}</span>
            </div>
            ${narrativeHtml}
            <div class="wt-arbiter-decision-meta">
                ${tradePlaced ? '<i class="bi bi-check-circle-fill" style="color:var(--wt-green);font-size:0.7rem" title="Trade placed"></i>' : '<i class="bi bi-dash-circle" style="color:var(--wt-text-muted);font-size:0.7rem" title="No trade"></i>'}
                ${outcomeHtml}
                <span style="font-size:0.68rem;color:var(--wt-text-muted)">${latency}ms</span>
                <span style="font-size:0.68rem;color:var(--wt-text-muted)">${time}</span>
            </div>
        </div>`;
    }

    container.innerHTML = html;
}

// ── Handle SSE Arbiter Events ───────────────────────────────────────────────

function handleArbiterSSE(data) {
    // Prepend new decision to list
    _arbiterDecisions.unshift(data);
    _arbiterDecisions = _arbiterDecisions.slice(0, 100);
    renderArbiterDecisions();

    // Refresh stats
    loadArbiterStats();
}

async function loadArbiterStats() {
    try {
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/arbiter/stats?model=${encodeURIComponent(model)}`);
        const data = await resp.json();
        updateArbiterStats(data);
    } catch (err) {
        // Silent
    }
}

function updateArbiterStats(stats) {
    setText('arbiter-stat-total', stats.total_decisions || 0);
    setText('arbiter-stat-approved', stats.approvals || 0);
    setText('arbiter-stat-vetoed', stats.vetoes || 0);
    setText('arbiter-stat-overrides', stats.overrides || 0);
    setText('arbiter-stat-veto-saved', stats.veto_would_have_lost || 0);
    setText('arbiter-stat-veto-missed', stats.veto_would_have_won || 0);
    setText('arbiter-stat-latency', (stats.avg_latency_ms || 0) + 'ms');
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ── Detail Modal ────────────────────────────────────────────────────────────

function openArbiterDetail(decisionId) {
    const dec = _arbiterDecisions.find(d => d.decision_id === decisionId);
    if (!dec) return;

    const modal = document.getElementById('arbiter-detail-modal');
    if (!modal) return;

    // Populate fields
    const action = dec.action || 'APPROVE';
    const actionBadge = document.getElementById('arbiter-detail-action-badge');
    if (actionBadge) {
        actionBadge.textContent = action;
        actionBadge.className = `wt-arbiter-action-badge ${action.toLowerCase()}`;
    }

    setText('arbiter-detail-signal', `${dec.original_signal || '?'} → ${dec.modified_signal || dec.original_signal || '?'}`);
    setText('arbiter-detail-pair', dec.pair || '');

    // Strategy info
    const stratSection = document.getElementById('arbiter-detail-strategy-section');
    if (stratSection) {
        if (dec.strategy_name) {
            stratSection.style.display = 'block';
            setText('arbiter-detail-strategy-name', dec.strategy_name);
            setText('arbiter-detail-strategy-id', dec.strategy_id || '');
        } else {
            stratSection.style.display = 'none';
        }
    }

    // Narrative (strategy voice)
    const narrativeSection = document.getElementById('arbiter-detail-narrative-section');
    const narrativeEl = document.getElementById('arbiter-detail-narrative');
    if (narrativeSection && narrativeEl) {
        if (dec.narrative) {
            narrativeSection.style.display = 'block';
            narrativeEl.textContent = dec.narrative;
        } else {
            narrativeSection.style.display = 'none';
        }
    }

    document.getElementById('arbiter-detail-reasoning').textContent = dec.reasoning || 'No reasoning provided';

    // Risk notes
    const riskSection = document.getElementById('arbiter-detail-risk-section');
    const riskNotes = document.getElementById('arbiter-detail-risk-notes');
    if (dec.risk_notes) {
        riskSection.style.display = 'block';
        riskNotes.textContent = dec.risk_notes;
    } else {
        riskSection.style.display = 'none';
    }

    // Signal details
    setText('arbiter-detail-orig-signal', dec.original_signal || '?');
    setText('arbiter-detail-confidence', dec.original_confidence ? (dec.original_confidence * 100).toFixed(1) + '%' : '?');
    setText('arbiter-detail-entry', dec.entry_price ? dec.entry_price.toFixed(5) : '?');
    setText('arbiter-detail-conf-adj', dec.confidence_adjustment ? (dec.confidence_adjustment > 0 ? '+' : '') + (dec.confidence_adjustment * 100).toFixed(1) + '%' : 'None');
    setText('arbiter-detail-trade-placed', dec.trade_placed ? '✅ Yes' : '❌ No');
    setText('arbiter-detail-llm-model', dec.model_used || '?');
    setText('arbiter-detail-latency', dec.latency_ms ? Math.round(dec.latency_ms) + 'ms' : '?');

    // Outcome
    const outcomeEl = document.getElementById('arbiter-detail-outcome');
    if (dec.outcome && dec.outcome.pnl !== undefined) {
        const pnl = parseFloat(dec.outcome.pnl);
        const cls = pnl >= 0 ? 'wt-pnl-positive' : 'wt-pnl-negative';
        const result = pnl >= 0 ? 'WON' : 'LOST';
        outcomeEl.innerHTML = `
            <div style="display:flex;gap:16px;align-items:center">
                <span class="${cls}" style="font-size:1.2rem;font-weight:700">$${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>
                <span style="font-size:0.78rem;font-weight:600;color:${pnl >= 0 ? 'var(--wt-green)' : 'var(--wt-red, #f85149)'}">${result}</span>
                ${dec.outcome.exit_reason ? `<span style="font-size:0.72rem;color:var(--wt-text-muted)">${dec.outcome.exit_reason}</span>` : ''}
            </div>`;

        // Show if LLM decision was correct
        if (action === 'VETO') {
            const simPnl = dec.outcome.simulated_pnl || pnl;
            const saved = simPnl < 0;
            outcomeEl.innerHTML += `
                <div style="margin-top:8px;padding:6px 10px;border-radius:4px;background:${saved ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)'};font-size:0.75rem">
                    ${saved
                        ? `<i class="bi bi-shield-check" style="color:var(--wt-green)"></i> Veto <b>saved</b> us $${Math.abs(simPnl).toFixed(2)}`
                        : `<i class="bi bi-exclamation-triangle" style="color:var(--wt-red, #f85149)"></i> Veto <b>missed</b> $${simPnl.toFixed(2)} profit`
                    }
                </div>`;
        }
    } else {
        outcomeEl.innerHTML = '<span style="color:var(--wt-text-muted);font-size:0.75rem">Outcome pending — will update when trade closes or simulation period ends</span>';
    }

    // Calendar context
    const calSection = document.getElementById('arbiter-detail-calendar-section');
    const calEl = document.getElementById('arbiter-detail-calendar');
    const context = dec.context || {};
    const events = context.calendar_events || [];
    if (events.length > 0) {
        calSection.style.display = 'block';
        calEl.innerHTML = events.map(e => {
            const impactClass = e.impact === 'high' ? 'color:var(--wt-red, #f85149);font-weight:600' : e.impact === 'medium' ? 'color:var(--wt-yellow)' : 'color:var(--wt-text-muted)';
            return `<div style="padding:2px 0"><span style="${impactClass}">[${(e.impact || '').toUpperCase()}]</span> ${e.currency}: ${e.event} (Prev: ${e.previous}, Fcst: ${e.forecast})</div>`;
        }).join('');
    } else {
        calSection.style.display = 'none';
    }

    // Show modal
    modal.style.display = 'flex';

    // Hide the decision list behind the modal
    const decList = document.getElementById('arbiter-decisions-list');
    if (decList) decList.style.display = 'none';
}

function closeArbiterDetail() {
    const modal = document.getElementById('arbiter-detail-modal');
    if (modal) modal.style.display = 'none';

    const decList = document.getElementById('arbiter-decisions-list');
    if (decList) decList.style.display = '';
}

// ── Initialize on DOM ready ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initArbiterPanel();
});


// ── Manual Market Inspection ────────────────────────────────────────────────

let _inspectionRunning = false;

async function runInspection() {
    if (_inspectionRunning) return;
    _inspectionRunning = true;

    const btn = document.getElementById('inspect-market-btn');
    const status = document.getElementById('inspect-status');
    if (btn) {
        btn.disabled = true;
        btn.style.opacity = '0.6';
        btn.innerHTML = '<i class="bi bi-hourglass-split"></i> Analysing…';
    }
    if (status) status.textContent = 'Calling Gemini Pro — this may take 10-20s…';

    try {
        const model = (typeof currentModel !== 'undefined') ? currentModel : 'mtf';
        const resp = await fetch(`/api/live/arbiter/inspect?model=${encodeURIComponent(model)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        const data = await resp.json();
        displayInspectionResult(data);

        if (status) status.textContent = `Done in ${Math.round(data.latency_ms || 0)}ms`;
    } catch (err) {
        console.error('Inspection failed:', err);
        if (status) status.textContent = 'Inspection failed: ' + err.message;
    } finally {
        _inspectionRunning = false;
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.innerHTML = '<i class="bi bi-lightning-charge-fill"></i> Inspect Market';
        }
    }
}

function displayInspectionResult(data) {
    const panel = document.getElementById('inspection-result-panel');
    if (!panel) return;
    panel.style.display = 'block';

    // Time + latency
    const timeEl = document.getElementById('inspection-time');
    if (timeEl) timeEl.textContent = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '';
    const latEl = document.getElementById('inspection-latency');
    if (latEl) latEl.textContent = data.latency_ms ? `${Math.round(data.latency_ms)}ms` : '';

    // Analysis text
    const analysisEl = document.getElementById('inspection-analysis');
    if (analysisEl) analysisEl.textContent = data.analysis || 'No analysis returned.';

    // Risk warnings
    const warningsEl = document.getElementById('inspection-warnings');
    const warnings = data.risk_warnings || [];
    if (warningsEl) {
        if (warnings.length > 0) {
            warningsEl.style.display = 'block';
            warningsEl.innerHTML = warnings.map(w =>
                `<div style="padding:4px 8px;margin-bottom:3px;border-radius:4px;background:rgba(210,153,34,0.1);font-size:0.72rem;color:var(--wt-yellow)">
                    <i class="bi bi-exclamation-triangle-fill" style="margin-right:4px"></i>${w}
                </div>`
            ).join('');
        } else {
            warningsEl.style.display = 'none';
        }
    }

    // Trade action
    const tradeEl = document.getElementById('inspection-trade');
    const ta = data.trade_action;
    const te = data.trade_executed;
    if (tradeEl) {
        if (ta) {
            const sigColor = ta.signal === 'BUY' ? 'var(--wt-green, #3fb950)' : 'var(--wt-red, #f85149)';
            let executedHtml = '';
            if (te && !te.error) {
                executedHtml = `
                    <div style="margin-top:6px;padding:6px 10px;border-radius:4px;background:rgba(63,185,80,0.1);font-size:0.72rem">
                        <i class="bi bi-check-circle-fill" style="color:var(--wt-green)"></i>
                        <b>Trade placed</b> on ${te.model_id} @ ${(te.price || 0).toFixed(3)}
                    </div>`;
            } else if (te && te.error) {
                executedHtml = `
                    <div style="margin-top:6px;padding:6px 10px;border-radius:4px;background:rgba(248,81,73,0.1);font-size:0.72rem">
                        <i class="bi bi-x-circle-fill" style="color:var(--wt-red, #f85149)"></i>
                        Trade failed: ${te.error}
                    </div>`;
            }
            tradeEl.style.display = 'block';
            tradeEl.innerHTML = `
                <div style="padding:8px 10px;border-radius:6px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2)">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                        <span style="font-weight:700;color:${sigColor};font-size:0.82rem">${ta.signal}</span>
                        <span style="font-size:0.72rem;color:var(--wt-text-muted)">via ${ta.model_id}</span>
                        <span style="font-size:0.72rem;color:var(--wt-text-muted)">SL: ${ta.sl_pips}p | TP: ${ta.tp_pips}p</span>
                        <span style="font-size:0.72rem;color:var(--wt-text-muted)">Conf: ${((ta.confidence || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div style="font-size:0.72rem;color:var(--wt-text);opacity:0.85">${ta.reasoning || ''}</div>
                    ${executedHtml}
                </div>`;
        } else {
            tradeEl.style.display = 'block';
            tradeEl.innerHTML = `
                <div style="font-size:0.72rem;color:var(--wt-text-muted);padding:4px 0">
                    <i class="bi bi-info-circle"></i> No trade recommended — market unclear or models already positioned correctly.
                </div>`;
        }
    }

    // Scroll the arbiter tab into view
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
