/**
 * event-cards.js — Per-event-type card renderers.
 *
 * Each event type has a unique icon SVG and rendered card body.
 * Cards are rendered into the event timeline.
 */

window.EventCards = (() => {

  // ── SVG Icons per event type ──────────────────────────────────────────
  const ICONS = {
    ActionEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`,
    ObservationEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`,
    MessageEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
    StartupEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
    HeartbeatEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
    EvaluateEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
    FinishEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12l5 5L20 7"/></svg>`,
    AgentErrorEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
    LLMCompletionLogEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6v6l4 2"/><circle cx="18" cy="6" r="4"/></svg>`,
    PauseEvent: `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`,
    ConversationStateUpdateEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 2H3v16h5l3 3 3-3h7V2z"/><line x1="9" y1="9" x2="15" y2="9"/><line x1="9" y1="13" x2="12" y2="13"/></svg>`,
    default: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`,
  };

  function getIcon(type) {
    return ICONS[type] || ICONS.default;
  }

  // ── Time formatting ───────────────────────────────────────────────────
  function fmtRelTime(ts) {
    if (!ts) return '';
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    const now = Date.now();
    const diff = Math.floor((now - date.getTime()) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
  }

  function fmtAbsTime(ts) {
    if (!ts) return '';
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  // ── Body renderers per type ───────────────────────────────────────────
  const _bodyRenderers = {
    ActionEvent(ev) {
      const raw = ev.raw || {};
      const tool = raw.tool_name || raw.tool || '';
      const args = raw.tool_input || raw.arguments || {};
      const argsStr = typeof args === 'string' ? args : JSON.stringify(args);
      return `<div class="ev-summary">[${tool}] ${ev.summary?.replace(/^\[Action:[^\]]+\]\s*/, '') || ''}</div>
              ${argsStr ? `<div class="ev-code-preview"><code>${_truncStr(argsStr, 180)}</code></div>` : ''}`;
    },
    ObservationEvent(ev) {
      const raw = ev.raw || {};
      const tool = raw.tool_name || '';
      const result = raw.result || raw.content || '';
      return `<div class="ev-summary">[${tool}] observation</div>
              ${result ? `<div class="ev-code-preview"><code>${_truncStr(String(result), 200)}</code></div>` : ''}`;
    },
    MessageEvent(ev) {
      const raw = ev.raw || {};
      const role = raw.role || 'agent';
      const content = raw.content || ev.summary || '';
      return `<div class="ev-summary"><span class="ev-role-badge ev-role-${role}">${role}</span> ${_truncStr(String(content).replace(/\[Message:[^\]]+\]\s*/, ''), 200)}</div>`;
    },
    StartupEvent(ev) {
      return `<div class="ev-summary">Container started · ${ev.llm_model || ''}</div>
              <div class="ev-tags-row">
                ${ev.max_iterations ? `<span class="ev-tag">max_iter: ${ev.max_iterations}</span>` : ''}
                ${ev.workspace_base ? `<span class="ev-tag">${ev.workspace_base}</span>` : ''}
              </div>`;
    },
    HeartbeatEvent(ev) {
      return `<div class="ev-summary">♡ ${ev.phase || ''} · iter ${ev.iteration ?? '?'} · uptime ${Math.round(ev.uptime_seconds ?? 0)}s</div>`;
    },
    EvaluateEvent(ev) {
      return `<div class="ev-summary">Status: <strong>${ev.status || ''}</strong> · cost $${(ev.accumulated_cost || 0).toFixed(4)} · ${ev.total_llm_calls || 0} LLM calls · ${ev.iterations || 0} iterations</div>`;
    },
    FinishEvent(ev) {
      const ok = (ev.exit_code || 0) === 0;
      return `<div class="ev-summary ${ok ? '' : 'ev-summary--error'}">Exit ${ev.exit_code ?? 0} · ${ev.phase || ''} · ${ev.reason || ''}</div>`;
    },
    AgentErrorEvent(ev) {
      const raw = ev.raw || {};
      const err = raw.error || ev.summary || '';
      return `<div class="ev-summary ev-summary--error">${_truncStr(String(err), 220)}</div>`;
    },
    LLMCompletionLogEvent(ev) {
      const raw = ev.raw || {};
      const tokens = raw.usage ? `${raw.usage.total_tokens ?? '?'} tokens` : '';
      const cost = raw.cost ? `$${Number(raw.cost).toFixed(5)}` : '';
      return `<div class="ev-summary">LLM call${tokens ? ' · ' + tokens : ''}${cost ? ' · ' + cost : ''}</div>`;
    },
    ConversationStateUpdateEvent(ev) {
      const raw = ev.raw || {};
      return `<div class="ev-summary"><span class="ev-tag">${raw.key || ''}=${_truncStr(String(raw.value || ''), 60)}</span></div>`;
    },
    default(ev) {
      return `<div class="ev-summary">${_truncStr(ev.summary || '', 220)}</div>`;
    },
  };

  function _truncStr(s, n) {
    return s.length > n ? s.slice(0, n) + '…' : s;
  }

  // ── Public: render a card DOM element ────────────────────────────────
  function render(ev, opts = {}) {
    const type = ev.event_type || 'unknown';
    const renderer = _bodyRenderers[type] || _bodyRenderers.default;
    const isHidden = ev.visible === 0 || ev.visible === false;
    const isExit = type === 'FinishEvent' && (ev.exit_code || 0) !== 0;

    const card = document.createElement('div');
    card.className = `ev-card ${type}${isExit ? ' exit-error' : ''}${isHidden ? ' ev-hidden' : ''}`;
    card.dataset.id = ev.id || '';
    card.dataset.type = type;
    card.dataset.visible = isHidden ? '0' : '1';

    const ts = ev.received_at || ev.timestamp;
    const absTime = fmtAbsTime(ts);
    const relTime = fmtRelTime(ts);

    const eyeIcon = isHidden
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;

    card.innerHTML = `
      <div class="ev-icon">${getIcon(type)}</div>
      <div class="ev-body">
        <div class="ev-header">
          <span class="ev-type">${type.replace('Event', '')}</span>
          <span class="ev-source">${ev.source || ''}</span>
        </div>
        ${renderer(ev)}
        <div class="ev-meta">
          <span class="ev-time" title="${absTime}">${relTime}</span>
          ${ev.id ? `<span class="ev-time">#${ev.id}</span>` : ''}
        </div>
      </div>
      <div class="ev-actions">
        <button class="ev-vis-btn${isHidden ? ' hidden-state' : ''}" data-ev-id="${ev.id}" title="${isHidden ? 'Show event' : 'Hide event'}">${eyeIcon}</button>
      </div>`;

    // Click opens drawer
    card.addEventListener('click', (e) => {
      if (e.target.closest('.ev-vis-btn')) return;
      opts.onOpen && opts.onOpen(ev);
    });

    // Visibility toggle
    card.querySelector('.ev-vis-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      opts.onToggleVisible && opts.onToggleVisible(ev, card);
    });

    return card;
  }

  // ── Time gap divider ──────────────────────────────────────────────────
  function renderGap(seconds) {
    const el = document.createElement('div');
    el.className = 'time-gap';
    const label = seconds < 60
      ? `${seconds.toFixed(1)}s`
      : seconds < 3600
        ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
        : `${Math.floor(seconds / 3600)}h`;
    el.innerHTML = `<div class="time-gap-line"></div><span class="time-gap-label">${label} gap</span><div class="time-gap-line"></div>`;
    return el;
  }

  // ── Collapsed group ───────────────────────────────────────────────────
  function renderGroup(type, count, onExpand) {
    const el = document.createElement('div');
    el.className = 'ev-group';
    el.innerHTML = `${getIcon(type)}<span class="ev-group-count">${count}</span> <span>${type.replace('Event', '')} events collapsed</span> <span style="margin-left:auto;font-size:10px">click to expand</span>`;
    el.addEventListener('click', onExpand);
    return el;
  }

  return { render, renderGap, renderGroup, getIcon, fmtRelTime, fmtAbsTime };
})();

// Extra styles for event card internals
const _evCardStyle = document.createElement('style');
_evCardStyle.textContent = `
.ev-code-preview { margin-top: 3px; background: var(--bg-base); border-radius: 3px; padding: 3px 6px; }
.ev-code-preview code { font-family: var(--font-mono); font-size: 10px; color: var(--cyan); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; max-width: 100%; }
.ev-summary--error { color: var(--red) !important; }
.ev-role-badge { font-size: 9px; font-weight: 700; text-transform: uppercase; padding: 1px 4px; border-radius: 2px; margin-right: 4px; }
.ev-role-user      { background: var(--blue-dim); color: var(--blue); }
.ev-role-assistant { background: var(--purple-dim); color: var(--purple); }
.ev-role-system    { background: var(--gray-dim); color: var(--gray); }
.ev-role-tool      { background: var(--cyan-dim); color: var(--cyan); }
.ev-tags-row { display: flex; gap: 4px; margin-top: 3px; flex-wrap: wrap; }
.ev-tag { font-size: 10px; padding: 1px 5px; background: var(--bg-base); border: 1px solid var(--border); border-radius: 3px; color: var(--text-muted); font-family: var(--font-mono); }
`;
document.head.appendChild(_evCardStyle);
