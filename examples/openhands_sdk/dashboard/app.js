/**
 * app.js — Core SPA: API client, router, state, SSE manager, and app init.
 */

// ── API Client ────────────────────────────────────────────────────────────
window.API = (() => {
  const BASE = '/api/v1';

  async function request(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
      opts.headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(BASE + path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    return res.json();
  }

  return {
    get: (p) => request('GET', p),
    post: (p, b) => request('POST', p, b),
    patch: (p, b) => request('PATCH', p, b),
    del: (p) => request('DELETE', p),

    sessions: {
      list: (status = 'all') => request('GET', `/sessions?status=${status}`),
      del: (id) => request('DELETE', `/sessions/${id}`),
      patch: (id, body) => request('PATCH', `/sessions/${id}`, body),
    },
    events: {
      list: (id, params = {}) => {
        const qs = new URLSearchParams(params).toString();
        return request('GET', `/sessions/${id}/events${qs ? '?' + qs : ''}`);
      },
      payload: (sid, dbId) => request('GET', `/sessions/${sid}/events/${dbId}/payload`),
      patchVisible: (sid, dbId, visible) => request('PATCH', `/sessions/${sid}/events/${dbId}`, { visible }),
    },
    state: (id) => request('GET', `/sessions/${id}/state`),
    kpi: (id) => request('GET', `/sessions/${id}/kpi`),
    stats: (id) => id ? request('GET', `/sessions/${id}/stats`) : request('GET', '/stats'),
    control: {
      get: (id) => request('GET', `/sessions/${id}/control`),
      pause: (id) => request('POST', `/sessions/${id}/control`, { pause: true }),
      resume: (id) => request('POST', `/sessions/${id}/control`, { resume: true }),
    },
  };
})();

// ── SSE Manager ───────────────────────────────────────────────────────────
window.SSEManager = (() => {
  const _sources = new Map(); // sessionId → EventSource
  const _handlers = new Map(); // sessionId → { state: fn, event: fn }

  function connect(sessionId, { onState, onEvent }) {
    if (_sources.has(sessionId)) return; // already connected
    const es = new EventSource(`/api/v1/sessions/${sessionId}/stream`);
    _sources.set(sessionId, es);
    _handlers.set(sessionId, { onState, onEvent });

    es.addEventListener('state', (e) => {
      try { onState && onState(JSON.parse(e.data)); } catch (_) {}
    });
    es.addEventListener('event', (e) => {
      try { onEvent && onEvent(JSON.parse(e.data)); } catch (_) {}
    });
    es.addEventListener('ping', () => {}); // keepalive
    es.onerror = () => {
      // Auto-reconnect is handled by browser EventSource
    };
  }

  function disconnect(sessionId) {
    const es = _sources.get(sessionId);
    if (es) { es.close(); _sources.delete(sessionId); _handlers.delete(sessionId); }
  }

  function disconnectAll() {
    for (const [id] of _sources) disconnect(id);
  }

  return { connect, disconnect, disconnectAll };
})();

// ── Toast ─────────────────────────────────────────────────────────────────
window.toast = (msg, type = '', duration = 3000) => {
  const el = document.createElement('div');
  el.className = `toast${type ? ' ' + type : ''}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), duration);
};

// ── App State ─────────────────────────────────────────────────────────────
const AppState = {
  sessions: [],
  currentSessionId: null,
  currentKPI: null,
  currentEvents: [],
  eventFilter: { type: '*', search: '', visible: '' },
  autoScroll: true,
  liveConnected: false,
};

// ── Uptime timer ──────────────────────────────────────────────────────────
let _uptimeInterval = null;
let _uptimeStart = null;
function startUptimeTimer(startTime) {
  if (_uptimeInterval) clearInterval(_uptimeInterval);
  _uptimeStart = startTime;
  function tick() {
    const el = document.getElementById('kpi-uptime-val');
    if (!el) return;
    const s = Math.floor(Date.now() / 1000 - _uptimeStart);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    el.textContent = h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
  }
  tick();
  _uptimeInterval = setInterval(tick, 1000);
}
function stopUptimeTimer() {
  if (_uptimeInterval) { clearInterval(_uptimeInterval); _uptimeInterval = null; }
}

// ── Session list rendering ────────────────────────────────────────────────
function renderSessionList(sessions) {
  AppState.sessions = sessions;
  const list = document.getElementById('session-list');
  const empty = document.getElementById('session-empty');
  const search = document.getElementById('session-search').value.toLowerCase();
  const statusFilter = document.getElementById('status-filter').value;
  const sortBy = document.getElementById('sort-filter').value;

  let filtered = sessions.filter(s => {
    const label = (s.label || s.session_id || '').toLowerCase();
    const matchSearch = !search || label.includes(search) || s.session_id.toLowerCase().includes(search);
    let phase = s.phase || 'unknown';
    if (s.is_running && phase === 'unknown') phase = 'running';
    const matchStatus = statusFilter === 'all' ||
      (statusFilter === 'running' && s.is_running) ||
      (statusFilter !== 'running' && phase === statusFilter);
    return matchSearch && matchStatus;
  });

  if (sortBy === 'created') filtered.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  else if (sortBy === 'name') filtered.sort((a, b) => (a.label || a.session_id).localeCompare(b.label || b.session_id));
  // default: by updated (API already sorted)

  // Update sidebar stats
  const running = sessions.filter(s => s.is_running).length;
  const errored = sessions.filter(s => s.phase === 'error').length;
  const finished = sessions.filter(s => !s.is_running && s.phase !== 'error').length;
  document.getElementById('stat-running').textContent = `${running} running`;
  document.getElementById('stat-finished').textContent = `${finished} done`;
  document.getElementById('stat-error').textContent = `${errored} err`;
  document.getElementById('global-session-count').textContent = `${sessions.length} sessions`;
  document.getElementById('global-running-count').textContent = `${running} running`;

  // Clear and re-render
  list.innerHTML = '';
  if (filtered.length === 0) { list.appendChild(empty); return; }

  for (const s of filtered) {
    const phase = s.is_running
      ? (s.phase || 'running')
      : (s.phase || 'finished');
    const phaseClass = s.is_running ? 'running' : (s.phase === 'error' ? 'error' : (s.phase === 'paused' ? 'paused' : 'finished'));
    const cost = s.accumulated_cost ? `$${Number(s.accumulated_cost).toFixed(4)}` : '';
    const uptime = s.uptime_seconds ? fmtUptime(s.uptime_seconds) : '';
    const label = s.label || s.session_id;
    const item = document.createElement('div');
    item.className = `session-item ${phaseClass}${s.session_id === AppState.currentSessionId ? ' active' : ''}`;
    item.dataset.sid = s.session_id;
    item.innerHTML = `
      <div class="session-item-header">
        <div class="phase-dot ${phaseClass}"></div>
        <div class="session-item-label" title="${label}">${label}</div>
      </div>
      <div class="session-item-meta">
        <span>iter <span class="session-meta-val">${s.iteration || 0}</span></span>
        ${s.event_count ? `<span>events <span class="session-meta-val">${s.event_count}</span></span>` : ''}
        ${cost ? `<span class="session-meta-val">${cost}</span>` : ''}
        ${uptime ? `<span class="session-meta-val">${uptime}</span>` : ''}
      </div>
      <div class="session-item-tags">
        ${s.llm_model ? `<span class="tag" title="${s.llm_model}">${s.llm_model.split('/').pop()}</span>` : ''}
        ${s.task_preview ? `<span class="tag" title="${s.task_preview}">${s.task_preview.slice(0, 40)}</span>` : ''}
      </div>`;
    item.addEventListener('click', () => selectSession(s.session_id));
    list.appendChild(item);
  }
}

function fmtUptime(s) {
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

// ── KPI panel ─────────────────────────────────────────────────────────────
function renderKPI(kpi, state) {
  const v = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  const phase = state?.phase || kpi?.phase || '—';
  const iter = state?.iteration ?? kpi?.iteration ?? '—';
  const cost = state?.accumulated_cost ?? kpi?.accumulated_cost;
  const llmCalls = state?.total_llm_calls ?? kpi?.total_llm_calls ?? '—';
  const events = kpi?.total_events ?? '—';
  const uptime = state?.uptime_seconds ?? kpi?.uptime_seconds;
  const errors = kpi?.error_count ?? 0;

  v('kpi-phase-val', phase);
  v('kpi-iter-val', iter);
  v('kpi-iter-sub', `/ ${state?.max_iterations || '?'} max`);
  v('kpi-cost-val', cost != null ? `$${Number(cost).toFixed(4)}` : '—');
  v('kpi-cost-sub', `${llmCalls} LLM calls`);
  v('kpi-events-val', events);
  v('kpi-events-sub', kpi ? `${(kpi.events_per_minute || 0).toFixed(1)}/min` : '');
  v('kpi-errors-val', errors);

  const phaseBadge = document.getElementById('session-phase-badge');
  if (phaseBadge) {
    phaseBadge.textContent = phase.replace(/_/g, ' ');
    phaseBadge.className = `phase-badge ${phase}`;
  }

  // Live uptime counter
  if (state?.is_running && state?.start_time) {
    startUptimeTimer(state.start_time);
    document.getElementById('kpi-uptime-sub').textContent = 'running';
  } else {
    stopUptimeTimer();
    if (uptime != null) document.getElementById('kpi-uptime-val').textContent = fmtUptime(uptime);
    document.getElementById('kpi-uptime-sub').textContent = 'total';
  }

  // Live badge
  const liveBadge = document.getElementById('header-live-badge');
  if (liveBadge) {
    if (state?.is_running) liveBadge.classList.remove('hidden');
    else liveBadge.classList.add('hidden');
  }

  // Pause / Resume buttons
  const pauseBtn = document.getElementById('ctrl-pause');
  const resumeBtn = document.getElementById('ctrl-resume');
  if (state?.is_running) {
    pauseBtn?.classList.remove('hidden');
    resumeBtn?.classList.add('hidden');
  } else if (state?.phase === 'paused') {
    pauseBtn?.classList.add('hidden');
    resumeBtn?.classList.remove('hidden');
  }

  // Phase timeline
  renderPhaseTimeline(kpi);
}

function renderPhaseTimeline(kpi) {
  const container = document.getElementById('phase-timeline');
  if (!container || !kpi?.phase_transitions?.length) return;

  const transitions = kpi.phase_transitions;
  const totalDuration = (kpi.last_event_at || 0) - (kpi.first_event_at || 0);
  if (totalDuration <= 0) return;

  const durationEl = document.getElementById('phase-duration-label');
  if (durationEl) durationEl.textContent = `${fmtUptime(kpi.duration_seconds || 0)} total`;

  container.innerHTML = '';
  for (let i = 0; i < transitions.length; i++) {
    const seg = transitions[i];
    const nextAt = transitions[i + 1]?.at || kpi.last_event_at;
    const segDur = nextAt - seg.at;
    const pct = Math.max(1, (segDur / totalDuration) * 100);
    const el = document.createElement('div');
    el.className = `phase-segment ${seg.phase}`;
    el.style.flex = pct.toFixed(2);
    el.title = `${seg.phase} (${fmtUptime(segDur)})`;
    el.textContent = pct > 10 ? seg.phase.replace(/_/g, ' ') : '';
    container.appendChild(el);
  }
}

// ── Tab management ────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden', 'active') && p.classList.remove('active'));
      btn.classList.add('active');
      const panel = document.getElementById(`tab-panel-${btn.dataset.tab}`);
      if (panel) { panel.classList.remove('hidden'); panel.classList.add('active'); }
      if (btn.dataset.tab === 'state') loadRawState();
      if (btn.dataset.tab === 'commands') loadCommands();
      if (btn.dataset.tab === 'llm-context') loadLLMContext();
      if (btn.dataset.tab === 'config') loadConfig();
    });
  });
}

// ── Config tab ────────────────────────────────────────────────────────────
async function loadConfig() {
  const state = await API.state(AppState.currentSessionId).catch(() => null);
  const container = document.getElementById('config-content');
  if (!state || !container) return;
  const fields = [
    ['session_id', state.session_id],
    ['session_label', state.session_label],
    ['llm_model', state.llm_model],
    ['llm_base_url_prefix', state.llm_base_url_prefix],
    ['workspace_base', state.workspace_base],
    ['max_iterations', state.max_iterations],
    ['conversation_id', state.conversation_id],
    ...Object.entries(state.env_vars || {}).map(([k, v]) => [k, v]),
  ];
  container.innerHTML = fields.map(([k, v]) =>
    `<div class="config-row"><div class="config-key">${k}</div><div class="config-val">${v ?? ''}</div></div>`
  ).join('');
}

// ── Raw state tab ─────────────────────────────────────────────────────────
async function loadRawState() {
  const state = await API.state(AppState.currentSessionId).catch(() => null);
  const el = document.getElementById('raw-state-content');
  if (el) el.textContent = JSON.stringify(state, null, 2);
}

// ── Commands tab ──────────────────────────────────────────────────────────
async function loadCommands() {
  const state = await API.state(AppState.currentSessionId).catch(() => null);
  const container = document.getElementById('commands-content');
  if (!container) return;
  const cmds = state?.command_history || [];
  if (!cmds.length) { container.innerHTML = '<div class="empty-state"><p>No commands recorded</p></div>'; return; }
  const rows = cmds.map(c => {
    const ok = (c.exit_code || 0) === 0;
    return `<tr>
      <td class="cmd-text">${escHtml(c.cmd || c.command || '')}</td>
      <td><span class="${ok ? 'badge-ok' : 'badge-err'}">${c.exit_code ?? '?'}</span></td>
      <td class="output-preview">${escHtml(String(c.stdout || '').slice(0, 100))}</td>
    </tr>`;
  }).join('');
  container.innerHTML = `<table class="data-table"><thead><tr><th>Command</th><th>Exit</th><th>Output preview</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// ── LLM Context tab ───────────────────────────────────────────────────────
async function loadLLMContext() {
  const state = await API.state(AppState.currentSessionId).catch(() => null);
  const container = document.getElementById('llm-context-content');
  if (!container) return;
  const msgs = state?.llm_context_snapshot || [];
  if (!msgs.length) { container.innerHTML = '<div class="empty-state"><p>No LLM context snapshot</p></div>'; return; }
  container.innerHTML = msgs.map(m => {
    const role = m.role || 'unknown';
    const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
    return `<div class="llm-msg ${role}"><div class="llm-role">${role}</div><div class="llm-content">${escHtml(content.slice(0, 600))}${content.length > 600 ? '…' : ''}</div></div>`;
  }).join('');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Event Timeline ────────────────────────────────────────────────────────
const LIFECYCLE_TYPES = new Set(['StartupEvent','HeartbeatEvent','EvaluateEvent','FinishEvent']);
const COLLAPSE_TYPES = new Set(['HeartbeatEvent','ConversationStateUpdateEvent']);
const GAP_THRESHOLD = 5; // seconds before showing a gap divider

async function loadEventTimeline(sessionId) {
  const loading = document.getElementById('timeline-loading');
  const empty = document.getElementById('timeline-empty');
  if (loading) loading.classList.remove('hidden');
  if (empty) empty.classList.add('hidden');

  const filter = AppState.eventFilter;
  const params = { limit: 500 };
  if (filter.visible) params.visible = filter.visible;

  let events;
  try {
    const res = await API.events.list(sessionId, params);
    events = res.items || [];
  } catch (e) {
    if (loading) loading.classList.add('hidden');
    toast('Failed to load events: ' + e.message, 'error');
    return;
  }

  AppState.currentEvents = events;
  if (loading) loading.classList.add('hidden');
  renderTimeline(events);
}

function renderTimeline(events) {
  const container = document.getElementById('event-timeline');
  const empty = document.getElementById('timeline-empty');
  if (!container) return;

  // Apply client-side filters
  const f = AppState.eventFilter;
  let filtered = events.filter(ev => {
    if (f.type !== '*') {
      if (f.type === 'lifecycle') {
        if (!LIFECYCLE_TYPES.has(ev.event_type)) return false;
      } else if (ev.event_type !== f.type) {
        return false;
      }
    }
    if (f.search && !String(ev.summary || '').toLowerCase().includes(f.search.toLowerCase())) return false;
    return true;
  });

  // Remove old event nodes (keep loading/empty)
  Array.from(container.children).forEach(c => {
    if (!c.id) c.remove();
  });

  if (!filtered.length) { empty?.classList.remove('hidden'); return; }
  empty?.classList.add('hidden');

  const frag = document.createDocumentFragment();
  let prevTs = null;
  let groupType = null, groupCount = 0, groupStart = null;

  const flushGroup = () => {
    if (groupCount > 1) {
      frag.appendChild(EventCards.renderGroup(groupType, groupCount, () => {
        // Expand: re-render without collapsing
        AppState._noCollapse = groupType;
        renderTimeline(events);
        AppState._noCollapse = null;
      }));
    } else if (groupCount === 1 && groupStart) {
      frag.appendChild(EventCards.render(groupStart, {
        onOpen: openDrawer,
        onToggleVisible: toggleEventVisible,
      }));
    }
    groupType = null; groupCount = 0; groupStart = null;
  };

  for (const ev of filtered) {
    const ts = ev.received_at || 0;

    // Gap indicator
    if (prevTs !== null && ts - prevTs > GAP_THRESHOLD) {
      flushGroup();
      frag.appendChild(EventCards.renderGap(ts - prevTs));
    }

    const shouldCollapse = COLLAPSE_TYPES.has(ev.event_type) && AppState._noCollapse !== ev.event_type;
    if (shouldCollapse) {
      if (groupType === ev.event_type) {
        groupCount++;
      } else {
        flushGroup();
        groupType = ev.event_type;
        groupCount = 1;
        groupStart = ev;
      }
    } else {
      flushGroup();
      frag.appendChild(EventCards.render(ev, {
        onOpen: openDrawer,
        onToggleVisible: toggleEventVisible,
      }));
    }
    prevTs = ts;
  }
  flushGroup();
  container.appendChild(frag);

  if (AppState.autoScroll) {
    container.scrollTop = container.scrollHeight;
  }
}

// ── Event detail drawer ───────────────────────────────────────────────────
async function openDrawer(ev) {
  const drawer = document.getElementById('event-drawer');
  const backdrop = document.getElementById('drawer-backdrop');
  const typeEl = document.getElementById('drawer-event-type');
  const summaryEl = document.getElementById('drawer-event-summary');
  const metaEl = document.getElementById('drawer-meta');
  const payloadEl = document.getElementById('drawer-payload');

  if (typeEl) { typeEl.textContent = ev.event_type; typeEl.className = 'drawer-type-badge'; }
  if (summaryEl) summaryEl.textContent = ev.summary || '';

  if (metaEl) {
    metaEl.innerHTML = [
      ['ID', ev.id],
      ['event_id', ev.event_id || '—'],
      ['source', ev.source || '—'],
      ['timestamp', EventCards.fmtAbsTime(ev.received_at || ev.timestamp)],
      ['visible', ev.visible === 0 ? 'hidden' : 'visible'],
    ].map(([k, v]) => `<div class="drawer-meta-row"><span class="drawer-meta-key">${k}</span><span class="drawer-meta-val">${v}</span></div>`).join('');
  }

  // Load full payload
  if (payloadEl) {
    payloadEl.textContent = 'Loading…';
    try {
      const payload = await API.events.payload(AppState.currentSessionId, ev.id);
      payloadEl.textContent = JSON.stringify(payload, null, 2);
    } catch (_) {
      payloadEl.textContent = JSON.stringify(ev, null, 2);
    }
  }

  drawer?.classList.add('open');
  backdrop?.classList.remove('hidden');
}

function closeDrawer() {
  document.getElementById('event-drawer')?.classList.remove('open');
  document.getElementById('drawer-backdrop')?.classList.add('hidden');
}

// ── Toggle event visibility ───────────────────────────────────────────────
async function toggleEventVisible(ev, cardEl) {
  const newVisible = !(ev.visible !== 0 && ev.visible !== false);
  try {
    await API.events.patchVisible(AppState.currentSessionId, ev.id, newVisible);
    ev.visible = newVisible ? 1 : 0;
    cardEl.dataset.visible = newVisible ? '1' : '0';
    cardEl.classList.toggle('ev-hidden', !newVisible);
    toast(newVisible ? 'Event visible' : 'Event hidden');
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

// ── Session selection ─────────────────────────────────────────────────────
async function selectSession(sessionId) {
  // Clear previous SSE
  if (AppState.currentSessionId && AppState.currentSessionId !== sessionId) {
    SSEManager.disconnect(AppState.currentSessionId);
  }

  AppState.currentSessionId = sessionId;

  // UI: show session view
  document.getElementById('welcome-view')?.classList.add('hidden');
  const sessionView = document.getElementById('session-view');
  sessionView?.classList.remove('hidden');

  // Highlight sidebar item
  document.querySelectorAll('.session-item').forEach(el => {
    el.classList.toggle('active', el.dataset.sid === sessionId);
  });

  // Set header info from session list cache
  const sess = AppState.sessions.find(s => s.session_id === sessionId) || {};
  const label = sess.label || sess.session_id || sessionId;
  const labelDisplay = document.getElementById('session-label-display');
  if (labelDisplay) labelDisplay.textContent = label;
  const headerSid = document.getElementById('header-session-id');
  if (headerSid) headerSid.textContent = sessionId.slice(0, 16) + '…';
  const headerModel = document.getElementById('header-llm-model');
  if (headerModel) headerModel.textContent = sess.llm_model || '';

  // Load KPI
  const [kpiData, stateData] = await Promise.all([
    API.kpi(sessionId).catch(() => null),
    API.state(sessionId).catch(() => null),
  ]);
  AppState.currentKPI = kpiData;
  renderKPI(kpiData, stateData);

  // Load events
  await loadEventTimeline(sessionId);

  // Connect SSE for live sessions
  if (sess.is_running) {
    SSEManager.connect(sessionId, {
      onState: (state) => {
        renderKPI(AppState.currentKPI, state);
      },
      onEvent: (ev) => {
        // Append new event card live
        const timeline = document.getElementById('event-timeline');
        const card = EventCards.render(ev, {
          onOpen: openDrawer,
          onToggleVisible: toggleEventVisible,
        });
        const emptyEl = document.getElementById('timeline-empty');
        emptyEl?.classList.add('hidden');
        timeline?.appendChild(card);
        if (AppState.autoScroll) timeline.scrollTop = timeline.scrollHeight;
      },
    });
  }
}

// ── Session label edit ────────────────────────────────────────────────────
function initLabelEdit() {
  const display = document.getElementById('session-label-display');
  const editInput = document.getElementById('session-label-edit');
  if (!display || !editInput) return;

  display.addEventListener('click', () => {
    display.classList.add('hidden');
    editInput.classList.remove('hidden');
    editInput.value = display.textContent;
    editInput.focus();
    editInput.select();
  });

  const save = async () => {
    const newLabel = editInput.value.trim();
    editInput.classList.add('hidden');
    display.classList.remove('hidden');
    if (!newLabel || newLabel === display.textContent || !AppState.currentSessionId) return;
    try {
      await API.sessions.patch(AppState.currentSessionId, { label: newLabel });
      display.textContent = newLabel;
      toast('Label updated', 'success');
      refreshSessions();
    } catch (e) {
      toast('Failed: ' + e.message, 'error');
    }
  };

  editInput.addEventListener('blur', save);
  editInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') save();
    if (e.key === 'Escape') { editInput.blur(); }
  });
}

// ── Filter logic ──────────────────────────────────────────────────────────
function initFilters() {
  // Event type quick chips
  document.querySelectorAll('.quick-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.quick-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      AppState.eventFilter.type = chip.dataset.type;
      renderTimeline(AppState.currentEvents);
    });
  });

  // Search
  let searchTimer;
  document.getElementById('event-search')?.addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      AppState.eventFilter.search = e.target.value;
      renderTimeline(AppState.currentEvents);
    }, 200);
  });

  // Visibility filter
  document.getElementById('event-visible-filter')?.addEventListener('change', (e) => {
    AppState.eventFilter.visible = e.target.value;
    if (AppState.currentSessionId) loadEventTimeline(AppState.currentSessionId);
  });

  // Auto-scroll toggle
  document.getElementById('auto-scroll-toggle')?.addEventListener('change', (e) => {
    AppState.autoScroll = e.target.checked;
  });

  // Scroll to bottom
  document.getElementById('btn-scroll-bottom')?.addEventListener('click', () => {
    const tl = document.getElementById('event-timeline');
    if (tl) tl.scrollTop = tl.scrollHeight;
  });
}

// ── Session list filters ──────────────────────────────────────────────────
function initSessionFilters() {
  document.getElementById('session-search')?.addEventListener('input', () => renderSessionList(AppState.sessions));
  document.getElementById('status-filter')?.addEventListener('change', () => renderSessionList(AppState.sessions));
  document.getElementById('sort-filter')?.addEventListener('change', () => renderSessionList(AppState.sessions));
}

// ── Control buttons ───────────────────────────────────────────────────────
function initControls() {
  document.getElementById('ctrl-pause')?.addEventListener('click', async () => {
    if (!AppState.currentSessionId) return;
    try { await API.control.pause(AppState.currentSessionId); toast('Pause signal sent', 'success'); }
    catch (e) { toast('Pause failed: ' + e.message, 'error'); }
  });
  document.getElementById('ctrl-resume')?.addEventListener('click', async () => {
    if (!AppState.currentSessionId) return;
    try { await API.control.resume(AppState.currentSessionId); toast('Resume signal sent', 'success'); }
    catch (e) { toast('Resume failed: ' + e.message, 'error'); }
  });
  document.getElementById('ctrl-delete')?.addEventListener('click', async () => {
    if (!AppState.currentSessionId) return;
    if (!confirm(`Delete session ${AppState.currentSessionId}? This cannot be undone.`)) return;
    try {
      await API.sessions.del(AppState.currentSessionId);
      toast('Session deleted', 'success');
      document.getElementById('session-view')?.classList.add('hidden');
      document.getElementById('welcome-view')?.classList.remove('hidden');
      AppState.currentSessionId = null;
      stopUptimeTimer();
      refreshSessions();
    } catch (e) { toast('Delete failed: ' + e.message, 'error'); }
  });
  document.getElementById('copy-state-btn')?.addEventListener('click', () => {
    const text = document.getElementById('raw-state-content')?.textContent;
    if (text) { navigator.clipboard.writeText(text); toast('Copied to clipboard'); }
  });
}

// ── Drawer close ──────────────────────────────────────────────────────────
function initDrawer() {
  document.getElementById('drawer-close')?.addEventListener('click', closeDrawer);
  document.getElementById('drawer-backdrop')?.addEventListener('click', closeDrawer);
}

// ── Refresh sessions ──────────────────────────────────────────────────────
async function refreshSessions() {
  try {
    const sessions = await API.sessions.list();
    renderSessionList(sessions);
  } catch (e) {
    // Gateway may not be running; fail silently
  }
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────
function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'r' || e.key === 'R') { e.preventDefault(); refreshSessions(); }
    if (e.key === '/') { e.preventDefault(); document.getElementById('session-search')?.focus(); }
    if (e.key === 'Escape') closeDrawer();
    if (e.key === 'j' || e.key === 'J') {
      const items = [...document.querySelectorAll('.session-item')];
      const cur = items.findIndex(el => el.dataset.sid === AppState.currentSessionId);
      const next = items[cur + 1];
      if (next) selectSession(next.dataset.sid);
    }
    if (e.key === 'k' || e.key === 'K') {
      const items = [...document.querySelectorAll('.session-item')];
      const cur = items.findIndex(el => el.dataset.sid === AppState.currentSessionId);
      const prev = items[Math.max(0, cur - 1)];
      if (prev) selectSession(prev.dataset.sid);
    }
  });
}

// ── Tab panel active state fix ────────────────────────────────────────────
function fixTabPanels() {
  // Ensure correct initial state
  document.querySelectorAll('.tab-panel').forEach(p => {
    if (!p.classList.contains('active')) p.classList.add('hidden');
    else p.classList.remove('hidden');
  });
}

// ── Main init ─────────────────────────────────────────────────────────────
async function init() {
  fixTabPanels();
  initTabs();
  initFilters();
  initSessionFilters();
  initControls();
  initDrawer();
  initLabelEdit();
  initKeyboard();

  document.getElementById('refresh-btn')?.addEventListener('click', refreshSessions);

  // Initial load
  await refreshSessions();

  // Auto-refresh sessions every 15s
  setInterval(refreshSessions, 15000);
}

document.addEventListener('DOMContentLoaded', init);
