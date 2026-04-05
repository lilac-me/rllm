# OpenHands SDK Example — Observability & Control System

This documentation covers the **observability and control** infrastructure added to the
`openhands_sdk` example. It explains how to monitor running containers, stream events,
and send pause/resume commands.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ rllm Training Host                                                   │
│                                                                      │
│  openhands_agent.py ──docker run──► Container                        │
│                                     ├─ entrypoint.py (thin shell)    │
│                                     └─ rllm_entrypoint/              │
│                                          ├─ config.py                │
│                                          ├─ state.py (RunState)      │
│                                          ├─ events.py (hook)         │
│                                          ├─ api_client.py ──POST──►  │
│                                          ├─ uploader.py (thread)     │◄┐
│                                          ├─ pause_ctrl.py (thread) ──┘│
│                                          └─ runner.py                  │
│                                                                         │
│  observer_gateway.py ◄─── POST /state, POST /events ──────────────────┘
│  (HTTP :8765)         ─── GET  /control ──────────────────────────────►
│                                                                      │
│  observer_cli.py      ─── interacts with gateway ────────────────►  │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Components

| File | Role |
|------|------|
| `workspace/entrypoint.py` | Container ENTRYPOINT — thin shell, delegates to package |
| `workspace/rllm_entrypoint/config.py` | All env-var config, frozen dataclass |
| `workspace/rllm_entrypoint/state.py` | Thread-safe `RunState` with all observable data |
| `workspace/rllm_entrypoint/events.py` | OpenHands event callback → `RunState` |
| `workspace/rllm_entrypoint/api_client.py` | Stdlib-only REST client (push/poll) |
| `workspace/rllm_entrypoint/uploader.py` | Daemon thread: periodic state push |
| `workspace/rllm_entrypoint/pause_ctrl.py` | Daemon thread: pause/resume polling |
| `workspace/rllm_entrypoint/runner.py` | Full conversation lifecycle |
| `observer_gateway.py` | Standalone REST server (no deps, stdlib only) |
| `observer_cli.py` | CLI for humans / scripts |

---

## Quick Start

### 1. Start the Observer Gateway (with SQLite persistence)

```bash
# Data stored in ./observer_data.db (auto-created)
python examples/openhands_sdk/observer_gateway.py --port 8765

# Custom DB path
python examples/openhands_sdk/observer_gateway.py --port 8765 --db /data/runs.db
```

The gateway persists **all** uploaded data to SQLite:
- Every state snapshot (with timestamp and client IP)
- Every OpenHands event (deduplicated by event ID)
- Every HTTP request (method, path, timing, status)
- Every control action (pause/resume) with direction and IP

### 2. Launch a Container with Observability Enabled

Add `OBSERVER_API_URL` and `OBSERVER_SESSION_ID` env vars to your `docker run`:

```bash
docker run --rm \
  -e LLM_BASE_URL="http://host.docker.internal:4000/.../v1" \
  -e LLM_MODEL="openai/openhands-model" \
  -e LLM_API_KEY="EMPTY" \
  -e TASK_INSTRUCTION="Fix the bug in main.py" \
  -e WORKSPACE_BASE=/opt/workspace \
  -e MAX_ITERATIONS=30 \
  -e OBSERVER_API_URL="http://host.docker.internal:8765" \
  -e OBSERVER_SESSION_ID="run-001" \
  -e OBSERVER_SESSION_LABEL="my-experiment" \
  -e OBSERVER_UPLOAD_INTERVAL="5" \
  -e OBSERVER_PAUSE_POLL_INTERVAL="2" \
  --add-host host.docker.internal:host-gateway \
  -v /path/to/workspace:/opt/workspace \
  rllm-openhands
```

### 3. Monitor from the CLI

```bash
# List all sessions
python observer_cli.py sessions

# Watch a session live (refreshes every 3s)
python observer_cli.py watch run-001

# Show detailed state
python observer_cli.py state run-001 --verbose

# Show last 20 events
python observer_cli.py events run-001 --limit 20

# Pause the container
python observer_cli.py pause run-001

# Resume the container
python observer_cli.py resume run-001
```

---

## REST API Reference

Base URL: `http://<host>:8765`

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/sessions` | List all known sessions (summary) |
| `DELETE` | `/api/v1/sessions/{id}` | Remove a session record |

### State

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sessions/{id}/state` | Upload full state (container → gateway) |
| `GET` | `/api/v1/sessions/{id}/state` | Retrieve latest state snapshot |

### Events

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sessions/{id}/events` | Upload a single event |
| `GET` | `/api/v1/sessions/{id}/events?limit=N` | Get recent events (default limit=100) |

### Control

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/sessions/{id}/control` | Get current control flags |
| `POST` | `/api/v1/sessions/{id}/control` | Set `{"pause": true}` or `{"resume": true}` |

---

## Observable State Fields

The `/state` endpoint returns a JSON object with these fields:

```json
{
  "session_id": "run-001",
  "session_label": "my-experiment",
  "task_instruction": "Fix the bug in...",
  "workspace_base": "/opt/workspace",
  "llm_model": "openai/openhands-model",
  "llm_base_url_prefix": "http://host.docker.internal:4000/...",
  "max_iterations": 30,
  "is_running": true,
  "phase": "executing_command",
  "uptime_seconds": 47.3,
  "start_time": 1712000000.0,
  "end_time": null,
  "iteration": 5,
  "total_llm_calls": 8,
  "accumulated_cost": 0.0032,
  "conversation_execution_status": "running",
  "conversation_id": "3f9a1b2c-...",
  "events_count": 42,
  "events": [...],
  "command_history": [...],
  "llm_context_snapshot": [{"role": "user", "content": "..."}, ...],
  "env_vars": {"WORKSPACE_BASE": "/opt/workspace", ...},
  "extra_metadata": {},
  "last_error": ""
}
```

### Phases

| Phase | Meaning |
|-------|---------|
| `initializing` | SDK objects being built |
| `waiting_for_reply` | LLM call in flight |
| `executing_command` | Tool (terminal/file) executing |
| `paused` | Paused by gateway request |
| `finished` | Completed normally |
| `error` | Unhandled exception |

### Event Object Shape

```json
{
  "event_type": "ActionEvent",
  "event_id": "3f8a...",
  "source": "agent",
  "timestamp": "2024-04-01T12:00:00",
  "summary": "[Action:terminal] Running pytest...",
  "raw": {...}
}
```

---

## Environment Variables Reference

### Core (required)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | — | Proxied rllm LiteLLM URL |
| `LLM_API_KEY` | `EMPTY` | API key |
| `LLM_MODEL` | `openai/openhands-model` | Model name |
| `TASK_INSTRUCTION` | (from INSTRUCTIONS.md) | Task instruction text |
| `WORKSPACE_BASE` | `/opt/workspace` | Workspace directory |
| `MAX_ITERATIONS` | `30` | Max agent iterations |
| `NPU_OPERATOR_TASK` | `0` | `1` = kernel/operator task |
| `OPERATOR_BACKEND` | `triton` | `triton` \| `ascendc` |

### Observability (all optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVER_API_URL` | `""` | Base URL of observer gateway; disabled if empty |
| `OBSERVER_SESSION_ID` | random UUID | Session identifier |
| `OBSERVER_SESSION_LABEL` | first 12 chars of ID | Human label |
| `OBSERVER_UPLOAD_INTERVAL` | `5.0` | State push interval (seconds) |
| `OBSERVER_PAUSE_POLL_INTERVAL` | `2.0` | Pause/resume poll interval (seconds) |
| `OBSERVER_EXTRA_METADATA` | `""` | JSON string with arbitrary metadata |
| `SYSTEM_PROMPT_PATH` | `/tmp/rllm_minimal_system.j2` | System prompt template path |

---

## Pause / Resume Flow

1. **Pause**: Call `POST /api/v1/sessions/{id}/control` with `{"pause": true}`.
2. The container's `pause_ctrl` thread polls the gateway (default: every 2s).
3. On seeing `pause: true`, it calls `conversation.pause()` (SDK thread-safe method).
4. The conversation loop breaks between agent steps and the phase becomes `paused`.
5. **Resume**: Call `POST /api/v1/sessions/{id}/control` with `{"resume": true}`.
6. The `pause_ctrl` thread sets `resume_requested = True`.
7. The main runner loop detects this flag, calls `conversation.run()` again.

> **Note**: The pause takes effect at the **next agent step boundary**. If the agent
> is mid-LLM-call (waiting for a token stream), it will pause only after that call
> completes.

---

## Project Structure

```
examples/openhands_sdk/
├── observer_gateway.py          ← host-side REST gateway server
├── observer_cli.py              ← CLI for humans/scripts
├── openhands_agent.py           ← rllm rollout function (unchanged API)
├── train_openhands.py
├── train_openhands.sh
└── workspace/
    ├── Dockerfile               ← builds rllm-openhands image
    ├── entrypoint.py            ← thin container ENTRYPOINT shell
    ├── INSTRUCTIONS.md
    ├── rllm_entrypoint/         ← NEW sub-package
    │   ├── __init__.py
    │   ├── config.py            ← env-var config
    │   ├── state.py             ← RunState dataclass
    │   ├── events.py            ← OpenHands event hook
    │   ├── api_client.py        ← stdlib REST client
    │   ├── uploader.py          ← background uploader thread
    │   ├── pause_ctrl.py        ← background pause/resume thread
    │   └── runner.py            ← main conversation runner
    ├── src/
    └── tools/
```
