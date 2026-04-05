"""
events.py — OpenHands event hook that drives event-driven push to the
observer gateway.

Design (v2 — event-driven):
  Each OpenHands SDK event is immediately serialised and pushed to the
  gateway via ObserverClient.push_event().  There is no background upload
  thread; the push happens synchronously inside the callback (with a short
  timeout so it never stalls the agent loop).

  Additional lifecycle helpers are provided for structured events that are
  NOT native OpenHands SDK events:
    - push_startup_event()     — emitted once at container start
    - push_heartbeat_event()   — emitted periodically by a lightweight timer
    - push_evaluate_event()    — emitted when evaluation metrics are available
    - push_finish_event()      — emitted when the conversation ends

Usage:
    from rllm_entrypoint.events import make_event_callback, push_startup_event
    cb = make_event_callback(run_state, client)
    conversation = Conversation(agent=agent, workspace=..., callbacks=[cb])
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from rllm_entrypoint.state import AgentPhase, RunState

if TYPE_CHECKING:
    from rllm_entrypoint.api_client import ObserverClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(v: Any, max_len: int = 2000) -> str:
    try:
        s = str(v)
        return s[:max_len] if len(s) > max_len else s
    except Exception:
        return "<repr failed>"


def _truncate_dict(d: Any, max_str_len: int = 500) -> Any:
    """Recursively truncate strings in a dict/list structure."""
    if isinstance(d, dict):
        return {k: _truncate_dict(v, max_str_len) for k, v in d.items()}
    if isinstance(d, list):
        return [_truncate_dict(v, max_str_len) for v in d]
    if isinstance(d, str) and len(d) > max_str_len:
        return d[:max_str_len] + "…"
    return d


def _make_lifecycle_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
    """Build a structured lifecycle event dict (non-SDK events)."""
    return {
        "event_type": event_type,
        "event_id": str(uuid.uuid4()),
        "source": "rllm_entrypoint",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": f"[{event_type}]",
        **kwargs,
    }

# ---------------------------------------------------------------------------
# SDK event serialisation
# ---------------------------------------------------------------------------

def _event_to_dict(event: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert any OpenHands SDK Event to a flat serialisable dict."""
    # Try pydantic model_dump first (SDK Events are Pydantic models).
    try:
        raw = event.model_dump()
    except AttributeError:
        raw = {}

    event_type = type(event).__name__
    source = getattr(event, "source", "unknown")
    timestamp = getattr(event, "timestamp", None) or time.time()
    event_id = getattr(event, "id", None) or ""

    # Build a human-readable summary per event type.
    summary = _build_summary(event, event_type)

    return {
        "event_type": event_type,
        "event_id": str(event_id),
        "source": str(source),
        "timestamp": str(timestamp),
        "summary": summary,
        "raw": _truncate_dict(raw, max_str_len=500),
    }


def _build_summary(event: Any, event_type: str) -> str:
    """Build a concise one-line summary for the event."""
    try:
        # ActionEvent: show tool name + thought preview
        if event_type == "ActionEvent":
            tool = getattr(event, "tool_name", "")
            thought = getattr(event, "thought", [])
            thought_text = " ".join(
                getattr(t, "text", str(t)) for t in thought
            )[:200]
            return f"[Action:{tool}] {thought_text}"

        # ObservationEvent: show tool + result preview
        if event_type == "ObservationEvent":
            tool = getattr(event, "tool_name", "")
            obs = getattr(event, "observation", None)
            if obs is not None:
                try:
                    content_list = obs.to_llm_content
                    text = " ".join(
                        getattr(c, "text", str(c)) for c in content_list
                    )[:300]
                except Exception:
                    text = _safe_str(obs, 300)
            else:
                text = ""
            return f"[Observation:{tool}] {text}"

        # MessageEvent
        if event_type == "MessageEvent":
            msg = getattr(event, "llm_message", None)
            if msg is not None:
                content = getattr(msg, "content", [])
                text = " ".join(
                    getattr(c, "text", str(c)) for c in content
                )[:200]
                role = getattr(msg, "role", "?")
                return f"[Message:{role}] {text}"

        # AgentErrorEvent
        if event_type == "AgentErrorEvent":
            return f"[Error] {getattr(event, 'error', '')[:300]}"

        # PauseEvent
        if event_type == "PauseEvent":
            return "[Paused]"

        # ConversationStateUpdateEvent
        if event_type == "ConversationStateUpdateEvent":
            key = getattr(event, "key", "")
            value = getattr(event, "value", "")
            return f"[StateUpdate] {key}={_safe_str(value, 80)}"

        # LLMCompletionLogEvent
        if event_type == "LLMCompletionLogEvent":
            return "[LLMCompletion]"

        return f"[{event_type}]"
    except Exception:
        return f"[{event_type}] <summary failed>"


def _phase_from_event(event_type: str) -> AgentPhase | None:
    """Map event type → AgentPhase transition (None = no change)."""
    _map = {
        "ActionEvent": AgentPhase.EXECUTING_COMMAND,
        "ObservationEvent": AgentPhase.WAITING_FOR_REPLY,
        "MessageEvent": AgentPhase.WAITING_FOR_REPLY,
        "PauseEvent": AgentPhase.PAUSED,
    }
    return _map.get(event_type)

# ---------------------------------------------------------------------------
# Main callback factory
# ---------------------------------------------------------------------------

def make_event_callback(
    run_state: RunState,
    client: "ObserverClient",
) -> Callable[[Any], None]:
    """Return a callback suitable for Conversation(callbacks=[...]).

    Each event is:
      1. Serialised to a dict.
      2. Recorded in RunState (for local state tracking).
      3. Immediately pushed to the gateway via client.push_event().
    """

    def on_event(event: Any) -> None:  # noqa: ANN401
        try:
            event_dict = _event_to_dict(event)
            run_state.record_event(event_dict)

            # Phase transitions
            phase = _phase_from_event(event_dict["event_type"])
            if phase is not None:
                run_state.set_phase(phase)

            # Track iterations: count ActionEvents
            if event_dict["event_type"] == "ActionEvent":
                run_state.increment_iteration()

            # ── Direct push to gateway (event-driven) ──────────────────
            client.push_event(event_dict)

        except Exception:
            # Never crash the agent loop
            pass

    return on_event

# ---------------------------------------------------------------------------
# Lifecycle event helpers  (call from runner.py at key moments)
# ---------------------------------------------------------------------------

def push_startup_event(client: "ObserverClient", run_state: RunState) -> None:
    """Emit a 'startup' lifecycle event when the container starts."""
    ev = _make_lifecycle_event(
        "StartupEvent",
        summary="[Startup] Container initialised and runner started",
        session_id=run_state.session_id,
        session_label=getattr(run_state, "session_label", ""),
        llm_model=getattr(run_state, "llm_model", ""),
        llm_base_url=getattr(run_state, "llm_base_url", ""),
        workspace_base=getattr(run_state, "workspace_base", ""),
        max_iterations=getattr(run_state, "max_iterations", 0),
        task_preview=getattr(run_state, "task_instruction", "")[:300],
    )
    logger.info("[events] Pushing StartupEvent")
    client.push_event(ev)


def push_heartbeat_event(client: "ObserverClient", run_state: RunState) -> None:
    """Emit a lightweight heartbeat (call from a periodic timer in runner)."""
    with run_state._lock:
        phase = run_state.phase.value if hasattr(run_state.phase, "value") else str(run_state.phase)
        iteration = run_state.iteration
        is_running = run_state.is_running

    ev = _make_lifecycle_event(
        "HeartbeatEvent",
        summary=f"[Heartbeat] phase={phase} iteration={iteration} running={is_running}",
        phase=phase,
        iteration=iteration,
        is_running=is_running,
        uptime_seconds=run_state.uptime_seconds if hasattr(run_state, "uptime_seconds") else 0,
    )
    logger.debug("[events] Pushing HeartbeatEvent (phase=%s iter=%d)", phase, iteration)
    client.push_event(ev)


def push_evaluate_event(
    client: "ObserverClient",
    run_state: RunState,
    *,
    status: str,
    cost: float = 0.0,
    llm_calls: int = 0,
    iterations: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit an evaluation/summary event at conversation end."""
    ev = _make_lifecycle_event(
        "EvaluateEvent",
        summary=f"[Evaluate] status={status} cost={cost:.4f} llm_calls={llm_calls} iterations={iterations}",
        status=status,
        accumulated_cost=cost,
        total_llm_calls=llm_calls,
        iterations=iterations,
        **(extra or {}),
    )
    logger.info("[events] Pushing EvaluateEvent (status=%s cost=%.4f)", status, cost)
    client.push_event(ev)


def push_finish_event(
    client: "ObserverClient",
    run_state: RunState,
    *,
    exit_code: int,
    reason: str = "",
) -> None:
    """Emit a 'finish' lifecycle event as the last thing the container does."""
    with run_state._lock:
        phase = run_state.phase.value if hasattr(run_state.phase, "value") else str(run_state.phase)
        iteration = run_state.iteration

    ev = _make_lifecycle_event(
        "FinishEvent",
        summary=f"[Finish] exit_code={exit_code} phase={phase} reason={reason}",
        exit_code=exit_code,
        phase=phase,
        iteration=iteration,
        reason=reason,
    )
    logger.info("[events] Pushing FinishEvent (exit_code=%d, reason=%s)", exit_code, reason)
    client.push_event(ev)
