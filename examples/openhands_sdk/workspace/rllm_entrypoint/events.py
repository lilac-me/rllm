"""
events.py — OpenHands event hook that feeds RunState.

Usage:
    from rllm_entrypoint.events import make_event_callback
    cb = make_event_callback(run_state)
    conversation = Conversation(agent=agent, workspace=..., callbacks=[cb])

The callback is called by OpenHands inside the conversation run loop
(possibly from additional threads for token streaming but ``on_event``
is always the main loop callback).  It is thread-safe because RunState
uses its own lock.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from rllm_entrypoint.state import AgentPhase, RunState

if TYPE_CHECKING:
    pass

# We import lazily at call time to avoid circular deps with OpenHands internals.


def _safe_str(v: Any, max_len: int = 2000) -> str:
    try:
        s = str(v)
        return s[:max_len] if len(s) > max_len else s
    except Exception:
        return "<repr failed>"


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


def _truncate_dict(d: Any, max_str_len: int = 500) -> Any:
    """Recursively truncate strings in a dict/list structure."""
    if isinstance(d, dict):
        return {k: _truncate_dict(v, max_str_len) for k, v in d.items()}
    if isinstance(d, list):
        return [_truncate_dict(v, max_str_len) for v in d]
    if isinstance(d, str) and len(d) > max_str_len:
        return d[:max_str_len] + "…"
    return d


def _phase_from_event(event_type: str) -> AgentPhase | None:
    """Map event type → AgentPhase transition (None = no change)."""
    _map = {
        "ActionEvent": AgentPhase.EXECUTING_COMMAND,
        "ObservationEvent": AgentPhase.WAITING_FOR_REPLY,
        "MessageEvent": AgentPhase.WAITING_FOR_REPLY,
        "PauseEvent": AgentPhase.PAUSED,
    }
    return _map.get(event_type)


def make_event_callback(run_state: RunState) -> Callable[[Any], None]:
    """Return a callback suitable for Conversation(callbacks=[...])."""

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

        except Exception:
            # Never crash the agent loop
            pass

    return on_event
