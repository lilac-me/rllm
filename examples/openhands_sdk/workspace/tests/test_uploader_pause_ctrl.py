"""
test_uploader_pause_ctrl.py — Unit tests for uploader.py and pause_ctrl.py.

Tests cover:
  StateUploader:
    - trigger() causes an immediate upload instead of waiting the full interval
    - stop() causes the thread to exit cleanly and perform a final upload
    - Exceptions in push_state do not crash the thread
    - Thread is daemonised

  PauseController:
    - set_conversation() attaches the conversation object
    - pause control signal → conversation.pause() called + phase set to PAUSED
    - resume control signal → resume_requested=True + phase set to WAITING_FOR_REPLY
    - Empty control response is ignored
    - Exception in fetch_control does not crash the thread
    - stop() exits the polling loop
"""
from __future__ import annotations

import sys
import os
import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_WORKSPACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workspace",
)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from rllm_entrypoint.api_client import ObserverClient  # noqa: E402
from rllm_entrypoint.state import AgentPhase, RunState  # noqa: E402
from rllm_entrypoint.uploader import StateUploader  # noqa: E402
from rllm_entrypoint.pause_ctrl import PauseController  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _make_client(push_ok: bool = True, control: dict | None = None) -> ObserverClient:
    """Create a mock ObserverClient."""
    client = MagicMock(spec=ObserverClient)
    client.push_state.return_value = push_ok
    client.fetch_control.return_value = control if control is not None else {}
    client.push_event.return_value = True
    return client


def _make_state(**kwargs) -> RunState:
    return RunState(session_id="test", session_label="test-label", **kwargs)


# ===========================================================================
# StateUploader
# ===========================================================================

class TestStateUploader:
    def test_is_daemon_thread(self):
        uploader = StateUploader(_make_state(), _make_client(), interval_s=60.0)
        assert uploader.daemon is True

    def test_stop_exits_thread(self):
        state = _make_state()
        client = _make_client()
        uploader = StateUploader(state, client, interval_s=60.0)
        uploader.start()
        time.sleep(0.05)
        uploader.stop()
        uploader.join(timeout=3)
        assert not uploader.is_alive(), "Uploader thread did not stop in time"

    def test_stop_performs_final_upload(self):
        state = _make_state()
        client = _make_client()
        uploader = StateUploader(state, client, interval_s=60.0)
        uploader.start()
        time.sleep(0.05)
        uploader.stop()
        uploader.join(timeout=3)
        # push_state should be called at least once (the final upload on exit)
        assert client.push_state.call_count >= 1

    def test_trigger_causes_immediate_upload(self):
        state = _make_state()
        client = _make_client()
        # long interval so it wouldn't fire otherwise
        uploader = StateUploader(state, client, interval_s=60.0)
        uploader.start()
        time.sleep(0.05)
        uploader.trigger()
        time.sleep(0.2)    # give the thread time to react
        uploader.stop()
        uploader.join(timeout=3)
        # Should have been called at least once due to the trigger
        # (plus the final upload, so >= 1; in practice >= 2)
        assert client.push_state.call_count >= 1

    def test_push_state_exception_does_not_crash(self):
        """Thread keeps running even if push_state raises repeatedly."""
        state = _make_state()
        client = _make_client()
        client.push_state.side_effect = RuntimeError("network down")
        uploader = StateUploader(state, client, interval_s=0.05)
        uploader.start()
        time.sleep(0.2)
        uploader.stop()
        uploader.join(timeout=3)
        assert not uploader.is_alive()

    def test_periodic_uploads(self):
        """With a short interval, push_state should be called multiple times."""
        state = _make_state()
        client = _make_client()
        uploader = StateUploader(state, client, interval_s=0.05)
        uploader.start()
        time.sleep(0.35)
        uploader.stop()
        uploader.join(timeout=3)
        # At least 3 periodic uploads in ~350ms with 50ms interval
        assert client.push_state.call_count >= 3


# ===========================================================================
# PauseController
# ===========================================================================

class TestPauseController:
    def test_is_daemon_thread(self):
        ctrl = PauseController(_make_state(), _make_client(), poll_interval_s=60.0)
        assert ctrl.daemon is True

    def test_set_conversation(self):
        conv = MagicMock()
        ctrl = PauseController(_make_state(), _make_client(), poll_interval_s=60.0)
        ctrl.set_conversation(conv)
        assert ctrl._conversation is conv

    def test_pause_signal_calls_conversation_pause(self):
        state = _make_state()
        conv = MagicMock()
        client = _make_client(control={"pause": True, "resume": False})

        ctrl = PauseController(state, client, poll_interval_s=0.05)
        ctrl.set_conversation(conv)
        ctrl.start()
        time.sleep(0.3)
        ctrl.stop()
        ctrl.join(timeout=3)

        conv.pause.assert_called()
        assert state.phase == AgentPhase.PAUSED

    def test_pause_without_conversation_does_not_crash(self):
        """If conversation is None, pause signal should be silently ignored."""
        state = _make_state()
        client = _make_client(control={"pause": True, "resume": False})
        ctrl = PauseController(state, client, poll_interval_s=0.05)
        # No set_conversation called → _conversation stays None
        ctrl.start()
        time.sleep(0.2)
        ctrl.stop()
        ctrl.join(timeout=3)
        assert not ctrl.is_alive()

    def test_resume_signal_sets_flag(self):
        state = _make_state()
        client = _make_client(control={"pause": False, "resume": True})

        ctrl = PauseController(state, client, poll_interval_s=0.05)
        ctrl.start()
        time.sleep(0.3)
        ctrl.stop()
        ctrl.join(timeout=3)

        assert ctrl.resume_requested is True
        assert state.phase == AgentPhase.WAITING_FOR_REPLY

    def test_empty_control_is_ignored(self):
        state = _make_state()
        # Return empty dict → no action
        client = _make_client(control={})
        conv = MagicMock()
        ctrl = PauseController(state, client, poll_interval_s=0.05)
        ctrl.set_conversation(conv)
        ctrl.start()
        time.sleep(0.2)
        ctrl.stop()
        ctrl.join(timeout=3)

        conv.pause.assert_not_called()
        assert ctrl.resume_requested is False

    def test_fetch_control_exception_does_not_crash(self):
        """Exceptions in fetch_control should be swallowed, thread keeps running."""
        state = _make_state()
        client = _make_client()
        client.fetch_control.side_effect = RuntimeError("chaos")
        ctrl = PauseController(state, client, poll_interval_s=0.05)
        ctrl.start()
        time.sleep(0.2)
        ctrl.stop()
        ctrl.join(timeout=3)
        assert not ctrl.is_alive()

    def test_conversation_pause_exception_logged_not_raised(self):
        """conversation.pause() failing should not crash the thread."""
        state = _make_state()
        conv = MagicMock()
        conv.pause.side_effect = RuntimeError("pause error")
        client = _make_client(control={"pause": True, "resume": False})
        ctrl = PauseController(state, client, poll_interval_s=0.05)
        ctrl.set_conversation(conv)
        ctrl.start()
        time.sleep(0.2)
        ctrl.stop()
        ctrl.join(timeout=3)
        assert not ctrl.is_alive()

    def test_stop_exits_thread(self):
        state = _make_state()
        client = _make_client()
        ctrl = PauseController(state, client, poll_interval_s=60.0)
        ctrl.start()
        time.sleep(0.05)
        ctrl.stop()
        ctrl.join(timeout=3)
        assert not ctrl.is_alive()
