"""
uploader.py — Background thread that periodically pushes RunState to the
observer gateway.

Design:
  - Runs in a daemon thread (auto-exits when the main thread exits).
  - Posts every ``interval_s`` seconds, or immediately when
    ``trigger()`` is called (e.g., right after a significant event).
  - Catches all exceptions so it never silently kills the process.
"""
from __future__ import annotations

import logging
import threading
import time

from rllm_entrypoint.api_client import ObserverClient
from rllm_entrypoint.state import RunState

logger = logging.getLogger(__name__)


class StateUploader(threading.Thread):
    """Daemon thread that pushes RunState on a fixed interval."""

    def __init__(
        self,
        run_state: RunState,
        client: ObserverClient,
        interval_s: float = 5.0,
    ) -> None:
        super().__init__(name="rllm-state-uploader", daemon=True)
        self._state = run_state
        self._client = client
        self._interval = interval_s
        self._stop_event = threading.Event()
        self._trigger_event = threading.Event()
        self._last_event_count = 0

    def trigger(self) -> None:
        """Wake the uploader for an immediate upload."""
        self._trigger_event.set()

    def stop(self) -> None:
        """Signal the uploader to stop after current iteration."""
        self._stop_event.set()
        self._trigger_event.set()  # unblock any wait

    def run(self) -> None:
        logger.info("[uploader] Started (interval=%.1fs)", self._interval)
        while not self._stop_event.is_set():
            # Wait for timeout OR an explicit trigger.
            self._trigger_event.wait(timeout=self._interval)
            self._trigger_event.clear()

            if self._stop_event.is_set():
                break

            try:
                payload = self._state.to_json()
                ok = self._client.push_state(payload)
                if ok:
                    logger.debug("[uploader] State pushed (%d bytes)", len(payload))
                else:
                    logger.debug("[uploader] push_state returned False (gateway unavailable?)")
            except Exception:
                logger.exception("[uploader] Unexpected error during state push")

        # Final upload on exit
        try:
            self._client.push_state(self._state.to_json())
            logger.info("[uploader] Final state pushed. Thread stopping.")
        except Exception:
            logger.debug("[uploader] Final push failed")
