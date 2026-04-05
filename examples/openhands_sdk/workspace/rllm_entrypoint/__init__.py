"""
rllm_entrypoint — container-side sub-project for OpenHands rllm training.

Package structure:
    config.py       — Environment variable parsing
    state.py        — RunState dataclass (shared, thread-safe)
    events.py       — OpenHands event hook + lifecycle event push
    api_client.py   — REST API client (push events, query pause)
    pause_ctrl.py   — Background pause-poller thread
    runner.py       — Main conversation runner logic

Note: uploader.py (background state-uploader) is deprecated and no longer
used. Events are now pushed directly in the event callback (event-driven
model), with a lightweight heartbeat timer in runner.py for liveness.
"""
