#!/usr/bin/env python3
"""
rllm OpenHands entrypoint — runs inside the OpenHands Docker container.

This file is the container ENTRYPOINT (thin shell). All logic lives in
the ``rllm_entrypoint`` package located next to this file.

Configuration via environment variables (set by openhands_agent.py):

    LLM_BASE_URL            Proxied rllm LiteLLM URL with embedded metadata slug.
    LLM_API_KEY             API key for the proxy (default: EMPTY)
    LLM_MODEL               Model name on the LiteLLM proxy
    TASK_INSTRUCTION        Task text; falls back to reading INSTRUCTIONS.md
    WORKSPACE_BASE          Workspace directory (default: /opt/workspace)
    MAX_ITERATIONS          Max agent iterations (default: 30)
    NPU_OPERATOR_TASK       1 = operator / kernel task prompt
    OPERATOR_BACKEND        triton | ascendc (for task_scope hint)

Observability & control (all optional):

    OBSERVER_API_URL        Base URL of external observer REST gateway.
                            If unset, the container runs standalone (no upload).
                            e.g. http://host.docker.internal:8765
    OBSERVER_SESSION_ID     Unique ID for this session (default: random UUID).
    OBSERVER_SESSION_LABEL  Human-readable label (default: first 12 chars of ID).
    OBSERVER_UPLOAD_INTERVAL    State upload period in seconds (default: 5.0).
    OBSERVER_PAUSE_POLL_INTERVAL Pause poll period in seconds (default: 2.0).
    OBSERVER_EXTRA_METADATA     JSON string with arbitrary metadata to include.

Exit codes:
    0   Completed successfully
    1   Fatal error
"""
from __future__ import annotations

import logging
import sys
import os

# os.environ["LLM_BASE_URL"] = "https://api.silra.cn/v1"
# os.environ["LLM_API_KEY"] = "sk-ZkzodY0fm4YmAqEuplsBLcLT9mX44wLlLpBXEhgnHOPGJIFn"
# os.environ["LLM_MODEL"] = "openai/glm-4.7"
# os.environ["TASK_INSTRUCTION"] = "Write an add operator in Ascend-triton."
# os.environ["WORKSPACE_BASE"] = "/data/home/3120235672/huawei-intern/code_agent/rllm/examples/openhands_sdk/workspace/agent_workdir"
# os.environ["MAX_ITERATIONS"] = "10"
# os.environ["NPU_OPERATOR_TASK"] = "1"
# os.environ["OPERATOR_BACKEND"] = "triton"

# os.environ["OBSERVER_API_URL"] = "http://127.0.0.1:18858"


# ---------------------------------------------------------------------------
# Logging setup (before importing anything else so SDK loggers inherit this)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

# Ensure the package directory is importable when running the script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rllm_entrypoint.runner import run
    sys.exit(run())
