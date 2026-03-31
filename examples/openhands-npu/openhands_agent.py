"""
OpenHands agent — rollout function for NPU operator generation (sandbox mode).

This module defines ``rollout(task, config)`` which is the entry point
called by ``worker_server.py`` inside a Docker sandbox. It:

1. Creates a per-rollout workspace with the NPU operator task instruction
2. Writes a mock profiling script (profile_wrapper.sh) into the workspace
3. Invokes OpenHands via official SDK conversation APIs
4. Evaluates the result by checking kernel.cpp existence and mock profiling
5. Returns reward as an rllm trajectory

The function does NOT import rllm.sdk.session — session tracking is
handled externally by worker_server.py via the metadata slug mechanism.

OpenHands SDK invocation
------------------------
``_run_openhands`` uses the stable public SDK flow from the official examples:
``LLM -> get_default_agent -> Conversation -> send_message -> run``.
This avoids direct dependency on internal ``openhands.core.*`` modules.

Environment Variables
---------------------
OPENHANDS_MODEL_NAME : str, default ``"openhands-model"``
    Model name as exposed by the LiteLLM proxy.
OPENHANDS_API_KEY : str, default ``"EMPTY"``
    API key passed to the OpenHands SDK ``LLM`` constructor.
OPENHANDS_MAX_ITERATIONS : int, default 30
    Maximum agent iterations per episode.
DOCKER_HOST : str, optional
    Override Docker daemon socket (e.g. ``unix:///run/docker.sock``).
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
import uuid
from typing import Any

from rllm.agents.agent import Trajectory

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))

# ---------------------------------------------------------------------------
# Mock profiler script content
# This script simulates 'msopgen compile + msprof profiling' inside the
# sandbox.  In real deployment, replace with actual Ascend toolchain calls.
# ---------------------------------------------------------------------------

_MOCK_PROFILE_SCRIPT = r"""#!/bin/bash
# Mock NPU Profiler — simulates compilation + profiling for kernel development.
# Usage: bash profile_wrapper.sh <source.cpp>
set -e

SOURCE_FILE="$1"
WORKSPACE_DIR="$(pwd)"
OUTPUT_JSON="${WORKSPACE_DIR}/profiling_results.json"

if [ -z "$SOURCE_FILE" ]; then
    echo '{"success": false, "bandwidth_gbps": 0.0, "error": "No source file"}' > "$OUTPUT_JSON"
    exit 1
fi

if [ ! -f "$SOURCE_FILE" ]; then
    echo '{"success": false, "bandwidth_gbps": 0.0, "error": "File not found"}' > "$OUTPUT_JSON"
    exit 1
fi

# Mock: check basic syntax with g++ if available, otherwise just check file exists
if command -v g++ &> /dev/null; then
    if ! g++ -fsyntax-only -std=c++11 "$SOURCE_FILE" 2>/dev/null; then
        echo '{"success": false, "bandwidth_gbps": 0.0, "error": "Compilation failed"}' > "$OUTPUT_JSON"
        exit 1
    fi
fi

# Mock profiling result: random bandwidth between 100-600 GB/s
MOCK_BW=$(( RANDOM % 500 + 100 ))
cat << EOF > "$OUTPUT_JSON"
{
  "success": true,
  "bandwidth_gbps": ${MOCK_BW}.0,
  "execution_time_ms": 1.25,
  "error": null
}
EOF

echo "[mock_profiler] Compilation OK. Bandwidth=${MOCK_BW} GB/s. Results written to $OUTPUT_JSON"
exit 0
"""


# ---------------------------------------------------------------------------
# NPU operator task instruction template
# ---------------------------------------------------------------------------

_NPU_INSTRUCTION_TEMPLATE = """# NPU Operator Task

{instruction}

## Requirements

1. Write the kernel implementation in C++ and save it as `kernel.cpp` in the current directory.
2. After writing the kernel, run the profiler to verify:
   ```bash
   bash /workspace/tools/profile_wrapper.sh ./kernel.cpp
   ```
3. Check the output file `profiling_results.json` — it should show `"success": true`.
4. If compilation fails, examine the error, fix `kernel.cpp`, and re-run the profiler.
5. Once `profiling_results.json` shows success, your task is complete.
"""


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _default_reward(task: dict[str, Any], workspace_dir: str, agent_output: str) -> float:
    """Evaluate the agent's result and compute a scalar reward.

    For NPU operator generation, we check:
    1. Whether kernel.cpp was created in the workspace
    2. Whether profiling_results.json exists and shows success
    3. Mock bandwidth score for graded reward

    Args:
        task: The raw task dict from the dataset.
        workspace_dir: Path to the sandbox directory the agent operated in.
        agent_output: Final text output emitted by the agent.

    Returns:
        Scalar reward in [0.0, 1.0].
    """
    import json

    kernel_path = os.path.join(workspace_dir, "kernel.cpp")
    profiler_path = os.path.join(workspace_dir, "profiling_results.json")

    # Check if kernel.cpp was generated at all
    if not os.path.exists(kernel_path):
        logger.info("[reward] kernel.cpp not found — reward=0.0")
        return 0.0

    # Check if profiler was run and produced results
    if not os.path.exists(profiler_path):
        # Kernel exists but profiler was never invoked — partial credit
        logger.info("[reward] kernel.cpp exists but no profiling — reward=0.2")
        return 0.2

    try:
        with open(profiler_path, "r") as f:
            perf: dict[str, object] = json.load(f)

        is_success = bool(perf.get("success", False))
        if not is_success:
            # Profiler ran but compilation/logic failed
            logger.info("[reward] Profiling failed — reward=0.2")
            return 0.2

        # Success! Compute graded reward based on mock bandwidth
        bandwidth = float(perf.get("bandwidth_gbps", 0.0))  # type: ignore[arg-type]
        # Theoretical peak for mock scoring (adjust for real hardware)
        theoretical_peak = 1200.0
        optimization_ratio = min(bandwidth / theoretical_peak, 1.0)
        reward = 0.5 + 0.5 * optimization_ratio
        logger.info("[reward] Success! bandwidth=%.1f GB/s — reward=%.3f", bandwidth, reward)
        return reward

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("[reward] Failed to parse profiling_results.json: %s", exc)
        return 0.1


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def _setup_workspace(task: dict[str, Any]) -> str:
    """Create a per-rollout workspace with task files and tools.

    Creates:
    - INSTRUCTIONS.md with the NPU operator task description
    - /workspace/tools/profile_wrapper.sh mock profiling tool
    """
    workspace = tempfile.mkdtemp(prefix=f"openhands-npu-{uuid.uuid4().hex[:8]}-")

    # Write task instruction
    instruction = task.get("instruction", "Implement a simple vector_add NPU kernel.")
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(_NPU_INSTRUCTION_TEMPLATE.format(instruction=instruction))

    # Write mock profiler tool into workspace tools directory
    tools_dir = os.path.join(workspace, "tools")
    os.makedirs(tools_dir, exist_ok=True)
    profiler_path = os.path.join(tools_dir, "profile_wrapper.sh")
    with open(profiler_path, "w") as f:
        f.write(_MOCK_PROFILE_SCRIPT)
    # Make executable
    os.chmod(profiler_path, os.stat(profiler_path).st_mode | stat.S_IEXEC)

    return workspace


# ---------------------------------------------------------------------------
# OpenHands invocation (SDK conversation)
# ---------------------------------------------------------------------------

def _run_openhands(workspace: str, base_url: str, instruction: str) -> str:
    """Run OpenHands with the official SDK interface.

    Args:
        workspace: Path to the per-rollout working directory.
        base_url: Proxied LiteLLM URL.
        instruction: Task instruction text.

    Returns:
        Agent's final output text.
    """
    try:
        from openhands.sdk import Conversation, LLM, Workspace
        from openhands.sdk.event import MessageEvent
        from openhands.sdk.llm import content_to_str
        from openhands.tools.preset.default import get_default_agent
    except ImportError as e:
        logger.warning(
            "OpenHands SDK or required dependencies not installed: %s. Falling back to no-op.",
            e,
        )
        return "OpenHands not available"

    model_name = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")
    api_key = os.environ.get("OPENHANDS_API_KEY", "EMPTY")

    llm = LLM(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    agent = get_default_agent(
        llm=llm,
        cli_mode=True,
    )
    workspace_obj = Workspace(working_dir=workspace)

    logger.info(
        "[openhands] Starting agent with SDK conversation (workspace=%s, max_iterations=%d)",
        workspace,
        _MAX_ITERATIONS,
    )

    conversation = None
    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace_obj,
            max_iteration_per_run=_MAX_ITERATIONS,
        )
        conversation.send_message(instruction)
        conversation.run()

        # Extract the latest agent text message as the final output.
        for event in reversed(list(conversation.state.events)):
            if isinstance(event, MessageEvent) and event.source == "agent":
                text = "".join(content_to_str(event.llm_message.content)).strip()
                if text:
                    return text
        return "Task completed"
    except Exception as e:
        logger.exception("[openhands] Error during SDK conversation execution")
        return f"Error: {str(e)}"
    finally:
        if conversation is not None:
            try:
                conversation.close()
            except Exception:
                logger.debug(
                    "[openhands] Failed to close conversation cleanly",
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# Sandbox rollout entry point
# ---------------------------------------------------------------------------

def rollout(
    task: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Trajectory]:
    """Sandbox rollout entry point for NPU operator generation."""
    if task is None:
        task = dict(kwargs)
    if config is None:
        config = kwargs.get("config", {}) or {}

    base_url: str = config.get("base_url", "http://127.0.0.1:4001/v1")
    instruction: str = (
        kwargs.get("instruction")
        or task.get("instruction", "Implement a simple vector_add NPU kernel.")
    )
    task["instruction"] = instruction

    workspace = _setup_workspace(task)

    try:
        agent_output = _run_openhands(
            workspace=workspace,
            base_url=base_url,
            instruction=instruction,
        )
        reward = _default_reward(task, workspace, agent_output)
        logger.info(
            "[openhands-npu] reward=%.2f | instruction=%s",
            reward,
            instruction[:80],
        )
    except Exception:
        logger.exception("[openhands-npu] Rollout failed")
        reward = 0.0
    finally:
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

    return [
        Trajectory(
            name="openhands-npu",
            steps=[],
            reward=reward,
        )
    ]
