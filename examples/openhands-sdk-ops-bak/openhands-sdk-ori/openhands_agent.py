"""
OpenHands agent — rollout function for sandbox execution.

This module defines ``rollout(task, config)`` which is the entry point
called by ``worker_server.py`` inside a Docker sandbox. It:

1. Creates a per-rollout workspace with the task instruction
2. Optionally clones a git repository into the workspace
3. Invokes OpenHands headless with ``runtime="docker"`` so that OpenHands
   manages its own isolated runtime container for code execution
4. Evaluates the result via pytest or keyword heuristics
5. Returns reward as an rllm trajectory dict

The function does NOT import rllm.sdk.session — session tracking is
handled externally by worker_server.py via the metadata slug mechanism.

OpenHands Docker isolation
--------------------------
``_run_openhands`` sets ``runtime="docker"`` in the OpenHands ``AppConfig``.
This tells OpenHands to spawn a **per-task Docker container** (using the
runtime image) as the code-execution sandbox.  The OpenHands Python process
itself runs inside the outer rllm sandbox container  — meaning the rollout
environment has **two layers of Docker isolation**:

  Host  →  rllm sandbox (DockerSandbox)  →  OpenHands app (in-process)
                                         →  OpenHands runtime container (Docker)

For OpenHands to spawn the runtime container, the outer rllm sandbox
container must have access to the Docker daemon, which ``SandboxOrchestrator``
achieves by mounting ``/var/run/docker.sock`` (controlled via
``SandboxConfig.extra["docker_volumes"]``).

Environment Variables
---------------------
OPENHANDS_RUNTIME_IMAGE : str
    Image used by OpenHands as the code-execution sandbox.
    Default: ``docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik``
OPENHANDS_MODEL_NAME : str, default ``"openhands-model"``
    Model name as exposed by the LiteLLM proxy.
OPENHANDS_MAX_ITERATIONS : int, default 30
    Maximum agent iterations per episode.
OPENHANDS_SANDBOX_USER_ID : int, default 1000
    UID passed to the OpenHands sandbox (``SandboxConfig.user_id``).
DOCKER_HOST : str, optional
    Override Docker daemon socket (e.g. ``unix:///run/docker.sock``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
_SANDBOX_USER_ID = int(os.environ.get("OPENHANDS_SANDBOX_USER_ID", "1000"))
_DEFAULT_RUNTIME_IMAGE = (
    "docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik"
)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _default_reward(task: dict[str, Any], workspace_dir: str, agent_output: str) -> float:
    """Evaluate the agent's result and compute a scalar reward.

    Tries to run ``pytest`` inside the workspace; falls back to checking
    whether the agent explicitly reported success. Returns 1.0 on success,
    0.0 otherwise.

    Args:
        task: The raw task dict from the dataset.
        workspace_dir: Path to the sandbox directory the agent operated in.
        agent_output: Final text output emitted by the agent.

    Returns:
        Scalar reward in [0.0, 1.0].
    """
    # --- attempt pytest ---
    test_target = task.get("test_file") or task.get("test_dir")
    if test_target:
        test_path = os.path.join(workspace_dir, test_target)
        if os.path.exists(test_path):
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_path, "-x", "-q", "--tb=no"],
                    capture_output=True,
                    cwd=workspace_dir,
                    timeout=60,
                )
                return 1.0 if result.returncode == 0 else 0.0
            except subprocess.TimeoutExpired:
                logger.warning("pytest timed out in %s", workspace_dir)
                return 0.0

    # --- fallback: keyword heuristic in agent output ---
    success_keywords = task.get("success_keywords", [])
    if success_keywords:
        agent_output_lower = agent_output.lower()
        if any(kw.lower() in agent_output_lower for kw in success_keywords):
            return 1.0

    # If no evaluation criteria are provided, default to 0.0 so training
    # doesn't trivially converge; replace with a domain-specific reward.
    logger.warning(
        "No evaluation criteria found in task (expected 'test_file', "
        "'test_dir', or 'success_keywords'). Reward defaults to 0.0. "
        "Set these fields in your dataset or provide a custom reward function."
    )
    return 0.0


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def _setup_workspace(task: dict[str, Any]) -> str:
    """Create a per-rollout workspace with task files.

    If the task contains a ``repo_url``, it will be cloned into the workspace.
    Otherwise, an empty workspace with ``INSTRUCTIONS.md`` is created.
    """
    workspace = tempfile.mkdtemp(prefix=f"openhands-{uuid.uuid4().hex[:8]}-")

    # Optionally clone a repo into the workspace
    repo_url = task.get("repo_url")
    if repo_url:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, workspace],
            check=True,
            capture_output=True,
            timeout=120,
        )

    # Write task instruction
    instruction = task.get("instruction", "Fix the bug in the repository.")
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(f"# Task\n\n{instruction}\n")

    return workspace


# ---------------------------------------------------------------------------
# OpenHands invocation (with Docker runtime)
# ---------------------------------------------------------------------------

def _run_openhands(workspace: str, base_url: str, instruction: str) -> str:
    """Run OpenHands headless with its own Docker runtime container.

    OpenHands is invoked in-process (``openhands-ai`` must be installed),
    but with ``runtime="docker"`` so that all code execution happens inside
    an isolated OpenHands runtime container — not directly in the calling
    process.

    The outer rllm sandbox container must have ``/var/run/docker.sock``
    mounted (or ``DOCKER_HOST`` set) so that OpenHands can start its
    runtime container.

    Args:
        workspace: Path to the per-rollout working directory.
        base_url: Proxied LiteLLM URL (contains metadata slug for session
                  tracking).  OpenHands LLM calls are routed here.
        instruction: Task instruction text.

    Returns:
        Agent's final output text.
    """
    try:
        from openhands.core.config import AppConfig, LLMConfig  # type: ignore[import]
        from openhands.core.config import SandboxConfig as OHSandboxConfig  # type: ignore[import]
        from openhands.headless import run_openhands  # type: ignore[import]
    except ImportError:
        logger.warning(
            "OpenHands not installed. Falling back to no-op. "
            "Install with: pip install openhands-ai"
        )
        return "OpenHands not available"

    runtime_image = os.environ.get("OPENHANDS_RUNTIME_IMAGE", _DEFAULT_RUNTIME_IMAGE)
    model_name = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")

    # Optionally override the Docker socket path (e.g. when running inside
    # a container that has /var/run/docker.sock mounted).
    docker_host = os.environ.get("DOCKER_HOST")

    llm_cfg = LLMConfig(
        model=model_name,
        base_url=base_url,
        api_key="EMPTY",
    )

    # OpenHands SandboxConfig (distinct from rllm's SandboxConfig) controls
    # the runtime container that OpenHands uses for code execution.
    oh_sandbox_cfg = OHSandboxConfig(
        runtime_container_image=runtime_image,
        user_id=_SANDBOX_USER_ID,
        **({"docker_url": docker_host} if docker_host else {}),
    )

    app_cfg = AppConfig(
        workspace_base=workspace,
        max_iterations=_MAX_ITERATIONS,
        headless_mode=True,
        # "docker": OpenHands spawns an isolated runtime container per task.
        # This requires /var/run/docker.sock (or DOCKER_HOST) to be accessible.
        runtime="docker",
        sandbox=oh_sandbox_cfg,
    )
    app_cfg.set_llm_config(llm_cfg)

    logger.info(
        "[openhands] Starting agent (runtime=docker, image=%s, workspace=%s)",
        runtime_image,
        workspace,
    )
    result = run_openhands(
        task_str=instruction,
        config=app_cfg,
    )
    output_text = getattr(result, "final_output", "") or str(result)
    return output_text


# ---------------------------------------------------------------------------
# Sandbox rollout entry point
# ---------------------------------------------------------------------------

def rollout(task: dict, config: dict) -> list[dict]:
    """Sandbox rollout entry point.

    Called by ``worker_server.py`` with the proxied base_url already
    containing the metadata slug for session tracking.

    Args:
        task: Task dict from the dataset (instruction, repo_url, test_file, ...)
        config: Agent config dict with ``base_url`` and ``session_uid``
                (injected by worker_server.py).

    Returns:
        List of trajectory dicts (single trajectory for OpenHands agent).
    """
    base_url = config.get("base_url", "http://localhost:4000/v1")
    instruction = task.get("instruction", "Fix the bug in the repository.")

    workspace = _setup_workspace(task)

    try:
        # Run OpenHands agent with its own Docker runtime container
        agent_output = _run_openhands(
            workspace=workspace,
            base_url=base_url,
            instruction=instruction,
        )

        # Evaluate the result and compute reward
        reward = _default_reward(task, workspace, agent_output)
        logger.info(
            "[openhands] reward=%.2f | instruction=%s",
            reward,
            instruction[:80],
        )

    except Exception:
        logger.exception("[openhands] Rollout failed")
        reward = 0.0

    finally:
        # Clean up workspace (the OpenHands runtime container is already
        # cleaned up by the OpenHands library itself)
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

    # Return trajectory in the format expected by worker_server.py
    return [
        {
            "name": "openhands",
            "steps": [],  # steps are tracked via LiteLLM proxy traces
            "reward": reward,
        }
    ]