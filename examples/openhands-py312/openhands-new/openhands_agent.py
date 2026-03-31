"""
OpenHands agent — rollout function for rllm training.

Architecture:
    rllm training process
    └─ rollout(task, config)
         ├─ _build_proxied_base_url(proxy_url, metadata)  ← metadata slug
         └─ docker run rllm-openhands  (workspace/Dockerfile)
              └─ workspace/entrypoint.py
                   └─ OpenHands SDK (LLM, Agent, Conversation, Tool)
                        └─ LLM calls → LLM_BASE_URL (proxied) → rllm proxy

No openhands Python library is imported in this file. OpenHands runs entirely
inside its own container (built from workspace/Dockerfile). The proxied base
URL with embedded metadata slug is passed as LLM_BASE_URL into the container
so the rllm proxy can attribute all LLM calls to the correct session.

Environment Variables:
    OPENHANDS_IMAGE             : Custom rllm-openhands image
                                  (built from workspace/Dockerfile)
                                  Default: rllm-openhands
    OPENHANDS_SANDBOX_IMAGE     : Runtime image used by OpenHands internally
                                  Default: docker.all-hands.dev/all-hands-ai/runtime:0.28-nikolaik
    OPENHANDS_MODEL_NAME        : Model name on the LiteLLM proxy
                                  Default: openai/openhands-model
    OPENHANDS_MAX_ITERATIONS    : Max agent iterations, default 30
    OPENHANDS_CONTAINER_TIMEOUT : Seconds to wait for container, default 600
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from urllib.parse import urlparse, urlunparse
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

# Custom image built from workspace/Dockerfile (based on official OpenHands
# image but with workspace/entrypoint.py pre-installed as ENTRYPOINT).
_OPENHANDS_IMAGE = os.environ.get("OPENHANDS_IMAGE", "rllm-openhands")
_MODEL_NAME = os.environ.get("OPENHANDS_MODEL_NAME", "openai/openhands-model")
_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
_CONTAINER_TIMEOUT = int(os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# Inlined metadata slug helpers
# (mirrors rllm.sdk.proxy.metadata_slug — self-contained, no rllm import)
# ---------------------------------------------------------------------------

_SLUG_PREFIX = "rllm1:"


def _encode_metadata_slug(metadata: dict) -> str:
    body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=")
    return f"{_SLUG_PREFIX}{encoded.decode('ascii')}"


def _build_proxied_base_url(base_url: str, metadata: dict) -> str:
    slug = _encode_metadata_slug(metadata)
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    has_v1 = path.endswith("/v1")
    if has_v1:
        path = path[:-3]
    new_path = f"{path}/meta/{slug}"
    if has_v1:
        new_path += "/v1"
    if not new_path.startswith("/"):
        new_path = "/" + new_path
    rebuilt = parsed._replace(path=new_path)
    return urlunparse(rebuilt)


def _to_container_url(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal so the URL
    is reachable from inside an OpenHands Docker container."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1"):
        netloc = parsed.netloc.replace(host, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _setup_workspace(task: dict[str, Any]) -> str:
    workspace = tempfile.mkdtemp(prefix=f"openhands-{uuid.uuid4().hex[:8]}-")
    repo_url = task.get("repo_url")
    if repo_url:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, workspace],
            check=True, capture_output=True, timeout=120,
        )
    instruction = task.get("instruction", "Fix the bug in the repository.")
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(f"# Task\n\n{instruction}\n")
    return workspace


# ---------------------------------------------------------------------------
# Reward evaluation
# ---------------------------------------------------------------------------

def _default_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    test_target = task.get("test_file") or task.get("test_dir")
    if test_target:
        test_path = os.path.join(workspace_dir, test_target)
        if os.path.exists(test_path):
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_path, "-x", "-q", "--tb=no"],
                    capture_output=True, cwd=workspace_dir, timeout=60,
                )
                return 1.0 if result.returncode == 0 else 0.0
            except subprocess.TimeoutExpired:
                return 0.0

    success_keywords = task.get("success_keywords", [])
    if success_keywords:
        if any(kw.lower() in output.lower() for kw in success_keywords):
            return 1.0

    logger.warning("[openhands] No evaluation criteria in task; reward=0.0")
    return 0.0


# ---------------------------------------------------------------------------
# OpenHands container launch
# ---------------------------------------------------------------------------

def _run_openhands_container(
    workspace: str,
    proxied_url: str,
    instruction: str,
) -> str:
    """Start an OpenHands headless container, wait for completion, return logs.

    Args:
        workspace:    Host-side workspace directory (mounted into container).
        proxied_url:  LiteLLM proxy URL with embedded rllm metadata slug.
                      Passed as LLM_BASE_URL so rllm can track all LLM calls.
        instruction:  Task instruction (also written to INSTRUCTIONS.md).
    """
    container_name = f"rllm-openhands-{uuid.uuid4().hex[:12]}"

    cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,

        # --- LLM routing ---
        # proxied_url carries the rllm metadata slug so every LLM call made
        # by OpenHands SDK (via workspace/entrypoint.py) is attributed to
        # this training session by the rllm LiteLLM proxy.
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL={_MODEL_NAME}",

        # --- Task ---
        # Passed as env var; entrypoint.py reads it and sends to Conversation.
        "-e", f"TASK_INSTRUCTION={instruction}",

        # --- Workspace ---
        # The host workspace dir is mounted; the agent operates directly
        # inside the container — no inner sandbox is created.
        "-e", "WORKSPACE_BASE=/opt/workspace",
        "-v", f"{workspace}:/opt/workspace",

        # --- Agent iterations ---
        "-e", f"MAX_ITERATIONS={_MAX_ITERATIONS}",

        # --- Network ---
        # host.docker.internal resolves to the training host so the agent
        # can reach the LiteLLM proxy.
        "--add-host", "host.docker.internal:host-gateway",

        # Custom image built from workspace/Dockerfile.
        # ENTRYPOINT = workspace/entrypoint.py (new OpenHands SDK).
        # No docker.sock mount needed — no inner sandbox.
        _OPENHANDS_IMAGE,
    ]

    logger.info(
        "[openhands] Launching container %s (image=%s, proxied_url=%s...)",
        container_name, _OPENHANDS_IMAGE, proxied_url[:70],
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_CONTAINER_TIMEOUT,
        )
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if result.returncode != 0:
            logger.warning(
                "[openhands] Container %s exited with code %d",
                container_name, result.returncode,
            )
        return output
    except subprocess.TimeoutExpired:
        logger.error("[openhands] Container %s timed out", container_name)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        return ""
    except Exception:
        logger.exception("[openhands] Failed to run container %s", container_name)
        return ""


# ---------------------------------------------------------------------------
# Rollout entry point
# ---------------------------------------------------------------------------

def rollout(task: dict, config: dict) -> list[dict]:
    """rllm rollout entry point for OpenHands.

    Generates a session-specific metadata slug, builds a proxied LiteLLM URL,
    and launches OpenHands in an isolated Docker container. All LLM calls made
    by OpenHands will carry the metadata slug so the rllm proxy can attribute
    them to this training session.

    Args:
        task:   Task dict (instruction, repo_url, test_file, …).
        config: Config dict from rllm. Expected keys:
                  base_url  — raw LiteLLM proxy URL, e.g.
                              "http://127.0.0.1:4000/v1"

    Returns:
        List with one trajectory dict: {name, steps, reward}.
    """
    # Raw proxy URL — rllm passes this before any slug is applied
    proxy_url = config.get("base_url", "http://127.0.0.1:4000/v1")

    # Generate a unique session identifier for this rollout.
    # Encoding it in the URL lets the rllm proxy associate every OpenHands
    # LLM call with this specific training episode.
    session_uid = str(uuid.uuid4())
    metadata: dict[str, Any] = {
        "session_uids": [session_uid],
        "session_name": f"openhands-{session_uid[:8]}",
    }

    # Build the proxied URL with embedded metadata slug.
    # _build_proxied_base_url inlines the same logic as rllm.sdk.proxy.
    proxied_url = _build_proxied_base_url(proxy_url, metadata)

    # Rewrite localhost/127.0.0.1 → host.docker.internal so the URL is
    # reachable from inside the OpenHands Docker container.
    proxied_url = _to_container_url(proxied_url)

    instruction = task.get("instruction", "Fix the bug in the repository.")
    workspace = _setup_workspace(task)
    reward = 0.0

    try:
        output = _run_openhands_container(workspace, proxied_url, instruction)
        reward = _default_reward(task, workspace, output)
        logger.info(
            "[openhands] session=%s reward=%.2f instruction=%s",
            session_uid[:8], reward, instruction[:80],
        )
    except Exception:
        logger.exception("[openhands] Rollout failed (session=%s)", session_uid[:8])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return [
        {
            "name": "openhands",
            "steps": [],      # LLM call traces are tracked via metadata slug
            "reward": reward,
            "session_uid": session_uid,
        }
    ]
