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
import stat
import subprocess
import tempfile
import uuid
from urllib.parse import urlparse, urlunparse
from typing import Any

from rllm.types import Trajectory
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NPU operator mock (align with openhands-npu bring-up)
# ---------------------------------------------------------------------------

_MOCK_PROFILE_SCRIPT = r"""#!/bin/bash
# Mock NPU profiler — simulates compile + profiling for kernel bring-up.
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
if command -v g++ &> /dev/null; then
    if ! g++ -fsyntax-only -std=c++11 "$SOURCE_FILE" 2>/dev/null; then
        echo '{"success": false, "bandwidth_gbps": 0.0, "error": "Compilation failed"}' > "$OUTPUT_JSON"
        exit 1
    fi
fi
MOCK_BW=$(( RANDOM % 500 + 100 ))
cat << EOF > "$OUTPUT_JSON"
{
  "success": true,
  "bandwidth_gbps": ${MOCK_BW}.0,
  "execution_time_ms": 1.25,
  "error": null
}
EOF
echo "[mock_profiler] OK bandwidth=${MOCK_BW} GB/s -> $OUTPUT_JSON"
exit 0
"""

_NPU_INSTRUCTION_TEMPLATE = """# NPU operator task

{instruction}

## Requirements

1. Write the kernel implementation in C++ and save it as `kernel.cpp` in the workspace root (`/opt/workspace`).
2. Run the mock profiler:
   ```bash
   bash /opt/workspace/tools/profile_wrapper.sh ./kernel.cpp
   ```
3. Check `profiling_results.json` — it should contain `"success": true`.
4. If compilation fails, fix `kernel.cpp` and re-run the profiler.
"""


def _is_npu_operator_task(task: dict[str, Any]) -> bool:
    return task.get("scenario") == "npu_operator" or task.get("task_type") == "npu_operator"


def _setup_npu_operator_workspace(task: dict[str, Any]) -> str:
    workspace = tempfile.mkdtemp(prefix=f"openhands-npu-{uuid.uuid4().hex[:8]}-")
    instruction = task.get("instruction", "Implement a simple vector_add NPU kernel.")
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(_NPU_INSTRUCTION_TEMPLATE.format(instruction=instruction))
    tools_dir = os.path.join(workspace, "tools")
    os.makedirs(tools_dir, exist_ok=True)
    profiler_path = os.path.join(tools_dir, "profile_wrapper.sh")
    with open(profiler_path, "w") as f:
        f.write(_MOCK_PROFILE_SCRIPT)
    os.chmod(profiler_path, os.stat(profiler_path).st_mode | stat.S_IEXEC)
    return workspace


def _npu_operator_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    del task, output  # optional hooks for logging extensions
    kernel_path = os.path.join(workspace_dir, "kernel.cpp")
    profiler_json = os.path.join(workspace_dir, "profiling_results.json")
    if not os.path.exists(kernel_path):
        logger.info("[openhands-npu] kernel.cpp missing -> reward=0.0")
        return 0.0
    if not os.path.exists(profiler_json):
        logger.info("[openhands-npu] kernel present, no profiling -> reward=0.2")
        return 0.2
    try:
        with open(profiler_json) as f:
            perf: dict[str, Any] = json.load(f)
        if not bool(perf.get("success", False)):
            logger.info("[openhands-npu] profiling failed -> reward=0.2")
            return 0.2
        bandwidth = float(perf.get("bandwidth_gbps", 0.0))
        theoretical_peak = 1200.0
        optimization_ratio = min(bandwidth / theoretical_peak, 1.0)
        reward = 0.5 + 0.5 * optimization_ratio
        logger.info("[openhands-npu] success bandwidth=%.1f GB/s reward=%.3f", bandwidth, reward)
        return reward
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("[openhands-npu] bad profiling_results.json: %s", exc)
        return 0.1


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


# def _encode_metadata_slug_fallback(metadata: dict) -> str:
#     body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
#     encoded = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=")
#     return f"{_SLUG_PREFIX}{encoded.decode('ascii')}"


# def _build_proxied_base_url_fallback(base_url: str, metadata: dict) -> str:
#     slug = _encode_metadata_slug_fallback(metadata)
#     parsed = urlparse(base_url)
#     path = parsed.path.rstrip("/")
#     has_v1 = path.endswith("/v1")
#     if has_v1:
#         path = path[:-3]
#     new_path = f"{path}/meta/{slug}"
#     if has_v1:
#         new_path += "/v1"
#     if not new_path.startswith("/"):
#         new_path = "/" + new_path
#     rebuilt = parsed._replace(path=new_path)
#     return urlunparse(rebuilt)


def _trace_label_from_routing_metadata(metadata: dict[str, Any]) -> str:
    uids = metadata.get("session_uids") or []
    if uids:
        return str(uids[-1])[:18]
    name = metadata.get("session_name")
    return str(name or "none")[:18]


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
    if _is_npu_operator_task(task):
        return _setup_npu_operator_workspace(task)
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

# def _default_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
#     if _is_npu_operator_task(task):
#         return _npu_operator_reward(task, workspace_dir, output)
#     test_target = task.get("test_file") or task.get("test_dir")
#     if test_target:
#         test_path = os.path.join(workspace_dir, test_target)
#         if os.path.exists(test_path):
#             try:
#                 result = subprocess.run(
#                     ["python", "-m", "pytest", test_path, "-x", "-q", "--tb=no"],
#                     capture_output=True, cwd=workspace_dir, timeout=60,
#                 )
#                 return 1.0 if result.returncode == 0 else 0.0
#             except subprocess.TimeoutExpired:
#                 return 0.0

#     success_keywords = task.get("success_keywords", [])
#     if success_keywords:
#         if any(kw.lower() in output.lower() for kw in success_keywords):
#             return 1.0

#     logger.warning("[openhands] No evaluation criteria in task; reward=0.0")
#     return 0.0

import random  # 必须引入 random 库
from typing import Any
import os
import subprocess

def _default_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    # 如果你希望整个函数在任何情况下都只返回随机值，
    # 可以直接在函数开头返回：return random.random()
    
    # if _is_npu_operator_task(task):
    #     return _npu_operator_reward(task, workspace_dir, output)

    # test_target = task.get("test_file") or task.get("test_dir")
    # if test_target:
    #     test_path = os.path.join(workspace_dir, test_target)
    #     if os.path.exists(test_path):
    #         try:
    #             subprocess.run(
    #                 ["python", "-m", "pytest", test_path, "-x", "-q", "--tb=no"],
    #                 capture_output=True, cwd=workspace_dir, timeout=60,
    #             )
    #             # 原本这里成功返回 1.0，失败返回 0.0
    #             # 现在统一返回 0 到 1 之间的随机浮点数
    #             return random.random() 
    #         except subprocess.TimeoutExpired:
    #             return random.random()

    # success_keywords = task.get("success_keywords", [])
    # if success_keywords:
    #     if any(kw.lower() in output.lower() for kw in success_keywords):
    #         return random.random()

    # 原本兜底返回 0.0，现在也改为随机
    return random.random()


def _routing_metadata_for_rollout(
    explicit_uids: list[str] | None,
    explicit_name: str | None,
) -> dict[str, Any]:
    """Slug payload: ``assemble_routing_metadata`` + optional ``_rllm_proxy_session_*`` overrides."""
    extra: dict[str, Any] = {}
    if explicit_uids is not None:
        extra["session_uids"] = list(explicit_uids)
    if explicit_name is not None:
        extra["session_name"] = explicit_name
    return assemble_routing_metadata(extra=extra if extra else None)


# ---------------------------------------------------------------------------
# OpenHands container launch
# ---------------------------------------------------------------------------

def _run_openhands_container(
    workspace: str,
    proxied_url: str,
    instruction: str,
    *,
    npu_operator: bool = False,
) -> str:
    """Start an OpenHands headless container, wait for completion, return logs.

    Args:
        workspace:    Host-side workspace directory (mounted into container).
        proxied_url:  LiteLLM proxy URL with embedded rllm metadata slug.
                      Passed as LLM_BASE_URL so rllm can track all LLM calls.
        instruction:  Task instruction (also written to INSTRUCTIONS.md).
    """
    container_name = f"rllm-openhands-{uuid.uuid4().hex[:12]}"
    # breakpoint()
    cmd = [
        "docker", "run",
        "--rm",
        "-it",
        "--name", container_name,

        # --- LLM routing ---
        # proxied_url carries the rllm metadata slug so every LLM call made
        # by OpenHands SDK (via workspace/entrypoint.py) is attributed to
        # this training session by the rllm LiteLLM proxy.
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL=openai/{_MODEL_NAME}",
        "-e", f"NPU_OPERATOR_TASK={'1' if npu_operator else '0'}",
    ]
        
    if npu_operator:
        cmd.extend(["-e", f"TASK_INSTRUCTION=\"{instruction}\""])
    
    cmd.extend([
            "-e", "http_proxy=", 
            "-e", "https_proxy=",
            "-e", "no_proxy=host.docker.internal,127.0.0.1,localhost,172.17.0.1"
        ])

    cmd.extend(
        [
        # --- Workspace ---
        # The host workspace dir is mounted; the agent operates directly
        # inside the container — no inner sandbox is created.
        "-e", "WORKSPACE_BASE=/opt/workspace",
        # "-v", f"{workspace}:/opt/workspace",
        "-v", f"/home/g00841271/rllm-071/examples/openhands_sdk/workspace_debug:/opt/workspace",
        # "--entrypoint", f"/app/rllm_entrypoint.py",
        "--entrypoint", f"/opt/workspace/entrypoint.py",
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
    )
    
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

# Keys passed through AgentSdkEngine / session wrapper as kwargs alongside
# extra_info fields; they must not be treated as task payload.
_ROLLOUT_CONFIG_KEYS = frozenset({"config", "base_url", "session_uid", "is_validation"})


def _rollout_task_and_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize call shapes.

    - AgentSdkEngine: ``partial(wrapped, metadata, **extra_info)`` → session
      calls ``rollout(instruction=..., scenario=..., ...)`` (kwargs only).
    - Legacy / manual: ``rollout(task_dict, config_dict)``.
    """
    config: dict[str, Any] = {}
    raw_cfg = kwargs.get("config")
    if isinstance(raw_cfg, dict):
        config = dict(raw_cfg)
    if "base_url" in kwargs:
        config["base_url"] = kwargs["base_url"]
    if "session_uid" in kwargs:
        config["session_uid"] = kwargs["session_uid"]

    if len(args) >= 1 and isinstance(args[0], dict):
        task = dict(args[0])
        if len(args) >= 2 and isinstance(args[1], dict):
            config = {**config, **args[1]}
        return task, config

    task = {k: v for k, v in kwargs.items() if k not in _ROLLOUT_CONFIG_KEYS}
    return task, config


def rollout(*args: Any, **kwargs: Any) -> list[dict]:

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
    slug_uids = kwargs.get("_rllm_proxy_session_uids")
    slug_name = kwargs.get("_rllm_proxy_session_name")
    if slug_uids is not None and not isinstance(slug_uids, list):
        slug_uids = list(slug_uids) if slug_uids else None
    
    task, config = _rollout_task_and_config(args, kwargs)
    
    # Raw proxy URL — rllm passes this before any slug is applied
    proxy_url = config.get("base_url", "http://127.0.0.1:4000/v1")

    # Generate a unique session identifier for this rollout.
    # Encoding it in the URL lets the rllm proxy associate every OpenHands
    # LLM call with this specific training episode.
    # session_uid = str(uuid.uuid4())
    # metadata: dict[str, Any] = {
    #     "session_uids": [session_uid],
    #     "session_name": f"openhands-{session_uid[:8]}",
    # }
    
    
    # breakpoint()
    
    metadata = _routing_metadata_for_rollout(slug_uids, slug_name)
    trace_label = _trace_label_from_routing_metadata(metadata)
    _uids = metadata.get("session_uids") or []
    logger.info(
        "[openhands] proxy slug: n_uids=%d session_name=%r trace_tail=%s",
        len(_uids),
        metadata.get("session_name"),
        _uids[-1][-12:] if _uids else "",
    )

    # Build the proxied URL with embedded metadata slug.
    # _build_proxied_base_url inlines the same logic as rllm.sdk.proxy.
    # proxied_url = _build_proxied_base_url(proxy_url, metadata)
    proxied_url = build_proxied_base_url(proxy_url, metadata)

    # Rewrite localhost/127.0.0.1 → host.docker.internal so the URL is
    # reachable from inside the OpenHands Docker container.
    proxied_url = _to_container_url(proxied_url)

    npu = _is_npu_operator_task(task)
    workspace = _setup_workspace(task)
    instruction = task.get("instruction", "Fix the bug in the repository.")
    reward = 0.0

    try:
        output = _run_openhands_container(
            workspace, proxied_url, instruction, npu_operator=npu
        )
        reward = _default_reward(task, workspace, output)
        logger.info(
            "[openhands] trace_label=%s reward=%.2f instruction=%s",
            trace_label, reward, instruction[:80],
        )
    except Exception:
        logger.exception("[openhands] Rollout failed (trace_label=%s)", trace_label)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return reward
    # return [
    #     Trajectory(
    #             name="openhands",
    #             steps=[],
    #             reward=reward,
    #             metadata={"session_uid": session_uid},
    #         )
    # ]
