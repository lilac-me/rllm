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

_OPENHANDS_IMAGE = os.environ.get("OPENHANDS_IMAGE", "rllm-openhands")
_MODEL_NAME = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")
_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
_CONTAINER_TIMEOUT = int(os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "600"))
_ARTIFACT_DIR = os.environ.get("OPENHANDS_ARTIFACT_DIR", "")
_OPENHANDS_MOCK_PIPELINE = os.environ.get("OPENHANDS_MOCK_PIPELINE", "0") == "1"
# _OPENHANDS_ASCEND_VISIBLE_DEVICES = os.environ.get("OPENHANDS_ASCEND_VISIBLE_DEVICES", "").strip()

# ---------------------------------------------------------------------------
# NPU operator workspace setup
# ---------------------------------------------------------------------------

_SDK_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_PKG = os.path.join(_SDK_DIR, "workspace")
_OPERATOR_SEED_NAMES = (
    "AGENTS.md",
    "INSTRUCTIONS.md",
    ".agents",
    "tools",
    "src",
)


def _chmod_executable_scripts(directory: str) -> None:
    """Recursively set +x on .sh and .py files under *directory*."""
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith((".sh", ".py")):
                path = os.path.join(root, name)
                try:
                    os.chmod(
                        path,
                        os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                    )
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Mock pipeline (bring-up testing without real NPU)
#
# When OPENHANDS_MOCK_PIPELINE=1, this script **replaces**
# tools/operator_pipeline.sh in the temp workspace.  The interface
# (--op_name flag, metrics.json output schema) is identical to the real
# pipeline, so INSTRUCTIONS.md and reward logic work unchanged.
# ---------------------------------------------------------------------------

_MOCK_PIPELINE_SCRIPT = r"""#!/bin/bash
# Mock operator pipeline — drop-in replacement for operator_pipeline.sh.
# Performs Python syntax + ModelNew class check; outputs metrics.json
# with randomised speedup. No NPU or torch_npu required.
set -euo pipefail

OP_NAME="operator"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --op_name) OP_NAME="$2"; shift 2;;
        *) shift;;
    esac
done

IMPL="src/${OP_NAME}_triton_ascend_impl.py"
M="metrics.json"

if [ ! -f "$IMPL" ]; then
    printf '{"success":false,"error":"File not found: %s","ast_check_ok":false,"correctness_ok":false}\n' "$IMPL" > "$M"
    echo "[mock] FAIL: $IMPL not found"; exit 0
fi

# Python syntax check
if ! python3 -c "import ast; ast.parse(open('${IMPL}').read())" 2>/dev/null; then
    printf '{"success":false,"error":"Syntax error in %s","ast_check_ok":false,"correctness_ok":false}\n' "$IMPL" > "$M"
    echo "[mock] FAIL: syntax error"; exit 0
fi

# ModelNew class check
if ! python3 -c "
import ast
tree = ast.parse(open('${IMPL}').read())
assert any(n.name == 'ModelNew' for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
" 2>/dev/null; then
    printf '{"success":false,"error":"ModelNew class not found in %s","ast_check_ok":true,"correctness_ok":false}\n' "$IMPL" > "$M"
    echo "[mock] FAIL: ModelNew not found"; exit 0
fi

# Mock success with random speedup
SP=$(python3 -c "import random; print(f'{random.uniform(0.8, 2.0):.2f}')")
TL=$(python3 -c "print(f'{1.0/float(${SP}):.4f}')")
cat > "$M" <<EOFM
{
  "success": true,
  "ast_check_ok": true,
  "correctness_ok": true,
  "perf_data": {"speedup_vs_torch": ${SP}, "torch_latency_ms": 1.0, "triton_latency_ms": ${TL}},
  "error": null
}
EOFM
echo "[mock] OK: speedup=${SP}x"
"""


# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------

_NPU_INSTRUCTION_TEMPLATE = """# 当前任务

- 算子名称: **{op_name}**
- 目标架构: **{arch}**

{instruction}

## 任务格式（KernelBench）

任务文件: `src/{op_name}.py`（包含 `Model`、`get_inputs()`、`get_init_inputs()`）。

## 要求

1. 阅读 `AGENTS.md` 了解全局约定和工作流。
2. 在 `src/{op_name}_triton_ascend_impl.py` 中实现 `ModelNew` 类。
3. 运行验证流水线：
   ```bash
   bash tools/operator_pipeline.sh --op_name {op_name}
   ```
4. 读取 `metrics.json`，根据 `error` 字段修复并重试，直至 `"success": true`。
"""


def _setup_npu_operator_workspace(task: dict[str, Any]) -> str:
    workspace = tempfile.mkdtemp(prefix=f"openhands-npu-{uuid.uuid4().hex[:8]}-")
    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    instruction = task.get("instruction", "Implement a simple vector_add-style operator.")
    task_code = task.get("task_code", "")

    for name in _OPERATOR_SEED_NAMES:
        src = os.path.join(_WORKSPACE_PKG, name)
        dst = os.path.join(workspace, name)
        if not os.path.exists(src):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    _chmod_executable_scripts(os.path.join(workspace, "tools"))

    if _OPENHANDS_MOCK_PIPELINE:
        mock_path = os.path.join(workspace, "tools", "operator_pipeline.sh")
        with open(mock_path, "w") as f:
            f.write(_MOCK_PIPELINE_SCRIPT)
        os.chmod(mock_path, 0o755)

    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(_NPU_INSTRUCTION_TEMPLATE.format(op_name=op_name, arch=arch, instruction=instruction))

    if task_code:
        src_dir = os.path.join(workspace, "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, f"{op_name}.py"), "w") as f:
            f.write(task_code)

    return workspace


# ---------------------------------------------------------------------------
# Metrics & reward
# ---------------------------------------------------------------------------

def _load_metrics(workspace_dir: str) -> dict[str, Any] | None:
    path = os.path.join(workspace_dir, "metrics.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[openhands-npu] bad metrics.json: %s", exc)
    return None


def _reward_from_metrics(perf: dict[str, Any]) -> float:
    if not bool(perf.get("success", False)):
        ast_ok = bool(perf.get("ast_check_ok", False))
        corr_ok = bool(perf.get("correctness_ok", False))
        if corr_ok:
            return 0.4
        elif ast_ok:
            return 0.3
        return 0.2

    perf_data = perf.get("perf_data") or {}
    speedup = float(perf_data.get("speedup_vs_torch", 1.0))
    return min(0.5 + 0.5 * (speedup / 2.0), 1.0)


def _npu_operator_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    del output
    op_name = task.get("op_name", "operator")

    impl_file = os.path.join(workspace_dir, "src", f"{op_name}_triton_ascend_impl.py")
    if not os.path.exists(impl_file):
        logger.info("[openhands-npu] no impl file %s -> reward=0.0", impl_file)
        return 0.0

    # TODO: consider rejecting empty/stub-only impl files (os.path.getsize check)

    perf = _load_metrics(workspace_dir)
    if not perf:
        logger.info("[openhands-npu] impl present but no metrics -> reward=0.2")
        return 0.2

    reward = _reward_from_metrics(perf)

    best_path = os.path.join(workspace_dir, "metrics_best.json")
    if os.path.exists(best_path):
        try:
            with open(best_path) as f:
                best = json.load(f)
            best_reward = _reward_from_metrics(best)
            if best_reward > reward:
                logger.info(
                    "[openhands-npu] best version has higher reward: "
                    "current=%.3f best=%.3f -> using best",
                    reward, best_reward,
                )
                reward = best_reward
        except (json.JSONDecodeError, OSError):
            pass

    logger.info("[openhands-npu] final reward=%.3f", reward)
    return reward


def _archive_npu_artifacts(
    workspace_dir: str,
    task: dict[str, Any],
    trace_label: str,
    reward: float,
) -> None:
    """Copy key rollout artifacts to a persistent directory before cleanup.

    Controlled by env OPENHANDS_ARTIFACT_DIR. No-op if unset/empty.
    Never raises — archival failure must not affect training.
    """
    if not _ARTIFACT_DIR:
        return
    if not os.path.isdir(workspace_dir):
        return

    op_name = task.get("op_name", "operator")
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    dest = os.path.join(_ARTIFACT_DIR, f"{trace_label}_{op_name}_{ts}_{short_id}")

    try:
        os.makedirs(dest, exist_ok=True)

        candidates = [
            (os.path.join(workspace_dir, "src", f"{op_name}_triton_ascend_impl.py"),
             f"{op_name}_triton_ascend_impl.py"),
            (os.path.join(workspace_dir, "src", f"{op_name}_triton_ascend_impl_best.py"),
             f"{op_name}_triton_ascend_impl_best.py"),
            (os.path.join(workspace_dir, "metrics.json"), "metrics.json"),
            (os.path.join(workspace_dir, "metrics_best.json"), "metrics_best.json"),
            (os.path.join(workspace_dir, "INSTRUCTIONS.md"), "INSTRUCTIONS.md"),
        ]

        copied = 0
        for src_path, dst_name in candidates:
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(dest, dst_name))
                copied += 1

        manifest = {
            "trace_label": trace_label,
            "op_name": op_name,
            "arch": task.get("arch", "ascend910b1"),
            "reward": reward,
            "timestamp": ts,
        }
        with open(os.path.join(dest, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if copied == 0:
            shutil.rmtree(dest, ignore_errors=True)
            logger.debug("[openhands] No artifacts to archive for trace=%s", trace_label)
        else:
            logger.info("[openhands] Archived %d artifacts → %s (reward=%.3f)", copied, dest, reward)
    except Exception:
        logger.warning("[openhands] Failed to archive artifacts for trace=%s", trace_label, exc_info=True)



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
# OpenHands container launch
# ---------------------------------------------------------------------------

def _run_openhands_container(
    workspace: str,
    proxied_url: str,
    instruction: str,
    *,
    task: dict[str, Any] | None = None,
) -> str:
    """Start an OpenHands headless container, wait for completion, return logs.

    Args:
        workspace:    Host-side workspace directory (mounted into container).
        proxied_url:  LiteLLM proxy URL with embedded rllm metadata slug.
                      Passed as LLM_BASE_URL so rllm can track all LLM calls.
        instruction:  Task instruction (also written to INSTRUCTIONS.md).
    """
    # TODO(遗留): 禁止 agent 外网（如误 apt）目前仅靠 workspace/AGENTS.md 软约束。若需硬隔离，在此
    # 组装 docker cmd 时加入 --network（例如宿主机预先 docker network create --internal …），并验证
    # 仍能访问 host.docker.internal 上的 LiteLLM；勿用 network=none 除非 LLM 不依赖宿主机 HTTP。
    task = task or {}
    container_name = f"rllm-openhands-{uuid.uuid4().hex[:12]}"
    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    operator_backend = str(task.get("operator_backend", "triton"))

    # TODO(遗留): Ascend NPU 入容器所需的 docker --device / 额外 -v 挂载（如 /dev/davinci*、驱动相关路径）
    # 待按 CANN 与所用基础镜像文档确定，并与 OPENHANDS_ASCEND_VISIBLE_DEVICES 一起在真机验证。
    # 当前仅挂载 workspace、entrypoint 及注入 ASCEND_RT_VISIBLE_DEVICES（若设置）。
    cmd = [
        "docker", "run",
        "--rm",
        "-d",
        "--name", container_name,
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL=openai/{_MODEL_NAME}",
        "-e", f"OPERATOR_BACKEND={operator_backend}",
        "-e", f"OPERATOR_ARCH={arch}",
        "-e", f"OPERATOR_NAME={op_name}",
        "-e", f"TASK_INSTRUCTION=\"{instruction}\"",
        
        "-e", "http_proxy=", 
        "-e", "https_proxy=",
        "-e", "no_proxy=host.docker.internal,127.0.0.1,localhost,172.17.0.1"
        "-e", "WORKSPACE_BASE=/opt/workspace/agent_workdir",
        "-e", f"OBSERVER_API_URL=http://127.0.0.1:18858"
        # os.environ["OBSERVER_API_URL"] = "http://127.0.0.1:18858"
        "-v", f"{workspace}:/opt/workspace",
        # "-v", f"/home/g00841271/rllm-071/examples/openhands_sdk/workspace_debug:/opt/workspace",
        "--entrypoint", f"/opt/workspace/entrypoint.py",
        "-e", f"MAX_ITERATIONS={_MAX_ITERATIONS}",
        "--add-host", "host.docker.internal:host-gateway",
    ]
    cmd.extend([_OPENHANDS_IMAGE,])
    
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
_ROLLOUT_CONFIG_KEYS = frozenset({
    "config",
    "base_url",
    "session_uid",
    "is_validation",
})


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
    
    
    metadata = assemble_routing_metadata()
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

    workspace = _setup_npu_operator_workspace(task)
    instruction = task.get("instruction", "")
    reward = 0.0

    try:
        output = _run_openhands_container(
            workspace, proxied_url, instruction, task=task
        )
        reward = _npu_operator_reward(task, workspace, output)
        logger.info(
            "[openhands] trace_label=%s reward=%.2f instruction=%s",
            trace_label, reward, instruction[:80],
        )
    except Exception:
        logger.exception("[openhands] Rollout failed (trace_label=%s)", trace_label)
    finally:
        _archive_npu_artifacts(workspace, task, trace_label, reward)
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
