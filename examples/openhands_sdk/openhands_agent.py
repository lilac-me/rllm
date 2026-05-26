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

import time
import base64
import json
import logging
import os
import random
import shutil
import stat
import subprocess
import tempfile
import uuid
from urllib.parse import urlparse, urlunparse
from typing import Any
from pathlib import Path

from rllm.types import Trajectory
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url

import os
import sys
import time
import socket
import traceback
import faulthandler
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
faulthandler.enable(all_threads=True)


def dbg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    host = socket.gethostname()
    pid = os.getpid()
    rank = os.getenv("RANK", "NA")
    local_rank = os.getenv("LOCAL_RANK", "NA")
    msg=f"[{ts}] [host={host}] [pid={pid}] [rank={rank}] [local_rank={local_rank}] {msg}"
    path = f"/workspace/results/rllm_{socket.gethostname()}_{os.getpid()}.log"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.time()} {msg}\n")
        f.flush()
    
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


# _OPERATOR_SEED_NAMES = (
#     "AGENTS.md",
#     "INSTRUCTIONS.md",
#     ".agents",
#     "tools",
#     "src",
# )


# def _chmod_executable_scripts(directory: str) -> None:
#     """Recursively set +x on .sh and .py files under *directory*."""
#     for root, _, files in os.walk(directory):
#         for name in files:
#             if name.endswith((".sh", ".py")):
#                 path = os.path.join(root, name)
#                 try:
#                     os.chmod(
#                         path,
#                         os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
#                     )
#                 except OSError:
#                     pass


# ---------------------------------------------------------------------------
# Mock pipeline (bring-up testing without real NPU)
#
# When OPENHANDS_MOCK_PIPELINE=1, this script **replaces**
# tools/operator_pipeline.sh in the temp workspace.  The interface
# (--op_name flag, metrics.json output schema) is identical to the real
# pipeline, so INSTRUCTIONS.md and reward logic work unchanged.
# ---------------------------------------------------------------------------

_MOCK_PIPELINE_SCRIPT = r"""#!/bin/bash
# Mock operator pipeline for AGENTS.md flow
# Checks model_new_ascendc.py existence and ModelNew class
set -euo pipefail

OP_NAME="operator"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --op_name) OP_NAME="$2"; shift 2;;
        *) shift;;
    esac
done

OUTPUT_DIR="output/${OP_NAME}"
IMPL="${OUTPUT_DIR}/model_new_ascendc.py"
TRACE="${OUTPUT_DIR}/trace.md"
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
cat > "$M" <<EOFM
{
  "success": true,
  "ast_check_ok": true,
  "correctness_ok": true,
  "perf_data": {"speedup_vs_torch": ${SP}},
  "error": null
}
EOFM

# Create mock trace.md
mkdir -p "$OUTPUT_DIR"
cat > "$TRACE" <<EOFTRACE
# Trace for ${OP_NAME}

## Phase 0: 参数确认
- 成功

## Phase 1: 环境准备
- 成功

## Phase 2: 测试用例精简
- 成功

## Phase 3: TileLang 设计表达
- 成功

## Phase 4: AscendC 转译与验证
- 成功

## Phase 5: 性能分析
- speedup_vs_torch: ${SP}

## Phase 6: 全量用例验证
- 成功

## Phase 7: Trace 记录
- 成功
EOFTRACE

echo "[mock] OK: speedup=${SP}x"
"""


# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------

_NPU_INSTRUCTION_TEMPLATE = """生成ascendC算子，npu=0，算子描述文件为 src/{op_name}.py，输出到 ./output/{op_name}/
"""


def _setup_npu_operator_workspace(task: dict[str, Any], trace_label: str) -> str:
    pwd = Path(__file__).parent
    _WORKSPACE_PKG = pwd / "workspace"
    # DooD: must be a host path also bind-mounted into the main container at the
    # SAME path, since the sibling container's `-v {workspace}:/opt/workspace` is
    # resolved by the host dockerd. Requires `-v /home/docker/openhands_workspace:/home/docker/openhands_workspace`
    # on main container startup. See plan §13.23.
    workspace_temp = Path("/home/docker/openhands_workspace")
    workspace_temp.mkdir(parents=True, exist_ok=True)
    workspace = tempfile.mkdtemp(prefix=f"trajectory-{trace_label}-", dir=workspace_temp)
    op_name = task.get("op_name", "operator")
    arch = task.get("arch", "ascend910b1")
    instruction = task.get("instruction", "Implement a simple vector_add-style operator.")
    task_code = task.get("task_code", "")


    shutil.copytree(_WORKSPACE_PKG, workspace, dirs_exist_ok=True)
    os.utime(workspace, None)

    # for name in _OPERATOR_SEED_NAMES:
    #     src = os.path.join(_WORKSPACE_PKG, name)
    #     dst = os.path.join(workspace, name)
    #     if not os.path.exists(src):
    #         continue
    #     if os.path.isdir(src):
    #         shutil.copytree(src, dst)
    #     else:
    #         shutil.copy2(src, dst)
    # _chmod_executable_scripts(os.path.join(workspace, "tools"))

    if _OPENHANDS_MOCK_PIPELINE:
        mock_path = os.path.join(workspace, "agent_workdir", "tools", "operator_pipeline.sh")
        with open(mock_path, "w") as f:
            f.write(_MOCK_PIPELINE_SCRIPT)
        os.chmod(mock_path, 0o755)

    with open(os.path.join(workspace, "agent_workdir", "INSTRUCTIONS.md"), "w") as f:
        f.write(_NPU_INSTRUCTION_TEMPLATE.format(op_name=op_name))

    if task_code:
        src_dir = os.path.join(workspace, "agent_workdir", "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, f"{op_name}.py"), "w") as f:
            f.write(task_code)

        # 从 NPUKernelBench 查找并复制 .json 测试用例文件
        benchmark_dir = os.path.join(workspace, "agent_workdir", "src", "NPUKernelBench")
        json_found = False
        for level in ["level0", "level1", "level2", "level3", "level4"]:
            level_dir = os.path.join(benchmark_dir, level)
            if os.path.exists(level_dir):
                candidate_json = os.path.join(level_dir, f"{op_name}.json")
                if os.path.exists(candidate_json):
                    shutil.copy2(candidate_json, os.path.join(src_dir, f"{op_name}.json"))
                    json_found = True
                    break

        # 如果找不到，创建空的 .json 文件（AGENTS.md Phase 2 需要）
        if not json_found:
            with open(os.path.join(src_dir, f"{op_name}.json"), "w") as f:
                f.write("")
    return workspace


# ---------------------------------------------------------------------------
# Metrics & reward
# ---------------------------------------------------------------------------

def _has_metrics(path: str) -> dict[str, Any] | None:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            dbg("[openhands-npu] bad metrics.json: %s, %s", exc, path) # TODO
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

    # AGENTS.md 流程的输出文件: output/{op_name}/model_new_ascendc.py
    output_dir = os.path.join(workspace_dir, "output", op_name)
    impl_file = os.path.join(output_dir, "model_new_ascendc.py")
    if not os.path.exists(impl_file):
        logger.info("[openhands-npu] no impl file %s -> reward=0.0", impl_file)
        return 0.0

    # 检查 trace.md 判断各阶段成功/失败
    trace_file = os.path.join(output_dir, "trace.md")
    reward = 0.2  # 默认: 有实现但不确定成功

    if os.path.exists(trace_file):
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_content = f.read()

            # 根据 trace.md 内容判断 reward
            if "Phase 4: 成功" in trace_content or "AscendC 验证通过" in trace_content:
                reward = 0.8
                logger.info("[openhands-npu] AscendC implementation verified -> reward=0.8")
            elif "Phase 3: 成功" in trace_content or "TileLang 验证通过" in trace_content:
                reward = 0.5
                logger.info("[openhands-npu] TileLang implementation verified -> reward=0.5")
            elif "Phase 4: 失败" in trace_content or "AscendC 验证失败" in trace_content:
                reward = 0.3
                logger.info("[openhands-npu] AscendC verification failed -> reward=0.3")
            elif "Phase 3: 失败" in trace_content or "TileLang 验证失败" in trace_content:
                reward = 0.2
                logger.info("[openhands-npu] TileLang verification failed -> reward=0.2")
            else:
                logger.info("[openhands-npu] trace exists but no clear status -> reward=0.2")
        except Exception as e:
            logger.warning("[openhands-npu] failed to read trace.md: %s", e)
    else:
        logger.info("[openhands-npu] no trace.md found -> reward=0.2")

    # 检查 preformance.json 获取性能数据（可选）
    perf_file = os.path.join(output_dir, "preformance.json")
    if os.path.exists(perf_file) and reward >= 0.5:
        try:
            with open(perf_file, "r") as f:
                perf_data = json.load(f)
            # 如果有性能数据，可以微调 reward
            speedup = perf_data.get("speedup_vs_torch", 1.0)
            if speedup > 1.0:
                reward = min(reward + 0.1 * (speedup - 1.0), 1.0)
                logger.info("[openhands-npu] speedup=%.2f -> reward=%.3f", speedup, reward)
        except Exception:
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
    metric_dir = workspace_dir + "/agent_workdir"
    if not _ARTIFACT_DIR:
        return
    if not os.path.isdir(metric_dir):
        return

    op_name = task.get("op_name", "operator")
    ts = time.strftime("%Y%m%d_%H%M%S")
    # short_id = uuid.uuid4().hex[:6]
    dest = os.path.join(_ARTIFACT_DIR, f"{workspace_dir.split('/')[-1]}_{trace_label}_{op_name}_{ts}")

    try:
        os.makedirs(dest, exist_ok=True)

        # AGENTS.md 流程的产物在 output/{op_name}/ 目录下
        output_dir = os.path.join(metric_dir, "output", op_name)
        candidates = [
            (os.path.join(metric_dir, "conversation.log"), "conversation.log"),
            (os.path.join(metric_dir, "conversation_result.json"), "conversation_result.json"),
            # AGENTS.md 流程的核心产物
            (os.path.join(output_dir, "model_new_ascendc.py"), "model_new_ascendc.py"),
            (os.path.join(output_dir, "model_new_tilelang.py"), "model_new_tilelang.py"),
            (os.path.join(output_dir, "trace.md"), "trace.md"),
            (os.path.join(output_dir, "preformance.json"), "preformance.json"),
            # 设计文件
            (os.path.join(output_dir, "design", "block_level", "design.md"), "design_block_level.md"),
            (os.path.join(output_dir, "design", "tile_level", "design.md"), "design_tile_level.md"),
            # kernel 文件
            (os.path.join(output_dir, "kernel", "kernel.cpp"), "kernel.cpp"),
            (os.path.join(metric_dir, "INSTRUCTIONS.md"), "INSTRUCTIONS.md"),
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
    # uids = metadata.get("session_uids") or []
    # if uids:
    #     return str(uids[-1])[:18]
    name = metadata.get("session_name")
    trace_label = str(name or "none").split('-')[-1].replace(':', '-')
    return trace_label


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
    operator_backend = str(task.get("operator_backend", "ascendc"))

    path = Path("/tmp/shared_npu_lock")
    path.mkdir(mode=0o755, parents=True, exist_ok=True)

    # TODO(遗留): Ascend NPU 入容器所需的 docker --device / 额外 -v 挂载（如 /dev/davinci*、驱动相关路径）
    # 待按 CANN 与所用基础镜像文档确定，并与 OPENHANDS_ASCEND_VISIBLE_DEVICES 一起在真机验证。
    # 当前仅挂载 workspace、entrypoint 及注入 ASCEND_RT_VISIBLE_DEVICES（若设置）。
    cmd = [
        "docker", "run",
        #"--rm",  # TODO
        "-d",
        "--name", container_name,
        "--network", "host",
        "--ipc", "host",
        "--shm-size", "500g",
        "--privileged",
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL=openai/{_MODEL_NAME}",
        "-e", f"OPERATOR_BACKEND={operator_backend}",
        "-e", f"OPERATOR_ARCH={arch}",
        "-e", f"OPERATOR_NAME={op_name}",
        
        "-e", "http_proxy=", 
        "-e", "https_proxy=",
        "-e", "no_proxy=host.docker.internal,127.0.0.1,localhost,172.17.0.1",
        "-e", "WORKSPACE_BASE=/opt/workspace/agent_workdir",
        "-e", f"OBSERVER_API_URL=http://host.docker.internal:18858",
        "-e", f"MAX_ITERATIONS={_MAX_ITERATIONS}",
        
        # 评估专用
        "-e", "EVAL_LOCK_DIR=/shared/device-locks",
        "-e", "EVAL_DEVICE_PREFIX=npu",
        "-e", "EVAL_DEVICE_COUNT=1",
        "-e", "EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES",
        "-e", "EVAL_RETRY_INTERVAL=1.0",
        "-e", "EVAL_TIMEOUT=None",
        "-e", "EVAL_VERBOSE=true",
        
        # "--device", "/dev/davinci0",
        # "--device", "/dev/davinci1",
        # "--device", "/dev/davinci2",
        # "--device", "/dev/davinci3",
        # "--device", "/dev/davinci4",
        # "--device", "/dev/davinci5",
        # "--device", "/dev/davinci6",
        # "--device", "/dev/davinci7",
        # "--device", "/dev/davinci_manager",
        # "--device", "/dev/hisi_hdc",
        # "--device", "/dev/devmm_svm",
        
        "-v", "/dev:/dev",
        "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
        "-v", "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
        "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
        "-v", "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
        "-v", "/etc/ascend_install.info:/etc/ascend_install.info:ro",
        "-v", "/usr/local/sbin:/usr/local/sbin:ro",
        "-v", "/etc/localtime:/etc/localtime:ro",
        "-v", "/etc/timezone:/etc/timezone:ro",
        "-v", "/mnt/pipeline-data:/mnt/pipeline-data",
        #"-v", "/opt/DPC:/opt/DPC",

        # os.environ["OBSERVER_API_URL"] = "http://127.0.0.1:18858"
        "-v", f"{workspace}:/opt/workspace",
        "-v", f"/tmp/shared_npu_lock:/shared/device-locks",   # 共享文件锁所在路径
        # "-v", f"/home/g00841271/rllm-071/examples/openhands_sdk/workspace_debug:/opt/workspace",
        "--entrypoint", f"/opt/workspace/entrypoint.py",
        "--add-host", "host.docker.internal:host-gateway",
    ]
    cmd.extend([_OPENHANDS_IMAGE,])
    
    dbg(
        f"[openhands] Launching container {container_name} (proxied_url={proxied_url[:70]}...)"
    )
    import shlex
    print("DEBUG CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    dbg(" ".join(shlex.quote(c) for c in cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            #timeout=_CONTAINER_TIMEOUT, # TODO
        )
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if result.returncode != 0:
            logger.warning(
                "[openhands] Container %s exited with code %d",
                container_name, result.returncode,
            )
            return output
    
        wait_result = subprocess.run(
            ["docker", "wait", container_name],
            capture_output=True,
            #timeout=_CONTAINER_TIMEOUT, # TODO
        )
        
        dbg(f"wait_result={wait_result}, {wait_result.stdout.decode().strip()}")
        try:
            exit_code = int(wait_result.stdout.decode().strip())
        except Exception as e: # TODO
            logger.error(f"exit_code error, {e}")
        
        # 获取并保存日志
        logs_result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
        )

        logs = logs_result.stdout + logs_result.stderr
        with open(workspace+"/agent_workdir/conversation.log", "wb") as f:
            f.write(logs)

        with open(workspace+"/agent_workdir/conversation_result.json", "w") as f:
            json.dump({
                "container": container_name,
                "exit_code": exit_code,
                "timestamp": time.time(),
            }, f, indent=2)

        return exit_code
    
    except subprocess.TimeoutExpired:
        logger.error("[openhands] Container %s timed out", container_name)
        subprocess.run(["docker", "kill", container_name])
        return -1
    except Exception:
        logger.exception("[openhands] Failed to run container %s", container_name)
        return -1
    finally:
        # DEBUG: keep container for inspection
        # subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL)
        print(f"[DEBUG] kept container {container_name} for inspection")

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
    base_url_port = os.getenv("OPENHANDS_BASE_URL_PORT", 4000)
    proxy_url = config.get("base_url", f"http://127.0.0.1:{base_url_port}/v1")

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

    logger.info(
        "[openhands] proxy slug: session_uids=%r session_name=%r",
        metadata.get("session_uids"),
        metadata.get("session_name"),
    )

    # Build the proxied URL with embedded metadata slug.
    # _build_proxied_base_url inlines the same logic as rllm.sdk.proxy.
    # proxied_url = _build_proxied_base_url(proxy_url, metadata)
    proxied_url = build_proxied_base_url(proxy_url, metadata)

    # Rewrite localhost/127.0.0.1 → host.docker.internal so the URL is
    # reachable from inside the OpenHands Docker container.
    proxied_url = _to_container_url(proxied_url)

    workspace = _setup_npu_operator_workspace(task, trace_label)
    instruction = task.get("instruction", "")

    # =========================================================================
    # ⚠️  STAGE1 MOCK REWARD FALLBACK — MUST REMOVE BEFORE W3 REAL TRAINING ⚠️
    # =========================================================================
    # Why this block exists (W2.22 + W2.24, see plan §13.25 / §13.27):
    #   In stage1 we force MAX_ITERATIONS=1 to lock OpenHands prompt at first-turn
    #   ~30k tokens (otherwise it grows past max_model_len within 2 turns). At
    #   MAX_ITERATIONS=1 the agent has no time to write any kernel file, so
    #   `_npu_operator_reward` returns 0.0 ("no impl file" branch). All 4
    #   rollouts returning 0.0 → GRPO `std=0` → NaN advantage → PPO step crashes.
    #   Random fallback gives variance so GRPO + PPO step can run.
    #
    # Removal criteria (when to delete this block):
    #   1. OpenHands condenser is integrated (LLM-based summarization or Recent)
    #      so prompts don't grow past max_model_len in multi-turn
    #   2. MAX_ITERATIONS raised back from 1 to a real value (typical: 30+)
    #   3. Real rollouts produce reward > 0 reliably (verify by grepping
    #      "real_reward=0.0" in production logs; should be rare/empty)
    #   4. Variance across rollouts comes from real signal, not noise
    #
    # Tracked in plan §9 "stage2 TODO" + §13.27. The removal is a code-only
    # change (no env, no config); just revert this block to:
    #     reward = 0.0
    #     try:
    #         output = _run_openhands_container(...)
    #         reward = _npu_operator_reward(task, workspace+"/agent_workdir", output)
    #     except Exception:
    #         logger.exception(...)
    #     finally:
    #         _archive_npu_artifacts(...)
    #     return reward
    # and remove `import random` if no other code uses it.
    # =========================================================================
    reward = random.random()

    try:
        output = _run_openhands_container(
            workspace, proxied_url, instruction, task=task
        )
        metrics_dir = workspace + "/agent_workdir"
        real_reward = _npu_operator_reward(task, metrics_dir, output)
        if real_reward > 0.0:
            reward = real_reward
            logger.info(
                "[openhands] trace_label=%s real_reward=%.3f instruction=%s",
                trace_label, reward, instruction[:80],
            )
        else:
            # STAGE1 MOCK: real_reward == 0 (no impl file), keep random fallback
            logger.info(
                "[openhands] trace_label=%s real_reward=0.0 (no impl file); STAGE1_MOCK keeping random=%.3f",
                trace_label, reward,
            )
    except Exception:
        logger.exception(
            "[openhands] Rollout failed (trace_label=%s); STAGE1_MOCK keeping random=%.3f",
            trace_label, reward,
        )
    finally:
        _archive_npu_artifacts(workspace, task, trace_label, reward)
        # shutil.rmtree(workspace, ignore_errors=True) # TODO
    return reward

