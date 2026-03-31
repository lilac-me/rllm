"""
OpenHands agent — rollout function for NPU operator generation (sandbox mode).

This module defines ``rollout(task, config)`` which is the entry point
called by ``worker_server.py`` inside a Docker sandbox. It:

1. Creates a per-rollout workspace with the NPU operator task instruction
2. Writes a mock profiling script (profile_wrapper.sh) into the workspace
3. Invokes OpenHands headless with ``runtime="docker"`` so that OpenHands
   manages its own isolated runtime container for code execution
4. Evaluates the result by checking kernel.cpp existence and mock profiling
5. Returns reward as an rllm trajectory dict

The function does NOT import rllm.sdk.session — session tracking is
handled externally by worker_server.py via the metadata slug mechanism.

OpenHands Docker isolation
--------------------------
``_run_openhands`` sets ``runtime="docker"`` in the OpenHands ``AppConfig``.
This tells OpenHands to spawn a **per-task Docker container** (using the
runtime image) as the code-execution sandbox.  The OpenHands Python process
itself runs inside the outer rllm sandbox container — meaning the rollout
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
import traceback # 确保文件顶部导入了 traceback
import sys

import logging
import os
import random
import shutil
import stat
import subprocess
import tempfile
import uuid
from typing import Any
from rllm.agents.agent import Trajectory

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
_SANDBOX_USER_ID = int(os.environ.get("OPENHANDS_SANDBOX_USER_ID", "1000"))
_DEFAULT_RUNTIME_IMAGE = (
    "docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik"
)

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
# OpenHands invocation (with Docker runtime)
# ---------------------------------------------------------------------------

# def _run_openhands(workspace: str, base_url: str, instruction: str) -> str:
#     """Run OpenHands headless with its own Docker runtime container.

#     OpenHands is invoked in-process (``openhands-ai`` must be installed),
#     but with ``runtime="docker"`` so that all code execution happens inside
#     an isolated OpenHands runtime container — not directly in the calling
#     process.

#     The outer rllm sandbox container must have ``/var/run/docker.sock``
#     mounted (or ``DOCKER_HOST`` set) so that OpenHands can start its
#     runtime container.

#     Args:
#         workspace: Path to the per-rollout working directory.
#         base_url: Proxied LiteLLM URL (contains metadata slug for session
#                   tracking).  OpenHands LLM calls are routed here.
#         instruction: Task instruction text.

#     Returns:
#         Agent's final output text.
#     """
#     # try:
#     from openhands.core.config import AppConfig, LLMConfig  # type: ignore[import]
#     from openhands.core.config import SandboxConfig as OHSandboxConfig  # type: ignore[import]
#     from openhands.headless import run_openhands  # type: ignore[import]
#     # except ImportError:
#     #     logger.warning(
#     #         "OpenHands not installed. Falling back to no-op. "
#     #         "Install with: pip install openhands-ai"
#     #     )
#     #     return "OpenHands not available"

#     runtime_image = os.environ.get("OPENHANDS_RUNTIME_IMAGE", _DEFAULT_RUNTIME_IMAGE)
#     model_name = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")

#     # Optionally override the Docker socket path (e.g. when running inside
#     # a container that has /var/run/docker.sock mounted).
#     docker_host = os.environ.get("DOCKER_HOST")

#     llm_cfg = LLMConfig(
#         model=model_name,
#         base_url=base_url,
#         api_key="EMPTY",
#     )

#     # OpenHands SandboxConfig (distinct from rllm's SandboxConfig) controls
#     # the runtime container that OpenHands uses for code execution.
#     oh_sandbox_cfg = OHSandboxConfig(
#         runtime_container_image=runtime_image,
#         user_id=_SANDBOX_USER_ID,
#         **({} if docker_host is None else {"docker_url": docker_host}),
#     )

#     app_cfg = AppConfig(
#         workspace_base=workspace,
#         max_iterations=_MAX_ITERATIONS,
#         headless_mode=True,
#         # "docker": OpenHands spawns an isolated runtime container per task.
#         # This requires /var/run/docker.sock (or DOCKER_HOST) to be accessible.
#         runtime="docker",
#         sandbox=oh_sandbox_cfg,
#     )
#     app_cfg.set_llm_config(llm_cfg)

#     logger.info(
#         "[openhands] Starting agent (runtime=docker, image=%s, workspace=%s)",
#         runtime_image,
#         workspace,
#     )
#     result = run_openhands(
#         task_str=instruction,
#         config=app_cfg,
#     )
#     output_text = getattr(result, "final_output", "") or str(result)
#     return output_text


# def _run_openhands(workspace: str, base_url: str, instruction: str) -> str:
#     """Run OpenHands headless with its own Docker runtime container."""
#     import sys
#     import asyncio
#     import os

#     print("\n[DEBUG] === Enter _run_openhands ===", file=sys.stderr)
    
#     # 1. 拆除 try...except！如果沙箱里没装包，让它立刻崩溃爆红！
#     # from openhands.core.config import AppConfig, LLMConfig
#     # from openhands.core.config import SandboxConfig as OHSandboxConfig
#     # from openhands.headless import run_openhands
    
#     from pydantic import SecretStr
#     from openhands.core.config.openhands_config import OpenHandsConfig
#     from openhands.core.config.llm_config import LLMConfig
#     from openhands.core.config.sandbox_config import SandboxConfig
#     from openhands.core.main import run_controller, generate_sid
#     from openhands.events.action import MessageAction

#     runtime_image = os.environ.get("OPENHANDS_RUNTIME_IMAGE", "docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik")
#     model_name = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")
#     docker_host = os.environ.get("DOCKER_HOST")

#     print(f"[DEBUG] Setup: runtime_image={runtime_image}, base_url={base_url}", file=sys.stderr)

#     llm_cfg = LLMConfig(
#         model=model_name,
#         base_url=base_url,
#         api_key=SecretStr("EMPTY"),
#     )

#     oh_sandbox_cfg = SandboxConfig(
#         runtime_container_image=runtime_image,
#         user_id=_SANDBOX_USER_ID, # _SANDBOX_USER_ID 替换成你的实际变量
#     )

#     app_cfg = OpenHandsConfig(
#         workspace_base=workspace,
#         max_iterations=_MAX_ITERATIONS, # _MAX_ITERATIONS 替换成你的实际变量
#         # headless_mode=True,
#         runtime="docker",
#         sandbox=oh_sandbox_cfg,
#     )
#     app_cfg.set_llm_config(llm_cfg)

#     print("[DEBUG] Calling run_openhands...", file=sys.stderr)
    
#     # 2. 防御异步陷阱：如果是 async 函数，就用 asyncio.run 来跑
#     async def _async_run():
#         state = await run_controller(
#             config=app_cfg,
#             initial_user_action=MessageAction(content=instruction),
#             sid=generate_sid(app_cfg),
#             headless_mode=True,
#         )
#         return state

#     try:
#         final_state = asyncio.run(_async_run())
#         # Extract final output from state if available
#         output_text = "Task completed" # Default fallback
#         if final_state and hasattr(final_state, 'outputs'):
#              output_text = str(final_state.outputs)
#         return output_text
#     except Exception as e:
#         logger.exception("[openhands] Error during run_controller execution")
#         return f"Error: {str(e)}"
    
#     # 3. 强制打印 result 到底是个什么东西
#     print(f"[DEBUG] run_openhands finished. Raw result type: {type(result)}", file=sys.stderr)
#     print(f"[DEBUG] Raw result: {result}", file=sys.stderr)

    
#     print("\n[DEBUG] === Exit _run_openhands ===", file=sys.stderr)
#     return output_text


def _run_openhands(workspace: str, base_url: str, instruction: str) -> str:
    """Run OpenHands headless with its own Docker runtime container."""
    import sys
    import asyncio
    import os
    import traceback
    import threading
    import signal
    print("\n[DEBUG] === Enter _run_openhands ===", file=sys.stderr)
    
    from pydantic import SecretStr
    from openhands.core.config import AppConfig, LLMConfig, SandboxConfig, finalize_config
    from openhands.core.main import run_controller, generate_sid
    # from openhands.events.action import MessageAction

    runtime_image = os.environ.get("OPENHANDS_RUNTIME_IMAGE", "docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik")
    model_name = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")
    # docker_host = os.environ.get("DOCKER_HOST")

    # 【核心护盾】：骗过 OpenHands，防止它在子线程里注册信号报错
    if threading.current_thread() is not threading.main_thread():
        print("[DEBUG] Patching signal.signal for background thread execution", file=sys.stderr)
        # 将 signal.signal 替换为一个什么都不做的假函数
        signal.signal = lambda sig, handler: None
    print(f"[DEBUG] Setup: runtime_image={runtime_image}, base_url={base_url}", file=sys.stderr)

    llm_cfg = LLMConfig(
        model=model_name,
        base_url=base_url,
        api_key="EMPTY",
    )

    oh_sandbox_cfg = SandboxConfig(
        runtime_container_image=runtime_image,
        user_id=_SANDBOX_USER_ID, # _SANDBOX_USER_ID 替换成你的实际变量
    )

    app_cfg = AppConfig(
        workspace_base=workspace,
        max_iterations=_MAX_ITERATIONS, # _MAX_ITERATIONS 替换成你的实际变量
        runtime="eventstream",
        sandbox=oh_sandbox_cfg,
    )
    app_cfg.set_llm_config(llm_cfg)
    finalize_config(app_cfg)

    print("[DEBUG] Calling run_openhands...", file=sys.stderr)
    
    async def _async_run():
        state = await run_controller(
            config=app_cfg,
            task_str=instruction,
            sid=generate_sid(app_cfg),
            headless_mode=True,
        )
        return state

    try:
        final_state = asyncio.run(_async_run())
        
        # 3. 强制打印 final_state 到底是个什么东西 (移到了 return 前面)
        print(f"[DEBUG] run_openhands finished. Raw result type: {type(final_state)}", file=sys.stderr)
        print(f"[DEBUG] Raw result: {final_state}", file=sys.stderr)

        # Extract final output from state if available
        output_text = "Task completed" # Default fallback
        if final_state and hasattr(final_state, 'outputs'):
             output_text = str(final_state.outputs)
             
        print("\n[DEBUG] === Exit _run_openhands ===", file=sys.stderr)
        return output_text

    except Exception as e:
        # 拆除消音器！一旦出错，让它在屏幕上炸开，并强行阻断程序！
        print("\n\n" + "="*50, file=sys.stderr)
        print("💥 OPENHANDS CONTROLLER CRASHED!!! 💥", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("="*50 + "\n\n", file=sys.stderr)
        raise e  # 必须抛出异常，否则又会被框架当成空轨迹吃掉！
# ---------------------------------------------------------------------------
# Sandbox rollout entry point
# ---------------------------------------------------------------------------

def rollout(task: dict = None, config: dict = None, **kwargs) -> list[dict[str, object]]:
    """Sandbox rollout entry point for NPU operator generation."""
    
    # 1. 如果框架没有传 task，我们就把所有 kwargs 当作 task 数据聚合起来
    if task is None:
        task = kwargs
        
    # 2. 如果框架没有传 config，我们给个空字典，防止后面的 .get() 报错
    if config is None:
        # 有些框架会把 config 塞在 kwargs 里，捞一把试试
        config = kwargs.get("config", {})

    # 注意：这里改成 4001，因为你前面把 proxy 端口改成了 4001！
    base_url: str = config.get("base_url", "http://127.0.0.1:4001/v1")
    # 把 127.0.0.1 换成 host.docker.internal 
    # base_url: str = config.get("base_url", "http://host.docker.internal:4001/v1")
    
    # 3. 双保险获取 instruction，并确保 task 字典里有它（防后续函数报错）
    instruction: str = kwargs.get("instruction") or task.get("instruction", "Implement a simple vector_add NPU kernel.")
    task["instruction"] = instruction

    workspace = _setup_workspace(task)

    try:
        # Run OpenHands agent with its own Docker runtime container
        agent_output = _run_openhands(
            workspace=workspace,
            base_url=base_url,
            instruction=instruction,
        )
        import sys
        print(f"\n[DEBUG] _run_openhands output: {agent_output}\n", file=sys.stderr)

        # Evaluate the result and compute reward using NPU-specific logic
        reward = _default_reward(task, workspace, agent_output)
        logger.info(
            "[openhands-npu] reward=%.2f | instruction=%s",
            reward,
            instruction[:80],
        )

    except Exception as e:
        # 暴力破解法：不依赖 logger，直接写到系统的标准错误输出里
        print("\n\n" + "="*50, file=sys.stderr)
        print("💥 OPENHANDS AGENT CRASHED!!! 💥", file=sys.stderr)
        print(f"Base URL was: {base_url}", file=sys.stderr)
        print("Traceback details:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("="*50 + "\n\n", file=sys.stderr)
        
        # 强行终止当前 worker 进程，迫使整个训练立刻停止并打印 stderr
        sys.exit(1)

    finally:
        try:
            import shutil # 确保你顶部 import 了 shutil
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