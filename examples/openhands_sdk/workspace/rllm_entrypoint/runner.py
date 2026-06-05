"""
runner.py — Main conversation execution logic for the rllm entrypoint.

Responsibilities:
  1. Build OpenHands SDK objects (LLM, Agent, Conversation).
  2. Wire the event callback into the Conversation.
  3. Start background pause controller thread.
  4. Run the conversation loop, honoring pause/resume signals.
  5. Emit lifecycle events (startup / heartbeat / evaluate / finish).
  6. Report final metrics and clean up background threads.

v2 (event-driven):
  The StateUploader background thread has been removed.  Instead, each
  OpenHands SDK event is pushed to the gateway immediately inside the
  event callback.  Heartbeats are emitted from a lightweight timer thread
  so the gateway always has a liveness signal even during long LLM calls.
"""
from __future__ import annotations

import functools
import inspect
import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from rllm_entrypoint.api_client import ObserverClient
from rllm_entrypoint.config import cfg
from rllm_entrypoint.events import (
    make_event_callback,
    push_evaluate_event,
    push_finish_event,
    push_heartbeat_event,
    push_startup_event,
)
from rllm_entrypoint.pause_ctrl import PauseController
from rllm_entrypoint.state import AgentPhase, RunState

from openhands.sdk.context import Skill
from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_HARDCODED_SYSTEM_PROMPT = """You are a helpful assistant. Use the provided tools to complete the task. Be concise.
"""


def _write_system_prompt() -> None:
    os.makedirs(os.path.dirname(cfg.system_prompt_path) or "/tmp", exist_ok=True)
    with open(cfg.system_prompt_path, "w", encoding="utf-8") as f:
        f.write(_HARDCODED_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Workspace skill merging
# ---------------------------------------------------------------------------

def merge_workspace_skills(workspace_base: str, task_scope: Skill = None) -> list:
    """Merge AGENTS.md + .agents/skills/* + inline task_scope."""
    ws = Path(workspace_base)
    skills: list = []

    loaded = load_project_skills(work_dir=str(ws))
    if loaded:
        skills.extend(loaded if isinstance(loaded, list) else list(loaded))

    # agents_skills_root = ws / ".agents" / "skills"
    # if agents_skills_root.is_dir():
    #     _repo, _knowledge, agent_skills = load_skills_from_dir(str(agents_skills_root))
    #     skills.extend(agent_skills.values())

    # skills.append(task_scope)
    return skills


# ---------------------------------------------------------------------------
# Env-var snapshot
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _safe_env_snapshot() -> dict[str, str]:
    result = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(pat in upper for pat in _SECRET_PATTERNS):
            result[k] = "***REDACTED***"
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# LLM latency probe
# ---------------------------------------------------------------------------

def install_llm_latency_probe(
    llm,
    run_state: RunState | None = None,
    log_path: str | None = None,
):
    """
    Patch OpenHands SDK LLM instance to log latency for each LLM inference call.

    It tries to wrap common LLM call methods:
      - completion
      - acompletion
      - chat_completion
      - achat_completion
      - __call__

    The exact method name depends on the OpenHands SDK version.
    """

    if log_path is None:
        log_path = os.path.join(os.environ.get("RLLM_LOG_DIR", "/tmp"), f"llm_latency_{os.getpid()}.log")

    pid = os.getpid()
    rank = os.getenv("RANK", "NA")
    local_rank = os.getenv("LOCAL_RANK", "NA")
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "NA")

    call_counter = itertools.count()

    def _write_latency_log(msg: str) -> None:
        """
        Write logs both to logger and to an independent file.

        File logging is important because Ray may deduplicate stdout/stderr logs.
        """

        logger.info(msg)

        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.debug("[llm-latency] failed to write latency log", exc_info=True)

    def _summarize_args(args, kwargs) -> str:
        """
        Avoid printing full prompt/messages because they may be huge.

        Only print rough size/shape information.
        """

        parts = []

        if args:
            parts.append(f"args_len={len(args)}")

        if kwargs:
            parts.append(f"kwargs_keys={list(kwargs.keys())}")

            messages = kwargs.get("messages")
            if isinstance(messages, list):
                parts.append(f"messages_len={len(messages)}")
                try:
                    char_len = 0
                    for m in messages:
                        if isinstance(m, dict):
                            char_len += len(str(m.get("content", "")))
                        else:
                            char_len += len(str(m))
                    parts.append(f"messages_chars={char_len}")
                except Exception:
                    pass

            prompt = kwargs.get("prompt")
            if prompt is not None:
                parts.append(f"prompt_chars={len(str(prompt))}")

            tools = kwargs.get("tools")
            if tools is not None:
                try:
                    parts.append(f"tools_len={len(tools)}")
                except Exception:
                    parts.append("tools_present=True")

        return " ".join(parts)

    def _wrap_sync_method(method_name: str, orig_method):
        @functools.wraps(orig_method)
        def wrapper(*args, **kwargs):
            call_id = next(call_counter)
            start_wall = time.time()
            start_perf = time.perf_counter()

            if run_state is not None:
                try:
                    run_state.total_llm_calls += 1
                except Exception:
                    pass

            arg_summary = _summarize_args(args, kwargs)

            _write_latency_log(
                f"[LLM_CALL_START] "
                f"call_id={call_id} method={method_name} "
                f"wall={start_wall:.6f} "
                f"pid={pid} rank={rank} local_rank={local_rank} "
                f"ASCEND_RT_VISIBLE_DEVICES={visible_devices} "
                f"llm_type={type(llm).__name__} llm_id={id(llm)} "
                f"{arg_summary}"
            )

            ok = False
            try:
                ret = orig_method(*args, **kwargs)
                ok = True
                return ret
            except Exception as e:
                elapsed = time.perf_counter() - start_perf
                _write_latency_log(
                    f"[LLM_CALL_ERROR] "
                    f"call_id={call_id} method={method_name} "
                    f"elapsed_s={elapsed:.6f} "
                    f"error={repr(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise
            finally:
                elapsed = time.perf_counter() - start_perf
                end_wall = time.time()

                _write_latency_log(
                    f"[LLM_CALL_END] "
                    f"call_id={call_id} method={method_name} ok={ok} "
                    f"elapsed_s={elapsed:.6f} "
                    f"start_wall={start_wall:.6f} end_wall={end_wall:.6f} "
                    f"pid={pid} rank={rank} local_rank={local_rank}"
                )

        return wrapper

    def _wrap_async_method(method_name: str, orig_method):
        @functools.wraps(orig_method)
        async def wrapper(*args, **kwargs):
            call_id = next(call_counter)
            start_wall = time.time()
            start_perf = time.perf_counter()

            if run_state is not None:
                try:
                    run_state.total_llm_calls += 1
                except Exception:
                    pass

            arg_summary = _summarize_args(args, kwargs)

            _write_latency_log(
                f"[LLM_CALL_START] "
                f"call_id={call_id} method={method_name} "
                f"wall={start_wall:.6f} "
                f"pid={pid} rank={rank} local_rank={local_rank} "
                f"ASCEND_RT_VISIBLE_DEVICES={visible_devices} "
                f"llm_type={type(llm).__name__} llm_id={id(llm)} "
                f"{arg_summary}"
            )

            ok = False
            try:
                ret = await orig_method(*args, **kwargs)
                ok = True
                return ret
            except Exception as e:
                elapsed = time.perf_counter() - start_perf
                _write_latency_log(
                    f"[LLM_CALL_ERROR] "
                    f"call_id={call_id} method={method_name} "
                    f"elapsed_s={elapsed:.6f} "
                    f"error={repr(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise
            finally:
                elapsed = time.perf_counter() - start_perf
                end_wall = time.time()

                _write_latency_log(
                    f"[LLM_CALL_END] "
                    f"call_id={call_id} method={method_name} ok={ok} "
                    f"elapsed_s={elapsed:.6f} "
                    f"start_wall={start_wall:.6f} end_wall={end_wall:.6f} "
                    f"pid={pid} rank={rank} local_rank={local_rank}"
                )

        return wrapper

    candidate_methods = [
        "completion",
        "acompletion",
        "chat_completion",
        "achat_completion",
        "__call__",
    ]

    patched = []

    for method_name in candidate_methods:
        if not hasattr(llm, method_name):
            continue

        orig_method = getattr(llm, method_name)

        if not callable(orig_method):
            continue

        if getattr(orig_method, "_llm_latency_probe_patched", False):
            continue

        if inspect.iscoroutinefunction(orig_method):
            wrapped = _wrap_async_method(method_name, orig_method)
        else:
            wrapped = _wrap_sync_method(method_name, orig_method)

        setattr(wrapped, "_llm_latency_probe_patched", True)

        try:
            object.__setattr__(llm, method_name, wrapped)
            patched.append(method_name)
        except Exception:
            logger.exception("[llm-latency] failed to patch method: %s", method_name)

    _write_latency_log(
        f"[LLM_PROBE_INSTALLED] "
        f"pid={pid} rank={rank} local_rank={local_rank} "
        f"llm_type={type(llm)} llm_id={id(llm)} "
        f"patched_methods={patched} log_path={log_path}"
    )

    if not patched:
        candidate_attrs = []
        try:
            for name in dir(llm):
                try:
                    attr = getattr(llm, name)
                except Exception:
                    continue
                if callable(attr) and any(
                    k in name.lower()
                    for k in ["completion", "chat", "call", "response", "generate", "invoke"]
                ):
                    candidate_attrs.append(name)
        except Exception:
            pass

        _write_latency_log(
            f"[LLM_PROBE_WARNING] no method patched. "
            f"llm_type={type(llm)} "
            f"candidate_attrs={candidate_attrs}"
        )

    return patched


# ---------------------------------------------------------------------------
# Heartbeat timer
# ---------------------------------------------------------------------------

class _HeartbeatTimer:
    """
    Periodically emits a HeartbeatEvent via push_heartbeat_event().

    Uses a daemon thread so it never blocks program exit.
    """

    def __init__(
        self,
        client: ObserverClient,
        run_state: RunState,
        interval_s: float = 15.0,
    ) -> None:
        self._client = client
        self._state = run_state
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="rllm-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        if self._client._enabled:
            self._thread.start()
            logger.info("[heartbeat] Timer started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                push_heartbeat_event(self._client, self._state)
            except Exception:
                logger.debug("[heartbeat] push failed", exc_info=True)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def build_run_state() -> RunState:
    """Populate a RunState with all static config values."""
    state = RunState(
        session_id=cfg.session_id,
        session_label=cfg.session_label,
        task_instruction=cfg.task_instruction,
        workspace_base=cfg.workspace_base,
        llm_model=cfg.llm_model,
        llm_base_url=cfg.llm_base_url,
        max_iterations=cfg.max_iterations,
        ops_name=cfg.operator_name,
        ops_arch=cfg.operator_arch,
        is_running=False,
        phase=AgentPhase.INITIALIZING,
        env_vars=_safe_env_snapshot(),
        extra_metadata=_parse_extra_metadata(),
    )
    return state


def _parse_extra_metadata() -> dict[str, Any]:
    if not cfg.extra_metadata:
        return {}
    try:
        return json.loads(cfg.extra_metadata)
    except json.JSONDecodeError:
        return {"raw": cfg.extra_metadata}


def run() -> int:
    """Build SDK objects, run the conversation, return exit code."""
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.sdk.tool import Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    # ── Validate config ───────────────────────────────────────────────────
    if not cfg.llm_base_url:
        logger.error("LLM_BASE_URL is not set. Exiting.")
        return 1

    if not cfg.task_instruction:
        logger.error("No task instruction provided. Exiting.")
        return 1

    logger.info("LLM_BASE_URL : %s...", cfg.llm_base_url[:80])
    logger.info("LLM_MODEL    : %s", cfg.llm_model)
    logger.info("WORKSPACE    : %s", cfg.workspace_base)
    logger.info("MAX_ITER     : %d", cfg.max_iterations)
    logger.info("SESSION_ID   : %s", cfg.session_id)
    logger.info("TASK         : %.120s", cfg.task_instruction)
    logger.info(
        "ASCEND_RT_VISIBLE_DEVICES : %s...",
        os.getenv("ASCEND_RT_VISIBLE_DEVICES", "NA"),
    )

    if cfg.observer_api_url:
        logger.info("OBSERVER_URL : %s", cfg.observer_api_url)
    else:
        logger.info("OBSERVER_URL : (disabled — running standalone)")

    # ── Write system prompt ───────────────────────────────────────────────
    _write_system_prompt()

    # ── Initialise RunState ───────────────────────────────────────────────
    run_state = build_run_state()

    # ── Initialise observer client ────────────────────────────────────────
    client = ObserverClient(
        base_url=cfg.observer_api_url,
        session_id=cfg.session_id,
    )

    # ── Push startup lifecycle event ──────────────────────────────────────
    push_startup_event(client, run_state)

    # ── Start heartbeat timer ─────────────────────────────────────────────
    heartbeat = _HeartbeatTimer(
        client=client,
        run_state=run_state,
        interval_s=cfg.upload_interval_s,
    )
    heartbeat.start()

    # ── Start pause controller ────────────────────────────────────────────
    pause_ctrl = PauseController(
        run_state,
        client,
        poll_interval_s=cfg.pause_poll_interval_s,
    )
    pause_ctrl.start()

    # ── Build event callback ──────────────────────────────────────────────
    _event_cb = make_event_callback(run_state, client)

    # ── Build OpenHands SDK objects ───────────────────────────────────────
    llm = LLM(
        usage_id="rllm-openhands",
        model=cfg.llm_model,
        api_key=SecretStr(cfg.llm_api_key),
        base_url=cfg.llm_base_url or None,
        # Sampling temperature: 0.0 default (greedy, as before). Eval sets LLM_TEMPERATURE>0
        # so N independent rollouts diverge → pass@k is meaningful. The worker passes it via -e.
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
        # Raised 2048→8192: reasoning models (qwen3.6 CoT) spend output budget on
        # thinking before the tool call; 2048 truncated the kernel write. Env-tunable.
        max_output_tokens=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "8192")),
    )

    def debug_llm_identity(llm, agent):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pid = os.getpid()
        rank = os.getenv("RANK", "NA")
        local_rank = os.getenv("LOCAL_RANK", "NA")

        msg = (
            f"[{ts}] "
            f"[pid={pid}] "
            f"[rank={rank}] "
            f"[local_rank={local_rank}] "
            f"[input llm type={type(llm)}] "
            f"[input llm id={id(llm)}] "
            f"[agent.llm type={type(agent.llm)}] "
            f"[agent.llm id={id(agent.llm)}] "
            f"[same_obj={llm is agent.llm}]"
        )

        path = os.path.join(os.environ.get("RLLM_LOG_DIR", "/tmp"), f"openhands_{os.getpid()}.log")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{time.time()} {msg}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.debug("[debug_llm_identity] failed to write log", exc_info=True)

        logger.info(msg)

    try:
        _merged_skills = merge_workspace_skills(cfg.workspace_base)
    except Exception:
        logger.exception("merge_workspace_skills failed; falling back to empty skills")
        _merged_skills = []

    agent_context = AgentContext(
        skills=_merged_skills,
        load_public_skills=False,
        system_message_suffix=(
            f"Workspace directory: {cfg.workspace_base}. "
            f"Maximum iterations budget: {cfg.max_iterations}. "
            f"Session ID: {cfg.session_id}."
        ),
    )

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ],
        agent_context=agent_context,
        system_prompt_filename=cfg.system_prompt_path,
    )

    debug_llm_identity(llm, agent)

    # ── Install LLM latency probe ─────────────────────────────────────────
    #
    # Important:
    #   Patch agent.llm instead of only llm, because Agent may internally keep,
    #   copy, or wrap the LLM object depending on OpenHands SDK version.
    #
    install_llm_latency_probe(
        agent.llm,
        run_state=run_state,
        log_path=os.path.join(os.environ.get("RLLM_LOG_DIR", "/tmp"), f"llm_latency_{cfg.session_id}_{os.getpid()}.log"),
    )

    conversation = Conversation(
        agent=agent,
        workspace=cfg.workspace_base,
        max_iteration_per_run=cfg.max_iterations,
        callbacks=[_event_cb],
    )

    # Wire pause controller to the live conversation object
    pause_ctrl.set_conversation(conversation)

    # Reflect conversation ID in state
    run_state.set_conversation_status("idle", str(conversation.id))

    # ── Run ───────────────────────────────────────────────────────────────
    exit_code = 0
    run_state.set_running(True)
    run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)

    try:
        conversation.send_message(cfg.task_instruction)

        while True:
            run_state.set_conversation_status(
                conversation.state.execution_status.value,
                str(conversation.id),
            )
            run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)

            conversation.run()

            # Check if resume was requested by pause controller
            if pause_ctrl.resume_requested:
                logger.info("[runner] Resume requested — re-entering conversation.run()")
                pause_ctrl.resume_requested = False
                continue

            # Check conversation terminal status
            exec_status = conversation.state.execution_status.value
            run_state.set_conversation_status(exec_status, str(conversation.id))

            if exec_status in ("finished", "error", "stuck"):
                logger.info("[runner] Conversation reached terminal status: %s", exec_status)
                break

            # Paused but no resume pending: wait for resume signal
            if exec_status == "paused":
                logger.info("[runner] Conversation paused. Waiting for resume signal...")

                while (
                    not pause_ctrl.resume_requested
                    and not pause_ctrl._stop_event.is_set()
                ):
                    time.sleep(0.5)
                    exec_status = conversation.state.execution_status.value

                    if exec_status not in ("paused",):
                        break

                if pause_ctrl.resume_requested:
                    pause_ctrl.resume_requested = False
                    logger.info("[runner] Resuming after pause...")
                    continue

                break

            # All other statuses, for example idle / waiting_for_confirmation
            break

        # Log LLM cost
        cost = 0.0
        llm_calls = run_state.total_llm_calls

        if llm.metrics is not None:
            cost = float(llm.metrics.accumulated_cost or 0.0)
            run_state.update_metrics(cost=cost, llm_calls=llm_calls)
            logger.info("EXAMPLE_COST: %s", cost)

        exec_status = conversation.state.execution_status.value
        logger.info("Conversation completed. Status=%s", exec_status)
        run_state.set_phase(AgentPhase.FINISHED)

        # Eval calibration aid: persist turn/iteration counts so the eval driver can
        # size --max-iterations (read back from agent_workdir/run_meta.json).
        try:
            _calls = int(getattr(run_state, "total_llm_calls", 0))
            _max = int(cfg.max_iterations)
            with open(os.path.join(cfg.workspace_base, "run_meta.json"), "w", encoding="utf-8") as _mf:
                json.dump({"total_llm_calls": _calls, "iterations": int(getattr(run_state, "iteration", 0)),
                           "max_iterations": _max, "exec_status": exec_status,
                           "hit_cap": _calls >= _max}, _mf, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("[runner] failed to write run_meta.json", exc_info=True)

        # ── Evaluation event ──────────────────────────────────────────────
        push_evaluate_event(
            client,
            run_state,
            status=exec_status,
            cost=cost,
            llm_calls=llm_calls,
            iterations=run_state.iteration,
        )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        run_state.set_phase(AgentPhase.ERROR)
        run_state.set_error("KeyboardInterrupt")
        exit_code = 1

    except Exception as exc:
        logger.exception("Unhandled exception in runner.")
        run_state.set_phase(AgentPhase.ERROR)
        run_state.set_error(str(exc))
        exit_code = 1

    finally:
        run_state.set_running(False)

        # ── Finish event ─────────────────────────────────────────────────
        try:
            push_finish_event(
                client,
                run_state,
                exit_code=exit_code,
                reason="error" if exit_code != 0 else "completed",
            )
        except Exception:
            logger.debug("[runner] push_finish_event failed", exc_info=True)

        # ── Stop background threads ───────────────────────────────────────
        heartbeat.stop()
        pause_ctrl.stop()
        heartbeat.join(timeout=5)

    return exit_code