"""
runner.py — Main conversation execution logic for the rllm entrypoint.

Responsibilities:
  1. Build OpenHands SDK objects (LLM, Agent, Conversation).
  2. Wire the event callback into the Conversation.
  3. Start background threads (uploader, pause controller).
  4. Run the conversation loop, honoring pause/resume signals.
  5. Report final metrics and clean up background threads.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from rllm_entrypoint.api_client import ObserverClient
from rllm_entrypoint.config import cfg
from rllm_entrypoint.events import make_event_callback
from rllm_entrypoint.pause_ctrl import PauseController
from rllm_entrypoint.state import AgentPhase, RunState
from rllm_entrypoint.uploader import StateUploader

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
# Workspace skill merging (kept here to avoid monolithic entrypoint)
# ---------------------------------------------------------------------------

def _merge_workspace_skills(workspace_base: str, task_skill: Any) -> list:
    from openhands.sdk.context import Skill
    from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir

    ws = Path(workspace_base)
    skills: list = []

    if any((ws / name).exists() for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md")):
        try:
            loaded = load_project_skills(work_dir=str(ws))
            if loaded:
                skills.extend(loaded if isinstance(loaded, list) else list(loaded))
        except Exception:
            logger.exception("load_project_skills failed")

    agents_skills_root = ws / ".agents" / "skills"
    if agents_skills_root.is_dir():
        try:
            _repo, _knowledge, agent_skills = load_skills_from_dir(str(agents_skills_root))
            skills.extend(agent_skills.values())
        except Exception:
            logger.exception("load_skills_from_dir failed")

    skills.append(task_skill)
    return skills


# ---------------------------------------------------------------------------
# Env-var snapshot (redact secrets)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _safe_env_snapshot() -> dict[str, str]:
    result = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(pat in upper for pat in _SECRET_PATTERNS):
            result[k] = "***REDACTED***"
        else:
            result[k] = v[:200]
    return result


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
    from openhands.sdk import LLM, Agent, AgentContext, Conversation, get_logger
    from openhands.sdk.context import Skill
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
    if cfg.observer_api_url:
        logger.info("OBSERVER_URL : %s", cfg.observer_api_url)
    else:
        logger.info("OBSERVER_URL : (disabled — running standalone)")

    # ── Write system prompt ───────────────────────────────────────────────
    _write_system_prompt()

    # ── Initialise RunState ───────────────────────────────────────────────
    run_state = build_run_state()

    # ── Initialise observer infrastructure ────────────────────────────────
    client = ObserverClient(
        base_url=cfg.observer_api_url,
        session_id=cfg.session_id,
    )
    uploader = StateUploader(run_state, client, interval_s=cfg.upload_interval_s)
    pause_ctrl = PauseController(run_state, client, poll_interval_s=cfg.pause_poll_interval_s)

    uploader.start()
    pause_ctrl.start()

    # ── Build event callback ──────────────────────────────────────────────
    user_event_cb = make_event_callback(run_state)

    # Wrap callback to also trigger an uploader flush on every event
    def _event_cb(event: Any) -> None:
        user_event_cb(event)
        uploader.trigger()

    # ── Build OpenHands SDK objects ───────────────────────────────────────
    llm = LLM(
        usage_id="rllm-openhands",
        model=cfg.llm_model,
        api_key=SecretStr(cfg.llm_api_key),
        base_url=cfg.llm_base_url or None,
        max_output_tokens=4096,
    )

    if cfg.npu_operator_task:
        _task_scope = (
            f"You are a custom kernel / operator agent (backend hint: {cfg.operator_backend}). "
            "Follow AGENTS.md and INSTRUCTIONS.md. Implement under src/triton/ or src/ascendc/ as directed. "
            "Do not modify tools/. Run `bash tools/operator_pipeline.sh` after changes; "
            "iterate until metrics.json reports success. Summarize results when done."
        )
    else:
        _task_scope = (
            "You are a software engineering agent. "
            "Complete the task described in the TASK section. "
            "Work inside the provided workspace directory. "
            "When done, summarize what you accomplished."
        )

    task_skill = Skill(name="task_scope", content=_task_scope, trigger=None)

    try:
        merged_skills = _merge_workspace_skills(cfg.workspace_base, task_skill)
    except Exception:
        logger.exception("merge_workspace_skills failed; falling back to task_scope only")
        merged_skills = [task_skill]

    agent_context = AgentContext(
        skills=merged_skills,
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
                while not pause_ctrl.resume_requested and not pause_ctrl._stop_event.is_set():
                    time.sleep(0.5)
                    exec_status = conversation.state.execution_status.value
                    if exec_status not in ("paused",):
                        break
                if pause_ctrl.resume_requested:
                    pause_ctrl.resume_requested = False
                    logger.info("[runner] Resuming after pause...")
                    continue
                break

            # All other (idle / waiting_for_confirmation) → done
            break

        # Log LLM cost
        if llm.metrics is not None:
            cost = llm.metrics.accumulated_cost
            run_state.update_metrics(cost=float(cost or 0.0), llm_calls=run_state.total_llm_calls)
            logger.info("EXAMPLE_COST: %s", cost)

        logger.info("Conversation completed. Status=%s", conversation.state.execution_status.value)
        run_state.set_phase(AgentPhase.FINISHED)

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
        # Stop background threads and do one final upload
        pause_ctrl.stop()
        uploader.stop()
        uploader.join(timeout=10)

    return exit_code
