#!/usr/bin/env python3
"""
rllm OpenHands entrypoint — runs inside the OpenHands Docker container.

Launched by openhands_agent.py (rllm side) via ``docker run``.
Uses the new OpenHands Python SDK (LLM, Agent, Conversation, Tool) directly.
No inner sandbox is created — the container itself IS the execution environment.

Configuration via environment variables (set by openhands_agent.py):

    LLM_BASE_URL        Proxied rllm LiteLLM URL with embedded metadata slug.
                        All LLM calls route here for rllm session tracking.
    LLM_API_KEY         API key for the proxy (default: EMPTY)
    LLM_MODEL           Model name on LiteLLM proxy
                        (default: openai/openhands-model)
    TASK_INSTRUCTION    Task text; falls back to reading INSTRUCTIONS.md
    WORKSPACE_BASE      Workspace directory (default: /opt/workspace)
    MAX_ITERATIONS      Max agent iterations (default: 30)

Exit codes:
    0   Completed
    1   Fatal error
"""

from __future__ import annotations

import os
import sys

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, AgentContext, Conversation, get_logger
from openhands.sdk.context import Skill
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Read configuration from environment
# ---------------------------------------------------------------------------

LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "openai/openhands-model")
WORKSPACE_BASE: str = os.environ.get("WORKSPACE_BASE", "/opt/workspace")
MAX_ITERATIONS: int = int(os.environ.get("MAX_ITERATIONS", "30"))

# Task instruction: env var takes priority, then fall back to INSTRUCTIONS.md
TASK_INSTRUCTION: str = os.environ.get("TASK_INSTRUCTION", "")
if not TASK_INSTRUCTION:
    _md = os.path.join(WORKSPACE_BASE, "INSTRUCTIONS.md")
    if os.path.exists(_md):
        with open(_md) as _f:
            _lines = [l for l in _f.read().splitlines()
                      if l.strip() and not l.startswith("#")]
        TASK_INSTRUCTION = "\n".join(_lines).strip()

if not LLM_BASE_URL:
    logger.error("LLM_BASE_URL is not set. Exiting.")
    sys.exit(1)

if not TASK_INSTRUCTION:
    logger.error("No task instruction provided. Exiting.")
    sys.exit(1)

logger.info("LLM_BASE_URL : %s...", LLM_BASE_URL[:80])
logger.info("LLM_MODEL    : %s", LLM_MODEL)
logger.info("WORKSPACE    : %s", WORKSPACE_BASE)
logger.info("MAX_ITER     : %d", MAX_ITERATIONS)
logger.info("TASK         : %.120s", TASK_INSTRUCTION)


# ---------------------------------------------------------------------------
# Build OpenHands SDK objects
# ---------------------------------------------------------------------------

# LLM — LLM_BASE_URL already contains the rllm metadata slug, so every
# call the agent makes is automatically attributed to this training session.
llm = LLM(
    usage_id="rllm-openhands",
    model=LLM_MODEL,
    api_key=SecretStr(LLM_API_KEY),
    base_url=LLM_BASE_URL if LLM_BASE_URL else None,
    max_output_tokens=MAX_ITERATIONS * 2048,  # generous upper bound
)

# AgentContext — inject a skill that scopes the agent to the task
agent_context = AgentContext(
    skills=[
        Skill(
            name="task_scope",
            content=(
                "You are a software engineering agent. "
                "Complete the task described in the TASK section. "
                "Work inside the provided workspace directory. "
                "When done, summarize what you accomplished."
            ),
            trigger=None,  # always active
        )
    ],
    system_message_suffix=(
        f"Workspace directory: {WORKSPACE_BASE}. "
        f"Maximum iterations budget: {MAX_ITERATIONS}."
    ),
)

# Agent with standard coding tools.
# No sandbox / DockerWorkspace — the container itself is the sandbox.
agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
    ],
    agent_context=agent_context,
)

# Conversation rooted in the workspace (mounted from the host)
conversation = Conversation(
    agent=agent,
    workspace=WORKSPACE_BASE,
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = 0
    try:
        conversation.send_message(TASK_INSTRUCTION)
        conversation.run()

        # Log cost for observability
        if llm.metrics is not None:
            cost = llm.metrics.accumulated_cost
            logger.info("EXAMPLE_COST: %s", cost)

        logger.info("Conversation completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        exit_code = 1
    except Exception:
        logger.exception("Unhandled exception in entrypoint.")
        exit_code = 1

    sys.exit(exit_code)
