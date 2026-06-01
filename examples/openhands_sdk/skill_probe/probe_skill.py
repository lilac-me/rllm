#!/usr/bin/env python3
"""
probe_skill.py — minimal control experiment for OpenHands skill invocation.

Goal: prove that, with a skill registered the NORMAL way
(load_project_skills -> AgentContext -> <available_skills>), a model
  (a) knows WHEN to consult a skill, and
  (b) is ABLE to read the skill's SKILL.md and the files it points to.

It builds a throwaway workspace with ONE skill, "secret-keeper", whose:
  - SKILL.md       body holds a random PRIMARY code + a pointer to references/codes.md
  - references/codes.md  holds a random BACKUP code
Then it asks the model for BOTH codes without revealing them or their location.
Because the codes are random tokens, a correct answer can ONLY come from the
model actually reading the files — no guessing, no training-data leak.

Wiring is copied from the production container driver
examples/openhands_sdk/workspace/rllm_entrypoint/runner.py:
  LLM -> AgentContext(skills=load_project_skills(ws)) -> Agent(Terminal+FileEditor)
  -> Conversation(callbacks=[...]) -> send_message -> run

Two modes (env PROBE_MODE):
  implicit  (default)  AGENTS.md does NOT mention the skill. The skill is only
                       discoverable via the <available_skills> listing the SDK
                       injects. Tests JUDGMENT: does the model proactively use it?
  explicit             AGENTS.md orders the model to read the SKILL.md first.
                       Tests pure CAPABILITY: can it complete the read chain?

Verdict (tokens counted ONLY from the model's own assistant text, never from
tool observations, so a file echo can't false-pass):
  PASS     both codes reported  -> knew-when + able + references reachable
  PARTIAL  only PRIMARY reported -> read SKILL.md but didn't follow to references
  FAIL     neither              -> didn't use the skill (try PROBE_MODE=explicit)

Run inside the worker image (real SDK), pointing LLM at any reachable endpoint:
  docker run --rm -it \
    -e LLM_BASE_URL="http://host.docker.internal:5000/v1" \
    -e LLM_MODEL="openai/openhands-model" \
    -e LLM_API_KEY="EMPTY" \
    -e PROBE_MODE=implicit \
    --add-host host.docker.internal:host-gateway \
    -v "/ABS/PATH/TO/skill_probe:/probe" \   # host dir holding this script
    --entrypoint python \
    openhands-triton-env:1 /probe/probe_skill.py

Or on any host that has openhands-sdk installed:
  LLM_BASE_URL=... LLM_MODEL=... LLM_API_KEY=... python probe_skill.py

Tip: for a clean positive baseline, first run with a capable instruct model at
temperature 0 (PROBE_TEMPERATURE=0). Then re-run with your actual actor model to
compare. A weak/early-RL checkpoint failing does NOT mean the mechanism is broken.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

from pydantic import SecretStr


# --------------------------------------------------------------------------- #
# Config (env names mirror runner.py)
# --------------------------------------------------------------------------- #
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/openhands-model")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MODE = os.environ.get("PROBE_MODE", "implicit").strip().lower()
MAX_ITER = int(os.environ.get("PROBE_MAX_ITERATIONS", "15"))
TEMPERATURE = float(os.environ.get("PROBE_TEMPERATURE", "0.0"))
WS = Path(
    os.environ.get("PROBE_WS", os.path.join(tempfile.gettempdir(), "skill_probe_ws"))
).resolve()

if MODE not in ("implicit", "explicit"):
    print(f"[probe] bad PROBE_MODE={MODE!r}; use 'implicit' or 'explicit'", file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------- #
# 1. Build a throwaway workspace with one registered skill
# --------------------------------------------------------------------------- #
PRIMARY = "PRIMARY-" + secrets.token_hex(4).upper()
BACKUP = "BACKUP-" + secrets.token_hex(4).upper()

skill_dir = WS / ".agents" / "skills" / "secret-keeper"
refs_dir = skill_dir / "references"
refs_dir.mkdir(parents=True, exist_ok=True)

# SKILL.md: valid AgentSkills frontmatter (name == dir name). PRIMARY lives in the
# body; the BACKUP is deliberately NOT here — only a pointer to references/codes.md.
(skill_dir / "SKILL.md").write_text(
    "---\n"
    "name: secret-keeper\n"
    "description: >\n"
    "  Provides deployment/access codes. Use this skill whenever the user asks for a\n"
    "  deployment code, access code, credential, or secret. It gives the PRIMARY code\n"
    "  directly and tells you where to find the BACKUP code.\n"
    "---\n\n"
    "# Secret Keeper\n\n"
    "Follow these steps when the user needs deployment credentials:\n\n"
    f"1. The PRIMARY deployment code is: `{PRIMARY}`\n"
    "2. The BACKUP deployment code is NOT in this file. Read the reference file\n"
    "   `.agents/skills/secret-keeper/references/codes.md` (relative to the workspace\n"
    "   root) and copy the BACKUP code from there.\n"
    "3. Report BOTH codes back to the user, exactly as written.\n",
    encoding="utf-8",
)

(refs_dir / "codes.md").write_text(
    "# Backup codes (reference)\n\n"
    f"The BACKUP deployment code is: `{BACKUP}`\n\n"
    "Report it exactly as written.\n",
    encoding="utf-8",
)

# AGENTS.md is always injected (loaded as the third-party 'agents' skill -> REPO_CONTEXT).
if MODE == "implicit":
    # Says nothing about the skill: the ONLY pointer the model gets is the
    # <available_skills> listing the SDK injects. This tests "knows when".
    agents_md = (
        "# Workspace\n\n"
        "You are a helpful assistant. Use the available tools and skills when "
        "relevant to answer the user's request accurately. Never invent values.\n"
    )
else:
    # Explicitly orders skill use (like the AscendC AGENTS.md '重要：Skill 使用方法'
    # block). This tests pure capability: can the model complete the read chain?
    agents_md = (
        "# Workspace\n\n"
        "You are a helpful assistant.\n\n"
        "## How to use skills\n"
        "The skills listed in <available_skills> are guidance documents, not callable "
        "tools. Before answering, if a skill is relevant you MUST read its SKILL.md "
        "with the file_editor `view` command, then follow its instructions step by "
        "step — including reading any file it points to under references/.\n"
    )
(WS / "AGENTS.md").write_text(agents_md, encoding="utf-8")

# Minimal system prompt file (runner.py passes one via system_prompt_filename).
sys_prompt_path = WS / "_system_prompt.txt"
sys_prompt_path.write_text(
    "You are a helpful assistant. Use the provided tools to complete the task. "
    "Be concise.\n",
    encoding="utf-8",
)

TASK = (
    "I'm deploying service ATLAS to production and need the deployment credentials. "
    "Please report the PRIMARY deployment code and the BACKUP deployment code. "
    "Do not make up values — use whatever resources are available to you."
)


# --------------------------------------------------------------------------- #
# 2. Build SDK objects (same wiring as runner.py)
# --------------------------------------------------------------------------- #
try:
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.sdk.context.skills import load_project_skills
    from openhands.sdk.tool import Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool
except Exception as exc:  # pragma: no cover
    print(
        "[probe] could not import openhands SDK — run this INSIDE the worker image "
        f"(openhands-triton-env:1) or a host with openhands-sdk installed.\n  {exc!r}",
        file=sys.stderr,
    )
    sys.exit(3)

llm = LLM(
    usage_id="skill-probe",
    model=LLM_MODEL,
    api_key=SecretStr(LLM_API_KEY),
    base_url=LLM_BASE_URL or None,
    temperature=TEMPERATURE,
    max_output_tokens=2048,
)

skills = load_project_skills(work_dir=str(WS))

print("=" * 72)
print(f"PROBE_MODE        : {MODE}")
print(f"workspace         : {WS}")
print(f"LLM_MODEL         : {LLM_MODEL}")
print(f"LLM_BASE_URL      : {LLM_BASE_URL or '(none / provider default)'}")
print(f"temperature       : {TEMPERATURE}")
print(f"PRIMARY (SKILL.md): {PRIMARY}")
print(f"BACKUP  (ref)     : {BACKUP}")
print("-" * 72)
print(f"registered skills : {[(s.name, s.is_agentskills_format, s.source) for s in skills]}")

agent_context = AgentContext(
    skills=skills,
    load_public_skills=False,
    system_message_suffix=f"Workspace directory: {WS}.",
)

# Show the model EXACTLY what the SDK injects about skills (proves registration).
try:
    suffix = agent_context.get_system_message_suffix(llm_model=LLM_MODEL)
    print("-" * 72)
    print("system_message_suffix the model receives (skills block included):")
    print(suffix)
except Exception as exc:
    print(f"[probe] could not render system_message_suffix: {exc!r}")
print("=" * 72)

agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
    agent_context=agent_context,
    system_prompt_filename=str(sys_prompt_path),
)


# --------------------------------------------------------------------------- #
# 3. Trajectory capture (grounded in events.py)
#    - filenames read: ONLY from the model's own ActionEvents
#    - tokens reported: ONLY from the model's assistant text (never observations)
# --------------------------------------------------------------------------- #
read_files: set[str] = set()
assistant_text: list[str] = []
trajectory: list[str] = []


def on_event(event) -> None:
    etype = type(event).__name__
    try:
        raw_str = json.dumps(event.model_dump(), default=str, ensure_ascii=False)
    except Exception:
        raw_str = ""

    summary = etype
    if etype == "ActionEvent":
        tool = getattr(event, "tool_name", "")
        thought = " ".join(getattr(t, "text", "") for t in getattr(event, "thought", []) or [])
        # which skill files did the MODEL choose to open?
        for fn in ("secret-keeper/SKILL.md", "references/codes.md", "SKILL.md", "codes.md"):
            if fn in raw_str:
                read_files.add(fn)
        if thought:
            assistant_text.append(thought)  # pre-read thoughts won't contain post-read tokens
        summary = f"Action[{tool}] {thought[:140]}"
    elif etype == "MessageEvent":
        msg = getattr(event, "llm_message", None)
        role = getattr(msg, "role", "?") if msg is not None else "?"
        text = ""
        if msg is not None:
            text = " ".join(getattr(c, "text", "") for c in getattr(msg, "content", []) or [])
        if role == "assistant":
            assistant_text.append(text)
        summary = f"Message[{role}] {text[:140]}"
    elif etype == "ObservationEvent":
        tool = getattr(event, "tool_name", "")
        summary = f"Observation[{tool}]"  # content intentionally NOT scanned for tokens
    elif etype == "AgentErrorEvent":
        summary = f"Error: {getattr(event, 'error', '')}"

    trajectory.append(f"  [{etype}] {summary}")
    print(trajectory[-1][:240])


conversation = Conversation(
    agent=agent,
    workspace=str(WS),
    max_iteration_per_run=MAX_ITER,
    callbacks=[on_event],
)

print("\n----- TRAJECTORY -----")
conversation.send_message(TASK)
conversation.run()


# --------------------------------------------------------------------------- #
# 4. Verdict
# --------------------------------------------------------------------------- #
answer = "\n".join(assistant_text)
read_skill = any("SKILL.md" in f for f in read_files)
read_ref = any("codes.md" in f for f in read_files)
has_primary = PRIMARY in answer
has_backup = BACKUP in answer

if has_primary and has_backup:
    verdict = "PASS  — knew WHEN to use the skill, was ABLE to read SKILL.md AND its reference"
elif has_primary:
    verdict = "PARTIAL — read SKILL.md but did NOT follow the pointer to references/"
else:
    verdict = "FAIL  — did not use the skill at all (try PROBE_MODE=explicit to isolate capability)"

print("\n" + "=" * 72)
print("RESULT")
print(f"  model opened SKILL.md      : {read_skill}   files_read={sorted(read_files)}")
print(f"  model opened references/   : {read_ref}")
print(f"  reported PRIMARY ({PRIMARY}) : {has_primary}")
print(f"  reported BACKUP  ({BACKUP}) : {has_backup}")
print(f"  VERDICT: {verdict}")
print("=" * 72)

sys.exit(0 if (has_primary and has_backup) else 1)
