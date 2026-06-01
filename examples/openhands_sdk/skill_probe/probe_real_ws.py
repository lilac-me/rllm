#!/usr/bin/env python3
r"""
probe_real_ws.py — does the model actually READ skills on a REAL task?

Unlike probe_skill.py (a synthetic 1-skill control), this points the agent at a
REAL agent_workdir (e.g. the openhands-observability triton skills + its
AGENTS.md) plus a REAL operator task, then logs which skill files the model
actually opens. It does NOT score kernel correctness — it answers one question:

  under the real AGENTS.md + real skills, does the model read
  .agents/skills/<skill>/SKILL.md and their references/ — or does it skip them,
  or try to "call" a skill as a tool (the misfire we saw in probe_skill.py)?

Wiring matches the production container driver runner.py:
  LLM -> AgentContext(skills=load_project_skills(ws)) -> Agent[Terminal+FileEditor]
      -> Conversation -> send_message -> run

The real agent_workdir is COPIED to a scratch dir first (the agent writes files),
and a small KernelBench-style reference op is seeded into src/ so there is a
concrete task. The operator pipeline will likely fail without NPU — that's fine;
we only care about whether skills get read in the early trajectory.

Env:
  LLM_BASE_URL / LLM_MODEL / LLM_API_KEY    same as probe_skill.py
  PROBE_SRC_WORKSPACE   REQUIRED. Path to the real agent_workdir to test, e.g.
                        the openhands-observability checkout's
                        examples/openhands_sdk/workspace/agent_workdir
  PROBE_OP_NAME         default "softmax"
  PROBE_TASK            optional task override (else built from the op name)
  PROBE_RUN_DIR         scratch copy dir, default /tmp/probe_real_ws
  PROBE_MAX_ITERATIONS  default 30
  PROBE_TEMPERATURE     default 0.0

Run inside the worker image (real SDK), e.g. with the observability workdir:
  docker run --rm -it --network host \
    -e LLM_BASE_URL="http://127.0.0.1:8003/v1" -e LLM_MODEL="openai/qwen3" \
    -e LLM_API_KEY="EMPTY" -e LITELLM_LOCAL_MODEL_COST_MAP=True \
    -e PROBE_SRC_WORKSPACE=/real_ws -e PROBE_OP_NAME=softmax \
    -v /abs/openhands-observability/.../agent_workdir:/real_ws \
    -v /ABS/PATH/TO/skill_probe:/probe \
    --entrypoint python openhands-triton-env:v1 /probe/probe_real_ws.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from pydantic import SecretStr


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/qwen3")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
OP_NAME = os.environ.get("PROBE_OP_NAME", "softmax")
MAX_ITER = int(os.environ.get("PROBE_MAX_ITERATIONS", "30"))
TEMPERATURE = float(os.environ.get("PROBE_TEMPERATURE", "0.0"))
SRC_WS = os.environ.get("PROBE_SRC_WORKSPACE", "")
RUN_DIR = Path(os.environ.get("PROBE_RUN_DIR", os.path.join(tempfile.gettempdir(), "probe_real_ws"))).resolve()

if not SRC_WS or not Path(SRC_WS).is_dir():
    print(f"[probe] set PROBE_SRC_WORKSPACE to a real agent_workdir (got {SRC_WS!r})", file=sys.stderr)
    sys.exit(2)

# Fresh copy of the real workspace so the agent can write into it.
if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)
shutil.copytree(SRC_WS, RUN_DIR)

# Seed a KernelBench-style reference op so there is a concrete task.
src_dir = RUN_DIR / "src"
src_dir.mkdir(parents=True, exist_ok=True)
op_file = src_dir / f"{OP_NAME}.py"
if not op_file.exists():
    op_file.write_text(
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        "        return torch.softmax(x, dim=-1)\n\n"
        "def get_inputs():\n"
        "    return [torch.randn(1024, 1024)]\n\n"
        "def get_init_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )

TASK = os.environ.get("PROBE_TASK", "").strip() or (
    f"任务：为算子 `{OP_NAME}` 生成 Triton-Ascend 实现。参考实现见 `src/{OP_NAME}.py`"
    f"（含 Model / get_inputs / get_init_inputs）。请严格按 AGENTS.md 的工作流执行，"
    f"在 `src/{OP_NAME}_triton_ascend_impl.py` 中实现 `ModelNew`，并运行 "
    f"`bash tools/operator_pipeline.sh --op_name {OP_NAME}` 验证，迭代修复直到 metrics.json 报 success。"
)

# Minimal system prompt file (runner.py passes one).
sys_prompt_path = RUN_DIR / "_system_prompt.txt"
sys_prompt_path.write_text(
    "You are a helpful assistant. Use the provided tools to complete the task. Be concise.\n",
    encoding="utf-8",
)

try:
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.sdk.context.skills import load_project_skills
    from openhands.sdk.tool import Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool
except Exception as exc:  # pragma: no cover
    print(f"[probe] run inside openhands image; SDK import failed: {exc!r}", file=sys.stderr)
    sys.exit(3)

llm = LLM(
    usage_id="skill-probe-realws",
    model=LLM_MODEL,
    api_key=SecretStr(LLM_API_KEY),
    base_url=LLM_BASE_URL or None,
    temperature=TEMPERATURE,
    max_output_tokens=2048,
)

skills = load_project_skills(work_dir=str(RUN_DIR))
skill_names = [s.name for s in skills]

print("=" * 72)
print(f"workspace (copy)  : {RUN_DIR}")
print(f"src op            : src/{OP_NAME}.py")
print(f"LLM_MODEL         : {LLM_MODEL}   base_url={LLM_BASE_URL or '(provider default)'}   T={TEMPERATURE}")
print(f"registered skills : {[(s.name, s.is_agentskills_format, s.source) for s in skills]}")
agent_context = AgentContext(skills=skills, load_public_skills=False,
                             system_message_suffix=f"Workspace directory: {RUN_DIR}.")
try:
    print("-" * 72)
    print("available_skills block the model receives:")
    print(agent_context.get_system_message_suffix(llm_model=LLM_MODEL))
except Exception as exc:
    print(f"[probe] suffix render failed: {exc!r}")
print("=" * 72)

agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
    agent_context=agent_context,
    system_prompt_filename=str(sys_prompt_path),
)

# ── Trajectory capture ──────────────────────────────────────────────────────
skill_md_reads: set[str] = set()
reference_reads: set[str] = set()
skill_tool_misfires: list[str] = []
impl_created = False
pipeline_run = False
action_log: list[str] = []


def on_event(event) -> None:
    global impl_created, pipeline_run
    etype = type(event).__name__
    try:
        raw = json.dumps(event.model_dump(), default=str, ensure_ascii=False)
    except Exception:
        raw = ""

    if etype == "ActionEvent":
        tool = getattr(event, "tool_name", "")
        # which skill's SKILL.md did the model open?
        for n in skill_names:
            if f"{n}/SKILL.md" in raw or (n in raw and "SKILL.md" in raw):
                skill_md_reads.add(n)
        if "/references/" in raw or "references/" in raw:
            for tok in raw.replace('\\', '/').split('"'):
                if "references/" in tok and tok.endswith(".md"):
                    reference_reads.add(tok.split("references/")[-1])
        if f"{OP_NAME}_triton_ascend_impl.py" in raw and '"create"' in raw:
            impl_created = True
        if "operator_pipeline.sh" in raw:
            pipeline_run = True
        thought = " ".join(getattr(t, "text", "") for t in getattr(event, "thought", []) or [])
        action_log.append(f"Action[{tool}] {thought[:120]}")
        print(f"  [Action:{tool}] {thought[:160]}")
    elif etype == "AgentErrorEvent":
        err = str(getattr(event, "error", "")) or raw
        if "not found" in err.lower():
            for n in skill_names:
                if n in err:
                    skill_tool_misfires.append(n)
        action_log.append(f"ERROR {err[:120]}")
        print(f"  [Error] {err[:180]}")
    elif etype == "MessageEvent":
        msg = getattr(event, "llm_message", None)
        role = getattr(msg, "role", "?") if msg is not None else "?"
        if role == "assistant":
            txt = " ".join(getattr(c, "text", "") for c in getattr(msg, "content", []) or [])
            print(f"  [Msg:assistant] {txt[:160]}")


conversation = Conversation(agent=agent, workspace=str(RUN_DIR),
                            max_iteration_per_run=MAX_ITER, callbacks=[on_event])

print("\n----- TRAJECTORY -----")
try:
    conversation.send_message(TASK)
    conversation.run()
except Exception as exc:
    print(f"\n[probe] conversation ended with exception (often expected — no NPU): {exc!r}")

# ── Behavioral summary ──────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("RESULT — did the model engage the skills on a real task?")
print(f"  skills registered          : {skill_names}")
print(f"  SKILL.md opened by model    : {sorted(skill_md_reads) or 'NONE'}")
print(f"  references opened by model  : {sorted(reference_reads) or 'NONE'}")
print(f"  skill-as-tool misfires      : {skill_tool_misfires or 'none'}")
print(f"  created impl file           : {impl_created}    ran operator_pipeline.sh: {pipeline_run}")
if skill_md_reads and reference_reads:
    obs = "READ skills AND references"
elif skill_md_reads:
    obs = "read SKILL.md but NOT references"
else:
    obs = "did NOT open any skill doc (skills not engaged under this AGENTS.md)"
print(f"  OBSERVATION: {obs}")
if skill_tool_misfires:
    print(f"  NOTE: tried to call skill(s) as a tool first: {skill_tool_misfires} "
          f"(skills are docs, not callable tools in OpenHands)")
print("=" * 72)
