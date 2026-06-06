"""Build rllm dataset tasks for Triton operator-gen rollouts on Polar (topology a).

Each task carries the per-operator Polar specs under ``metadata`` so that
``PolarRuntime.build_task_request`` forwards them verbatim (``meta['polar_runtime']`` /
``meta['polar_evaluator']``). The STATIC bits (rollout_url, agent, poll/timeout) live in the
trainer config (``rllm.remote_runtime.polar`` — see ``remote_runtime.polar.example.yaml``); only the
PER-OP bits (op_name, the ``src/{op}.py`` upload, the judge command) vary here.

Conventions are the SAME as the standalone smoke ``ProRL-Agent-Server/examples/ascend/submit_operator.py``
and are locked by ``CANNBOT_SKILLS_MIGRATION_GUIDE.md`` §B:
  workdir /opt/workspace/agent_workdir · submit output/submission/{op}_impl.py ·
  eval tools/triton_eval_pipeline.sh -> judge_out/metrics.json · reward via operator_judge (fresh judge).

Anti-cheat + no wasted RL steps: skills/pipeline are a read-only SOURCE (/opt/canonical); each
container cp's tools into its OWN writable {workdir}/tools (agent never hits a read-only write), while
the judge re-copies from the untouched source in its clean container.
"""

from __future__ import annotations

import json
from typing import Any

WORKDIR = "/opt/workspace/agent_workdir"
CANON = "/opt/canonical"  # read-only source mount (skills/ + tools/)


def _instruction(op: str) -> str:
    return (
        f"Implement a Triton operator for Ascend NPU. The reference task is at src/{op}.py. "
        f"Write your implementation as class ModelNew to output/submission/{op}_impl.py. "
        f"Use the triton-op-verifier skill (it runs tools/triton_eval_pipeline.sh) to test and iterate "
        f"until correctness passes, then optimize for speed. Do NOT edit anything under tools/. "
        f"Your kernel is scored by re-running the canonical pipeline in a clean environment."
    )


def build_operator_task(
    op_name: str,
    *,
    image: str,
    device_ids: str,
    lock_dir: str,
    skills_dir: str,
    tasks_dir: str,
    backend: str = "docker",
    task_json: bool = False,
) -> dict[str, Any]:
    """One operator -> an rllm dataset task (PolarRuntime reads metadata.polar_*).

    ``image`` is the agent image; ``skills_dir`` (host) holds skills/ + tools/; ``tasks_dir`` (host)
    holds ``{op}.py`` (+ optional ``{op}.json``). agent (harness/model) comes from the static cfg.
    """
    sub = f"output/submission/{op_name}_impl.py"
    task_src = f"src/{op_name}.py"
    judge_command = (
        f"bash tools/triton_eval_pipeline.sh --op_name {op_name} "
        f"--impl {sub} --task {task_src} --out_dir judge_out"
    )
    mk = f"mkdir -p {WORKDIR}/output/submission {WORKDIR}/judge_out"
    cp_tools = f"cp -r {CANON}/tools {WORKDIR}/tools"  # writable copy per container
    cp_agents = f"cp {CANON}/AGENTS.md {WORKDIR}/AGENTS.md"  # orchestrator into agent cwd (Claude Code reads it)
    place = [{"type": "upload_file", "source": f"{tasks_dir}/{op_name}.py", "target": f"{WORKDIR}/{task_src}"}]
    if task_json:
        place.append(
            {"type": "upload_file", "source": f"{tasks_dir}/{op_name}.json", "target": f"{WORKDIR}/src/{op_name}.json"}
        )

    runtime = {
        "backend": backend,
        "image": (f"docker-daemon:{image}" if backend == "apptainer" else image),
        "network": "host",
        "workdir": WORKDIR,
        "kwargs": {
            "ascend": {"device_ids": device_ids, "lock_dir": lock_dir},
            "volumes": [f"{skills_dir}:{CANON}:ro"],  # read-only SOURCE only
        },
        "prepare": [*place, {"type": "exec", "command": f"{mk} && {cp_tools} && {cp_agents} && command -v claude"}],
        "eval_prepare": [*place, {"type": "exec", "command": f"{mk} && {cp_tools}"}],
    }
    evaluator = {
        "strategy": "operator_judge",
        "refresh_runtime": True,  # fresh clean judge runtime (anti-cheat); inherits kwargs.ascend
        "config": {
            "op_name": op_name,
            "judge_command": judge_command,
            "submission_path": sub,
            "metrics_path": "judge_out/metrics.json",
            "workdir": WORKDIR,
        },
    }
    return {
        "op_name": op_name,
        "instruction": _instruction(op_name),
        "metadata": {"polar_runtime": runtime, "polar_evaluator": evaluator},
    }


def write_operator_dataset(ops: list[str], path: str, **kwargs: Any) -> int:
    """Write a JSONL dataset (one operator per line). Returns the count. kwargs -> build_operator_task."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for op in ops:
            f.write(json.dumps(build_operator_task(op, **kwargs), ensure_ascii=False) + "\n")
            n += 1
    return n
