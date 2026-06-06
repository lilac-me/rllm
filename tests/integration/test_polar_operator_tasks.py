"""operator_tasks.build_operator_task -> PolarRuntime.build_task_request round-trip (§0.5.8 #1, ②).

Proves the rllm->Polar operator path: a dataset task built here, fed through the (pure) request
builder, yields a well-formed operator rollout request (ascend passthrough + operator_judge + the
locked conventions). Pure — no rllm/httpx/torch. Standalone: `python tests/integration/test_polar_operator_tasks.py`.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rllm.integrations.polar.operator_tasks import build_operator_task  # noqa: E402
from rllm.integrations.polar.runtime import build_task_request  # noqa: E402

_KW = dict(image="polar-op-agent:latest", device_ids="8,9,10,11", lock_dir="/shared/npu-locks",
           skills_dir="/data/cannbot-skills", tasks_dir="/data/op-tasks")
_CFG = {"agent": {"harness": "claude_code", "skills_path": "/opt/canonical/skills"}, "timeout_seconds": 3600.0}


def test_task_carries_per_op_specs():
    t = build_operator_task("add", **_KW)
    assert t["op_name"] == "add" and "add" in t["instruction"]
    ev = t["metadata"]["polar_evaluator"]
    assert ev["strategy"] == "operator_judge" and ev["refresh_runtime"] is True
    assert ev["config"]["op_name"] == "add" and "add_impl.py" in ev["config"]["judge_command"]
    rt = t["metadata"]["polar_runtime"]
    assert rt["kwargs"]["ascend"]["device_ids"] == "8,9,10,11"


def test_writable_copy_invariant():
    rt = build_operator_task("relu", **_KW)["metadata"]["polar_runtime"]
    # exactly one read-only SOURCE mount; nothing :ro under the agent's workdir.
    assert rt["kwargs"]["volumes"] == ["/data/cannbot-skills:/opt/canonical:ro"]
    cmds = " ".join(a.get("command", "") for a in rt["prepare"] + rt["eval_prepare"])
    assert "cp -r /opt/canonical/tools" in cmds  # writable copy, both containers
    assert ":ro" not in " ".join(str(rt["prepare"]))  # agent prepare has no read-only write target


def test_roundtrip_through_build_task_request():
    t = build_operator_task("softmax", **_KW, task_json=True)
    sub = SimpleNamespace(task=t, task_id="op-softmax-1", session_id="sess-9", inference_url="")
    req = build_task_request(sub, _CFG)
    assert req["task_id"] == "op-softmax-1" and req["num_samples"] == 1
    assert req["metadata"]["rllm_session_id"] == "sess-9"
    assert req["agent"] == _CFG["agent"]                                  # static agent from cfg
    assert req["runtime"]["kwargs"]["ascend"]["lock_dir"] == "/shared/npu-locks"
    assert req["evaluator"]["config"]["op_name"] == "softmax"            # per-op from task metadata
    # task_json adds the .json upload to prepare
    targets = [a.get("target", "") for a in req["runtime"]["prepare"]]
    assert any(p.endswith("src/softmax.json") for p in targets)


def test_cfg_agent_required_else_raises():
    t = build_operator_task("add", **_KW)
    sub = SimpleNamespace(task=t, task_id="x", session_id="s", inference_url="")
    try:
        build_task_request(sub, {})  # no agent anywhere
    except ValueError:
        return
    raise AssertionError("expected ValueError when no agent in cfg or task metadata")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
