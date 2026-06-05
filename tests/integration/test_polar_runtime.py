"""Unit tests for rllm.integrations.polar.runtime (PolarRuntime helpers, design §0.5.8 #1).

Pure helpers (build_task_request / parse_task_status / _last_trace_reward) need no rllm/httpx.
The RemoteTaskResult construction is importorskip'd (needs the remote_runtime protocol).
"""

from types import SimpleNamespace

import pytest

from rllm.integrations.polar.runtime import (
    _last_trace_reward,
    build_task_request,
    parse_task_status,
)


def _sub(task=None, task_id="t1", session_id="s1"):
    return SimpleNamespace(task=task if task is not None else {"instruction": "do x"},
                           task_id=task_id, session_id=session_id)


# ----------------------------- build_task_request -----------------------------

def test_build_task_request_basic():
    cfg = {"agent": {"harness": "claude_code", "model_name": "qwen35"},
           "runtime": {"backend": "docker"}, "evaluator": {"strategy": "test_on_output"}}
    req = build_task_request(_sub(task={"instruction": "fix bug"}), cfg)
    assert req["task_id"] == "t1"
    assert req["instruction"] == "fix bug"
    assert req["num_samples"] == 1
    assert req["agent"] == {"harness": "claude_code", "model_name": "qwen35"}
    assert req["builder"] == {"strategy": "per_request"}  # locked default
    assert req["runtime"] == {"backend": "docker"}
    assert req["evaluator"] == {"strategy": "test_on_output"}
    assert req["metadata"]["rllm_session_id"] == "s1"


def test_build_task_request_requires_agent():
    with pytest.raises(ValueError):
        build_task_request(_sub(), {})  # no agent in cfg or metadata


def test_build_task_request_metadata_override():
    cfg = {"agent": {"harness": "claude_code"}}
    sub = _sub(task={"instruction": "x", "metadata": {
        "polar_agent": {"harness": "codex"}, "polar_runtime": {"backend": "apptainer"}}})
    req = build_task_request(sub, cfg)
    assert req["agent"] == {"harness": "codex"}          # metadata wins over cfg
    assert req["runtime"] == {"backend": "apptainer"}


def test_build_task_request_builder_override():
    req = build_task_request(_sub(), {"agent": {"harness": "x"}, "builder": "prefix_merging"})
    assert req["builder"] == {"strategy": "prefix_merging"}


def test_build_task_request_omits_optional_when_absent():
    req = build_task_request(_sub(), {"agent": {"harness": "x"}})
    assert "runtime" not in req and "evaluator" not in req


# ----------------------------- parse_task_status -----------------------------

def test_parse_task_status_running():
    assert parse_task_status({"status": "running", "results": []}) == (False, [])


def test_parse_task_status_completed():
    done, results = parse_task_status({"status": "completed", "results": [{"session_id": "s"}]})
    assert done is True and results == [{"session_id": "s"}]


def test_parse_task_status_failed_is_terminal():
    done, _ = parse_task_status({"status": "failed", "results": []})
    assert done is True


# ----------------------------- _last_trace_reward -----------------------------

def test_last_trace_reward_takes_last_numeric():
    sr = {"trajectory": {"traces": [{"reward": 0.0}, {"reward": 0.7}]}}
    assert _last_trace_reward(sr) == 0.7


def test_last_trace_reward_none_when_absent():
    assert _last_trace_reward({"trajectory": {"traces": [{}]}}) is None
    assert _last_trace_reward({}) is None


# ----------------------- RemoteTaskResult construction -----------------------

def test_session_result_to_remote_result():
    pytest.importorskip("rllm.experimental.engine.remote_runtime.protocol")
    from rllm.integrations.polar.runtime import session_result_to_remote_result

    sr = {"session_id": "s1", "status": "COMPLETED",
          "trajectory": {"traces": [{"reward": 1.0}]}, "metadata": {"k": "v"}}
    r = session_result_to_remote_result(sr, "t1")
    assert r.finished is True
    assert r.session_id == "s1"
    assert r.task_id == "t1"
    assert r.reward == 1.0
    assert r.raw_result is sr          # the adapter consumes this verbatim
    assert r.metadata == {"k": "v"}


def test_session_result_error_not_finished():
    pytest.importorskip("rllm.experimental.engine.remote_runtime.protocol")
    from rllm.integrations.polar.runtime import session_result_to_remote_result

    r = session_result_to_remote_result({"session_id": "s", "status": "ERROR", "trajectory": {"traces": []}}, "t")
    assert r.finished is False
    assert r.reward is None
