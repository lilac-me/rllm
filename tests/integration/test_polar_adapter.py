"""Unit tests for rllm.integrations.polar.adapter (Polar Trace -> rllm Step/Episode, design §0.5.8 #2).

Pure-core tests (normalize_trace / assert_per_request) run anywhere rllm imports.
rllm-construction tests need the rollout-engine deps; they importorskip if absent.
"""

import pytest

from rllm.integrations.polar.adapter import (
    assert_per_request,
    normalize_trace,
    polar_trace_to_step,
    session_result_to_episode,
)

GOOD_TRACE = {
    "prompt_ids": [1, 2, 3, 4],
    "response_ids": [10, 11, 12],
    "loss_mask": [1, 1, 1],
    "response_logprobs": [-0.1, -0.2, -0.3],
    "prompt_messages": [{"role": "user", "content": "hi"}],
    "response_messages": [{"role": "assistant", "content": "hello there"}],
    "finish_reason": "stop",
    "reward": 1.0,
}


# ----------------------------- pure core -----------------------------

def test_normalize_good():
    n = normalize_trace(GOOD_TRACE)
    assert n.prompt_ids == [1, 2, 3, 4]
    assert n.response_ids == [10, 11, 12]
    assert n.logprobs == [-0.1, -0.2, -0.3]
    assert n.finish_reason == "stop"
    assert n.content == "hello there"
    assert n.reward == 1.0


def test_normalize_logprob_mismatch_raises():
    with pytest.raises(ValueError):
        normalize_trace({**GOOD_TRACE, "response_logprobs": [-0.1, -0.2]})


def test_normalize_mask_mismatch_raises():
    with pytest.raises(ValueError):
        normalize_trace({**GOOD_TRACE, "loss_mask": [1, 1]})


def test_normalize_mask_bad_value_raises():
    with pytest.raises(ValueError):
        normalize_trace({**GOOD_TRACE, "loss_mask": [1, 2, 1]})


def test_normalize_no_logprobs_ok():
    t = {k: v for k, v in GOOD_TRACE.items() if k != "response_logprobs"}
    assert normalize_trace(t).logprobs == []


def test_normalize_rejects_logprob_integrity_violation():
    # record_utils flags token_id<->logprob misattribution / missing-logprob in metadata (the
    # length check passes); normalize_trace must REFUSE such a trace so it never trains GRPO.
    for integ in ({"misattributed": 1, "missing": 0}, {"misattributed": 0, "missing": 2}):
        with pytest.raises(ValueError):
            normalize_trace({**GOOD_TRACE, "metadata": {"logprob_integrity": integ}})
    # clean flag (no violation) passes
    normalize_trace({**GOOD_TRACE, "metadata": {"logprob_integrity": {"misattributed": 0, "missing": 0}}})


def test_assistant_text_from_blocks():
    t = {**GOOD_TRACE, "response_messages": [
        {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}
    assert normalize_trace(t).content == "ab"


def test_assert_per_request_ok():
    assert_per_request(GOOD_TRACE)  # all 1 -> no raise


def test_assert_per_request_raises_on_zero():
    with pytest.raises(ValueError):
        assert_per_request({**GOOD_TRACE, "loss_mask": [1, 0, 1]})


# ----------------------- rllm construction -----------------------

def _need_rollout():
    pytest.importorskip("rllm.experimental.rollout")


def test_polar_trace_to_step():
    _need_rollout()
    step = polar_trace_to_step(GOOD_TRACE)
    assert step.prompt_ids == [1, 2, 3, 4]
    assert step.response_ids == [10, 11, 12]
    assert step.logprobs == [-0.1, -0.2, -0.3]
    assert step.reward == 1.0
    assert step.model_response == "hello there"
    assert step.model_output.completion_ids == [10, 11, 12]


def test_step_length_guard_propagates():
    _need_rollout()
    with pytest.raises((ValueError, AssertionError)):
        polar_trace_to_step({**GOOD_TRACE, "response_logprobs": [-0.1]})


def test_session_result_to_episode():
    _need_rollout()
    result = {
        "session_id": "s1", "task_id": "t1", "status": "COMPLETED",
        "trajectory": {"status": "COMPLETED", "traces": [GOOD_TRACE, dict(GOOD_TRACE)]},
    }
    ep = session_result_to_episode(result, uid="t1:0", task={"id": "t1", "instruction": "x"})
    assert ep.session_id == "s1"
    assert len(ep.trajectories) == 1
    assert len(ep.trajectories[0].steps) == 2
    assert ep.trajectories[0].reward == 1.0
    assert ep.is_correct is True
    assert ep.metrics["steps_collected"] == 2
    assert ep.metrics["empty"] == 0


def test_session_result_empty_traces():
    _need_rollout()
    result = {"session_id": "s2", "status": "ERROR", "trajectory": {"status": "ERROR", "traces": []}}
    ep = session_result_to_episode(result, uid="t2:0", task={"id": "t2", "instruction": "x"})
    assert ep.trajectories == []
    assert ep.is_correct is False
    assert ep.metrics["empty"] == 1
    assert ep.metrics["steps_collected"] == 0


def test_session_expect_per_request_raises():
    _need_rollout()
    bad = {**GOOD_TRACE, "loss_mask": [1, 0, 1]}
    result = {"session_id": "s3", "status": "COMPLETED", "trajectory": {"traces": [bad]}}
    with pytest.raises(ValueError):
        session_result_to_episode(result, uid="t3:0", task={"id": "t3", "instruction": "x"}, expect_per_request=True)
