"""Integration tests for the gateway-less RemoteAgentFlowEngine + Polar episode_builder (§0.5.8 #1, 2b).

Exercises the ``gateway=None`` branch with a fake runtime (no Polar server). Needs rllm.
"""

import asyncio

import pytest

pytest.importorskip("rllm.experimental.rollout")  # adapter Episode construction needs the rollout engine

from rllm.experimental.engine.remote_agent_flow_engine import RemoteAgentFlowEngine  # noqa: E402
from rllm.integrations.polar.runtime import (  # noqa: E402
    polar_episode_builder,
    session_result_to_remote_result,
)


def _session(reward=1.0, status="COMPLETED"):
    return {
        "session_id": "sess-1", "task_id": "t1", "status": status,
        "trajectory": {"status": status, "traces": [{
            "prompt_ids": [1, 2], "response_ids": [3, 4],
            "loss_mask": [1, 1], "response_logprobs": [-0.1, -0.2],
            "response_messages": [{"role": "assistant", "content": "ok"}],
            "finish_reason": "stop", "reward": reward,
        }]},
    }


class _FakeRuntime:
    """Returns a RemoteTaskResult whose raw_result is a Polar SessionResult (no HTTP)."""

    def __init__(self, session):
        self.session = session

    def initialize(self):
        pass

    async def execute_tasks(self, submissions, timeout=None):
        return [session_result_to_remote_result(self.session, submissions[0].task_id)]

    def shutdown(self):
        pass


def test_requires_gateway_or_builder():
    with pytest.raises(ValueError):
        RemoteAgentFlowEngine(runtime=_FakeRuntime(_session()), gateway=None, episode_builder=None)


def test_gatewayless_builds_episode_via_polar_builder():
    eng = RemoteAgentFlowEngine(
        runtime=_FakeRuntime(_session(reward=1.0)),
        gateway=None,
        episode_builder=polar_episode_builder,
        n_parallel_tasks=2,
    )
    task_id, _ridx, _idx, episode = asyncio.run(eng.process_task_with_retry({"instruction": "x"}, "t1", 0, 0))
    assert task_id == "t1"
    assert episode.session_id == "sess-1"
    assert len(episode.trajectories) == 1
    assert len(episode.trajectories[0].steps) == 1
    assert episode.is_correct is True


def test_polar_episode_builder_direct():
    rr = session_result_to_remote_result(_session(reward=0.0), "t9")
    ep = polar_episode_builder(rr, "t9:0", {"id": "t9", "instruction": "x"})
    assert ep.session_id == "sess-1"
    assert ep.is_correct is False


def test_error_session_with_traces_is_not_trainable():
    # ERROR session (e.g. operator_judge raised on an infra failure) carrying token traces but
    # reward=None must NOT become a 0.0-reward trainable episode — that would poison GRPO with a
    # false negative. Adapter must drop it to non-trainable (no trajectories).
    rr = session_result_to_remote_result(_session(reward=None, status="ERROR"), "terr")
    assert rr.finished is False                       # status != COMPLETED -> retry/skip upstream
    ep = polar_episode_builder(rr, "terr:0", {"id": "terr", "instruction": "x"})
    assert ep.trajectories == [] and ep.is_correct is False   # non-trainable, not reward-0.0
