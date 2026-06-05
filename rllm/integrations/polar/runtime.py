"""PolarRuntime — drive a Polar rollout server as an rllm RemoteAgentRuntime (topology a, §0.5.8 #1).

Mirrors AgentCoreRuntime / HarborRuntime. One Polar task per rllm rollout
(``num_samples=1``; rllm repeats each task ``rollout.n`` times for the GRPO group), polled to
completion; returns a ``RemoteTaskResult`` whose ``raw_result`` is the Polar SessionResult dict,
which ``rllm.integrations.polar.adapter.session_result_to_episode`` converts to an Episode.

Testability: the pure helpers (``build_task_request`` / ``parse_task_status`` /
``_last_trace_reward``) have NO rllm/httpx import and are unit-tested standalone; rllm/httpx are
imported lazily so the helpers stay importable without the training stack. ``execute_tasks`` is a
thin async-httpx poll loop around them.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

_TERMINAL_TASK_STATUS = {"completed", "failed"}  # Polar TaskStatus.status vocabulary


def build_task_request(submission: "TaskSubmission", cfg: dict[str, Any]) -> dict[str, Any]:  # noqa: F821
    """rllm TaskSubmission + polar cfg -> Polar ``TaskRequest`` JSON.

    Per-task overrides may ride in ``task.metadata`` (``polar_agent`` / ``polar_runtime`` /
    ``polar_evaluator``). ``num_samples=1``: rllm does the GRPO grouping by repeating each task.
    Polar routes inference via its own topology (gateway -> vllm-ascend), so the rllm
    ``TaskSubmission.inference_url`` is intentionally unused here.
    """
    task = getattr(submission, "task", None) or {}
    meta = task.get("metadata") or {}
    agent = meta.get("polar_agent") or cfg.get("agent")
    if not agent:
        raise ValueError("polar config needs an 'agent' spec, e.g. {'harness': 'claude_code', 'model_name': ...}")
    req: dict[str, Any] = {
        "task_id": submission.task_id,
        "instruction": task.get("instruction") or meta.get("instruction") or "",
        "num_samples": 1,
        "timeout_seconds": float(cfg.get("timeout_seconds", 600.0)),
        "agent": agent,
        "builder": {"strategy": cfg.get("builder", "per_request")},
        "metadata": {"rllm_session_id": submission.session_id},
    }
    runtime = meta.get("polar_runtime") or cfg.get("runtime")
    if runtime is not None:
        req["runtime"] = runtime
    evaluator = meta.get("polar_evaluator") or cfg.get("evaluator")
    if evaluator is not None:
        req["evaluator"] = evaluator
    return req


def parse_task_status(status: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """(done, results) from a Polar ``TaskStatus`` dict. done = task status is terminal."""
    done = str(status.get("status", "")).lower() in _TERMINAL_TASK_STATUS
    return done, list(status.get("results") or [])


def _last_trace_reward(session_result: dict[str, Any]) -> float | None:
    traces = ((session_result.get("trajectory") or {}).get("traces")) or []
    for tr in reversed(traces):
        r = tr.get("reward")
        if isinstance(r, (int, float)):
            return float(r)
    return None


def session_result_to_remote_result(sr: dict[str, Any], task_id: str):
    """One Polar SessionResult dict -> rllm ``RemoteTaskResult`` (``raw_result`` carries the SessionResult)."""
    from rllm.experimental.engine.remote_runtime.protocol import RemoteTaskResult

    return RemoteTaskResult(
        finished=(sr.get("status") == "COMPLETED"),
        session_id=sr.get("session_id", ""),
        task_id=task_id,
        reward=_last_trace_reward(sr),
        error=sr.get("error"),
        termination_reason=None,
        raw_result=sr,
        metadata=sr.get("metadata") or {},
    )


def _failed_result(task_id: str, error: str):
    from rllm.experimental.engine.remote_runtime.protocol import RemoteTaskResult

    return RemoteTaskResult(finished=False, session_id="", task_id=task_id, reward=0.0, error=error)


def polar_episode_builder(result, uid: str, task):
    """``RemoteAgentFlowEngine.episode_builder`` hook for the Polar path.

    PolarRuntime stamps the Polar SessionResult into ``result.raw_result``; turn it into an
    rllm Episode via the adapter (which mirrors rllm's own ``_build_episode``).
    """
    from rllm.integrations.polar.adapter import session_result_to_episode

    return session_result_to_episode(getattr(result, "raw_result", None) or {}, uid, task)


class PolarRuntime:
    """``RemoteAgentRuntime`` (Protocol, structural) backed by a Polar rollout server."""

    def __init__(self, config: Any, exp_id: str = "", model_id: str = "") -> None:
        self.cfg: dict[str, Any] = dict(getattr(config, "polar", {}) or {})
        self.rollout_url: str = self.cfg.get("rollout_url", "http://127.0.0.1:8080").rstrip("/")
        self.poll_interval: float = float(self.cfg.get("poll_interval_seconds", 5.0))
        # Default the trainer's served model into the agent spec if the config didn't pin one.
        agent = self.cfg.get("agent")
        if model_id and isinstance(agent, dict) and not agent.get("model_name"):
            agent["model_name"] = model_id
        self._client: Any = None

    def initialize(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(base_url=self.rollout_url, timeout=httpx.Timeout(None, connect=30.0))

    async def execute_tasks(self, submissions: list, timeout: float | None = None) -> list:
        if self._client is None:
            self.initialize()
        return await asyncio.gather(*[self._run_one(s, timeout) for s in submissions])

    async def _run_one(self, submission, timeout: float | None):
        req = build_task_request(submission, self.cfg)
        resp = await self._client.post("/rollout/task/submit", json=req)
        resp.raise_for_status()
        task_id = resp.json()["task_id"]

        budget = timeout if timeout is not None else float(self.cfg.get("timeout_seconds", 600.0))
        deadline = time.monotonic() + budget + 60.0  # grace beyond the per-task budget
        while True:
            await asyncio.sleep(self.poll_interval)
            status = (await self._client.get(f"/rollout/task/{task_id}")).json()
            done, results = parse_task_status(status)
            if done:
                if not results:
                    return _failed_result(submission.task_id, f"task {task_id} {status.get('status')} with no results")
                return session_result_to_remote_result(results[0], submission.task_id)
            if time.monotonic() > deadline:
                return _failed_result(submission.task_id, f"poll timeout after {budget}s for task {task_id}")

    def shutdown(self) -> None:
        # External, long-lived rollout server; the AsyncClient is released by the caller's loop
        # teardown / GC. Avoid touching a possibly-running event loop here.
        self._client = None
