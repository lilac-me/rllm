"""Polar SessionResult/Trace -> rllm Episode/Step adapter (topology a, design §0.5.8 #2).

Split by testability:
  * PURE core (`normalize_trace`, `assert_per_request`) — NO rllm import, where all the
    risk lives (field extraction + the length/mask contract). Fully unit-tested anywhere.
  * rllm construction (`polar_trace_to_step`, `session_result_to_episode`) — imports rllm
    LAZILY, so the core stays importable without the training stack. Mirrors rllm's own
    `_build_episode` / `trace_record_to_step` so behaviour matches the verl bridge already in use.

Contract (both sides enforce the same length discipline — see Polar Trace._validate_response_lengths
and rllm Step.model_post_init): len(loss_mask)==len(response_logprobs)==len(response_ids); raise otherwise.
rllm `Step` has NO loss_mask field (it derives the mask in transform.py), so per_request
(loss_mask all 1) is required — `assert_per_request` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormStep:
    """rllm-free normalized view of one Polar Trace."""
    prompt_ids: list[int]
    response_ids: list[int]
    logprobs: list[float]       # [] when Polar provided none
    finish_reason: str | None
    content: str
    reward: float | None


def _assistant_text(messages: list[dict] | None) -> str:
    parts: list[str] = []
    for m in messages or []:
        if m.get("role") not in (None, "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # content blocks
            parts.extend(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return "".join(parts)


def normalize_trace(trace: dict) -> NormStep:
    """Extract + validate one Polar Trace. Raises ValueError on any length/mask violation."""
    response_ids = list(trace.get("response_ids") or [])
    prompt_ids = list(trace.get("prompt_ids") or [])

    logprobs = trace.get("response_logprobs")
    if logprobs is not None and len(logprobs) != len(response_ids):
        raise ValueError(f"response_logprobs len {len(logprobs)} != response_ids len {len(response_ids)}")

    mask = trace.get("loss_mask")
    if mask is not None:
        if len(mask) != len(response_ids):
            raise ValueError(f"loss_mask len {len(mask)} != response_ids len {len(response_ids)}")
        if any(m not in (0, 1) for m in mask):
            raise ValueError("loss_mask values must be 0 or 1")

    reward = trace.get("reward")
    return NormStep(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        logprobs=list(logprobs) if logprobs is not None else [],
        finish_reason=trace.get("finish_reason"),
        content=_assistant_text(trace.get("response_messages")),
        reward=float(reward) if isinstance(reward, (int, float)) else None,
    )


def assert_per_request(trace: dict) -> None:
    """Enforce the locked per_request builder: loss_mask must be all 1 (whole response trainable)."""
    mask = trace.get("loss_mask")
    if mask and any(m != 1 for m in mask):
        raise ValueError("expected per_request builder (loss_mask all 1) but found 0s")


def polar_trace_to_step(trace: dict):
    """One Polar Trace -> rllm Step (via ModelOutput). Imports rllm lazily."""
    from rllm.experimental.rollout import ModelOutput
    from rllm.types import Step

    n = normalize_trace(trace)
    mo = ModelOutput(
        content=n.content or None,
        prompt_ids=n.prompt_ids,
        completion_ids=n.response_ids,
        logprobs=n.logprobs,
        finish_reason=n.finish_reason,
        prompt_length=len(n.prompt_ids),
        completion_length=len(n.response_ids),
    )
    # Step.model_post_init backfills prompt_ids/response_ids/logprobs from model_output and
    # re-asserts len(response_ids)==len(logprobs) — a second, independent guard on the contract.
    return Step(
        model_output=mo,
        reward=n.reward or 0.0,
        done=False,
        model_response=n.content,
        chat_completions=list(trace.get("prompt_messages") or []) + list(trace.get("response_messages") or []),
    )


def session_result_to_episode(result: dict, uid: str, task, *, expect_per_request: bool = False):
    """Polar SessionResult -> rllm Episode (one Trajectory). Mirrors rllm's own _build_episode."""
    from rllm.types import Episode, Trajectory
    from rllm.workflows.workflow import TerminationReason

    traj = result.get("trajectory") or {}
    traces = traj.get("traces") or []
    if expect_per_request:
        for tr in traces:
            assert_per_request(tr)

    steps = [polar_trace_to_step(tr) for tr in traces]

    # Session reward = last trace's reward (matches Polar examples/calculator run.py session_reward).
    reward: float | None = None
    for tr in reversed(traces):
        r = tr.get("reward")
        if isinstance(r, (int, float)):
            reward = float(r)
            break

    status = result.get("status") or traj.get("status")
    completed = str(status).upper() == "COMPLETED"

    # ONLY a COMPLETED session yields a TRAINABLE trajectory. ERROR/TIMEOUT — including operator_judge
    # RAISING on an infra failure (-> trajectory.status=ERROR, reward stays None) — must NOT become a
    # 0.0-reward trainable episode: that would defeat the infra-vs-operator retry contract and poison
    # GRPO with a false negative (see §3.4 / operator_reward.INFRA_ERROR_TYPES). Non-COMPLETED ->
    # non-trainable (empty trajectories, which transform.py drops); RemoteTaskResult.finished=False
    # already routes the task to retry/skip upstream.
    trajectories = []
    if completed and (steps or reward is not None):
        trajectories.append(Trajectory(name="default", task=task, steps=steps,
                                       reward=reward if reward is not None else 0.0))

    return Episode(
        id=uid,
        task=task,
        is_correct=bool(completed and reward and reward >= 1.0),
        session_id=result.get("session_id"),
        trajectories=trajectories,
        termination_reason=TerminationReason.UNKNOWN,  # TODO: map finish_reason==length -> MAX_RESPONSE_LENGTH_EXCEEDED
        metrics={"empty": int(not traces), "steps_collected": len(traces),
                 "polar_status": status, "trainable": int(bool(trajectories))},
    )
