"""Polar integration — consume a Polar rollout server as an rllm remote runtime (topology a).

Mirrors ``rllm/integrations/harbor/``: Polar runs the agent + captures token-faithful
trajectories; this package turns a Polar ``SessionResult`` into rllm ``Episode``/``Step``
(adapter) and (next) drives Polar's rollout API (``runtime.PolarRuntime``).

Shipped: adapter (Polar Trace -> rllm Step/Episode).
Wiring (next increment): ``create_remote_runtime`` ``backend=="polar"`` branch +
``RemoteRuntimeConfig.polar`` field; ``runtime.PolarRuntime`` (HTTP submit->callback).
"""

from rllm.integrations.polar.adapter import (
    NormStep,
    assert_per_request,
    normalize_trace,
    polar_trace_to_step,
    session_result_to_episode,
)

__all__ = [
    "NormStep",
    "normalize_trace",
    "assert_per_request",
    "polar_trace_to_step",
    "session_result_to_episode",
]
