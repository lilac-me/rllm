"""KernelGYM environment for rllm.

Directly reuses the KernelGYM package (from /home/robomaster/Research/KernelGYM)
for kernel evaluation. Delegates to ``kernel.rewards.kernel_reward`` for
batch reward computation and to ``kernel.rewards.reward_client.KernelRewardClient``
for HTTP communication with the KernelGYM API server.

This avoids re-implementing the HTTP client logic and reward formulation,
keeping the rllm integration aligned with upstream KernelGYM / DrKernel.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ensure KernelGYM packages are importable
# ---------------------------------------------------------------------------
_KERNELGYM_ROOT = ""

for _p in [_KERNELGYM_ROOT, f"{_KERNELGYM_ROOT}/drkernel"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Lazy import helpers – only imported when actually needed
# ---------------------------------------------------------------------------

def _get_reward_client_class():
    """Lazily import KernelRewardClient from drkernel."""
    from kernel.rewards.reward_client import KernelRewardClient
    return KernelRewardClient


def _get_compute_kernel_reward_batch():
    """Lazily import the batch reward function from drkernel."""
    from kernel.rewards.kernel_reward import (
        compute_kernel_reward_batch,
        extract_kernel_code as _kg_extract,
    )
    return compute_kernel_reward_batch, _kg_extract


# ---------------------------------------------------------------------------
# Fallback kernel-code extraction (same logic as current, used as default)
# ---------------------------------------------------------------------------

def _extract_kernel_code(text: str) -> str:
    """Extract kernel code from an LLM response.

    Priority:
    1. ``<kernel>...</kernel>`` tags.
    2. Last ```python ... ``` (or bare ``` ... ```) fenced block.
    3. Entire text as fallback.
    """
    # 1. Explicit tags
    tag_match = re.search(r"<kernel>(.*?)</kernel>", text, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    # 2. Fenced code blocks (last one wins – usually the final answer)
    fence_matches = re.findall(r"```(?:python|cuda|triton)?\n?(.*?)```", text, re.DOTALL)
    if fence_matches:
        return fence_matches[-1].strip()

    # 3. Fallback
    return text.strip()


# ---------------------------------------------------------------------------
# Minimal reward-config shim for KernelRewardClient
# ---------------------------------------------------------------------------

class _RewardPolicy:
    """Thin shim that mimics the OmegaConf object expected by KernelRewardClient."""

    def __init__(self, penalties: Optional[Dict[str, float]] = None):
        self.penalties = _DictAccessor(penalties or {
            "penalty_score": -1.0,
            "compilation_fail": -0.5,
            "correctness_fail": -0.3,
            "perf_degrade": -0.1,
        })


class _CoverageReward:
    """Shim for ``reward_config.coverage_reward``."""

    def __init__(self, enable: bool = False, weight: float = 0.0,
                 reward_type: str = "time_coverage"):
        self.enable = enable
        self.weight = weight
        self.reward_type = reward_type


class _DictAccessor(dict):
    """dict subclass that also supports attribute-style access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class _RewardConfig:
    """Minimal reward config that satisfies ``KernelRewardClient.__init__``."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: int = 300,
        max_retries: int = 3,
        rate_limit: int = 8,
        max_concurrent: int = 8,
        acquire_timeout: int = 120,
        reward_func_name: str = "calculate_reward_like_kernel",
        init_correct_weight: float = 0.3,
        init_performance_weight: float = 0.6,
        speedup_eps: float = 0.05,
        speedup_reward_upper_bound: float = 10.0,
        speedup_reward_lower_bound: float = 0.0,
        task_timeout_in_client: Optional[int] = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        enable_profiling: bool = False,
        verbose_errors: bool = True,
        detect_decoy_kernel: bool = True,
        reference_backend: Optional[str] = None,
        penalty_score: float = -1.0,
        coverage_enable: bool = False,
        coverage_weight: float = 0.0,
        coverage_reward_type: str = "time_coverage",
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.acquire_timeout = acquire_timeout
        self.reward_func_name = reward_func_name
        self.init_correct_weight = init_correct_weight
        self.init_performance_weight = init_performance_weight
        self.speedup_eps = speedup_eps
        self.speedup_reward_upper_bound = speedup_reward_upper_bound
        self.speedup_reward_lower_bound = speedup_reward_lower_bound
        self.task_timeout_in_client = task_timeout_in_client or timeout
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.enable_profiling = enable_profiling
        self.verbose_errors = verbose_errors
        self.detect_decoy_kernel = detect_decoy_kernel
        self.reference_backend = reference_backend
        self.reward_policy = _RewardPolicy({"penalty_score": penalty_score})
        self.coverage_reward = _CoverageReward(
            enable=coverage_enable,
            weight=coverage_weight,
            reward_type=coverage_reward_type,
        )


# ---------------------------------------------------------------------------
# Simple httpx-based fallback client (no Ray dependency)
# ---------------------------------------------------------------------------

class _SimpleEvalClient:
    """Lightweight evaluation client using httpx when Ray is not available.

    Follows the same submit → poll → fetch pattern as KernelGYM server but
    without requiring Ray or the full ``KernelRewardClient`` stack.
    """

    def __init__(self, server_url: str, timeout: int = 300, request_timeout: int = 360):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.request_timeout = request_timeout

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to /evaluate synchronously and return the response dict."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for KernelGymEnv. "
                "Install it via 'pip install httpx'."
            ) from exc

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                resp = client.post(f"{self.server_url}/evaluate", json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("KernelGYM evaluation failed for task %s: %s",
                           payload.get("task_id", "?"), exc)
            return {
                "compiled": False,
                "correctness": False,
                "speedup": None,
                "status": "failed",
                "error_message": str(exc),
            }


# ---------------------------------------------------------------------------
# Reward computation – try drkernel first, fallback to simple formula
# ---------------------------------------------------------------------------

# Default weights (mirrors DrKernel's calculate_reward_like_kernel)
_W_COMPILED = 0.1
_W_CORRECT = 0.3
_W_SPEEDUP = 0.6
_SPEEDUP_CLIP_MAX = 10.0


def _compute_reward_simple(
    compiled: bool,
    correctness: Optional[bool],
    speedup: Optional[float],
) -> float:
    """Simple fallback reward when drkernel is not available."""
    r = _W_COMPILED * float(compiled)
    if correctness:
        r += _W_CORRECT
    if speedup is not None and speedup > 0:
        clipped = min(speedup, _SPEEDUP_CLIP_MAX)
        r += _W_SPEEDUP * (clipped / _SPEEDUP_CLIP_MAX)
    return round(r, 4)


# ---------------------------------------------------------------------------
# KernelGymEnv
# ---------------------------------------------------------------------------

class KernelGymEnv(MultiTurnEnvironment):
    """Multi-turn RL environment backed by the KernelGYM evaluation server.

    This environment directly reuses the KernelGYM package from
    ``/home/robomaster/Research/KernelGYM`` for evaluation and reward
    computation, rather than re-implementing the HTTP client logic.

    **Evaluation modes** (selected automatically):

    1. **Ray + KernelRewardClient** (preferred): When Ray is available and
       ``use_ray=True``, uses ``KernelRewardClient`` from drkernel for the
       full submit → poll → fetch pattern with rate limiting.
    2. **Simple httpx client** (fallback): Direct synchronous POST to
       ``/evaluate`` when Ray is not available.

    Each episode:
    - ``reset()``  → returns the task description (reference PyTorch code).
    - ``step(action)`` → submits the kernel code to KernelGYM, receives
      compiled / correctness / speedup metrics, compute reward.

    Args:
        task: Task dict with fields:
            - ``problem_id`` (str): Unique problem identifier.
            - ``reference_code`` (str): PyTorch reference implementation.
            - ``description`` (str, optional): Human-readable problem description.
            - ``entry_point`` (str, optional): Class name to evaluate (default "Model").
        kernel_server_url: Base URL of the running KernelGYM API server.
        max_turns: Maximum LLM refinement rounds per episode (default 3).
        num_correct_trials: Correctness trials passed to KernelGYM (default 5).
        num_perf_trials: Performance timing trials (default 100).
        timeout: Per-evaluation timeout in seconds (default 300).
        toolkit: Default toolkit name (default "kernelbench").
        backend_adapter: Default backend adapter (default "kernelbench").
        backend: Default backend type (default "cuda").
        request_timeout: HTTP request timeout in seconds (default 360).
        use_ray: Whether to attempt using KernelRewardClient with Ray.
        reward_func_name: Name of reward function in KernelRewardClient
            (default "calculate_reward_like_kernel").
        reward_config: Optional dict of extra reward-config overrides.
    """

    def __init__(
        self,
        task: dict | None = None,
        kernel_server_url: str = "http://localhost:8000",
        max_turns: int = 3,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timeout: int = 300,
        toolkit: str = "kernelbench",
        backend_adapter: str = "kernelbench",
        backend: str = "cuda",
        request_timeout: int = 360,
        use_ray: bool = False,
        reward_func_name: str = "calculate_reward_like_kernel",
        reward_config: dict | None = None,
    ):
        super().__init__(task=task, max_turns=max_turns)
        self.kernel_server_url = kernel_server_url.rstrip("/")
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.timeout = timeout
        self.toolkit = toolkit
        self.backend_adapter = backend_adapter
        self.backend = backend
        self.request_timeout = request_timeout
        self.use_ray = use_ray
        self.reward_func_name = reward_func_name
        self._reward_config_overrides = reward_config or {}

        # State reset by reset()
        self._last_error: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None

        # Lazy-initialised clients
        self._reward_client = None  # KernelRewardClient instance (Ray mode)
        self._simple_client = None  # _SimpleEvalClient instance (no-Ray mode)
        self._reward_cfg = None     # _RewardConfig object

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _get_reward_config(self) -> _RewardConfig:
        """Build (or return cached) _RewardConfig object."""
        if self._reward_cfg is None:
            defaults = {
                "server_url": self.kernel_server_url,
                "timeout": self.timeout,
                "num_correct_trials": self.num_correct_trials,
                "num_perf_trials": self.num_perf_trials,
                "reward_func_name": self.reward_func_name,
            }
            defaults.update(self._reward_config_overrides)
            self._reward_cfg = _RewardConfig(**defaults)
        return self._reward_cfg

    def _get_simple_client(self) -> _SimpleEvalClient:
        """Build (or return cached) simple HTTP client."""
        if self._simple_client is None:
            self._simple_client = _SimpleEvalClient(
                server_url=self.kernel_server_url,
                timeout=self.timeout,
                request_timeout=self.request_timeout,
            )
        return self._simple_client

    def _try_get_reward_client(self):
        """Try to initialise KernelRewardClient (requires Ray)."""
        if self._reward_client is not None:
            return self._reward_client

        if not self.use_ray:
            return None

        try:
            import ray
            if not ray.is_initialized():
                logger.info("Ray not initialised; falling back to simple client")
                return None

            KernelRewardClient = _get_reward_client_class()
            cfg = self._get_reward_config()
            self._reward_client = KernelRewardClient(reward_config=cfg)
            logger.info("Initialised KernelRewardClient (Ray mode)")
            return self._reward_client
        except Exception as exc:
            logger.warning("Failed to init KernelRewardClient, using simple client: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, task: dict | None = None, seed: int | None = None) -> Tuple[dict, dict]:
        """Reset the environment and return the initial observation."""
        if task is not None:
            self.task = task

        assert self.task is not None, "Task must be set before calling reset()"

        self.done = False
        self.current_turn = 0
        self.history = []
        self._last_error = None
        self._last_result = None

        return self.task, {}

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, dict]:
        """Submit the kernel code to KernelGYM and compute the reward."""
        self.history.append(action)

        kernel_code = _extract_kernel_code(action)
        reward, next_obs = self.get_reward_and_next_obs(self.task, kernel_code)

        self.current_turn += 1
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(
        self, task: dict, action: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the kernel and return (reward, next_obs).

        Tries to use drkernel's ``compute_kernel_reward_batch`` for reward
        computation; falls back to a simple formula if unavailable.
        """
        result = self._evaluate_kernel(task, action)
        self._last_result = result

        compiled: bool = result.get("compiled", False)
        correctness: Optional[bool] = result.get("correctness")
        speedup: Optional[float] = result.get("speedup")
        error_msg: Optional[str] = result.get("error_message")

        # --- Reward computation ---
        # Prefer drkernel reward if available in the result (Ray path injects it)
        if "reward" in result and isinstance(result["reward"], (int, float)):
            reward = float(result["reward"])
        else:
            reward = _compute_reward_simple(compiled, correctness, speedup)

        # Build next observation for the agent
        if self.done or (compiled and correctness):
            next_obs: Dict[str, Any] = {}
        else:
            feedback = self._build_feedback(compiled, correctness, speedup, error_msg)
            next_obs = {
                "feedback": feedback,
                "compiled": compiled,
                "correctness": correctness,
                "speedup": speedup,
                "error_message": error_msg,
            }

        return reward, next_obs

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate_kernel(self, task: dict, kernel_code: str) -> Dict[str, Any]:
        """Evaluate a kernel using KernelGYM.

        First tries the Ray-based KernelRewardClient from drkernel.
        Falls back to the simple httpx client.
        """
        task_id = f"{task.get('problem_id', 'task')}_{uuid.uuid4().hex[:8]}"
        server_url = task.get("kernel_server_url", self.kernel_server_url).rstrip("/")
        entry_point = task.get("entry_point", "Model")

        payload = {
            "task_id": task_id,
            "reference_code": task.get("reference_code", ""),
            "kernel_code": kernel_code,
            "toolkit": task.get("toolkit", self.toolkit),
            "backend_adapter": task.get("backend_adapter", self.backend_adapter),
            "backend": task.get("backend", self.backend),
            "entry_point": entry_point,
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
            "timeout": self.timeout,
            "priority": "normal",
            "workflow": task.get("workflow", "kernelbench"),
        }

        # --- Try Ray-based KernelRewardClient ---
        reward_client = self._try_get_reward_client()
        if reward_client is not None:
            return self._evaluate_via_reward_client(reward_client, task, payload)

        # --- Fallback: simple httpx POST ---
        return self._get_simple_client().evaluate(payload)

    def _evaluate_via_reward_client(
        self,
        client,
        task: dict,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use KernelRewardClient.compute_batch_rewards for a single task.

        This follows the exact patterns used in drkernel's
        ``compute_kernel_reward_batch`` — submit via Ray workers,
        poll for completion, compute reward.
        """
        import asyncio

        cfg = self._get_reward_config()
        reward_task = {
            "reference_code": payload["reference_code"],
            "kernel_code": payload["kernel_code"],
            "entry_point": payload["entry_point"],
            "use_reference_cache": False,
            "uuid": task.get("uuid", ""),
            "is_valid": False,
            "task_timeout": None,
            "task_timeout_in_client": None,
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
            "enable_profiling": cfg.enable_profiling,
            "verbose_errors": cfg.verbose_errors,
            "detect_decoy_kernel": cfg.detect_decoy_kernel,
            "reference_backend": cfg.reference_backend,
        }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside an async context (e.g. Ray); create new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    results = pool.submit(
                        lambda: asyncio.new_event_loop().run_until_complete(
                            client.compute_batch_rewards([reward_task])
                        )
                    ).result(timeout=self.request_timeout)
            else:
                results = loop.run_until_complete(
                    client.compute_batch_rewards([reward_task])
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                client.compute_batch_rewards([reward_task])
            )

        if results and len(results) > 0:
            return results[0]

        return {
            "compiled": False,
            "correctness": False,
            "speedup": None,
            "status": "failed",
            "error_message": "KernelRewardClient returned empty results",
        }

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feedback(
        compiled: bool,
        correctness: Optional[bool],
        speedup: Optional[float],
        error_message: Optional[str],
    ) -> str:
        """Human-readable feedback string to feed back to the agent."""
        lines = []
        if not compiled:
            lines.append("❌ Compilation FAILED.")
            if error_message:
                lines.append(f"Error:\n{error_message}")
            lines.append("Please fix the compilation error and resubmit your kernel.")
        elif not correctness:
            lines.append("✅ Compilation succeeded.")
            lines.append("❌ Correctness check FAILED — outputs do not match the reference.")
            if error_message:
                lines.append(f"Error:\n{error_message}")
            lines.append("Please fix the numerical correctness issue and resubmit.")
        else:
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
            lines.append("✅ Compilation succeeded.")
            lines.append("✅ Correctness check PASSED.")
            lines.append(f"Speedup over reference: {speedup_str}")
            lines.append("Can you further optimize the kernel?")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    # Keys that are recognised as environment *configuration* (constructor params).
    # Everything else in the dict coming from `extra_info` is treated as task data.
    _ENV_CONFIG_KEYS = frozenset({
        "kernel_server_url", "max_turns", "num_correct_trials",
        "num_perf_trials", "timeout", "toolkit", "backend_adapter",
        "request_timeout", "use_ray", "reward_func_name", "reward_config",
    })

    @staticmethod
    def from_dict(env_args: dict) -> "KernelGymEnv":
        """Create a KernelGymEnv from a configuration dictionary.

        When called from ``AgentPPOTrainer.init_envs_and_agents``, *env_args*
        is the union of the dataset row (``extra_info``) and the base env
        config.  Dataset rows contain task-level fields such as
        ``problem_id``, ``reference_code``, ``description``, etc. — these
        must be collected into the ``task`` dict rather than forwarded as
        ``**kwargs`` to ``__init__`` (which would cause a TypeError).

        The split logic: any key in ``_ENV_CONFIG_KEYS`` is an env config
        parameter; everything else is task data.
        """
        env_args = dict(env_args)  # avoid mutating caller's dict

        # If the caller already wrapped the task, honour that.
        if "task" in env_args:
            task = env_args.pop("task")
        else:
            # Separate env config from task data
            config = {k: env_args.pop(k) for k in list(env_args) if k in KernelGymEnv._ENV_CONFIG_KEYS}
            task = env_args if env_args else None  # remaining keys = task data
            env_args = config

        return KernelGymEnv(task=task, **env_args)

    @staticmethod
    def is_multithread_safe() -> bool:
        """KernelGymEnv is stateless per instance; safe to use across threads."""
        return True
