"""KernelGYM hybrid reward logic (aligned with rllm-071 ``KernelGymEnv``).

Used by fully-async ``kernelgym_rollout`` so reward matches upstream without
replacing the legacy ``kernelgym_env.py`` used by sync training.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KernelGymHybridRewardParams:
    """Scalar + table config for reward methods (OmegaConf-compatible layout)."""

    reward_func_name: str = "calculate_reward_like_kernel"
    init_correct_weight: float = 0.3
    init_performance_weight: float = 0.6
    speedup_eps: float = 0.05
    penalty_score: float = -1.0
    speedup_reward_upper_bound: float = 10.0
    speedup_reward_lower_bound: float = 0.0
    compilation_fail: float = -0.5
    correctness_fail: float = -0.3
    perf_degrade: float = -0.1
    coverage_enable: bool = False
    coverage_weight: float = 0.0
    coverage_reward_type: str = "time_coverage"  # time_coverage | number_coverage
    like_kernel_incomplete_reward: float = -1.0  # when status != "completed" for like_kernel

    @classmethod
    def from_kernel_mapping(cls, kernel_cfg: dict) -> KernelGymHybridRewardParams:
        """Build params from Hydra ``kernel:`` and optional nested ``reward_config``."""
        rc = kernel_cfg.get("reward_config") or {}
        if not isinstance(rc, dict):
            rc = {}
        penalties = rc.get("penalties")
        if not isinstance(penalties, dict):
            penalties = {}
        cov = rc.get("coverage_reward")
        if not isinstance(cov, dict):
            cov = {}

        def _f(key: str, default: float) -> float:
            for d in (penalties, rc, kernel_cfg):
                if isinstance(d, dict) and key in d and d[key] is not None:
                    return float(d[key])
            return default

        cov_enable = cov.get("enable", kernel_cfg.get("coverage_enable"))
        if cov_enable is None:
            cov_enable = False
        cov_w = cov.get("weight", kernel_cfg.get("coverage_weight"))
        if cov_w is None:
            cov_w = 0.0
        cov_rt = cov.get("reward_type", kernel_cfg.get("coverage_reward_type", "time_coverage"))
        if cov_rt is None:
            cov_rt = "time_coverage"

        return cls(
            reward_func_name=str(kernel_cfg.get("reward_func_name", cls.reward_func_name)),
            init_correct_weight=float(kernel_cfg.get("init_correct_weight", 0.3)),
            init_performance_weight=float(kernel_cfg.get("init_performance_weight", 0.6)),
            speedup_eps=float(kernel_cfg.get("speedup_eps", 0.05)),
            penalty_score=_f("penalty_score", -1.0),
            speedup_reward_upper_bound=float(
                kernel_cfg.get("speedup_reward_upper_bound", rc.get("speedup_reward_upper_bound", 10.0))
            ),
            speedup_reward_lower_bound=float(
                kernel_cfg.get("speedup_reward_lower_bound", rc.get("speedup_reward_lower_bound", 0.0))
            ),
            compilation_fail=_f("compilation_fail", -0.5),
            correctness_fail=_f("correctness_fail", -0.3),
            perf_degrade=_f("perf_degrade", -0.1),
            coverage_enable=bool(cov_enable),
            coverage_weight=float(cov_w),
            coverage_reward_type=str(cov_rt),
            like_kernel_incomplete_reward=float(
                kernel_cfg.get("like_kernel_incomplete_reward", -1.0)
            ),
        )


class _PenaltiesView:
    def __init__(self, p: KernelGymHybridRewardParams) -> None:
        self.p = p

    def get(self, key: str, default: float) -> float:
        if key == "penalty_score":
            return self.p.penalty_score
        if key == "compilation_fail":
            return self.p.compilation_fail
        if key == "correctness_fail":
            return self.p.correctness_fail
        if key == "perf_degrade":
            return self.p.perf_degrade
        return default


class _RewardPolicyView:
    def __init__(self, p: KernelGymHybridRewardParams) -> None:
        self.penalties = _PenaltiesView(p)


class _CoverageView:
    def __init__(self, p: KernelGymHybridRewardParams) -> None:
        self.enable = p.coverage_enable
        self.weight = p.coverage_weight
        self.reward_type = p.coverage_reward_type


class _CfgView:
    """Mimic ``self.config`` attribute access in rllm-071 ``KernelGymEnv`` methods."""

    def __init__(self, p: KernelGymHybridRewardParams) -> None:
        self.reward_policy = _RewardPolicyView(p)
        self.coverage_reward = _CoverageView(p)


class KernelGymRewardOps:
    """Stateless reward calculators; same formulas as rllm-071 ``KernelGymEnv``."""

    def __init__(self, params: KernelGymHybridRewardParams) -> None:
        self.p = params
        self._cfg = _CfgView(params)

    def get_reward_function(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        name = (self.p.reward_func_name or "calculate_reward_like_kernel").strip()
        fn = getattr(self, name, None)
        if callable(fn):
            return fn
        logger.warning("Unknown reward_func_name=%r; using calculate_reward_like_kernel", name)
        return self.calculate_reward_like_kernel

    def default_failed_merge(self) -> Dict[str, Any]:
        return {
            "reward": self.p.penalty_score,
            "speedup": 0.0,
            "success": False,
            "correctness": False,
            "compiled": False,
            "error": "Unknown error",
            "num_custom_kernel": 0,
            "num_total_kernels": 0,
            "custom_kernel_cuda_time_in_profiling_us": 0,
            "total_kernel_run_time_in_profiling_us": 0,
        }

    def apply_reward(self, ret_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if ret_data is None:
            return self.default_failed_merge()
        rfunc = self.get_reward_function()
        summary = rfunc(ret_data)
        merged: Dict[str, Any] = {**ret_data, **(summary or {})}
        return merged

    def score_from_merged(self, merged: Dict[str, Any]) -> float:
        v = merged.get("score", merged.get("reward", 0.0))
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    # --- reward functions (rllm-071) -----------------------------------------

    def calculate_reward_like_kernel(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result.get("status") != "completed":
            err = result.get("error_message", "Task failed")
            if err == "Task failed":
                err = result.get("error", "Task failed")
            logger.debug("calculate_reward_like_kernel: task not completed: %s", err)
            return {
                "reward": self.p.like_kernel_incomplete_reward,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": err,
            }
        if result.get("decoy_kernel", False):
            logger.info("Decoy kernel detected; forcing penalty reward")
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": -1.0,
            }

        penalties = self._cfg.reward_policy.penalties
        compilation_fail = float(penalties.get("compilation_fail", -0.5))
        correctness_fail = float(penalties.get("correctness_fail", -0.3))
        perf_degrade = float(penalties.get("perf_degrade", -0.1))

        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0) or 0.0
        compiled = result.get("compiled", False)
        if not compiled:
            reward = compilation_fail
        elif not correctness:
            reward = correctness_fail
        else:
            if speedup >= 3.0:
                reward = 1.0
            elif speedup >= 2.0:
                reward = 0.8
            elif speedup >= 1.5:
                reward = 0.6
            elif speedup >= 1.2:
                reward = 0.4
            elif speedup >= 1.0:
                reward = 0.2
            else:
                reward = perf_degrade
        return {
            "reward": reward,
            "speedup": speedup,
            "success": bool(compiled and correctness),
            "correctness": correctness,
            "compiled": compiled,
            "score": reward,
        }

    def compute_coverage_reward(self, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata") or {}

        def _get_field(*keys: str, default: int = 0) -> int:
            for k in keys:
                if k in metadata:
                    return int(metadata.get(k) or default)
                if k in result:
                    v = result.get(k)
                    if v is not None:
                        return int(v)
            return default

        num_custom_kernel = _get_field("num_custom_kernels", "num_custom_kernel")
        num_total_kernels = _get_field("num_total_kernels", "num_total_kernel", "num_total_kernels")
        custom_cuda = _get_field("custom_kernel_cuda_time_in_profiling_us")
        total_time = _get_field("total_kernel_run_time_in_profiling_us")

        num_cov = 0.0
        if num_total_kernels > 0:
            num_cov = num_custom_kernel / num_total_kernels
        time_cov = 0.0
        if total_time > 0:
            time_cov = custom_cuda / total_time

        rt = self._cfg.coverage_reward.reward_type
        if rt == "time_coverage":
            coverage = time_cov
        elif rt == "number_coverage":
            coverage = num_cov
        else:
            raise ValueError(f"Invalid coverage reward type: {rt!r}")
        return {
            "coverage": coverage,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_cuda,
            "total_kernel_run_time_in_profiling_us": total_time,
        }

    def calculate_reward_weighted(self, result: Dict[str, Any]) -> Dict[str, Any]:
        penalty = self.p.penalty_score
        if result.get("status") != "completed":
            err = result.get("error_message", "Task failed")
            if err == "Task failed":
                err = result.get("error", "Task failed")
            out: Dict[str, Any] = {
                "reward": penalty,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": err,
            }
            for k, v in result.items():
                if k not in out:
                    out[k] = v
            return out
        if result.get("decoy_kernel", False):
            return {
                "reward": penalty,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0) or 0.0
        compiled = result.get("compiled", False)
        profiling = result.get("profiling")
        # TODO: status=="completed" but compiled==False is currently not penalised
        # here (reward falls through to base=0.0). rllm-071 behaviour should be
        # confirmed; if compilation_fail penalty is desired, add:
        #   if not compiled: return {"reward": self.p.compilation_fail, ...}
        is_pos = speedup >= (1.0 + self.p.speedup_eps)
        base = self.p.init_correct_weight * float(correctness) + self.p.init_performance_weight * float(
            is_pos
        )
        final_r = base
        num_custom_kernel = 0
        num_total = 0
        cu1 = 0
        cu2 = 0
        if correctness:
            cdict = self.compute_coverage_reward(result)
            cov = cdict["coverage"]
            num_custom_kernel = cdict["num_custom_kernel"]
            num_total = cdict["num_total_kernels"]
            cu1 = cdict["custom_kernel_cuda_time_in_profiling_us"]
            cu2 = cdict["total_kernel_run_time_in_profiling_us"]
            if self._cfg.coverage_reward.enable:
                final_r += self._cfg.coverage_reward.weight * cov
        return {
            "reward": final_r,
            "speedup": speedup,
            "success": bool(compiled and correctness),
            "correctness": correctness,
            "compiled": compiled,
            "score": final_r,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total,
            "custom_kernel_cuda_time_in_profiling_us": cu1,
            "total_kernel_run_time_in_profiling_us": cu2,
        }

    def calculate_reward_speedup(self, result: Dict[str, Any]) -> Dict[str, Any]:
        penalty = self.p.penalty_score
        if result.get("status") != "completed":
            err = result.get("error_message", "Task failed")
            if err == "Task failed":
                err = result.get("error", "Task failed")
            out: Dict[str, Any] = {
                "reward": penalty,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": err,
            }
            for k, v in result.items():
                if k not in out:
                    out[k] = v
            return out
        if result.get("decoy_kernel", False):
            return {
                "reward": penalty,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0) or 0.0
        compiled = result.get("compiled", False)
        profiling = result.get("profiling")
        # TODO: status=="completed" but compiled==False is currently not penalised
        # here (reward falls through to base=0.0). rllm-071 behaviour should be
        # confirmed; if compilation_fail penalty is desired, add:
        #   if not compiled: return {"reward": self.p.compilation_fail, ...}
        reward_speedup = min(speedup, self.p.speedup_reward_upper_bound)
        if reward_speedup < self.p.speedup_reward_lower_bound:
            reward_speedup = 0.0
        base = self.p.init_correct_weight * float(correctness) + self.p.init_performance_weight * float(
            reward_speedup
        )
        final_r = base
        nck, ntk, c1, c2 = 0, 0, 0, 0
        if correctness:
            cdict = self.compute_coverage_reward(result)
            cov = cdict["coverage"]
            nck = cdict["num_custom_kernel"]
            ntk = cdict["num_total_kernels"]
            c1 = cdict["custom_kernel_cuda_time_in_profiling_us"]
            c2 = cdict["total_kernel_run_time_in_profiling_us"]
            if self._cfg.coverage_reward.enable:
                final_r += self._cfg.coverage_reward.weight * cov
        return {
            "reward": final_r,
            "speedup": speedup,
            "success": bool(compiled and correctness),
            "correctness": correctness,
            "compiled": compiled,
            "score": final_r,
            "profiling": profiling,
            "num_custom_kernel": nck,
            "num_total_kernels": ntk,
            "custom_kernel_cuda_time_in_profiling_us": c1,
            "total_kernel_run_time_in_profiling_us": c2,
        }


def preflight_validate(
    reference_code: str, kernel_code: str, entry_point: str
) -> Tuple[bool, str]:
    """Same regex checks as rllm-071 ``KernelGymEnv._preflight_validate`` (stdlib ``re``)."""
    try:
        ep = re.escape(entry_point)
        ref_ok = bool(
            re.search(rf"class {ep}\s*\(.*Module\s*\)", reference_code or "")
        )
        ker_ok = bool(
            re.search(rf"class {ep}New\s*\(.*Module\s*\)", kernel_code or "")
        )
        if ref_ok and ker_ok:
            return True, ""
        missing: list[str] = []
        if not ref_ok:
            missing.append(f"class {entry_point}(nn.Module)")
        if not ker_ok:
            missing.append(f"class {entry_point}New(nn.Module)")
        return False, ", ".join(missing)
    except Exception as exc:
        logger.debug("preflight skipped due to error: %s", exc)
        return True, ""


def preflight_failure_result(
    error_message: str, *, penalty: float, task_id: str
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error_message": error_message,
        "error": error_message,
        "task_id": task_id,
        "compiled": False,
        "correctness": False,
        "speedup": 0.0,
        "reward": penalty,
    }
