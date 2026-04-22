"""KernelGYM fully-async training entry point.

Uses ``AsyncAgentTrainer`` from the RLLM fully-async framework to run
decoupled rollout + PPO training with message-queue hand-off.

Usage:
    python -m examples.kernelgym.train_kernelgym_fully_async [hydra overrides ...]
"""

from __future__ import annotations

import random
import time

import hydra

from rllm.environments.kernelgym.kernelgym_reward_ops import (
    KernelGymHybridRewardParams,
    KernelGymRewardOps,
)
from rllm.experimental.fully_async.runner import AsyncAgentTrainer

from .kernelgym_rollout import kernelgym_rollout


# ---------------------------------------------------------------------------
# Factory: build rollout_fn with kernel config captured via closure
# ---------------------------------------------------------------------------

def _row_kwargs_to_task(kwargs: dict) -> dict:
    """Normalize a dataloader row for ``kernelgym_rollout``.

    ``prepare_kernelbench_data`` registers ``{"task": {...}, "backend": ...}``.
    Mock / hand-rolled sets often use a flat dict with ``reference_code`` at
    the top level. ``RolloutExecutor`` expands each parquet/row dict as
    ``**kwargs``.
    """
    inner = kwargs.get("task")
    if isinstance(inner, dict):
        task = dict(inner)
        for k, v in kwargs.items():
            if k == "task":
                continue
            task.setdefault(k, v)
        return task
    return dict(kwargs)


def make_rollout_fn(kernel_cfg: dict):
    """Return an ``async def rollout_fn(client, tokenizer, **kwargs)`` with
    KernelGYM evaluation parameters baked in via closure.

    ``RolloutExecutor.generate_trajectory`` passes **only** the dataset-row
    fields as ``kwargs`` (e.g. ``reference_code``, ``problem_id``).  Kernel
    server URL, max turns, backend, etc. come from the Hydra config and must
    be captured here.
    """
    # M4/M5: OmegaConf DictConfig is NOT a plain dict — isinstance(rc, dict)
    # returns False, which causes reward_config.penalties.* to be silently
    # dropped in KernelGymHybridRewardParams.from_kernel_mapping.  Convert to
    # a plain Python dict here so all downstream .get() and isinstance() calls
    # work correctly regardless of whether Hydra is used.
    try:
        from omegaconf import OmegaConf
        kernel_cfg = OmegaConf.to_container(kernel_cfg, resolve=True, throw_on_missing=False)
    except Exception:  # omegaconf not installed or already a plain dict
        pass
    if not isinstance(kernel_cfg, dict):
        kernel_cfg = dict(kernel_cfg)

    kernel_server_url = kernel_cfg.get("server_url", "http://localhost:8000")
    max_turns = int(kernel_cfg.get("max_turns", 3))
    kernel_eval_timeout = int(kernel_cfg.get("timeout", 300))
    kernel_eval_client_timeout = kernel_cfg.get("task_timeout_in_client")
    if kernel_eval_client_timeout is not None:
        kernel_eval_client_timeout = int(kernel_eval_client_timeout)
    num_correct_trials = int(kernel_cfg.get("num_correct_trials", 5))
    num_perf_trials = int(kernel_cfg.get("num_perf_trials", 100))
    toolkit = kernel_cfg.get("toolkit", "kernelbench")
    backend_adapter = kernel_cfg.get("backend_adapter", toolkit)
    backend = kernel_cfg.get("backend", "cuda")
    reference_backend = kernel_cfg.get("reference_backend")
    system_prompt = kernel_cfg.get("system_prompt", None)
    max_retries = kernel_cfg.get("max_retries", 3)
    rate_limit = int(kernel_cfg.get("rate_limit", 8))
    acquire_timeout = int(kernel_cfg.get("acquire_timeout", 120))
    is_valid = bool(kernel_cfg.get("is_valid", True))
    detect_decoy_kernel = bool(kernel_cfg.get("detect_decoy_kernel", True))
    enable_profiling = bool(kernel_cfg.get("enable_profiling", True))
    verbose_errors = bool(kernel_cfg.get("verbose_errors", True))
    workflow = str(kernel_cfg.get("workflow", "kernelbench"))
    rerun_on_anomaly = bool(kernel_cfg.get("rerun_on_anomaly_speedup", True))
    early_exit_on_correct = bool(kernel_cfg.get("early_exit_on_correct", False))

    reward_params = KernelGymHybridRewardParams.from_kernel_mapping(kernel_cfg)
    reward_ops = KernelGymRewardOps(reward_params)

    async def rollout_fn(client, tokenizer, **kwargs):
        start_time = time.time()
        param_version_start = client.cur_version

        task = _row_kwargs_to_task(kwargs)

        result = await kernelgym_rollout(
            client=client,
            task=task,
            max_turns=max_turns,
            system_prompt=system_prompt,
            kernel_server_url=kernel_server_url,
            kernel_eval_timeout=kernel_eval_timeout,
            kernel_eval_client_timeout=kernel_eval_client_timeout,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            toolkit=toolkit,
            backend_adapter=backend_adapter,
            backend=backend,
            reference_backend=reference_backend,
            max_retries=max_retries,
            rate_limit=rate_limit,
            acquire_timeout=acquire_timeout,
            is_valid=is_valid,
            detect_decoy_kernel=detect_decoy_kernel,
            enable_profiling=enable_profiling,
            verbose_errors=verbose_errors,
            workflow=workflow,
            rerun_on_anomaly_speedup=rerun_on_anomaly,
            early_exit_on_correct=early_exit_on_correct,
            reward_ops=reward_ops,
        )

        trajectory = result["trajectory"]
        trajectory.reward = result["reward"]
        metrics = result["metrics"]

        end_time = time.time()
        param_version_end = client.cur_version
        processing_time = end_time - start_time

        metadata = {
            "processing_time": processing_time,
            "param_version_start": param_version_start,
            "param_version_end": param_version_end,
            "param_version": param_version_end,
            "is_partial": param_version_start != param_version_end,
        }
        metadata.update(metrics)
        trajectory.metadata = metadata

        if random.random() < 0.05:
            print(
                f"\n{'=' * 70}\n"
                f"[KernelGYM] problem_id={task.get('problem_id', 'N/A')}\n"
                f"  turns={metrics.get('num_turns')}  reward={result['reward']:.4f}\n"
                f"  compiled={metrics.get('compiled')}  correctness={metrics.get('correctness')}\n"
                f"  speedup={metrics.get('speedup')}  "
                f"gen_time={metrics.get('total_generation_time', 0):.1f}s  "
                f"eval_time={metrics.get('total_eval_time', 0):.1f}s\n"
                f"  param_version: {param_version_start} → {param_version_end}\n"
                f"{'=' * 70}\n"
            )

        return trajectory

    return rollout_fn


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="pkg://rllm.experimental.fully_async.config",
    config_name="fully_async_ppo_megatron_trainer",
    version_base=None,
)
def main(config):
    kernel_cfg = config.get("kernel", {})

    rollout_fn = make_rollout_fn(kernel_cfg)

    trainer = AsyncAgentTrainer(
        config=config,
        rollout_fn=rollout_fn,
        val_rollout_fn=rollout_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()
