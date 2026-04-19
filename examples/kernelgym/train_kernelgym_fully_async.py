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

from rllm.experimental.fully_async.runner import AsyncAgentTrainer

from .kernelgym_rollout import kernelgym_rollout


# ---------------------------------------------------------------------------
# Factory: build rollout_fn with kernel config captured via closure
# ---------------------------------------------------------------------------

def make_rollout_fn(kernel_cfg: dict):
    """Return an ``async def rollout_fn(client, tokenizer, **kwargs)`` with
    KernelGYM evaluation parameters baked in via closure.

    ``RolloutExecutor.generate_trajectory`` passes **only** the dataset-row
    fields as ``kwargs`` (e.g. ``reference_code``, ``problem_id``).  Kernel
    server URL, max turns, backend, etc. come from the Hydra config and must
    be captured here.
    """

    kernel_server_url = kernel_cfg.get("server_url", "http://localhost:8000")
    max_turns = int(kernel_cfg.get("max_turns", 3))
    kernel_eval_timeout = int(kernel_cfg.get("timeout", 300))
    num_correct_trials = int(kernel_cfg.get("num_correct_trials", 5))
    num_perf_trials = int(kernel_cfg.get("num_perf_trials", 100))
    toolkit = kernel_cfg.get("toolkit", "kernelbench")
    backend_adapter = kernel_cfg.get("backend_adapter", toolkit)
    backend = kernel_cfg.get("backend", "cuda")
    system_prompt = kernel_cfg.get("system_prompt", None)

    async def rollout_fn(client, tokenizer, **kwargs):
        start_time = time.time()
        param_version_start = client.cur_version

        task = dict(kwargs)

        result = await kernelgym_rollout(
            client=client,
            task=task,
            max_turns=max_turns,
            system_prompt=system_prompt,
            kernel_server_url=kernel_server_url,
            kernel_eval_timeout=kernel_eval_timeout,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            toolkit=toolkit,
            backend_adapter=backend_adapter,
            backend=backend,
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
    config_name="fully_async_ppo_trainer",
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
