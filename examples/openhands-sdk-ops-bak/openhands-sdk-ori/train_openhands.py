"""
Training entry point: OpenHands + AgentSDKEngine/AgentSDKTrainer.

Loads an SWE-bench-style dataset (or a custom one via DatasetRegistry) and
trains a model with PPO/GRPO using OpenHands as the agent runtime.

The agent runs inside Docker containers via worker_server.py (sandbox mode).
SandboxOrchestrator manages container lifecycle, metadata slug injection,
and result collection.

Run via train_openhands.sh or directly:

    python3 -m examples.openhands.train_openhands \\
        actor_rollout_ref.model.path=<your_model> \\
        rllm.sdk.proxy.mode=subprocess \\
        rllm.sdk.store.path=${HOME}/rllm-traces.db

Dataset format
--------------
Each row must have at least one extra_info field::

    {
        "instruction": "Fix the failing test in src/utils.py",
        "repo_url": "https://github.com/user/repo",   # optional
        "test_file": "tests/test_utils.py",            # optional
        "success_keywords": ["PASSED", "All tests ok"] # optional
    }

Register a custom dataset via DatasetRegistry or supply a parquet file
directly through `data.train_files` / `data.val_files` Hydra overrides.
"""

import hydra

from examples.openhands.openhands_agent import rollout
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # ------------------------------------------------------------------
    # Load datasets
    # Use "swe_bench" if registered; otherwise fall back to "countdown"
    # for a quick smoke-test with the default rllm datasets.
    # ------------------------------------------------------------------
    try:
        train_dataset = DatasetRegistry.load_dataset("swe_bench", "train")
        val_dataset = DatasetRegistry.load_dataset("swe_bench", "test")
    except Exception:
        import warnings
        warnings.warn(
            "swe_bench dataset not found in DatasetRegistry. "
            "Falling back to 'countdown' for a quick smoke-test. "
            "To use SWE-bench, register it via DatasetRegistry or override "
            "data.train_files / data.val_files in the shell launcher.",
            stacklevel=1,
        )
        train_dataset = DatasetRegistry.load_dataset("countdown", "train")
        val_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, (
        "Train dataset not found. Pass a dataset path via "
        "data.train_files=<path> or register a dataset in DatasetRegistry."
    )

    # ------------------------------------------------------------------
    # Build trainer with OpenHands rollout as the agent_run_func.
    #
    # The rollout function runs inside Docker containers managed by
    # SandboxOrchestrator. worker_server.py handles metadata slug
    # injection and result submission.
    # ------------------------------------------------------------------
    trainer = AgentTrainer(
        agent_run_func=rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()