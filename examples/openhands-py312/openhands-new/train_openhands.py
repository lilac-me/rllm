"""
Training entry point: OpenHands + rllm (PPO/GRPO).

OpenHands runs entirely inside Docker containers — no openhands Python
package is needed in the training environment.

The rollout() function in openhands_agent.py:
  1. Calls _build_proxied_base_url() to embed rllm metadata into the LLM URL
  2. Launches an OpenHands container with LLM_BASE_URL set to the proxied URL
  3. Waits for completion and returns a trajectory dict

Run via train_openhands.sh or directly::

    python3 -m examples.openhands_sdk.train_openhands \\
        actor_rollout_ref.model.path=<model> \\
        rllm.sdk.proxy.mode=subprocess \\
        rllm.sdk.sandbox.enabled=False
"""

import hydra

from examples.openhands_sdk.openhands_agent import rollout
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    try:
        train_dataset = DatasetRegistry.load_dataset("swe_bench", "train")
        val_dataset = DatasetRegistry.load_dataset("swe_bench", "test")
    except Exception:
        import warnings
        warnings.warn(
            "swe_bench not found; falling back to 'countdown' for smoke-test.",
            stacklevel=1,
        )
        train_dataset = DatasetRegistry.load_dataset("countdown", "train")
        val_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, "Train dataset not found."

    trainer = AgentTrainer(
        agent_run_func=rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
