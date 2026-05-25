"""
Stage1 training entry point: Qwen3.6-35B-A3B × OpenHands × Ascend NPU.

This is the dedicated stage1 entry point per QWEN36_OPENHANDS_AGENT_RL_PLAN.md
§7.1 / §4.3. Functionally equivalent to ``train_open_megatron.py``; kept as a
separate file so stage1 hydra defaults / dataset selection / future Qwen3.6-
specific tweaks do not perturb the legacy entry point.

OpenHands runs entirely inside Docker containers — no openhands Python
package is needed in the training environment.

The rollout() function in openhands_agent.py:
  1. Calls _build_proxied_base_url() to embed rllm metadata into the LLM URL
  2. Launches an OpenHands container with LLM_BASE_URL set to the proxied URL
  3. Waits for completion and returns a trajectory dict

Companion shell wrapper: ``train_openhands_qwen36_npu.sh``.

Stage1 baselines (per plan v2.4):
  - router_replay disabled (no-op on AgentPPOTrainer path; see §13.3)
  - KL loss disabled (use_kl_loss=False, kl_loss_coef=0); rely on PPO clip
  - Monitor rollout_probs_diff / pg_clipfrac / approx_kl (see §13.7)
  - max_model_len = 32k (per plan §5.2)
  - TP=2 EP=4 single-node 8-NPU layout (per user decision 2026-05-25)

Run::

    bash examples/openhands_sdk/train_openhands_qwen36_npu.sh

Or directly (rare — most config comes from the .sh wrapper)::

    python3 -m examples.openhands_sdk.train_openhands_qwen36_npu \\
        actor_rollout_ref.model.path=<model> \\
        rllm.sdk.proxy.mode=subprocess

算子 agent bring-up（mock 数据 + 容器内 mock profiling）::

    export OPENHANDS_DATASET=mock_npu
    bash examples/openhands_sdk/train_openhands_qwen36_npu.sh
"""

import os
import warnings

import hydra

# Layered testing hatch: STAGE1_MOCK_ROLLOUT=1 swaps in mock_rollout.rollout
# (deterministic fake trajectory, no docker/LLM) so Layer 5 (training step)
# can be exercised in isolation from Layer 4 (real rollout infra).
if os.environ.get("STAGE1_MOCK_ROLLOUT", "0") not in ("", "0", "false", "False"):
    from examples.openhands_sdk.mock_rollout import rollout

    print("[stage1] STAGE1_MOCK_ROLLOUT=1 set; using mock_rollout.rollout (no docker/LLM)")
else:
    from examples.openhands_sdk.openhands_agent import rollout

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

_EX_DIR = os.path.dirname(os.path.abspath(__file__))
_MOCK_NPU_PARQUET = os.path.join(_EX_DIR, "rl_single_ops.parquet")


class _MockNPUOperatorParquetDataset:
    """占位数据集：仅提供 verl 所需的 parquet 路径（extra_info 含 scenario=npu_operator）。"""

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def get_verl_data_path(self) -> str:
        return _MOCK_NPU_PARQUET

    def get_data(self) -> list:
        return []


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    dataset_mode = os.environ.get("OPENHANDS_DATASET", "mock_npu").strip().lower()

    if dataset_mode in ("mock_npu", "npu_operator", "npu"):
        train_dataset = _MockNPUOperatorParquetDataset("train")
        val_dataset = _MockNPUOperatorParquetDataset("test")
    else:
        try:
            train_dataset = DatasetRegistry.load_dataset("swe_bench", "train")
            val_dataset = DatasetRegistry.load_dataset("swe_bench", "test")
        except Exception:
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

    # Layered testing hatch: skip .train() when running preflight-only.
    # The companion .sh wrapper sets PREFLIGHT_ONLY=1 to exercise Layer 1
    # (config + dataset + AgentTrainer construct) without spinning up Ray /
    # Megatron / docker / vllm.
    if os.environ.get("PREFLIGHT_ONLY", "0") not in ("", "0", "false", "False"):
        print("[stage1] PREFLIGHT_ONLY=1 set; skipping trainer.train() and exiting OK")
        return

    trainer.train()


if __name__ == "__main__":
    main()
