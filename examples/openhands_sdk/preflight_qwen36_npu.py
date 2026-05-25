"""
Stage1 preflight: Layer 1 (config + import + dataset + trainer-construct)
isolation check.

Runs the cheap parts of the train pipeline (no Ray, no NPU, no Megatron load,
no docker, no LLM) so that Layer 1 problems (broken imports, hydra config
typos, missing parquet, AgentTrainer constructor mismatch) fail in seconds
instead of after a 5-minute warm-up of the real training stack.

Usage::

    # Quick check (no overrides, uses the same hydra config as the train script)
    python3 -m examples.openhands_sdk.preflight_qwen36_npu

    # With the same overrides as the .sh wrapper (pass-through to hydra)
    python3 -m examples.openhands_sdk.preflight_qwen36_npu \\
        actor_rollout_ref.model.path=<model> \\
        +rllm.algorithm.router_replay=disabled

Exit codes:
    0   all checks passed
    1   one of the checks failed (last line of stderr says which)
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Layer 1.1: hard imports (stage1 critical modules)
# ---------------------------------------------------------------------------
_STAGE1_IMPORTS = [
    # rllm core
    "rllm",
    "rllm.sdk",
    "rllm.engine",
    "rllm.engine.agent_execution_engine",
    "rllm.engine.rollout.openai_engine",
    "rllm.engine.rollout.verl_engine",
    "rllm.engine.rollout.rollout_engine",
    "rllm.parser.chat_template_parser",
    # trainer entry chain
    "rllm.trainer.agent_trainer",
    "rllm.trainer.verl.train_agent_ppo",
    "rllm.trainer.verl.agent_ppo_trainer",
    "rllm.trainer.verl.ray_runtime_env",
    # experimental (cherry-picked machinery: rollout_correction, metrics)
    "rllm.experimental.verl.metrics",
    "rllm.experimental.verl.transform",
    "rllm.experimental.verl.dataclass",
    "rllm.experimental.common.config",
    # stage1 entry + rollout func
    "examples.openhands_sdk.openhands_agent",
    "examples.openhands_sdk.train_openhands_qwen36_npu",
]


def _check_imports() -> list[str]:
    """Return list of failure messages; empty list = all OK."""
    fails: list[str] = []
    for mod in _STAGE1_IMPORTS:
        try:
            importlib.import_module(mod)
            print(f"  [import] OK   {mod}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            fails.append(f"{mod}: {msg}")
            print(f"  [import] FAIL {mod}: {msg}")
            traceback.print_exc(limit=3)
    return fails


# ---------------------------------------------------------------------------
# Layer 1.2: mock_npu dataset is wired and parquet file is readable
# ---------------------------------------------------------------------------
def _check_dataset() -> list[str]:
    fails: list[str] = []
    try:
        from examples.openhands_sdk.train_openhands_qwen36_npu import (
            _MOCK_NPU_PARQUET,
            _MockNPUOperatorParquetDataset,
        )

        ds = _MockNPUOperatorParquetDataset("train")
        path = ds.get_verl_data_path()
        if path != _MOCK_NPU_PARQUET:
            fails.append(f"dataset.get_verl_data_path() returned {path!r} != _MOCK_NPU_PARQUET {_MOCK_NPU_PARQUET!r}")

        if not os.path.isfile(_MOCK_NPU_PARQUET):
            fails.append(
                f"mock_npu parquet not found at {_MOCK_NPU_PARQUET}; "
                "run `python3 -m examples.openhands_sdk.create_mock_npu_operator_data` first"
            )
            print(f"  [dataset] FAIL parquet missing: {_MOCK_NPU_PARQUET}")
            return fails

        # Try to read with pyarrow (verl-internal) to confirm schema is loadable.
        try:
            import pyarrow.parquet as pq

            tbl = pq.read_table(_MOCK_NPU_PARQUET)
            n_rows = tbl.num_rows
            cols = tbl.column_names
            print(f"  [dataset] OK   parquet rows={n_rows} cols={cols}")
            # Sanity: verl expects at least `prompt` (string|list[dict]) and `extra_info` cols.
            # Be lenient — different bring-up scripts may use slightly different names.
            required_any = [{"prompt", "extra_info"}, {"messages", "extra_info"}]
            if not any(req.issubset(cols) for req in required_any):
                fails.append(
                    f"parquet schema missing expected verl columns; got {cols}. "
                    "Expected at least one of: {prompt, extra_info} or {messages, extra_info}."
                )
        except ImportError:
            print("  [dataset] WARN pyarrow not installed; skipping parquet schema check")
        except Exception as e:
            fails.append(f"parquet read failed: {type(e).__name__}: {e}")
            print(f"  [dataset] FAIL parquet read: {e}")
    except Exception as e:
        fails.append(f"dataset construction crashed: {type(e).__name__}: {e}")
        print(f"  [dataset] FAIL {e}")
        traceback.print_exc(limit=3)
    return fails


# ---------------------------------------------------------------------------
# Layer 1.3: rollout func signature matches AgentTrainer expectation
# ---------------------------------------------------------------------------
def _check_rollout_signature() -> list[str]:
    fails: list[str] = []
    try:
        import inspect

        from examples.openhands_sdk.openhands_agent import rollout

        if not callable(rollout):
            fails.append("openhands_agent.rollout is not callable")
            return fails

        sig = inspect.signature(rollout)
        params = sig.parameters
        # rollout(*args, **kwargs) per current implementation;
        # require either varargs or at least known positional names.
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_varkwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not (has_varargs and has_varkwargs):
            fails.append(
                f"rollout signature {sig} unusual; expected (*args, **kwargs) "
                "to match AgentTrainer.agent_run_func contract"
            )
        else:
            print(f"  [rollout] OK   signature {sig}")
    except Exception as e:
        fails.append(f"rollout import crashed: {type(e).__name__}: {e}")
        print(f"  [rollout] FAIL {e}")
        traceback.print_exc(limit=3)
    return fails


# ---------------------------------------------------------------------------
# Layer 1.4: hydra config composes + has the stage1-required keys
# ---------------------------------------------------------------------------
_STAGE1_REQUIRED_KEYS = [
    # We assert presence in the *composed* config (after hydra defaults),
    # not the value (script wrapper sets values via CLI overrides).
    "algorithm.adv_estimator",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.kl_loss_coef",
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size",
    "actor_rollout_ref.rollout.calculate_log_probs",
    "actor_rollout_ref.rollout.max_model_len",
    "trainer.n_gpus_per_node",
    "trainer.total_epochs",
]


def _check_hydra_config(overrides: list[str]) -> list[str]:
    fails: list[str] = []
    try:
        from hydra import compose, initialize_config_module
        from omegaconf import OmegaConf

        with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
            cfg = compose(config_name="agent_ppo_trainer_megatron", overrides=overrides)

        for key in _STAGE1_REQUIRED_KEYS:
            try:
                val = OmegaConf.select(cfg, key, throw_on_missing=False)
                if val is None:
                    fails.append(f"hydra config missing key: {key}")
                    print(f"  [hydra] FAIL missing: {key}")
                else:
                    print(f"  [hydra] OK   {key} = {val}")
            except Exception as e:
                fails.append(f"hydra config key access {key} failed: {e}")
                print(f"  [hydra] FAIL key {key}: {e}")
    except Exception as e:
        fails.append(f"hydra compose crashed: {type(e).__name__}: {e}")
        print(f"  [hydra] FAIL compose: {e}")
        traceback.print_exc(limit=3)
    return fails


# ---------------------------------------------------------------------------
# Layer 1.5: AgentTrainer can be constructed (no .train() call)
# ---------------------------------------------------------------------------
def _check_trainer_construct(overrides: list[str]) -> list[str]:
    """Construct AgentTrainer with the composed config but do not call train().

    This is the deepest preflight check — if it passes, you're confident
    the Layer 1 pipeline (config + dataset + agent_run_func) is wired right.
    """
    fails: list[str] = []
    try:
        from hydra import compose, initialize_config_module

        from examples.openhands_sdk.openhands_agent import rollout
        from examples.openhands_sdk.train_openhands_qwen36_npu import (
            _MockNPUOperatorParquetDataset,
        )
        from rllm.trainer.agent_trainer import AgentTrainer

        with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
            cfg = compose(config_name="agent_ppo_trainer_megatron", overrides=overrides)

        train_ds = _MockNPUOperatorParquetDataset("train")
        val_ds = _MockNPUOperatorParquetDataset("test")

        trainer = AgentTrainer(
            agent_run_func=rollout,
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
        )
        # Don't call trainer.train()
        print(f"  [trainer] OK   AgentTrainer constructed: {type(trainer).__name__}")
    except Exception as e:
        fails.append(f"AgentTrainer construct crashed: {type(e).__name__}: {e}")
        print(f"  [trainer] FAIL {e}")
        traceback.print_exc(limit=5)
    return fails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    # Pass-through CLI overrides (same form as the .sh wrapper).
    overrides = sys.argv[1:]

    print("=" * 70)
    print("Stage1 preflight (Layer 1 isolation, no NPU/Megatron/docker)")
    print(f"Overrides ({len(overrides)}):")
    for o in overrides:
        print(f"    {o}")
    print("=" * 70)

    layer_checks = [
        ("imports", _check_imports, ()),
        ("dataset", _check_dataset, ()),
        ("rollout signature", _check_rollout_signature, ()),
        ("hydra config", _check_hydra_config, (overrides,)),
        ("AgentTrainer construct", _check_trainer_construct, (overrides,)),
    ]

    total_fails: list[tuple[str, str]] = []
    for label, fn, args in layer_checks:
        print(f"\n--- Layer 1.{label} ---")
        fails = fn(*args)
        for f in fails:
            total_fails.append((label, f))
        if fails:
            # Fail-fast (per user decision): don't move to next layer
            break

    print("\n" + "=" * 70)
    if total_fails:
        print(f"PREFLIGHT FAILED ({len(total_fails)} issues):")
        for layer, msg in total_fails:
            print(f"  [{layer}] {msg}")
        print("=" * 70)
        return 1
    print("PREFLIGHT OK — Layer 1 healthy, ready for Layer 2+ smoke")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
