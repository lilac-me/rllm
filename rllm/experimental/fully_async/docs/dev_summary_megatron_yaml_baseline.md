# 全异步配置基线迁移：PPO Megatron Trainer

本文档总结将 rllm 全异步训练配置从「以 `ppo_trainer` 为基座 + 运行时切 Megatron」迁移为「以 `ppo_megatron_trainer` 为官方基座」的改动，便于与 verl 默认 Megatron 行为对齐，并与仓库内已有全异步 Megatron 启动脚本（如 `examples/fully_async/verl_fully_async_mega.sh`）保持一致。

## 背景与动机

- `ppo_trainer.yaml` 通过 `model_engine` 在 `dp_actor` / `megatron_actor` 等之间切换，属于通用模板。
- `ppo_megatron_trainer.yaml` 在 Hydra `defaults` 中直接固定 `megatron_actor`、`megatron_ref`、`megatron_critic`，并在 `_self_` 中附带 Megatron 场景常用的 `actor_rollout_ref.model`（如 `override_config`、`lora`）、`rollout.layer_name_map`、`algorithm`、`trainer` 等默认值。
- 全异步训练要求 **独立 rollout 进程**，因此必须 **`actor_rollout_ref.hybrid_engine: False`**；该覆盖在专用全异步 yaml 中集中声明，避免各 shell 重复传参、遗漏或与基座 `hybrid_engine: True` 冲突。

## 本轮改动清单

| 文件 | 说明 |
|------|------|
| `rllm/experimental/fully_async/config/fully_async_ppo_megatron_trainer.yaml` | **新增**：`defaults` 首项为 `ppo_megatron_trainer`，`_self_` 中覆盖 `hybrid_engine: False`，挂载 `async_training`、`rollout`（全异步侧步数/拓扑等）、`data.gen_batch_size` 与 DatasetRegistry 相关占位字段；`actor_rollout_ref.checkpoint_engine` 与 `actor.use_rollout_log_probs` 与 `async_training` 联动。 |
| `examples/kernelgym/train_kernelgym_fully_async.py` | `@hydra.main` 的 `config_name` 由 `fully_async_ppo_trainer` 改为 `fully_async_ppo_megatron_trainer`。 |
| `examples/kernelgym/train_kernelgym_fully_async.sh` | 移除与 yaml 重复的 `actor_rollout_ref.hybrid_engine=False`；Megatron `override_transformer_config` 中与基座 schema 不一致的键仍使用 Hydra `+` 追加（与 `verl_fully_async_mega.sh` 一致）。 |

## 与 verl / 仓库脚本的对齐要点

- **配置名**：与 `examples/fully_async/verl_fully_async_mega.sh` 中 `--config-name='fully_async_ppo_megatron_trainer.yaml'` 一致（此前脚本已引用该文件名，本次补齐包内实现）。
- **`hybrid_engine`**：在 `fully_async_ppo_megatron_trainer.yaml` 中统一为 `False`，符合全异步「训练与推理分离」拓扑。
- **Shell 覆盖**：`actor_rollout_ref.actor.megatron.*` 等已在 `verl` 的 `engine/megatron.yaml` 中存在的字段可直接覆盖；对 `override_transformer_config` 下可能不在默认 schema 中的键（如部分环境下的 `context_parallel_size`、`use_flash_attn`）继续使用 `+key=value` 追加，降低 Hydra 结构校验风险。

## 与此前全异步改动（KernelGym + vLLM）的关系

- **KernelGym 集成**与 **RolloutClient / InferenceManager / RolloutExecutor** 等 vLLM 适配代码仅依赖 `OmegaConf` 上的字段（如 `actor_rollout_ref.rollout.name`、`async_training.*`），**不依赖** `ppo_trainer` 与 `ppo_megatron_trainer` 二选一；因此本轮仅调整 Hydra 组合与示例入口，无需修改上述 Python 实现。
- **`fully_async_ppo_trainer.yaml`** 仍保留，供非 Megatron（如 FSDP / `model_engine=dp`）全异步场景使用。

## 使用说明

- Megatron 全异步：在 Hydra 入口指定 `config_name=fully_async_ppo_megatron_trainer`（或等价 CLI `--config-name`），并按任务覆盖 `data.*`、`actor_rollout_ref.model.path`、`trainer.device` 等。
- 若需与 verl 上游 `ppo_megatron_trainer` 行为完全一致，请避免在未理解差异的情况下删除 `_self_` 中对 `hybrid_engine` 的覆盖。

## 后续可选工作

- 将其他示例（如 deepresearch）在明确使用 Megatron 时迁移到同一 `config_name`，减少「通用模板 + 大量 CLI 覆盖」带来的漂移。
- 在 CI 或文档中增加一次「仅 compose 配置、不拉起 Ray」的 Hydra smoke test，防止 config 文件名与包内文件不同步。
