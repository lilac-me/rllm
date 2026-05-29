# Stage2 Fit-Loop Audit: verl `RayPPOTrainer.fit` vs rllm `AgentSdkTrainer.fit_agent`

> 生成时间：2026-05-27
> 目的：W3 真训练前，一次性扫完"verl 自家 trainer 做了 X，rllm 没做"的同源遗漏（W2.25/26/28/29 都是这类 bug）。
> 范围：仅 stage2 目标 = "管线 enable + 单节点 + 真 reward + 短 N 步 sanity"。多节点 / condenser / reward 上升 / 100 步长跑全部 defer 到后续阶段。
> 配套：[QWEN36_OPENHANDS_AGENT_RL_PLAN.md](./QWEN36_OPENHANDS_AGENT_RL_PLAN.md) §9 stage2 TODO + §14 stage1 pre-context

---

## 0. 用户拍板的 stage2 锁定项（前置）

| Config | 锁定值 | 理由 |
|---|---|---|
| `use_critic` | **`False`** | stage2 GRPO 路径，不用 critic |
| `actor.calculate_sum_pi_squared` | **`False`** | stage1 默认；开了会撞 nested tensor 漏 post-process |
| `algorithm.rollout_correction` / router_replay | **`disabled`** | stage1 §13.7 已锁；19983fe4 入口在 AgentPPOTrainer 不在 AgentSdkTrainer |
| `rllm.rejection_sample.enable` | **`False`** | 单 turn 真 reward `is_correct=False` 概率极高，开了整 batch 被 drop → step skip 饿死 |
| `use_remove_padding` (model + megatron) | **`False`** | W2.29 已锁；GDN packed seq 不支持 |
| `OPENHANDS_MAX_ITERATIONS` | **`1`** | W2.22 已锁；condenser 接入前不能拉 |

**未来要解锁任何一项前**，必须重审下面对应 finding。

---

## 1. Audit 范围与方法

按 8 项对照 verl 在 batch 上的每一处 set/modify：

1. `batch.meta_info[...]` set 内容
2. `batch.non_tensor_batch[...]` set 内容
3. worker 调用前是否 `to_tensordict`
4. worker 调用前是否 `left_right_2_no_padding`
5. worker 调用前 `assign_non_tensor` 塞了哪些 metadata
6. worker 返回后 `no_padding_2_padding` 处理
7. 返回 key 是否 `rename_dict` / 重命名
8. metrics 如何包回 `DataProto.meta_info["metrics"]`

读取范围：
- verl `RayPPOTrainer.fit` (`verl/trainer/ppo/ray_trainer.py:1318-1730`)
- verl `_compute_old_log_prob` (1212-1247)
- verl `_compute_ref_log_prob` (1188-1210)
- verl `_update_actor` (1249-1289)
- verl `_compute_values` (1174-1186) / `_update_critic` (1291-1316)
- verl `_balance_batch` (1104-1172)
- rllm `AgentSdkTrainer.fit_agent` (`rllm/trainer/verl/agent_sdk_trainer.py:230-732`)
- rllm `AgentSdkTrainer.generate_trajectories` (823-841)
- rllm `AgentSdkExecutionEngine.execute_tasks_verl` (`rllm/engine/agent_sdk_engine.py:432-462`)
- rllm `AgentSdkExecutionEngine.transform_results_for_verl` (464-757)

---

## 2. Findings 总结

### 🔴 必做（stage2 阻塞 / 必锁）

| # | 差异 | 位置 | 影响 | 修法 |
|---|---|---|---|---|
| **F1** | `_compute_values` 无 bridge | `agent_sdk_trainer.py:529` 直接 `self.critic_wg.compute_values(batch)` 传 DataProto | use_critic=True 立即崩 | **lock `use_critic=False`** (§0)；不补 bridge |
| **F2** | `_update_critic` 无 bridge | `agent_sdk_trainer.py:592` 直接 `critic_wg.update_critic(batch)` | 同上 | 同上 |
| **F3** | 整 batch 全部失败时下游崩 | `fit_agent.py:351-353` 兜底被注释 `# if len(drop_uids) == len(unique_uids): continue` | stage2 真 reward 启动初期可能整批 `is_correct=False` 或 `repeat_counts` 全 0 → 后续 `union`/`balance_batch`/`old_log_prob` 在空 batch 上炸 | 打开 continue + skip log + 累加 metric `batch/skipped_all_drop` |
| **F4** | `agent_sdk_engine.py:624` 硬编码 `max_prompt_length = 16384` 覆盖 config | 同 line | stage1 容忍（mock reward 不在乎扔 14K tokens）；stage2 真 reward 需要真实 32K 信号 | 删硬编码行；让 line 622 读到的 config 值生效；同步删 `[DEBUG format]` print 残留（line 620/628/636/642/710-732） |

### 🟡 建议做（observability / silent fallback）

| # | 差异 | 位置 | 影响 | 修法 |
|---|---|---|---|---|
| **F5** | `rollout_log_probs` silent fallback：仅在 `all(len(logprobs) == len(response))` 时填 | `agent_sdk_engine.py:663-672` | 失败时 silently 跳过 `rollout_correction` / `calculate_debug_metrics`；下游不知道；训练信号不一致 | 加 metric `rollout/log_probs_fill_rate`；不一致时 warning log |
| **F6** | `_compute_old_log_prob` 不读 worker output 的 `mfu` 字段 | `agent_sdk_trainer.py:459-465` | verl `ray_trainer.py:1233/1518` 提取后写 `metrics["perf/mfu/actor_infer"]`；rllm 丢这个 observability | line 459 后加 `_mfu = tu.get(old_log_prob_td, "metrics", {}).get("mfu", 0.0)`，line 478 附近 `metrics["perf/mfu/actor_infer"] = _mfu` |

### 🟢 已确认对齐（W2.27 + W2.28 + W2.30 成果）

| 项 | verl 位置 | rllm 位置 | 状态 |
|---|---|---|---|
| `batch.meta_info["temperature"]` per-step set | 1395 / 1253 | 426 (W2.28) | ✅ |
| `batch.meta_info["multi_turn"]` | 1251 | 427 (W2.28) | ✅ |
| `batch.meta_info["global_token_num"]` | 1471 | 416 | ✅ |
| `batch.meta_info["images_seqlens"]` | 1478 | 436（多了 if guard） | ✅ |
| `_compute_old_log_prob` 桥接 8 步全套 | 1212-1247 | 440-484 (W2.27+W2.30) | ✅ |
| `_compute_ref_log_prob` 桥接 | 1188-1210 | 503-524 (W2.27+W2.30) | ✅ |
| `_update_actor` 桥接 + `rename_dict("actor/")` + `perf/mfu/actor` swap | 1249-1289 | 600-631 (W2.27+W2.30) | ✅ |
| `assign_non_tensor` 三处 metadata 内容 | 1194/1220/1271 | 444/506/609 | ✅ 全对齐 |
| `transform_results_for_verl` 输出 `response_mask` | (verl 1461-1462 兜底) | `agent_sdk_engine.py:700` 已填 | ✅ rllm 已填，verl 兜底不需要 |

### 🟢 设计差异（非漏 port，无需修）

| # | 差异 | verl | rllm | 备注 |
|---|---|---|---|---|
| **D1** | reward 路径 | `_compute_reward_colocate` + `extract_reward` + `reward_extra_infos_dict` | 直接读 `traj_rewards` / `step_rewards`（OpenHands 内算好） | 设计差异 |
| **D2** | `uid` naming + 设置时点 | step 开头 line 1398 直接设 `non_tensor_batch["uid"]` | line 287 设 `task_ids`；line 536 才 `uid = step_ids` | 下游 advantage / KL 不重叠；只要不开 `PrefixGrouper` (line 1119) 就 OK；**未来开 prefix_grouper 要复审** |
| **D3** | REMAX 路径 | 1409-1421 + 1443-1455 | 完全没实现 | stage1/2 不用 REMAX |
| **D4** | `response_mask` 语义 | `compute_response_mask` 算 attention-style mask | `agent_sdk_engine.py:700` 写入的是 `traj_mask` | **per-step row schema 下两者等价**：每行 response 就是单个 LLM completion，traj_mask 全 1 等于 attention-style response_mask。未来若改 trajectory-level row schema（一行装多 turn + tool_result）才会不等价 — 当前不走这条路。建议 `agent_sdk_engine.py:700` 加一行注释说明 |
| **D5** | `meta_info["timing"]` 由 generate 输出 | line 1436-1437 pop | rllm 用 `marked_timer("gen")` 替代 | 等价 |
| **D6** | `gen_batch.meta_info["global_steps"]` | line 1405 设 | `execute_tasks_verl` 不消费 | stage1/2 OpenHands 不需要 step 号；trace 想要可后补 |
| **D7** | `compute_variance_proxy_metrics` / GDPO reward keys / `train_dataset.on_batch_end` | 1690-1706 / 1725-1727 | 缺 | 纯 observability 或 experimental |
| **D8** | `async_calls_finalize_fn_exec` (verl 主线最近加的 async hook) | 1383-1384 / 1716-1717 | 不调 | **vllm-ascend 和 verl-fork workers 都不暴露此方法**（grep 验证）；rllm 不调是正确的 |

---

## 3. Out of audit scope（在 plan §9 已 track）

这些不是 fit-loop 桥接漏 port 问题，所以不在 audit 主清单，但与 stage2 相关：

| 项 | 类型 | plan §9 状态 | 备注 |
|---|---|---|---|
| W2.8 worker re-export shim 移回 rllm 侧 | 侧选择 / fork 债清理 | 🟡 推迟 | 功能完全正常；revert verl-fork `85159408` + rllm 4 处 import 改新名；技术上可做到 "verl 一行不动" 但保留 shim 当 dead code；行为差异风险 (`AsyncActorRolloutRefWorker` → `ActorRolloutRefWorker`) 没坏不修 |
| `§0.1 / §0.2` 软约束实测后状态回填 | 文档动作 | 待办 | audit lesson #22；stage2 之前过一遍 plan 表把 "可调 / 起步" 改成 "已实测 = X" |

---

## 4. 行动清单（stage2 范围）

| Pri | 动作 | 改动文件 | 工作量 |
|---|---|---|---|
| 🔴 P0-1 | Plan §14.5 加 §0 列的 6 项 stage2 锁定项 | `QWEN36_OPENHANDS_AGENT_RL_PLAN.md` | 半小时文档 |
| 🔴 P0-2 | 打开 `fit_agent.py:351-353` 整 batch 全 drop 兜底 continue + skip log + metric | `rllm/trainer/verl/agent_sdk_trainer.py` | ~10 行 |
| 🔴 P0-3 | 删 `agent_sdk_engine.py:624` 硬编码 + 删 `[DEBUG format]` print 残留 | `rllm/engine/agent_sdk_engine.py` | ~15 行净删 |
| 🟡 P1-1 | `rollout_log_probs` 填充率 metric + 不一致 warning | `rllm/engine/agent_sdk_engine.py` | ~10 行 |
| 🟡 P1-2 | `_compute_old_log_prob` 提取 worker mfu 写入 metrics | `rllm/trainer/verl/agent_sdk_trainer.py` | ~3 行 |
| 🟢 nice-to-have | `agent_sdk_engine.py:700` 加注释解释 schema | 同 | 1 行 |
| 🟢 defer | F1/F2 `_compute_values`/`_update_critic` bridge | — | use_critic=False 锁定后不必做 |
| 🟢 defer | W2.8 shim 移回 rllm 侧 | — | plan §9 已 track |

---

## 5. 验收

🔴 三项做完后跑：

```bash
# NPU 节点
cd /workspace/rllm-071
git pull   # 拿到 P0-2/P0-3 改动
bash examples/openhands_sdk/stage1_test_layered.sh L3
```

期望：
- L3 dry-step 仍然 PASS（不引入 regression）
- 看到 `Saved checkpoint` log
- 真 reward 函数被调用（grep production log 看 `real_reward=` 出现）

🟡 两项做完后短跑 10-20 step 单节点 8-NPU，关注：
- `batch/skipped_all_drop` metric（F3 兜底触发频率）
- `rollout/log_probs_fill_rate` metric（F5 silent fallback 频率）
- `perf/mfu/actor_infer` metric（F6 observability）
- `rollout_probs_diff` 不发散
- ckpt save/resume 数值一致

不要求 reward 上升 / 收敛——那是后续阶段目标。
