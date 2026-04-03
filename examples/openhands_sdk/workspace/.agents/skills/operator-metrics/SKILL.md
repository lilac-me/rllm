---
name: operator-metrics
description: >
  metrics.json / profiling_results.json 字段约定与 success 判定。
  在用户提到 metrics、profiling、reward、性能指标时使用。
triggers:
  - metrics
  - profiling
  - benchmark
---

# 指标文件约定

由 `tools/operator_pipeline.sh` 在 workspace 根目录写入：

## metrics.json（完整）

- `schema_version`：整数，当前为 `1`。
- `backend`：`triton` | `ascendc` | `mock` | `unknown`。
- `compile_ok` / `correctness_ok`：布尔。
- `success`：通常为 `compile_ok && correctness_ok`（profile 失败时仍可为 true，由流水线策略决定）。
- `latency_ms` / `execution_time_ms` / `bandwidth_gbps`：数值或 `null`。
- `error`：字符串或 `null`。

## profiling_results.json（兼容训练）

- `success`：布尔（与 `metrics.success` 一致）。
- `bandwidth_gbps`：浮点（占位或实测）。
- `execution_time_ms`：可选。
- `error`：可选。
