---
name: operator-metrics
description: >
  metrics.json / profiling_results.json 字段约定、success 判定逻辑、
  性能指标解读与调试要点。
  在查看或调试 metrics.json 时使用。
triggers:
  - metrics
  - profiling
  - benchmark
  - reward
  - success
  - latency_ms
---

# 算子指标文件 Skill

> 由 `tools/operator_pipeline.sh` 调用 `tools/render_metrics.py` 自动写入，**不要手动创建或修改**。

---

## 1. metrics.json — 完整字段说明

```json
{
  "schema_version": 1,
  "backend": "ascendc",
  "compile_ok": true,
  "correctness_ok": true,
  "profile_ok": true,
  "success": true,
  "latency_ms": 0.42,
  "execution_time_ms": 0.42,
  "bandwidth_gbps": 312.5,
  "error": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | int | 固定为 `1`，供训练侧版本识别 |
| `backend` | string | `ascendc` / `triton` / `mock` / `unknown` |
| `compile_ok` | bool | 编译步骤是否成功 |
| `correctness_ok` | bool | 正确性验证是否通过 |
| `profile_ok` | bool | 性能采集是否成功（可为 false 而不影响 success）|
| `success` | bool | **整体判定**，见下方逻辑说明 |
| `latency_ms` | float \| null | 算子执行延迟（毫秒）；采集失败时为 null |
| `execution_time_ms` | float \| null | 与 `latency_ms` 相同（兼容字段）|
| `bandwidth_gbps` | float | 访存带宽（GB/s）；采集失败时为 0.0 |
| `error` | string \| null | 错误描述；无错误时为 null |

---

## 2. profiling_results.json — 训练侧 reward 解析格式

```json
{
  "success": true,
  "bandwidth_gbps": 312.5,
  "execution_time_ms": 0.42,
  "error": null
}
```

与 `metrics.json` 保持一致；训练侧 reward 函数读取此文件。

---

## 3. success 判定逻辑

```
success = compile_ok AND correctness_ok
```

- `profile_ok = false`（性能采集失败）**不影响** `success`，但会影响 reward 评分。
- `error` 字段包含 `"fail"` 关键字时，强制 `success = false`。
- **含义**：编译通过 + 结果正确 = 基础成功；性能达标 = 高分奖励。

---

## 4. 读取与调试

```bash
# 读取完整指标
cat metrics.json

# 快速检查是否成功
python3 -c "import json; d=json.load(open('metrics.json')); print('SUCCESS' if d['success'] else 'FAILED:', d.get('error'))"

# 查看性能数字
python3 -c "import json; d=json.load(open('metrics.json')); print(f"latency={d['latency_ms']}ms  bw={d['bandwidth_gbps']}GB/s")"
```

---

## 5. 常见异常状态

| 状态 | compile_ok | correctness_ok | success | 含义与处理 |
|------|-----------|---------------|---------|-----------|
| 编译失败 | false | false | false | 检查 `error` 字段；修复代码后重跑流水线 |
| 编译通过但结果错误 | true | false | false | 对比 reference；检查数值精度与边界 |
| 全部通过但性能差 | true | true | true | success=true 但 latency 过高；继续优化 |
| 性能采集失败 | true | true | true | profile_ok=false；性能数字缺失，reward 可能降低 |
| 完全成功 | true | true | true | latency / bandwidth 均有值；正常完成 |

---

## 6. 何时认为任务完成

满足以下全部条件：

1. `compile_ok: true`
2. `correctness_ok: true`
3. `success: true`
4. 若 `INSTRUCTIONS.md` 中有性能目标（如 `latency < 1ms`），`latency_ms` 满足该目标
5. `error: null`
