---
name: triton-op-verifier
description: >
  算子代码验证——通过固定入口 tools/triton_eval_pipeline.sh 跑 AST 退化检查 + NPU 数值正确性
  + 性能采集，读 metrics.json 判定。禁止自写测试。触发：需要验证 Triton 算子正确性/性能时。
triggers:
  - verify
---

# Triton 算子验证 Skill

你负责验证 `output/submission/{op_name}_impl.py` 是否编译运行正确、与 `src/{op_name}.py` 参考实现输出一致，并采集性能数据。

⚠️ **禁止自写测试代码、禁止直接调用或修改 `scripts/` 下脚本、禁止改评测参数**——所有验证与性能采集**统一通过固定入口** `tools/triton_eval_pipeline.sh` 执行。它内部按固定顺序跑 AST 退化检查 → `verify.py`（NPU 数值）→ `benchmark.py`（性能），warmup/repeats/精度阈值全部写死。

## 用法

```bash
bash tools/triton_eval_pipeline.sh --op_name {op_name} \
    --impl output/submission/{op_name}_impl.py \
    --task src/{op_name}.py [--json src/{op_name}.json]
```

结果写入 `metrics.json`（位于 `--out_dir`，默认 `agent_workdir` 根）。

## metrics.json 关键字段

| 字段 | 含义 |
|------|------|
| `success` | 全流程是否成功（ast + correctness + perf 均通过） |
| `ast_check_ok` | AST 退化预检查是否通过 |
| `correctness_ok` | NPU 数值正确性是否通过 |
| `perf_data.speedup_vs_torch` | 相比 PyTorch 的加速比（**全 shape 几何平均**，异常 shape 不计入） |
| `error` / `error_type` | 失败信息（含 traceback）与分类 |

## 常见失败模式与修复方向（按 `error_type` 索引）

| error_type / 特征 | 修复方向 |
|---------|---------|
| AST 退化（Type 1/2/3） | 见 `AGENTS.md` 退化子类型：把 forward 的 torch 计算移进 `@triton.jit` kernel |
| `correctness_failed` / shape mismatch | 检查 grid/block 配置、输出 buffer 形状、broadcasting、规约轴 |
| `correctness_failed` / 数值超差 | 检查 dtype 转换、累加顺序（必要时 fp32）、边界处理 |
| `triton_compile_failed` / `triton_lowering_failed` | 检查 `tl.constexpr` 参数、内存对齐、指针表达式（别传 `.data_ptr()`） |
| `triton_ub_overflow` / OOM | 减小 `BLOCK_SIZE_{M,N,K}`，优化内存复用（多为 B 类，可能需终止） |
| `benchmark_failed` | 简化 kernel、减少访存、调 block size |
| 超时 | 检查无限循环或过大 grid |

## 精度判定（评测器内置，了解即可、不可改）

`verify.py` 用基于 dtype 的**相对误差双门限**（MERE/MARE，NPU Benchmark 标准），非 `torch.allclose`：比较前两侧升 float32；先查形状/NaN/Inf 位置一致，再在有限值上算相对误差。dtype 阈值随精度变化（如 float16 紧、bfloat16 松）。这些阈值由入口写死，**agent 不可调整**。
