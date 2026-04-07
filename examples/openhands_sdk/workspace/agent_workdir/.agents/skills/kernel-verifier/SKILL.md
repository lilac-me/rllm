---
name: kernel-verifier
description: >
  算子代码验证——AST 退化检查 + NPU 数值正确性 + torch_npu.profiler 性能采集。
  统一通过 tools/operator_pipeline.sh 执行。
triggers:
  - verify
---

# Kernel Verifier Skill

你是算子代码验证专家。负责验证生成的 Triton Ascend 算子是否编译运行正确、与 `src/{op_name}.py` 中参考实现输出一致，并采集性能数据。

⚠️ **禁止自写测试代码**——所有验证和性能采集统一通过 `operator_pipeline.sh` 执行。

## 验证流程（pipeline 内部自动执行）

```
AST 退化预检查 → 数值正确性验证（NPU） → 性能采集（torch_npu.profiler）
```

| 阶段 | 脚本 | 说明 |
|------|------|------|
| AST 检查 | `validate_triton_impl.py` | 静态分析，检测 PyTorch 退化（Type 1/2/3），无需 NPU |
| 正确性 | `verify.py` | NPU 上运行，对比 Model vs ModelNew 输出 |
| 性能 | `benchmark.py` | NPU 上运行，采集延迟和内存，计算 speedup |

## 使用方式

```bash
bash tools/operator_pipeline.sh --op_name <op_name>
```

结果写入 `metrics.json`，schema 见 `AGENTS.md`。

## 结果解读

### metrics.json 关键字段

| 字段 | 含义 |
|------|------|
| `ast_check_ok` | AST 退化预检查是否通过 |
| `correctness_ok` | 数值正确性是否通过 |
| `success` | 全流程是否成功（ast + correctness + perf 均通过） |
| `error` | 失败时的错误信息（含 traceback） |
| `perf_data.speedup_vs_torch` | 相比原生 PyTorch 的加速比 |

### 常见失败模式与修复方向

| 失败模式 | error 特征 | 修复方向 |
|---------|-----------|---------|
| PyTorch 退化 | `regression_type: 1/2/3` | 见 `AGENTS.md` 退化子类型说明，移除 forward() 中的 torch 计算 |
| 形状不匹配 | `shape mismatch` | 检查 kernel 的 grid/block 配置和输出 buffer 形状 |
| 数值超差 | `relative error exceeds` | 检查数据类型转换、累加顺序、边界处理 |
| 编译错误 | `triton.CompilationError` | 检查 tl.constexpr 参数、内存对齐 |
| NPU OOM | `out of memory` | 减小 BLOCK_SIZE 或优化内存复用（B 类错误，可能需终止） |
| 超时 | `timeout` | 检查是否有无限循环或过大的 grid |

## 精度阈值

验证使用基于数据类型的**相对误差**比较：

| 数据类型 | 阈值 | 说明 |
|---------|------|------|
| `float16` | 0.004 | 半精度 |
| `bfloat16` | 0.03 | BF16 精度较低 |
| `int8` | 0.01 | 整数量化 |
| 其他 (float32) | 0.02 | 默认 |

比较规则：形状一致 → NaN 位置一致 → Inf 位置/符号一致 → 有限值相对误差不超过阈值比例。
