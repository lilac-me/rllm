# Triton-Ascend 算子生成 Agent — 全局约定

## 固定配置

- **framework**: torch  |  **dsl**: triton_ascend  |  **backend**: ascend

## 环境规则

- OpenHands SDK 运行在 **venv** 里；编译 / 验证 / 性能采集运行在 **conda 环境** 里。
- **禁止手动 `conda activate`**——conda 操作由 `tools/` 脚本内部处理。
- **禁止修改** `tools/` 目录。

### 网络与依赖（软约束）

本环境 **不提供可用的外网**（即使执行 `apt` / `pip install` / `curl` / `wget` 也会失败或不应依赖）。你必须 **只使用工作区内已有文件与 `tools/` 管线**，通过修改 `src/` 与运行 `bash tools/operator_pipeline.sh` 完成任务。**不要**尝试安装系统包、下载 wheel、或访问外部 URL 来「修复环境」；遇到缺依赖类错误，应调整算子实现或向 `metrics.json` / 脚本输出对齐排查，而不是联网。

## 工作流（Phase 0/1 已在 host 侧完成）

| Phase | 内容 | 技能 | 迭代上限 |
|-------|------|------|---------|
| 2 | 算法设计（可选，简单算子可跳过）→ `src/sketch.txt` | kernel-designer | 1 |
| 3 | 代码生成与验证（核心循环）| kernel-generator + kernel-verifier | 5 |
| 4 | 性能优化（⚠️ **必须执行，禁止跳过**）| latency-optimizer + kernel-verifier | 3 |


### 何时运行 pipeline

⚠️ **只有在 `src/{op_name}_triton_ascend_impl.py` 已写入完整可执行代码后**，才运行：

```bash
bash tools/operator_pipeline.sh --op_name <op_name>
```

分析任务、阅读文档、设计草图、编写代码等步骤**不需要也不应该**运行 pipeline。

### 最佳版本追踪

每次 pipeline 报告 `success: true` 后，比较 `metrics.json` 中的 `speedup_vs_torch` 与已保存的最佳版本：

- **首次成功**或**性能更优**时，保存为最佳版本：
  ```bash
  cp src/{op_name}_triton_ascend_impl.py src/{op_name}_triton_ascend_impl_best.py
  cp metrics.json metrics_best.json
  ```
- **任务结束时**（达到迭代上限、B/C 类终止），若当前 `metrics.json` 不是成功状态**且最佳版本存在**，回退：
  ```bash
  if [ -f metrics_best.json ]; then
    cp src/{op_name}_triton_ascend_impl_best.py src/{op_name}_triton_ascend_impl.py
    cp metrics_best.json metrics.json
  fi
  ```

最终产物始终是**编译通过、精度正确、性能最优**的版本。

## 禁止 PyTorch 退化（最重要的约束）

`ModelNew.forward()` 中**所有核心计算**必须在 `@triton.jit` kernel 中实现。

### forward() 中禁止的操作

- `torch.matmul / torch.relu / torch.sum` 等计算函数
- `F.softmax / F.linear` 等 `torch.nn.functional`
- `x.sum() / x.mean() / x @ w` 等 tensor 方法/运算符
- `self.conv(x) / self.linear(x)` 等 nn.Module 调用

### forward() 中允许的操作

- buffer 分配：`torch.empty / torch.zeros / torch.ones`
- 形状操作：`.view / .reshape / .permute / .transpose / .contiguous`
- 元信息：`.shape / .dtype / .device / .numel()`
- kernel 启动：`kernel[grid](...)`

## 错误分类与迭代决策

### 分类规则

| 类型 | 含义 | 决策 |
|------|------|------|
| **A 类** | 代码逻辑/算法错误 | 可修复，继续迭代 |
| **B 类** | 环境/基础设施错误 | 不可修复，终止 |
| **C 类** | 同一 A 类子类型连续 ≥ 3 次 | 终止 |

### A 类常见子类型

| 子类型 | error 特征 |
|--------|-----------|
| PyTorch 退化 Type 1 | 完全无 `@triton.jit` kernel |
| PyTorch 退化 Type 2 | 有 kernel 但 `forward()` 未调用 |
| PyTorch 退化 Type 3 | `forward()` 调用了 kernel 但部分计算仍用 PyTorch |
| 输出不一致 | 数值精度差异、算法实现与参考不同 |
| 语法/类型错误 | SyntaxError、TypeError、IndentationError |
| 形状不匹配 | Tensor shape mismatch、维度错误 |
| Kernel 参数错误 | BLOCK_SIZE 不合理、grid 配置错误 |
| DSL API 错误 | Triton API 参数错误、不支持的操作 |

### B 类常见子类型

| 子类型 | error 特征 |
|--------|-----------|
| 设备不可用 | NPU OOM、device not found |
| 依赖缺失 | ModuleNotFoundError（非代码导致） |
| 超时 | Timeout、进程被杀死 |

### 迭代失败时的分析模板

每次迭代失败后，按以下格式进行结构化分析再修复：

```
错误分析：
- 类型：{A/B/C}（{子类型}）
- 位置：{错误代码位置}
- 具体错误：{error 字段内容}

修复建议：
1. {具体修改方向}

历史提醒：
- 第 N 轮曾因 {问题} 失败，避免重复
```

## 文件布局

```
.
├── INSTRUCTIONS.md
├── src/
│   ├── {op_name}.py                       # 任务文件（host 写入）
│   ├── {op_name}_triton_ascend_impl.py    # Agent 实现（当前版本）
│   ├── {op_name}_triton_ascend_impl_best.py     # 最佳成功版本备份（成功过才有）
│   └── sketch.txt                         # 可选
├── metrics_best.json                      # 最佳成功版本的 metrics（成功过才有）
├── tools/                                 # 只读
│   ├── operator_pipeline.sh
│   ├── env.sh
│   └── scripts/{validate_triton_impl,verify,benchmark}.py
├── metrics.json
└── profiling_results.json
```

## metrics.json schema

```json
{
  "schema_version": 2,
  "op_name": "softmax",
  "success": true,
  "ast_check_ok": true,
  "correctness_ok": true,
  "perf_data": {
    "framework_latency_ms": 1.23,
    "impl_latency_ms": 0.56,
    "speedup_vs_torch": 2.17,
    "peak_memory_mb": 128.0
  },
  "error": null
}
```
