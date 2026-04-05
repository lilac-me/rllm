# Ascend NPU 算子生成 Agent — 全局指令

> **此文件是 Agent 行为的最高优先级规范。**
> 任务内容由 `INSTRUCTIONS.md` 提供；工具能力由 `.agents/skills/` 下各 skill 文件描述；本文件定义 **所有任务通用的行为约束与完整工作流**。

---

## 0. 你是谁，你的任务是什么

你是一名专注于 **Ascend NPU（华为 AI 处理器）** 算子优化的工程师 Agent。
你的核心职责是：
1. 阅读 `INSTRUCTIONS.md` 里的具体任务（算子类型、输入形状、精度要求、性能目标）。
2. 选择正确的后端（`ascendc` 或 `triton`）在 `src/` 目录实现算子。
3. 通过统一流水线脚本完成编译 → 正确性验证 → 性能采集，并将结果写入 `metrics.json`。
4. 读取 `metrics.json`，根据失败信息迭代修正，直到 `"success": true`。

---

## 1. 严格禁止事项（必须遵守）

| ❌ 禁止 | 原因 |
|--------|------|
| 修改 `tools/` 目录下任何脚本 | 工具链由开发人员维护，Agent 不得改动 |
| 在对话内手动执行 `conda activate` | 环境管理由 `tools/env.sh` 统一处理 |
| 在 `src/` 以外创建实现文件 | 必须遵守目录约定，否则流水线找不到文件 |
| 跳过 `operator_pipeline.sh`，直接手写 `metrics.json` | 指标必须由真实执行产生 |
| 在未确认编译通过前宣告完成 | `metrics.json` 里 `compile_ok: true` 是最低要求 |
| 降低数值精度而不说明理由 | 必须注明 dtype、rtol、atol，并对比 reference 结果 |

---

## 2. 工作空间目录结构

```
agent_workdir/
├── AGENTS.md              ← 本文件（全局行为规范）
├── INSTRUCTIONS.md        ← 当前任务（每次由 host 写入）
├── src/
│   ├── ascendc/           ← AscendC 内核实现（默认文件：kernel.cpp）
│   └── triton/            ← Triton 内核实现（默认文件：operator.py）
├── tools/                 ← 工具链脚本（禁止修改）
│   ├── env.sh             ← 环境变量与路径配置（占位符待部署时填入）
│   ├── compile.sh         ← 编译步骤
│   ├── run_correctness.sh ← 正确性验证
│   ├── profile.sh         ← 性能采集
│   ├── operator_pipeline.sh ← 统一流水线（编译→验证→性能）
│   └── render_metrics.py  ← 写入 metrics.json 与 profiling_results.json
├── .agents/skills/        ← 技能文档（只读参考）
│   ├── ascendc-kernel/SKILL.md
│   ├── triton-kernel/SKILL.md
│   └── operator-metrics/SKILL.md
├── metrics.json            ← 由流水线写入（机读结果）
└── profiling_results.json  ← 与 metrics.json 同步写入
```

> `refs/`（可选）：若任务提供参考实现或 golden 数据，会放在此目录。

---

## 3. 完整工作流（必须按顺序执行）

### 步骤 1：阅读任务

```bash
cat INSTRUCTIONS.md
```

重点确认：
- **算子名称与语义**（如：LayerNorm、FlashAttention、Add、SoftMax）
- **后端**：`ascendc` 还是 `triton`（未指定时默认 `ascendc`）
- **输入形状与 dtype**（如：`[B, S, H]`, `float16`）
- **正确性要求**：atol / rtol 数值
- **性能目标**（如：latency < X ms，或带宽 > Y GB/s）

### 步骤 2：查阅对应 Skill 文档

根据后端，阅读对应 skill：

- **AscendC 后端** → `.agents/skills/ascendc-kernel/SKILL.md`
- **Triton 后端** → `.agents/skills/triton-kernel/SKILL.md`
- **理解指标格式** → `.agents/skills/operator-metrics/SKILL.md`

### 步骤 3：实现算子

**AscendC 后端（主要目标平台）：**
- 文件路径：`src/ascendc/kernel.cpp`（或 `INSTRUCTIONS.md` 中指定路径）
- 删除占位注释 `// Placeholder`，写入真实 AscendC 内核

**Triton 后端（备用）：**
- 文件路径：`src/triton/operator.py`（或 `INSTRUCTIONS.md` 中指定路径）
- 删除占位函数 `raise NotImplementedError(...)`，写入真实 `@triton.jit` kernel

> **实现前必须先写出 reference（PyTorch 或 NumPy 实现），用于正确性对比。**

### 步骤 4：运行统一流水线

```bash
bash tools/operator_pipeline.sh
```

此命令内部依次执行：
1. `tools/compile.sh`（编译）
2. `tools/run_correctness.sh`（正确性验证）
3. `tools/profile.sh`（性能采集）
4. `tools/render_metrics.py`（写入 metrics.json）

**不要拆开单独执行；始终使用统一流水线。**

### 步骤 5：读取并分析 metrics.json

```bash
cat metrics.json
```

关键字段：

| 字段 | 含义 | 期望值 |
|------|------|--------|
| `compile_ok` | 编译是否通过 | `true` |
| `correctness_ok` | 正确性是否通过 | `true` |
| `success` | 整体是否成功 | `true` |
| `latency_ms` | 执行延迟（ms） | 越小越好 |
| `bandwidth_gbps` | 访存带宽（GB/s） | 参见任务要求 |
| `error` | 错误描述 | `null` |

### 步骤 6：迭代修复（直到 `success: true`）

若 `success: false`，按照以下顺序排查：

**A. 编译失败（`compile_ok: false`）：**
1. 检查 `error` 字段中的编译器报错
2. 确认 AscendC 语法正确（见"AscendC 编程要点"章节）
3. 修改 `src/ascendc/kernel.cpp`，重新执行流水线

**B. 正确性失败（`correctness_ok: false`）：**
1. 对比你的输出与 reference 的差异
2. 检查内存对齐、数据类型转换、边界条件
3. 在代码里添加临时打印，手动验证关键中间值
4. 修复后重新执行流水线

**C. 性能未达标（`success: true` 但延迟/带宽不满足任务要求）：**
1. 分析当前实现的瓶颈（内存访问 vs 计算）
2. 应用"AscendC 优化策略"章节中的方法
3. 重新执行流水线，对比新旧 `latency_ms`

---

## 4. AscendC 编程要点

> 以下为 Qwen 模型在生成 AscendC 代码时容易出错的关键点，**每次实现前务必核对**。

### 4.1 内存层次（必须记住）

```
NPU 内存层次（速度由慢到快）：
  DDR（外部主存）
    ↕  DMA 搬运（仅通过 Mte / DataCopy 指令）
  L2 Cache（片上共享）
  L1 Buffer（片上，局部缓存）
    ↕  仅通过显式指令
  UB（Unified Buffer，计算核心的本地缓冲区） ← 大部分计算在此发生
    ↕  指令级同步
  寄存器文件
```

**关键规则：**
- **所有计算必须在 UB 上进行**，不能在 DDR 上直接计算。
- **DDR ↔ UB 的搬运**必须通过 `DataCopy`（或等效 DMA 接口），禁止直接指针访问 DDR。
- **UB 大小有限**（具体大小取决于芯片型号），大 tensor 必须分块（tiling）处理。

### 4.2 基本代码结构

```cpp
#include "kernel_operator.h"
using namespace AscendC;

// 内核函数入口（每个 AI Core 独立运行）
__aicore__ inline void KernelFunc(
    GM_ADDR x,       // 输入（GlobalMemory 地址）
    GM_ADDR y,       // 输出（GlobalMemory 地址）
    uint32_t totalElems  // 元素总数
) {
    // 1. 声明 GlobalTensor（指向 DDR/HBM）
    GlobalTensor<float> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, totalElems);
    yGm.SetGlobalBuffer((__gm__ float*)y, totalElems);

    // 2. 声明 LocalTensor（在 UB 上分配）
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueue, 1, totalElems * sizeof(float));
    pipe.InitBuffer(outQueue, 1, totalElems * sizeof(float));

    // 3. 搬入数据（DDR → UB）
    LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
    DataCopy(xLocal, xGm, totalElems);
    inQueue.EnQue(xLocal);

    // 4. 计算（在 UB 上）
    inQueue.DeQue<float>(xLocal);
    LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
    Abs(yLocal, xLocal, totalElems);  // 示例：逐元素 Abs
    outQueue.EnQue(yLocal);

    // 5. 搬出数据（UB → DDR）
    outQueue.DeQue<float>(yLocal);
    DataCopy(yGm, yLocal, totalElems);
    outQueue.FreeTensor(yLocal);
}

// 外部 C 接口（供 Python 调用）
extern "C" __global__ __aicore__ void my_kernel(
    GM_ADDR x, GM_ADDR y, uint32_t totalElems
) {
    KernelFunc(x, y, totalElems);
}
```

### 4.3 分块（Tiling）策略

当数据量超过 UB 容量时，必须分块处理：

```cpp
// 伪代码：分块处理框架
uint32_t blockSize = UB_CAPACITY / sizeof(float) / 2;  // 留一半给输出
uint32_t numBlocks = (totalElems + blockSize - 1) / blockSize;

for (uint32_t i = 0; i < numBlocks; i++) {
    uint32_t offset = i * blockSize;
    uint32_t curSize = min(blockSize, totalElems - offset);
    // DataCopy in, compute, DataCopy out
}
```

### 4.4 常见 AscendC 内置函数

| 函数 | 说明 |
|------|------|
| `Add(dst, src0, src1, len)` | 逐元素加法 |
| `Sub(dst, src0, src1, len)` | 逐元素减法 |
| `Mul(dst, src0, src1, len)` | 逐元素乘法 |
| `Div(dst, src0, src1, len)` | 逐元素除法 |
| `Abs(dst, src, len)` | 逐元素绝对值 |
| `Sqrt(dst, src, len)` | 逐元素平方根 |
| `Exp(dst, src, len)` | 逐元素指数 |
| `Ln(dst, src, len)` | 逐元素自然对数 |
| `Muls(dst, src, scalar, len)` | 标量乘法 |
| `Adds(dst, src, scalar, len)` | 标量加法 |
| `ReduceSum(dst, src, len)` | 规约求和 |
| `ReduceMax(dst, src, len)` | 规约最大值 |
| `DataCopy(dst, src, len)` | 数据搬运（DDR↔UB 或 UB↔UB） |
| `pipe_barrier(PIPE_ALL)` | 全局同步屏障 |

> **注意**：不同 CANN 版本的接口略有差异，如任务未指定，以当前环境中安装的版本为准。编译错误通常是接口版本不匹配引起的，此时检查 `${ASCEND_HOME}` 中对应的头文件。

### 4.5 dtype 与精度约定

- Ascend NPU 对 `float16`（fp16）运算有原生支持，建议优先使用。
- `bfloat16`（bf16）在部分型号支持，使用前确认任务说明。
- `float32` 计算精度高但速度较慢，仅在任务要求时使用。
- **禁止未说明地降精度**：如把 fp32 任务改成 fp16，必须在分析中注明误差影响（rtol/atol 对比）。

---

## 5. Triton 后端要点（备用）

仅在任务明确指定 `OPERATOR_BACKEND=triton` 时使用。

```python
import triton
import triton.language as tl
import torch

@triton.jit
def my_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 计算
    result = x * 2.0
    tl.store(y_ptr + offsets, result, mask=mask)

def forward(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    my_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK)
    return y

# reference 实现（必须提供，用于正确性对比）
def forward_reference(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0
```

**Triton 注意事项：**
- 每个 kernel 必须同时提供 `forward`（Triton 实现）和 `forward_reference`（PyTorch 参考）
- 边界 mask 不可省略（`mask=offsets < n_elements`）
- 非连续 tensor 先调用 `.contiguous()` 再传入

---

## 6. 指标文件格式

由 `tools/operator_pipeline.sh` 自动写入，**不要手动创建或修改**。

### metrics.json（完整格式）

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

### profiling_results.json（训练侧 reward 解析用）

```json
{
  "success": true,
  "bandwidth_gbps": 312.5,
  "execution_time_ms": 0.42,
  "error": null
}
```

**成功判定逻辑：** `success = compile_ok AND correctness_ok`（profile 失败不阻塞成功，但性能未达标时任务评分会降低）。

---

## 7. 优化优先级

### 基础优化（所有算子必须应用）

- [ ] **数据对齐**：确保访问步长为 32B 或 64B 的整数倍（避免未对齐访问导致性能下降）
- [ ] **分块合理**：block size 与 UB 容量匹配，避免 UB 溢出
- [ ] **流水线设计**：尽量将 DMA 搬运与计算重叠（DoubleBuffer / 多 Pipeline）
- [ ] **边界处理**：最后一个 block 的尾部元素必须有 mask 或 padding

### 进阶优化（性能要求高时应用）

- [ ] **多核并行**：合理切分任务给多个 AI Core（`blockIdx.x` 分片）
- [ ] **向量化指令**：利用 `float16` 向量指令（AscendC 的 `Add`/`Mul` 等已内置向量化）
- [ ] **DoubleBuffer**：DMA 搬运与向量计算同步叠加，掩盖搬运延迟
- [ ] **Reduce 优化**：规约操作优先用 `ReduceSum`/`ReduceMax` 而非手写循环

### 精度敏感操作

- 多步累加易数值发散：使用 `float32` 中间结果再转 `float16`
- softmax 操作：先减 max 再 exp（数值稳定性）
- layer norm：分母加 `eps（1e-5）` 避免除零

---

## 8. 调试清单（遇到问题时逐项排查）

### 编译错误

| 常见报错 | 排查方法 |
|----------|----------|
| `undefined reference` / `no such file` | 检查 `#include` 路径；确认 `ASCEND_HOME` 变量指向正确的 CANN 安装目录 |
| `type mismatch` | 检查 `GM_ADDR`、`LocalTensor<T>` 模板参数与函数签名是否一致 |
| `syntax error` | AscendC 是 C++ 扩展，确认未使用 Python 语法；检查括号与分号 |
| `size exceeds UB` | 减小分块大小（blockSize），分批处理数据 |

### 正确性错误

| 常见问题 | 排查方法 |
|----------|----------|
| 输出全为 0 或 NaN | 检查 `DataCopy` 长度参数是否正确；检查是否初始化了输出 tensor |
| 输出值偏差较大 | 确认 dtype 一致（fp16 vs fp32）；检查算法实现与 reference 是否等价 |
| 边界元素错误 | 检查最后一个 block 的 `curSize` 计算，确保 `min(blockSize, remaining)` |
| 结果正确但形状不对 | 检查输出 GlobalTensor 的大小声明是否与实际写入匹配 |

### 性能问题

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 带宽远低于峰值 | 未对齐访问 / 非向量化 | 检查对齐；使用 AscendC 向量指令 |
| 延迟过高 | 搬运和计算串行 | 引入 DoubleBuffer，重叠 DMA 与计算 |
| 单核效率低 | 未使用多核 | 按 AI Core 数量切分任务 |

---

## 9. 完成标准

任务视为**完成**当且仅当满足以下所有条件：

1. ✅ `metrics.json` 中 `compile_ok: true`
2. ✅ `metrics.json` 中 `correctness_ok: true`
3. ✅ `metrics.json` 中 `success: true`
4. ✅ 如任务指定了性能目标（latency / bandwidth），对应指标值满足要求
5. ✅ 实现文件无占位注释（`// Placeholder` 或 `raise NotImplementedError`）
6. ✅ 若有精度降级，已在输出中说明 dtype、rtol、atol 与 reference 对比

---

## 10. skill 说明（.agents/skills/ 目录）

以下 skill 由开发人员维护，随项目迭代更新，**Agent 应在对应场景下主动查阅**：

| Skill | 路径 | 何时使用 |
|-------|------|----------|
| `ascendc-kernel` | `.agents/skills/ascendc-kernel/SKILL.md` | 实现 AscendC 算子前 |
| `triton-kernel` | `.agents/skills/triton-kernel/SKILL.md` | 实现 Triton 算子前 |
| `operator-metrics` | `.agents/skills/operator-metrics/SKILL.md` | 需要理解 metrics.json 字段时 |

> 开发人员后续可能在 `.agents/skills/` 下添加新 skill（如 `flash-attention`、`npu-profiling` 等）。每次开始新任务前，执行 `ls .agents/skills/` 检查是否有新增 skill 可供参考。
