---
name: triton-kernel
description: >
  Triton 算子完整实现指南：@triton.jit 语法、tile/mask、dtype 约定、
  reference 对比、autotuner 使用、Ascend NPU 适配差异。
  任务后端为 triton 时必须查阅。
triggers:
  - triton
  - tile
  - jit
  - tl.program_id
---

# Triton 算子实现 Skill

> 本 Skill 是 `AGENTS.md` 第 5 节的技术补充，提供更完整的 Triton 编程参考。

---

## 1. Triton 与 AscendC 的关系

- **AscendC** 是华为官方 NPU 编程接口，直接控制 AI Core / UB，性能上限高，但语法复杂。
- **Triton**（OpenAI）在 NPU 上通过适配层运行，语法更接近 Python，适合快速原型。
- 任务中若未特别指定，**默认优先使用 `ascendc`**；`triton` 作为备用后端。

---

## 2. 最小可运行模板

```python
# src/triton/operator.py
import triton
import triton.language as tl
import torch


# ─── Triton Kernel ────────────────────────────────────────────────────────────
@triton.jit
def my_kernel(
    x_ptr,           # 输入指针
    y_ptr,           # 输出指针
    n_elements,      # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 编译期常量（tile 大小）
):
    # 1. 确定当前 program 负责的数据区间
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # 必须有 mask，处理尾部

    # 2. 加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # 3. 计算（示例：乘以 2）
    result = x * 2.0

    # 4. 存储结果
    tl.store(y_ptr + offsets, result, mask=mask)


# ─── Python 启动器（必须提供）────────────────────────────────────────────────
def forward(x: torch.Tensor) -> torch.Tensor:
    """Triton 算子启动函数，由工具链调用。"""
    x = x.contiguous()          # 确保内存连续
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    my_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK)
    return y


# ─── Reference 实现（必须提供，用于正确性对比）────────────────────────────────
def forward_reference(x: torch.Tensor) -> torch.Tensor:
    """等价的 PyTorch 实现，作为 ground truth。"""
    return x * 2.0
```

> **`forward_reference` 是必须项**：正确性验证脚本会调用它与 `forward` 的输出做对比。

---

## 3. 关键语法速查

### Program ID 与分片
```python
pid = tl.program_id(axis=0)          # 1D grid
pid_x = tl.program_id(axis=0)        # 2D grid，x 方向
pid_y = tl.program_id(axis=1)        # 2D grid，y 方向
```

### 数据加载与存储
```python
x = tl.load(ptr + offsets, mask=mask, other=0.0)   # 带 mask 加载
tl.store(ptr + offsets, value, mask=mask)           # 带 mask 存储
```

### 常用数学操作
```python
tl.exp(x)        # e^x
tl.log(x)        # ln(x)
tl.sqrt(x)       # √x
tl.abs(x)        # |x|
tl.maximum(x, y) # max(x, y) 逐元素
tl.minimum(x, y) # min(x, y) 逐元素
tl.where(cond, x, y)  # 条件选择
```

### 规约（在 tile 内部）
```python
s = tl.sum(x, axis=0)     # 在 BLOCK_M 轴上求和
m = tl.max(x, axis=0)     # 在 BLOCK_M 轴上取最大值
```

### 矩阵乘（GEMM 相关）
```python
# 加载 2D tile
a = tl.load(a_ptr + offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak, mask=...)
b = tl.load(b_ptr + offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn, mask=...)
c = tl.dot(a, b)  # 内积，输出为 float32（自动提升）
```

### dtype 转换
```python
x = x.to(tl.float16)   # 转换为 fp16
x = x.to(tl.float32)   # 转换为 fp32（规约前常用）
```

---

## 4. Autotuner（性能自动调优）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],   # 当此参数变化时重新调优
)
@triton.jit
def my_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

> Autotuner 会在第一次调用时测试所有配置并缓存最优结果；适合性能要求高的场景。

---

## 5. 常见边界问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 最后一个 tile 结果错误 | 忘记 `mask` | 加 `mask = offsets < n_elements` |
| 访问越界（Segfault / NaN） | mask 未传给 `tl.load` | `tl.load(..., mask=mask, other=0.0)` |
| 非连续 tensor 结果不对 | 未调用 `.contiguous()` | 在 `forward()` 开头加 `x = x.contiguous()` |
| 精度差异大 | 规约用 fp16 溢出 | 规约前 `x = x.to(tl.float32)` |

---

## 6. 正确性验证要求

工具链会自动对比 `forward(x)` 与 `forward_reference(x)` 的结果：

```python
# 验证标准（来自任务 INSTRUCTIONS.md，以下为默认值）
assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), \
    f"Max diff: {(out - ref).abs().max()}"
```

- 若精度下降必须说明（如：使用 fp16 导致误差增大，在输出中注明）
- bfloat16 与 float16 的精度特性不同，不可混用

---

## 7. 执行流程

1. 确认 `OPERATOR_BACKEND=triton`（`INSTRUCTIONS.md` 或环境变量）
2. 参考本 Skill 的模板写 `src/triton/operator.py`
3. 同时提供 `forward_reference` 函数（PyTorch 参考实现）
4. 执行 `bash tools/operator_pipeline.sh`
5. 读取 `metrics.json`；失败则按第 5 节排查

> **不要**修改 `tools/` 目录下任何文件；所有编译命令由工具链管理。
