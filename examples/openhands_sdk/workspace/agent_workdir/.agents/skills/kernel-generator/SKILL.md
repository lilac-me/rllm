---
name: kernel-generator
description: >
  Triton Ascend 算子代码生成——根据任务和可选 sketch 生成 @triton.jit kernel + ModelNew。
  严格禁止 PyTorch 退化。支持首次生成和基于错误反馈的迭代修复。
triggers:
  - generate
---

# Triton Ascend 代码生成 Skill

你是高性能算子代码生成专家。根据任务描述和可选的算法草图（sketch），生成完整的 `@triton.jit` kernel + `ModelNew` 类。

⚠️ **禁止 PyTorch 退化**：forward() 中所有核心计算必须在 `@triton.jit` kernel 中实现。
禁止/允许操作的完整规则见 `AGENTS.md`。

### ❌ 错误示例

```python
# Type 1：完全无 kernel
def forward(self, x, w):
    return torch.matmul(x, w)

# Type 2：有 kernel 但 forward 未调用
@triton.jit
def matmul_kernel(...):
    pass

def forward(self, x, w):
    return torch.matmul(x, w)  # kernel 定义了但没用

# Type 3：混合实现（部分 kernel + 部分 torch）
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y.sum(dim=-1)  # ← 违规
```

### ✅ 正确示例

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    tl.store(output_ptr + idx, x + y)

class ModelNew(nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        add_kernel[(1,)](x, y, output, x.numel(), BLOCK_SIZE=128)
        return output
```

---

## 参考文档索引

按需读取（路径相对于 workspace 根目录）：

- **硬件规格**：`.agents/skills/kernel-generator/references/hw-ascend910*.md`（按目标 arch 选择）
- **编程基础**：`.agents/skills/kernel-generator/references/triton-ascend-fundamentals.md`
- **完整示例**：`.agents/skills/kernel-generator/references/triton-ascend-examples.md`
- **按算子类型**：
  - Elementwise → `.agents/skills/kernel-generator/references/triton-ascend-elementwise.md`
  - MatMul → `.agents/skills/kernel-generator/references/triton-ascend-matmul.md`
  - Reduce → `.agents/skills/kernel-generator/references/triton-ascend-reduce.md`
  - Attention → `.agents/skills/kernel-generator/references/triton-ascend-attention.md`

融合算子加载所有相关文档。

---

## 算法草图

若 `src/sketch.txt` 存在，**必须以草图为基础**实现。无草图则自行设计。

---

## 代码生成模式

### 首次生成

1. 阅读 `src/{op_name}.py` 中 `Model.forward()` 的参考实现
2. 判断算子类型，读取对应参考文档
3. 选择并行化策略和内存访问模式
4. 生成 kernel 函数和 `ModelNew` 类

### 迭代修复（验证失败后）

1. 分析 `metrics.json` 中的错误，理解失败原因
2. 保留正确部分，只改有问题的地方
3. 不做不必要的大规模重构
4. 避免重犯历史错误

---

## 输出要求

完整的单文件 Python，结构如下：

```python
import torch, torch.nn as nn, triton, triton.language as tl

@triton.jit
def {op_name}_kernel(...):
    ...

class ModelNew(nn.Module):
    def __init__(self, <与原 Model 相同的参数>):
        super().__init__()
        ...
    def forward(self, <与原 Model 相同的输入>):
        ...
        return output
```

关键约束：
- 类名必须为 **`ModelNew`**（不是 `Model`）
- `__init__` / `forward` 签名与原 `Model` 完全一致
- 所有 kernel 和辅助函数在同一文件内，可直接导入运行

---

## 含随机权重算子的权重一致性

当原 `Model` 含 `nn.Conv2d` / `nn.Linear` / `nn.Parameter(torch.randn(...))` 等随机参数时，
验证框架会在创建两个模型前分别调用 `torch.manual_seed(0)`，因此 `ModelNew.__init__` 中
**必须按原 Model 相同顺序**创建参数以保证权重一致。

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ...):
        super().__init__()
        torch.manual_seed(0)  # 第一行固定种子
        # 按原 Model 相同顺序创建模块并提取权重
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if conv.bias is not None else None

    def forward(self, x):
        return custom_conv_kernel(x, self.weight, self.bias, ...)
```

要点：`__init__` 第一行 `torch.manual_seed(0)`；参数创建顺序与原 Model 完全一致；
通过创建同类型 `nn.Module` 提取权重（而非手动 `torch.randn`）。
