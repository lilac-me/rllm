# Vector Compare 优化模式

## 概述

在 Triton NPU kernel 中，**当代码中涉及数值比较操作时**，如果比较操作数是整数类型（如 i64/i32），cmp 操作会退化为 scalar 计算，导致性能下降。**通过将比较数值转换为 fp32 类型**，可以启用 NPU 的 vector 加速单元，利用向量化加速，显著提升计算效率。

## 触发条件

**当 Triton 代码中出现数值比较操作时，应考虑应用此优化**，特别是以下场景：

1. **tl.where 中的条件判断**：使用整数索引与数据进行比较
2. **条件选择操作**：根据索引范围选择不同的计算路径
3. **边界处理**：处理不规则数据块（尾块、边界块）
4. **任何涉及整数类型比较的运算**：如 `cols < N`、`idx >= start` 等

## 优化方法

### 原始代码（scalar 计算）

```python
cols = tl.arange(0, BLOCK_N)  # cols 类型为 i64
xbar = tl.where(cols < N, x - mean, 0.0)  # 退化为 scalar 计算
```

### 优化后代码（vector 计算）

```python
cols = tl.arange(0, BLOCK_N)  # cols 类型为 i64
cols_cmp = cols.to(tl.float32)  # 转换为 fp32
xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # 启用 vector 计算
```

## 关键点

1. **类型转换**：使用 `.to(tl.float32)` 将整数索引转换为浮点数
2. **适用范围**：`tl.where` 中的比较操作需要手动转换
3. **自动优化**：在 `tl.load`/`tl.store` 的 mask 参数中使用 cmp 时，编译器通常会自动优化，无需手动处理

## 性能收益

将 scalar cmp 转换为 vector cmp 操作，可在 NPU 上获得显著的性能提升，尤其在处理不规则数据块时效果明显。
