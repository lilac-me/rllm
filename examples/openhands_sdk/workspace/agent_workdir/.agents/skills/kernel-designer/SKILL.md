---
name: kernel-designer
description: >
  Triton Ascend 算法草图设计——根据任务描述设计算法思路（sketch），
  用于指导后续代码生成。不生成可执行代码。
triggers:
  - sketch
---

# Triton Ascend 算法草图设计 Skill

你是高性能计算的算法设计专家。根据任务描述设计算法思路（sketch），用于指导 kernel-generator 生成代码。

⚠️ **仅生成算法草图**（UnifiedSketch DSL），不生成可执行代码。

## 参考文档索引

按需读取（路径相对于 workspace 根目录）：

- **必选**：`.agents/skills/kernel-designer/references/sketch-design.md`（UnifiedSketch DSL 语法、核心操作、设计模式）
- **硬件规格**：`.agents/skills/kernel-generator/references/hw-ascend910*.md`（按目标 arch 选择）
- **hint 模式**：任务描述含 `@hint:` / `@range_hint` 时加载 `.agents/skills/kernel-designer/references/hint-mode.md`

### 手写优化案例（按算子类型选最相关的 2 个）

选择依据：算子类型匹配 > 数据规模接近 > 优化模式相似。

| 类别 | 案例文件 | 核心优化 |
|------|---------|---------|
案例路径前缀：`.agents/skills/kernel-designer/references/cases/`

| **Elementwise** | `elemwise-broadcast-2d.md` | 2D 广播：小维不切分 |
| | `elemwise-broadcast-3d.md` | 跨轴 3D 广播：两阶段 kernel |
| | `elemwise-cast.md` | int8→fp16：二次切分 |
| | `elemwise-concat.md` | Slice+Concat 融合 |
| | `elemwise-zeros.md` | 小 shape：少核减调度 |
| **Index** | `index-histogram.md` | 预排序 + 二分查找 |
| | `index-put.md` | 批量 load 索引 |
| **MatMul** | `matmul-swizzle2d.md` | Swizzle2D 块重排 |
| **Reduction** | `reduction-amax-large.md` | reduce 轴多核 + 原子 + 二次切分 |
| | `reduction-amax-medium.md` | 矩阵累加再归约 |
| | `reduction-amax-small.md` | 极小 shape：grid=1 |
| | `reduction-amin-atomic.md` | 两种原子方案对比 |
| | `reduction-amin-large.md` | 超大 1D 二次切分 |
| | `reduction-amin-medium.md` | 大 N 维矩阵 min |
| | `reduction-amin-small.md` | 1D 并行度平衡 |
| | `reduction-mean-large.md` | mean 行二次切分 |
| | `reduction-mean-medium.md` | mean reduce 重组 |
| | `reduction-prod-small.md` | tl.reduce + 自定义 mul |
| | `reduction-sum-fused.md` | elemwise + sum 融合 |
| | `reduction-sum-large.md` | 大规模 sum 重组 |
| | `reduction-weighted-swiglu.md` | 3D SwiGLU backward |

---

## 设计流程

1. 阅读 `src/{op_name}.py` 中 `Model.forward()` 的参考实现
2. 判断算子类型（elementwise / reduce / matmul / attention / 复合）
3. 根据硬件架构选择并行化策略、Tile 大小（考虑 NPU UB 容量和对齐）
4. 用 UnifiedSketch DSL 设计算法草图

## 输出要求

直接输出 `sketch op_name { ... }` 格式的算法草图。
若任务描述含 hint 标记，在草图末尾附"设计适用范围"注释（格式见 `hint-mode.md`）。

## 设计原则

- 关注算法逻辑和优化策略，而非实现细节
- 遵循 Ascend NPU 硬件特性（core 级并行、内存层次）
- 标注优化点和权衡决策（使用 `@llm_hint` 注解）
- 数值正确性优先，性能次之
