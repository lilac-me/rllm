---
name: ascendc-kernel
description: >
  AscendC 算子完整实现指南：内存层次（UB/L1/DDR）、API 速查、
  分块策略、DoubleBuffer、多 AI Core 并行、编译调试。
  凡任务后端为 ascendc / ascend / npu kernel 时必须查阅。
triggers:
  - ascendc
  - ascend
  - npu
  - cann
---

# AscendC 算子实现 Skill

> 本 Skill 是 `AGENTS.md` 第 4 节的技术补充，提供更完整的 AscendC 编程参考。

---

## 1. 内存层次速记

| 层次 | 符号 | 关键特性 |
|------|------|----------|
| 全局内存（DDR/HBM） | `GM_ADDR` / `GlobalTensor<T>` | 容量大；不可直接计算，只能用 DataCopy 搬运 |
| Unified Buffer | `LocalTensor<T>` | 在此做向量运算；容量有限（约 256 KB 级别）
| L1 Cache | （自动管理或显式 Reload）| 部分场景下可缓存复用数据 |

**核心原则**：数据流向必须是 `DDR → UB（DataCopy）→ 计算 → UB → DDR（DataCopy）`。

---

## 2. 标准内核框架

```cpp
#include "kernel_operator.h"
using namespace AscendC;

// ① 声明全局内存视图
GlobalTensor<DTYPE> xGm, yGm;
xGm.SetGlobalBuffer((__gm__ DTYPE*)x_addr, total_elems);
yGm.SetGlobalBuffer((__gm__ DTYPE*)y_addr, total_elems);

// ② 创建 Pipeline 管理 UB 队列
TPipe pipe;
TQue<QuePosition::VECIN,  1> inQ;
TQue<QuePosition::VECOUT, 1> outQ;
pipe.InitBuffer(inQ,  1, block_size * sizeof(DTYPE));
pipe.InitBuffer(outQ, 1, block_size * sizeof(DTYPE));

// ③ 主循环（分块）
for each block:
    LocalTensor<DTYPE> xL = inQ.AllocTensor<DTYPE>();
    DataCopy(xL, xGm[offset], cur_size);     // DDR → UB
    inQ.EnQue(xL);
    inQ.DeQue<DTYPE>(xL);

    LocalTensor<DTYPE> yL = outQ.AllocTensor<DTYPE>();
    YOUR_COMPUTE_OP(yL, xL, cur_size);        // UB 上计算

    outQ.EnQue(yL);
    outQ.DeQue<DTYPE>(yL);
    DataCopy(yGm[offset], yL, cur_size);      // UB → DDR
    outQ.FreeTensor(yL);
    inQ.FreeTensor(xL);  // 某些版本需要显式释放
```

---

## 3. 向量计算 API 速查（AscendC 内置）

### 逐元素二元运算
```cpp
Add(dst, src0, src1, len);   // dst[i] = src0[i] + src1[i]
Sub(dst, src0, src1, len);   // dst[i] = src0[i] - src1[i]
Mul(dst, src0, src1, len);   // dst[i] = src0[i] * src1[i]
Div(dst, src0, src1, len);   // dst[i] = src0[i] / src1[i]
Maximum(dst, src0, src1, len); // dst[i] = max(src0[i], src1[i])
Minimum(dst, src0, src1, len); // dst[i] = min(src0[i], src1[i])
```

### 逐元素一元运算
```cpp
Abs(dst, src, len);
Sqrt(dst, src, len);
Exp(dst, src, len);
Ln(dst, src, len);
Reciprocal(dst, src, len);  // 1/x
Relu(dst, src, len);
```

### 标量运算
```cpp
Muls(dst, src, scalar, len);  // dst[i] = src[i] * scalar
Adds(dst, src, scalar, len);  // dst[i] = src[i] + scalar
```

### 规约
```cpp
ReduceSum(dst, src, work_local, len);
ReduceMax(dst, src, work_local, len);
ReduceMin(dst, src, work_local, len);
```
> `work_local` 是临时 LocalTensor，大小至少为 `len * sizeof(DTYPE)`。

### 同步
```cpp
pipe_barrier(PIPE_ALL);  // 等待所有 DMA 与计算完成
pipe_barrier(PIPE_MTE2); // 仅等待 DMA 搬入
pipe_barrier(PIPE_V);    // 仅等待向量计算
```

---

## 4. 分块大小计算参考

```cpp
// 假设 UB 可用空间为 UB_SIZE bytes，需要 2 个 buffer（输入 + 输出）
constexpr uint32_t UB_SIZE  = 200 * 1024;  // 200 KB（保守估计）
constexpr uint32_t DTYPE_SZ = sizeof(float16_t);
constexpr uint32_t BLOCK    = UB_SIZE / DTYPE_SZ / 2;

// 对齐到 32 的整数倍（访问对齐优化）
constexpr uint32_t BLOCK_ALIGNED = (BLOCK / 32) * 32;
```

---

## 5. 多 AI Core 并行

```cpp
// 每个 AI Core 处理不同数据段
uint32_t coreId   = GetBlockIdx();    // 当前 AI Core 编号（0 ~ numCores-1）
uint32_t numCores = GetBlockNum();    // 总 AI Core 数量
uint32_t perCore  = (totalElems + numCores - 1) / numCores;
uint32_t myStart  = coreId * perCore;
uint32_t myEnd    = min(myStart + perCore, totalElems);
uint32_t myLen    = myEnd - myStart;
// 处理 xGm[myStart : myEnd]
```

---

## 6. DoubleBuffer（DMA 与计算重叠）

当性能要求高时，用双队列深度掩盖 DMA 延迟：

```cpp
TQue<QuePosition::VECIN, 2> inQ;  // depth=2 → DoubleBuffer
pipe.InitBuffer(inQ, 2, block_size * sizeof(DTYPE));
// 系统会自动交替使用两个 buffer slot，在计算 block[k] 时预取 block[k+1]
```

---

## 7. 常见编译错误与解决

| 报错关键字 | 可能原因 | 解决 |
|-----------|---------|------|
| `no member 'XXX' in 'AscendC'` | API 名字拼写错误 / CANN 版本不匹配 | 查询 `${ASCEND_HOME}` 下的头文件 |
| `size of local variable ... exceeds` | UB 分配过大 | 减小 blockSize |
| `invalid conversion from GM_ADDR` | 指针类型转换错误 | 加 `(__gm__ DTYPE*)` 强转 |
| `undefined reference to 'GetBlockIdx'` | 缺少 CANN 链接 / 未正确 source env.sh | 检查 compile.sh 是否 source env.sh |

---

## 8. 执行流程（本 Skill 内容使用顺序）

1. 确认任务后端为 `ascendc`（`INSTRUCTIONS.md` 或 `OPERATOR_BACKEND`）
2. 参考本 Skill 的框架写 `src/ascendc/kernel.cpp`
3. 执行 `bash tools/operator_pipeline.sh`
4. 读取 `metrics.json`；失败则按第 7 节排查
5. 性能不足时应用第 5、6 节优化

> **不要**修改 `tools/` 目录下任何文件；所有编译命令由工具链管理。
