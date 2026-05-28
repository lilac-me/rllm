# AscendOpGenAgent Triton Reference

This note is a compact OpenHands-oriented reference adapted from
Just-it/AscendOpGenAgent:

- Repository: https://github.com/Just-it/AscendOpGenAgent
- Triton agent guide:
  https://github.com/Just-it/AscendOpGenAgent/blob/main/agents/triton-ascend-coder.md

Use it as guidance for generating Ascend NPU Triton kernels in this workspace.

## Role

Act as an Ascend NPU operator engineer. Given a PyTorch reference module, produce
a compatible `ModelNew` implementation using Triton kernels and optimize it for
latency while preserving numerical correctness.

## Planning Checklist

- Extract the reference `Model.__init__`, `Model.forward`, `get_inputs`, and
  `get_init_inputs` behavior.
- Identify tensor ranks, shapes, strides, dtype, device assumptions, reduction
  axes, broadcasting, and output layout.
- Separate pure shape/view logic from compute-heavy logic.
- Look for fusion opportunities, especially elementwise chains, reductions with
  post-processing, activation after matmul/conv-like operations, and normalization
  patterns.
- Pick simple correct kernels before tuning.

## Implementation Pattern

- Import `torch`, `triton`, and `triton.language as tl`.
- Put `@triton.jit` kernels at module scope.
- Wrap kernel launches in small Python helper functions when useful.
- Define `class ModelNew(torch.nn.Module)` with an interface compatible with the
  reference `Model`.
- Allocate outputs with `torch.empty_like`, `torch.empty`, or similar support
  operations. Keep core math inside Triton kernels.
- Use masks for boundary conditions.
- Make tensors contiguous when the reference semantics allow it.

## Verification Loop

1. Write a complete implementation file.
2. Run the pipeline:

   ```bash
   bash tools/operator_pipeline.sh --op_name <op_name>
   ```

3. Read `metrics.json`.
4. If AST fails, remove PyTorch fallback compute from `ModelNew.forward`.
5. If correctness fails, fix shape, dtype, indexing, broadcasting, and numerical
   behavior.
6. If benchmark fails, simplify and retest.
7. If success is true, tune block sizes and memory access patterns for speed.

## Optimization Hints

- Favor contiguous memory access.
- Avoid unnecessary intermediate tensors.
- Fuse simple elementwise operations into one kernel.
- For reductions, use block reductions and stable accumulation where needed.
- Tune block sizes conservatively; correctness and compile reliability matter
  more than aggressive specialization.
- Keep one robust implementation rather than many speculative variants.

## What Not To Do

- Do not call PyTorch compute ops for the optimized computation path.
- Do not modify validation scripts or pipeline scripts.
- Do not rely on network access or package installation.
- Do not hide failures. Let `metrics.json` reflect the real outcome.
