# Triton-Ascend Operator Agent

This workspace is for generating Ascend NPU Triton implementations from
KernelBench-style PyTorch reference modules.

The workflow is adapted for OpenHands from the Triton operator-generation
ideas in Just-it/AscendOpGenAgent. Use `references/ascend_op_gen_agent_triton.md`
as a compact guide when planning kernels.

## Target

- Backend: Ascend NPU.
- DSL: Triton / Triton-Ascend.
- Reference framework: PyTorch `Model`.
- Required output file: `src/{op_name}_triton_ascend_impl.py`.
- Required class: `ModelNew`.
- `ModelNew.__init__` and `ModelNew.forward` must be compatible with the
  reference `Model` in the task.
- `ModelNew` must subclass `torch.nn.Module` or `nn.Module`.
- Triton kernels must be defined at module scope, not nested inside `forward`.
- Treat executable Python code as the source of truth. In particular,
  `forward`, `get_inputs()`, and `get_init_inputs()` override comments,
  docstrings, and shape descriptions if they disagree.

## Ground Rules

- Do not modify `tools/`.
- Do not install packages, download files, run `apt`, run `pip install`, or use
  external network access.
- Use only files already in this workspace.
- Preserve correctness before optimizing speed.
- Keep generated code self-contained in the implementation file.
- Do not add test-only code, print spam, or `if __name__ == "__main__"` blocks
  to the implementation file.
- Do not add `get_inputs()` or `get_init_inputs()` to the implementation file;
  they belong only to the mirrored reference task file.
- Do not create skeletons, placeholders, TODO-only code, or `pass` bodies. If a
  complete high-performance implementation is hard, still write the simplest
  executable Triton implementation you can validate.
- Do not inspect `tools/` or `tools/operator_pipeline.sh`; the validation
  command is fixed in the task. Avoid whole-file replacement edits after the
  initial create step; use targeted edits so the conversation history stays
  compact.
- Do not use the think tool or write private reasoning transcripts. Act with
  file edits and validation commands directly.
- After the initial create, never use a whole-file replacement edit. Patch the
  smallest failing block so the conversation history stays compact.

## Workflow

1. Read the user message. It includes the full PyTorch reference code, so you do
   not need to inspect a separate task file before starting. Your first file
   operation should create `src/{op_name}_triton_ascend_impl.py` unless the
   prompt is missing code.
2. Identify inputs, initialization inputs, tensor shapes, dtypes, reductions,
   broadcasts, layout transforms, and fusion opportunities.
3. Write `src/{op_name}_triton_ascend_impl.py` with module-level `@triton.jit`
   kernels and a compatible `ModelNew(nn.Module)`.
4. Run:

   ```bash
   bash tools/operator_pipeline.sh --op_name {op_name}
   ```

5. Read `metrics.json` after every pipeline run, including failed runs. Use
   `error_type` and `error` as the primary debugging signal; terminal logs may
   be clipped. If `metrics.json` has `"error_truncated": true` or an
   `error_file`, read that file before deciding the next code change. Fix the
   code and rerun the pipeline.
6. When the pipeline reports success or `metrics.json` has `"success": true`,
   stop editing the implementation file. Low speedup after a successful
   benchmark is still success in this rollout. Only read `metrics.json`,
   compare speedup with any existing `metrics_best.json`, save the best
   implementation, and finish:

   ```bash
   cp src/{op_name}_triton_ascend_impl.py src/{op_name}_triton_ascend_impl_best.py
   cp metrics.json metrics_best.json
   ```

## PyTorch Fallback Policy

Core computation in `ModelNew.forward()` should be implemented in Triton kernels.
The static checker rejects common PyTorch fallback patterns.

Avoid using these for core math in `forward`:

- `torch.matmul`, `torch.sum`, `torch.relu`, `torch.softmax`, and similar
  compute operators.
- `torch.nn.functional` compute calls such as `F.linear` or `F.softmax`.
- Tensor compute methods and operators such as `.sum()`, `.mean()`, `@`, `+`,
  `*` when they are replacing the target operation.
- Calling reference submodules such as `self.conv(x)` or `self.linear(x)` for
  the optimized path.

Allowed support operations include allocation, shape inspection, reshaping,
contiguity conversion, and launching Triton kernels.
When launching Triton kernels, pass tensor objects directly, for example
`kernel[grid](A, B, C, ...)`. Do not pass `.data_ptr()` or integer pointer
values; Triton needs tensor arguments so `tl.load` and `tl.store` receive valid
pointers.
For scalar outputs such as reductions or losses, the final scalar must also be
produced by Triton kernels. Do not use PyTorch post-processing like `.sum()`,
`.mean()`, tensor indexing plus arithmetic, or scalar division in `forward`.

## Short Kernel Rules

Use these rules silently. Do not print a plan before creating the file.

- Elementwise/broadcast: use one vectorized tile kernel over output elements.
  `diag(A) @ B` is `C[i, j] = A[i] * B[i, j]`; never materialize `diag(A)`.
- Matmul/linear: use a tiled `tl.dot` kernel, tensor arguments, fp32
  accumulation when needed, and no dtype casts unless the reference does them.
- Reductions: reduce inside a tile; for large axes use a small first-stage
  partial reduction plus a final reduction kernel.
- Transpose/slice/layout ops: allocate the output and write the exact indexed
  layout in Triton. Keep stride, shape, and dtype identical to the reference.
- Softmax/cumsum/conv/loss: write the simplest correct Triton version first,
  then reduce tile sizes if compilation or memory fails.

## Error Triage

- Syntax/import/type errors: fix the implementation file.
- `pass`, duplicate `forward`, or nested kernel definitions: replace them with
  a complete module-level kernel and a single valid `forward`.
- `Unsupported ptr type ... int64 in tl.load`: remove `.data_ptr()` from the
  launch and pass tensors directly.
- `Mask argument cannot be block type if pointer argument is not a block`: make
  the pointer expression block-shaped too.
- AST fallback failures: move the rejected PyTorch compute into Triton kernels.
- Correctness failures: compare shape, dtype, broadcasting, reduction axes, and
  numerical tolerances with the reference.
- Triton-Ascend compile failures with `ub overflow`: reduce tile sizes first,
  especially `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, and `BLOCK_SIZE_K`.
- For tiled kernels, ensure the launch grid rank matches the `tl.program_id`
  axes used in the kernel. If the kernel uses only `tl.program_id(0)`, compute
  all block coordinates from that single linear id. If it uses
  `tl.program_id(1)`, launch a 2D grid.
- Benchmark failures after correctness passed: simplify the kernel, reduce
  memory traffic, or tune block sizes. If benchmark completed successfully,
  do not tune for higher speedup in the same rollout.
- Device/environment failures: do not try to repair the environment with
  installs. Record the issue from `metrics.json` and stop if it is not code
  fixable.

## Metrics Contract

The pipeline writes `metrics.json`:

```json
{
  "schema_version": 2,
  "success": true,
  "ast_check_ok": true,
  "correctness_ok": true,
  "perf_data": {
    "framework_latency_ms": 1.23,
    "impl_latency_ms": 0.56,
    "speedup_vs_torch": 2.17
  },
  "error": null
}
```

Failed runs may also include `error_type`, `error_file`, and `error_truncated`.
The RL reward is computed from this file. Always run the pipeline after code
changes so the final reward reflects the actual implementation.
