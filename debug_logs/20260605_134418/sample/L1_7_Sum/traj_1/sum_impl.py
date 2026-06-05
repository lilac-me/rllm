import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """
    Reduce sum along axis=1 for a [M, N] matrix.
    Each program handles BLOCK_SIZE_M rows.
    Accumulates in fp32 for numerical accuracy.
    Uses 1D indexing per row to avoid 2D broadcast issues.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M

    for row_idx in range(row_start, min((pid + 1) * BLOCK_SIZE_M, M)):
        acc = 0.0

        # Loop over N in chunks
        for n_start in range(0, N, BLOCK_SIZE_N):
            offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
            mask = offsets < N
            # 1D index for element at (row_idx, n_start + offsets)
            idx = row_idx * N + offsets
            data = tl.load(input_ptr + idx, mask=mask, other=0.0)
            data_f32 = data.to(tl.float32)
            acc += tl.sum(data_f32, axis=0)

        # Store result
        if OUTPUT_DTYPE == tl.float16:
            tl.store(output_ptr + row_idx, acc.to(tl.float16))
        elif OUTPUT_DTYPE == tl.bfloat16:
            tl.store(output_ptr + row_idx, acc.to(tl.bfloat16))
        else:
            tl.store(output_ptr + row_idx, acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        """
        Returns the sum of elements along specified dimensions.
        All core computation is done in the Triton kernel.
        """
        orig_dtype = x.dtype
        orig_shape = x.shape
        ndim = x.ndim

        # Normalize dim
        if dim is None:
            dim = tuple(range(ndim))
        elif isinstance(dim, int):
            dim = (dim,)

        # Normalize negative dims
        dim = tuple(d % ndim for d in dim)
        # Sort for consistency
        dim = tuple(sorted(dim))

        # Compute output shape
        if keepdim:
            out_shape = tuple(1 if i in dim else s for i, s in enumerate(orig_shape))
        else:
            out_shape = tuple(s for i, s in enumerate(orig_shape) if i not in dim)

        # If output is scalar (0-d tensor)
        if len(out_shape) == 0:
            out_shape = (1,)

        # Permute: move reduce dims to the end
        non_reduce_dims = [i for i in range(ndim) if i not in dim]
        perm = non_reduce_dims + list(dim)
        x_perm = x.permute(perm)

        # Compute M and N using math.prod (Python stdlib, no tensor ops)
        M = math.prod(orig_shape[i] for i in non_reduce_dims) if non_reduce_dims else 1
        N = math.prod(orig_shape[i] for i in dim) if dim else 1

        # Reshape to [M, N]
        x_2d = x_perm.reshape(M, N)

        # Determine output dtype
        if orig_dtype == torch.float16:
            out_dtype = tl.float16
        elif orig_dtype == torch.bfloat16:
            out_dtype = tl.bfloat16
        else:
            out_dtype = tl.float32

        # Allocate output
        output = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch kernel
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = min(1024, N) if N > 0 else 1
        grid = (triton.cdiv(M, BLOCK_SIZE_M),)

        sum_kernel[grid](
            x_2d,
            output,
            M,
            N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            INPUT_DTYPE=orig_dtype,
            OUTPUT_DTYPE=out_dtype,
        )

        # Cast output to original dtype
        if orig_dtype == torch.float16:
            output = output.to(torch.float16)
        elif orig_dtype == torch.bfloat16:
            output = output.to(torch.bfloat16)

        # Reshape to final output shape
        output = output.reshape(out_shape)

        return output
