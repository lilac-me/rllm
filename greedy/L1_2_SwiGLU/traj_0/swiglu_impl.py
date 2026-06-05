import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    dim: tl.constexpr,
    dim_size: tl.constexpr,
    half_dim: tl.constexpr,
    stride_dim: tl.constexpr,
    dim_stride_product: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SwiGLU kernel: silu(a) * b where a, b = chunk(x, 2, dim)

    For each output element at flat index `o`:
    - a's flat index = o (same multi-dim index, first half of chunk)
    - b's flat index = o + dim_stride_product * half_dim
      (same except dim shifted by half_dim)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a values (first half of chunk along dim)
    a = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Compute b's flat index: same as output index but with dim offset
    # b_offset = dim_stride_product * half_dim
    b_offset = dim_stride_product * half_dim
    b = tl.load(in_ptr + offsets + b_offset, mask=mask, other=0.0)

    # Compute silu(a) = a * sigmoid(a) = a / (1.0 + exp(-a))
    # Numerical stability: use sigmoid directly
    sig = 1.0 / (1.0 + tl.exp(-a))
    silu_a = a * sig

    # Compute result = silu(a) * b
    result = silu_a * b

    tl.store(out_ptr + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Normalize dim to positive
        ndim = x.ndim
        if dim < 0:
            dim = ndim + dim

        # Get dim size and compute half_dim
        dim_size = x.shape[dim]
        half_dim = dim_size // 2

        # Compute stride along dim and the stride product for dim offset
        # stride_dim = x.stride(dim)
        # dim_stride_product = stride_dim (since we only shift along dim)
        stride_dim = x.stride(dim)
        dim_stride_product = stride_dim

        # Make input contiguous for efficient memory access
        x_cont = x.contiguous()

        # Compute output shape: same as input except dim is halved
        out_shape = list(x.shape)
        out_shape[dim] = half_dim

        # Allocate output tensor
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        # Total output elements
        n_elements = output.numel()

        if n_elements == 0:
            return output

        # BLOCK_SIZE: use 1024 for elementwise operations
        BLOCK_SIZE = 1024

        # Grid: 1D grid over output elements
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Launch kernel
        swiglu_kernel[grid](
            x_cont,
            output,
            n_elements,
            dim,
            dim_size,
            half_dim,
            stride_dim,
            dim_stride_product,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output
