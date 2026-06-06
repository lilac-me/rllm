import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    half_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 2D grid: each program handles a tile of (BLOCK_SIZE_M, BLOCK_SIZE_N) output elements
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets within the tile
    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Boundary masks
    row_mask = row_offsets < num_rows
    col_mask = col_offsets < half_dim
    mask = row_mask[:, None] & col_mask[None, :]

    # Use block pointers for proper 2D access
    # Input: (num_rows, last_dim=half_dim*2), contiguous row-major
    # Output: (num_rows, half_dim), contiguous row-major
    in_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(num_rows, half_dim * 2),
        strides=(half_dim * 2, 1),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    a = tl.load(in_block_ptr, boundary_check=(0, 1))

    # b is in the second half of the input dimension
    in_block_ptr_b = tl.make_block_ptr(
        base=x_ptr,
        shape=(num_rows, half_dim * 2),
        strides=(half_dim * 2, 1),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N + half_dim),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    b = tl.load(in_block_ptr_b, boundary_check=(0, 1))

    # Silu(a) = a / (1 + exp(-a))
    silu = a / (1.0 + tl.exp(-a))

    # SwiGLU = silu(a) * b
    out = silu * b

    # Store to output
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(num_rows, half_dim),
        strides=(half_dim, 1),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(out_block_ptr, out, boundary_check=(0, 1))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Move target dimension to last axis
        x_moved = x.movedim(dim, -1)

        # Get shape info
        orig_shape = x_moved.shape
        last_dim = orig_shape[-1]
        half_dim = last_dim // 2
        num_rows = int(x_moved.numel() // last_dim)

        # Reshape to (num_rows, last_dim) where last axis is [a..., b...]
        x_flat = x_moved.reshape(num_rows, last_dim)

        # Output tensor: (num_rows, half_dim)
        out_flat = torch.empty((num_rows, half_dim), device=x.device, dtype=x.dtype)

        # Launch 2D kernel with small blocks to avoid UB overflow
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        grid_m = triton.cdiv(num_rows, BLOCK_SIZE_M)
        grid_n = triton.cdiv(half_dim, BLOCK_SIZE_N)
        _swiglu_kernel[grid_m, grid_n](
            x_flat, out_flat,
            num_rows, half_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        # Reshape back to original shape (with dim halved)
        out = out_flat.reshape(orig_shape[:-1] + (half_dim,))

        # Move dimension back to original position
        out = out.movedim(-1, dim)

        return out
