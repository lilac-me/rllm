import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sort_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    sort_size,
    BLOCK_SIZE: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    """Selection sort kernel: each program handles one row."""
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Pointers for this row
    row_ptr = input_ptr + row_idx * sort_size
    out_ptr = output_ptr + row_idx * sort_size

    # Selection sort: for each position, find min in remaining unsorted portion
    for pos in range(sort_size):
        # Load the row data with boundary check
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < sort_size
        data = tl.load(row_ptr + offsets, mask=mask, other=SENTINEL)

        # Find minimum in unsorted portion [pos, sort_size)
        min_val = SENTINEL
        min_idx = -1

        for i in range(pos, sort_size):
            if i < sort_size:
                val = tl.load(row_ptr + i, mask=(i < sort_size), other=SENTINEL)
                # Replace sentinel with +inf for comparison
                val_cmp = tl.where(val == SENTINEL, float('inf'), val)
                min_cmp = tl.where(min_val == SENTINEL, float('inf'), min_val)
                is_new_min = val_cmp < min_cmp
                min_val = tl.where(is_new_min, val, min_val)
                min_idx = tl.where(is_new_min, i, min_idx)

        # Store the minimum at current position
        tl.store(out_ptr + pos, min_val, mask=(pos < sort_size))

        # Mark this index as "sorted" by writing sentinel back
        tl.store(row_ptr + min_idx, SENTINEL, mask=(min_idx >= 0) & (min_idx < sort_size))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
        # Normalize dim
        ndim = x.ndim
        if dim < 0:
            dim = dim + ndim

        # Handle edge cases
        if x.numel() <= 1:
            return x.clone()

        # Handle descending: negate values, sort ascending, negate back
        if descending:
            x = -x

        # Transpose so sort_dim is last
        dims = list(range(ndim))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(dims).contiguous()

        # Reshape to 2D
        orig_shape = x.shape
        n_rows = orig_shape[:-1].numel() if orig_shape[:-1].numel() > 0 else 1
        sort_size = orig_shape[-1]

        if n_rows == 0 or sort_size <= 1:
            # Reshape back
            out = x.reshape(orig_shape)
            if descending:
                out = -out
            return out

        x_2d = x.reshape(-1, sort_size)

        # Allocate output
        out_2d = torch.empty_like(x_2d)

        # Choose sentinel value based on dtype
        if x_2d.dtype == torch.float16:
            sentinel = 60000.0
        else:
            sentinel = 1e38

        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_rows, 1),)
        sort_kernel[grid](
            x_2d, out_2d,
            n_rows, sort_size,
            BLOCK_SIZE=BLOCK_SIZE,
            SENTINEL=sentinel,
        )

        # Reshape back to original shape
        out = out_2d.reshape(orig_shape)

        # Reverse descending
        if descending:
            out = -out

        return out
