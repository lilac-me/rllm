import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sort_kernel(
    in_ptr,
    out_ptr,
    row_len,
    descending,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load all elements of this row
    offsets = pid * row_len + tl.arange(0, BLOCK_SIZE)
    mask = offsets < pid * row_len + row_len

    # Create a local buffer for sorting
    arr = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)
    arr = tl.where(mask, tl.load(in_ptr + offsets, mask=mask, other=0.0), arr)

    # Selection sort
    for step in range(BLOCK_SIZE):
        # Find minimum element in arr[step:row_len]
        idx = step + tl.arange(0, BLOCK_SIZE)
        idx_mask = idx < row_len

        val = tl.load(arr + idx, mask=idx_mask, other=1e10)
        is_cand = idx_mask & (val < arr[step])
        min_idx = tl.where(is_cand, idx, step)
        min_val = tl.where(is_cand, val, arr[step])

        # Swap arr[step] and arr[min_idx]
        val_step = tl.load(arr + step, mask=(step < row_len), other=0.0)
        val_min = tl.load(arr + min_idx, mask=(min_idx < row_len), other=0.0)
        do_swap = step < row_len
        new_val_step = tl.where(do_swap, val_min, val_step)
        new_val_min = tl.where(do_swap, val_step, val_min)
        tl.store(arr + step, new_val_step, mask=(step < row_len))
        tl.store(arr + min_idx, new_val_min, mask=(min_idx < row_len))

    # Store sorted result
    tl.store(out_ptr + offsets, arr, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
        original_shape = x.shape
        ndim = x.ndim

        # Normalize dim
        if dim < 0:
            dim = dim + ndim

        # Transpose so that sort dim becomes the last dimension
        dims = list(range(ndim))
        dims.append(dims.pop(dim))
        x_t = x.permute(dims)
        row_len = x_t.shape[-1]
        num_rows = x_t.numel() // row_len

        # Flatten to 2D
        x_2d = x_t.reshape(-1, row_len)

        # Allocate output
        out_2d = torch.empty_like(x_2d)

        # Launch kernel
        BLOCK_SIZE = triton.next_power_of_2(row_len)
        grid = (num_rows,)
        sort_kernel[grid](x_2d, out_2d, row_len, descending, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape back
        sorted_t = out_2d.reshape(original_shape)

        # Reverse transpose to restore original dimension order
        inv_dims = [0] * ndim
        for i, d in enumerate(dims):
            inv_dims[d] = i
        result = sorted_t.permute(inv_dims)

        return result
