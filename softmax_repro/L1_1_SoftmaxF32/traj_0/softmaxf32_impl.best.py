import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, M, N, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_cores = 48

    # Each program handles multiple rows via strided loop
    for row_idx in range(pid, M, num_cores):
        row_start = row_idx * stride_x

        # Load entire row
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))

        # Compute max, exp, sum, normalize in one pass
        max_val = tl.max(x, axis=0)
        exp_val = tl.exp(x - max_val)
        sum_val = tl.sum(exp_val, axis=0)
        result = exp_val / sum_val

        tl.store(y_ptr + row_start + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        M, N = x.shape
        y = torch.empty_like(x)

        # Fixed grid of 48 cores, each handles M/48 rows
        grid = (48,)

        softmax_kernel[grid](
            x, y,
            M, N,
            x.stride(0), y.stride(0),
            BLOCK_SIZE=2048,
        )
        return y
