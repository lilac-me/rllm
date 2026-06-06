import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr, M, N, stride_x_row, stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_x_row
    out_ptr = out_ptr + row * stride_out_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # ---- pass 1: max ----
    x = tl.load(x_ptr + cols, mask=mask, other=float('-inf'))
    x_fp32 = x.to(tl.float32)
    max_val = tl.max(x_fp32, axis=0)

    # ---- pass 2: exp + sum ----
    x_shifted = x_fp32 - max_val
    e = tl.exp(x_shifted)
    sum_e = tl.sum(e, axis=0)

    # ---- pass 3: normalize ----
    y_fp32 = e / sum_e

    # write back in original dtype
    tl.store(out_ptr + cols, y_fp32.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        M, N = x.shape
        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE > 2048:
            BLOCK_SIZE = 2048
        grid = (M,)
        softmax_kernel[grid](
            x, out, M, N,
            x.stride(0), out.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
