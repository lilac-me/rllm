import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for i in range(3):
        cur_offset = i * STRIDE + offset
        mask = cur_offset < n
        x = tl.load(x_ptr + cur_offset, mask=mask)
        out = tl.where(x >= 0, x, -x)
        tl.store(out_ptr + cur_offset, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        num_blocks = triton.cdiv(n, 1024)
        if num_blocks > 65535:
            num_blocks = 65535
        stride = num_blocks * 1024
        grid = (num_blocks,)
        abs_kernel[grid](x.contiguous(), out, n, BLOCK_SIZE=1024, STRIDE=stride)
        return out
