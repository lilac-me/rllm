import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Elementwise absolute value kernel for Triton-Ascend."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.abs(x)
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 4096
        num_blocks = triton.cdiv(n, BLOCK_SIZE)
        grid = (num_blocks,)
        abs_kernel[grid](x.contiguous(), out, n, BLOCK_SIZE=BLOCK_SIZE)
        return out
