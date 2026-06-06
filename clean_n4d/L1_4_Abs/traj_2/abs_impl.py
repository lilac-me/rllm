import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    """Elementwise absolute value kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    # Use simple arithmetic: abs(x) = x * sign(x), but sign(0) = 1
    # Alternative: x * (2 * (x >= 0).to(x.dtype) - 1)
    sign = 2 * (x >= 0).to(x.dtype) - 1
    out = x * sign
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 1024),)
        abs_kernel[grid](x.contiguous(), out, n, BLOCK_SIZE=1024)
        return out
