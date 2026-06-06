import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(in_ptr, out_ptr, xnumel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < xnumel

    x = tl.load(in_ptr + offsets, mask=mask)
    ret = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, ret, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        gelu_kernel[grid](x.contiguous(), out, n, BLOCK_SIZE=BLOCK_SIZE)
        return out
