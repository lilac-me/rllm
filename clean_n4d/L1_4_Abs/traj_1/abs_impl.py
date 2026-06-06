import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.maximum(x, -x)
    tl.store(output_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 1024),)
        abs_kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
