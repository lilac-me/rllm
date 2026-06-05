import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Elementwise absolute value kernel for Triton Ascend."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x < 0, -x, x)
    tl.store(output_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return output
        # Ensure contiguous for linear indexing
        x_contig = x.contiguous()
        # Use BLOCK_SIZE=1024; for very large tensors this creates many blocks
        # but avoids UB overflow issues
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        abs_kernel[grid](x_contig, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output
