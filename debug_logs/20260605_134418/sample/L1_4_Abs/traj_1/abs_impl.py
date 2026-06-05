import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.abs(data)
    tl.store(output_ptr + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten to 1D for kernel
        x_flat = x.reshape(-1)
        output = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        block_size = 2048
        grid = (triton.cdiv(n_elements, block_size),)
        abs_kernel[grid](x_flat, output, n_elements, BLOCK_SIZE=block_size)
        return output.reshape(x.shape)
