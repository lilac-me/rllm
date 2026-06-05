import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_numel: tl.constexpr,
    y_numel: tl.constexpr,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_numel

    # Compute broadcast factor: x_numel / y_numel
    # Broadcasting always happens on leading dims, so:
    #   x_index = i
    #   y_index = i // broadcast_factor
    broadcast_factor = x_numel // y_numel

    x_val = tl.load(x_ptr + offsets, mask=mask)
    y_val = tl.load(y_ptr + (offsets // broadcast_factor), mask=mask)

    out_val = x_val + alpha * y_val

    tl.store(out_ptr + offsets, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Flatten both tensors to 1D (no computation, just layout)
        x_flat = x.contiguous().view(-1)
        y_flat = y.contiguous().view(-1)
        x_numel = x_flat.numel()
        y_numel = y_flat.numel()

        # Allocate output
        out = torch.empty_like(x)

        # Launch kernel — broadcasting handled inside kernel via broadcast_factor
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(x_numel, BLOCK_SIZE),)
        add_kernel[grid](x_flat, y_flat, out, x_numel, y_numel, alpha, BLOCK_SIZE=BLOCK_SIZE)

        return out
