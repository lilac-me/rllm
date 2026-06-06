import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr: tl.constexpr,
    y_ptr: tl.constexpr,
    out_ptr: tl.constexpr,
    n_elements,
    alpha: tl.constexpr,
):
    # Each thread handles one element
    idx = tl.program_id(0).to(tl.int32)
    mask = idx < n_elements
    xv = tl.load(x_ptr + idx, mask=mask)
    yv = tl.load(y_ptr + idx, mask=mask)
    result = (xv.to(tl.float32) + alpha * yv.to(tl.float32)).to(xv.dtype)
    tl.store(out_ptr + idx, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Simple case: output shape = x.shape
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

        # Flatten inputs for 1D kernel
        x_flat = x.contiguous().view(-1)
        y_flat = y.contiguous().view(-1)
        n_elements = out.numel()

        grid = (n_elements,)
        add_kernel[grid](
            x_flat.data_ptr(), y_flat.data_ptr(), out.data_ptr(),
            n_elements, alpha,
        )
        return out
