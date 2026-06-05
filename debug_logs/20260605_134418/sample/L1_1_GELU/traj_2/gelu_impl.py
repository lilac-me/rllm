import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(in_ptr, out_ptr, xnumel, is_tanh: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """GELU activation kernel supporting both 'none' and 'tanh' approximation modes."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < xnumel
    x = tl.load(in_ptr + offsets, mask=mask)

    if is_tanh:
        # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) — numerically stable
        x_f32 = x.to(tl.float32)
        coeff = tl.sqrt(2.0 / 3.141592653589793)
        cube = x_f32 * x_f32 * x_f32
        inner = coeff * (x_f32 + 0.044715 * cube)
        # Clamp for numerical stability
        inner = tl.minimum(inner, 10.0)
        inner = tl.maximum(inner, -10.0)
        # tanh via exp: (e^(2*inner) - 1) / (e^(2*inner) + 1)
        exp2 = tl.exp(2.0 * inner)
        tanh_val = (exp2 - 1.0) / (exp2 + 1.0)
        result = 0.5 * x_f32 * (1.0 + tanh_val)
    else:
        # GELU standard: 0.5 * x * (1 + erf(x / sqrt(2)))
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))

    tl.store(out_ptr + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            approximate (str, optional): The gelu approximation algorithm to use: 'none'|'tanh'.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        # Ensure contiguous for efficient memory access
        if not x.is_contiguous():
            x = x.contiguous()

        output = torch.empty_like(x)
        xnumel = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(xnumel, BLOCK_SIZE),)

        _gelu_kernel[grid](x, output, xnumel, is_tanh=(approximate == 'tanh'), BLOCK_SIZE=BLOCK_SIZE)

        return output
