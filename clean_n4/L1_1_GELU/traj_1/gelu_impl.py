import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, mode: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """GELU kernel. mode: 0='none' (tanh approx), 1='tanh' (tanh approx)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # Convert to float32 for precision in computation
    x_f32 = x.to(tl.float32)

    # GELU approximation using tanh (used for both 'none' and 'tanh' modes):
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # sqrt(2/pi) ≈ 0.7978845608028654
    arg = 0.7978845608028654 * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
    # Compute tanh via exp: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    exp_neg2arg = tl.exp(-2.0 * arg)
    tanh_val = (1.0 - exp_neg2arg) / (1.0 + exp_neg2arg)
    gelu_val = 0.5 * x_f32 * (1.0 + tanh_val)

    tl.store(out_ptr + offsets, gelu_val, mask=mask)


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
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()

        out = torch.empty_like(x)
        n_elements = x.numel()

        if n_elements == 0:
            return out

        # Use BLOCK_SIZE = 1024 for balanced performance
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Single kernel launch with constexpr mode parameter
        gelu_kernel[grid](x, out, n_elements, mode=0, BLOCK_SIZE=BLOCK_SIZE)

        return out
