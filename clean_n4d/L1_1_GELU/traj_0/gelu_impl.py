import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(in_ptr, out_ptr, numel, is_tanh, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute gelu_none: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_none = x_f32 * 0.5 * (1.0 + tl.erf(x_f32 / tl.sqrt(2.0)))

    # Compute gelu_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coef = 0.044715
    inner = x_f32 + coef * x_f32 * x_f32 * x_f32
    abs_inner = tl.abs(inner)
    exp_neg = tl.exp(-2.0 * abs_inner)
    tanh_val = (1.0 - exp_neg) / (1.0 + exp_neg)
    tanh_val = tl.where(inner >= 0, tanh_val, -tanh_val)
    gelu_tanh = x_f32 * 0.5 * (1.0 + tanh_val)

    # Select based on is_tanh using arithmetic (avoids tl.where with scalar)
    ret = (1 - is_tanh) * gelu_none + is_tanh * gelu_tanh
    tl.store(out_ptr + offsets, ret, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
        out = torch.empty_like(x)
        numel = x.numel()
        is_tanh = 1 if approximate == 'tanh' else 0

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)

        gelu_kernel[grid](x.contiguous(), out, numel, is_tanh, BLOCK_SIZE=BLOCK_SIZE)

        return out
