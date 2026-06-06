import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _erf_approx(x):
    """Approximate erf using the rational approximation (Abramowitz & Stegun 7.1.26)."""
    # erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/pi + a*x^2) / (1 + a*x^2)))
    # where a = 0.147
    # More accurate: use the polynomial approximation
    # erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    # where t = 1 / (1 + p*x), p = 0.3275911
    sign = tl.where(x >= 0, 1.0, -1.0)
    ax = tl.abs(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * ax)
    poly = a1 * t + a2 * t * t + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    result = 1.0 - poly * tl.exp(-ax * ax)
    return sign * result


@triton.jit
def gelu_kernel(
    in_ptr, out_ptr, n_elements, approximate_int: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    # Convert to float32 for numerical accuracy (important for f16/bf16)
    x_fp32 = x.to(tl.float32)

    if approximate_int == 1:
        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = 0.79788456
        coef = 0.044715
        x3 = x_fp32 * x_fp32 * x_fp32
        inner = sqrt_2_pi * (x_fp32 + coef * x3)
        # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        e_pos = tl.exp(inner)
        e_neg = tl.exp(-inner)
        tanh_val = (e_pos - e_neg) / (e_pos + e_neg)
        y = 0.5 * x_fp32 * (1.0 + tanh_val)
    else:
        # default (none): 0.5 * x * (1 + erf(x / sqrt(2)))
        y = 0.5 * x_fp32 * (1.0 + _erf_approx(x_fp32 * 0.7071067811865476))

    # Convert back to original dtype
    y = y.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        # Ensure contiguous for flat memory access
        x_cont = x.contiguous()
        out = torch.empty_like(x_cont)
        n = x_cont.numel()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        # Convert string to int for constexpr (0='none', 1='tanh')
        approx_int = 1 if approximate == "tanh" else 0

        gelu_kernel[grid](
            x_cont,
            out,
            n,
            approximate_int=approx_int,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
