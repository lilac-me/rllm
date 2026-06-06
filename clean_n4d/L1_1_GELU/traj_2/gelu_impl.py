import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr, n, approximate: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)

    # Promote to float32 for computation (critical for f16/bf16 correctness)
    x_f32 = x.to(tl.float32)

    if approximate == 0:
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        result = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 / tl.sqrt(2.0)))
    else:
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # sqrt(2/pi) ≈ 0.7978845608
        result = 0.5 * x_f32 * (
            1.0
            + tl.math.tanh(
                0.7978845608 * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
            )
        )

    # Convert back to original dtype
    out = result.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        approximate_flag = 0 if approximate == "none" else 1
        grid = (triton.cdiv(n, 1024),)
        gelu_kernel[grid](x.contiguous(), out, n, approximate_flag, BLOCK_SIZE=1024)
        return out
