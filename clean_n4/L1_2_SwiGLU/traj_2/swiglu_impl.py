import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # silu(a) = a * sigmoid(a) = a / (1 + exp(-a))
    neg_a = -a
    exp_neg_a = tl.exp(neg_a)
    sigmoid_a = a.to(tl.float32) / (1.0 + exp_neg_a.to(tl.float32))
    sila = a * sigmoid_a

    result = sila * b
    result = result.to(a_ptr.dtype.element_ty)

    tl.store(out_ptr + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        orig_shape = x.shape
        ndim = x.dim()

        # Normalize dim to positive
        if dim < 0:
            dim = ndim + dim

        dim_size = orig_shape[dim]
        half = dim_size // 2

        # Reshape to [..., 2, half, ...]
        new_shape = (
            list(orig_shape[:dim]) + [2, half] + list(orig_shape[dim + 1 :])
        )
        x_reshaped = x.reshape(new_shape)

        # Extract a (first half) and b (second half) along the chunk dimension
        # Build index tuple to slice at position `dim` in the reshaped tensor
        idx = [slice(None)] * len(new_shape)
        idx[dim] = 0
        a = x_reshaped[tuple(idx)].contiguous().reshape(-1)

        idx[dim] = 1
        b = x_reshaped[tuple(idx)].contiguous().reshape(-1)

        total = a.numel()
        out = torch.empty_like(a)

        BLOCK_SIZE = 2048
        grid = (triton.cdiv(total, BLOCK_SIZE),)

        swiglu_kernel[grid](a, b, out, total, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape output to [..., half, ...]
        out_shape = list(orig_shape[:dim]) + [half] + list(orig_shape[dim + 1 :])
        return out.reshape(out_shape)
