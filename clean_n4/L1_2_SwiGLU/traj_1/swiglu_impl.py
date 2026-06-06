import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    in_ptr,
    out_ptr,
    numel,
    chunk_size,
    chunk_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    pos = offsets
    # offset within the chunk dimension
    offset_in_chunk = (pos // chunk_stride) % chunk_size
    # threshold for first half vs second half
    half = chunk_size // 2

    # First half: in chunk a, pair is in chunk b at pos + chunk_size * chunk_stride
    mask_a = offset_in_chunk < half
    mask_b = ~mask_a

    # Load from both halves using mask
    a_val_a = tl.load(in_ptr + pos, mask=mask & mask_a)
    b_val_a = tl.load(in_ptr + pos + chunk_size * chunk_stride, mask=mask & mask_a)

    a_val_b = tl.load(in_ptr + pos - chunk_size * chunk_stride, mask=mask & mask_b)
    b_val_b = tl.load(in_ptr + pos, mask=mask & mask_b)

    # silu(a) * b
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    sig_a_a = 1.0 / (1.0 + tl.exp(-a_val_a))
    result_a = a_val_a * sig_a_a * b_val_a

    sig_a_b = 1.0 / (1.0 + tl.exp(-a_val_b))
    result_b = a_val_b * sig_a_b * b_val_b

    # Combine results using tl.where
    result = tl.where(mask_a, result_a, result_b)

    tl.store(out_ptr + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Normalize dim
        if dim < 0:
            dim = x.ndim + dim

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        # Compute output shape (dim is halved)
        orig_shape = list(x.shape)
        orig_shape[dim] = orig_shape[dim] // 2
        out_shape = tuple(orig_shape)

        # Prepare output tensor
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        # Flatten for 1D kernel
        x_flat = x.view(-1)
        out_flat = out.view(-1)
        numel = x_flat.numel()

        # Chunk parameters
        chunk_size = x.shape[dim]
        chunk_stride = numel // chunk_size

        # Grid configuration
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)

        # Launch kernel
        swiglu_kernel[grid](
            x_flat,
            out_flat,
            numel,
            chunk_size,
            chunk_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out
