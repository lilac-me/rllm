import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements, alpha,
    BLOCK_SIZE: tl.constexpr,
    is_fp16: tl.constexpr,
    is_bf16: tl.constexpr,
):
    """Element-wise add kernel: out = x + alpha * y with broadcasting support.

    x and y are already flattened to 1D and broadcasted to the same shape.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # For low precision dtypes, accumulate in fp32 for numerical accuracy
    if is_fp16 or is_bf16:
        x_fp32 = x.to(tl.float32)
        y_fp32 = y.to(tl.float32)
        result_fp32 = x_fp32 + alpha * y_fp32
        # Store back to original dtype
        if is_fp16:
            result = result_fp32.to(tl.float16)
        else:
            result = result_fp32.to(tl.bfloat16)
    else:
        result = x + alpha * y

    tl.store(out_ptr + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Broadcast y to match x's shape (layout operation, not core computation)
        x_orig_shape = x.shape
        y_broadcast = y.broadcast_to(x_orig_shape)

        # Flatten to 1D contiguous tensors for the kernel
        x_flat = x.contiguous().view(-1)
        y_flat = y_broadcast.contiguous().view(-1)
        n_elements = x_flat.numel()

        # Allocate output
        out = torch.empty_like(x)

        # Determine dtype flags for the kernel
        is_fp16 = x.dtype == torch.float16
        is_bf16 = x.dtype == torch.bfloat16

        # Grid configuration
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Launch kernel
        _add_kernel[grid](
            x_flat, y_flat, out.view(-1),
            n_elements, alpha,
            BLOCK_SIZE,
            is_fp16,
            is_bf16,
        )

        return out
