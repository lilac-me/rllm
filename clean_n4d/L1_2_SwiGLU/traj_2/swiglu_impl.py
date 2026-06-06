import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    input_ptr,
    output_ptr,
    out_numel,
    chunk_size,
    strides_after,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_threads = tl.num_programs(0)

    for i in range(pid, out_numel, num_threads):
        # flat index in first half: (i // strides_after) * chunk_size + (i % strides_after)
        idx_a = (i // strides_after) * chunk_size + (i % strides_after)
        idx_b = idx_a + chunk_size * strides_after

        val_a = tl.load(input_ptr + idx_a)
        val_b = tl.load(input_ptr + idx_b)

        # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        neg_x = -val_a
        neg_x = tl.minimum(neg_x, 88.0)
        neg_x = tl.maximum(neg_x, -88.0)
        exp_neg_x = tl.exp(neg_x)
        silu_a = val_a / (1.0 + exp_neg_x)

        out = silu_a * val_b
        tl.store(output_ptr + i, out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Normalize negative dim
        if dim < 0:
            dim = x.dim() + dim

        # Make contiguous for flat indexing
        x = x.contiguous()
        orig_shape = x.shape
        numel = x.numel()

        # For contiguous tensors, stride(dim+1) = product of all dims after dim
        if dim + 1 < x.dim():
            strides_after = x.stride(dim + 1)
        else:
            strides_after = 1

        chunk_size = orig_shape[dim] // 2
        out_numel = numel // 2

        # Build output shape: same as input but with dim halved
        out_shape = list(orig_shape)
        out_shape[dim] = chunk_size

        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        output = output.contiguous()

        BLOCK_SIZE = 256
        num_threads = min(256, triton.cdiv(out_numel, BLOCK_SIZE))
        grid = (num_threads,)

        swiglu_kernel[grid](
            x,
            output,
            out_numel,
            chunk_size,
            strides_after,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output
