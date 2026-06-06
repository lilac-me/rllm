import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim = x.dim() + dim

        out_shape = list(x.shape)
        out_shape[dim] = x.shape[dim] // 2
        out_shape = tuple(out_shape)

        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        if out.numel() == 0:
            return out

        # Flatten both to 1D
        input_flat = x.contiguous().view(-1)
        output_flat = out.contiguous().view(-1)
        numel = output_flat.numel()

        BLOCK_SIZE = 1024
        n_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE

        copy_kernel[(n_blocks,)](
            input_flat,
            output_flat,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out
