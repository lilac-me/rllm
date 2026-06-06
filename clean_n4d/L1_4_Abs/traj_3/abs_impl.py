import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x_ptr, out_ptr, xnumel, NUM_TILES: tl.constexpr):
    pid = tl.program_id(0)
    for tile in range(NUM_TILES):
        idx = pid * 1024 + tile * 65536 + tl.arange(0, 1024)
        mask = idx < xnumel
        x = tl.load(x_ptr + idx, mask=mask)
        out = tl.where(x < 0, -x, x)
        tl.store(out_ptr + idx, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        num_cores = 64
        # Calculate tiles needed per core to cover all elements
        elements_per_core = triton.cdiv(n, num_cores)
        num_tiles = triton.cdiv(elements_per_core, 65536)
        if num_tiles < 1:
            num_tiles = 1
        abs_kernel[(num_cores,)](x.contiguous(), out, n, num_tiles)
        return out
