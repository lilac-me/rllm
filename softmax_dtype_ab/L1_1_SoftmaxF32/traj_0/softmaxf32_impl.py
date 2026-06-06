import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmaxf32_kernel(x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Softmax along last dimension (dim=-1) for float32 input.

    Each program handles one row:
      1. Load row data
      2. max_val = max(row)
      3. row = row - max_val  (numerical stability)
      4. row = exp(row)
      5. sum_val = sum(row)
      6. row = row / sum_val
      7. Store result
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load the entire row
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)

    # Phase 1: max reduction
    max_val = tl.max(x, axis=0)
    # Subtract max for numerical stability
    x = x - max_val

    # Phase 2: exp
    x = tl.exp(x)

    # Phase 3: sum reduction
    sum_val = tl.sum(x, axis=0)

    # Phase 4: divide
    y = x / sum_val

    # Store result
    tl.store(out_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n)
        # Ensure BLOCK_SIZE is at least 1
        BLOCK_SIZE = max(BLOCK_SIZE, 1)
        grid = (m,)
        softmaxf32_kernel[grid](
            x.contiguous(),
            out,
            m,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
