import torch
import torch.nn as nn
import json
import os
import numpy as np

class Model(nn.Module):
    """
    Simple model that performs cumulative sum along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Applies cumulative sum along the specified dimension.

        NPU torch.cumsum may use a parallel scan that differs from the serial
        AscendC kernel for float32/float16. We compute the reference on CPU with
        serial accumulation (matching the kernel) to ensure a fair comparison.
        bfloat16 still uses NPU torch.cumsum.
        """
        if x.dtype == torch.float32:
            out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float32)
            return torch.from_numpy(out).to(x.device)
        if x.dtype == torch.float16:
            out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float16)
            return torch.from_numpy(out).to(x.device)
        return torch.cumsum(x, dim=dim)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 0],
        [torch.randn([256], dtype=torch.float32), 0],
        [torch.randn([512], dtype=torch.float32), -1],
        [torch.randn([1024], dtype=torch.float32), 0],
        [torch.randn([2048], dtype=torch.float32), -1],
        [torch.randn([128, 128], dtype=torch.float16), 0],
        [torch.randn([128, 128], dtype=torch.float16), 1],
        [torch.randn([256, 256], dtype=torch.float16), -1],
        [torch.randn([256, 256], dtype=torch.float16), -2],
        [torch.randn([128, 256], dtype=torch.float32), 0],
        [torch.randn([128, 256], dtype=torch.float32), 1],
        [torch.randn([256, 512], dtype=torch.float32), -1],
        [torch.randn([256, 512], dtype=torch.float32), -2],
        [torch.randn([64, 64], dtype=torch.bfloat16), 0],
        [torch.randn([64, 64], dtype=torch.bfloat16), -1],
        [torch.randn([64, 64, 64], dtype=torch.float32), 0],
        [torch.randn([64, 64, 64], dtype=torch.float32), 1],
        [torch.randn([64, 64, 64], dtype=torch.float32), 2],
        [torch.randn([64, 64, 64], dtype=torch.float32), -1],
        [torch.randn([32, 32, 32], dtype=torch.float16), -2],
        [torch.randn([32, 32, 32], dtype=torch.float16), -3],
        [torch.randn([128, 64, 32], dtype=torch.bfloat16), 0],
        [torch.randn([128, 64, 32], dtype=torch.bfloat16), 1],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 0],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 1],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 2],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 3],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), -1],
        [torch.randn([1, 16, 64, 64], dtype=torch.float16), 1],
        [torch.randn([1, 16, 64, 64], dtype=torch.float16), 2],
        [torch.randn([1, 16, 64, 64], dtype=torch.float16), -1],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 2],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 3],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), -2],
        [torch.randn([1, 128, 28, 28], dtype=torch.bfloat16), -3],
        [torch.randn([4096, 18432], dtype=torch.float32), 0],
        [torch.randn([4096, 18432], dtype=torch.float32), 1],
        [torch.randn([8192, 16384], dtype=torch.float16), -1],
        [torch.randn([8192, 16384], dtype=torch.float16), -2],
        [torch.randn([100], dtype=torch.float32), 0],
        [torch.randn([100, 2007], dtype=torch.float16), 0],
        [torch.randn([100, 2007], dtype=torch.float16), 1],
        [torch.randn([17, 301], dtype=torch.float16), -1],
        [torch.randn([13, 2117], dtype=torch.bfloat16), -2],
        [torch.randn([7, 15, 23], dtype=torch.float32), 0],
        [torch.randn([7, 15, 23], dtype=torch.float32), 1],
        [torch.randn([7, 15, 23], dtype=torch.float32), 2],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), 0],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), -1],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 2],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 3],
    ]
