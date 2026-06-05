import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that repeats elements of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats, dim: int = None, output_size: int = None) -> torch.Tensor:
        """
        Repeats elements of a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (int or torch.Tensor): Number of repetitions for each element.
            dim (int, optional): The dimension along which to repeat values.
            output_size (int, optional): Total output size for the repeated dimension.

        Returns:
            torch.Tensor: Tensor with repeated elements.
        """
        if output_size is not None:
            return torch.repeat_interleave(x, repeats, dim=dim, output_size=output_size)
        return torch.repeat_interleave(x, repeats, dim=dim)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 2, 0],
        [torch.randn([256], dtype=torch.float16), 3, 0],
        [torch.randn([512], dtype=torch.float16), 4, 0],
        [torch.randn([128, 64], dtype=torch.float32), 2, 0],
        [torch.randn([128, 64], dtype=torch.float16), 2, 1],
        [torch.randn([64, 128], dtype=torch.float16), 3, 0],
        [torch.randn([64, 128], dtype=torch.float16), 3, 1],
        [torch.randn([256, 256], dtype=torch.bfloat16), 2, 0],
        [torch.randn([256, 256], dtype=torch.bfloat16), 2, 1],
        [torch.randn([128, 256], dtype=torch.float32), 4, 0],
        [torch.randn([128, 256], dtype=torch.float16), 4, 1],
        [torch.randn([64, 64, 64], dtype=torch.float32), 2, 0],
        [torch.randn([64, 64, 64], dtype=torch.float16), 2, 1],
        [torch.randn([64, 64, 64], dtype=torch.float16), 2, 2],
        [torch.randn([32, 64, 128], dtype=torch.float16), 2, 0],
        [torch.randn([32, 64, 128], dtype=torch.float16), 2, -1],
        [torch.randn([16, 128, 256], dtype=torch.bfloat16), 2, 1],
        [torch.randn([1, 64, 56, 56], dtype=torch.float32), 2, 1],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 2, 1],
        [torch.randn([1, 128, 56, 56], dtype=torch.float16), 2, 1],
        [torch.randn([1, 256, 28, 28], dtype=torch.float16), 2, 1],
        [torch.randn([1, 512, 14, 14], dtype=torch.bfloat16), 2, 1],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 2, 2],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 2, 3],
        [torch.randn([1, 4096, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 4096, 128], dtype=torch.float16), 2, 2],
        [torch.randn([1, 8192, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 4096, 512], dtype=torch.float16), 2, 1],
        [torch.randn([1, 4096, 128], dtype=torch.bfloat16), 2, 1],
        [torch.randn([4, 4096, 128], dtype=torch.float16), 2, 1],
        [torch.randn([8, 4096, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 3584, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 5120, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 6144, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 3072, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 2048, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 1536, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 2560, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 2880, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 3840, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 7168, 128], dtype=torch.float16), 2, 1],
        [torch.randn([128, 256], dtype=torch.float16), 2, 0],
        [torch.randn([128, 256], dtype=torch.float16), 2, 1],
        [torch.randn([256, 512], dtype=torch.float16), 2, 0],
        [torch.randn([512, 1024], dtype=torch.float16), 2, 0],
        [torch.randn([1024, 4096], dtype=torch.float16), 2, 0],
        [torch.randn([4096, 4096], dtype=torch.float16), 2, 0],
        [torch.randn([4096, 8192], dtype=torch.float16), 2, 0],
        [torch.randn([8192, 8192], dtype=torch.float16), 2, 0],
        [torch.randn([1, 12, 64, 64], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 128, 128], dtype=torch.float16), 2, 1],
        [torch.randn([1, 128, 256, 256], dtype=torch.float16), 2, 1],
        [torch.randn([1, 128, 128, 4096], dtype=torch.float16), 2, 1],
        [torch.randn([1, 128, 128, 8192], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 64, 1536], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 64, 2048], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 64, 3072], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 64, 5120], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 64, 6144], dtype=torch.float16), 2, 1],
        [torch.randn([1, 64, 64, 7168], dtype=torch.float16), 2, 1],
        [torch.randn([1, 100, 200], dtype=torch.float16), 2, 1],
        [torch.randn([1, 50, 75, 100], dtype=torch.float16), 2, 1],
        [torch.randn([1, 17, 33, 65], dtype=torch.float16), 2, 1],
        [torch.randn([1, 128, 128, 128], dtype=torch.float32), 2, 1],
        [torch.randn([123, 6144], dtype=torch.float16), 2, 0],
        [torch.randn([789, 12288], dtype=torch.bfloat16), 2, 0],
        [torch.randn([4096, 11008], dtype=torch.float32), 2, 0],
        [torch.randn([4096, 13824], dtype=torch.float16), 2, 0],
        [torch.randn([4096, 24576], dtype=torch.bfloat16), 2, 0],
        [torch.randn([8192, 16384], dtype=torch.float32), 2, 0],
        [torch.randn([2048, 13824], dtype=torch.float16), 2, 0],
        [torch.randn([1, 3584], dtype=torch.float16), 2, 1],
        [torch.randn([1, 5120], dtype=torch.float32), 2, 1],
        [torch.randn([1, 8192], dtype=torch.float16), 2, 1],
        [torch.randn([4096, 4096], dtype=torch.float16), 2, 1],
    ]
