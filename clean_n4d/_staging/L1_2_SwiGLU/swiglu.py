import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs a SwiGLU activation.
    SwiGLU(x, dim) = Swish(a) * b, where a and b are chunks of x along dim.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Applies SwiGLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor where the size of dim must be even.
            dim (int, optional): The dimension along which to chunk the tensor.

        Returns:
            torch.Tensor: Output tensor with SwiGLU applied, shape is same as x except
                          dim is halved.
        """
        a, b = torch.chunk(x, 2, dim=dim)
        return torch.nn.functional.silu(a) * b


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([130], dtype=torch.float32), -1],
        [torch.randn([256], dtype=torch.float32), -1],
        [torch.randn([512], dtype=torch.float32), -1],
        [torch.randn([1024], dtype=torch.float32), -1],
        [torch.randn([2048], dtype=torch.float32), -1],
        [torch.randn([4096], dtype=torch.float32), -1],
        [torch.randn([128, 128], dtype=torch.float16), -1],
        [torch.randn([128, 128], dtype=torch.float16), 0],
        [torch.randn([256, 256], dtype=torch.float16), -1],
        [torch.randn([256, 256], dtype=torch.float16), 0],
        [torch.randn([512, 512], dtype=torch.float16), -1],
        [torch.randn([512, 512], dtype=torch.float16), 0],
        [torch.randn([1024, 1024], dtype=torch.float16), -1],
        [torch.randn([1024, 1024], dtype=torch.float16), 0],
        [torch.randn([64, 64], dtype=torch.bfloat16), -1],
        [torch.randn([64, 64], dtype=torch.bfloat16), 0],
        [torch.randn([128, 256], dtype=torch.float32), -1],
        [torch.randn([128, 256], dtype=torch.float32), 0],
        [torch.randn([256, 512], dtype=torch.float32), -1],
        [torch.randn([256, 512], dtype=torch.float32), 0],
        [torch.randn([64, 64, 64], dtype=torch.float16), -1],
        [torch.randn([64, 64, 64], dtype=torch.float16), 0],
        [torch.randn([64, 64, 64], dtype=torch.float16), 1],
        [torch.randn([32, 32, 32], dtype=torch.float16), -1],
        [torch.randn([32, 32, 32], dtype=torch.float16), 0],
        [torch.randn([32, 32, 32], dtype=torch.float16), 1],
        [torch.randn([128, 64, 32], dtype=torch.bfloat16), -1],
        [torch.randn([128, 64, 32], dtype=torch.bfloat16), 0],
        [torch.randn([128, 64, 32], dtype=torch.bfloat16), 1],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), -1],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 0],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 1],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), -1],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), 1],
        [torch.randn([1, 64, 64, 64], dtype=torch.float32), -1],
        [torch.randn([1, 64, 64, 64], dtype=torch.float32), 1],
        [torch.randn([1, 128, 32, 32], dtype=torch.float16), -1],
        [torch.randn([1, 128, 32, 32], dtype=torch.float16), 1],
        [torch.randn([1, 256, 16, 16], dtype=torch.bfloat16), -1],
        [torch.randn([1, 256, 16, 16], dtype=torch.bfloat16), 1],
        [torch.randn([4096, 8192], dtype=torch.float32), -1],
        [torch.randn([349, 1536], dtype=torch.float16), -1],
        [torch.randn([5007, 3840], dtype=torch.float16), -1],
        [torch.randn([1829, 3072], dtype=torch.float16), -1],
        [torch.randn([109, 5120], dtype=torch.bfloat16), -1],
        [torch.randn([147, 10240], dtype=torch.bfloat16), -1],
        [torch.randn([221, 12288], dtype=torch.float32), -1],
        [torch.randn([2677, 13824], dtype=torch.float32), -1],
        [torch.randn([1001, 18432], dtype=torch.float16), -1],
        [torch.randn([1776, 24576], dtype=torch.float16), 0],
    ]
