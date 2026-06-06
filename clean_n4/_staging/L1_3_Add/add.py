import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs element-wise addition with broadcasting support.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Applies element-wise addition to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.
            alpha (float, optional): The multiplier for y.

        Returns:
            torch.Tensor: Output tensor x + alpha * y, shape follows broadcasting rules.
        """
        return torch.add(x, y, alpha=alpha)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), torch.randn([128], dtype=torch.float32), 1.0],
        [torch.randn([256], dtype=torch.float32), torch.randn([256], dtype=torch.float32), 2.0],
        [torch.randn([512], dtype=torch.float32), torch.randn([512], dtype=torch.float32), 0.5],
        [torch.randn([128, 128], dtype=torch.float16), torch.randn([128, 128], dtype=torch.float16), 1.0],
        [torch.randn([256, 256], dtype=torch.float16), torch.randn([256, 256], dtype=torch.float16), 1.5],
        [torch.randn([512, 512], dtype=torch.float16), torch.randn([512, 512], dtype=torch.float16), -1.0],
        [torch.randn([64, 64], dtype=torch.bfloat16), torch.randn([64, 64], dtype=torch.bfloat16), 1.0],
        [torch.randn([128, 256], dtype=torch.float32), torch.randn([128, 256], dtype=torch.float32), 0.25],
        [torch.randn([256, 512], dtype=torch.float32), torch.randn([256, 512], dtype=torch.float32), 3.0],
        [torch.randn([64, 64, 64], dtype=torch.float16), torch.randn([64, 64, 64], dtype=torch.float16), 1.0],
        [torch.randn([128, 128], dtype=torch.float32), torch.randn([128, 1], dtype=torch.float32), 1.0],
        [torch.randn([128, 128], dtype=torch.float32), torch.randn([1, 128], dtype=torch.float32), 1.0],
        [torch.randn([256, 256], dtype=torch.float16), torch.randn([256, 1], dtype=torch.float16), 2.0],
        [torch.randn([256, 256], dtype=torch.float16), torch.randn([1, 256], dtype=torch.float16), 0.5],
        [torch.randn([512, 512], dtype=torch.bfloat16), torch.randn([512, 1], dtype=torch.bfloat16), 1.0],
        [torch.randn([512, 512], dtype=torch.bfloat16), torch.randn([1, 512], dtype=torch.bfloat16), -0.5],
        [torch.randn([128, 256], dtype=torch.float32), torch.randn([128, 1], dtype=torch.float32), 1.0],
        [torch.randn([128, 256], dtype=torch.float32), torch.randn([1, 256], dtype=torch.float32), 1.0],
        [torch.randn([64, 64, 64], dtype=torch.float16), torch.randn([64, 1, 64], dtype=torch.float16), 1.0],
        [torch.randn([64, 64, 64], dtype=torch.float16), torch.randn([1, 64, 64], dtype=torch.float16), 1.0],
        [torch.randn([128, 128], dtype=torch.float32), torch.randn([128], dtype=torch.float32), 1.0],
        [torch.randn([256, 256], dtype=torch.float16), torch.randn([256], dtype=torch.float16), 0.5],
        [torch.randn([512, 512], dtype=torch.bfloat16), torch.randn([512], dtype=torch.bfloat16), 2.0],
        [torch.randn([64, 64, 64], dtype=torch.float32), torch.randn([64, 64], dtype=torch.float32), 1.0],
        [torch.randn([32, 32, 32], dtype=torch.float16), torch.randn([32, 32], dtype=torch.float16), 1.0],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), torch.randn([64, 64], dtype=torch.float32), 1.0],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), torch.randn([56, 56], dtype=torch.float16), 1.0],
        [torch.randn([2, 64, 56, 56], dtype=torch.bfloat16), torch.randn([56, 56], dtype=torch.bfloat16), 0.5],
        [torch.randn([1, 128, 28, 28], dtype=torch.float32), torch.randn([28, 28], dtype=torch.float32), 1.0],
        [torch.randn([1, 256, 14, 14], dtype=torch.float16), torch.randn([14, 14], dtype=torch.float16), 1.0],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), torch.randn([1, 16, 1, 1], dtype=torch.float32), 1.0],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), torch.randn([1, 64, 1, 1], dtype=torch.float16), 1.0],
        [torch.randn([2, 64, 56, 56], dtype=torch.bfloat16), torch.randn([1, 64, 1, 1], dtype=torch.bfloat16), 2.0],
        [torch.randn([1, 128, 28, 28], dtype=torch.float32), torch.randn([1, 128, 1, 1], dtype=torch.float32), 0.5],
        [torch.randn([1, 256, 14, 14], dtype=torch.float16), torch.randn([1, 256, 1, 1], dtype=torch.float16), 1.0],
        [torch.randn([4096, 8192], dtype=torch.float32), torch.randn([4096, 8192], dtype=torch.float32), 1.0],
        [torch.randn([4096, 8192], dtype=torch.float32), torch.randn([4096, 1], dtype=torch.float32), 1.0],
        [torch.randn([4096, 8192], dtype=torch.float32), torch.randn([1, 8192], dtype=torch.float32), 1.0],
        [torch.randn([8192, 4096], dtype=torch.bfloat16), torch.randn([8192, 4096], dtype=torch.bfloat16), 1.0],
        [torch.randn([8192, 4096], dtype=torch.bfloat16), torch.randn([8192, 1], dtype=torch.bfloat16), 0.5],
        [torch.randn([64, 128, 256], dtype=torch.float16), torch.randn([64, 128, 256], dtype=torch.float16), 1.0],
        [torch.randn([64, 128, 256], dtype=torch.float16), torch.randn([1, 128, 256], dtype=torch.float16), 1.0],
        [torch.randn([64, 128, 256], dtype=torch.float16), torch.randn([64, 1, 256], dtype=torch.float16), 1.0],
        [torch.randn([100, 200], dtype=torch.float32), torch.randn([100, 200], dtype=torch.float32), 1.0],
        [torch.randn([17, 31], dtype=torch.float32), torch.randn([17, 31], dtype=torch.float32), 1.0],
        [torch.randn([128, 256, 512], dtype=torch.float16), torch.randn([128, 256, 512], dtype=torch.float16), 1.0],
        [torch.randn([128, 256, 512], dtype=torch.float16), torch.randn([128, 1, 512], dtype=torch.float16), 1.0],
        [torch.randn([32, 64, 128, 256], dtype=torch.bfloat16), torch.randn([32, 64, 128, 256], dtype=torch.bfloat16), 1.0],
        [torch.randn([32, 64, 128, 256], dtype=torch.bfloat16), torch.randn([1, 64, 1, 256], dtype=torch.bfloat16), 1.0],
        [torch.randn([1024, 2048], dtype=torch.float32), torch.randn([1024, 2048], dtype=torch.float32), 1.0],
    ]
