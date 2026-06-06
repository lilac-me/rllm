import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            approximate (str, optional): The gelu approximation algorithm to use: 'none'|'tanh'.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x, approximate=approximate)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 'none'],
        [torch.randn([256], dtype=torch.float32), 'tanh'],
        [torch.randn([512], dtype=torch.float32), 'none'],
        [torch.randn([1024], dtype=torch.float32), 'tanh'],
        [torch.randn([2048], dtype=torch.float32), 'none'],
        [torch.randn([4096], dtype=torch.float32), 'tanh'],
        [torch.randn([128, 128], dtype=torch.float16), 'none'],
        [torch.randn([256, 256], dtype=torch.float16), 'tanh'],
        [torch.randn([512, 512], dtype=torch.float16), 'none'],
        [torch.randn([1024, 1024], dtype=torch.float16), 'tanh'],
        [torch.randn([64, 64], dtype=torch.bfloat16), 'none'],
        [torch.randn([32, 32], dtype=torch.bfloat16), 'tanh'],
        [torch.randn([128, 256], dtype=torch.float32), 'none'],
        [torch.randn([256, 512], dtype=torch.float32), 'tanh'],
        [torch.randn([512, 1024], dtype=torch.float32), 'none'],
        [torch.randn([1024, 2048], dtype=torch.float32), 'tanh'],
        [torch.randn([64, 64, 64], dtype=torch.float16), 'tanh'],
        [torch.randn([32, 32, 32], dtype=torch.float16), 'none'],
        [torch.randn([16, 16, 16], dtype=torch.float16), 'tanh'],
        [torch.randn([128, 64, 32], dtype=torch.bfloat16), 'none'],
        [torch.randn([64, 32, 16], dtype=torch.bfloat16), 'tanh'],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 'tanh'],
        [torch.randn([8, 8, 8, 8], dtype=torch.float32), 'none'],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), 'none'],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 'tanh'],
        [torch.randn([1, 64, 56, 56], dtype=torch.float32), 'none'],
        [torch.randn([2, 64, 56, 56], dtype=torch.float32), 'tanh'],
        [torch.randn([4, 64, 56, 56], dtype=torch.float16), 'none'],
        [torch.randn([1, 128, 28, 28], dtype=torch.float16), 'tanh'],
        [torch.randn([1, 256, 14, 14], dtype=torch.bfloat16), 'none'],
        [torch.randn([1, 512, 7, 7], dtype=torch.bfloat16), 'tanh'],
        [torch.randn([4096, 18432], dtype=torch.float32), 'none'],
        [torch.randn([4096, 24576], dtype=torch.bfloat16), 'tanh'],
        [torch.randn([8192, 16384], dtype=torch.float16), 'none'],
        [torch.randn([2048, 13824], dtype=torch.float32), 'tanh'],
        [torch.randn([100], dtype=torch.float32), 'none'],
        [torch.randn([18432], dtype=torch.float32), 'tanh'],
        [torch.randn([100, 2007], dtype=torch.float16), 'none'],
        [torch.randn([17, 301], dtype=torch.float16), 'tanh'],
        [torch.randn([13, 2117], dtype=torch.bfloat16), 'none'],
        [torch.randn([7, 15, 23], dtype=torch.float32), 'tanh'],
        [torch.randn([11, 19, 29], dtype=torch.float32), 'none'],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), 'tanh'],
        [torch.randn([1, 7, 13, 17], dtype=torch.float16), 'none'],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 'tanh'],
        [torch.randn([2, 33, 55, 77], dtype=torch.bfloat16), 'none'],
        [torch.randn([1, 13, 65, 65], dtype=torch.float32), 'tanh'],
        [torch.randn([1, 11, 113, 113], dtype=torch.float32), 'none'],
        [torch.randn([123, 6144], dtype=torch.float16), 'tanh'],
        [torch.randn([789, 12288], dtype=torch.float16), 'none'],
    ]
