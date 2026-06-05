# torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.sum.html#torch.sum

import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the sum of elements along specified dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        """
        Returns the sum of elements along specified dimensions.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int or tuple of ints, optional): Dimension(s) to reduce.
            keepdim (bool): Whether to keep the reduced dimension(s).

        Returns:
            torch.Tensor: Tensor with sum along specified dimensions.
        """
        return torch.sum(x, dim=dim, keepdim=keepdim)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 0, False],
        [torch.randn([256], dtype=torch.float32), 0, True],
        [torch.randn([512], dtype=torch.float32), -1, False],
        [torch.randn([1024], dtype=torch.float32), 0, True],
        [torch.randn([128, 128], dtype=torch.float16), 0, False],
        [torch.randn([128, 128], dtype=torch.float16), 1, True],
        [torch.randn([256, 256], dtype=torch.float16), -1, False],
        [torch.randn([256, 256], dtype=torch.float16), -2, True],
        [torch.randn([64, 64], dtype=torch.bfloat16), 0, False],
        [torch.randn([128, 256], dtype=torch.bfloat16), 1, True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 0, False],
        [torch.randn([64, 64, 64], dtype=torch.float32), 1, True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 2, False],
        [torch.randn([64, 64, 64], dtype=torch.float32), -1, True],
        [torch.randn([32, 32, 32], dtype=torch.float16), -2, False],
        [torch.randn([32, 32, 32], dtype=torch.float16), -3, True],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 0, False],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 1, True],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 2, False],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), 3, True],
        [torch.randn([16, 16, 16, 16], dtype=torch.float32), -1, False],
        [torch.randn([1, 16, 64, 64], dtype=torch.float16), 1, True],
        [torch.randn([1, 16, 64, 64], dtype=torch.float16), 2, False],
        [torch.randn([1, 16, 64, 64], dtype=torch.float16), -1, True],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 2, False],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 3, True],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), -2, False],
        [torch.randn([1, 128, 28, 28], dtype=torch.bfloat16), -3, True],
        [torch.randn([4096, 18432], dtype=torch.float32), 0, False],
        [torch.randn([4096, 18432], dtype=torch.float32), 1, True],
        [torch.randn([8192, 16384], dtype=torch.float16), -1, False],
        [torch.randn([8192, 16384], dtype=torch.float16), -2, True],
        [torch.randn([100], dtype=torch.float32), 0, False],
        [torch.randn([100, 2007], dtype=torch.float16), 0, True],
        [torch.randn([100, 2007], dtype=torch.float16), 1, False],
        [torch.randn([17, 301], dtype=torch.float16), -1, True],
        [torch.randn([13, 2117], dtype=torch.bfloat16), -2, False],
        [torch.randn([7, 15, 23], dtype=torch.float32), 0, True],
        [torch.randn([7, 15, 23], dtype=torch.float32), 1, False],
        [torch.randn([7, 15, 23], dtype=torch.float32), 2, True],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), 0, False],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), -1, True],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 2, False],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 3, True],
    ]
