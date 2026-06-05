# torch.histc(input, bins=100, min=0, max=0, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.histc.html#torch.histc

import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the histogram of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, bins: int, min: int, max: int) -> torch.Tensor:
        """
        Computes the histogram of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            bins (int): Number of histogram bins.
            min (int): Lower end of the range (inclusive).
            max (int): Upper end of the range (inclusive).

        Returns:
            torch.Tensor: Histogram tensor of shape (bins,).
        """
        return torch.histc(x, bins=bins, min=min, max=max)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 100, 0.0, 1.0],
        [torch.randn([256], dtype=torch.float32), 50, -1.0, 1.0],
        [torch.randn([512], dtype=torch.float32), 64, 0.0, 10.0],
        [torch.randn([1024], dtype=torch.float32), 128, 0.0, 1.0],
        [torch.randn([4096], dtype=torch.float32), 256, -5.0, 5.0],
        [torch.randn([16384], dtype=torch.float32), 100, 0.0, 1.0],
        [torch.randn([32768], dtype=torch.float16), 64, 0.0, 1.0],
        [torch.randn([65536], dtype=torch.float16), 128, -1.0, 1.0],
        [torch.randn([100], dtype=torch.float32), 32, 0.0, 100.0],
        [torch.randn([18432], dtype=torch.float32), 100, -2.0, 2.0],
        [torch.randn([24576], dtype=torch.float16), 64, 0.0, 1.0],
        [torch.randn([128, 128], dtype=torch.float16), 100, 0.0, 1.0],
        [torch.randn([256, 256], dtype=torch.float16), 50, -1.0, 1.0],
        [torch.randn([64, 64], dtype=torch.float32), 32, 0.0, 10.0],
        [torch.randn([128, 256], dtype=torch.float32), 128, -3.0, 3.0],
    ]
