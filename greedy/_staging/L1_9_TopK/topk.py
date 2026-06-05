import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that finds the k largest elements along a given dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> torch.Tensor:
        """
        Finds the k largest/smallest elements along a given dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            k (int): The number of elements to return.
            dim (int, optional): The dimension to find the top k along.
            largest (bool, optional): If True, return the largest elements.
            sorted (bool, optional): If True, the elements are returned sorted.

        Returns:
            torch.Tensor: The top k values tensor.
        """
        return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)[0]


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 10, -1, True],
        [torch.randn([256], dtype=torch.float32), 20, -1, False],
        [torch.randn([512], dtype=torch.float32), 50, 0, True],
        [torch.randn([1024], dtype=torch.float32), 100, -1, True],
        [torch.randn([128, 128], dtype=torch.float16), 16, 0, True],
        [torch.randn([128, 128], dtype=torch.float16), 32, 1, False],
        [torch.randn([256, 256], dtype=torch.float16), 64, -1, True],
        [torch.randn([256, 256], dtype=torch.float16), 16, -2, False],
        [torch.randn([64, 64], dtype=torch.bfloat16), 8, 0, True],
        [torch.randn([128, 256], dtype=torch.bfloat16), 32, -1, True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 16, 0, True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 8, 1, False],
        [torch.randn([64, 64, 64], dtype=torch.float32), 32, 2, True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 16, -1, False],
        [torch.randn([32, 32, 32], dtype=torch.float16), 8, -2, True],
        [torch.randn([16, 16, 16, 16], dtype=torch.float16), 4, 0, True],
        [torch.randn([16, 16, 16, 16], dtype=torch.float16), 8, -1, False],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), 16, 1, True],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 32, -1, True],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), 8, 2, False],
        [torch.randn([4096, 18432], dtype=torch.float32), 64, 0, True],
        [torch.randn([8192, 16384], dtype=torch.float16), 128, -1, True],
        [torch.randn([100], dtype=torch.float32), 10, 0, False],
        [torch.randn([100, 2007], dtype=torch.float16), 50, 1, True],
        [torch.randn([17, 301], dtype=torch.float16), 15, -1, True],
        [torch.randn([7, 15, 23], dtype=torch.float32), 5, 0, True],
        [torch.randn([7, 15, 23], dtype=torch.float32), 8, 1, False],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), 3, -1, True],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 16, 2, False],
    ]
