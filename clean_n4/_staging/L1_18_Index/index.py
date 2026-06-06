import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that selects elements from a tensor along a dimension using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        """
        Selects elements from input tensor along the specified dimension using index.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension in which to index.
            index (torch.Tensor): The 1-D tensor containing the indices to index.

        Returns:
            torch.Tensor: Tensor with selected elements.
        """
        return torch.index_select(x, dim, index)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 0, torch.randint(0, 100, [32], dtype=torch.int64)],
        [torch.randn([256], dtype=torch.float32), 0, torch.randint(0, 100, [64], dtype=torch.int64)],
        [torch.randn([512], dtype=torch.float32), 0, torch.randint(0, 100, [128], dtype=torch.int64)],
        [torch.randn([128, 128], dtype=torch.float16), 0, torch.randint(0, 100, [32], dtype=torch.int64)],
        [torch.randn([128, 128], dtype=torch.float16), 1, torch.randint(0, 100, [64], dtype=torch.int64)],
        [torch.randn([128, 128], dtype=torch.float16), -1, torch.randint(0, 100, [48], dtype=torch.int64)],
        [torch.randn([256, 256], dtype=torch.float16), 0, torch.randint(0, 100, [64], dtype=torch.int64)],
        [torch.randn([256, 256], dtype=torch.float16), -2, torch.randint(0, 100, [128], dtype=torch.int64)],
        [torch.randn([64, 64], dtype=torch.bfloat16), 0, torch.randint(0, 100, [16], dtype=torch.int64)],
        [torch.randn([128, 256], dtype=torch.bfloat16), 1, torch.randint(0, 100, [64], dtype=torch.int64)],
        [torch.randn([64, 64, 64], dtype=torch.float32), 0, torch.randint(0, 100, [16], dtype=torch.int64)],
        [torch.randn([64, 64, 64], dtype=torch.float32), 1, torch.randint(0, 100, [32], dtype=torch.int64)],
        [torch.randn([64, 64, 64], dtype=torch.float32), 2, torch.randint(0, 100, [48], dtype=torch.int64)],
        [torch.randn([64, 64, 64], dtype=torch.float32), -1, torch.randint(0, 100, [24], dtype=torch.int64)],
        [torch.randn([32, 32, 32], dtype=torch.float16), -2, torch.randint(0, 100, [8], dtype=torch.int64)],
        [torch.randn([16, 16, 16, 16], dtype=torch.float16), 0, torch.randint(0, 100, [4], dtype=torch.int64)],
        [torch.randn([16, 16, 16, 16], dtype=torch.float16), -1, torch.randint(0, 100, [8], dtype=torch.int64)],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), 1, torch.randint(0, 100, [8], dtype=torch.int64)],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), -1, torch.randint(0, 100, [32], dtype=torch.int64)],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), 2, torch.randint(0, 100, [16], dtype=torch.int64)],
        [torch.randn([4096, 4096], dtype=torch.float32), 0, torch.randint(0, 100, [128], dtype=torch.int64)],
        [torch.randn([4096, 11008], dtype=torch.float16), 1, torch.randint(0, 100, [256], dtype=torch.int64)],
        [torch.randn([3584, 18944], dtype=torch.float16), -1, torch.randint(0, 100, [512], dtype=torch.int64)],
        [torch.randn([5120, 27648], dtype=torch.bfloat16), 0, torch.randint(0, 100, [1024], dtype=torch.int64)],
        [torch.randn([8192, 8192], dtype=torch.float32), -2, torch.randint(0, 100, [2048], dtype=torch.int64)],
        [torch.randn([8, 128, 4096], dtype=torch.float16), 1, torch.randint(0, 100, [64], dtype=torch.int64)],
        [torch.randn([4, 256, 5120], dtype=torch.float16), -1, torch.randint(0, 100, [256], dtype=torch.int64)],
        [torch.randn([2, 512, 4096], dtype=torch.bfloat16), 0, torch.randint(0, 100, [1], dtype=torch.int64)],
        [torch.randn([1, 1024, 3072], dtype=torch.float32), 1, torch.randint(0, 100, [128], dtype=torch.int64)],
        [torch.randn([2, 4, 256, 4096], dtype=torch.float16), 2, torch.randint(0, 100, [64], dtype=torch.int64)],
        [torch.randn([1, 8, 512, 3584], dtype=torch.bfloat16), -1, torch.randint(0, 100, [512], dtype=torch.int64)],
        [torch.randn([100], dtype=torch.float32), 0, torch.randint(0, 100, [10], dtype=torch.int64)],
        [torch.randn([100, 200], dtype=torch.float16), 1, torch.randint(0, 100, [50], dtype=torch.int64)],
        [torch.randn([17, 31], dtype=torch.float16), 0, torch.randint(0, 100, [8], dtype=torch.int64)],
        [torch.randn([7, 15, 23], dtype=torch.float32), 0, torch.randint(0, 100, [3], dtype=torch.int64)],
        [torch.randn([7, 15, 23], dtype=torch.float32), 1, torch.randint(0, 100, [5], dtype=torch.int64)],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), -1, torch.randint(0, 100, [4], dtype=torch.int64)],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 2, torch.randint(0, 100, [16], dtype=torch.int64)],
        [torch.randn([34, 66], dtype=torch.float32), 0, torch.randint(0, 100, [17], dtype=torch.int64)],
        [torch.randn([17, 33, 65], dtype=torch.float16), -2, torch.randint(0, 100, [8], dtype=torch.int64)],
        [torch.randn([9, 17, 31, 63], dtype=torch.bfloat16), 3, torch.randint(0, 100, [16], dtype=torch.int64)],
    ]
