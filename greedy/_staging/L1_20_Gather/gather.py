import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that gathers values along a dimension using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, sparse_grad: bool = False) -> torch.Tensor:
        """
        Gathers values along an axis specified by dim.

        Args:
            x (torch.Tensor): Input tensor (src).
            dim (int): The axis along which to index.
            index (torch.Tensor): The indices of elements to gather.
            sparse_grad (bool, optional): If True, gradient w.r.t. input will be a sparse tensor.

        Returns:
            torch.Tensor: Tensor with gathered values, same shape as index.
        """
        return torch.gather(x, dim, index, sparse_grad=sparse_grad)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 0, torch.randint(0, 100, [64], dtype=torch.int64), False],
        [torch.randn([256], dtype=torch.float32), 0, torch.randint(0, 100, [128], dtype=torch.int64), True],
        [torch.randn([512], dtype=torch.float32), 0, torch.randint(0, 100, [256], dtype=torch.int64), False],
        [torch.randn([128, 128], dtype=torch.float16), 0, torch.randint(0, 100, [64, 128], dtype=torch.int64), False],
        [torch.randn([128, 128], dtype=torch.float16), 1, torch.randint(0, 100, [128, 64], dtype=torch.int64), True],
        [torch.randn([128, 128], dtype=torch.float16), -1, torch.randint(0, 100, [128, 48], dtype=torch.int64), False],
        [torch.randn([256, 256], dtype=torch.float16), 0, torch.randint(0, 100, [128, 256], dtype=torch.int64), True],
        [torch.randn([256, 256], dtype=torch.float16), -2, torch.randint(0, 100, [64, 256], dtype=torch.int64), False],
        [torch.randn([64, 64], dtype=torch.bfloat16), 0, torch.randint(0, 100, [32, 64], dtype=torch.int64), False],
        [torch.randn([128, 256], dtype=torch.bfloat16), 1, torch.randint(0, 100, [128, 64], dtype=torch.int64), True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 0, torch.randint(0, 100, [32, 64, 64], dtype=torch.int64), False],
        [torch.randn([64, 64, 64], dtype=torch.float32), 1, torch.randint(0, 100, [64, 32, 64], dtype=torch.int64), True],
        [torch.randn([64, 64, 64], dtype=torch.float32), 2, torch.randint(0, 100, [64, 64, 48], dtype=torch.int64), False],
        [torch.randn([64, 64, 64], dtype=torch.float32), -1, torch.randint(0, 100, [64, 64, 32], dtype=torch.int64), True],
        [torch.randn([32, 32, 32], dtype=torch.float16), -2, torch.randint(0, 100, [16, 32, 32], dtype=torch.int64), False],
        [torch.randn([16, 16, 16, 16], dtype=torch.float16), 0, torch.randint(0, 100, [8, 16, 16, 16], dtype=torch.int64), False],
        [torch.randn([16, 16, 16, 16], dtype=torch.float16), -1, torch.randint(0, 100, [16, 16, 16, 8], dtype=torch.int64), True],
        [torch.randn([1, 16, 64, 64], dtype=torch.float32), 1, torch.randint(0, 100, [1, 8, 64, 64], dtype=torch.int64), False],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), -1, torch.randint(0, 100, [1, 3, 224, 112], dtype=torch.int64), True],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), 2, torch.randint(0, 100, [1, 64, 28, 56], dtype=torch.int64), False],
        [torch.randn([4096, 4096], dtype=torch.float32), 0, torch.randint(0, 100, [2048, 4096], dtype=torch.int64), False],
        [torch.randn([4096, 11008], dtype=torch.float16), 1, torch.randint(0, 100, [4096, 5504], dtype=torch.int64), True],
        [torch.randn([3584, 18944], dtype=torch.float16), -1, torch.randint(0, 100, [3584, 9472], dtype=torch.int64), False],
        [torch.randn([5120, 27648], dtype=torch.bfloat16), 0, torch.randint(0, 100, [2560, 27648], dtype=torch.int64), True],
        [torch.randn([8192, 8192], dtype=torch.float32), -2, torch.randint(0, 100, [4096, 8192], dtype=torch.int64), False],
        [torch.randn([8, 128, 4096], dtype=torch.float16), 1, torch.randint(0, 100, [8, 64, 4096], dtype=torch.int64), True],
        [torch.randn([4, 256, 5120], dtype=torch.float16), -1, torch.randint(0, 100, [4, 256, 2560], dtype=torch.int64), False],
        [torch.randn([2, 512, 4096], dtype=torch.bfloat16), 0, torch.randint(0, 100, [1, 512, 4096], dtype=torch.int64), True],
        [torch.randn([1, 1024, 3072], dtype=torch.float32), 1, torch.randint(0, 100, [1, 512, 3072], dtype=torch.int64), False],
        [torch.randn([2, 4, 256, 4096], dtype=torch.float16), 2, torch.randint(0, 100, [2, 4, 128, 4096], dtype=torch.int64), True],
        [torch.randn([1, 8, 512, 3584], dtype=torch.bfloat16), -1, torch.randint(0, 100, [1, 8, 512, 1792], dtype=torch.int64), False],
        [torch.randn([100], dtype=torch.float32), 0, torch.randint(0, 100, [50], dtype=torch.int64), False],
        [torch.randn([100, 200], dtype=torch.float16), 1, torch.randint(0, 100, [100, 100], dtype=torch.int64), True],
        [torch.randn([17, 31], dtype=torch.float16), 0, torch.randint(0, 100, [8, 31], dtype=torch.int64), False],
        [torch.randn([7, 15, 23], dtype=torch.float32), 0, torch.randint(0, 100, [3, 15, 23], dtype=torch.int64), True],
        [torch.randn([7, 15, 23], dtype=torch.float32), 1, torch.randint(0, 100, [7, 7, 23], dtype=torch.int64), False],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), -1, torch.randint(0, 100, [3, 5, 7, 5], dtype=torch.int64), True],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 2, torch.randint(0, 100, [1, 5, 55, 111], dtype=torch.int64), False],
        [torch.randn([34, 66], dtype=torch.float32), 0, torch.randint(0, 100, [17, 66], dtype=torch.int64), True],
        [torch.randn([17, 33, 65], dtype=torch.float16), -2, torch.randint(0, 100, [8, 33, 65], dtype=torch.int64), False],
        [torch.randn([9, 17, 31, 63], dtype=torch.bfloat16), 3, torch.randint(0, 100, [9, 17, 31, 31], dtype=torch.int64), True],
        [torch.randn([1536, 1536], dtype=torch.float32), 0, torch.randint(0, 100, [768, 1536], dtype=torch.int64), False],
        [torch.randn([2048, 2048], dtype=torch.float16), 1, torch.randint(0, 100, [2048, 1024], dtype=torch.int64), True],
        [torch.randn([3072, 3072], dtype=torch.float16), -1, torch.randint(0, 100, [3072, 1536], dtype=torch.int64), False],
        [torch.randn([4096, 4096], dtype=torch.bfloat16), 0, torch.randint(0, 100, [2048, 4096], dtype=torch.int64), True],
        [torch.randn([5120, 5120], dtype=torch.float16), 1, torch.randint(0, 100, [5120, 2560], dtype=torch.int64), False],
        [torch.randn([6144, 6144], dtype=torch.float32), -2, torch.randint(0, 100, [3072, 6144], dtype=torch.int64), True],
    ]
