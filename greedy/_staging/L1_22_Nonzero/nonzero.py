import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that returns indices of non-zero elements.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, as_tuple: bool = False):
        """
        Returns indices of non-zero elements in the tensor.

        Args:
            x (torch.Tensor): Input tensor.
            as_tuple (bool, optional): If True, returns a tuple of 1-D tensors, one for each dimension.

        Returns:
            torch.Tensor or tuple: Indices of non-zero elements.
        """
        return torch.nonzero(x, as_tuple=as_tuple)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), False],
        [torch.randn([256], dtype=torch.float32), True],
        [torch.randn([512], dtype=torch.float16), False],
        [torch.randn([1024], dtype=torch.bfloat16), True],
        [torch.randn([128, 256], dtype=torch.float32), False],
        [torch.randn([256, 512], dtype=torch.float32), True],
        [torch.randn([512, 1024], dtype=torch.float16), False],
        [torch.randn([1024, 2048], dtype=torch.bfloat16), True],
        [torch.randn([64, 128, 256], dtype=torch.float32), False],
        [torch.randn([32, 64, 128], dtype=torch.float32), True],
        [torch.randn([16, 128, 256], dtype=torch.float16), False],
        [torch.randn([8, 256, 512], dtype=torch.bfloat16), True],
        [torch.randn([1, 64, 128, 128], dtype=torch.float32), False],
        [torch.randn([1, 128, 64, 64], dtype=torch.float32), True],
        [torch.randn([1, 256, 32, 32], dtype=torch.float16), False],
        [torch.randn([1, 512, 16, 16], dtype=torch.bfloat16), True],
        [torch.randint(0, 100, [128], dtype=torch.int32), False],
        [torch.randint(0, 100, [256], dtype=torch.int64), True],
        [torch.randint(0, 100, [64, 128], dtype=torch.int32), False],
        [torch.randint(0, 100, [128, 256], dtype=torch.int64), True],
        [torch.randn([1536], dtype=torch.float32), False],
        [torch.randn([4096], dtype=torch.float16), True],
        [torch.randn([8192], dtype=torch.bfloat16), False],
        [torch.randn([4096, 4096], dtype=torch.float32), True],
        [torch.randn([4096, 11008], dtype=torch.float16), False],
        [torch.randn([5120, 13824], dtype=torch.bfloat16), True],
        [torch.randn([3584, 18944], dtype=torch.float32), False],
        [torch.randn([5120, 27648], dtype=torch.float16), True],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), False],
        [torch.randn([1, 3, 224, 224], dtype=torch.float16), True],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), False],
        [torch.randn([1, 128, 28, 28], dtype=torch.float32), True],
        [torch.randn([1, 256, 14, 14], dtype=torch.float32), False],
        [torch.randn([1, 512, 7, 7], dtype=torch.float16), True],
        [torch.randn([100], dtype=torch.float32), False],
        [torch.randn([200], dtype=torch.float16), True],
        [torch.randn([34, 66], dtype=torch.float32), False],
        [torch.randn([17, 33], dtype=torch.float16), True],
        [torch.randn([65, 129], dtype=torch.bfloat16), False],
        [torch.randn([1, 16, 100, 100], dtype=torch.float32), True],
        [torch.randn([1, 32, 50, 50], dtype=torch.float16), False],
        [torch.randn([1, 64, 25, 25], dtype=torch.bfloat16), True],
        [torch.randn([2048, 2048], dtype=torch.float32), False],
        [torch.randn([3072, 3072], dtype=torch.float16), True],
        [torch.randn([6144, 6144], dtype=torch.float32), False],
        [torch.randn([8192, 8192], dtype=torch.float16), True],
        [torch.randn([32, 128, 128], dtype=torch.float32), False],
        [torch.randn([16, 64, 64], dtype=torch.float16), True],
        [torch.randn([1, 64, 64, 64], dtype=torch.float32), False],
        [torch.randn([1, 128, 32, 32], dtype=torch.float16), True],
    ]
