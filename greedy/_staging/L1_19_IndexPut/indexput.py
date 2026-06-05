import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that puts values into a tensor at specified indices (1D case).
    For a 1D tensor x, uses a single index tensor to put values.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, values: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        """
        Puts values into the tensor at the specified indices.

        Args:
            x (torch.Tensor): Input tensor.
            index (torch.Tensor): 1-D index tensor for the first dimension.
            values (torch.Tensor): Values to put at the specified indices.
            accumulate (bool, optional): Whether to accumulate values at the indices.

        Returns:
            torch.Tensor: Tensor with values put at specified indices.
        """
        x.index_put_((index,), values, accumulate=accumulate)
        return x


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), torch.randint(0, 100, [64], dtype=torch.int64), torch.randn([64], dtype=torch.float32), False],
        [torch.randn([256], dtype=torch.float32), torch.randint(0, 100, [128], dtype=torch.int64), torch.randn([128], dtype=torch.float32), True],
        [torch.randn([512], dtype=torch.float32), torch.randint(0, 100, [256], dtype=torch.int64), torch.randn([256], dtype=torch.float32), False],
        [torch.randn([1024], dtype=torch.float32), torch.randint(0, 100, [512], dtype=torch.int64), torch.randn([512], dtype=torch.float32), True],
        [torch.randn([128], dtype=torch.float16), torch.randint(0, 100, [32], dtype=torch.int64), torch.randn([32], dtype=torch.float16), False],
        [torch.randn([256], dtype=torch.float16), torch.randint(0, 100, [64], dtype=torch.int64), torch.randn([64], dtype=torch.float16), True],
        [torch.randn([512], dtype=torch.float16), torch.randint(0, 100, [128], dtype=torch.int64), torch.randn([128], dtype=torch.float16), False],
        [torch.randn([64], dtype=torch.bfloat16), torch.randint(0, 100, [16], dtype=torch.int64), torch.randn([16], dtype=torch.bfloat16), False],
        [torch.randn([128], dtype=torch.bfloat16), torch.randint(0, 100, [32], dtype=torch.int64), torch.randn([32], dtype=torch.bfloat16), True],
        [torch.randn([2048], dtype=torch.float32), torch.randint(0, 100, [1024], dtype=torch.int64), torch.randn([1024], dtype=torch.float32), False],
        [torch.randn([4096], dtype=torch.float16), torch.randint(0, 100, [2048], dtype=torch.int64), torch.randn([2048], dtype=torch.float16), True],
        [torch.randn([8192], dtype=torch.float16), torch.randint(0, 100, [4096], dtype=torch.int64), torch.randn([4096], dtype=torch.float16), False],
        [torch.randn([16384], dtype=torch.bfloat16), torch.randint(0, 100, [8192], dtype=torch.int64), torch.randn([8192], dtype=torch.bfloat16), True],
        [torch.randn([32768], dtype=torch.float32), torch.randint(0, 100, [16384], dtype=torch.int64), torch.randn([16384], dtype=torch.float32), False],
        [torch.randn([65536], dtype=torch.float16), torch.randint(0, 100, [32768], dtype=torch.int64), torch.randn([32768], dtype=torch.float16), True],
        [torch.randn([100], dtype=torch.float32), torch.randint(0, 100, [50], dtype=torch.int64), torch.randn([50], dtype=torch.float32), False],
        [torch.randn([100], dtype=torch.float16), torch.randint(0, 100, [25], dtype=torch.int64), torch.randn([25], dtype=torch.float16), True],
        [torch.randn([17], dtype=torch.float32), torch.randint(0, 100, [8], dtype=torch.int64), torch.randn([8], dtype=torch.float32), False],
        [torch.randn([31], dtype=torch.float16), torch.randint(0, 100, [15], dtype=torch.int64), torch.randn([15], dtype=torch.float16), True],
        [torch.randn([7], dtype=torch.bfloat16), torch.randint(0, 100, [3], dtype=torch.int64), torch.randn([3], dtype=torch.bfloat16), False],
        [torch.randn([15], dtype=torch.float32), torch.randint(0, 100, [7], dtype=torch.int64), torch.randn([7], dtype=torch.float32), True],
        [torch.randn([23], dtype=torch.float16), torch.randint(0, 100, [11], dtype=torch.int64), torch.randn([11], dtype=torch.float16), False],
        [torch.randn([33], dtype=torch.float32), torch.randint(0, 100, [16], dtype=torch.int64), torch.randn([16], dtype=torch.float32), True],
        [torch.randn([65], dtype=torch.float16), torch.randint(0, 100, [32], dtype=torch.int64), torch.randn([32], dtype=torch.float16), False],
        [torch.randn([111], dtype=torch.bfloat16), torch.randint(0, 100, [55], dtype=torch.int64), torch.randn([55], dtype=torch.bfloat16), True],
        [torch.randn([34], dtype=torch.float32), torch.randint(0, 100, [17], dtype=torch.int64), torch.randn([17], dtype=torch.float32), False],
        [torch.randn([66], dtype=torch.float16), torch.randint(0, 100, [33], dtype=torch.int64), torch.randn([33], dtype=torch.float16), True],
        [torch.randn([1536], dtype=torch.float32), torch.randint(0, 100, [768], dtype=torch.int64), torch.randn([768], dtype=torch.float32), False],
        [torch.randn([2048], dtype=torch.float16), torch.randint(0, 100, [1024], dtype=torch.int64), torch.randn([1024], dtype=torch.float16), True],
        [torch.randn([2560], dtype=torch.float16), torch.randint(0, 100, [1280], dtype=torch.int64), torch.randn([1280], dtype=torch.float16), False],
        [torch.randn([3072], dtype=torch.bfloat16), torch.randint(0, 100, [1536], dtype=torch.int64), torch.randn([1536], dtype=torch.bfloat16), True],
        [torch.randn([3584], dtype=torch.float32), torch.randint(0, 100, [1792], dtype=torch.int64), torch.randn([1792], dtype=torch.float32), False],
        [torch.randn([4096], dtype=torch.float16), torch.randint(0, 100, [2048], dtype=torch.int64), torch.randn([2048], dtype=torch.float16), True],
        [torch.randn([5120], dtype=torch.float16), torch.randint(0, 100, [2560], dtype=torch.int64), torch.randn([2560], dtype=torch.float16), False],
        [torch.randn([6144], dtype=torch.bfloat16), torch.randint(0, 100, [3072], dtype=torch.int64), torch.randn([3072], dtype=torch.bfloat16), True],
        [torch.randn([7168], dtype=torch.float32), torch.randint(0, 100, [3584], dtype=torch.int64), torch.randn([3584], dtype=torch.float32), False],
        [torch.randn([8192], dtype=torch.float16), torch.randint(0, 100, [4096], dtype=torch.int64), torch.randn([4096], dtype=torch.float16), True],
        [torch.randn([11008], dtype=torch.float32), torch.randint(0, 100, [5504], dtype=torch.int64), torch.randn([5504], dtype=torch.float32), False],
        [torch.randn([12288], dtype=torch.float16), torch.randint(0, 100, [6144], dtype=torch.int64), torch.randn([6144], dtype=torch.float16), True],
        [torch.randn([13824], dtype=torch.float16), torch.randint(0, 100, [6912], dtype=torch.int64), torch.randn([6912], dtype=torch.float16), False],
        [torch.randn([16384], dtype=torch.bfloat16), torch.randint(0, 100, [8192], dtype=torch.int64), torch.randn([8192], dtype=torch.bfloat16), True],
        [torch.randn([18432], dtype=torch.float32), torch.randint(0, 100, [9216], dtype=torch.int64), torch.randn([9216], dtype=torch.float32), False],
        [torch.randn([18944], dtype=torch.float16), torch.randint(0, 100, [9472], dtype=torch.int64), torch.randn([9472], dtype=torch.float16), True],
        [torch.randn([20480], dtype=torch.float16), torch.randint(0, 100, [10240], dtype=torch.int64), torch.randn([10240], dtype=torch.float16), False],
        [torch.randn([24576], dtype=torch.bfloat16), torch.randint(0, 100, [12288], dtype=torch.int64), torch.randn([12288], dtype=torch.bfloat16), True],
        [torch.randn([27648], dtype=torch.float32), torch.randint(0, 100, [13824], dtype=torch.int64), torch.randn([13824], dtype=torch.float32), False],
    ]
