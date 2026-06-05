import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that repeats a tensor along specified dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats: tuple) -> torch.Tensor:
        """
        Repeats the tensor along each dimension.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (tuple): Number of repeats for each dimension.

        Returns:
            torch.Tensor: Repeated tensor.
        """
        return x.repeat(*repeats)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), [2]],
        [torch.randn([256], dtype=torch.float32), [4]],
        [torch.randn([512], dtype=torch.float16), [2]],
        [torch.randn([1024], dtype=torch.bfloat16), [3]],
        [torch.randn([128, 256], dtype=torch.float32), [2, 2]],
        [torch.randn([256, 512], dtype=torch.float32), [1, 4]],
        [torch.randn([512, 1024], dtype=torch.float16), [2, 1]],
        [torch.randn([1024, 2048], dtype=torch.bfloat16), [1, 2]],
        [torch.randn([64, 128, 256], dtype=torch.float32), [2, 1, 2]],
        [torch.randn([32, 64, 128], dtype=torch.float32), [1, 2, 1]],
        [torch.randn([16, 128, 256], dtype=torch.float16), [2, 2, 2]],
        [torch.randn([8, 256, 512], dtype=torch.bfloat16), [1, 1, 2]],
        [torch.randn([1, 64, 128, 128], dtype=torch.float32), [1, 2, 1, 1]],
        [torch.randn([1, 128, 64, 64], dtype=torch.float32), [1, 1, 2, 2]],
        [torch.randn([1, 256, 32, 32], dtype=torch.float16), [2, 1, 1, 1]],
        [torch.randn([1, 512, 16, 16], dtype=torch.bfloat16), [1, 2, 2, 2]],
        [torch.randn([1536], dtype=torch.float32), [2]],
        [torch.randn([4096], dtype=torch.float16), [4]],
        [torch.randn([8192], dtype=torch.bfloat16), [2]],
        [torch.randn([4096, 4096], dtype=torch.float32), [1, 2]],
        [torch.randn([4096, 11008], dtype=torch.float16), [2, 1]],
        [torch.randn([5120, 13824], dtype=torch.bfloat16), [1, 1]],
        [torch.randn([3584, 18944], dtype=torch.float32), [2, 2]],
        [torch.randn([5120, 27648], dtype=torch.float16), [1, 2]],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), [1, 1, 1, 1]],
        [torch.randn([1, 3, 224, 224], dtype=torch.float16), [2, 1, 1, 1]],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), [1, 2, 1, 1]],
        [torch.randn([1, 128, 28, 28], dtype=torch.float32), [1, 1, 2, 2]],
        [torch.randn([1, 256, 14, 14], dtype=torch.float32), [2, 1, 1, 2]],
        [torch.randn([1, 512, 7, 7], dtype=torch.float16), [1, 2, 2, 1]],
        [torch.randn([100], dtype=torch.float32), [3]],
        [torch.randn([200], dtype=torch.float16), [2]],
        [torch.randn([34, 66], dtype=torch.float32), [2, 2]],
        [torch.randn([17, 33], dtype=torch.float16), [1, 3]],
        [torch.randn([65, 129], dtype=torch.bfloat16), [2, 1]],
        [torch.randn([1, 16, 100, 100], dtype=torch.float32), [2, 1, 1, 1]],
        [torch.randn([1, 32, 50, 50], dtype=torch.float16), [1, 2, 2, 2]],
        [torch.randn([1, 64, 25, 25], dtype=torch.bfloat16), [2, 1, 1, 2]],
        [torch.randn([1, 3, 112, 112], dtype=torch.float32), [1, 1, 2, 1]],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), [1, 2, 1, 2]],
        [torch.randn([2048, 2048], dtype=torch.float32), [2, 2]],
        [torch.randn([3072, 3072], dtype=torch.float16), [1, 3]],
        [torch.randn([6144, 6144], dtype=torch.float32), [2, 1]],
        [torch.randn([8192, 8192], dtype=torch.float16), [1, 2]],
        [torch.randn([32, 128, 128], dtype=torch.float32), [2, 1, 2]],
        [torch.randn([16, 64, 64], dtype=torch.float16), [1, 2, 1]],
        [torch.randn([1, 64, 64, 64], dtype=torch.float32), [1, 2, 1, 2]],
        [torch.randn([1, 128, 32, 32], dtype=torch.float16), [2, 1, 2, 1]],
        [torch.randn([1, 256, 16, 16], dtype=torch.bfloat16), [1, 1, 2, 2]],
    ]
