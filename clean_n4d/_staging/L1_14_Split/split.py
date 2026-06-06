import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that splits a tensor into chunks.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, split_size_or_sections, dim: int = 0):
        """
        Splits the tensor into chunks.

        Args:
            x (torch.Tensor): Input tensor to split.
            split_size_or_sections (int or list): If int, size of each chunk. If list, sizes of each chunk.
            dim (int, optional): Dimension along which to split the tensor.

        Returns:
            tuple: Tuple of tensors resulting from the split.
        """
        return torch.split(x, split_size_or_sections, dim=dim)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), 32, 0],
        [torch.randn([256], dtype=torch.float32), 64, 0],
        [torch.randn([512], dtype=torch.float16), 128, 0],
        [torch.randn([1024], dtype=torch.float16), 256, -1],
        [torch.randn([2048], dtype=torch.bfloat16), 512, 0],
        [torch.randn([128, 256], dtype=torch.float32), 64, 0],
        [torch.randn([128, 256], dtype=torch.float32), 64, 1],
        [torch.randn([128, 512], dtype=torch.float32), 128, -1],
        [torch.randn([256, 256], dtype=torch.float16), 64, 0],
        [torch.randn([256, 256], dtype=torch.float16), 128, 1],
        [torch.randn([256, 512], dtype=torch.float16), 256, -2],
        [torch.randn([64, 1024], dtype=torch.bfloat16), 256, -1],
        [torch.randn([32, 128, 256], dtype=torch.float32), 16, 0],
        [torch.randn([32, 128, 256], dtype=torch.float32), 64, 1],
        [torch.randn([32, 128, 512], dtype=torch.float32), 256, -1],
        [torch.randn([16, 64, 1024], dtype=torch.float16), 256, 2],
        [torch.randn([16, 64, 1024], dtype=torch.float16), 512, -1],
        [torch.randn([8, 32, 2048], dtype=torch.bfloat16), 1024, -1],
        [torch.randn([1, 64, 56, 56], dtype=torch.float32), 32, 1],
        [torch.randn([1, 64, 56, 56], dtype=torch.float32), 28, 2],
        [torch.randn([1, 128, 28, 28], dtype=torch.float16), 64, 1],
        [torch.randn([1, 128, 28, 28], dtype=torch.float16), 14, -1],
        [torch.randn([1, 256, 14, 14], dtype=torch.bfloat16), 128, 1],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), 112, 2],
        [torch.randn([1, 3, 224, 224], dtype=torch.float16), 112, -1],
        [torch.randn([4096, 4096], dtype=torch.float32), 1024, 0],
        [torch.randn([4096, 4096], dtype=torch.float32), 2048, 1],
        [torch.randn([4096, 8192], dtype=torch.float16), 2048, 0],
        [torch.randn([4096, 11008], dtype=torch.float16), 2752, -1],
        [torch.randn([8192, 16384], dtype=torch.bfloat16), 4096, 0],
        [torch.randn([256], dtype=torch.float32), [64, 128, 64], 0],
        [torch.randn([512], dtype=torch.float32), [128, 256, 128], -1],
        [torch.randn([128, 256], dtype=torch.float16), [64, 64], 0],
        [torch.randn([128, 512], dtype=torch.float16), [256, 256], 1],
        [torch.randn([64, 1024], dtype=torch.bfloat16), [512, 512], -1],
        [torch.randn([32, 128, 512], dtype=torch.float32), [256, 256], 2],
        [torch.randn([4096, 4096], dtype=torch.float16), [2048, 2048], 1],
        [torch.randn([4096, 8192], dtype=torch.bfloat16), [2048, 4096, 2048], -1],
        [torch.randn([100], dtype=torch.float32), 25, 0],
        [torch.randn([100, 200], dtype=torch.float32), 50, 0],
        [torch.randn([100, 200], dtype=torch.float16), 100, 1],
        [torch.randn([100, 200], dtype=torch.float16), [50, 100, 50], -1],
        [torch.randn([17, 32], dtype=torch.bfloat16), 16, 0],
        [torch.randn([17, 64], dtype=torch.float32), 32, -1],
        [torch.randn([13, 100], dtype=torch.float16), [50, 30, 20], 1],
        [torch.randn([7, 15, 23], dtype=torch.bfloat16), 7, 0],
        [torch.randn([7, 15, 23], dtype=torch.float32), 5, -1],
        [torch.randn([7, 15, 23], dtype=torch.float16), 11, -1],
        [torch.randn([3, 5, 7, 11], dtype=torch.bfloat16), 1, 0],
        [torch.randn([3, 5, 7, 11], dtype=torch.float32), [2, 3], 1],
        [torch.randn([3, 5, 7, 11], dtype=torch.float16), [5, 6], -1],
        [torch.randn([1, 5, 111, 111], dtype=torch.bfloat16), 55, 2],
        [torch.randn([1, 5, 111, 111], dtype=torch.float32), [50, 61], -1],
        [torch.randn([3584, 18944], dtype=torch.float16), 4736, -1],
        [torch.randn([5120, 13824], dtype=torch.bfloat16), 3456, 1],
        [torch.randn([5120, 27648], dtype=torch.float32), [9216, 9216, 9216], -1],
        [torch.randn([6144, 20480], dtype=torch.float16), [4096, 8192, 8192], 1],
    ]
