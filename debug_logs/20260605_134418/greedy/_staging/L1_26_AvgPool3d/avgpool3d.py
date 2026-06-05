import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs 3D average pooling.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                ceil_mode: bool = False, count_include_pad: bool = True,
                divisor_override=None) -> torch.Tensor:
        """
        Applies 3D average pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            count_include_pad (bool, optional): Include zero-padding in averaging.
            divisor_override (optional): If specified, will be used as divisor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return torch.nn.functional.avg_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.float32), [2, 2, 2], [2, 2, 2], 0],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.float32), [3, 3, 3], [1, 1, 1], 1],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.float32), [2, 2, 2], [2, 2, 2], 0],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.float32), [3, 3, 3], [2, 2, 2], 1],
        [torch.randn([2, 64, 4, 28, 28], dtype=torch.float32), 2, 2, 0],
        [torch.randn([2, 128, 4, 14, 14], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 1, 64, 64, 64], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 32, 32, 32, 32], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 64, 16, 16, 16], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 128, 8, 8, 8], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 3, 8, 112, 112], dtype=torch.float32), [2, 3, 3], [2, 2, 2], 0],
        [torch.randn([1, 64, 4, 56, 56], dtype=torch.float32), [3, 3, 3], [1, 2, 2], 1],
        [torch.randn([1, 3, 16, 224, 224], dtype=torch.float32), [2, 7, 7], [2, 7, 7], 0],
        [torch.randn([1, 64, 4, 28, 28], dtype=torch.float32), [3, 7, 7], [1, 1, 1], 1],
        [torch.randn([1, 3, 16, 128, 128], dtype=torch.float32), 3, 2, 1],
        [torch.randn([1, 64, 8, 32, 32], dtype=torch.float32), 3, 1, 1],
        [torch.randn([1, 128, 4, 16, 16], dtype=torch.float32), 3, 1, 1],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.float32), 2, 2, 0, True],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.float32), 3, 2, 1, True],
        [torch.randn([1, 32, 32, 32, 32], dtype=torch.float32), 2, 2, 0, True],
        [torch.randn([1, 3, 8, 64, 64], dtype=torch.float32), 2, 2, 1, False],
        [torch.randn([1, 64, 4, 32, 32], dtype=torch.float32), 3, 1, 1, False],
        [torch.randn([1, 1, 32, 32, 32], dtype=torch.float32), 2, 2, 1, False],
        [torch.randn([1, 64, 8, 28, 28], dtype=torch.float32), 3, 2, 1, True, False],
        [torch.randn([1, 3, 16, 112, 112], dtype=torch.float32), 2, 2, 0, True, False],
        [torch.randn([4, 64, 8, 28, 28], dtype=torch.float32), 2, 2, 0],
        [torch.randn([8, 64, 4, 14, 14], dtype=torch.float32), 2, 2, 0],
        [torch.randn([16, 128, 4, 7, 7], dtype=torch.float32), 2, 2, 0],
        [torch.randn([4, 3, 8, 56, 56], dtype=torch.float32), [2, 3, 3], [2, 2, 2], 1],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.float32), [2, 2, 2], [2, 2, 2], [1, 1, 1]],
        [torch.randn([1, 64, 8, 32, 32], dtype=torch.float32), [3, 3, 3], [1, 2, 2], [1, 1, 1]],
        [torch.randn([1, 64, 8, 32, 32], dtype=torch.float32), [2, 2, 2], [2, 2, 2], [0, 0, 0]],
        [torch.randn([1, 3, 16, 32, 32], dtype=torch.float32), 2, 2, 0, 8],
        [torch.randn([1, 64, 4, 16, 16], dtype=torch.float32), 3, 1, 1, 27],
        [torch.randn([1, 32, 16, 16, 16], dtype=torch.float32), 2, 2, 0, 8],
        [torch.randn([1, 3, 7, 55, 55], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 64, 5, 27, 27], dtype=torch.float32), 3, 2, 1],
        [torch.randn([1, 32, 9, 33, 33], dtype=torch.float32), 2, 2, 0, True],
        [torch.randn([1, 128, 5, 13, 13], dtype=torch.float32), 3, 1, 1],
        [torch.randn([1, 3, 11, 111, 111], dtype=torch.float32), 2, 2, 0],
        [torch.randn([1, 64, 7, 57, 57], dtype=torch.float32), 3, 2, 1, True],
        [torch.randn([1, 32, 15, 29, 29], dtype=torch.float32), 2, 2, 0, False],
        [torch.randn([1, 3, 16, 32, 32], dtype=torch.float32), [3, 5, 5], [1, 1, 1], [1, 2, 2]],
        [torch.randn([1, 64, 8, 16, 16], dtype=torch.float32), [5, 3, 3], [1, 1, 1], [2, 1, 1]],
        [torch.randn([1, 128, 4, 8, 8], dtype=torch.float32), [2, 3, 3], [2, 1, 1], [0, 1, 1]],
        [torch.randn([1, 256, 2, 4, 4], dtype=torch.float32), 2, 1, 0],
        [torch.randn([1, 512, 2, 2, 2], dtype=torch.float32), 2, 1, 0],
        [torch.randn([1, 3, 8, 16, 16], dtype=torch.float32), 4, 2, 1],
        [torch.randn([1, 64, 8, 8, 8], dtype=torch.float32), 4, 2, 1],
        [torch.randn([1, 128, 6, 8, 8], dtype=torch.float32), 4, 1, 1],
        [torch.randn([1, 3, 16, 48, 48], dtype=torch.float32), 2, 2, 0, True, False],
        [torch.randn([1, 64, 8, 24, 24], dtype=torch.float32), 3, 2, 1, True, False],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.float16), [2, 2, 2], [2, 2, 2], 0],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.float16), [3, 3, 3], [1, 1, 1], 1],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.float16), [2, 2, 2], [2, 2, 2], 0],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.float16), [3, 3, 3], [2, 2, 2], 1],
        [torch.randn([2, 64, 4, 28, 28], dtype=torch.float16), 2, 2, 0],
        [torch.randn([2, 128, 4, 14, 14], dtype=torch.float16), 2, 2, 0],
        [torch.randn([1, 1, 64, 64, 64], dtype=torch.float16), 2, 2, 0],
        [torch.randn([1, 32, 32, 32, 32], dtype=torch.float16), 2, 2, 0],
        [torch.randn([1, 64, 16, 16, 16], dtype=torch.float16), 2, 2, 0],
        [torch.randn([1, 128, 8, 8, 8], dtype=torch.float16), 2, 2, 0],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.bfloat16), [2, 2, 2], [2, 2, 2], 0],
        [torch.randn([1, 3, 16, 64, 64], dtype=torch.bfloat16), [3, 3, 3], [1, 1, 1], 1],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.bfloat16), [2, 2, 2], [2, 2, 2], 0],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.bfloat16), [3, 3, 3], [2, 2, 2], 1],
        [torch.randn([2, 64, 4, 28, 28], dtype=torch.bfloat16), 2, 2, 0],
        [torch.randn([2, 128, 4, 14, 14], dtype=torch.bfloat16), 2, 2, 0],
        [torch.randn([1, 1, 64, 64, 64], dtype=torch.bfloat16), 2, 2, 0],
        [torch.randn([1, 32, 32, 32, 32], dtype=torch.bfloat16), 2, 2, 0],
        [torch.randn([1, 64, 16, 16, 16], dtype=torch.bfloat16), 2, 2, 0],
        [torch.randn([1, 128, 8, 8, 8], dtype=torch.bfloat16), 2, 2, 0],
    ]
