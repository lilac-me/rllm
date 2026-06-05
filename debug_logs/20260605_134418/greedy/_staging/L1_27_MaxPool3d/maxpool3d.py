import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs 3D max pooling.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                dilation: int = 1, ceil_mode: bool = False,
                return_indices: bool = False):
        """
        Applies 3D max pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            dilation (int, optional): Spacing between kernel elements.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            return_indices (bool, optional): Return indices of max values.

        Returns:
            torch.Tensor or tuple: Pooled tensor (and indices if return_indices=True).
        """
        return torch.nn.functional.max_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            dilation=dilation, ceil_mode=ceil_mode,
            return_indices=return_indices
        )


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([1, 64, 16, 32, 32], dtype=torch.float32), 2, 2, 0, 1, False, False],
        [torch.randn([2, 128, 8, 64, 64], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([1, 256, 16, 56, 56], dtype=torch.float16), 3, 1, 1, 1, False, False],
        [torch.randn([4, 64, 8, 28, 28], dtype=torch.float16), 2, 2, 0, 1, False, False],
        [torch.randn([1, 32, 32, 112, 112], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([2, 512, 4, 14, 14], dtype=torch.float32), 3, 1, 1, 1, False, False],
        [torch.randn([1, 128, 16, 32, 32], dtype=torch.float16), 5, 1, 2, 1, False, False],
        [torch.randn([1, 64, 8, 16, 16], dtype=torch.float16), 2, 1, 0, 1, True, False],
        [torch.randn([1, 64, 16, 32, 32], dtype=torch.float32), 2, 2, 0, 1, False, False],
        [torch.randn([2, 256, 8, 28, 28], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([1, 128, 8, 64, 64], dtype=torch.bfloat16), 2, 2, 0, 1, False, False],
        [torch.randn([1, 64, 16, 56, 56], dtype=torch.bfloat16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 32, 24, 48, 48], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([4, 128, 4, 32, 32], dtype=torch.float16), 2, 2, 0, 1, False, False],
        [torch.randn([2, 64, 12, 24, 24], dtype=torch.float32), 3, 1, 1, 1, False, False],
        [torch.randn([1, 256, 8, 14, 14], dtype=torch.float16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 96, 8, 224, 224], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([2, 192, 4, 56, 56], dtype=torch.float16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 64, 8, 32, 32], dtype=torch.bfloat16), 2, 1, 0, 1, True, False],
        [torch.randn([1, 128, 6, 17, 17], dtype=torch.float32), 3, 2, 1, 1, True, False],
        [torch.randn([1, 64, 10, 33, 33], dtype=torch.float16), 2, 2, 0, 1, True, False],
        [torch.randn([2, 64, 5, 25, 25], dtype=torch.bfloat16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 32, 7, 21, 21], dtype=torch.float32), 5, 1, 2, 1, False, False],
        [torch.randn([1, 48, 9, 19, 19], dtype=torch.float16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 80, 11, 31, 31], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([1, 64, 8, 224, 224], dtype=torch.float16), 2, 2, 0, 1, False, False],
        [torch.randn([1, 128, 8, 14, 14], dtype=torch.bfloat16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 256, 4, 7, 7], dtype=torch.float32), 3, 1, 1, 1, False, False],
        [torch.randn([2, 96, 6, 28, 28], dtype=torch.float16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 160, 10, 18, 18], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([1, 32, 4, 112, 112], dtype=torch.float16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 64, 8, 56, 56], dtype=torch.bfloat16), 2, 2, 0, 1, False, False],
        [torch.randn([1, 128, 8, 16, 16], dtype=torch.float32), 2, 2, 0, 1, False, False],
        [torch.randn([4, 64, 4, 14, 14], dtype=torch.float16), 3, 1, 1, 1, False, False],
        [torch.randn([2, 32, 8, 64, 64], dtype=torch.float32), 2, 1, 0, 1, False, False],
        [torch.randn([1, 48, 6, 35, 35], dtype=torch.float16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 96, 8, 29, 29], dtype=torch.bfloat16), 3, 2, 1, 1, True, False],
        [torch.randn([1, 64, 5, 27, 27], dtype=torch.float32), 3, 1, 1, 1, False, False],
        [torch.randn([1, 128, 4, 11, 11], dtype=torch.float16), 5, 1, 2, 1, False, False],
        [torch.randn([2, 64, 6, 23, 23], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([1, 32, 8, 13, 13], dtype=torch.float16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 128, 8, 7, 7], dtype=torch.bfloat16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 256, 6, 9, 9], dtype=torch.float32), 3, 2, 1, 1, False, False],
        [torch.randn([1, 64, 10, 15, 15], dtype=torch.float16), 2, 2, 0, 1, False, False],
        [torch.randn([1, 48, 8, 12, 12], dtype=torch.float32), 3, 1, 1, 1, False, False],
        [torch.randn([2, 128, 5, 10, 10], dtype=torch.float16), 3, 2, 1, 1, False, False],
        [torch.randn([1, 96, 6, 11, 11], dtype=torch.bfloat16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 64, 4, 8, 8], dtype=torch.float32), 2, 2, 0, 1, False, False],
        [torch.randn([1, 32, 3, 5, 5], dtype=torch.float16), 3, 1, 1, 1, False, False],
        [torch.randn([1, 16, 2, 4, 4], dtype=torch.float32), 2, 2, 0, 1, False, False],
    ]
