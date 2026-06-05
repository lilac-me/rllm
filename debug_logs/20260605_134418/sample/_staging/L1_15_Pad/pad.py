import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs padding on a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, pad: tuple, mode: str = 'constant', value: float = None) -> torch.Tensor:
        """
        Pads tensor with specified padding mode.

        Args:
            x (torch.Tensor): Input tensor.
            pad (tuple): Padding sizes in the form (pad_left, pad_right, pad_top, pad_bottom, ...).
            mode (str, optional): Padding mode: 'constant', 'reflect', 'replicate', 'circular'.
            value (float, optional): Fill value for 'constant' padding.

        Returns:
            torch.Tensor: Padded tensor.
        """
        return torch.nn.functional.pad(x, pad, mode=mode, value=value)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128], dtype=torch.float32), [4, 4], 'constant', 0.0],
        [torch.randn([256], dtype=torch.float32), [8, 8], 'constant', 0.0],
        [torch.randn([512], dtype=torch.float16), [16, 16], 'constant', 0.0],
        [torch.randn([1024], dtype=torch.bfloat16), [32, 32], 'constant', 0.0],
        [torch.randn([128, 256], dtype=torch.float32), [4, 4, 2, 2], 'constant', 0.0],
        [torch.randn([256, 512], dtype=torch.float32), [8, 8, 4, 4], 'constant', 0.0],
        [torch.randn([512, 1024], dtype=torch.float16), [16, 16, 8, 8], 'constant', 0.0],
        [torch.randn([1024, 2048], dtype=torch.bfloat16), [32, 32, 16, 16], 'constant', 0.0],
        [torch.randn([64, 128, 256], dtype=torch.float32), [4, 4, 2, 2, 1, 1], 'constant', 0.0],
        [torch.randn([32, 64, 128], dtype=torch.float32), [2, 2, 1, 1, 1, 1], 'constant', 0.0],
        [torch.randn([16, 128, 256], dtype=torch.float16), [8, 8, 4, 4, 2, 2], 'constant', 0.0],
        [torch.randn([8, 256, 512], dtype=torch.bfloat16), [16, 16, 8, 8, 4, 4], 'constant', 0.0],
        [torch.randn([1, 64, 128, 128], dtype=torch.float32), [4, 4, 2, 2, 1, 1, 0, 0], 'constant', 0.0],
        [torch.randn([1, 128, 64, 64], dtype=torch.float32), [2, 2, 1, 1], 'reflect'],
        [torch.randn([1, 256, 32, 32], dtype=torch.float16), [2, 2, 2, 2], 'replicate'],
        [torch.randn([1, 512, 16, 16], dtype=torch.bfloat16), [2, 2, 1, 1], 'circular'],
        [torch.randn([1536], dtype=torch.float32), [64, 64], 'constant', 0.0],
        [torch.randn([4096], dtype=torch.float16), [128, 128], 'constant', 1.0],
        [torch.randn([8192], dtype=torch.bfloat16), [256, 256], 'constant', -1.0],
        [torch.randn([4096, 4096], dtype=torch.float32), [64, 64, 32, 32], 'constant', 0.0],
        [torch.randn([4096, 11008], dtype=torch.float16), [128, 128, 64, 64], 'constant', 0.0],
        [torch.randn([5120, 13824], dtype=torch.bfloat16), [256, 256, 128, 128], 'constant', 0.0],
        [torch.randn([3584, 18944], dtype=torch.float32), [64, 64, 32, 32], 'constant', 0.5],
        [torch.randn([5120, 27648], dtype=torch.float16), [128, 128, 64, 64], 'constant', 0.0],
        [torch.randn([1, 3, 224, 224], dtype=torch.float32), [3, 3, 3, 3], 'constant', 0.0],
        [torch.randn([1, 3, 224, 224], dtype=torch.float16), [1, 1, 1, 1], 'reflect'],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), [1, 1, 1, 1], 'replicate'],
        [torch.randn([1, 128, 28, 28], dtype=torch.float32), [2, 2, 2, 2], 'circular'],
        [torch.randn([1, 256, 14, 14], dtype=torch.float32), [1, 1, 1, 1], 'constant', 0.0],
        [torch.randn([1, 512, 7, 7], dtype=torch.float16), [1, 1, 1, 1], 'constant', 0.0],
        [torch.randn([100], dtype=torch.float32), [5, 5], 'constant', 0.0],
        [torch.randn([200], dtype=torch.float16), [3, 3], 'constant', 0.0],
        [torch.randn([34, 66], dtype=torch.float32), [2, 2, 1, 1], 'constant', 0.0],
        [torch.randn([17, 33], dtype=torch.float16), [1, 1, 1, 1], 'constant', 0.0],
        [torch.randn([65, 129], dtype=torch.bfloat16), [2, 2, 2, 2], 'constant', 0.0],
        [torch.randn([1, 16, 100, 100], dtype=torch.float32), [5, 5, 5, 5], 'constant', 0.0],
        [torch.randn([1, 32, 50, 50], dtype=torch.float16), [2, 2, 2, 2], 'reflect'],
        [torch.randn([1, 64, 25, 25], dtype=torch.bfloat16), [1, 1, 1, 1], 'circular'],
        [torch.randn([1, 3, 112, 112], dtype=torch.float32), [1, 0, 1, 0], 'constant', 0.0],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), [2, 0, 2, 0], 'replicate'],
        [torch.randn([128, 256], dtype=torch.float32), [0, 0, 4, 4], 'constant', 0.0],
        [torch.randn([256, 512], dtype=torch.float16), [8, 8, 0, 0], 'constant', 0.0],
        [torch.randn([32, 128, 128], dtype=torch.float32), [0, 4, 0, 4, 0, 2], 'constant', 0.0],
        [torch.randn([16, 64, 64], dtype=torch.float16), [2, 2, 0, 0, 1, 1], 'constant', 0.0],
        [torch.randn([1, 64, 64, 64], dtype=torch.float32), [1, 1, 1, 1, 1, 1, 0, 0], 'constant', 0.0],
        [torch.randn([1, 128, 32, 32], dtype=torch.float16), [2, 2, 2, 2], 'reflect'],
        [torch.randn([1, 256, 16, 16], dtype=torch.bfloat16), [1, 1, 1, 1], 'circular'],
        [torch.randn([2048, 2048], dtype=torch.float32), [32, 32, 16, 16], 'constant', 0.0],
        [torch.randn([3072, 3072], dtype=torch.float16), [64, 64, 32, 32], 'constant', 0.0],
        [torch.randn([6144, 6144], dtype=torch.float32), [128, 128, 64, 64], 'constant', 0.0],
        [torch.randn([8192, 8192], dtype=torch.float16), [256, 256, 128, 128], 'constant', 0.0],
    ]
