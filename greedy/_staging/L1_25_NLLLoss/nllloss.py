import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the negative log likelihood loss.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor = None, ignore_index: int = -100,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the negative log likelihood loss.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C) or (N, C, d1, d2, ...).
            target (torch.Tensor): Target tensor of shape (N,) or (N, d1, d2, ...).
            weight (torch.Tensor, optional): Manual rescaling weight given to each class.
            ignore_index (int, optional): Target value that is ignored and does not contribute to gradient.
            reduction (str, optional): Reduction method ('none', 'mean', 'sum').

        Returns:
            torch.Tensor: NLL loss value.
        """
        return torch.nn.functional.nll_loss(input, target, weight=weight, ignore_index=ignore_index, reduction=reduction)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([32, 1000], dtype=torch.float32), torch.randint(0, 100, [32], dtype=torch.int64), torch.randn([1000], dtype=torch.float32), -100, 'mean'],
        [torch.randn([64, 512], dtype=torch.float32), torch.randint(0, 100, [64], dtype=torch.int64), -100, 'mean'],
        [torch.randn([128, 2048], dtype=torch.float16), torch.randint(0, 100, [128], dtype=torch.int64), -100, 'sum'],
        [torch.randn([256, 4096], dtype=torch.float16), torch.randint(0, 100, [256], dtype=torch.int64), torch.randn([4096], dtype=torch.float16), -100, 'mean'],
        [torch.randn([16, 8192], dtype=torch.bfloat16), torch.randint(0, 100, [16], dtype=torch.int64), -100, 'mean'],
        [torch.randn([8, 32000], dtype=torch.float32), torch.randint(0, 100, [8], dtype=torch.int64), -100, 'mean'],
        [torch.randn([4, 128000], dtype=torch.float16), torch.randint(0, 100, [4], dtype=torch.int64), -100, 'mean'],
        [torch.randn([2, 256000], dtype=torch.bfloat16), torch.randint(0, 100, [2], dtype=torch.int64), -100, 'mean'],
        [torch.randn([32, 1000], dtype=torch.float32), torch.randint(0, 100, [32], dtype=torch.int64), -100, 'none'],
        [torch.randn([64, 512], dtype=torch.float16), torch.randint(0, 100, [64], dtype=torch.int64), torch.randn([512], dtype=torch.float16), -100, 'none'],
        [torch.randn([32, 512], dtype=torch.float32), torch.randint(0, 100, [32], dtype=torch.int64), torch.randn([512], dtype=torch.float32), 0, 'mean'],
        [torch.randn([64, 1024], dtype=torch.float16), torch.randint(0, 100, [64], dtype=torch.int64), torch.randn([1024], dtype=torch.float16), -1, 'sum'],
        [torch.randn([2, 1024, 256], dtype=torch.float32), torch.randint(0, 100, [2, 256], dtype=torch.int64), -100, 'mean'],
        [torch.randn([4, 4096, 512], dtype=torch.float16), torch.randint(0, 100, [4, 512], dtype=torch.int64), -100, 'mean'],
        [torch.randn([8, 8192, 1024], dtype=torch.bfloat16), torch.randint(0, 100, [8, 1024], dtype=torch.int64), -100, 'sum'],
        [torch.randn([2, 32000, 2048], dtype=torch.float16), torch.randint(0, 100, [2, 2048], dtype=torch.int64), -100, 'mean'],
        [torch.randn([1, 128000, 4096], dtype=torch.bfloat16), torch.randint(0, 100, [1, 4096], dtype=torch.int64), -100, 'mean'],
        [torch.randn([2, 1024, 256], dtype=torch.float32), torch.randint(0, 100, [2, 256], dtype=torch.int64), torch.randn([1024], dtype=torch.float32), -100, 'none'],
        [torch.randn([4, 512, 128], dtype=torch.float16), torch.randint(0, 100, [4, 128], dtype=torch.int64), torch.randn([512], dtype=torch.float16), 0, 'mean'],
        [torch.randn([1, 10, 16, 64, 64], dtype=torch.float32), torch.randint(0, 100, [1, 16, 64, 64], dtype=torch.int64), -100, 'mean'],
        [torch.randn([2, 21, 32, 128, 128], dtype=torch.float16), torch.randint(0, 100, [2, 32, 128, 128], dtype=torch.int64), torch.randn([21], dtype=torch.float16), -100, 'mean'],
        [torch.randn([50, 200], dtype=torch.float32), torch.randint(0, 100, [50], dtype=torch.int64), -100, 'mean'],
        [torch.randn([100, 300], dtype=torch.float16), torch.randint(0, 100, [100], dtype=torch.int64), torch.randn([300], dtype=torch.float16), -100, 'sum'],
        [torch.randn([33, 127], dtype=torch.float32), torch.randint(0, 100, [33], dtype=torch.int64), -100, 'mean'],
        [torch.randn([67, 255], dtype=torch.float16), torch.randint(0, 100, [67], dtype=torch.int64), -100, 'mean'],
        [torch.randn([17, 511, 63], dtype=torch.bfloat16), torch.randint(0, 100, [17, 63], dtype=torch.int64), torch.randn([511], dtype=torch.bfloat16), -100, 'mean'],
        [torch.randn([35, 1023, 129], dtype=torch.float32), torch.randint(0, 100, [35, 129], dtype=torch.int64), -1, 'sum'],
        [torch.randn([1, 4096, 128], dtype=torch.float16), torch.randint(0, 100, [1, 128], dtype=torch.int64), -100, 'mean'],
        [torch.randn([1, 8192, 256], dtype=torch.bfloat16), torch.randint(0, 100, [1, 256], dtype=torch.int64), -100, 'mean'],
        [torch.randn([1, 1024, 512], dtype=torch.float32), torch.randint(0, 100, [1, 512], dtype=torch.int64), torch.randn([1024], dtype=torch.float32), -100, 'mean'],
        [torch.randn([1, 10, 1, 128, 128], dtype=torch.float16), torch.randint(0, 100, [1, 1, 128, 128], dtype=torch.int64), 255, 'mean'],
        [torch.randn([128, 3584], dtype=torch.float32), torch.randint(0, 100, [128], dtype=torch.int64), -100, 'mean'],
        [torch.randn([256, 5120], dtype=torch.float16), torch.randint(0, 100, [256], dtype=torch.int64), -100, 'mean'],
        [torch.randn([64, 6144], dtype=torch.bfloat16), torch.randint(0, 100, [64], dtype=torch.int64), -100, 'sum'],
        [torch.randn([32, 7168], dtype=torch.float16), torch.randint(0, 100, [32], dtype=torch.int64), -100, 'mean'],
        [torch.randn([4, 15258, 2048], dtype=torch.float32), torch.randint(0, 100, [4, 2048], dtype=torch.int64), -100, 'mean'],
        [torch.randn([2, 32000, 4096], dtype=torch.float16), torch.randint(0, 100, [2, 4096], dtype=torch.int64), torch.randn([32000], dtype=torch.float16), -100, 'mean'],
        [torch.randn([1, 128256, 8192], dtype=torch.bfloat16), torch.randint(0, 100, [1, 8192], dtype=torch.int64), -100, 'mean'],
        [torch.randn([2, 250000, 1024], dtype=torch.float16), torch.randint(0, 100, [2, 1024], dtype=torch.int64), -100, 'mean'],
        [torch.randn([1, 200000, 512], dtype=torch.float32), torch.randint(0, 100, [1, 512], dtype=torch.int64), -100, 'mean'],
        [torch.randn([1, 511, 33], dtype=torch.float16), torch.randint(0, 100, [1, 33], dtype=torch.int64), -100, 'mean'],
        [torch.randn([1, 1023, 65], dtype=torch.bfloat16), torch.randint(0, 100, [1, 65], dtype=torch.int64), -100, 'sum'],
        [torch.randn([1, 2047, 127], dtype=torch.float32), torch.randint(0, 100, [1, 127], dtype=torch.int64), torch.randn([2047], dtype=torch.float32), -100, 'none'],
        [torch.randn([1, 4095, 255], dtype=torch.float16), torch.randint(0, 100, [1, 255], dtype=torch.int64), -1, 'mean'],
        [torch.randn([1, 8191, 511], dtype=torch.float32), torch.randint(0, 100, [1, 511], dtype=torch.int64), -100, 'mean'],
        [torch.randn([512, 1536], dtype=torch.float16), torch.randint(0, 100, [512], dtype=torch.int64), -100, 'mean'],
        [torch.randn([256, 2560], dtype=torch.bfloat16), torch.randint(0, 100, [256], dtype=torch.int64), torch.randn([2560], dtype=torch.bfloat16), -100, 'sum'],
        [torch.randn([128, 2880], dtype=torch.float32), torch.randint(0, 100, [128], dtype=torch.int64), -100, 'mean'],
        [torch.randn([64, 3072], dtype=torch.float16), torch.randint(0, 100, [64], dtype=torch.int64), -100, 'mean'],
        [torch.randn([128, 3840], dtype=torch.float16), torch.randint(0, 100, [128], dtype=torch.int64), torch.randn([3840], dtype=torch.float16), -100, 'none'],
    ]
