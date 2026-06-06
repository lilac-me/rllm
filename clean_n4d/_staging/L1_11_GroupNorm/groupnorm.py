import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Group Normalization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, num_groups: int, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Group Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, *].
            num_groups (int): Number of groups to separate channels into.
            weight (torch.Tensor, optional): Weight tensor of shape [C].
            bias (torch.Tensor, optional): Bias tensor of shape [C].

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.group_norm(x, num_groups, weight=weight, bias=bias)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([1, 64, 56, 56], dtype=torch.float32), 8, torch.randn([64], dtype=torch.float32), torch.randn([64], dtype=torch.float32)],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 112, 112], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 224, 224], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 128, 56, 56], dtype=torch.float16), 16, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 128, 112, 112], dtype=torch.float16), 16, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 128, 224, 224], dtype=torch.float16), 16, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 256, 56, 56], dtype=torch.float16), 32, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 256, 112, 112], dtype=torch.float16), 32, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 256, 224, 224], dtype=torch.float16), 32, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 512, 28, 28], dtype=torch.float16), 32, torch.randn([512], dtype=torch.float16), torch.randn([512], dtype=torch.float16)],
        [torch.randn([1, 512, 56, 56], dtype=torch.float16), 32, torch.randn([512], dtype=torch.float16), torch.randn([512], dtype=torch.float16)],
        [torch.randn([1, 512, 112, 112], dtype=torch.float16), 32, torch.randn([512], dtype=torch.float16), torch.randn([512], dtype=torch.float16)],
        [torch.randn([1, 768, 14, 14], dtype=torch.float16), 32, torch.randn([768], dtype=torch.float16), torch.randn([768], dtype=torch.float16)],
        [torch.randn([1, 768, 28, 28], dtype=torch.float16), 32, torch.randn([768], dtype=torch.float16), torch.randn([768], dtype=torch.float16)],
        [torch.randn([1, 768, 56, 56], dtype=torch.float16), 32, torch.randn([768], dtype=torch.float16), torch.randn([768], dtype=torch.float16)],
        [torch.randn([1, 1024, 14, 14], dtype=torch.float16), 32, torch.randn([1024], dtype=torch.float16), torch.randn([1024], dtype=torch.float16)],
        [torch.randn([1, 1024, 28, 28], dtype=torch.float16), 32, torch.randn([1024], dtype=torch.float16), torch.randn([1024], dtype=torch.float16)],
        [torch.randn([1, 1024, 56, 56], dtype=torch.float16), 32, torch.randn([1024], dtype=torch.float16), torch.randn([1024], dtype=torch.float16)],
        [torch.randn([1, 64, 56, 56], dtype=torch.bfloat16), 8, torch.randn([64], dtype=torch.bfloat16), torch.randn([64], dtype=torch.bfloat16)],
        [torch.randn([1, 256, 56, 56], dtype=torch.bfloat16), 32, torch.randn([256], dtype=torch.bfloat16), torch.randn([256], dtype=torch.bfloat16)],
        [torch.randn([1, 512, 56, 56], dtype=torch.bfloat16), 32, torch.randn([512], dtype=torch.bfloat16), torch.randn([512], dtype=torch.bfloat16)],
        [torch.randn([1, 1024, 56, 56], dtype=torch.bfloat16), 32, torch.randn([1024], dtype=torch.bfloat16), torch.randn([1024], dtype=torch.bfloat16)],
        [torch.randn([1, 2048, 14, 14], dtype=torch.float16), 32, torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 2048, 28, 28], dtype=torch.float16), 32, torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 2048, 56, 56], dtype=torch.float16), 32, torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 4096, 7, 7], dtype=torch.float16), 32, torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 4096, 14, 14], dtype=torch.float16), 32, torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 4096, 28, 28], dtype=torch.float16), 32, torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([4, 256, 56, 56], dtype=torch.float16), 32, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([8, 512, 28, 28], dtype=torch.float16), 32, torch.randn([512], dtype=torch.float16), torch.randn([512], dtype=torch.float16)],
        [torch.randn([16, 1024, 14, 14], dtype=torch.float16), 32, torch.randn([1024], dtype=torch.float16), torch.randn([1024], dtype=torch.float16)],
        [torch.randn([1, 64, 128, 128], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 128, 64, 64], dtype=torch.float16), 8, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 256, 32, 32], dtype=torch.float16), 8, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 512, 16, 16], dtype=torch.float16), 8, torch.randn([512], dtype=torch.float16), torch.randn([512], dtype=torch.float16)],
        [torch.randn([1, 1024, 8, 8], dtype=torch.float16), 8, torch.randn([1024], dtype=torch.float16), torch.randn([1024], dtype=torch.float16)],
        [torch.randn([1, 2048, 4, 4], dtype=torch.float16), 8, torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 4096, 2, 2], dtype=torch.float16), 8, torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 4, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 16, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 32, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 128, 56, 56], dtype=torch.float16), 8, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 128, 56, 56], dtype=torch.float16), 32, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 256, 56, 56], dtype=torch.float16), 8, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 256, 56, 56], dtype=torch.float16), 64, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 64, 50, 50], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 100, 100], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 128, 50, 50], dtype=torch.float16), 16, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 256, 50, 50], dtype=torch.float16), 32, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 56, 56], dtype=torch.float16), 8, torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 128, 128], dtype=torch.float16), 1, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 64, 128, 128], dtype=torch.float16), 64, torch.randn([64], dtype=torch.float16), torch.randn([64], dtype=torch.float16)],
        [torch.randn([1, 128, 128, 128], dtype=torch.float16), 1, torch.randn([128], dtype=torch.float16), torch.randn([128], dtype=torch.float16)],
        [torch.randn([1, 256, 64, 64], dtype=torch.float16), 1, torch.randn([256], dtype=torch.float16), torch.randn([256], dtype=torch.float16)],
        [torch.randn([1, 512, 32, 32], dtype=torch.float16), 1, torch.randn([512], dtype=torch.float16), torch.randn([512], dtype=torch.float16)],
        [torch.randn([1, 1024, 16, 16], dtype=torch.float16), 1, torch.randn([1024], dtype=torch.float16), torch.randn([1024], dtype=torch.float16)],
        [torch.randn([1, 1536, 8, 8], dtype=torch.float16), 1, torch.randn([1536], dtype=torch.float16), torch.randn([1536], dtype=torch.float16)],
        [torch.randn([1, 2048, 8, 8], dtype=torch.float16), 1, torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 2560, 4, 4], dtype=torch.float16), 1, torch.randn([2560], dtype=torch.float16), torch.randn([2560], dtype=torch.float16)],
        [torch.randn([1, 3072, 4, 4], dtype=torch.float16), 1, torch.randn([3072], dtype=torch.float16), torch.randn([3072], dtype=torch.float16)],
        [torch.randn([1, 3584, 4, 4], dtype=torch.float16), 1, torch.randn([3584], dtype=torch.float16), torch.randn([3584], dtype=torch.float16)],
        [torch.randn([1, 4096, 4, 4], dtype=torch.float16), 1, torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 5120, 4, 4], dtype=torch.float16), 1, torch.randn([5120], dtype=torch.float16), torch.randn([5120], dtype=torch.float16)],
        [torch.randn([1, 6144, 4, 4], dtype=torch.float16), 1, torch.randn([6144], dtype=torch.float16), torch.randn([6144], dtype=torch.float16)],
        [torch.randn([1, 7168, 4, 4], dtype=torch.float16), 1, torch.randn([7168], dtype=torch.float16), torch.randn([7168], dtype=torch.float16)],
        [torch.randn([1, 8192, 4, 4], dtype=torch.float16), 1, torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
        [torch.randn([1, 32, 56, 56], dtype=torch.float16), 8, torch.randn([32], dtype=torch.float16), torch.randn([32], dtype=torch.float16)],
        [torch.randn([1, 32, 128, 128], dtype=torch.float16), 8, torch.randn([32], dtype=torch.float16), torch.randn([32], dtype=torch.float16)],
        [torch.randn([1, 48, 56, 56], dtype=torch.float16), 8, torch.randn([48], dtype=torch.float16), torch.randn([48], dtype=torch.float16)],
        [torch.randn([1, 96, 56, 56], dtype=torch.float16), 8, torch.randn([96], dtype=torch.float16), torch.randn([96], dtype=torch.float16)],
        [torch.randn([1, 192, 56, 56], dtype=torch.float16), 8, torch.randn([192], dtype=torch.float16), torch.randn([192], dtype=torch.float16)],
    ]
