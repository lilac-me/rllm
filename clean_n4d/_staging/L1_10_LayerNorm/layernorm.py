import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Layer Normalization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, normalized_shape: list, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Layer Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [*, normalized_shape[0], ...].
            normalized_shape (list): Shape over which to normalize.
            weight (torch.Tensor, optional): Weight tensor of shape normalized_shape.
            bias (torch.Tensor, optional): Bias tensor of shape normalized_shape.

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([1, 128, 4096], dtype=torch.float32), [4096], torch.randn([4096], dtype=torch.float32), torch.randn([4096], dtype=torch.float32)],
        [torch.randn([1, 256, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 512, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 1024, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 2048, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 4096, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 128, 4096], dtype=torch.bfloat16), [4096], torch.randn([4096], dtype=torch.bfloat16), torch.randn([4096], dtype=torch.bfloat16)],
        [torch.randn([1, 2048, 4096], dtype=torch.bfloat16), [4096], torch.randn([4096], dtype=torch.bfloat16), torch.randn([4096], dtype=torch.bfloat16)],
        [torch.randn([1, 128, 3584], dtype=torch.float16), [3584], torch.randn([3584], dtype=torch.float16), torch.randn([3584], dtype=torch.float16)],
        [torch.randn([1, 512, 3584], dtype=torch.float16), [3584], torch.randn([3584], dtype=torch.float16), torch.randn([3584], dtype=torch.float16)],
        [torch.randn([1, 2048, 3584], dtype=torch.float16), [3584], torch.randn([3584], dtype=torch.float16), torch.randn([3584], dtype=torch.float16)],
        [torch.randn([1, 1024, 3584], dtype=torch.bfloat16), [3584], torch.randn([3584], dtype=torch.bfloat16), torch.randn([3584], dtype=torch.bfloat16)],
        [torch.randn([1, 128, 5120], dtype=torch.float16), [5120], torch.randn([5120], dtype=torch.float16), torch.randn([5120], dtype=torch.float16)],
        [torch.randn([1, 512, 5120], dtype=torch.float16), [5120], torch.randn([5120], dtype=torch.float16), torch.randn([5120], dtype=torch.float16)],
        [torch.randn([1, 2048, 5120], dtype=torch.float16), [5120], torch.randn([5120], dtype=torch.float16), torch.randn([5120], dtype=torch.float16)],
        [torch.randn([1, 1024, 5120], dtype=torch.bfloat16), [5120], torch.randn([5120], dtype=torch.bfloat16), torch.randn([5120], dtype=torch.bfloat16)],
        [torch.randn([1, 128, 6144], dtype=torch.float16), [6144], torch.randn([6144], dtype=torch.float16), torch.randn([6144], dtype=torch.float16)],
        [torch.randn([1, 512, 6144], dtype=torch.float16), [6144], torch.randn([6144], dtype=torch.float16), torch.randn([6144], dtype=torch.float16)],
        [torch.randn([1, 2048, 6144], dtype=torch.float16), [6144], torch.randn([6144], dtype=torch.float16), torch.randn([6144], dtype=torch.float16)],
        [torch.randn([1, 1024, 6144], dtype=torch.bfloat16), [6144], torch.randn([6144], dtype=torch.bfloat16), torch.randn([6144], dtype=torch.bfloat16)],
        [torch.randn([1, 128, 8192], dtype=torch.float16), [8192], torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
        [torch.randn([1, 512, 8192], dtype=torch.float16), [8192], torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
        [torch.randn([1, 2048, 8192], dtype=torch.float16), [8192], torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
        [torch.randn([1, 1024, 8192], dtype=torch.bfloat16), [8192], torch.randn([8192], dtype=torch.bfloat16), torch.randn([8192], dtype=torch.bfloat16)],
        [torch.randn([1, 128, 3072], dtype=torch.float16), [3072], torch.randn([3072], dtype=torch.float16), torch.randn([3072], dtype=torch.float16)],
        [torch.randn([1, 512, 3072], dtype=torch.float16), [3072], torch.randn([3072], dtype=torch.float16), torch.randn([3072], dtype=torch.float16)],
        [torch.randn([1, 2048, 3072], dtype=torch.float16), [3072], torch.randn([3072], dtype=torch.float16), torch.randn([3072], dtype=torch.float16)],
        [torch.randn([1, 128, 2048], dtype=torch.float16), [2048], torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 512, 2048], dtype=torch.float16), [2048], torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 2048, 2048], dtype=torch.float16), [2048], torch.randn([2048], dtype=torch.float16), torch.randn([2048], dtype=torch.float16)],
        [torch.randn([1, 128, 1536], dtype=torch.float16), [1536], torch.randn([1536], dtype=torch.float16), torch.randn([1536], dtype=torch.float16)],
        [torch.randn([1, 512, 1536], dtype=torch.float16), [1536], torch.randn([1536], dtype=torch.float16), torch.randn([1536], dtype=torch.float16)],
        [torch.randn([1, 2048, 1536], dtype=torch.float16), [1536], torch.randn([1536], dtype=torch.float16), torch.randn([1536], dtype=torch.float16)],
        [torch.randn([1, 128, 2560], dtype=torch.float16), [2560], torch.randn([2560], dtype=torch.float16), torch.randn([2560], dtype=torch.float16)],
        [torch.randn([1, 512, 2560], dtype=torch.float16), [2560], torch.randn([2560], dtype=torch.float16), torch.randn([2560], dtype=torch.float16)],
        [torch.randn([1, 2048, 2560], dtype=torch.float16), [2560], torch.randn([2560], dtype=torch.float16), torch.randn([2560], dtype=torch.float16)],
        [torch.randn([1, 128, 2880], dtype=torch.float16), [2880], torch.randn([2880], dtype=torch.float16), torch.randn([2880], dtype=torch.float16)],
        [torch.randn([1, 512, 2880], dtype=torch.float16), [2880], torch.randn([2880], dtype=torch.float16), torch.randn([2880], dtype=torch.float16)],
        [torch.randn([1, 2048, 2880], dtype=torch.float16), [2880], torch.randn([2880], dtype=torch.float16), torch.randn([2880], dtype=torch.float16)],
        [torch.randn([1, 128, 3840], dtype=torch.float16), [3840], torch.randn([3840], dtype=torch.float16), torch.randn([3840], dtype=torch.float16)],
        [torch.randn([1, 512, 3840], dtype=torch.float16), [3840], torch.randn([3840], dtype=torch.float16), torch.randn([3840], dtype=torch.float16)],
        [torch.randn([1, 2048, 3840], dtype=torch.float16), [3840], torch.randn([3840], dtype=torch.float16), torch.randn([3840], dtype=torch.float16)],
        [torch.randn([1, 128, 7168], dtype=torch.float16), [7168], torch.randn([7168], dtype=torch.float16), torch.randn([7168], dtype=torch.float16)],
        [torch.randn([1, 512, 7168], dtype=torch.float16), [7168], torch.randn([7168], dtype=torch.float16), torch.randn([7168], dtype=torch.float16)],
        [torch.randn([1, 2048, 7168], dtype=torch.float16), [7168], torch.randn([7168], dtype=torch.float16), torch.randn([7168], dtype=torch.float16)],
        [torch.randn([4, 128, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([8, 512, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([16, 1024, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([4, 128, 8192], dtype=torch.float16), [8192], torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
        [torch.randn([8, 512, 8192], dtype=torch.float16), [8192], torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
        [torch.randn([1, 100, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 200, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 500, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 1000, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 128, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 128, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 128, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 128, 4096, 64], dtype=torch.float16), [4096, 64], torch.randn([4096, 64], dtype=torch.float16), torch.randn([4096, 64], dtype=torch.float16)],
        [torch.randn([1, 64, 64, 4096], dtype=torch.float16), [4096], torch.randn([4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([1, 64, 64, 8192], dtype=torch.float16), [8192], torch.randn([8192], dtype=torch.float16), torch.randn([8192], dtype=torch.float16)],
    ]
