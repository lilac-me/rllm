import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs the backward pass for embedding with dense gradients.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, indices: torch.Tensor,
                num_weights: int, padding_idx: int = -1, scale_grad_by_freq: bool = False) -> torch.Tensor:
        """
        Computes the gradient for embedding layer with dense backward.

        Args:
            grad_output (torch.Tensor): Gradient of the output.
            indices (torch.Tensor): The indices tensor from forward pass.
            num_weights (int): Number of rows in the embedding weight matrix.
            padding_idx (int, optional): Index of padding token to zero out gradient.
            scale_grad_by_freq (bool, optional): Whether to scale gradients by frequency.

        Returns:
            torch.Tensor: Gradient tensor for embedding weights.
        """
        return torch.ops.aten.embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128, 768], dtype=torch.float32), torch.randint(0, 100, [128], dtype=torch.int64), 32000, -1, False],
        [torch.randn([256, 1024], dtype=torch.float32), torch.randint(0, 100, [256], dtype=torch.int64), 50257, -1, False],
        [torch.randn([512, 4096], dtype=torch.float16), torch.randint(0, 100, [512], dtype=torch.int64), 32000, 0, False],
        [torch.randn([1024, 4096], dtype=torch.float16), torch.randint(0, 100, [1024], dtype=torch.int64), 64000, -1, True],
        [torch.randn([2048, 3584], dtype=torch.bfloat16), torch.randint(0, 100, [2048], dtype=torch.int64), 152000, -1, False],
        [torch.randn([64, 512], dtype=torch.float32), torch.randint(0, 100, [64], dtype=torch.int64), 10000, 0, True],
        [torch.randn([128, 1280], dtype=torch.float16), torch.randint(0, 100, [128], dtype=torch.int64), 50257, -1, False],
        [torch.randn([256, 8192], dtype=torch.bfloat16), torch.randint(0, 100, [256], dtype=torch.int64), 128256, -1, False],
        [torch.randn([32, 256], dtype=torch.float32), torch.randint(0, 100, [32], dtype=torch.int64), 8000, -1, True],
        [torch.randn([512, 2048], dtype=torch.float16), torch.randint(0, 100, [512], dtype=torch.int64), 151936, 151643, False],
        [torch.randn([16, 64, 768], dtype=torch.float32), torch.randint(0, 100, [16, 64], dtype=torch.int64), 32000, -1, False],
        [torch.randn([8, 128, 4096], dtype=torch.float16), torch.randint(0, 100, [8, 128], dtype=torch.int64), 64000, -1, True],
        [torch.randn([4, 256, 5120], dtype=torch.bfloat16), torch.randint(0, 100, [4, 256], dtype=torch.int64), 100000, -1, False],
        [torch.randn([2, 512, 4096], dtype=torch.float16), torch.randint(0, 100, [2, 512], dtype=torch.int64), 128256, -1, False],
        [torch.randn([1, 1024, 3072], dtype=torch.float32), torch.randint(0, 100, [1, 1024], dtype=torch.int64), 151936, 151643, True],
        [torch.randn([2, 4, 256, 4096], dtype=torch.float16), torch.randint(0, 100, [2, 4, 256], dtype=torch.int64), 64000, -1, False],
        [torch.randn([1, 8, 512, 3584], dtype=torch.bfloat16), torch.randint(0, 100, [1, 8, 512], dtype=torch.int64), 152000, -1, True],
        [torch.randn([100, 768], dtype=torch.float32), torch.randint(0, 100, [100], dtype=torch.int64), 32000, -1, False],
        [torch.randn([34, 1024], dtype=torch.float16), torch.randint(0, 100, [34], dtype=torch.int64), 50257, -1, True],
        [torch.randn([17, 66, 512], dtype=torch.bfloat16), torch.randint(0, 100, [17, 66], dtype=torch.int64), 8000, 0, False],
        [torch.randn([50, 100, 384], dtype=torch.float32), torch.randint(0, 100, [50, 100], dtype=torch.int64), 30000, 1, True],
        [torch.randn([200, 512], dtype=torch.float16), torch.randint(0, 100, [200], dtype=torch.int64), 100000, -1, False],
        [torch.randn([33, 128, 2048], dtype=torch.bfloat16), torch.randint(0, 100, [33, 128], dtype=torch.int64), 64000, -1, True],
        [torch.randn([3, 150, 1024], dtype=torch.float32), torch.randint(0, 100, [3, 150], dtype=torch.int64), 128000, -1, False],
        [torch.randn([7, 200, 4096], dtype=torch.float16), torch.randint(0, 100, [7, 200], dtype=torch.int64), 151936, 151643, True],
        [torch.randn([1, 16, 64, 768], dtype=torch.float32), torch.randint(0, 100, [1, 16, 64], dtype=torch.int64), 32000, 0, False],
        [torch.randn([2, 3, 100, 2560], dtype=torch.bfloat16), torch.randint(0, 100, [2, 3, 100], dtype=torch.int64), 50257, -1, True],
        [torch.randn([4, 5, 80, 3072], dtype=torch.float16), torch.randint(0, 100, [4, 5, 80], dtype=torch.int64), 100000, -1, False],
        [torch.randn([1, 10, 120, 4096], dtype=torch.float32), torch.randint(0, 100, [1, 10, 120], dtype=torch.int64), 128256, -1, True],
        [torch.randn([2, 7, 90, 5120], dtype=torch.bfloat16), torch.randint(0, 100, [2, 7, 90], dtype=torch.int64), 152000, -1, False],
    ]
