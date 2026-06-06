import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([1024, 1024], dtype=torch.float32)],
        [torch.randn([512, 768], dtype=torch.float32)],
        [torch.randn([128, 2048], dtype=torch.float32)],
    ]
