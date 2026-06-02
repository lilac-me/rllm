import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference: elementwise add (same as good fixture)."""

    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.randn(4096), torch.randn(4096)]


def get_init_inputs():
    return []
