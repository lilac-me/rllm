import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """BAD fixture — PyTorch 退化 (Type 1): no @triton.jit kernel at all.

    Expected: validate_triton_impl.py rejects it -> ast_check_ok=false
    (so the pipeline never reaches verify; correctness_ok stays false).
    This exercises the AST gate without needing NPU.
    """

    def forward(self, x, y):
        return x + y
