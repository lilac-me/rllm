import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs interpolation (resizing) of tensors.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, size=None, scale_factor=None,
                mode: str = 'nearest', align_corners=None,
                recompute_scale_factor=None, antialias: bool = False) -> torch.Tensor:
        """
        Interpolates (resizes) the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, ...) where ... represents spatial dimensions.
            size (optional): Output spatial size.
            scale_factor (optional): Multiplier for spatial size.
            mode (str, optional): Algorithm used for interpolation: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'.
            align_corners (optional): How to align corners when resizing.
            recompute_scale_factor (optional): Recompute scale_factor for backward compatibility.
            antialias (bool, optional): Apply antialiasing.

        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return torch.nn.functional.interpolate(
            x, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([1, 3, 256, 256], dtype=torch.float32), [512, 512], 'bilinear', False],
        [torch.randn([1, 3, 512, 512], dtype=torch.float32), [256, 256], 'bilinear', False],
        [torch.randn([1, 3, 768, 768], dtype=torch.float32), [384, 384], 'bilinear', True],
        [torch.randn([1, 64, 256, 256], dtype=torch.float32), 2.0, 'bilinear', False],
        [torch.randn([1, 64, 512, 512], dtype=torch.float32), 0.5, 'bilinear', False],
        [torch.randn([1, 3, 1024, 1024], dtype=torch.float16), [512, 512], 'bilinear', False],
        [torch.randn([1, 3, 512, 512], dtype=torch.float16), [1024, 1024], 'bilinear', False],
        [torch.randn([1, 32, 1280, 720], dtype=torch.float16), [640, 360], 'bilinear', True],
        [torch.randn([1, 32, 640, 360], dtype=torch.float16), [1280, 720], 'bilinear', True],
        [torch.randn([1, 16, 1920, 1080], dtype=torch.float16), [960, 540], 'bilinear', False],
        [torch.randn([1, 3, 256, 256], dtype=torch.bfloat16), [512, 512], 'bilinear', False],
        [torch.randn([1, 3, 512, 512], dtype=torch.bfloat16), [256, 256], 'bilinear', False],
        [torch.randn([1, 3, 256, 256], dtype=torch.float32), [512, 512], 'bicubic', False],
        [torch.randn([1, 3, 512, 512], dtype=torch.float32), [256, 256], 'bicubic', False],
        [torch.randn([1, 3, 1024, 1024], dtype=torch.float32), 0.25, 'bicubic', True],
        [torch.randn([1, 3, 256, 256], dtype=torch.float32), 4.0, 'bicubic', True],
        [torch.randn([1, 3, 256, 256], dtype=torch.float32), [512, 512], 'nearest'],
        [torch.randn([1, 3, 512, 512], dtype=torch.float32), [256, 256], 'nearest'],
        [torch.randn([1, 64, 128, 128], dtype=torch.float32), 2.0, 'nearest'],
        [torch.randn([1, 64, 256, 256], dtype=torch.float32), 0.5, 'nearest'],
        [torch.randn([1, 3, 256, 256], dtype=torch.float32), [512, 512], 'area'],
        [torch.randn([1, 3, 512, 512], dtype=torch.float32), [256, 256], 'area'],
        [torch.randn([1, 128, 64, 64], dtype=torch.float32), 0.25, 'area'],
        [torch.randn([1, 128, 32, 32], dtype=torch.float16), 2.0, 'nearest'],
        [torch.randn([1, 64, 128, 128], dtype=torch.bfloat16), 2.0, 'nearest'],
        [torch.randn([2, 3, 512, 512], dtype=torch.float32), [256, 256], 'bilinear', False],
        [torch.randn([2, 64, 256, 256], dtype=torch.float32), 2.0, 'bilinear', False],
        [torch.randn([4, 3, 256, 256], dtype=torch.float32), [128, 128], 'bilinear', True],
        [torch.randn([4, 64, 128, 128], dtype=torch.float16), 2.0, 'bilinear', True],
        [torch.randn([1, 3, 512, 512], dtype=torch.float32), [1024, 768], 'bilinear', False],
        [torch.randn([1, 3, 1024, 768], dtype=torch.float32), [512, 512], 'bilinear', False],
        [torch.randn([1, 3, 800, 600], dtype=torch.float32), [400, 300], 'bilinear', True],
        [torch.randn([1, 3, 400, 300], dtype=torch.float32), [800, 600], 'bilinear', True],
        [torch.randn([1, 32, 640, 480], dtype=torch.float16), [320, 240], 'bilinear', False],
        [torch.randn([1, 32, 320, 240], dtype=torch.float16), [640, 480], 'bilinear', False],
        [torch.randn([1, 64, 224, 224], dtype=torch.float32), 2.0, 'bilinear', False],
        [torch.randn([1, 64, 448, 448], dtype=torch.float32), 0.5, 'bilinear', False],
        [torch.randn([1, 128, 112, 112], dtype=torch.float32), 2.0, 'bilinear', True],
        [torch.randn([1, 128, 224, 224], dtype=torch.float32), 0.5, 'bilinear', True],
        [torch.randn([1, 256, 56, 56], dtype=torch.float16), 2.0, 'nearest'],
        [torch.randn([1, 256, 112, 112], dtype=torch.float16), 0.5, 'nearest'],
        [torch.randn([1, 512, 28, 28], dtype=torch.bfloat16), 2.0, 'nearest'],
        [torch.randn([1, 512, 56, 56], dtype=torch.bfloat16), 0.5, 'nearest'],
        [torch.randn([1, 3, 384, 384], dtype=torch.float32), [768, 768], 'bicubic', False],
        [torch.randn([1, 3, 768, 768], dtype=torch.float32), [384, 384], 'bicubic', False],
        [torch.randn([1, 3, 256, 256], dtype=torch.float32), [128, 128], 'area'],
        [torch.randn([1, 3, 512, 512], dtype=torch.float32), [128, 128], 'area'],
        [torch.randn([1, 3, 1024, 1024], dtype=torch.float32), [256, 256], 'area'],
        [torch.randn([1, 3, 128, 128], dtype=torch.float32), [512, 512], 'bicubic', True],
        [torch.randn([1, 3, 64, 64], dtype=torch.float32), [256, 256], 'bilinear', False],
        [torch.randn([1, 3, 32, 32], dtype=torch.float32), [128, 128], 'bilinear', True],
        [torch.randn([1, 3, 300, 300], dtype=torch.float32), [600, 600], 'bilinear', False],
        [torch.randn([1, 3, 600, 600], dtype=torch.float32), [300, 300], 'bilinear', False],
        [torch.randn([1, 3, 450, 450], dtype=torch.float16), [225, 225], 'bilinear', True],
        [torch.randn([1, 3, 225, 225], dtype=torch.float16), [450, 450], 'bilinear', True],
        [torch.randn([1, 3, 720, 720], dtype=torch.bfloat16), [360, 360], 'bilinear', False],
        [torch.randn([1, 3, 360, 360], dtype=torch.bfloat16), [720, 720], 'bilinear', False],
        [torch.randn([1, 3, 333, 333], dtype=torch.float32), [666, 666], 'bilinear', False],
        [torch.randn([1, 3, 666, 666], dtype=torch.float32), [333, 333], 'bilinear', False],
        [torch.randn([1, 3, 513, 513], dtype=torch.float32), [257, 257], 'bilinear', True],
        [torch.randn([1, 3, 257, 257], dtype=torch.float32), [513, 513], 'bilinear', True],
        [torch.randn([1, 16, 277, 277], dtype=torch.float16), [554, 554], 'nearest'],
        [torch.randn([1, 16, 554, 554], dtype=torch.float16), [277, 277], 'nearest'],
        [torch.randn([1, 32, 411, 411], dtype=torch.float32), [822, 822], 'bilinear', False],
        [torch.randn([1, 32, 822, 822], dtype=torch.float32), [411, 411], 'bilinear', False],
        [torch.randn([1, 64, 189, 189], dtype=torch.bfloat16), 2.0, 'bilinear', True],
        [torch.randn([1, 64, 378, 378], dtype=torch.bfloat16), 0.5, 'bilinear', True],
        [torch.randn([1, 3, 1000, 1000], dtype=torch.float32), [500, 500], 'bilinear', False],
        [torch.randn([1, 3, 500, 500], dtype=torch.float32), [1000, 1000], 'bilinear', False],
        [torch.randn([1, 3, 1440, 900], dtype=torch.float16), [720, 450], 'bilinear', True],
        [torch.randn([1, 3, 720, 450], dtype=torch.float16), [1440, 900], 'bilinear', True],
        [torch.randn([1, 3, 2048, 1024], dtype=torch.float32), [1024, 512], 'area'],
        [torch.randn([1, 3, 1024, 512], dtype=torch.float32), [2048, 1024], 'nearest'],
    ]
