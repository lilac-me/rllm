import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs dynamic quantization on NPU.
    Pytorch native implemention
    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None,
                group_index: torch.Tensor = None, dst_type=None):
        if dst_type is None:
            dst_type = torch.int8

        x_float = x.float()

        if smooth_scales is not None:
            smooth_scales_float = smooth_scales.float()
            x_float = x_float * smooth_scales_float

        if group_index is not None:
            return self._quant_with_groups(x_float, group_index, dst_type)

        return self._quant_per_token(x_float, dst_type)

    def _quant_per_token(self, x: torch.Tensor, dst_type):
        if x.dim() == 2:
            max_abs = x.abs().max(dim=1, keepdim=True)[0]
            scale = max_abs / 127.0
            scale = scale.clamp(min=1e-10)
            quantized = torch.round(x / scale)
            quantized = quantized.clamp(-128, 127).to(dst_type)
            scale = scale.squeeze(1)
            return quantized, scale
        elif x.dim() == 3:
            max_abs = x.abs().max(dim=2, keepdim=True)[0]
            scale = max_abs / 127.0
            scale = scale.clamp(min=1e-10)
            quantized = torch.round(x / scale)
            quantized = quantized.clamp(-128, 127).to(dst_type)
            scale = scale.squeeze(2)
            return quantized, scale
        else:
            max_abs = x.abs().max()
            scale = max_abs / 127.0
            scale = torch.tensor(scale, device=x.device)
            quantized = torch.round(x / scale)
            quantized = quantized.clamp(-128, 127).to(dst_type)
            return quantized, scale

    def _quant_with_groups(self, x: torch.Tensor, group_index: torch.Tensor, dst_type):
        if x.dim() != 2:
            raise ValueError("Group quantization only supports 2D tensors")

        num_tokens = x.shape[0]
        quantized = torch.zeros_like(x, dtype=dst_type)
        scales = torch.zeros(num_tokens, device=x.device)

        num_groups = group_index.max().item() + 1 if group_index.numel() > 0 else 1

        for g in range(num_groups):
            mask = (group_index == g)
            if mask.sum() == 0:
                continue

            group_x = x[mask]
            max_abs = group_x.abs().max()
            scale = max_abs / 127.0
            scale = max(scale, 1e-10)

            group_quantized = torch.round(group_x / scale)
            group_quantized = group_quantized.clamp(-128, 127).to(dst_type)

            quantized[mask] = group_quantized
            scales[mask] = scale

        return quantized, scales
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None,
                group_index: torch.Tensor = None, dst_type=None):
        """
        Performs dynamic quantization on the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be quantized.
            smooth_scales (torch.Tensor, optional): Smooth scale factors.
            group_index (torch.Tensor, optional): Group indices for per-group quantization.
            dst_type (optional): Target data type for quantized output.

        Returns:
            tuple: (quantized_tensor, scale_tensor)
        """
        import torch_npu
        return torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales,
                                            group_index=group_index, dst_type=dst_type)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([128, 128], dtype=torch.bfloat16)],
        [torch.randn([256, 256], dtype=torch.bfloat16)],
        [torch.randn([512, 512], dtype=torch.bfloat16)],
        [torch.randn([1024, 1024], dtype=torch.bfloat16)],
        [torch.randn([128, 256], dtype=torch.float16)],
        [torch.randn([256, 512], dtype=torch.float16)],
        [torch.randn([512, 1024], dtype=torch.float16)],
        [torch.randn([1024, 2048], dtype=torch.float16)],
        [torch.randn([64, 64, 64], dtype=torch.bfloat16)],
        [torch.randn([32, 32, 32], dtype=torch.bfloat16)],
        [torch.randn([128, 64, 32], dtype=torch.float16)],
        [torch.randn([64, 32, 16], dtype=torch.float16)],
        [torch.randn([4096, 11008], dtype=torch.float16)],
        [torch.randn([4096, 14336], dtype=torch.float16)],
        [torch.randn([4096, 12288], dtype=torch.bfloat16)],
        [torch.randn([4096, 13824], dtype=torch.bfloat16)],
        [torch.randn([4096, 18432], dtype=torch.float16)],
        [torch.randn([4096, 24576], dtype=torch.float16)],
        [torch.randn([8192, 16384], dtype=torch.bfloat16)],
        [torch.randn([2048, 13824], dtype=torch.float16)],
        [torch.randn([5120, 27648], dtype=torch.float16)],
        [torch.randn([3584, 18944], dtype=torch.float16)],
        [torch.randn([5120, 13824], dtype=torch.bfloat16)],
        [torch.randn([1536, 8960], dtype=torch.float16)],
        [torch.randn([2560, 14592], dtype=torch.float16)],
        [torch.randn([3072, 12288], dtype=torch.float16)],
        [torch.randn([6144, 20480], dtype=torch.bfloat16)],
        [torch.randn([100, 200], dtype=torch.float16)],
        [torch.randn([17, 301], dtype=torch.bfloat16)],
        [torch.randn([13, 211], dtype=torch.float16)],
        [torch.randn([7, 15, 23], dtype=torch.bfloat16)],
        [torch.randn([11, 19, 29], dtype=torch.float16)],
        [torch.randn([123, 6144], dtype=torch.bfloat16)],
        [torch.randn([789, 12288], dtype=torch.float16)],
        [torch.randn([1, 4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([128, 4096], dtype=torch.float16), torch.randn([4096], dtype=torch.float16)],
        [torch.randn([256, 8192], dtype=torch.bfloat16), torch.randn([8192], dtype=torch.bfloat16)],
        [torch.randn([4096, 11008], dtype=torch.float16), torch.randn([11008], dtype=torch.float16)],
        [torch.randn([2048, 5120], dtype=torch.bfloat16), torch.randn([5120], dtype=torch.bfloat16)],
        [torch.randn([128, 4096], dtype=torch.float16)],
        [torch.randn([256, 8192], dtype=torch.float16)],
        [torch.randn([512, 4096], dtype=torch.bfloat16), torch.randn([4096], dtype=torch.bfloat16)],
    ]
