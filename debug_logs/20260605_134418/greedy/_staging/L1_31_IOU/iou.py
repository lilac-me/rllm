import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that computes IoU (Intersection over Union) on NPU.
    Pytorch native implemention
    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0) -> torch.Tensor:
        if mode not in [0, 1]:
            raise ValueError(f"mode must be 0 (IoU) or 1 (IoF), got {mode}")

        n = bboxes.shape[0]
        m = gtboxes.shape[0]

        bboxes_float = bboxes.float()
        gtboxes_float = gtboxes.float()

        lt = torch.max(bboxes_float[:, :2].unsqueeze(1), gtboxes_float[:, :2].unsqueeze(0))
        rb = torch.min(bboxes_float[:, 2:].unsqueeze(1), gtboxes_float[:, 2:].unsqueeze(0))

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area1 = (bboxes_float[:, 2] - bboxes_float[:, 0]) * (bboxes_float[:, 3] - bboxes_float[:, 1])
        area2 = (gtboxes_float[:, 2] - gtboxes_float[:, 0]) * (gtboxes_float[:, 3] - gtboxes_float[:, 1])

        if mode == 0:
            union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
            iou = inter / union.clamp(min=1e-10)
        else:
            iou = inter / area2.unsqueeze(0).clamp(min=1e-10)

        return iou.t().to(bboxes.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0) -> torch.Tensor:
        """
        Computes IoU between bounding boxes.

        Args:
            bboxes (torch.Tensor): First set of bounding boxes.
            gtboxes (torch.Tensor): Second set of bounding boxes (ground truth).
            mode (int, optional): IoU computation mode (0: IoU, 1: IoF).

        Returns:
            torch.Tensor: IoU values between bounding boxes.
        """
        import torch_npu
        return torch_npu.npu_iou(bboxes, gtboxes, mode=mode)


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([10, 4], dtype=torch.float32), 0],
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([10, 4], dtype=torch.float32), 1],
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([50, 4], dtype=torch.float32), 0],
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([100, 4], dtype=torch.float32), 0],
        [torch.randn([256, 4], dtype=torch.float16), torch.randn([10, 4], dtype=torch.float16), 0],
        [torch.randn([256, 4], dtype=torch.float16), torch.randn([50, 4], dtype=torch.float16), 0],
        [torch.randn([256, 4], dtype=torch.float16), torch.randn([100, 4], dtype=torch.float16), 0],
        [torch.randn([512, 4], dtype=torch.float32), torch.randn([10, 4], dtype=torch.float32), 0],
        [torch.randn([512, 4], dtype=torch.float32), torch.randn([50, 4], dtype=torch.float32), 1],
        [torch.randn([512, 4], dtype=torch.float32), torch.randn([100, 4], dtype=torch.float32), 0],
        [torch.randn([1024, 4], dtype=torch.float16), torch.randn([10, 4], dtype=torch.float16), 0],
        [torch.randn([1024, 4], dtype=torch.float16), torch.randn([50, 4], dtype=torch.float16), 0],
        [torch.randn([1024, 4], dtype=torch.float16), torch.randn([100, 4], dtype=torch.float16), 1],
        [torch.randn([2048, 4], dtype=torch.float32), torch.randn([10, 4], dtype=torch.float32), 0],
        [torch.randn([2048, 4], dtype=torch.float32), torch.randn([50, 4], dtype=torch.float32), 0],
        [torch.randn([2048, 4], dtype=torch.float32), torch.randn([100, 4], dtype=torch.float32), 0],
        [torch.randn([4096, 4], dtype=torch.float16), torch.randn([10, 4], dtype=torch.float16), 0],
        [torch.randn([4096, 4], dtype=torch.float16), torch.randn([50, 4], dtype=torch.float16), 1],
        [torch.randn([4096, 4], dtype=torch.float16), torch.randn([100, 4], dtype=torch.float16), 0],
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([1, 4], dtype=torch.float32), 0],
        [torch.randn([1000, 4], dtype=torch.float32), torch.randn([1, 4], dtype=torch.float32), 0],
        [torch.randn([100, 4], dtype=torch.float16), torch.randn([200, 4], dtype=torch.float16), 0],
        [torch.randn([500, 4], dtype=torch.float16), torch.randn([150, 4], dtype=torch.float16), 1],
        [torch.randn([1500, 4], dtype=torch.float32), torch.randn([75, 4], dtype=torch.float32), 0],
        [torch.randn([3000, 4], dtype=torch.float16), torch.randn([25, 4], dtype=torch.float16), 0],
        [torch.randn([8192, 4], dtype=torch.float32), torch.randn([10, 4], dtype=torch.float32), 0],
        [torch.randn([8192, 4], dtype=torch.float16), torch.randn([50, 4], dtype=torch.float16), 1],
        [torch.randn([8192, 4], dtype=torch.float32), torch.randn([100, 4], dtype=torch.float32), 0],
        [torch.randn([64, 4], dtype=torch.float16), torch.randn([8, 4], dtype=torch.float16), 0],
        [torch.randn([128, 4], dtype=torch.float32), torch.randn([16, 4], dtype=torch.float32), 1],
    ]
