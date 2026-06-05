import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        """
        Performs Non-Maximum Suppression (NMS) on bounding boxes.
        Pure PyTorch reference implementation.
        """
        boxes_f32 = boxes.float()
        scores_f32 = scores.float()

        score_mask = scores_f32 > scores_threshold
        filtered_boxes = boxes_f32[score_mask]
        filtered_scores = scores_f32[score_mask]
        original_indices = torch.where(score_mask)[0]

        selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)

        if filtered_boxes.shape[0] == 0:
            num_selected = torch.tensor(0, dtype=torch.int32, device=boxes.device)
            return selected_indices, num_selected

        sorted_indices = torch.argsort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_indices]
        sorted_original_indices = original_indices[sorted_indices]

        num_boxes = sorted_boxes.shape[0]
        selected_indices_list = []
        suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes.device)

        areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]) * (sorted_boxes[:, 3] - sorted_boxes[:, 1])

        for i in range(num_boxes):
            if suppressed[i]:
                continue

            selected_indices_list.append(sorted_original_indices[i].item())

            if len(selected_indices_list) >= max_output_size:
                break

            rest = torch.arange(i + 1, num_boxes, device=boxes.device)
            if rest.numel() == 0:
                break
            mask = ~suppressed[rest]
            if not mask.any():
                continue
            candidates = rest[mask]

            cur_box = sorted_boxes[i]
            cand_boxes = sorted_boxes[candidates]

            x1_inter = torch.maximum(cur_box[0].expand(cand_boxes.shape[0]), cand_boxes[:, 0])
            y1_inter = torch.maximum(cur_box[1].expand(cand_boxes.shape[0]), cand_boxes[:, 1])
            x2_inter = torch.minimum(cur_box[2].expand(cand_boxes.shape[0]), cand_boxes[:, 2])
            y2_inter = torch.minimum(cur_box[3].expand(cand_boxes.shape[0]), cand_boxes[:, 3])

            inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
            union_area = areas[i] + areas[candidates] - inter_area
            iou = inter_area / union_area.clamp(min=1e-6)

            suppress_mask = iou >= iou_threshold
            suppressed[candidates[suppress_mask]] = True

        num_selected = len(selected_indices_list)

        if num_selected > 0:
            selected_indices[:num_selected] = torch.tensor(
                selected_indices_list, dtype=torch.int32, device=boxes.device
            )

        num_selected_tensor = torch.tensor(num_selected, dtype=torch.int32, device=boxes.device)

        return selected_indices, num_selected_tensor


def _make_legal_boxes(shape, dtype):
    assert len(shape) == 2 and shape[1] == 4, f"boxes shape must be [N, 4], got {shape}"
    n = shape[0]

    raw = torch.randn(n, 2, 2, dtype=torch.float32)
    pt_a = raw[:, 0, :]
    pt_b = raw[:, 1, :]

    x1 = torch.minimum(pt_a[:, 0], pt_b[:, 0])
    y1 = torch.minimum(pt_a[:, 1], pt_b[:, 1])
    x2 = torch.maximum(pt_a[:, 0], pt_b[:, 0])
    y2 = torch.maximum(pt_a[:, 1], pt_b[:, 1])

    eps = 1e-3
    x2 = torch.where(x2 - x1 < eps, x1 + eps, x2)
    y2 = torch.where(y2 - y1 < eps, y1 + eps, y2)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).to(dtype)
    return boxes


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([100], dtype=torch.float32), 100, 0.5, 0.05],
        [torch.randn([256, 4], dtype=torch.float32), torch.randn([256], dtype=torch.float32), 256, 0.5, 0.05],
        [torch.randn([512, 4], dtype=torch.float32), torch.randn([512], dtype=torch.float32), 512, 0.5, 0.05],
        [torch.randn([1024, 4], dtype=torch.float32), torch.randn([1024], dtype=torch.float32), 1024, 0.5, 0.05],
        [torch.randn([2048, 4], dtype=torch.float32), torch.randn([2048], dtype=torch.float32), 1000, 0.5, 0.05],
        [torch.randn([4096, 4], dtype=torch.float32), torch.randn([4096], dtype=torch.float32), 1000, 0.5, 0.05],
        [torch.randn([8192, 4], dtype=torch.float16), torch.randn([8192], dtype=torch.float16), 1000, 0.5, 0.05],
        [torch.randn([8732, 4], dtype=torch.float16), torch.randn([8732], dtype=torch.float16), 200, 0.5, 0.01],
        [torch.randn([10000, 4], dtype=torch.float32), torch.randn([10000], dtype=torch.float32), 1000, 0.5, 0.05],
        [torch.randn([16192, 4], dtype=torch.float16), torch.randn([16192], dtype=torch.float16), 1000, 0.5, 0.05],
        [torch.randn([8450, 4], dtype=torch.float32), torch.randn([8450], dtype=torch.float32), 1000, 0.45, 0.25],
        [torch.randn([21125, 4], dtype=torch.float16), torch.randn([21125], dtype=torch.float16), 1000, 0.45, 0.25],
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([100], dtype=torch.float32), 100, 0.3, 0.1],
        [torch.randn([1000, 4], dtype=torch.float32), torch.randn([1000], dtype=torch.float32), 300, 0.7, 0.001],
        [torch.randn([2048, 4], dtype=torch.float16), torch.randn([2048], dtype=torch.float16), 100, 0.4, 0.5],
        [torch.randn([4096, 4], dtype=torch.float32), torch.randn([4096], dtype=torch.float32), 500, 0.6, 0.01],
        [torch.randn([8192, 4], dtype=torch.float16), torch.randn([8192], dtype=torch.float16), 2000, 0.5, 0.001],
        [torch.randn([100, 4], dtype=torch.float32), torch.randn([100], dtype=torch.float32), 100, 0.5, 0.05, True],
        [torch.randn([1000, 4], dtype=torch.float32), torch.randn([1000], dtype=torch.float32), 300, 0.5, 0.05, True],
        [torch.randn([2048, 4], dtype=torch.float16), torch.randn([2048], dtype=torch.float16), 100, 0.5, 0.05, True],
        [torch.randn([50, 4], dtype=torch.float32), torch.randn([50], dtype=torch.float32), 50, 0.5, 0.05],
        [torch.randn([150, 4], dtype=torch.float16), torch.randn([150], dtype=torch.float16), 100, 0.5, 0.05],
        [torch.randn([300, 4], dtype=torch.float32), torch.randn([300], dtype=torch.float32), 200, 0.5, 0.05],
        [torch.randn([500, 4], dtype=torch.float16), torch.randn([500], dtype=torch.float16), 300, 0.5, 0.05],
        [torch.randn([750, 4], dtype=torch.float32), torch.randn([750], dtype=torch.float32), 500, 0.5, 0.05],
        [torch.randn([1234, 4], dtype=torch.float16), torch.randn([1234], dtype=torch.float16), 500, 0.5, 0.05],
        [torch.randn([5678, 4], dtype=torch.float32), torch.randn([5678], dtype=torch.float32), 1000, 0.5, 0.05],
        [torch.randn([3456, 4], dtype=torch.float16), torch.randn([3456], dtype=torch.float16), 500, 0.5, 0.05],
        [torch.randn([7890, 4], dtype=torch.float32), torch.randn([7890], dtype=torch.float32), 1000, 0.5, 0.05],
        [torch.randn([16384, 4], dtype=torch.float16), torch.randn([16384], dtype=torch.float16), 1000, 0.5, 0.05],
        [torch.randn([32768, 4], dtype=torch.float32), torch.randn([32768], dtype=torch.float32), 1000, 0.5, 0.05],
    ]
