"""ViTDet Multi-Scale Detection Head, Loss Functions, and NMS Decoding.

Ref: Li et al., "Exploring Plain Vision Transformer Backbones for Object Detection"
(papers/vitdet/vitdet.md), §4.1, Appendix 0.A.1 (Fig. 5 RetinaNet / One-Stage Head)
& Appendix 0.A.2 "Architectures":
"We use 4 hidden convolution layers for the heads... These hidden convolution
layers are followed by LN."

Implements a multi-scale dense detector head operating across Simple Feature Pyramid
levels {P2, P3, P4, P5} with 4-layer (Conv3x3 + LayerNorm2d + GELU) classification
and box regression towers, Sigmoid Focal Loss, Generalized IoU (GIoU) Loss,
Centerness BCE Loss, and pure-PyTorch Non-Maximum Suppression (NMS).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from histojepa.models.vitdet.sfp import LayerNorm2d


@dataclass
class DetectionLossOutput:
    """Container for total and decomposed detection training losses."""

    total_loss: torch.Tensor
    cls_loss: torch.Tensor
    box_loss: torch.Tensor
    ctr_loss: torch.Tensor
    num_positives: int


class Scale(nn.Module):
    """Learnable scalar multiplier per feature pyramid level for box regression."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Computes Sigmoid Focal Loss for dense object detection (Lin et al., RetinaNet).

    Args:
        inputs: Raw logit tensor of arbitrary shape.
        targets: Binary target tensor in {0, 1} of the same shape as inputs.
        alpha: Weighting factor in [0, 1] balancing positive vs negative examples.
        gamma: Focusing exponent modulating easy vs hard examples.
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        Focal loss tensor.
    """
    inputs_f32 = inputs.float()
    targets_f32 = targets.float()
    p = torch.sigmoid(inputs_f32)
    ce_loss = F.binary_cross_entropy_with_logits(inputs_f32, targets_f32, reduction="none")
    p_t = p * targets_f32 + (1.0 - p) * (1.0 - targets_f32)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets_f32 + (1.0 - alpha) * (1.0 - targets_f32)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def giou_loss_xyxy(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes Generalized IoU (GIoU) loss between aligned [x1, y1, x2, y2] box pairs.

    Args:
        pred_boxes: Predicted boxes of shape (N, 4) in [x1, y1, x2, y2] format.
        target_boxes: Ground-truth boxes of shape (N, 4) in [x1, y1, x2, y2] format.
        weights: Optional per-box weight tensor of shape (N,).
        eps: Numerical stability epsilon.

    Returns:
        Scalar weighted GIoU loss (or 0.0 if N == 0).
    """
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    pred = pred_boxes.float()
    target = target_boxes.float()

    x1_p, y1_p, x2_p, y2_p = pred.unbind(dim=-1)
    x1_g, y1_g, x2_g, y2_g = target.unbind(dim=-1)

    # Intersection
    ix1 = torch.maximum(x1_p, x1_g)
    iy1 = torch.maximum(y1_p, y1_g)
    ix2 = torch.minimum(x2_p, x2_g)
    iy2 = torch.minimum(y2_p, y2_g)

    inter_w = (ix2 - ix1).clamp(min=0.0)
    inter_h = (iy2 - iy1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area_p = ((x2_p - x1_p).clamp(min=0.0)) * ((y2_p - y1_p).clamp(min=0.0))
    area_g = ((x2_g - x1_g).clamp(min=0.0)) * ((y2_g - y1_g).clamp(min=0.0))
    union_area = area_p + area_g - inter_area + eps

    iou = inter_area / union_area

    # Smallest enclosing convex box
    cx1 = torch.minimum(x1_p, x1_g)
    cy1 = torch.minimum(y1_p, y1_g)
    cx2 = torch.maximum(x2_p, x2_g)
    cy2 = torch.maximum(y2_p, y2_g)
    c_area = ((cx2 - cx1).clamp(min=0.0)) * ((cy2 - cy1).clamp(min=0.0)) + eps

    giou = iou - (c_area - union_area) / c_area
    loss = 1.0 - giou

    if weights is not None:
        w = weights.float()
        return (loss * w).sum() / w.sum().clamp(min=eps)
    return loss.mean()


def batched_nms_pure_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> torch.Tensor:
    """Pure-PyTorch class-aware Non-Maximum Suppression (NMS).

    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Tensor of shape (N,) with confidence scores.
        labels: Tensor of shape (N,) with integer class labels.
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.
        max_detections: Maximum number of kept detections.

    Returns:
        1D LongTensor of kept indices sorted by descending score.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # Offset boxes by class label so different classes never suppress each other
    max_coord = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coord + 4096.0)
    boxes_for_nms = boxes + offsets[:, None]

    x1 = boxes_for_nms[:, 0]
    y1 = boxes_for_nms[:, 1]
    x2 = boxes_for_nms[:, 2]
    y2 = boxes_for_nms[:, 3]
    areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)

    order = scores.argsort(descending=True)
    # Cap candidate pool before greedy loop for fast execution
    if order.numel() > 2000:
        order = order[:2000]

    keep: List[int] = []
    while order.numel() > 0 and len(keep) < max_detections:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)

        mask = iou <= iou_threshold
        order = rest[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class ViTDetHead(nn.Module):
    """Multi-scale Detection Head for ViTDet Simple Feature Pyramid.

    Ref: ViTDet (papers/vitdet/vitdet.md), Appendix 0.A.1 & 0.A.2:
    Uses 4 hidden 3x3 convolution layers with LayerNorm2d and GELU activation
    in both the classification and box regression towers across {P2, P3, P4, P5}.

    Args:
        in_channels: Input feature channels per pyramid level (default: 256).
        num_classes: Number of foreground object classes (default: 1 for Tumor Bud).
        num_convs: Number of 3x3 conv + LN + GELU layers per tower (default: 4).
        prior_prob: Prior foreground probability for focal loss bias initialization (default: 0.01).
        strides: Feature pyramid strides for P2, P3, P4, P5 (default: (4, 8, 16, 32)).
        scale_ranges: Object size assignment ranges per pyramid level in pixels.
        center_sampling_radius: Radius (in stride units) around GT center for positive assignment.
        focal_alpha: Focal loss alpha (default: 0.25).
        focal_gamma: Focal loss gamma (default: 2.0).
    """

    LEVEL_NAMES = ("p2", "p3", "p4", "p5")

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        num_convs: int = 4,
        prior_prob: float = 0.01,
        strides: Tuple[int, int, int, int] = (4, 8, 16, 32),
        scale_ranges: Tuple[Tuple[float, float], ...] = (
            (0.0, 48.0),
            (48.0, 96.0),
            (96.0, 192.0),
            (192.0, 1e8),
        ),
        center_sampling_radius: float = 1.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_convs = int(num_convs)
        self.strides = tuple(int(s) for s in strides)
        self.scale_ranges = tuple((float(lo), float(hi)) for lo, hi in scale_ranges)
        self.center_sampling_radius = float(center_sampling_radius)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

        # 4-layer Classification Tower (Conv3x3 + LayerNorm2d + GELU)
        cls_layers: List[nn.Module] = []
        for _ in range(self.num_convs):
            cls_layers.extend(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    LayerNorm2d(in_channels),
                    nn.GELU(),
                ]
            )
        self.cls_tower = nn.Sequential(*cls_layers)

        # 4-layer Box Regression Tower (Conv3x3 + LayerNorm2d + GELU)
        box_layers: List[nn.Module] = []
        for _ in range(self.num_convs):
            box_layers.extend(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    LayerNorm2d(in_channels),
                    nn.GELU(),
                ]
            )
        self.box_tower = nn.Sequential(*box_layers)

        # Prediction heads
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # Per-level learnable scale for regression distances (l, t, r, b)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(len(self.LEVEL_NAMES))])

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float) -> None:
        for m in [self.cls_tower, self.box_tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Focal loss bias initialization so initial foreground prob ~ prior_prob
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward_features(
        self,
        pyramid_features: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Runs shared towers across pyramid levels {p2, p3, p4, p5}.

        Returns:
            Tuple of lists across levels:
                - cls_logits_list: each `(B, num_classes, H_l, W_l)`
                - bbox_reg_list: each `(B, 4, H_l, W_l)` (non-negative normalized distances in stride units)
                - centerness_list: each `(B, 1, H_l, W_l)`
        """
        cls_logits_list: List[torch.Tensor] = []
        bbox_reg_list: List[torch.Tensor] = []
        centerness_list: List[torch.Tensor] = []

        for idx, level_name in enumerate(self.LEVEL_NAMES):
            feat = pyramid_features[level_name]
            cls_feat = self.cls_tower(feat)
            box_feat = self.box_tower(feat)

            cls_logits_list.append(self.cls_logits(cls_feat))
            centerness_list.append(self.centerness(box_feat))
            # Softplus / ReLU guarantees non-negative (l, t, r, b) box distances
            reg = F.relu(self.scales[idx](self.bbox_pred(box_feat)))
            bbox_reg_list.append(reg)

        return cls_logits_list, bbox_reg_list, centerness_list

    @staticmethod
    def _build_grid_points(
        h: int,
        w: int,
        stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generates `(H*W, 2)` pixel coordinates `(x, y)` of feature cell centers."""
        shifts_y = (torch.arange(0, h, device=device, dtype=dtype) + 0.5) * stride
        shifts_x = (torch.arange(0, w, device=device, dtype=dtype) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        return points

    def compute_loss(
        self,
        cls_logits_list: List[torch.Tensor],
        bbox_reg_list: List[torch.Tensor],
        centerness_list: List[torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> DetectionLossOutput:
        """Computes multi-scale Sigmoid Focal + GIoU + Centerness loss against GT targets."""
        device = cls_logits_list[0].device
        batch_size = cls_logits_list[0].shape[0]

        all_points: List[torch.Tensor] = []
        all_strides: List[torch.Tensor] = []
        all_ranges: List[torch.Tensor] = []

        # Flatten predictions per level
        flat_cls_per_level: List[torch.Tensor] = []
        flat_reg_per_level: List[torch.Tensor] = []
        flat_ctr_per_level: List[torch.Tensor] = []

        for lvl_idx, stride in enumerate(self.strides):
            cls_l = cls_logits_list[lvl_idx]   # (B, C, H, W)
            reg_l = bbox_reg_list[lvl_idx]     # (B, 4, H, W)
            ctr_l = centerness_list[lvl_idx]   # (B, 1, H, W)
            _, _, h_l, w_l = cls_l.shape

            pts = self._build_grid_points(h_l, w_l, stride, device, torch.float32)  # (N_l, 2)
            n_l = pts.shape[0]

            all_points.append(pts)
            all_strides.append(torch.full((n_l,), float(stride), device=device, dtype=torch.float32))
            lo, hi = self.scale_ranges[lvl_idx]
            all_ranges.append(
                torch.tensor([[lo, hi]], device=device, dtype=torch.float32).expand(n_l, 2)
            )

            flat_cls_per_level.append(cls_l.permute(0, 2, 3, 1).reshape(batch_size, n_l, self.num_classes))
            flat_reg_per_level.append(reg_l.permute(0, 2, 3, 1).reshape(batch_size, n_l, 4) * float(stride))
            flat_ctr_per_level.append(ctr_l.permute(0, 2, 3, 1).reshape(batch_size, n_l))

        points = torch.cat(all_points, dim=0)        # (N_total, 2)
        strides_vec = torch.cat(all_strides, dim=0)  # (N_total,)
        ranges_mat = torch.cat(all_ranges, dim=0)    # (N_total, 2)

        pred_cls_all = torch.cat(flat_cls_per_level, dim=1)  # (B, N_total, C)
        pred_reg_all = torch.cat(flat_reg_per_level, dim=1)  # (B, N_total, 4) in pixels
        pred_ctr_all = torch.cat(flat_ctr_per_level, dim=1)  # (B, N_total)

        px = points[:, 0]  # (N_total,)
        py = points[:, 1]  # (N_total,)

        cls_targets_batch: List[torch.Tensor] = []
        pos_pred_boxes_list: List[torch.Tensor] = []
        pos_gt_boxes_list: List[torch.Tensor] = []
        pos_pred_ctr_list: List[torch.Tensor] = []
        pos_gt_ctr_list: List[torch.Tensor] = []

        total_positives = 0

        for b_idx in range(batch_size):
            gt_boxes = targets[b_idx]["boxes"].to(device=device, dtype=torch.float32)
            gt_labels = targets[b_idx]["labels"].to(device=device, dtype=torch.long)

            n_total = points.shape[0]
            cls_target = torch.zeros((n_total, self.num_classes), device=device, dtype=torch.float32)

            if gt_boxes.shape[0] == 0:
                cls_targets_batch.append(cls_target)
                continue

            # Compute distances (l, t, r, b) from every grid point (N_total, 1) to every GT box (1, M)
            l_dist = px[:, None] - gt_boxes[None, :, 0]
            t_dist = py[:, None] - gt_boxes[None, :, 1]
            r_dist = gt_boxes[None, :, 2] - px[:, None]
            b_dist = gt_boxes[None, :, 3] - py[:, None]
            reg_targets_per_gt = torch.stack([l_dist, t_dist, r_dist, b_dist], dim=-1)  # (N_total, M, 4)

            # 1. Inside GT box condition
            is_in_box = reg_targets_per_gt.min(dim=-1).values > 0.0  # (N_total, M)

            # 2. Center sampling condition: within radius * stride of GT box center
            gt_cx = 0.5 * (gt_boxes[:, 0] + gt_boxes[:, 2])  # (M,)
            gt_cy = 0.5 * (gt_boxes[:, 1] + gt_boxes[:, 3])  # (M,)
            radius_px = self.center_sampling_radius * strides_vec[:, None]  # (N_total, 1)

            c_x1 = torch.maximum(gt_boxes[None, :, 0], gt_cx[None, :] - radius_px)
            c_y1 = torch.maximum(gt_boxes[None, :, 1], gt_cy[None, :] - radius_px)
            c_x2 = torch.minimum(gt_boxes[None, :, 2], gt_cx[None, :] + radius_px)
            c_y2 = torch.minimum(gt_boxes[None, :, 3], gt_cy[None, :] + radius_px)

            is_in_center = (
                (px[:, None] >= c_x1)
                & (px[:, None] <= c_x2)
                & (py[:, None] >= c_y1)
                & (py[:, None] <= c_y2)
            )

            # 3. Scale range condition per pyramid level
            max_reg_dist = reg_targets_per_gt.max(dim=-1).values  # (N_total, M)
            is_in_scale = (max_reg_dist >= ranges_mat[:, 0:1]) & (max_reg_dist <= ranges_mat[:, 1:2])

            candidate_mask = is_in_center & is_in_scale  # (N_total, M)

            # Fallback: if a GT box has no candidate points in center & scale, assign its closest inside/grid point
            gt_has_pos = candidate_mask.any(dim=0)  # (M,)
            if not bool(gt_has_pos.all()):
                dist_sq = (px[:, None] - gt_cx[None, :]) ** 2 + (py[:, None] - gt_cy[None, :]) ** 2
                # Prefer levels whose stride best matches box size
                gt_size = torch.sqrt(
                    (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
                    * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
                )
                for m_idx in range(gt_boxes.shape[0]):
                    if not bool(gt_has_pos[m_idx]):
                        in_scale_m = is_in_scale[:, m_idx]
                        if bool(in_scale_m.any()):
                            masked_d = torch.where(
                                in_scale_m, dist_sq[:, m_idx], torch.full_like(dist_sq[:, m_idx], 1e18)
                            )
                            best_pt = int(masked_d.argmin().item())
                        else:
                            best_pt = int(dist_sq[:, m_idx].argmin().item())
                        candidate_mask[best_pt, m_idx] = True

            # If a grid point matches multiple GT boxes, assign the GT box with minimal area
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # (M,)
            areas_matrix = torch.where(
                candidate_mask,
                gt_areas[None, :].expand(n_total, -1),
                torch.full((n_total, gt_boxes.shape[0]), 1e18, device=device, dtype=torch.float32),
            )
            min_areas, matched_gt_idx = areas_matrix.min(dim=1)  # (N_total,)
            pos_mask = min_areas < 1e17

            pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
            num_pos_b = int(pos_indices.numel())
            total_positives += num_pos_b

            if num_pos_b > 0:
                matched_gt = matched_gt_idx[pos_indices]
                matched_cls = gt_labels[matched_gt].clamp(min=0, max=self.num_classes - 1)
                cls_target[pos_indices, matched_cls] = 1.0

                # Decode predicted boxes at positive locations: [px - l, py - t, px + r, py + b]
                pos_px = px[pos_indices]
                pos_py = py[pos_indices]
                pos_reg_pred = pred_reg_all[b_idx, pos_indices]  # (N_pos, 4)
                pred_x1 = pos_px - pos_reg_pred[:, 0]
                pred_y1 = pos_py - pos_reg_pred[:, 1]
                pred_x2 = pos_px + pos_reg_pred[:, 2]
                pred_y2 = pos_py + pos_reg_pred[:, 3]
                pos_pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

                pos_gt_boxes = gt_boxes[matched_gt]
                pos_pred_boxes_list.append(pos_pred_boxes)
                pos_gt_boxes_list.append(pos_gt_boxes)

                # Centerness target: sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
                pos_reg_tgt = reg_targets_per_gt[pos_indices, matched_gt].clamp(min=1e-4)
                lr_min = torch.minimum(pos_reg_tgt[:, 0], pos_reg_tgt[:, 2])
                lr_max = torch.maximum(pos_reg_tgt[:, 0], pos_reg_tgt[:, 2])
                tb_min = torch.minimum(pos_reg_tgt[:, 1], pos_reg_tgt[:, 3])
                tb_max = torch.maximum(pos_reg_tgt[:, 1], pos_reg_tgt[:, 3])
                ctr_tgt = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max)).clamp(min=0.0, max=1.0)

                pos_pred_ctr_list.append(pred_ctr_all[b_idx, pos_indices])
                pos_gt_ctr_list.append(ctr_tgt)

            cls_targets_batch.append(cls_target)

        cls_targets_tensor = torch.stack(cls_targets_batch, dim=0)  # (B, N_total, C)
        avg_factor = float(max(total_positives, 1))

        cls_loss = (
            sigmoid_focal_loss(
                pred_cls_all,
                cls_targets_tensor,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )
            / avg_factor
        )

        if total_positives > 0:
            all_pos_pred_boxes = torch.cat(pos_pred_boxes_list, dim=0)
            all_pos_gt_boxes = torch.cat(pos_gt_boxes_list, dim=0)
            all_pos_pred_ctr = torch.cat(pos_pred_ctr_list, dim=0)
            all_pos_gt_ctr = torch.cat(pos_gt_ctr_list, dim=0)

            box_loss = giou_loss_xyxy(
                all_pos_pred_boxes,
                all_pos_gt_boxes,
                weights=all_pos_gt_ctr,
            )
            ctr_loss = F.binary_cross_entropy_with_logits(
                all_pos_pred_ctr.float(),
                all_pos_gt_ctr.float(),
                reduction="mean",
            )
        else:
            box_loss = pred_reg_all.sum() * 0.0
            ctr_loss = pred_ctr_all.sum() * 0.0

        total_loss = cls_loss + box_loss + 0.5 * ctr_loss
        return DetectionLossOutput(
            total_loss=total_loss,
            cls_loss=cls_loss.detach(),
            box_loss=box_loss.detach(),
            ctr_loss=ctr_loss.detach(),
            num_positives=total_positives,
        )

    @torch.no_grad()
    def predict(
        self,
        cls_logits_list: List[torch.Tensor],
        bbox_reg_list: List[torch.Tensor],
        centerness_list: List[torch.Tensor],
        image_size: Tuple[int, int] = (1280, 1280),
        score_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 300,
        pre_nms_topk: int = 1000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decodes multi-scale predictions into NMS-filtered bounding boxes per image.

        Args:
            cls_logits_list: Per-level classification logits `(B, C, H_l, W_l)`.
            bbox_reg_list: Per-level box regression `(B, 4, H_l, W_l)`.
            centerness_list: Per-level centerness logits `(B, 1, H_l, W_l)`.
            image_size: `(H, W)` image pixel bounds for clipping boxes.
            score_threshold: Minimum confidence score to retain before NMS.
            nms_iou_threshold: IoU threshold for NMS suppression.
            max_detections: Maximum detections returned per image.
            pre_nms_topk: Top-K candidates retained per level before NMS.

        Returns:
            List of `B` dictionaries, each containing:
                - `"boxes"`: FloatTensor of shape `(K, 4)` in `[x1, y1, x2, y2]`
                - `"scores"`: FloatTensor of shape `(K,)`
                - `"labels"`: LongTensor of shape `(K,)`
        """
        device = cls_logits_list[0].device
        batch_size = cls_logits_list[0].shape[0]
        img_h, img_w = image_size

        batch_boxes: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        batch_scores: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        batch_labels: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        for lvl_idx, stride in enumerate(self.strides):
            cls_l = cls_logits_list[lvl_idx].float()
            reg_l = bbox_reg_list[lvl_idx].float() * float(stride)
            ctr_l = centerness_list[lvl_idx].float()
            _, num_cls, h_l, w_l = cls_l.shape

            pts = self._build_grid_points(h_l, w_l, stride, device, torch.float32)  # (N_l, 2)
            n_l = pts.shape[0]

            probs = torch.sigmoid(cls_l).permute(0, 2, 3, 1).reshape(batch_size, n_l, num_cls)
            ctr_prob = torch.sigmoid(ctr_l).permute(0, 2, 3, 1).reshape(batch_size, n_l, 1)
            # Combine classification probability with centerness quality score
            scores_all = torch.sqrt(probs * ctr_prob)  # (B, N_l, C)
            reg_flat = reg_l.permute(0, 2, 3, 1).reshape(batch_size, n_l, 4)

            for b_idx in range(batch_size):
                scores_b = scores_all[b_idx].reshape(-1)  # (N_l * C,)
                keep_mask = scores_b > score_threshold
                if not bool(keep_mask.any()):
                    continue

                valid_idxs = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
                valid_scores = scores_b[valid_idxs]

                if valid_scores.numel() > pre_nms_topk:
                    valid_scores, topk_rel = valid_scores.topk(pre_nms_topk)
                    valid_idxs = valid_idxs[topk_rel]

                loc_idxs = torch.div(valid_idxs, num_cls, rounding_mode="floor")
                cls_idxs = valid_idxs % num_cls

                sel_pts = pts[loc_idxs]
                sel_reg = reg_flat[b_idx, loc_idxs]

                x1 = (sel_pts[:, 0] - sel_reg[:, 0]).clamp(min=0.0, max=float(img_w))
                y1 = (sel_pts[:, 1] - sel_reg[:, 1]).clamp(min=0.0, max=float(img_h))
                x2 = (sel_pts[:, 0] + sel_reg[:, 2]).clamp(min=0.0, max=float(img_w))
                y2 = (sel_pts[:, 1] + sel_reg[:, 3]).clamp(min=0.0, max=float(img_h))

                boxes_b = torch.stack([x1, y1, x2, y2], dim=-1)
                non_empty = (boxes_b[:, 2] - boxes_b[:, 0] > 1.0) & (boxes_b[:, 3] - boxes_b[:, 1] > 1.0)
                if bool(non_empty.any()):
                    batch_boxes[b_idx].append(boxes_b[non_empty])
                    batch_scores[b_idx].append(valid_scores[non_empty])
                    batch_labels[b_idx].append(cls_idxs[non_empty])

        results: List[Dict[str, torch.Tensor]] = []
        for b_idx in range(batch_size):
            if len(batch_boxes[b_idx]) == 0:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                    }
                )
                continue

            cat_boxes = torch.cat(batch_boxes[b_idx], dim=0)
            cat_scores = torch.cat(batch_scores[b_idx], dim=0)
            cat_labels = torch.cat(batch_labels[b_idx], dim=0)

            keep = batched_nms_pure_torch(
                cat_boxes,
                cat_scores,
                cat_labels,
                iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            results.append(
                {
                    "boxes": cat_boxes[keep],
                    "scores": cat_scores[keep],
                    "labels": cat_labels[keep],
                }
            )

        return results
