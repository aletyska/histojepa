"""Object Detection Evaluation Suite (IoU, AP, mAP@50, mAP@75, mAP@50:95, Precision, Recall, F1).

Implements pure PyTorch/NumPy COCO-style 101-point interpolated Average Precision
and F1-score calculation for Tumor Bud detection in Branch A.
Handles empty predictions and empty ground-truths gracefully without raising ZeroDivisionError
(in compliance with AGENTS.md Testing & Validation Expectations).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class DetectionMetricsReport:
    """Comprehensive evaluation report for object detection on a dataset split."""

    map50: float = 0.0
    map75: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    best_score_threshold: float = 0.25
    total_gt: int = 0
    total_pred: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    pr_curve_recall: List[float] = field(default_factory=list)
    pr_curve_precision: List[float] = field(default_factory=list)
    ap_per_iou: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_box_iou(
    boxes1: Union[torch.Tensor, np.ndarray],
    boxes2: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Computes pairwise IoU matrix between two sets of [x1, y1, x2, y2] boxes.

    Args:
        boxes1: Tensor or array of shape `(N, 4)`.
        boxes2: Tensor or array of shape `(M, 4)`.

    Returns:
        FloatTensor of shape `(N, M)` with pairwise IoU values in `[0, 1]`.
    """
    b1 = torch.as_tensor(boxes1, dtype=torch.float32)
    b2 = torch.as_tensor(boxes2, dtype=torch.float32)

    if b1.ndim == 1:
        b1 = b1.reshape(-1, 4)
    if b2.ndim == 1:
        b2 = b2.reshape(-1, 4)

    n = b1.shape[0]
    m = b2.shape[0]
    if n == 0 or m == 0:
        return torch.zeros((n, m), dtype=torch.float32, device=b1.device)

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0.0) * (b1[:, 3] - b1[:, 1]).clamp(min=0.0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0.0) * (b2[:, 3] - b2[:, 1]).clamp(min=0.0)

    lt = torch.maximum(b1[:, None, :2], b2[None, :, :2])  # (N, M, 2)
    rb = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter

    iou = torch.where(union > 0.0, inter / union.clamp(min=1e-7), torch.zeros_like(inter))
    return iou


def _compute_ap_at_iou(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes 101-point interpolated AP at a single IoU threshold across all images.

    Returns:
        Tuple of `(ap, precisions, recalls, sorted_scores)`.
    """
    total_gt = 0
    gt_by_img: Dict[int, torch.Tensor] = {}
    matched_by_img: Dict[int, np.ndarray] = {}

    for img_idx, tgt in enumerate(targets):
        boxes = torch.as_tensor(tgt.get("boxes", torch.zeros((0, 4))), dtype=torch.float32).cpu()
        if boxes.ndim == 1:
            boxes = boxes.reshape(-1, 4)
        gt_by_img[img_idx] = boxes
        matched_by_img[img_idx] = np.zeros((boxes.shape[0],), dtype=bool)
        total_gt += int(boxes.shape[0])

    all_preds: List[Tuple[float, int, torch.Tensor]] = []
    for img_idx, pred in enumerate(predictions):
        p_boxes = torch.as_tensor(pred.get("boxes", torch.zeros((0, 4))), dtype=torch.float32).cpu()
        p_scores = torch.as_tensor(pred.get("scores", torch.zeros((0,))), dtype=torch.float32).cpu()
        if p_boxes.ndim == 1:
            p_boxes = p_boxes.reshape(-1, 4)
        for k in range(p_boxes.shape[0]):
            score_val = float(p_scores[k].item()) if k < p_scores.numel() else 1.0
            all_preds.append((score_val, img_idx, p_boxes[k]))

    if total_gt == 0 or len(all_preds) == 0:
        return 0.0, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    # Sort predictions globally by descending confidence score
    all_preds.sort(key=lambda item: item[0], reverse=True)

    num_preds = len(all_preds)
    tp = np.zeros((num_preds,), dtype=np.float64)
    fp = np.zeros((num_preds,), dtype=np.float64)
    scores_arr = np.zeros((num_preds,), dtype=np.float64)

    for pred_idx, (score, img_idx, p_box) in enumerate(all_preds):
        scores_arr[pred_idx] = score
        gt_boxes = gt_by_img.get(img_idx, torch.zeros((0, 4)))
        if gt_boxes.shape[0] == 0:
            fp[pred_idx] = 1.0
            continue

        ious = compute_box_iou(p_box.unsqueeze(0), gt_boxes).squeeze(0).numpy()
        best_gt = int(np.argmax(ious))
        best_iou = float(ious[best_gt])

        if best_iou >= iou_threshold and not matched_by_img[img_idx][best_gt]:
            tp[pred_idx] = 1.0
            matched_by_img[img_idx][best_gt] = True
        else:
            fp[pred_idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # 101-point COCO interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap_sum = 0.0
    for r_thresh in recall_thresholds:
        inds = np.where(recalls >= r_thresh)[0]
        p_max = float(np.max(precisions[inds])) if inds.size > 0 else 0.0
        ap_sum += p_max
    ap = ap_sum / 101.0

    return float(ap), precisions, recalls, scores_arr


def evaluate_detections(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    iou_thresholds: Optional[Sequence[float]] = None,
) -> DetectionMetricsReport:
    """Evaluates predicted bounding boxes against ground-truth targets across a dataset split.

    Gracefully handles empty predictions (`len(predictions) == 0` or 0 boxes) and empty
    ground-truths without raising `ZeroDivisionError`.

    Args:
        predictions: List of dicts per image with `"boxes"` `(K, 4)` and `"scores"` `(K,)`.
        targets: List of dicts per image with `"boxes"` `(M, 4)`.
        iou_thresholds: Sequence of IoU thresholds (defaults to COCO `0.50:0.05:0.95`).

    Returns:
        DetectionMetricsReport with mAP@50, mAP@75, mAP@50:95, Precision, Recall, F1, and PR curve.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    total_gt = sum(
        int(torch.as_tensor(t.get("boxes", torch.zeros((0, 4)))).reshape(-1, 4).shape[0])
        for t in targets
    )
    total_pred = sum(
        int(torch.as_tensor(p.get("boxes", torch.zeros((0, 4)))).reshape(-1, 4).shape[0])
        for p in predictions
    )

    if len(predictions) == 0 or len(targets) == 0 or total_gt == 0 or total_pred == 0:
        return DetectionMetricsReport(
            map50=0.0,
            map75=0.0,
            map50_95=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            total_gt=total_gt,
            total_pred=total_pred,
            true_positives=0,
            false_positives=total_pred,
            false_negatives=total_gt,
        )

    ap_per_iou: Dict[str, float] = {}
    prec50 = np.zeros((0,), dtype=np.float64)
    rec50 = np.zeros((0,), dtype=np.float64)
    scores50 = np.zeros((0,), dtype=np.float64)

    for thresh in iou_thresholds:
        ap_val, p_curve, r_curve, s_curve = _compute_ap_at_iou(
            predictions, targets, iou_threshold=float(thresh)
        )
        key = f"{int(round(thresh * 100))}"
        ap_per_iou[key] = ap_val
        if abs(thresh - 0.50) < 1e-4:
            prec50 = p_curve
            rec50 = r_curve
            scores50 = s_curve

    if "50" not in ap_per_iou:
        ap50_val, prec50, rec50, scores50 = _compute_ap_at_iou(predictions, targets, iou_threshold=0.50)
        ap_per_iou["50"] = ap50_val

    if "75" not in ap_per_iou:
        ap75_val, _, _, _ = _compute_ap_at_iou(predictions, targets, iou_threshold=0.75)
        ap_per_iou["75"] = ap75_val

    map50 = float(ap_per_iou["50"])
    map75 = float(ap_per_iou["75"])
    map50_95 = float(np.mean([ap_per_iou[f"{int(round(t * 100))}"] for t in iou_thresholds]))

    # Find operating point on IoU=0.50 curve that maximizes F1
    if prec50.size > 0 and rec50.size > 0:
        denom = np.maximum(prec50 + rec50, 1e-12)
        f1_curve = 2.0 * prec50 * rec50 / denom
        best_idx = int(np.argmax(f1_curve))
        best_f1 = float(f1_curve[best_idx])
        best_prec = float(prec50[best_idx])
        best_rec = float(rec50[best_idx])
        best_score_thresh = float(scores50[best_idx])

        tp_count = int(round(best_rec * total_gt))
        fp_count = max(0, (best_idx + 1) - tp_count)
        fn_count = max(0, total_gt - tp_count)

        # Subsample PR curve to at most 200 points for lightweight JSON/plotting
        if rec50.size > 200:
            sample_idxs = np.linspace(0, rec50.size - 1, 200).astype(int)
            pr_rec = [float(x) for x in rec50[sample_idxs]]
            pr_prec = [float(x) for x in prec50[sample_idxs]]
        else:
            pr_rec = [float(x) for x in rec50]
            pr_prec = [float(x) for x in prec50]
    else:
        best_f1 = 0.0
        best_prec = 0.0
        best_rec = 0.0
        best_score_thresh = 0.25
        tp_count = 0
        fp_count = total_pred
        fn_count = total_gt
        pr_rec = []
        pr_prec = []

    return DetectionMetricsReport(
        map50=map50,
        map75=map75,
        map50_95=map50_95,
        precision=best_prec,
        recall=best_rec,
        f1=best_f1,
        best_score_threshold=best_score_thresh,
        total_gt=total_gt,
        total_pred=total_pred,
        true_positives=tp_count,
        false_positives=fp_count,
        false_negatives=fn_count,
        pr_curve_recall=pr_rec,
        pr_curve_precision=pr_prec,
        ap_per_iou=ap_per_iou,
    )
