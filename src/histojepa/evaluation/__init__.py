"""Evaluation metrics for HistoJEPA (Object Detection & Unsupervised Clustering)."""

from histojepa.evaluation.detection import (
    DetectionMetricsReport,
    compute_box_iou,
    evaluate_detections,
)

__all__ = [
    "DetectionMetricsReport",
    "compute_box_iou",
    "evaluate_detections",
]
