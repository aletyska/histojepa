"""Unit tests for detection evaluation metrics (IoU, mAP@50, mAP@50:95, F1)
and empty prediction handling.

Bound to: notebooks/03_Phase_2_Branch_A.py
Pytest Marker: @pytest.mark.nb03
"""

import pytest
import torch

from histojepa.evaluation.detection import compute_box_iou, evaluate_detections

pytestmark = pytest.mark.nb03


def test_compute_box_iou():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
    iou = compute_box_iou(b1, b2)

    assert iou.shape == (2, 2)
    assert pytest.approx(float(iou[0, 0].item()), rel=1e-4) == 1.0
    # Overlap is 5x10=50, union is 100+100-50=150 -> IoU = 1/3
    assert pytest.approx(float(iou[0, 1].item()), rel=1e-4) == 1.0 / 3.0
    assert pytest.approx(float(iou[1, 0].item()), abs=1e-6) == 0.0


def test_evaluate_detections_empty_predictions_no_zero_division():
    """Verifies AGENTS.md expectation: evaluation metrics handle empty bounding box
    predictions gracefully without raising ZeroDivisionError.
    """
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)},
    ]
    empty_preds = [
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)},
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)},
    ]

    report = evaluate_detections(empty_preds, targets)
    assert report.map50 == 0.0
    assert report.map50_95 == 0.0
    assert report.f1 == 0.0
    assert report.total_gt == 1
    assert report.total_pred == 0
    assert report.false_negatives == 1


def test_evaluate_detections_perfect_and_partial_match():
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0], [100.0, 100.0, 150.0, 150.0]]),
            "labels": torch.tensor([0, 0]),
        }
    ]
    perfect_preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0], [100.0, 100.0, 150.0, 150.0]]),
            "scores": torch.tensor([0.95, 0.90]),
            "labels": torch.tensor([0, 0]),
        }
    ]

    report = evaluate_detections(perfect_preds, targets)
    assert pytest.approx(report.map50, rel=1e-3) == 1.0
    assert pytest.approx(report.map50_95, rel=1e-3) == 1.0
    assert pytest.approx(report.f1, rel=1e-3) == 1.0
    assert report.true_positives == 2
    assert report.false_positives == 0
    assert report.false_negatives == 0
