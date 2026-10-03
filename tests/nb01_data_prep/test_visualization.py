"""Unit tests for visualization helpers and YOLO bounding box parsing.

Bound to: notebooks/01_Data_Preparation.py
Pytest Marker: @pytest.mark.nb01
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pytest

from histojepa.utils.visualization import load_yolo_annotations, draw_yolo_bboxes


pytestmark = pytest.mark.nb01


class TestVisualizationUtils:
    """Test suite for YOLO annotation loading and drawing."""

    def test_load_yolo_annotations(self, tmp_path: Path) -> None:
        """Verify parsing normalized YOLO text files into pixel coordinates."""
        label_file = tmp_path / "sample.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.1\n0 0.25 0.75 0.1 0.2\n")

        # Given 1000x500 image (W=1000, H=500)
        bboxes = load_yolo_annotations(label_file, img_width=1000, img_height=500)
        assert len(bboxes) == 2

        # First box: center (500, 250), size (200, 50) -> xmin=400, ymin=225
        cls1, xmin1, ymin1, w1, h1 = bboxes[0]
        assert cls1 == 0
        assert xmin1 == pytest.approx(400.0)
        assert ymin1 == pytest.approx(225.0)
        assert w1 == pytest.approx(200.0)
        assert h1 == pytest.approx(50.0)

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Verify missing label file returns empty list gracefully."""
        bboxes = load_yolo_annotations(tmp_path / "nonexistent.txt", 100, 100)
        assert bboxes == []

    def test_draw_yolo_bboxes(self) -> None:
        """Verify drawing bounding box patches on a matplotlib axes."""
        fig, ax = plt.subplots()
        bboxes = [(0, 10.0, 20.0, 30.0, 40.0)]
        draw_yolo_bboxes(ax, bboxes, edgecolor="#00FF00", linewidth=2.0)

        # Matplotlib axes patches should contain the added rectangle
        assert len(ax.patches) == 1
        rect = ax.patches[0]
        assert rect.get_x() == 10.0
        assert rect.get_y() == 20.0
        assert rect.get_width() == 30.0
        assert rect.get_height() == 40.0
        plt.close(fig)
