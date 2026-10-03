"""Visualization utilities for histopathology images and annotations.

Provides helpers for loading YOLO-format annotations and overlaying bounding boxes
onto matplotlib axes and images.
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional
import matplotlib.axes
import matplotlib.patches as patches
import numpy as np


def load_yolo_annotations(
    label_path: Union[str, Path],
    img_width: int,
    img_height: int,
) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO bounding boxes from a .txt file and convert to pixel coordinates.

    Args:
        label_path: Path to YOLO format annotation file. Each line should contain:
            <class_id> <x_center> <y_center> <width> <height>
            with normalized coordinates in [0, 1].
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of tuples: (class_id, x_min, y_min, box_width, box_height) in pixel units.
        Returns empty list if file does not exist or has no annotations.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    bboxes: List[Tuple[int, float, float, float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(float(parts[0]))
                    x_c, y_c, w, h = map(float, parts[1:5])
                    box_w = w * img_width
                    box_h = h * img_height
                    x_min = (x_c - w / 2.0) * img_width
                    y_min = (y_c - h / 2.0) * img_height
                    bboxes.append((cls_id, x_min, y_min, box_w, box_h))
                except (ValueError, IndexError):
                    continue

    return bboxes


def draw_yolo_bboxes(
    ax: matplotlib.axes.Axes,
    bboxes: List[Tuple[int, float, float, float, float]],
    edgecolor: str = "#00FF00",
    linewidth: float = 1.8,
    label: Optional[str] = None,
) -> None:
    """Draw bounding boxes onto a matplotlib Axes.

    Args:
        ax: Matplotlib axes to draw bounding boxes on.
        bboxes: List of (class_id, x_min, y_min, box_w, box_h) in pixel units.
        edgecolor: Bounding box line color (default: bright lime green).
        linewidth: Bounding box line width.
        label: Optional label for legend or title.
    """
    for idx, (_, x_min, y_min, box_w, box_h) in enumerate(bboxes):
        rect = patches.Rectangle(
            (x_min, y_min),
            box_w,
            box_h,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            label=label if idx == 0 else None,
        )
        ax.add_patch(rect)
