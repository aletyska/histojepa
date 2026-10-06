"""Visualization utilities for histopathology images, annotations, and I-JEPA masking.

Provides helpers for loading YOLO-format annotations, overlaying bounding boxes,
and visualizing I-JEPA multi-block context and target masks.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.patches as patches
import numpy as np
import torch


def load_yolo_annotations(
    label_path: Union[str, Path],
    img_width: int,
    img_height: int,
) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO bounding boxes from a .txt file and convert to pixel coordinates.

    Args:
        label_path: Path to YOLO format annotation file.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of tuples: (class_id, x_min, y_min, box_width, box_height) in pixel units.
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
    """Draw bounding boxes onto a matplotlib Axes."""
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


def draw_ijepa_masks(
    ax: matplotlib.axes.Axes,
    image_rgb: np.ndarray,
    context_idx: Union[Sequence[int], torch.Tensor],
    target_idx_list: Sequence[Union[Sequence[int], torch.Tensor]],
    grid_size: Tuple[int, int] = (80, 80),
    patch_size: int = 16,
) -> None:
    """Renders I-JEPA context and multi-block target masks overlaid onto an RGB image.

    Context patches are highlighted with a transparent blue wash.
    Each of the M target blocks is delineated with high-contrast color boundaries.

    Args:
        ax: Matplotlib axes to render onto.
        image_rgb: RGB image array of shape (H, W, 3).
        context_idx: Indices of context patches.
        target_idx_list: List of M target block patch index collections.
        grid_size: (H_patches, W_patches) spatial grid dimensions.
        patch_size: Pixel dimensions of each patch square (default: 16).
    """
    if isinstance(context_idx, torch.Tensor):
        context_idx = context_idx.cpu().tolist()

    H_g, W_g = grid_size
    img_h, img_w = image_rgb.shape[:2]

    # Show underlying histology image
    ax.imshow(image_rgb)

    # 1. Shade context patches with a subtle cyan/blue overlay
    ctx_set = set(context_idx)
    ctx_mask = np.zeros((H_g, W_g), dtype=bool)
    for idx in ctx_set:
        r = idx // W_g
        c = idx % W_g
        ctx_mask[r, c] = True

    # Build RGBA mask overlay
    overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    for r in range(H_g):
        for c in range(W_g):
            if ctx_mask[r, c]:
                y0, y1 = r * patch_size, (r + 1) * patch_size
                x0, x1 = c * patch_size, (c + 1) * patch_size
                overlay[y0:y1, x0:x1] = [0.0, 0.45, 0.85, 0.28]  # Translucent blue

    ax.imshow(overlay)

    # 2. Outline each of the M target blocks with vibrant distinguishable colors
    target_palette = ["#FF3366", "#FF9900", "#9900FF", "#00CC66", "#FFD700"]
    for m_idx, tgt_indices in enumerate(target_idx_list):
        if isinstance(tgt_indices, torch.Tensor):
            tgt_indices = tgt_indices.cpu().tolist()

        color = target_palette[m_idx % len(target_palette)]

        # Determine bounding box of target patches
        rows = [idx // W_g for idx in tgt_indices]
        cols = [idx % W_g for idx in tgt_indices]
        if not rows or not cols:
            continue

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        x_min = min_c * patch_size
        y_min = min_r * patch_size
        box_w = (max_c - min_c + 1) * patch_size
        box_h = (max_r - min_r + 1) * patch_size

        rect = patches.Rectangle(
            (x_min, y_min),
            box_w,
            box_h,
            linewidth=2.2,
            edgecolor=color,
            facecolor="none",
            label=f"Target {m_idx + 1} ({len(tgt_indices)} patches)",
        )
        ax.add_patch(rect)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")
