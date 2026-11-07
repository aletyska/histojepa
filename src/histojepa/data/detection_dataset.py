"""PyTorch Dataset and Collate utilities for TB-YOLO Object Detection (Branch A).

Loads preprocessed histopathology tiles from data/{split}/processed/ and pairs them
with ground-truth bounding box annotations in YOLO format from data/{split}/labels/.
Strictly respects patient-level splits (train, val, test) per AGENTS.md Hard Constraint #2.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def yolo_to_xyxy(
    boxes_yolo: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Converts normalized YOLO [x_center, y_center, w, h] boxes to pixel [x1, y1, x2, y2].

    Args:
        boxes_yolo: Array of shape (N, 4) in normalized [0, 1] coordinates.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Array of shape (N, 4) in pixel [x1, y1, x2, y2] coordinates, clipped to image bounds.
    """
    if boxes_yolo.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x_c = boxes_yolo[:, 0] * img_width
    y_c = boxes_yolo[:, 1] * img_height
    w = boxes_yolo[:, 2] * img_width
    h = boxes_yolo[:, 3] * img_height

    x1 = np.clip(x_c - 0.5 * w, 0.0, float(img_width))
    y1 = np.clip(y_c - 0.5 * h, 0.0, float(img_height))
    x2 = np.clip(x_c + 0.5 * w, 0.0, float(img_width))
    y2 = np.clip(y_c + 0.5 * h, 0.0, float(img_height))

    xyxy = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    # Filter degenerate boxes where width or height <= 0.5 px
    valid = (xyxy[:, 2] - xyxy[:, 0] > 0.5) & (xyxy[:, 3] - xyxy[:, 1] > 0.5)
    return xyxy[valid]


def load_yolo_boxes_and_labels(
    label_path: Union[str, Path],
    img_width: int = 1280,
    img_height: int = 1280,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a YOLO .txt label file and returns (boxes_xyxy, class_labels).

    Args:
        label_path: Path to .txt file with lines '<class> <xc> <yc> <w> <h>'.
        img_width: Target image width in pixels.
        img_height: Target image height in pixels.

    Returns:
        Tuple of:
            - boxes_xyxy: float32 array of shape (N, 4) in [x1, y1, x2, y2] pixel coords.
            - labels: int64 array of shape (N,) with 0-indexed class IDs.
    """
    path = Path(label_path)
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    raw_boxes: List[List[float]] = []
    raw_labels: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    raw_boxes.append([xc, yc, w, h])
                    raw_labels.append(cls_id)
                except (ValueError, IndexError):
                    continue

    if not raw_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes_yolo = np.array(raw_boxes, dtype=np.float32)
    labels_arr = np.array(raw_labels, dtype=np.int64)

    x_c = boxes_yolo[:, 0] * img_width
    y_c = boxes_yolo[:, 1] * img_height
    w = boxes_yolo[:, 2] * img_width
    h = boxes_yolo[:, 3] * img_height

    x1 = np.clip(x_c - 0.5 * w, 0.0, float(img_width))
    y1 = np.clip(y_c - 0.5 * h, 0.0, float(img_height))
    x2 = np.clip(x_c + 0.5 * w, 0.0, float(img_width))
    y2 = np.clip(y_c + 0.5 * h, 0.0, float(img_height))

    xyxy = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    valid = (xyxy[:, 2] - xyxy[:, 0] > 0.5) & (xyxy[:, 3] - xyxy[:, 1] > 0.5)
    return xyxy[valid], labels_arr[valid]


class TBYoloDetectionDataset(Dataset):
    """Dataset pairing preprocessed histopathology tiles with YOLO bounding box annotations.

    Prefers images from `data/{split}/processed/` (falling back to `data/{split}/images/`
    if `processed/` is empty in synthetic unit test fixtures).

    Args:
        data_root: Root directory containing train/val/test folders.
        split: Partition name ('train', 'val', or 'test').
        image_size: Target spatial resolution (default: 1280).
        mean: RGB normalization mean.
        std: RGB normalization std.
        augment: Whether to apply random dihedral augmentations (flips) with synchronized boxes.
        return_uint8: If True, returns unnormalized uint8 CHW tensors for fast GPU normalization.
        seed: Random seed for reproducible augmentations.
    """

    def __init__(
        self,
        data_root: Union[str, Path] = "data",
        split: str = "train",
        image_size: int = 1280,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        augment: bool = False,
        return_uint8: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'. Must be one of ('train', 'val', 'test').")

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.augment = bool(augment)
        self.return_uint8 = bool(return_uint8)
        self.rng = np.random.default_rng(seed)

        split_dir = self.data_root / self.split
        proc_dir = split_dir / "processed"
        raw_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"

        supported_exts = {".png", ".jpg", ".jpeg"}
        proc_imgs = (
            sorted([p for p in proc_dir.iterdir() if p.suffix.lower() in supported_exts])
            if proc_dir.exists()
            else []
        )

        if len(proc_imgs) > 0:
            self.image_dir = proc_dir
            self.image_paths = proc_imgs
        elif raw_dir.exists():
            raw_imgs = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in supported_exts])
            if len(raw_imgs) == 0:
                raise ValueError(f"No images found in {proc_dir} or {raw_dir}.")
            self.image_dir = raw_dir
            self.image_paths = raw_imgs
        else:
            raise FileNotFoundError(
                f"Neither processed ({proc_dir}) nor raw ({raw_dir}) image directories exist."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(
        self,
        rgb: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies histopathology-safe random horizontal and vertical flips."""
        h, w = rgb.shape[:2]
        boxes = boxes.copy()

        # Random Horizontal Flip (p=0.5)
        if self.rng.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            if len(boxes) > 0:
                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - old_x2
                boxes[:, 2] = w - old_x1

        # Random Vertical Flip (p=0.5)
        if self.rng.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[::-1, :, :])
            if len(boxes) > 0:
                old_y1 = boxes[:, 1].copy()
                old_y2 = boxes[:, 3].copy()
                boxes[:, 1] = h - old_y2
                boxes[:, 3] = h - old_y1

        return rgb, boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Loads image tensor (3, H, W) and target annotation dictionary."""
        img_path = self.image_paths[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read image at {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if h != self.image_size or w != self.image_size:
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA if (h > self.image_size or w > self.image_size) else cv2.INTER_LINEAR,
            )

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes_xyxy, labels = load_yolo_boxes_and_labels(
            label_path,
            img_width=self.image_size,
            img_height=self.image_size,
        )

        if self.augment:
            rgb, boxes_xyxy = self._apply_augmentations(rgb, boxes_xyxy)

        if self.return_uint8:
            img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        else:
            norm_rgb = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
            img_tensor = torch.from_numpy(norm_rgb).permute(2, 0, 1).contiguous()

        target = {
            "boxes": torch.from_numpy(boxes_xyxy).to(dtype=torch.float32),
            "labels": torch.from_numpy(labels).to(dtype=torch.long),
            "image_id": idx,
            "image_name": img_path.name,
        }
        return img_tensor, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collates variable-length bounding box targets alongside stacked image batch."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
