"""PyTorch Dataset for I-JEPA self-supervised pre-training.

Loads pre-processed histopathology tiles directly from data/{split}/processed/
without hand-crafted view augmentations (in accordance with I-JEPA §1 / §10).
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedImageDataset(Dataset):
    """Dataset of unlabelled histopathology tiles from data/{split}/processed/.

    Args:
        data_root: Root dataset directory containing split folders (default: "data").
        split: Split partition to load, e.g. "train" (default: "train").
        image_size: Expected image width and height in pixels (default: 1280).
        mean: Normalization RGB mean channels (default: ImageNet [0.485, 0.456, 0.406]).
        std: Normalization RGB standard deviations (default: ImageNet [0.229, 0.224, 0.225]).
    """

    def __init__(
        self,
        data_root: Union[str, Path] = "data",
        split: str = "train",
        image_size: int = 1280,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        self.processed_dir = self.data_root / self.split / "processed"
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed directory does not exist: {self.processed_dir}. "
                "Ensure DataPrepPipeline has been run for this split."
            )

        # Collect supported image files (.png, .jpg, .jpeg)
        supported_exts = {".png", ".jpg", ".jpeg"}
        self.image_paths: List[Path] = sorted(
            [p for p in self.processed_dir.iterdir() if p.suffix.lower() in supported_exts]
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No processed image files found in {self.processed_dir}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Loads and normalizes an image.

        Returns:
            torch.FloatTensor of shape (3, image_size, image_size) in RGB format.
        """
        img_path = self.image_paths[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Failed to load image at: {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Resize if physical dimensions deviate from target size
        h, w = rgb.shape[:2]
        if h != self.image_size or w != self.image_size:
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA if (h > self.image_size or w > self.image_size) else cv2.INTER_LINEAR,
            )

        # Convert to float32 in [0, 1] and apply channel standardisation
        norm_rgb = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std

        # Convert HWC -> CHW PyTorch tensor
        tensor_chw = torch.from_numpy(norm_rgb).permute(2, 0, 1).contiguous()
        return tensor_chw
