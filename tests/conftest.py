"""Shared pytest fixtures for HistoJEPA unit tests."""

from pathlib import Path
import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_histology_bgr() -> np.ndarray:
    """Creates a synthetic histology image in BGR format.

    Has a bright glass background (values 245-255) and a central darker stained
    tissue region (values 70-130) simulating Cytokeratin / Hematoxylin staining.
    Shape: (200, 200, 3), dtype: uint8.
    """
    img = np.full((200, 200, 3), 250, dtype=np.uint8)

    # Add subtle background noise
    noise = np.random.randint(-5, 5, (200, 200, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add central tissue block with brownish/purplish stain (BGR)
    # Tissue center: [50:150, 50:150]
    img[50:150, 50:150, 0] = 120  # Blue channel
    img[50:150, 50:150, 1] = 90   # Green channel
    img[50:150, 50:150, 2] = 150  # Red channel (brownish-pink)

    return img


@pytest.fixture
def temp_dataset_dir(tmp_path: Path, synthetic_histology_bgr: np.ndarray) -> Path:
    """Creates a mock dataset directory structure mimicking data/{split}/."""
    root = tmp_path / "mock_data"
    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        proc_dir = root / split / "processed"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)

        # Write 2 synthetic sample images and labels per split
        for i in (1, 2):
            img_path = img_dir / f"{i}.png"
            cv2.imwrite(str(img_path), synthetic_histology_bgr)
            lbl_path = lbl_dir / f"{i}.txt"
            lbl_path.write_text("0 0.5 0.5 0.5 0.5\n")

    return root
