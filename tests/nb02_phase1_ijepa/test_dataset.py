"""Unit tests for ProcessedImageDataset.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from histojepa.data.ijepa_dataset import ProcessedImageDataset


pytestmark = pytest.mark.nb02


def test_processed_image_dataset_loading_and_resizing(tmp_path: Path):
    split_dir = tmp_path / "mock_data" / "train" / "processed"
    split_dir.mkdir(parents=True)

    # Write 2 synthetic images: one 128x128, one 64x64 (to test resizing)
    img1 = np.full((128, 128, 3), 120, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(split_dir / "1.png"), img1)
    cv2.imwrite(str(split_dir / "2.png"), img2)

    ds = ProcessedImageDataset(
        data_root=tmp_path / "mock_data",
        split="train",
        image_size=128,
    )

    assert len(ds) == 2
    item0 = ds[0]
    assert isinstance(item0, torch.Tensor)
    assert item0.shape == (3, 128, 128)

    item1 = ds[1]
    assert item1.shape == (3, 128, 128)  # resized from 64 to 128


def test_processed_image_dataset_error_handling(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ProcessedImageDataset(data_root=tmp_path / "nonexistent", split="train")

    empty_dir = tmp_path / "empty_data" / "train" / "processed"
    empty_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        ProcessedImageDataset(data_root=tmp_path / "empty_data", split="train")
