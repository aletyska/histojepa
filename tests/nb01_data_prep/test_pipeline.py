"""Unit tests for the DataPrepPipeline and prepare_image orchestrator.

Bound to: notebooks/01_Data_Preparation.py
Pytest Marker: @pytest.mark.nb01
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from histojepa.data.pipeline import (
    DataPrepPipeline,
    prepare_image,
)


pytestmark = pytest.mark.nb01


class TestDataPrepPipeline:
    """Test suite for DataPrepPipeline batch execution and idempotency."""

    def test_prepare_image_creates_file(
        self, tmp_path: Path, synthetic_histology_bgr: np.ndarray
    ) -> None:
        """Verify prepare_image creates processed output with correct dimensions."""
        src_path = tmp_path / "sample.png"
        dst_path = tmp_path / "processed" / "sample.png"
        cv2.imwrite(str(src_path), synthetic_histology_bgr)

        result = prepare_image(src_path, dst_path, overwrite=True)
        assert result.exists()
        assert result == dst_path

        out_img = cv2.imread(str(dst_path))
        assert out_img is not None
        assert out_img.shape == synthetic_histology_bgr.shape

    def test_prepare_image_idempotency(
        self, tmp_path: Path, synthetic_histology_bgr: np.ndarray
    ) -> None:
        """Verify re-running prepare_image overwrites existing file cleanly."""
        src_path = tmp_path / "sample.png"
        dst_path = tmp_path / "processed" / "sample.png"
        cv2.imwrite(str(src_path), synthetic_histology_bgr)

        # Run 1
        prepare_image(src_path, dst_path, overwrite=True)
        mtime1 = dst_path.stat().st_mtime_ns

        # Run 2: idempotent overwrite
        prepare_image(src_path, dst_path, overwrite=True)
        assert dst_path.exists()
        out_img = cv2.imread(str(dst_path))
        assert out_img is not None

    def test_get_inventory(self, temp_dataset_dir: Path) -> None:
        """Verify inventory counts across splits."""
        pipeline = DataPrepPipeline(data_root=temp_dataset_dir)
        inventory = pipeline.get_inventory()

        assert "train" in inventory
        assert "val" in inventory
        assert "test" in inventory
        assert inventory["train"]["raw_images"] == 2
        assert inventory["train"]["labels"] == 2
        assert inventory["train"]["processed"] == 0

    def test_process_single_file(self, temp_dataset_dir: Path) -> None:
        """Verify processing a single file in a split."""
        pipeline = DataPrepPipeline(data_root=temp_dataset_dir)
        dst = pipeline.process_single_file("train", "1.png", overwrite=True)

        assert dst.exists()
        assert dst.name == "1.png"
        assert dst.parent.name == "processed"

        # Check inventory updated
        inventory = pipeline.get_inventory()
        assert inventory["train"]["processed"] == 1

    def test_process_split(self, temp_dataset_dir: Path) -> None:
        """Verify processing all files in a single split."""
        pipeline = DataPrepPipeline(data_root=temp_dataset_dir)
        stats = pipeline.process_split("val", overwrite=True)

        assert stats.split == "val"
        assert stats.total_images == 2
        assert stats.processed_images == 2
        assert len(stats.failed_images) == 0
        assert stats.elapsed_time > 0

        inventory = pipeline.get_inventory()
        assert inventory["val"]["processed"] == 2

    def test_process_all(self, temp_dataset_dir: Path) -> None:
        """Verify processing all dataset splits."""
        pipeline = DataPrepPipeline(data_root=temp_dataset_dir)
        all_stats = pipeline.process_all(overwrite=True)

        for split in ("train", "val", "test"):
            assert all_stats[split].processed_images == 2
            assert len(all_stats[split].failed_images) == 0

        inventory = pipeline.get_inventory()
        for split in ("train", "val", "test"):
            assert inventory[split]["processed"] == 2

    def test_missing_input_raises_error(self, tmp_path: Path) -> None:
        """Verify missing source file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            prepare_image(tmp_path / "nonexistent.png", tmp_path / "out.png")

    def test_invalid_split_raises_error(self, temp_dataset_dir: Path) -> None:
        """Verify invalid split name raises ValueError."""
        pipeline = DataPrepPipeline(data_root=temp_dataset_dir)
        with pytest.raises(ValueError, match="Invalid split"):
            pipeline.get_split_dir("unknown_split")
