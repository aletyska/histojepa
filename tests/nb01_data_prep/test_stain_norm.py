"""Unit tests for Reinhard stain normalization in Lab color space.

Bound to: notebooks/01_Data_Preparation.py
Pytest Marker: @pytest.mark.nb01
"""

import cv2
import numpy as np
import pytest

from histojepa.data.stain_norm import ReinhardNormalizer, normalize_stain


pytestmark = pytest.mark.nb01


class TestReinhardNormalizer:
    """Test suite for Reinhard color normalization."""

    def test_default_initialization(self) -> None:
        """Verify initialization with default CK reference statistics."""
        normalizer = ReinhardNormalizer()
        assert normalizer.target_mean is not None
        assert normalizer.target_std is not None
        assert len(normalizer.target_mean) == 3
        assert len(normalizer.target_std) == 3
        assert np.all(normalizer.target_std > 0)

    def test_fit_from_reference_image(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify fitting target statistics from a reference image."""
        normalizer = ReinhardNormalizer()
        normalizer.fit(synthetic_histology_bgr)

        assert normalizer.target_mean.shape == (3,)
        assert normalizer.target_std.shape == (3,)
        assert np.all(normalizer.target_std > 0)

    def test_fit_from_stats(self) -> None:
        """Verify explicit setting of target statistics."""
        target_m = np.array([100.0, 140.0, 130.0], dtype=np.float32)
        target_s = np.array([25.0, 5.0, 10.0], dtype=np.float32)
        normalizer = ReinhardNormalizer(target_mean=target_m, target_std=target_s)

        assert np.allclose(normalizer.target_mean, target_m)
        assert np.allclose(normalizer.target_std, target_s)

    def test_transform_shape_and_dtype(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify normalized image shape, dtype, and valid pixel value range."""
        normalizer = ReinhardNormalizer()
        normalized = normalizer.transform(synthetic_histology_bgr)

        assert normalized.shape == synthetic_histology_bgr.shape
        assert normalized.dtype == np.uint8
        assert normalized.min() >= 0
        assert normalized.max() <= 255

    def test_transform_with_mask_preserves_background(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify that background pixels remain intact when mask is provided."""
        # Create a mock mask: center is tissue (255), outer border is background (0)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        bg_color = (255, 255, 255)
        normalizer = ReinhardNormalizer()
        normalized = normalizer.transform(
            synthetic_histology_bgr, mask=mask, restore_background=True, bg_color=bg_color
        )

        # Background pixels must match bg_color
        bg_pixels = normalized[mask == 0]
        assert np.all(bg_pixels == bg_color)

    def test_color_transfer_shifts_distribution(self) -> None:
        """Verify that color transfer shifts the source image statistics toward target."""
        # Source image with a distinct dark tint
        src = np.full((100, 100, 3), 50, dtype=np.uint8)
        # Reference image with a brighter tint
        ref = np.full((100, 100, 3), 180, dtype=np.uint8)

        normalizer = ReinhardNormalizer()
        normalizer.fit(ref)
        trans = normalizer.transform(src)

        # The normalized image mean should be significantly closer to ref than src
        assert trans.mean() > src.mean()

    def test_convenience_function(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify normalize_stain convenience function."""
        result = normalize_stain(synthetic_histology_bgr)
        assert result.shape == synthetic_histology_bgr.shape
        assert result.dtype == np.uint8

    def test_invalid_input(self) -> None:
        """Verify error on invalid input dimensions."""
        normalizer = ReinhardNormalizer()
        with pytest.raises(ValueError):
            normalizer.transform(np.zeros((100, 100), dtype=np.uint8))
