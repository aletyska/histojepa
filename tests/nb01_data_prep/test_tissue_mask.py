"""Unit tests for tissue segmentation and Otsu thresholding.

Bound to: notebooks/01_Data_Preparation.py
Pytest Marker: @pytest.mark.nb01
"""

import numpy as np
import pytest

from histojepa.data.tissue_mask import OtsuSegmenter, segment_and_clean


pytestmark = pytest.mark.nb01


class TestOtsuSegmenter:
    """Test suite for Otsu automated thresholding and background noise removal."""

    def test_compute_mask_shape_and_dtype(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify binary mask dimensions, dtype, and valid binary values."""
        segmenter = OtsuSegmenter(blur_kernel_size=5, morph_kernel_size=3)
        mask = segmenter.compute_mask(synthetic_histology_bgr)

        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})
        assert 255 in unique_vals  # tissue present
        assert 0 in unique_vals    # background present

    def test_tissue_localization(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify that the stained tissue block is correctly detected as foreground."""
        segmenter = OtsuSegmenter(blur_kernel_size=5, morph_kernel_size=3)
        mask = segmenter.compute_mask(synthetic_histology_bgr)

        # Center [60:140, 60:140] should be classified as tissue (255)
        tissue_region = mask[60:140, 60:140]
        assert np.mean(tissue_region == 255) > 0.95

        # Border [0:40, 0:40] should be classified as glass background (0)
        background_region = mask[0:40, 0:40]
        assert np.mean(background_region == 0) > 0.95

    def test_remove_background_noise(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify that background noise pixels are cleanly replaced with the target background color."""
        custom_bg = (240, 240, 240)
        segmenter = OtsuSegmenter(default_bg_color=custom_bg)
        cleaned, mask = segmenter.remove_background_noise(synthetic_histology_bgr)

        assert cleaned.shape == synthetic_histology_bgr.shape
        assert cleaned.dtype == np.uint8

        # Background regions must match custom_bg
        bg_pixels = cleaned[mask == 0]
        assert np.all(bg_pixels == custom_bg)

        # Foreground tissue regions must retain original pixel values
        fg_pixels_cleaned = cleaned[mask == 255]
        fg_pixels_original = synthetic_histology_bgr[mask == 255]
        assert np.array_equal(fg_pixels_cleaned, fg_pixels_original)

    def test_convenience_function(self, synthetic_histology_bgr: np.ndarray) -> None:
        """Verify the top-level segment_and_clean convenience function."""
        cleaned, mask = segment_and_clean(synthetic_histology_bgr, bg_color=(255, 255, 255))
        assert cleaned.shape == synthetic_histology_bgr.shape
        assert mask.shape == (200, 200)

    def test_invalid_parameters(self) -> None:
        """Verify that invalid arguments raise appropriate exceptions."""
        with pytest.raises(ValueError, match="must be odd"):
            OtsuSegmenter(blur_kernel_size=4)

        segmenter = OtsuSegmenter()
        with pytest.raises(TypeError):
            segmenter.compute_mask("not_an_array")  # type: ignore

        with pytest.raises(ValueError, match="Expected 2D or 3-channel"):
            segmenter.compute_mask(np.zeros((10, 10, 5), dtype=np.uint8))
