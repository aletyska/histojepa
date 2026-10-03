"""Tissue segmentation and background noise removal using Otsu automated thresholding.

This module provides tissue segmentation for histopathology Whole Slide Images (WSIs)
and patches using Otsu's thresholding in OpenCV. In brightfield transmission microscopy,
glass background areas are bright/white and tissue structures are darker.
Otsu's threshold automatically determines the bimodal cutoff to separate foreground
tissue from background noise and glass artifacts.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class OtsuSegmenter:
    """Otsu automated thresholding for histology tissue segmentation.

    Attributes:
        blur_kernel_size: Odd integer kernel size for Gaussian smoothing to reduce
            high-frequency sensor noise prior to thresholding. Set to 0 to disable.
        morph_kernel_size: Kernel size for morphological opening and closing to remove
            isolated dust specks and fill tiny tissue lacunae. Set to 0 to disable.
        default_bg_color: BGR tuple used to replace background noise (default pure white).
    """

    def __init__(
        self,
        blur_kernel_size: int = 5,
        morph_kernel_size: int = 3,
        default_bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if blur_kernel_size > 0 and blur_kernel_size % 2 == 0:
            raise ValueError(f"blur_kernel_size must be odd or 0, got {blur_kernel_size}")
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.default_bg_color = default_bg_color

    def compute_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute binary tissue mask using Otsu thresholding.

        Args:
            image_bgr: Input image in BGR format (H, W, 3) or grayscale (H, W), dtype uint8.

        Returns:
            Binary mask of shape (H, W), dtype uint8, where 255 indicates foreground tissue
            and 0 indicates background glass / noise.
        """
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy ndarray")

        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif image_bgr.ndim == 2:
            gray = image_bgr.copy()
        else:
            raise ValueError(f"Expected 2D or 3-channel image, got shape {image_bgr.shape}")

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply Gaussian blur to suppress fine scanner sensor noise
        if self.blur_kernel_size > 1:
            blurred = cv2.GaussianBlur(
                gray, (self.blur_kernel_size, self.blur_kernel_size), 0
            )
        else:
            blurred = gray

        # Otsu thresholding:
        # In histology, tissue is darker and slide background is bright.
        # THRESH_BINARY_INV sets pixels < threshold (tissue) to 255, and glass to 0.
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological refinement
        if self.morph_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
            )
            # Remove isolated single-pixel dust in the background
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # Fill small pinhole gaps inside tissue
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def remove_background_noise(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bg_color: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the tissue mask to clean background noise.

        Pixels classified as background (mask == 0) are replaced by the specified
        background color (default: uniform white). Foreground tissue pixels (mask == 255)
        remain unmodified.

        Args:
            image_bgr: Input image in BGR format (H, W, 3), dtype uint8.
            mask: Optional precomputed binary mask. If None, it will be computed automatically.
            bg_color: BGR tuple for background replacement. Defaults to self.default_bg_color.

        Returns:
            Tuple of (cleaned_image_bgr, mask).
        """
        if mask is None:
            mask = self.compute_mask(image_bgr)

        if bg_color is None:
            bg_color = self.default_bg_color

        cleaned = image_bgr.copy()
        if cleaned.ndim == 3 and cleaned.shape[2] == 3:
            # Set background pixels to bg_color
            bg_indices = mask == 0
            cleaned[bg_indices] = bg_color
        elif cleaned.ndim == 2:
            cleaned[mask == 0] = bg_color[0]

        return cleaned, mask


def segment_and_clean(
    image_bgr: np.ndarray,
    blur_kernel_size: int = 5,
    morph_kernel_size: int = 3,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to segment tissue and remove background noise.

    Args:
        image_bgr: Input BGR image.
        blur_kernel_size: Gaussian blur kernel size.
        morph_kernel_size: Morphological structuring element size.
        bg_color: Replacement background BGR color.

    Returns:
        Tuple of (cleaned_image_bgr, binary_mask).
    """
    segmenter = OtsuSegmenter(
        blur_kernel_size=blur_kernel_size,
        morph_kernel_size=morph_kernel_size,
        default_bg_color=bg_color,
    )
    return segmenter.remove_background_noise(image_bgr, bg_color=bg_color)
