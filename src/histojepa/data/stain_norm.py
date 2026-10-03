"""Stain normalization using the Reinhard method in OpenCV Lab color space.

Reinhard color normalization standardizes the color distribution of histopathology
slides (specifically Cytokeratin / CK stained slides) by matching the channel means
and standard deviations in a decorrelated color space (Lab) against a reference slide.

Reference:
    E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley,
    "Color Transfer between Images", IEEE Computer Graphics and Applications, 2001.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


# Precomputed default reference statistics for CK (Cytokeratin) colorectal slides
# Derived from representative high-contrast reference slide data/train/images/1.png
DEFAULT_CK_LAB_MEAN = np.array([92.45, 138.94, 129.54], dtype=np.float32)
DEFAULT_CK_LAB_STD = np.array([26.68, 4.75, 15.00], dtype=np.float32)


class ReinhardNormalizer:
    """Reinhard color normalization for histopathology images.

    Transfers color characteristics from a target reference slide onto source slides
    by matching channel means and standard deviations in OpenCV CIELAB color space.

    Attributes:
        target_mean: Array of shape (3,) representing target [L, a, b] means.
        target_std: Array of shape (3,) representing target [L, a, b] standard deviations.
    """

    def __init__(
        self,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
    ) -> None:
        if target_mean is not None and target_std is not None:
            self.fit_from_stats(target_mean, target_std)
        else:
            self.target_mean = DEFAULT_CK_LAB_MEAN.copy()
            self.target_std = DEFAULT_CK_LAB_STD.copy()

    def fit_from_stats(self, target_mean: np.ndarray, target_std: np.ndarray) -> "ReinhardNormalizer":
        """Set normalization target parameters directly from channel statistics.

        Args:
            target_mean: Target channel mean array of shape (3,) for Lab space.
            target_std: Target channel std array of shape (3,) for Lab space.

        Returns:
            self for method chaining.
        """
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.target_std = np.maximum(self.target_std, 1e-6)
        return self

    def fit(
        self,
        ref_image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> "ReinhardNormalizer":
        """Fit target statistics from a reference BGR image.

        Args:
            ref_image_bgr: Reference histology image in BGR format (H, W, 3), uint8.
            mask: Optional binary mask of shape (H, W). If provided, statistics are
                computed exclusively on tissue pixels (mask > 0).

        Returns:
            self for method chaining.
        """
        if not isinstance(ref_image_bgr, np.ndarray) or ref_image_bgr.ndim != 3:
            raise ValueError("ref_image_bgr must be a 3-channel numpy array")

        lab = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

        if mask is not None and np.any(mask > 0):
            pixels = lab[mask > 0]
        else:
            pixels = lab.reshape(-1, 3)

        self.target_mean = np.mean(pixels, axis=0).astype(np.float32)
        self.target_std = np.std(pixels, axis=0).astype(np.float32)
        self.target_std = np.maximum(self.target_std, 1e-6)
        return self

    def transform(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        restore_background: bool = True,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Apply Reinhard color normalization to a source image.

        Args:
            image_bgr: Source histology image in BGR format (H, W, 3), uint8.
            mask: Optional binary mask of shape (H, W). If provided, source statistics
                are computed exclusively on tissue pixels (mask > 0).
            restore_background: If True and mask is provided, sets background pixels
                (mask == 0) to bg_color so background glass stays clean.
            bg_color: BGR tuple for background restoration (default pure white).

        Returns:
            Normalized image in BGR format (H, W, 3), uint8.
        """
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            raise ValueError("image_bgr must be a 3-channel numpy array")

        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("ReinhardNormalizer has not been fitted with target statistics.")

        # Convert source image to Lab space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

        # Extract source statistics (tissue pixels only if mask provided)
        if mask is not None and np.any(mask > 0):
            source_pixels = lab[mask > 0]
        else:
            source_pixels = lab.reshape(-1, 3)

        src_mean = np.mean(source_pixels, axis=0).astype(np.float32)
        src_std = np.std(source_pixels, axis=0).astype(np.float32)
        src_std = np.maximum(src_std, 1e-6)

        # Affine channel-wise transfer: (I_src - mu_src) * (std_target / std_src) + mu_target
        scale = self.target_std / src_std
        norm_lab = (lab - src_mean) * scale + self.target_mean

        # Clip to valid 8-bit Lab range [0, 255] and convert back to BGR
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
        norm_bgr = cv2.cvtColor(norm_lab, cv2.COLOR_Lab2BGR)

        # If background restoration is requested, ensure non-tissue regions remain clean
        if restore_background and mask is not None:
            norm_bgr[mask == 0] = bg_color

        return norm_bgr


def normalize_stain(
    image_bgr: np.ndarray,
    ref_image_bgr: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    restore_background: bool = True,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Convenience function for Reinhard color normalization.

    Args:
        image_bgr: Source BGR image.
        ref_image_bgr: Optional reference BGR image. If None, uses default CK statistics.
        mask: Optional binary tissue mask.
        restore_background: Whether to restore clean background color on non-tissue pixels.
        bg_color: Replacement background BGR color.

    Returns:
        Normalized BGR image.
    """
    normalizer = ReinhardNormalizer()
    if ref_image_bgr is not None:
        normalizer.fit(ref_image_bgr)
    return normalizer.transform(
        image_bgr, mask=mask, restore_background=restore_background, bg_color=bg_color
    )
