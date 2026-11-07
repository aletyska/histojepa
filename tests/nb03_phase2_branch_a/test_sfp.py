"""Unit tests for ViTDet Simple Feature Pyramid (SFP).

Bound to: notebooks/03_Phase_2_Branch_A.py
Pytest Marker: @pytest.mark.nb03
"""

import pytest
import torch

from histojepa.models.vitdet.sfp import SimpleFeaturePyramid

pytestmark = pytest.mark.nb03


@pytest.mark.parametrize(
    "image_size,grid_hw",
    [
        (224, (14, 14)),
        (512, (32, 32)),
        (1280, (80, 80)),
    ],
)
def test_simple_feature_pyramid_shapes_and_strides(image_size, grid_hw):
    """Verifies SimpleFeaturePyramid produces P2 (H/4), P3 (H/8), P4 (H/16), P5 (H/32)
    without throwing spatial dimension mismatch errors across resolutions.
    """
    h_g, w_g = grid_hw
    sfp = SimpleFeaturePyramid(in_channels=64, out_channels=32)

    spatial = torch.randn(2, 64, h_g, w_g)
    out = sfp(spatial)

    assert set(out.keys()) == {"p2", "p3", "p4", "p5"}
    assert out["p2"].shape == (2, 32, h_g * 4, w_g * 4)
    assert out["p3"].shape == (2, 32, h_g * 2, w_g * 2)
    assert out["p4"].shape == (2, 32, h_g, w_g)
    assert out["p5"].shape == (2, 32, (h_g + 1) // 2, (w_g + 1) // 2)


def test_simple_feature_pyramid_default_768_to_256():
    """Verifies canonical ViT-Base (768 -> 256) SFP tensor dimensions."""
    sfp = SimpleFeaturePyramid(in_channels=768, out_channels=256)
    spatial = torch.randn(1, 768, 16, 16)  # 256x256 image at stride 16
    out = sfp(spatial)

    assert out["p2"].shape == (1, 256, 64, 64)
    assert out["p3"].shape == (1, 256, 32, 32)
    assert out["p4"].shape == (1, 256, 16, 16)
    assert out["p5"].shape == (1, 256, 8, 8)
