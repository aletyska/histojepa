"""ViTDet Simple Feature Pyramid (SFP) implementation.

Ref: Li et al., "Exploring Plain Vision Transformer Backbones for Object Detection"
(papers/vitdet/vitdet.md), §3 "Simple feature pyramid" & Appendix 0.A.2 "Architectures".

Builds a multi-scale feature pyramid {P2, P3, P4, P5} directly in parallel from
only the last single-scale feature map (stride 16) of a plain Vision Transformer
(our strictly frozen Phase 1 I-JEPA Context Encoder).
"""

from typing import Dict

import torch
import torch.nn as nn
from transformers.models.vitdet.modeling_vitdet import VitDetLayerNorm


class LayerNorm2d(VitDetLayerNorm):
    """2D Channel-wise LayerNorm over (B, C, H, W) feature maps.

    Inherits directly from HuggingFace `transformers.models.vitdet.modeling_vitdet.VitDetLayerNorm`
    (Ref: ViTDet paper Appendix 0.A.2).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(normalized_shape=num_channels, eps=eps)


class SimpleFeaturePyramid(nn.Module):
    """ViTDet Simple Feature Pyramid (SFP) built on a single-scale stride-16 ViT map.

    Ref: ViTDet (papers/vitdet/vitdet.md), §3 "Simple feature pyramid" & App. 0.A.2:
    "We build a simple feature pyramid of scales {1/32, 1/16, 1/8, 1/4} (see Sec. 3).
    The 1/32 scale is built by stride-2 2x2 max pooling. The 1/16 scale simply uses
    the ViT's final feature map. Scale 1/8 (or 1/4) is built by one (or two) 2x2
    deconvolution layer(s) with stride=2. In the 1/4 scale case, the first deconvolution
    is followed by LayerNorm (LN) and GeLU. Then for each pyramid level, we apply a
    1x1 convolution with LN to reduce dimension to 256 and then a 3x3 convolution
    also with LN, similar to the per-level processing of FPN."

    Tensor Dimension Conventions (per AGENTS.md):
        Input spatial grid: `(B, in_channels, H/16, W/16)` where `in_channels = 768`
        Outputs:
            - `p2`: `(B, out_channels, H/4, W/4)`   (Stride 4, scale 1/4)
            - `p3`: `(B, out_channels, H/8, W/8)`   (Stride 8, scale 1/8)
            - `p4`: `(B, out_channels, H/16, W/16)` (Stride 16, scale 1/16)
            - `p5`: `(B, out_channels, H/32, W/32)` (Stride 32, scale 1/32)
    """

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}

        dim_half = max(out_channels, in_channels // 2)      # 384 for ViT-B (768)
        dim_quarter = max(out_channels // 2, in_channels // 4)  # 192 for ViT-B (768)

        # P2 (scale 1/4, stride 4): two 2x2 stride-2 deconvolutions + per-level 1x1 & 3x3 convs
        self.p2_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, dim_half, kernel_size=2, stride=2),
            LayerNorm2d(dim_half),
            nn.GELU(),
            nn.ConvTranspose2d(dim_half, dim_quarter, kernel_size=2, stride=2),
            nn.Conv2d(dim_quarter, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )

        # P3 (scale 1/8, stride 8): one 2x2 stride-2 deconvolution + per-level 1x1 & 3x3 convs
        self.p3_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, dim_half, kernel_size=2, stride=2),
            nn.Conv2d(dim_half, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )

        # P4 (scale 1/16, stride 16): identity on ViT map + per-level 1x1 & 3x3 convs
        self.p4_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )

        # P5 (scale 1/32, stride 32): stride-2 2x2 max pooling + per-level 1x1 & 3x3 convs
        self.p5_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Kaiming / Xavier initialization to convolutional layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spatial_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generates multi-scale pyramid maps {p2, p3, p4, p5} from a single-scale ViT feature grid.

        Args:
            spatial_features: Reshaped ViT token grid of shape `(B, D, H_g, W_g)` where
                `H_g = H / 16` and `W_g = W / 16` (e.g. `(B, 768, 80, 80)` for 1280x1280).

        Returns:
            Ordered dictionary of pyramid feature maps:
                - `"p2"`: `(B, 256, 4 * H_g, 4 * W_g)` (Stride 4)
                - `"p3"`: `(B, 256, 2 * H_g, 2 * W_g)` (Stride 8)
                - `"p4"`: `(B, 256, H_g, W_g)`         (Stride 16)
                - `"p5"`: `(B, 256, H_g // 2, W_g // 2)` (Stride 32)
        """
        if spatial_features.ndim != 4:
            raise ValueError(
                f"Expected 4D BCHW spatial_features tensor, got shape {tuple(spatial_features.shape)}"
            )
        if spatial_features.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {spatial_features.shape[1]}"
            )

        # Ensure even spatial dimensions before 2x2 MaxPool2d on P5 if odd grid size occurs
        _, _, h_g, w_g = spatial_features.shape
        if h_g % 2 != 0 or w_g % 2 != 0:
            pad_h = h_g % 2
            pad_w = w_g % 2
            p5_input = nn.functional.pad(spatial_features, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            p5_input = spatial_features

        return {
            "p2": self.p2_block(spatial_features),
            "p3": self.p3_block(spatial_features),
            "p4": self.p4_block(spatial_features),
            "p5": self.p5_block(p5_input),
        }
