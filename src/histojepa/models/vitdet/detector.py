"""HistoJEPA Phase 2 — Branch A Detector (Frozen I-JEPA Context Encoder + ViTDet SFP + Head).

Ref:
1. Li et al., "Exploring Plain Vision Transformer Backbones for Object Detection"
   (papers/vitdet/vitdet.md), §3 "Simple feature pyramid" & Appendix 0.A.2.
2. AGENTS.md Hard Constraint #1: "The ViT Context Encoder weights MUST remain
   strictly frozen (param.requires_grad = False) during Branch A (ViTDet) execution.
   Gradients in Branch A must flow only through the Simple Feature Pyramid (SFP)
   and the detection head."
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from histojepa.models.ijepa.vit_encoder import (
    IJepaViTEncoder,
    load_frozen_context_encoder,
)
from histojepa.models.vitdet.config import BranchAConfig
from histojepa.models.vitdet.head import DetectionLossOutput, ViTDetHead
from histojepa.models.vitdet.sfp import SimpleFeaturePyramid


class HistoJEPA_ViTDet(nn.Module):
    """End-to-end Tumor Bud detector pairing a strictly frozen I-JEPA Context Encoder
    with Meta AI's ViTDet Simple Feature Pyramid (SFP) and Multi-Scale Detection Head.

    Enforces Frozen Backbone Discipline:
    - All parameters in `self.context_encoder` have `requires_grad = False`.
    - `self.context_encoder` is permanently kept in `eval()` mode even when `self.train(True)` is called.
    - The forward pass through `self.context_encoder` executes under `torch.no_grad()`.
    """

    def __init__(
        self,
        context_encoder: IJepaViTEncoder,
        sfp: SimpleFeaturePyramid,
        head: ViTDetHead,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.sfp = sfp
        self.head = head

        self._enforce_frozen_backbone()

    def _enforce_frozen_backbone(self) -> None:
        """Strictly freezes all parameters of the I-JEPA Context Encoder."""
        self.context_encoder.eval()
        for param in self.context_encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "HistoJEPA_ViTDet":
        """Overrides nn.Module.train to ensure context_encoder never leaves eval() mode."""
        super().train(mode)
        self.context_encoder.eval()
        for param in self.context_encoder.parameters():
            param.requires_grad = False
        return self

    @classmethod
    def from_config(
        cls,
        config: BranchAConfig,
        context_encoder: Optional[IJepaViTEncoder] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "HistoJEPA_ViTDet":
        """Builds HistoJEPA_ViTDet from a BranchAConfig, loading the frozen encoder from disk if not provided."""
        if context_encoder is None:
            context_encoder = load_frozen_context_encoder(
                config.backbone.frozen_encoder_dir,
                device=str(device),
            )

        sfp = SimpleFeaturePyramid(
            in_channels=config.sfp.in_channels,
            out_channels=config.sfp.out_channels,
        )
        head = ViTDetHead(
            in_channels=config.head.in_channels,
            num_classes=config.head.num_classes,
            num_convs=config.head.num_convs,
            prior_prob=config.head.prior_prob,
            strides=config.head.strides,
            scale_ranges=config.head.scale_ranges,
            center_sampling_radius=config.head.center_sampling_radius,
            focal_alpha=config.head.focal_alpha,
            focal_gamma=config.head.focal_gamma,
        )
        detector = cls(context_encoder=context_encoder, sfp=sfp, head=head)
        return detector.to(device)

    @torch.no_grad()
    def extract_spatial_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes `(B, 3, H, W)` images with the frozen I-JEPA Context Encoder into `(B, D, H/16, W/16)`."""
        self.context_encoder.eval()
        _, _, h, w = images.shape
        p_sz = (
            self.context_encoder.patch_size[0]
            if isinstance(self.context_encoder.patch_size, (list, tuple))
            else int(self.context_encoder.patch_size)
        )
        grid_hw = (h // p_sz, w // p_sz)
        tokens = self.context_encoder(images, keep_idx=None)
        spatial_grid = self.context_encoder.to_spatial(tokens, grid_hw=grid_hw)
        return spatial_grid

    def forward_pyramid(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts frozen ViT spatial features and builds the trainable Simple Feature Pyramid."""
        with torch.no_grad():
            spatial = self.extract_spatial_features(images)
        # Detach explicitly so autograd never traverses into the frozen encoder
        spatial = spatial.detach()
        return self.sfp(spatial)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
        score_threshold: float = 0.25,
        nms_iou_threshold: float = 0.50,
        max_detections: int = 300,
    ) -> Union[DetectionLossOutput, List[Dict[str, torch.Tensor]]]:
        """Forward pass for training (when `targets` is provided) or inference (when `targets` is None)."""
        pyramid = self.forward_pyramid(images)
        cls_logits, bbox_reg, centerness = self.head.forward_features(pyramid)

        if targets is not None:
            return self.head.compute_loss(cls_logits, bbox_reg, centerness, targets)

        _, _, img_h, img_w = images.shape
        return self.head.predict(
            cls_logits,
            bbox_reg,
            centerness,
            image_size=(img_h, img_w),
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )

    def trainable_state_dict(self) -> Dict[str, Any]:
        """Returns state dictionary containing ONLY trainable modules (SFP + Head) to save disk space."""
        return {
            "sfp": self.sfp.state_dict(),
            "head": self.head.state_dict(),
        }

    def load_trainable_state_dict(self, state: Dict[str, Any]) -> None:
        """Loads SFP and Head weights while preserving the frozen Context Encoder."""
        if "sfp" in state and "head" in state:
            self.sfp.load_state_dict(state["sfp"])
            self.head.load_state_dict(state["head"])
        else:
            self.load_state_dict(state, strict=False)
        self._enforce_frozen_backbone()
