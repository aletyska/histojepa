"""Combined Image-based Joint-Embedding Predictive Architecture (I-JEPA) module.

Orchestrates Context Encoder, Target Encoder (EMA), and Predictor network.
Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), §3 "Method".
"""

import copy
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from histojepa.models.ijepa.config import Phase1Config
from histojepa.models.ijepa.ema import update_ema
from histojepa.models.ijepa.loss import ijepa_l2_loss
from histojepa.models.ijepa.masking import apply_masks
from histojepa.models.ijepa.predictor import IJepaPredictor
from histojepa.models.ijepa.vit_encoder import IJepaViTEncoder, build_ijepa_vit


@dataclass
class IJepaOutput:
    """Outputs from a single I-JEPA forward iteration."""

    loss: torch.Tensor
    preds: List[torch.Tensor]
    targets: List[torch.Tensor]
    target_std: float
    pred_std: float


class IJEPA(nn.Module):
    """Complete I-JEPA architecture.

    Components:
    - context_encoder (f_theta): Vision Transformer processing visible context patches.
    - target_encoder (f_theta_bar): Vision Transformer processing full images, updated via EMA.
    - predictor (g_phi): Narrow Vision Transformer predicting target representations.

    Args:
        config: Phase1Config specifying data, model, and masking parameters.
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        # 1. Instantiate Context Encoder
        hf_vit_ctx = build_ijepa_vit(
            config.model_encoder,
            image_size=config.data.image_size,
        )
        self.context_encoder = IJepaViTEncoder(hf_vit_ctx)

        # 2. Instantiate Target Encoder as a deep copy at initialization (App. A.1)
        hf_vit_tgt = copy.deepcopy(hf_vit_ctx)
        self.target_encoder = IJepaViTEncoder(hf_vit_tgt)
        self.target_encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 3. Instantiate Narrow ViT Predictor
        p_sz = config.model_encoder.patch_size
        img_sz = config.data.image_size
        grid_hw = (img_sz // p_sz, img_sz // p_sz)

        self.predictor = IJepaPredictor(
            grid_size=grid_hw,
            encoder_dim=config.model_encoder.hidden_size,
            embed_dim=config.model_predictor.embed_dim,
            depth=config.model_predictor.depth,
            num_heads=config.model_predictor.num_heads,
            mlp_ratio=config.model_predictor.mlp_ratio,
        )

    def train(self, mode: bool = True) -> "IJEPA":
        """Sets training mode for trainable components while keeping target encoder in eval mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns all parameters subject to gradient descent optimization."""
        params = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        return [p for p in params if p.requires_grad]

    def momentum_update(self, momentum: float) -> None:
        """Applies EMA step updating target encoder from context encoder."""
        update_ema(self.target_encoder, self.context_encoder, momentum=momentum)

    def forward(
        self,
        images: torch.Tensor,
        ctx_idx: torch.Tensor,
        tgt_idx_list: Sequence[torch.Tensor],
    ) -> IJepaOutput:
        """Executes one I-JEPA forward pass.

        Args:
            images: Batched image tensor (B, 3, H, W).
            ctx_idx: Context token indices (B, N_c).
            tgt_idx_list: List of M target block indices, each (B, N_t).

        Returns:
            IJepaOutput container with loss and monitoring statistics.
        """
        # --- 1. Target Side: compute full-image representation, mask outputs (Table 11) ---
        with torch.no_grad():
            s_y = self.target_encoder(images)  # (B, N, D)
            targets = [apply_masks(s_y, idx) for idx in tgt_idx_list]  # M x (B, N_t, D)

        # --- 2. Context Side: encode visible context tokens only ---
        s_x = self.context_encoder(images, keep_idx=ctx_idx)  # (B, N_c, D)

        # --- 3. Predictor Network: evaluate M times for the M target blocks ---
        preds = [self.predictor(s_x, ctx_idx, idx) for idx in tgt_idx_list]

        # --- 4. Loss: mean L2 distance across all M blocks ---
        loss = ijepa_l2_loss(preds, targets)

        # --- 5. Representation collapse diagnostic (standard deviation) ---
        with torch.no_grad():
            target_std = float(targets[0].detach().std(dim=-1).mean().item())
            pred_std = float(preds[0].detach().std(dim=-1).mean().item())

        return IJepaOutput(
            loss=loss,
            preds=preds,
            targets=targets,
            target_std=target_std,
            pred_std=pred_std,
        )
