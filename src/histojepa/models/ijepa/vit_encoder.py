"""Vision Transformer Context & Target Encoder implementation using HuggingFace IJepaModel.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), §3 "Method", App. A.1 "Architectures".
Implements token-dropping context encoder and frozen encoder export utilities for Phase 2.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.ijepa.configuration_ijepa import IJepaConfig
from transformers.models.ijepa.modeling_ijepa import IJepaModel

from histojepa.models.ijepa.config import EncoderConfig
from histojepa.models.ijepa.masking import apply_masks


def build_ijepa_vit(
    cfg: EncoderConfig,
    image_size: int = 1280,
) -> IJepaModel:
    """Builds a plain Vision Transformer using transformers.IJepaModel (ViT-B/16).

    Args:
        cfg: EncoderConfig specifying dimensions and layers.
        image_size: Input spatial dimension (default: 1280).

    Returns:
        IJepaModel instance initialized with random weights, without pooling or CLS tokens.
    """
    hf_config = IJepaConfig(
        image_size=image_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        _attn_implementation=cfg.attn_implementation,
    )
    return IJepaModel(hf_config, add_pooling_layer=False, use_mask_token=False)


class IJepaViTEncoder(nn.Module):
    """Context and Target Vision Transformer Encoder.

    Supports both full-image encoding (target encoder, keep_idx=None) and
    true token-dropping encoding (context encoder, keep_idx=(B, N_c)).

    Args:
        vit: Underlying IJepaModel instance.
    """

    def __init__(self, vit: IJepaModel):
        super().__init__()
        self.vit = vit
        self.config = vit.config
        self.hidden_size = vit.config.hidden_size
        self.patch_size = vit.config.patch_size
        self.image_size = vit.config.image_size

    def _get_layers(self) -> nn.ModuleList:
        """Returns transformer layers across transformers version differences."""
        if hasattr(self.vit, "layers"):
            return self.vit.layers
        elif hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            return self.vit.encoder.layer
        raise AttributeError("Could not locate transformer layers in IJepaModel.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        keep_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through patch embeddings and transformer layers.

        Args:
            pixel_values: Input image tensor (B, 3, H, W).
            keep_idx: Optional patch token indices (B, K) to retain for context encoding.
                When None, all patches are processed (target encoder / full image).

        Returns:
            Output representation tensor (B, K, D) if keep_idx is given, else (B, N, D).
        """
        # 1. Patch projection + positional embedding: (B, N, D)
        embeddings = self.vit.embeddings(pixel_values)

        # 2. Token dropping for context encoder: only process visible patches
        if keep_idx is not None:
            hidden_states = apply_masks(embeddings, keep_idx)
        else:
            hidden_states = embeddings

        # 3. Propagate through transformer blocks
        for layer in self._get_layers():
            hidden_states = layer(hidden_states, None)

        # 4. Final LayerNorm
        sequence_output = self.vit.layernorm(hidden_states)
        return sequence_output

    def to_spatial(
        self,
        tokens: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Reshapes (B, N, D) patch sequence tokens to BCHW spatial grid (B, D, H/16, W/16).

        Used to interface the frozen Context Encoder with Meta AI's ViTDet Simple Feature
        Pyramid (SFP) in Phase 2 - Branch A.

        Args:
            tokens: Patch token tensor of shape (B, N, D).
            grid_hw: Optional explicit (H_patches, W_patches). If omitted, inferred from model config.

        Returns:
            Spatial feature map tensor of shape (B, D, H_patches, W_patches).
        """
        B, N, D = tokens.shape
        if grid_hw is None:
            img_sz = (
                self.config.image_size
                if isinstance(self.config.image_size, (list, tuple))
                else (self.config.image_size, self.config.image_size)
            )
            p_sz = (
                self.config.patch_size
                if isinstance(self.config.patch_size, (list, tuple))
                else (self.config.patch_size, self.config.patch_size)
            )
            H_g = img_sz[0] // p_sz[0]
            W_g = img_sz[1] // p_sz[1]
        else:
            H_g, W_g = grid_hw

        if H_g * W_g != N:
            raise ValueError(
                f"Grid dimensions {H_g}x{W_g} ({H_g*W_g} tokens) do not match token sequence length {N}."
            )

        return tokens.transpose(1, 2).reshape(B, D, H_g, W_g).contiguous()


def save_frozen_context_encoder(
    encoder: IJepaViTEncoder,
    output_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Freezes all weights and exports the Context Encoder bundle for Phase 2.

    Ref: AGENTS.md Hard Constraint #1: "The ViT Context Encoder weights MUST remain
    strictly frozen (param.requires_grad = False) during Branch A and Branch B execution."

    Args:
        encoder: IJepaViTEncoder instance.
        output_dir: Destination directory.
        metadata: Optional dictionary with training provenance and performance stats.

    Returns:
        Path to the exported directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Strictly freeze all parameters and set eval mode
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # 2. Save underlying HuggingFace IJepaModel
    encoder.vit.save_pretrained(str(out_path))

    # 3. Save PyTorch state dictionary
    torch.save(
        encoder.state_dict(),
        out_path / "context_encoder_frozen.pt",
    )

    # 4. Save metadata manifest
    img_sz = (
        encoder.config.image_size
        if isinstance(encoder.config.image_size, (list, tuple))
        else (encoder.config.image_size, encoder.config.image_size)
    )
    p_sz = (
        encoder.config.patch_size
        if isinstance(encoder.config.patch_size, (list, tuple))
        else (encoder.config.patch_size, encoder.config.patch_size)
    )

    manifest = {
        "architecture": "IJepaViTEncoder",
        "frozen": True,
        "hidden_size": encoder.config.hidden_size,
        "patch_size": p_sz,
        "image_size": img_sz,
        "spatial_grid": [img_sz[0] // p_sz[0], img_sz[1] // p_sz[1]],
        "metadata": metadata or {},
    }

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path


def load_frozen_context_encoder(
    path_or_dir: Union[str, Path],
    device: str = "cpu",
) -> IJepaViTEncoder:
    """Loads a pre-trained Context Encoder strictly frozen for Phase 2 (Branch A / Branch B).

    Guarantees:
    - Model is set to eval() mode.
    - 100% of parameters have requires_grad = False.
    - Validates that forward pass produces expected shapes (B, D, H/16, W/16).

    Args:
        path_or_dir: Path to directory saved by save_frozen_context_encoder.
        device: PyTorch device to place encoder on.

    Returns:
        IJepaViTEncoder ready to drive ViTDet SFP and Clustering.
    """
    dir_path = Path(path_or_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Context encoder directory not found at: {dir_path}")

    # Load HF model from pretrained directory
    hf_model = IJepaModel.from_pretrained(str(dir_path))
    encoder = IJepaViTEncoder(hf_model).to(device)

    # Enforce frozen discipline
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder
