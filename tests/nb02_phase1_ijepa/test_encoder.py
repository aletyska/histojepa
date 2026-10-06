"""Unit tests for Vision Transformer Context Encoder and frozen export.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

from pathlib import Path
import pytest
import torch

from histojepa.models.ijepa.config import EncoderConfig
from histojepa.models.ijepa.vit_encoder import (
    IJepaViTEncoder,
    build_ijepa_vit,
    load_frozen_context_encoder,
    save_frozen_context_encoder,
)


pytestmark = pytest.mark.nb02


def test_ijepa_vit_encoder_forward_full_and_token_dropping():
    cfg = EncoderConfig(
        patch_size=16,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    # Image size 64 -> 4x4 = 16 patches
    vit = build_ijepa_vit(cfg, image_size=64)
    encoder = IJepaViTEncoder(vit)

    B = 2
    x = torch.randn(B, 3, 64, 64)

    # 1. Full image forward (target encoder path)
    full_out = encoder(x, keep_idx=None)
    assert full_out.shape == (B, 16, 64)

    # 2. Token dropping forward (context encoder path)
    keep_idx = torch.tensor([[0, 2, 5, 7], [1, 3, 6, 8]], dtype=torch.long)
    dropped_out = encoder(x, keep_idx=keep_idx)
    assert dropped_out.shape == (B, 4, 64)


def test_ijepa_vit_encoder_to_spatial():
    cfg = EncoderConfig(
        patch_size=16,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
    )
    vit = build_ijepa_vit(cfg, image_size=64)
    encoder = IJepaViTEncoder(vit)

    B = 2
    tokens = torch.randn(B, 16, 64)  # 4x4 grid
    spatial = encoder.to_spatial(tokens, grid_hw=(4, 4))
    assert spatial.shape == (B, 64, 4, 4)


def test_save_and_load_frozen_context_encoder_roundtrip(tmp_path: Path):
    cfg = EncoderConfig(
        patch_size=16,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    vit = build_ijepa_vit(cfg, image_size=64)
    encoder = IJepaViTEncoder(vit)

    export_dir = tmp_path / "exported_frozen_encoder"
    meta = {"loss": 0.42, "epoch": 10}

    saved_path = save_frozen_context_encoder(encoder, export_dir, metadata=meta)
    assert (saved_path / "manifest.json").exists()

    # Load frozen context encoder back
    loaded_encoder = load_frozen_context_encoder(saved_path, device="cpu")

    # Hard Constraint #1: Strictly frozen weights
    assert not loaded_encoder.training
    for name, param in loaded_encoder.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"

    # Forward check with spatial output
    dummy_img = torch.randn(1, 3, 64, 64)
    out_tokens = loaded_encoder(dummy_img)
    spatial_map = loaded_encoder.to_spatial(out_tokens, grid_hw=(4, 4))
    assert spatial_map.shape == (1, 64, 4, 4)
