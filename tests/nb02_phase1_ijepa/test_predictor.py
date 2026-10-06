"""Unit tests for Narrow Vision Transformer Predictor.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

import pytest
import torch

from histojepa.models.ijepa.predictor import IJepaPredictor


pytestmark = pytest.mark.nb02


def test_ijepa_predictor_forward_shape_and_bottleneck():
    B = 2
    N_c = 6
    N_t = 3
    encoder_dim = 64
    embed_dim = 32
    grid_size = (4, 4)

    predictor = IJepaPredictor(
        grid_size=grid_size,
        encoder_dim=encoder_dim,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )

    ctx_tokens = torch.randn(B, N_c, encoder_dim)
    ctx_idx = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7]], dtype=torch.long)
    tgt_idx = torch.tensor([[8, 9, 10], [12, 13, 14]], dtype=torch.long)

    out = predictor(ctx_tokens, ctx_idx, tgt_idx)

    # Output shape must be (B, N_t, encoder_dim)
    assert out.shape == (B, N_t, encoder_dim)
    assert predictor.mask_token.shape == (1, 1, embed_dim)
    assert predictor.mask_token.requires_grad
