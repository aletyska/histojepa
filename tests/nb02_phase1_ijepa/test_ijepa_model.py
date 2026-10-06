"""Unit tests for complete I-JEPA architecture module.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

import pytest
import torch

from histojepa.models.ijepa.config import (
    DataConfig,
    EncoderConfig,
    MaskingConfig,
    OptimConfig,
    Phase1Config,
    PredictorConfig,
)
from histojepa.models.ijepa.ijepa import IJEPA


pytestmark = pytest.mark.nb02


def test_ijepa_full_forward_and_gradient_flow():
    cfg = Phase1Config(
        data=DataConfig(image_size=64),
        model_encoder=EncoderConfig(
            patch_size=16,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        ),
        model_predictor=PredictorConfig(
            embed_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        ),
        masking=MaskingConfig(num_targets=2),
    )

    model = IJEPA(cfg)
    model.train()

    # Target encoder must remain in eval mode even after model.train()
    assert not model.target_encoder.training
    assert model.context_encoder.training
    assert model.predictor.training

    B = 2
    images = torch.randn(B, 3, 64, 64)
    ctx_idx = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    tgt_list = [
        torch.tensor([[6, 7], [8, 9]], dtype=torch.long),
        torch.tensor([[10, 11], [12, 13]], dtype=torch.long),
    ]

    out = model(images, ctx_idx, tgt_list)

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert len(out.preds) == 2
    assert len(out.targets) == 2
    assert out.target_std >= 0.0
    assert out.pred_std >= 0.0

    # Backpropagation test
    out.loss.backward()

    # Context encoder parameters must have gradients
    ctx_has_grad = any(p.grad is not None for p in model.context_encoder.parameters())
    assert ctx_has_grad, "Context encoder has no gradients!"

    # Predictor parameters must have gradients
    pred_has_grad = any(p.grad is not None for p in model.predictor.parameters())
    assert pred_has_grad, "Predictor has no gradients!"

    # Target encoder parameters must strictly have NO gradients
    for p in model.target_encoder.parameters():
        assert p.grad is None, "Target encoder has gradients! It must be completely frozen to backpropagation."
