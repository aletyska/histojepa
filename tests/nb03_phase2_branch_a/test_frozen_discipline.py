"""Unit tests for Frozen Backbone Discipline in HistoJEPA_ViTDet.

Bound to: notebooks/03_Phase_2_Branch_A.py
Pytest Marker: @pytest.mark.nb03
"""

import pytest
import torch

from histojepa.models.ijepa.config import EncoderConfig
from histojepa.models.ijepa.vit_encoder import IJepaViTEncoder, build_ijepa_vit
from histojepa.models.vitdet.detector import HistoJEPA_ViTDet
from histojepa.models.vitdet.head import ViTDetHead
from histojepa.models.vitdet.sfp import SimpleFeaturePyramid
from histojepa.models.vitdet.trainer import build_vitdet_param_groups

pytestmark = pytest.mark.nb03


def _build_tiny_detector(image_size: int = 128) -> HistoJEPA_ViTDet:
    enc_cfg = EncoderConfig(
        patch_size=16,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    vit = build_ijepa_vit(enc_cfg, image_size=image_size)
    context_encoder = IJepaViTEncoder(vit)
    sfp = SimpleFeaturePyramid(in_channels=64, out_channels=32)
    head = ViTDetHead(in_channels=32, num_classes=1, num_convs=2)
    return HistoJEPA_ViTDet(context_encoder=context_encoder, sfp=sfp, head=head)


def test_frozen_backbone_discipline_enforced():
    """Verifies AGENTS.md Hard Constraint #1:
    1. All context_encoder parameters have requires_grad = False.
    2. Calling detector.train(True) keeps context_encoder in eval() mode.
    3. Backpropagating detection loss produces gradients ONLY in SFP and Head,
       and zero gradients (None) in context_encoder.
    """
    detector = _build_tiny_detector(image_size=128)
    detector.train(True)

    # 1. Context encoder must stay in eval mode even when detector is in train mode
    assert detector.training is True
    assert detector.context_encoder.training is False

    for name, param in detector.context_encoder.named_parameters():
        assert not param.requires_grad, f"Context encoder param {name} has requires_grad=True!"

    # 2. Optimizer param groups must exclude context_encoder completely
    param_groups = build_vitdet_param_groups(detector, weight_decay=0.05)
    opt_param_ids = {id(p) for g in param_groups for p in g["params"]}
    ctx_param_ids = {id(p) for p in detector.context_encoder.parameters()}
    assert opt_param_ids.isdisjoint(ctx_param_ids)

    # 3. Forward + backward pass gradient check
    images = torch.randn(2, 3, 128, 128)
    targets = [
        {
            "boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        },
        {
            "boxes": torch.tensor([[32.0, 32.0, 96.0, 96.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        },
    ]

    loss_out = detector(images, targets=targets)
    assert torch.isfinite(loss_out.total_loss)
    loss_out.total_loss.backward()

    # Context encoder must receive NO gradients
    for name, param in detector.context_encoder.named_parameters():
        assert param.grad is None, f"Context encoder param {name} received a gradient!"

    # SFP and Head must receive gradients
    sfp_grads = [p.grad for p in detector.sfp.parameters() if p.grad is not None]
    head_grads = [p.grad for p in detector.head.parameters() if p.grad is not None]
    assert len(sfp_grads) > 0, "SFP received no gradients!"
    assert len(head_grads) > 0, "ViTDetHead received no gradients!"
