"""Unit tests for I-JEPA multi-block masking collator and token gather.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

import pytest
import torch

from histojepa.models.ijepa.masking import MultiBlockMaskCollator, apply_masks


pytestmark = pytest.mark.nb02


def test_apply_masks():
    B, N, D = 2, 10, 4
    x = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)
    idx = torch.tensor([[0, 2, 5], [1, 3, 9]], dtype=torch.long)

    gathered = apply_masks(x, idx)
    assert gathered.shape == (2, 3, 4)
    # Check specific element
    assert torch.equal(gathered[0, 0], x[0, 0])
    assert torch.equal(gathered[0, 1], x[0, 2])
    assert torch.equal(gathered[0, 2], x[0, 5])
    assert torch.equal(gathered[1, 0], x[1, 1])


def test_multiblock_mask_collator_shapes_and_zero_overlap():
    B = 3
    C, H, W = 3, 160, 160
    # 160x160 with patch_size 16 -> grid_size (10, 10) -> N = 100
    images = [torch.randn(C, H, W) for _ in range(B)]

    collator = MultiBlockMaskCollator(
        grid_size=(10, 10),
        num_targets=4,
        target_scale=(0.15, 0.20),
        target_aspect_ratio=(0.75, 1.5),
        context_scale=(0.85, 1.0),
        min_context_keep=5,
        seed=42,
    )

    batch_imgs, ctx_idx, tgt_list = collator(images)

    assert batch_imgs.shape == (B, C, H, W)
    assert ctx_idx.dim() == 2
    assert ctx_idx.shape[0] == B
    assert len(tgt_list) == 4

    N_t = tgt_list[0].shape[1]
    for tgt in tgt_list:
        assert tgt.shape == (B, N_t)
        assert tgt.dtype == torch.long

    # Crucial test: strictly zero overlap between context and ANY target block
    for b in range(B):
        ctx_set = set(ctx_idx[b].tolist())
        for m in range(4):
            tgt_set = set(tgt_list[m][b].tolist())
            overlap = ctx_set.intersection(tgt_set)
            assert len(overlap) == 0, f"Detected overlap between context and target {m} on image {b}: {overlap}"


def test_multiblock_mask_collator_determinism():
    images = [torch.randn(3, 80, 80) for _ in range(2)]

    c1 = MultiBlockMaskCollator(grid_size=(5, 5), num_targets=2, seed=123)
    c2 = MultiBlockMaskCollator(grid_size=(5, 5), num_targets=2, seed=123)

    _, ctx1, tgt1 = c1(images)
    _, ctx2, tgt2 = c2(images)

    assert torch.equal(ctx1, ctx2)
    for t1, t2 in zip(tgt1, tgt2):
        assert torch.equal(t1, t2)
