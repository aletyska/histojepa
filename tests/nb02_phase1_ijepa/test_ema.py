"""Unit tests for Exponential Moving Average target updates.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

import copy
import pytest
import torch
import torch.nn as nn

from histojepa.models.ijepa.ema import update_ema


pytestmark = pytest.mark.nb02


def test_update_ema_mathematical_correctness():
    online = nn.Linear(4, 4)
    target = copy.deepcopy(online)

    # Freeze target parameters
    for p in target.parameters():
        p.requires_grad = False

    with torch.no_grad():
        online.weight.fill_(2.0)
        target.weight.fill_(10.0)

    # Test m = 1.0 (no change)
    update_ema(target, online, momentum=1.0)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 10.0))

    # Test m = 0.0 (full copy)
    update_ema(target, online, momentum=0.0)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 2.0))

    # Test m = 0.9 (0.9 * 2.0 + 0.1 * 12.0 = 1.8 + 1.2 = 3.0)
    with torch.no_grad():
        online.weight.fill_(12.0)
        target.weight.fill_(2.0)
    update_ema(target, online, momentum=0.9)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 3.0))

    # Verify target parameters remain frozen
    for p in target.parameters():
        assert not p.requires_grad
