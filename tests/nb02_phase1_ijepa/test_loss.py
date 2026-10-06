"""Unit tests for I-JEPA L2 prediction loss.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

import pytest
import torch

from histojepa.models.ijepa.loss import ijepa_l2_loss


pytestmark = pytest.mark.nb02


def test_ijepa_l2_loss_values():
    t1 = torch.zeros(2, 4, 8)
    p1 = torch.zeros(2, 4, 8)

    t2 = torch.ones(2, 4, 8)
    p2 = torch.full((2, 4, 8), 3.0)  # (3 - 1)^2 = 4.0

    # Block 1 loss is 0.0, Block 2 loss is 4.0 -> average = 2.0
    loss = ijepa_l2_loss([p1, p2], [t1, t2])
    assert torch.isclose(loss, torch.tensor(2.0))


def test_ijepa_l2_loss_exceptions():
    with pytest.raises(ValueError):
        ijepa_l2_loss([], [])

    with pytest.raises(ValueError):
        ijepa_l2_loss([torch.zeros(1, 1)], [torch.zeros(1, 1), torch.zeros(1, 1)])
