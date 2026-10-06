"""L2 Prediction Loss in representation space for I-JEPA.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), §3 "Loss", Eq. (1).
"""

from typing import Sequence
import torch
import torch.nn.functional as F


def ijepa_l2_loss(
    preds: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Computes average L2 loss across M predicted target blocks in representation space.

    Formula (Eq. 1):
        Loss = (1 / M) * sum_{i=1}^M D(s_hat_y(i), s_y(i))
    where D is squared Euclidean distance between patch representations.

    Args:
        preds: List of M tensors containing predicted embeddings, each of shape (B, N_t, D).
        targets: List of M tensors containing ground-truth target embeddings, each (B, N_t, D).

    Returns:
        Scalar torch.Tensor representing the mean prediction error.
    """
    if len(preds) != len(targets):
        raise ValueError(f"Length mismatch: {len(preds)} predictions vs {len(targets)} targets.")
    if len(preds) == 0:
        raise ValueError("preds and targets sequences cannot be empty.")

    block_losses = [F.mse_loss(p, t) for p, t in zip(preds, targets)]
    total_loss = torch.stack(block_losses).mean()
    return total_loss
