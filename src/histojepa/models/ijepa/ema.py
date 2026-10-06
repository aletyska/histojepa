"""Exponential Moving Average (EMA) parameter update for I-JEPA target encoder.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), §3 "Loss",
Appendix A.1 "Optimization" (momentum 0.996 -> 1.0).
"""

import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(target_module: nn.Module, online_module: nn.Module, momentum: float) -> None:
    """Updates target encoder weights via Exponential Moving Average of context encoder weights.

    Formula:
        theta_bar <- m * theta_bar + (1.0 - m) * theta

    Args:
        target_module: Target encoder (f_theta_bar). Parameters updated in-place.
        online_module: Context encoder (f_theta). Source parameters.
        momentum: EMA momentum value in [0, 1] (e.g. 0.996 to 1.0).
    """
    target_params = [p for p in target_module.parameters() if p.dtype.is_floating_point]
    online_params = [p for p in online_module.parameters() if p.dtype.is_floating_point]

    if len(target_params) != len(online_params):
        raise ValueError(
            f"Parameter count mismatch between target ({len(target_params)}) and online ({len(online_params)}) models."
        )

    # In-place batched EMA calculation: theta_bar = m * theta_bar + (1 - m) * theta
    torch._foreach_mul_(target_params, momentum)
    torch._foreach_add_(target_params, online_params, alpha=1.0 - momentum)
