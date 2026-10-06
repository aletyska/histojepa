"""Optimization schedules for Learning Rate, Weight Decay, and EMA Momentum.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), Appendix A.1 "Optimization".
All schedules are stateless mathematical functions of the optimizer step, ensuring
exact deterministic restoration upon training resumption from checkpoints.
"""

import math
from typing import Optional


class WarmupCosineSchedule:
    """Linear warmup followed by cosine decay.

    Ref: I-JEPA App. A.1: linearly increase LR during warmup, then decay following cosine schedule.

    Args:
        start_value: Value at step 0 (e.g. start_lr).
        ref_value: Peak value at the end of warmup (e.g. ref_lr).
        final_value: Terminal value at total_steps (e.g. final_lr).
        warmup_steps: Number of optimizer steps for linear warmup phase.
        total_steps: Total number of optimizer steps across training.
    """

    def __init__(
        self,
        start_value: float,
        ref_value: float,
        final_value: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.start_value = float(start_value)
        self.ref_value = float(ref_value)
        self.final_value = float(final_value)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))

    def __call__(self, step: int) -> float:
        """Returns scheduled value for the given optimizer step."""
        step = max(0, min(step, self.total_steps))
        if step < self.warmup_steps:
            pct = step / max(1, self.warmup_steps)
            return self.start_value + (self.ref_value - self.start_value) * pct
        else:
            cosine_steps = max(1, self.total_steps - self.warmup_steps)
            current_cosine_step = step - self.warmup_steps
            decay = 0.5 * (1.0 + math.cos(math.pi * current_cosine_step / cosine_steps))
            return self.final_value + (self.ref_value - self.final_value) * decay


class LinearSchedule:
    """Monotonic linear interpolation between start_value and final_value.

    Used for Weight Decay (0.04 -> 0.4) and EMA Momentum (0.996 -> 1.0) schedules.

    Args:
        start_value: Initial value at step 0.
        final_value: Terminal value at total_steps.
        total_steps: Total number of optimizer steps.
    """

    def __init__(self, start_value: float, final_value: float, total_steps: int):
        self.start_value = float(start_value)
        self.final_value = float(final_value)
        self.total_steps = max(1, int(total_steps))

    def __call__(self, step: int) -> float:
        """Returns scheduled value for the given optimizer step."""
        step = max(0, min(step, self.total_steps))
        pct = step / self.total_steps
        return self.start_value + (self.final_value - self.start_value) * pct
