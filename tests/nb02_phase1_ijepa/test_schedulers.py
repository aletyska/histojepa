"""Unit tests for I-JEPA optimization schedules.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

import pytest

from histojepa.models.ijepa.config import OptimConfig
from histojepa.models.ijepa.schedulers import LinearSchedule, WarmupCosineSchedule


pytestmark = pytest.mark.nb02


def test_warmup_cosine_schedule():
    sched = WarmupCosineSchedule(
        start_value=1.0e-4,
        ref_value=1.0e-3,
        final_value=1.0e-6,
        warmup_steps=100,
        total_steps=1000,
    )

    assert pytest.approx(sched(0)) == 1.0e-4
    assert pytest.approx(sched(100)) == 1.0e-3
    assert pytest.approx(sched(1000)) == 1.0e-6

    # Monotonic warmup
    assert sched(0) < sched(50) < sched(100)
    # Cosine decay
    assert sched(100) > sched(500) > sched(1000)


def test_linear_schedule():
    sched = LinearSchedule(start_value=0.04, final_value=0.40, total_steps=100)

    assert pytest.approx(sched(0)) == 0.04
    assert pytest.approx(sched(50)) == 0.22
    assert pytest.approx(sched(100)) == 0.40


def test_effective_lrs_sqrt_scaling():
    cfg = OptimConfig(
        micro_batch_size=2,
        grad_accum_steps=16,  # eff_bs = 32
        start_lr=1.0e-4,
        ref_lr=1.0e-3,
        final_lr=1.0e-6,
        lr_scaling="sqrt",
    )
    # factor = sqrt(32 / 2048) = sqrt(1 / 64) = 1/8 = 0.125
    s_lr, r_lr, f_lr = cfg.get_effective_lrs()
    assert pytest.approx(s_lr) == 1.25e-5
    assert pytest.approx(r_lr) == 1.25e-4
    assert pytest.approx(f_lr) == 1.25e-7
