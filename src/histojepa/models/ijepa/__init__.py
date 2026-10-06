"""I-JEPA model components, configuration, masking, and training pipeline.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), CVPR 2023.
"""

from histojepa.models.ijepa.config import (
    DataConfig,
    EncoderConfig,
    MaskingConfig,
    OptimConfig,
    OutputConfig,
    Phase1Config,
    PredictorConfig,
)
from histojepa.models.ijepa.ema import update_ema
from histojepa.models.ijepa.ijepa import IJEPA, IJepaOutput
from histojepa.models.ijepa.loss import ijepa_l2_loss
from histojepa.models.ijepa.masking import MultiBlockMaskCollator, apply_masks
from histojepa.models.ijepa.predictor import IJepaPredictor
from histojepa.models.ijepa.schedulers import LinearSchedule, WarmupCosineSchedule
from histojepa.models.ijepa.trainer import IJepaTrainer, StepInfo, TrainingHistory
from histojepa.models.ijepa.vit_encoder import (
    IJepaViTEncoder,
    build_ijepa_vit,
    load_frozen_context_encoder,
    save_frozen_context_encoder,
)

__all__ = [
    "DataConfig",
    "EncoderConfig",
    "PredictorConfig",
    "MaskingConfig",
    "OptimConfig",
    "OutputConfig",
    "Phase1Config",
    "MultiBlockMaskCollator",
    "apply_masks",
    "IJepaViTEncoder",
    "build_ijepa_vit",
    "save_frozen_context_encoder",
    "load_frozen_context_encoder",
    "IJepaPredictor",
    "update_ema",
    "ijepa_l2_loss",
    "IJEPA",
    "IJepaOutput",
    "WarmupCosineSchedule",
    "LinearSchedule",
    "IJepaTrainer",
    "StepInfo",
    "TrainingHistory",
]
