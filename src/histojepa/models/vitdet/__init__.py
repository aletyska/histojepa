"""ViTDet Simple Feature Pyramid (SFP), Multi-Scale Detection Head, and Trainer for Branch A."""

from histojepa.models.vitdet.config import (
    BranchABackboneConfig,
    BranchAConfig,
    BranchADataConfig,
    BranchAEvalConfig,
    BranchAOptimConfig,
    BranchAOutputConfig,
    HeadConfig,
    SFPConfig,
)
from histojepa.models.vitdet.detector import HistoJEPA_ViTDet
from histojepa.models.vitdet.head import (
    DetectionLossOutput,
    Scale,
    ViTDetHead,
    batched_nms_pure_torch,
    giou_loss_xyxy,
    sigmoid_focal_loss,
)
from histojepa.models.vitdet.sfp import LayerNorm2d, SimpleFeaturePyramid
from histojepa.models.vitdet.trainer import (
    BranchAHistory,
    BranchAStepInfo,
    ViTDetTrainer,
    build_vitdet_param_groups,
)

__all__ = [
    "BranchAConfig",
    "BranchADataConfig",
    "BranchABackboneConfig",
    "SFPConfig",
    "HeadConfig",
    "BranchAOptimConfig",
    "BranchAEvalConfig",
    "BranchAOutputConfig",
    "LayerNorm2d",
    "SimpleFeaturePyramid",
    "Scale",
    "DetectionLossOutput",
    "ViTDetHead",
    "sigmoid_focal_loss",
    "giou_loss_xyxy",
    "batched_nms_pure_torch",
    "HistoJEPA_ViTDet",
    "BranchAStepInfo",
    "BranchAHistory",
    "ViTDetTrainer",
    "build_vitdet_param_groups",
]
