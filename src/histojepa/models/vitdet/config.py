"""Configuration dataclasses and YAML loader for Phase 2 — Branch A (ViTDet Detection).

Ref: Li et al., "Exploring Plain Vision Transformer Backbones for Object Detection"
(papers/vitdet/vitdet.md), §3 & Appendix 0.A.2.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import yaml


@dataclass
class BranchADataConfig:
    root: str = "data"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    image_size: int = 1280
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    augment_train: bool = True
    num_workers: int = 4


@dataclass
class BranchABackboneConfig:
    frozen_encoder_dir: str = "outputs/phase1_ijepa/context_encoder_frozen"
    hidden_size: int = 768
    patch_size: int = 16


@dataclass
class SFPConfig:
    in_channels: int = 768
    out_channels: int = 256


@dataclass
class HeadConfig:
    in_channels: int = 256
    num_classes: int = 1
    num_convs: int = 4
    prior_prob: float = 0.01
    strides: Tuple[int, int, int, int] = (4, 8, 16, 32)
    scale_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 48.0),
        (48.0, 96.0),
        (96.0, 192.0),
        (192.0, 1e8),
    )
    center_sampling_radius: float = 1.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class BranchAOptimConfig:
    epochs: int = 30
    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    warmup_epochs: int = 3
    lr: float = 1.0e-4
    min_lr: float = 1.0e-6
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    amp_dtype: str = "bfloat16"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


@dataclass
class BranchAEvalConfig:
    score_threshold: float = 0.25
    nms_iou_threshold: float = 0.50
    max_detections: int = 300
    eval_every_epochs: int = 1


@dataclass
class BranchAOutputConfig:
    dir: str = "outputs/phase2_branch_a"
    save_every_epochs: int = 1
    keep_last: int = 3


@dataclass
class BranchAConfig:
    data: BranchADataConfig = field(default_factory=BranchADataConfig)
    backbone: BranchABackboneConfig = field(default_factory=BranchABackboneConfig)
    sfp: SFPConfig = field(default_factory=SFPConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    optimization: BranchAOptimConfig = field(default_factory=BranchAOptimConfig)
    evaluation: BranchAEvalConfig = field(default_factory=BranchAEvalConfig)
    output: BranchAOutputConfig = field(default_factory=BranchAOutputConfig)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "BranchAConfig":
        path = Path(yaml_path)
        raw_dict: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_dict = loaded

        if overrides:
            for k, v in overrides.items():
                if isinstance(v, dict) and k in raw_dict and isinstance(raw_dict[k], dict):
                    raw_dict[k].update(v)
                else:
                    raw_dict[k] = v

        data_dict = dict(raw_dict.get("data", {}))
        backbone_dict = dict(raw_dict.get("backbone", {}))
        sfp_dict = dict(raw_dict.get("sfp", {}))
        head_dict = dict(raw_dict.get("head", {}))
        opt_dict = dict(raw_dict.get("optimization", {}))
        eval_dict = dict(raw_dict.get("evaluation", {}))
        out_dict = dict(raw_dict.get("output", {}))

        if "mean" in data_dict and isinstance(data_dict["mean"], list):
            data_dict["mean"] = tuple(data_dict["mean"])
        if "std" in data_dict and isinstance(data_dict["std"], list):
            data_dict["std"] = tuple(data_dict["std"])
        if "strides" in head_dict and isinstance(head_dict["strides"], list):
            head_dict["strides"] = tuple(int(s) for s in head_dict["strides"])
        if "scale_ranges" in head_dict and isinstance(head_dict["scale_ranges"], list):
            head_dict["scale_ranges"] = tuple(
                (float(pair[0]), float(pair[1])) for pair in head_dict["scale_ranges"]
            )

        return cls(
            data=BranchADataConfig(**data_dict),
            backbone=BranchABackboneConfig(**backbone_dict),
            sfp=SFPConfig(**sfp_dict),
            head=HeadConfig(**head_dict),
            optimization=BranchAOptimConfig(**opt_dict),
            evaluation=BranchAEvalConfig(**eval_dict),
            output=BranchAOutputConfig(**out_dict),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
