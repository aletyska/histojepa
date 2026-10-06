"""Configuration dataclasses and YAML loader for Phase 1 I-JEPA Pre-Training.

Provides strongly typed configuration containers with support for nested YAML loading,
environment defaults, and runtime overrides.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml


@dataclass
class DataConfig:
    root: str = "data"
    split: str = "train"
    image_size: int = 1280
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    num_workers: int = 4


@dataclass
class EncoderConfig:
    patch_size: int = 16
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    attn_implementation: str = "sdpa"


@dataclass
class PredictorConfig:
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0


@dataclass
class MaskingConfig:
    num_targets: int = 4
    target_scale: Tuple[float, float] = (0.15, 0.2)
    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    context_scale: Tuple[float, float] = (0.85, 1.0)
    context_aspect_ratio: Tuple[float, float] = (1.0, 1.0)
    min_context_keep: int = 10


@dataclass
class OptimConfig:
    epochs: int = 100
    micro_batch_size: int = 2
    grad_accum_steps: int = 16
    warmup_epochs: int = 15
    start_lr: float = 1.0e-4
    ref_lr: float = 1.0e-3
    final_lr: float = 1.0e-6
    lr_scaling: str = "sqrt"  # "none", "linear", or "sqrt"
    start_wd: float = 0.04
    final_wd: float = 0.4
    ema_momentum: Tuple[float, float] = (0.996, 1.0)
    amp_dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    def get_effective_lrs(self) -> Tuple[float, float, float]:
        """Calculates scaled learning rates based on lr_scaling mode."""
        eff_bs = self.effective_batch_size
        if self.lr_scaling == "sqrt":
            factor = (eff_bs / 2048.0) ** 0.5
        elif self.lr_scaling == "linear":
            factor = eff_bs / 2048.0
        else:
            factor = 1.0
        return self.start_lr * factor, self.ref_lr * factor, self.final_lr * factor


@dataclass
class OutputConfig:
    dir: str = "outputs/phase1_ijepa"
    save_every_epochs: int = 1
    keep_last: int = 3
    export_dir: str = "outputs/phase1_ijepa/context_encoder_frozen"


@dataclass
class Phase1Config:
    data: DataConfig = field(default_factory=DataConfig)
    model_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model_predictor: PredictorConfig = field(default_factory=PredictorConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    optimization: OptimConfig = field(default_factory=OptimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Phase1Config":
        path = Path(yaml_path)
        raw_dict: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_dict = loaded

        # Apply dictionary overrides if provided
        if overrides:
            for k, v in overrides.items():
                if isinstance(v, dict) and k in raw_dict and isinstance(raw_dict[k], dict):
                    raw_dict[k].update(v)
                else:
                    raw_dict[k] = v

        data_dict = raw_dict.get("data", {})
        enc_dict = raw_dict.get("model", {}).get("encoder", {})
        pred_dict = raw_dict.get("model", {}).get("predictor", {})
        mask_dict = raw_dict.get("masking", {})
        opt_dict = raw_dict.get("optimization", {})
        out_dict = raw_dict.get("output", {})

        # Ensure tuples for sequence fields
        if "mean" in data_dict and isinstance(data_dict["mean"], list):
            data_dict["mean"] = tuple(data_dict["mean"])
        if "std" in data_dict and isinstance(data_dict["std"], list):
            data_dict["std"] = tuple(data_dict["std"])
        if "target_scale" in mask_dict and isinstance(mask_dict["target_scale"], list):
            mask_dict["target_scale"] = tuple(mask_dict["target_scale"])
        if "target_aspect_ratio" in mask_dict and isinstance(mask_dict["target_aspect_ratio"], list):
            mask_dict["target_aspect_ratio"] = tuple(mask_dict["target_aspect_ratio"])
        if "context_scale" in mask_dict and isinstance(mask_dict["context_scale"], list):
            mask_dict["context_scale"] = tuple(mask_dict["context_scale"])
        if "context_aspect_ratio" in mask_dict and isinstance(mask_dict["context_aspect_ratio"], list):
            mask_dict["context_aspect_ratio"] = tuple(mask_dict["context_aspect_ratio"])
        if "ema_momentum" in opt_dict and isinstance(opt_dict["ema_momentum"], list):
            opt_dict["ema_momentum"] = tuple(opt_dict["ema_momentum"])

        return cls(
            data=DataConfig(**data_dict),
            model_encoder=EncoderConfig(**enc_dict),
            model_predictor=PredictorConfig(**pred_dict),
            masking=MaskingConfig(**mask_dict),
            optimization=OptimConfig(**opt_dict),
            output=OutputConfig(**out_dict),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
