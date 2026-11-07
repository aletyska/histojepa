"""ViTDet Training & Evaluation Engine for Phase 2 — Branch A.

Ref:
1. Li et al., "Exploring Plain Vision Transformer Backbones for Object Detection"
   (papers/vitdet/vitdet.md), §3 & Appendix 0.A.2.
2. AGENTS.md Hard Constraint #1 (Strictly Frozen Context Encoder) & #2 (Patient-Level Splits).
"""

from dataclasses import asdict, dataclass, field
import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from histojepa.data.detection_dataset import (
    TBYoloDetectionDataset,
    detection_collate_fn,
)
from histojepa.evaluation.detection import (
    DetectionMetricsReport,
    evaluate_detections,
)
from histojepa.models.ijepa.schedulers import WarmupCosineSchedule
from histojepa.models.ijepa.vit_encoder import IJepaViTEncoder
from histojepa.models.vitdet.config import BranchAConfig
from histojepa.models.vitdet.detector import HistoJEPA_ViTDet
from histojepa.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


@dataclass
class BranchAStepInfo:
    """Telemetry emitted after each optimizer step in Branch A."""

    epoch: int
    step: int
    total_steps: int
    total_loss: float
    cls_loss: float
    box_loss: float
    ctr_loss: float
    lr: float
    num_positives: int


@dataclass
class BranchAHistory:
    """Persisted step and epoch-level validation telemetry for Branch A."""

    steps: List[int] = field(default_factory=list)
    step_epochs: List[int] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    cls_losses: List[float] = field(default_factory=list)
    box_losses: List[float] = field(default_factory=list)
    ctr_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)

    eval_epochs: List[int] = field(default_factory=list)
    val_map50: List[float] = field(default_factory=list)
    val_map75: List[float] = field(default_factory=list)
    val_map50_95: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)

    def append_step(self, info: BranchAStepInfo) -> None:
        self.steps.append(info.step)
        self.step_epochs.append(info.epoch)
        self.total_losses.append(info.total_loss)
        self.cls_losses.append(info.cls_loss)
        self.box_losses.append(info.box_loss)
        self.ctr_losses.append(info.ctr_loss)
        self.lrs.append(info.lr)

    def append_eval(self, epoch: int, report: DetectionMetricsReport) -> None:
        self.eval_epochs.append(epoch)
        self.val_map50.append(report.map50)
        self.val_map75.append(report.map75)
        self.val_map50_95.append(report.map50_95)
        self.val_precision.append(report.precision)
        self.val_recall.append(report.recall)
        self.val_f1.append(report.f1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchAHistory":
        return cls(**data)


def build_vitdet_param_groups(
    detector: HistoJEPA_ViTDet,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Builds AdamW parameter groups strictly from trainable SFP and Head parameters.

    Verifies that zero parameters from `detector.context_encoder` are included.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in detector.named_parameters():
        if name.startswith("context_encoder"):
            if param.requires_grad:
                raise RuntimeError(
                    f"Hard Constraint #1 violation: Context Encoder parameter '{name}' has requires_grad=True!"
                )
            continue
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or "norm" in name or "scale" in name or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class ViTDetTrainer:
    """Supervised detection trainer & evaluator for Phase 2 — Branch A.

    Args:
        config: BranchAConfig instance.
        device: Compute device ('cuda' or 'cpu').
        context_encoder: Optional pre-loaded frozen IJepaViTEncoder instance.
    """

    def __init__(
        self,
        config: BranchAConfig,
        device: Optional[Union[str, torch.device]] = None,
        context_encoder: Optional[IJepaViTEncoder] = None,
    ):
        self.config = config
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(config.optimization.seed)
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(config.optimization.seed)
            torch.set_float32_matmul_precision("high")

        # 1. Build Detector (strictly freezing context_encoder)
        self.model = HistoJEPA_ViTDet.from_config(
            config,
            context_encoder=context_encoder,
            device=self.device,
        )

        # 2. Datasets & DataLoaders (Patient-isolated train / val / test splits)
        self.train_dataset = TBYoloDetectionDataset(
            data_root=config.data.root,
            split=config.data.train_split,
            image_size=config.data.image_size,
            mean=config.data.mean,
            std=config.data.std,
            augment=config.data.augment_train,
            return_uint8=(self.device.type == "cuda"),
            seed=config.optimization.seed,
        )
        self.val_dataset = TBYoloDetectionDataset(
            data_root=config.data.root,
            split=config.data.val_split,
            image_size=config.data.image_size,
            mean=config.data.mean,
            std=config.data.std,
            augment=False,
            return_uint8=(self.device.type == "cuda"),
            seed=config.optimization.seed,
        )

        self._gpu_mean = torch.tensor(
            config.data.mean, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self._gpu_std = torch.tensor(
            config.data.std, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        num_workers = config.data.num_workers
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.optimization.micro_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.optimization.micro_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

        # 3. Optimizer & LR Schedule (ONLY SFP + Detection Head parameters)
        micro_batches_per_epoch = max(1, len(self.train_loader))
        self.grad_accum = max(1, config.optimization.grad_accum_steps)
        self.optimizer_steps_per_epoch = max(
            1, (micro_batches_per_epoch + self.grad_accum - 1) // self.grad_accum
        )
        self.total_optimizer_steps = max(
            1, self.optimizer_steps_per_epoch * config.optimization.epochs
        )
        warmup_steps = self.optimizer_steps_per_epoch * config.optimization.warmup_epochs

        self.lr_schedule = WarmupCosineSchedule(
            start_value=config.optimization.lr * 0.1,
            ref_value=config.optimization.lr,
            final_value=config.optimization.min_lr,
            warmup_steps=warmup_steps,
            total_steps=self.total_optimizer_steps,
        )

        param_groups = build_vitdet_param_groups(
            self.model, weight_decay=config.optimization.weight_decay
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.optimization.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_map50: float = -1.0
        self.history: BranchAHistory = BranchAHistory()
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            return (
                images.to(self.device, dtype=torch.float32, non_blocking=True)
                .div_(255.0)
                .sub_(self._gpu_mean)
                .div_(self._gpu_std)
            )
        return images.to(self.device, dtype=torch.float32, non_blocking=True)

    def train_one_epoch(
        self,
        epoch: int,
        progress_callback: Optional[Callable[[BranchAStepInfo], None]] = None,
    ) -> float:
        """Trains SFP and ViTDetHead for one epoch while keeping Context Encoder frozen."""
        self.model.train()
        assert not self.model.context_encoder.training, "Context Encoder must stay in eval() mode!"

        amp_dtype = (
            torch.bfloat16
            if (self.config.optimization.amp_dtype == "bfloat16" and self.device.type == "cuda")
            else torch.float32
        )
        use_amp = self.device.type == "cuda" and amp_dtype != torch.float32

        self.optimizer.zero_grad(set_to_none=True)
        num_micro_batches = len(self.train_loader)

        accum_total = 0.0
        accum_cls = 0.0
        accum_box = 0.0
        accum_ctr = 0.0
        accum_pos = 0
        accum_count = 0
        epoch_losses: List[float] = []

        for micro_idx, (images, targets) in enumerate(self.train_loader):
            is_step_end = (
                (micro_idx + 1) % self.grad_accum == 0
                or (micro_idx + 1) == num_micro_batches
            )

            images = self._prepare_images(images)

            with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                loss_out = self.model(images, targets=targets)
                assert isinstance(loss_out, object) and hasattr(loss_out, "total_loss")
                loss_scaled = loss_out.total_loss / self.grad_accum

            loss_scaled.backward()

            accum_total += float(loss_out.total_loss.detach().item())
            accum_cls += float(loss_out.cls_loss.item())
            accum_box += float(loss_out.box_loss.item())
            accum_ctr += float(loss_out.ctr_loss.item())
            accum_pos += int(loss_out.num_positives)
            accum_count += 1

            if is_step_end:
                scheduled_lr = self.lr_schedule(self.global_step)
                for group in self.optimizer.param_groups:
                    group["lr"] = scheduled_lr

                if self.config.optimization.max_grad_norm > 0.0:
                    trainable_params = [
                        p for p in self.model.parameters() if p.requires_grad and p.grad is not None
                    ]
                    if trainable_params:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, self.config.optimization.max_grad_norm
                        )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                div = max(1, accum_count)
                step_total = accum_total / div
                step_cls = accum_cls / div
                step_box = accum_box / div
                step_ctr = accum_ctr / div
                epoch_losses.append(step_total)

                info = BranchAStepInfo(
                    epoch=epoch,
                    step=self.global_step,
                    total_steps=self.total_optimizer_steps,
                    total_loss=step_total,
                    cls_loss=step_cls,
                    box_loss=step_box,
                    ctr_loss=step_ctr,
                    lr=scheduled_lr,
                    num_positives=accum_pos,
                )
                self.history.append_step(info)
                self.global_step += 1

                accum_total = 0.0
                accum_cls = 0.0
                accum_box = 0.0
                accum_ctr = 0.0
                accum_pos = 0
                accum_count = 0

                if progress_callback is not None:
                    progress_callback(info)

        return float(sum(epoch_losses) / max(1, len(epoch_losses)))

    @torch.no_grad()
    def evaluate_split(
        self,
        split: str = "val",
        score_threshold: Optional[float] = None,
        nms_iou_threshold: Optional[float] = None,
        max_batches: Optional[int] = None,
    ) -> DetectionMetricsReport:
        """Evaluates the detector on `val` or `test` split and returns a DetectionMetricsReport."""
        self.model.eval()

        score_thresh = (
            float(score_threshold)
            if score_threshold is not None
            else self.config.evaluation.score_threshold
        )
        nms_thresh = (
            float(nms_iou_threshold)
            if nms_iou_threshold is not None
            else self.config.evaluation.nms_iou_threshold
        )

        if split == self.config.data.val_split:
            loader = self.val_loader
        else:
            eval_ds = TBYoloDetectionDataset(
                data_root=self.config.data.root,
                split=split,
                image_size=self.config.data.image_size,
                mean=self.config.data.mean,
                std=self.config.data.std,
                augment=False,
                return_uint8=(self.device.type == "cuda"),
            )
            loader = DataLoader(
                eval_ds,
                batch_size=self.config.optimization.micro_batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                collate_fn=detection_collate_fn,
            )

        amp_dtype = (
            torch.bfloat16
            if (self.config.optimization.amp_dtype == "bfloat16" and self.device.type == "cuda")
            else torch.float32
        )
        use_amp = self.device.type == "cuda" and amp_dtype != torch.float32

        all_preds: List[Dict[str, Any]] = []
        all_targets: List[Dict[str, Any]] = []

        for b_idx, (images, targets) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            images = self._prepare_images(images)
            with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                preds = self.model(
                    images,
                    targets=None,
                    score_threshold=score_thresh,
                    nms_iou_threshold=nms_thresh,
                    max_detections=self.config.evaluation.max_detections,
                )

            assert isinstance(preds, list)
            for p, t in zip(preds, targets):
                all_preds.append(
                    {
                        "boxes": p["boxes"].detach().cpu(),
                        "scores": p["scores"].detach().cpu(),
                        "labels": p["labels"].detach().cpu(),
                    }
                )
                all_targets.append(
                    {
                        "boxes": t["boxes"].detach().cpu(),
                        "labels": t["labels"].detach().cpu(),
                    }
                )

        return evaluate_detections(all_preds, all_targets)

    def fit(
        self,
        epochs: Optional[int] = None,
        progress_callback: Optional[Callable[[BranchAStepInfo], None]] = None,
        eval_callback: Optional[Callable[[int, DetectionMetricsReport], None]] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> BranchAHistory:
        """Runs full Branch A training and validation loop across epochs."""
        if resume_from is not None:
            if str(resume_from) == "auto":
                latest_ckpt = find_latest_checkpoint(self.output_dir)
                if latest_ckpt is not None:
                    self.load_checkpoint(latest_ckpt)
            else:
                self.load_checkpoint(resume_from)

        target_epochs = int(epochs) if epochs is not None else self.config.optimization.epochs

        for ep in range(self.current_epoch, target_epochs):
            self.current_epoch = ep
            self.train_one_epoch(ep, progress_callback=progress_callback)

            if (
                (ep + 1) % self.config.evaluation.eval_every_epochs == 0
                or (ep + 1) == target_epochs
            ):
                val_report = self.evaluate_split(split=self.config.data.val_split)
                self.history.append_eval(ep + 1, val_report)

                if val_report.map50 >= self.best_map50:
                    self.best_map50 = val_report.map50
                    self.save_checkpoint("best_map")

                if eval_callback is not None:
                    eval_callback(ep + 1, val_report)

            if (ep + 1) % self.config.output.save_every_epochs == 0 or (ep + 1) == target_epochs:
                self.save_checkpoint(f"ckpt_epoch{ep + 1}")
                self.save_checkpoint("latest")
                prune_checkpoints(self.output_dir, keep_last=self.config.output.keep_last)

            self.save_history()

        self.current_epoch = target_epochs
        return self.history

    def save_checkpoint(self, tag: str) -> Path:
        """Atomically saves trainable SFP + Head weights, optimizer state, and history."""
        state = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "best_map50": self.best_map50,
            "trainable_state_dict": self.model.trainable_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history.to_dict(),
        }
        ckpt_path = self.output_dir / f"{tag}.pt"
        return save_checkpoint(ckpt_path, state)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Restores SFP + Head weights, optimizer state, and telemetry from disk."""
        state = load_checkpoint(checkpoint_path, map_location=self.device)
        if "trainable_state_dict" in state:
            self.model.load_trainable_state_dict(state["trainable_state_dict"])
        elif "model_state_dict" in state:
            self.model.load_trainable_state_dict(state["model_state_dict"])

        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        self.current_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_map50 = float(state.get("best_map50", -1.0))

        if "history" in state:
            self.history = BranchAHistory.from_dict(state["history"])

    def save_history(self) -> Path:
        hist_path = self.output_dir / "history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history.to_dict(), f, indent=2)
        return hist_path
