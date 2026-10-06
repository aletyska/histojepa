"""I-JEPA Training Engine for Phase 1 self-supervised pre-training.

Implements micro-batch execution, AMP (bfloat16), gradient accumulation,
optimizer parameter groups with weight-decay exclusions, scheduled EMA updates,
checkpoint save/resume, and frozen Context Encoder export for Phase 2.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from histojepa.data.ijepa_dataset import ProcessedImageDataset
from histojepa.models.ijepa.config import Phase1Config
from histojepa.models.ijepa.ijepa import IJEPA
from histojepa.models.ijepa.masking import MultiBlockMaskCollator
from histojepa.models.ijepa.schedulers import LinearSchedule, WarmupCosineSchedule
from histojepa.models.ijepa.vit_encoder import save_frozen_context_encoder
from histojepa.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


@dataclass
class StepInfo:
    """Telemetry report emitted after each optimizer step."""

    epoch: int
    step: int
    total_steps: int
    loss: float
    lr: float
    wd: float
    momentum: float
    target_std: float
    pred_std: float


@dataclass
class TrainingHistory:
    """Historical telemetry for pre-training run."""

    steps: List[int] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)
    wds: List[float] = field(default_factory=list)
    momentums: List[float] = field(default_factory=list)
    target_stds: List[float] = field(default_factory=list)
    pred_stds: List[float] = field(default_factory=list)

    def append(self, info: StepInfo) -> None:
        self.steps.append(info.step)
        self.epochs.append(info.epoch)
        self.losses.append(info.loss)
        self.lrs.append(info.lr)
        self.wds.append(info.wd)
        self.momentums.append(info.momentum)
        self.target_stds.append(info.target_std)
        self.pred_stds.append(info.pred_std)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingHistory":
        return cls(**data)


def build_optimizer_param_groups(
    model: IJEPA,
    base_wd: float,
) -> List[Dict[str, Any]]:
    """Partitions trainable parameters into weight-decay and no-decay groups.

    Excludes 1D biases, LayerNorm parameters, and mask tokens from weight decay.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, 1D norm parameters, and mask token vector bypass weight decay
        if param.ndim <= 1 or "norm" in name or "mask_token" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": base_wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class IJepaTrainer:
    """Pre-training orchestrator for Phase 1 I-JEPA.

    Args:
        config: Phase1Config instance.
        device: PyTorch device ('cuda' or 'cpu'). If None, automatically detected.
    """

    def __init__(
        self,
        config: Phase1Config,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set manual seeds for reproducibility
        torch.manual_seed(config.optimization.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.optimization.seed)

        # 1. Instantiate Model
        self.model = IJEPA(config).to(self.device)

        # Optional gradient checkpointing
        if config.optimization.gradient_checkpointing and hasattr(
            self.model.context_encoder.vit, "gradient_checkpointing_enable"
        ):
            self.model.context_encoder.vit.gradient_checkpointing_enable()

        # 2. Dataset, Mask Collator, and DataLoader
        self.dataset = ProcessedImageDataset(
            data_root=config.data.root,
            split=config.data.split,
            image_size=config.data.image_size,
            mean=config.data.mean,
            std=config.data.std,
        )

        p_sz = config.model_encoder.patch_size
        img_sz = config.data.image_size
        grid_hw = (img_sz // p_sz, img_sz // p_sz)

        self.collator = MultiBlockMaskCollator(
            grid_size=grid_hw,
            num_targets=config.masking.num_targets,
            target_scale=config.masking.target_scale,
            target_aspect_ratio=config.masking.target_aspect_ratio,
            context_scale=config.masking.context_scale,
            context_aspect_ratio=config.masking.context_aspect_ratio,
            min_context_keep=config.masking.min_context_keep,
            seed=config.optimization.seed,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.optimization.micro_batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=self.collator,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        # 3. Steps calculation & Optimizer
        micro_batches_per_epoch = len(self.dataloader)
        self.grad_accum = max(1, config.optimization.grad_accum_steps)
        self.optimizer_steps_per_epoch = max(1, micro_batches_per_epoch // self.grad_accum)
        self.total_optimizer_steps = self.optimizer_steps_per_epoch * config.optimization.epochs
        warmup_steps = self.optimizer_steps_per_epoch * config.optimization.warmup_epochs

        # Scaled learning rates
        start_lr, ref_lr, final_lr = config.optimization.get_effective_lrs()

        # Optimization Schedulers
        self.lr_schedule = WarmupCosineSchedule(
            start_value=start_lr,
            ref_value=ref_lr,
            final_value=final_lr,
            warmup_steps=warmup_steps,
            total_steps=self.total_optimizer_steps,
        )

        self.wd_schedule = LinearSchedule(
            start_value=config.optimization.start_wd,
            final_value=config.optimization.final_wd,
            total_steps=self.total_optimizer_steps,
        )

        self.momentum_schedule = LinearSchedule(
            start_value=config.optimization.ema_momentum[0],
            final_value=config.optimization.ema_momentum[1],
            total_steps=self.total_optimizer_steps,
        )

        # Optimizer with weight-decay groups
        param_groups = build_optimizer_param_groups(
            self.model, base_wd=config.optimization.start_wd
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=start_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Tracking state
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_loss: float = float("inf")
        self.history: TrainingHistory = TrainingHistory()
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(
        self,
        epoch: int,
        progress_callback: Optional[Callable[[StepInfo], None]] = None,
    ) -> float:
        """Trains for a single epoch across micro-batches and gradient accumulation steps."""
        self.model.train()
        accum_loss = 0.0
        epoch_losses: List[float] = []

        # AMP precision setup
        amp_dtype = (
            torch.bfloat16
            if (self.config.optimization.amp_dtype == "bfloat16" and self.device.type == "cuda")
            else torch.float32
        )
        use_amp = self.device.type == "cuda" and amp_dtype != torch.float32

        self.optimizer.zero_grad(set_to_none=True)

        for micro_idx, (images, ctx_idx, tgt_idx_list) in enumerate(self.dataloader):
            images = images.to(self.device, non_blocking=True)
            ctx_idx = ctx_idx.to(self.device, non_blocking=True)
            tgt_idx_list = [t.to(self.device, non_blocking=True) for t in tgt_idx_list]

            with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                output = self.model(images, ctx_idx, tgt_idx_list)
                loss_scaled = output.loss / self.grad_accum

            loss_scaled.backward()
            accum_loss += float(output.loss.detach().item())

            # When gradient accumulation completes, execute optimizer step
            if (micro_idx + 1) % self.grad_accum == 0 or (micro_idx + 1) == len(self.dataloader):
                cur_step = self.global_step
                scheduled_lr = self.lr_schedule(cur_step)
                scheduled_wd = self.wd_schedule(cur_step)
                scheduled_momentum = self.momentum_schedule(cur_step)

                # Update optimizer hyperparameters
                for group in self.optimizer.param_groups:
                    group["lr"] = scheduled_lr
                    if group["weight_decay"] > 0.0:
                        group["weight_decay"] = scheduled_wd

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Target encoder EMA update
                self.model.momentum_update(scheduled_momentum)

                step_loss = accum_loss / min(self.grad_accum, (micro_idx % self.grad_accum) + 1)
                epoch_losses.append(step_loss)
                accum_loss = 0.0

                step_info = StepInfo(
                    epoch=epoch,
                    step=self.global_step,
                    total_steps=self.total_optimizer_steps,
                    loss=step_loss,
                    lr=scheduled_lr,
                    wd=scheduled_wd,
                    momentum=scheduled_momentum,
                    target_std=output.target_std,
                    pred_std=output.pred_std,
                )
                self.history.append(step_info)
                self.global_step += 1

                if progress_callback is not None:
                    progress_callback(step_info)

        avg_epoch_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        return avg_epoch_loss

    def fit(
        self,
        epochs: Optional[int] = None,
        progress_callback: Optional[Callable[[StepInfo], None]] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> TrainingHistory:
        """Runs full pre-training loop.

        Args:
            epochs: Total epochs to train for. Defaults to config.optimization.epochs.
            progress_callback: Optional callback invoked after each optimizer step.
            resume_from: Optional checkpoint path to resume from. If 'auto', loads latest.

        Returns:
            TrainingHistory instance.
        """
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
            avg_loss = self.train_one_epoch(ep, progress_callback=progress_callback)

            # Checkpoint saving
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
                self.save_checkpoint("best_loss")

            if (ep + 1) % self.config.output.save_every_epochs == 0 or (ep + 1) == target_epochs:
                self.save_checkpoint(f"ckpt_epoch{ep + 1}")
                self.save_checkpoint("latest")
                prune_checkpoints(self.output_dir, keep_last=self.config.output.keep_last)

            # Save training history JSON
            self.save_history()

        self.current_epoch = target_epochs
        return self.history

    def save_checkpoint(self, tag: str) -> Path:
        """Saves current training state to outputs/phase1_ijepa/{tag}.pt."""
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history.to_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)

        ckpt_path = self.output_dir / f"{tag}.pt"
        return save_checkpoint(ckpt_path, state)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Restores training state from checkpoint file."""
        state = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss", float("inf"))

        if "history" in state:
            self.history = TrainingHistory.from_dict(state["history"])

        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])

        if "cuda_rng_state" in state and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng_state"], self.device)

    def save_history(self) -> Path:
        """Persists training history to history.json."""
        hist_path = self.output_dir / "history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history.to_dict(), f, indent=2)
        return hist_path

    def export_frozen_context_encoder(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves and freezes the Context Encoder for downstream Phase 2 branches.

        Ref: AGENTS.md Hard Constraint #1.
        """
        dest_dir = Path(output_dir or self.config.output.export_dir)
        meta = {
            "trained_epochs": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            meta.update(metadata)

        return save_frozen_context_encoder(
            encoder=self.model.context_encoder,
            output_dir=dest_dir,
            metadata=meta,
        )
