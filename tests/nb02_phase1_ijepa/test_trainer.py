"""Unit tests for IJepaTrainer training loop, checkpoint resume, and frozen export.

Bound to: notebooks/02_Phase_1_IJEPA_PreTrain.py
Pytest Marker: @pytest.mark.nb02
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from histojepa.models.ijepa.config import (
    DataConfig,
    EncoderConfig,
    MaskingConfig,
    OptimConfig,
    OutputConfig,
    Phase1Config,
    PredictorConfig,
)
from histojepa.models.ijepa.trainer import IJepaTrainer
from histojepa.models.ijepa.vit_encoder import load_frozen_context_encoder


pytestmark = pytest.mark.nb02


def test_trainer_fit_and_checkpoint_resume(tmp_path: Path):
    # Setup mock data directory
    data_dir = tmp_path / "data"
    proc_dir = data_dir / "train" / "processed"
    proc_dir.mkdir(parents=True)

    for i in range(4):
        img = np.full((64, 64, 3), 100 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(proc_dir / f"{i}.png"), img)

    out_dir = tmp_path / "output"
    cfg = Phase1Config(
        data=DataConfig(
            root=str(data_dir),
            split="train",
            image_size=64,
            num_workers=0,
        ),
        model_encoder=EncoderConfig(
            patch_size=16,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        ),
        model_predictor=PredictorConfig(
            embed_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        ),
        masking=MaskingConfig(
            num_targets=2,
            min_context_keep=4,
        ),
        optimization=OptimConfig(
            epochs=2,
            micro_batch_size=2,
            grad_accum_steps=1,
            warmup_epochs=1,
            amp_dtype="float32",
        ),
        output=OutputConfig(
            dir=str(out_dir),
            save_every_epochs=1,
            export_dir=str(out_dir / "context_encoder_frozen"),
        ),
    )

    trainer = IJepaTrainer(cfg, device="cpu")

    # Capture initial target encoder weights to test EMA change
    init_tgt_weight = trainer.model.target_encoder.vit.embeddings.patch_embeddings.projection.weight.clone()

    history = trainer.fit(epochs=1)
    assert len(history.losses) >= 1
    assert torch.isfinite(torch.tensor(history.losses[-1]))

    # Target encoder weights should have updated via EMA
    post_tgt_weight = trainer.model.target_encoder.vit.embeddings.patch_embeddings.projection.weight
    assert not torch.equal(init_tgt_weight, post_tgt_weight)

    # Checkpoint was created
    assert (out_dir / "ckpt_epoch1.pt").exists()
    assert (out_dir / "history.json").exists()

    # Resume training for epoch 2
    trainer_resumed = IJepaTrainer(cfg, device="cpu")
    trainer_resumed.fit(epochs=2, resume_from=out_dir / "ckpt_epoch1.pt")
    assert trainer_resumed.current_epoch == 2

    # Export frozen context encoder test
    export_path = trainer_resumed.export_frozen_context_encoder()
    assert (export_path / "manifest.json").exists()

    # Load frozen context encoder back and verify
    frozen_encoder = load_frozen_context_encoder(export_path, device="cpu")
    assert not frozen_encoder.training
    for p in frozen_encoder.parameters():
        assert not p.requires_grad
