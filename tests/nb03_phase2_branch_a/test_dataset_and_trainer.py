"""Unit tests for TBYoloDetectionDataset, ViTDetHead, and ViTDetTrainer smoke run.

Bound to: notebooks/03_Phase_2_Branch_A.py
Pytest Marker: @pytest.mark.nb03
"""

from pathlib import Path
import pytest
import torch

from histojepa.data.detection_dataset import (
    TBYoloDetectionDataset,
    detection_collate_fn,
)
from histojepa.models.ijepa.config import EncoderConfig
from histojepa.models.ijepa.vit_encoder import (
    IJepaViTEncoder,
    build_ijepa_vit,
    save_frozen_context_encoder,
)
from histojepa.models.vitdet.config import BranchAConfig
from histojepa.models.vitdet.trainer import ViTDetTrainer

pytestmark = pytest.mark.nb03


def test_tb_yolo_detection_dataset_and_collate(temp_dataset_dir: Path):
    ds = TBYoloDetectionDataset(
        data_root=temp_dataset_dir,
        split="train",
        image_size=128,
        augment=True,
    )
    assert len(ds) == 2
    img, tgt = ds[0]
    assert img.shape == (3, 128, 128)
    assert tgt["boxes"].shape == (1, 4)
    assert tgt["labels"].shape == (1,)

    batch_imgs, batch_tgts = detection_collate_fn([ds[0], ds[1]])
    assert batch_imgs.shape == (2, 3, 128, 128)
    assert len(batch_tgts) == 2


def test_vitdet_trainer_one_epoch_smoke(temp_dataset_dir: Path, tmp_path: Path):
    # Export a tiny frozen context encoder
    enc_cfg = EncoderConfig(
        patch_size=16,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    vit = build_ijepa_vit(enc_cfg, image_size=128)
    frozen_dir = tmp_path / "frozen_enc"
    save_frozen_context_encoder(IJepaViTEncoder(vit), frozen_dir)

    out_dir = tmp_path / "branch_a_out"
    cfg = BranchAConfig.from_yaml(
        "configs/branch_a_vitdet.yaml",
        overrides={
            "data": {
                "root": str(temp_dataset_dir),
                "image_size": 128,
                "num_workers": 0,
            },
            "backbone": {
                "frozen_encoder_dir": str(frozen_dir),
                "hidden_size": 64,
                "patch_size": 16,
            },
            "sfp": {
                "in_channels": 64,
                "out_channels": 32,
            },
            "head": {
                "in_channels": 32,
                "num_classes": 1,
                "num_convs": 2,
            },
            "optimization": {
                "epochs": 1,
                "micro_batch_size": 2,
                "grad_accum_steps": 1,
                "warmup_epochs": 1,
            },
            "output": {
                "dir": str(out_dir),
            },
        },
    )

    trainer = ViTDetTrainer(config=cfg, device="cpu")
    history = trainer.fit(epochs=1)

    assert len(history.steps) == 1
    assert len(history.eval_epochs) == 1
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "best_map.pt").exists()
    assert (out_dir / "history.json").exists()

    # Verify test split evaluation also runs cleanly
    test_report = trainer.evaluate_split(split="test")
    assert test_report.total_gt == 2
