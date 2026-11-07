import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="full",
    app_title="03 - HistoJEPA Phase 2 - Branch A - Supervised Learning",
)


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import marimo as mo

    from histojepa.data import (
        TBYoloDetectionDataset,
        load_yolo_boxes_and_labels,
    )
    from histojepa.evaluation import (
        DetectionMetricsReport,
        compute_box_iou,
        evaluate_detections,
    )
    from histojepa.models.ijepa import load_frozen_context_encoder
    from histojepa.models.vitdet import (
        BranchAConfig,
        HistoJEPA_ViTDet,
        SimpleFeaturePyramid,
        ViTDetHead,
        ViTDetTrainer,
    )
    from histojepa.utils import draw_detection_comparison, load_checkpoint

    return (
        BranchAConfig,
        HistoJEPA_ViTDet,
        Path,
        SimpleFeaturePyramid,
        ViTDetHead,
        ViTDetTrainer,
        compute_box_iou,
        cv2,
        draw_detection_comparison,
        json,
        load_checkpoint,
        load_frozen_context_encoder,
        load_yolo_boxes_and_labels,
        mo,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎯 HistoJEPA Phase 2 — Branch A: Supervised Tumor Bud Detection (ViTDet SFP)
    **Notebook:** `03_Phase_2_Branch_A.py`
    **Backbone:** Strictly Frozen Phase 1 I-JEPA Context Encoder (`outputs/phase1_ijepa/context_encoder_frozen/`)
    **Reference:** Li et al., *"Exploring Plain Vision Transformer Backbones for Object Detection"* (**ViTDet**), ECCV 2022 (`papers/vitdet/vitdet.md`).

    ---

    ### 🏛️ Architectural Design & Frozen Backbone Discipline
    Unlike traditional hierarchical object detectors (e.g., ResNet/Swin + FPN), **ViTDet** demonstrates that a plain, single-scale Vision Transformer (stride $16$) is sufficient for multi-scale object detection when paired with a **Simple Feature Pyramid (SFP)**:

    1. **Strictly Frozen I-JEPA Context Encoder ($f_\theta$):**
       Each $1280 \times 1280$ histopathology tile is tokenized into $80 \times 80 = 6{,}400$ patches ($16 \times 16\text{ px}$) and encoded under `torch.no_grad()` (`param.requires_grad = False`, `eval()` mode) into a single-scale spatial representation grid $F_{1/16} \in \mathbb{R}^{B \times 768 \times 80 \times 80}$.
    2. **ViTDet Simple Feature Pyramid (SFP — §3 & App. 0.A.2):**
       Constructs multi-scale pyramid levels $\{P_2, P_3, P_4, P_5\}$ directly in parallel from $F_{1/16}$ without top-down or lateral FPN connections:
       - **$P_2$ (Stride 4, $320 \times 320 \times 256$):** Two $2 \times 2$ stride-2 transposed convolutions (first with `LayerNorm2d` + `GELU`) $\to 1\times 1\text{ Conv+LN} \to 3\times 3\text{ Conv+LN}$.
       - **$P_3$ (Stride 8, $160 \times 160 \times 256$):** One $2 \times 2$ stride-2 transposed convolution $\to 1\times 1\text{ Conv+LN} \to 3\times 3\text{ Conv+LN}$.
       - **$P_4$ (Stride 16, $80 \times 80 \times 256$):** Direct $1\times 1\text{ Conv+LN} \to 3\times 3\text{ Conv+LN}$ on the ViT feature map.
       - **$P_5$ (Stride 32, $40 \times 40 \times 256$):** Stride-2 $2 \times 2$ MaxPool $\to 1\times 1\text{ Conv+LN} \to 3\times 3\text{ Conv+LN}$.
    3. **Multi-Scale Detection Head (App. 0.A.1 & 0.A.2):**
       Shared 4-layer `Conv3x3 + LayerNorm2d + GELU` classification and box regression towers trained with **Sigmoid Focal Loss**, **Generalized IoU (GIoU) Loss**, and **Centerness BCE Loss**:
       $$\mathcal{L}_{\text{Branch A}} = \mathcal{L}_{\text{Focal}}(\hat{p}, y) + \mathcal{L}_{\text{GIoU}}(\hat{b}, b) + 0.5\,\mathcal{L}_{\text{Centerness}}(\hat{c}, c)$$
    """)
    return


@app.cell(hide_code=True)
def _(BranchAConfig, Path):
    default_cfg_path = Path("configs/branch_a_vitdet.yaml")
    base_config = BranchAConfig.from_yaml(default_cfg_path)
    return base_config, default_cfg_path


@app.cell(hide_code=True)
def _(
    Path,
    SimpleFeaturePyramid,
    ViTDetHead,
    base_config,
    json,
    load_frozen_context_encoder,
    load_yolo_boxes_and_labels,
    mo,
):
    enc_dir = Path(base_config.backbone.frozen_encoder_dir)
    manifest_path = enc_dir / "manifest.json"
    manifest_info = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest_info = json.load(mf)

    frozen_encoder = load_frozen_context_encoder(enc_dir, device="cpu")
    all_frozen = all(not p.requires_grad for p in frozen_encoder.parameters())
    is_eval_mode = not frozen_encoder.training
    ctx_param_count = sum(p.numel() for p in frozen_encoder.parameters())

    temp_sfp = SimpleFeaturePyramid(
        in_channels=base_config.sfp.in_channels,
        out_channels=base_config.sfp.out_channels,
    )
    temp_head = ViTDetHead(
        in_channels=base_config.head.in_channels,
        num_classes=base_config.head.num_classes,
        num_convs=base_config.head.num_convs,
    )
    sfp_param_count = sum(p.numel() for p in temp_sfp.parameters() if p.requires_grad)
    head_param_count = sum(p.numel() for p in temp_head.parameters() if p.requires_grad)
    total_trainable_count = sfp_param_count + head_param_count

    # Inventory of patient-isolated splits (train, val, test)
    split_rows = []
    for split_name, role_desc in [
        ("train", "Supervised training of SFP + Detection Head (with flips)"),
        ("val", "Per-epoch validation, checkpoint selection (`best_map.pt`)"),
        ("test", "Held-out final benchmark evaluation (unseen patients)"),
    ]:
        proc_dir = Path(base_config.data.root) / split_name / "processed"
        lbl_dir = Path(base_config.data.root) / split_name / "labels"
        imgs = sorted(list(proc_dir.glob("*.png"))) if proc_dir.exists() else []
        box_total = 0
        for img_p in imgs:
            boxes, _ = load_yolo_boxes_and_labels(
                lbl_dir / f"{img_p.stem}.txt",
                img_width=base_config.data.image_size,
                img_height=base_config.data.image_size,
            )
            box_total += len(boxes)
        split_rows.append((split_name, len(imgs), box_total, role_desc))

    meta_meta = manifest_info.get("metadata", {})
    phase1_epoch = meta_meta.get("epoch", "N/A")
    phase1_loss = meta_meta.get("best_loss", 0.0)
    phase1_loss_str = f"{phase1_loss:.5f}" if isinstance(phase1_loss, float) else str(phase1_loss)

    mo.md(
        f"""
        ### ❄️ Frozen Phase 1 Context Encoder & Patient-Split Inventory

        | Module | Status | Total Parameters | Trainable Parameters | Output Tensor Shape ($1280\\times 1280$) |
        | :--- | :--- | :---: | :---: | :--- |
        | **I-JEPA Context Encoder ($f_\\theta$)** | `eval()={is_eval_mode}`, `frozen={all_frozen}` (Epoch {phase1_epoch}, Loss `{phase1_loss_str}`) | **{ctx_param_count / 1e6:.2f}M** | **0 (Strictly Frozen)** | `(B, 768, 80, 80)` (Stride 16) |
        | **ViTDet Simple Feature Pyramid (SFP)** | 🔥 Trainable | **{sfp_param_count / 1e6:.2f}M** | **{sfp_param_count / 1e6:.2f}M** | $P_2: 320^2, P_3: 160^2, P_4: 80^2, P_5: 40^2$ (`256` ch) |
        | **ViTDet Multi-Scale Detection Head** | 🔥 Trainable | **{head_param_count / 1e6:.2f}M** | **{head_param_count / 1e6:.2f}M** | Class Logits, Box `(l,t,r,b)`, Centerness |
        | **Total HistoJEPA-ViTDet System** | Ready | **{(ctx_param_count + total_trainable_count) / 1e6:.2f}M** | **{total_trainable_count / 1e6:.2f}M ({total_trainable_count:,})** | Multi-Scale NMS Bounding Boxes |

        #### 📂 TB-YOLO Patient-Isolated Dataset Splits
        | Split | Processed Tiles | Ground-Truth Tumor Buds | Role in Branch A |
        | :--- | :---: | :---: | :--- |
        | **`{split_rows[0][0]}`** | **{split_rows[0][1]:,}** | **{split_rows[0][2]:,}** | {split_rows[0][3]} |
        | **`{split_rows[1][0]}`** | **{split_rows[1][1]:,}** | **{split_rows[1][2]:,}** | {split_rows[1][3]} |
        | **`{split_rows[2][0]}`** | **{split_rows[2][1]:,}** | **{split_rows[2][2]:,}** | {split_rows[2][3]} |
        """
    )
    return (frozen_encoder,)


@app.cell(hide_code=True)
def _(Path, base_config, mo):
    val_proc_dir = Path(base_config.data.root) / "val" / "processed"
    sample_files = sorted([p.name for p in val_proc_dir.glob("*.png")])[:40] if val_proc_dir.exists() else ["None"]

    pyramid_tile_ui = mo.ui.dropdown(
        options=sample_files,
        value=sample_files[0],
        label="**Select Tile (`val/processed`):**",
    )
    inspect_sfp_btn = mo.ui.run_button(
        label="🔬 Compute Frozen ViT + SFP Feature Pyramid",
    )

    mo.md(
        f"""
        ### 🔍 Interactive Frozen Context Encoder $\\to$ Simple Feature Pyramid (SFP) Inspector
        Select a histopathology tile and click to run a forward pass through the frozen I-JEPA Context Encoder and inspect the resulting multi-scale feature pyramid ($P_2, P_3, P_4, P_5$):

        {mo.hstack([pyramid_tile_ui, inspect_sfp_btn], justify="start")}
        """
    )
    return inspect_sfp_btn, pyramid_tile_ui


@app.cell(hide_code=True)
def _(
    Path,
    SimpleFeaturePyramid,
    base_config,
    cv2,
    draw_detection_comparison,
    frozen_encoder,
    inspect_sfp_btn,
    load_yolo_boxes_and_labels,
    mo,
    np,
    plt,
    pyramid_tile_ui,
    torch,
):
    sfp_preview_out = mo.md(
        "*Click **'🔬 Compute Frozen ViT + SFP Feature Pyramid'** above to run a live feature extraction.*"
    )

    if inspect_sfp_btn.value and pyramid_tile_ui.value != "None":
        tile_path = Path(base_config.data.root) / "val" / "processed" / pyramid_tile_ui.value
        lbl_path = Path(base_config.data.root) / "val" / "labels" / f"{tile_path.stem}.txt"

        if tile_path.exists():
            bgr = cv2.imread(str(tile_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt_boxes, _ = load_yolo_boxes_and_labels(
                lbl_path,
                img_width=base_config.data.image_size,
                img_height=base_config.data.image_size,
            )

            enc_device = next(frozen_encoder.parameters()).device
            mean_arr = np.array(base_config.data.mean, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.array(base_config.data.std, dtype=np.float32).reshape(1, 1, 3)
            norm_rgb = (rgb.astype(np.float32) / 255.0 - mean_arr) / std_arr
            img_tensor = (
                torch.from_numpy(norm_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
                .to(enc_device)
            )

            with torch.no_grad():
                tokens = frozen_encoder(img_tensor, keep_idx=None)
                spatial_768 = frozen_encoder.to_spatial(tokens, grid_hw=(80, 80))
                sfp_demo = (
                    SimpleFeaturePyramid(
                        in_channels=base_config.sfp.in_channels,
                        out_channels=base_config.sfp.out_channels,
                    )
                    .to(enc_device)
                    .eval()
                )
                pyr_maps = sfp_demo(spatial_768)

            fig_pyr, axes_grid = plt.subplots(2, 3, figsize=(15, 9.5))
            axes_pyr = axes_grid.flatten()

            # Panel 0: Tile + Ground Truth boxes
            draw_detection_comparison(
                axes_pyr[0],
                rgb,
                gt_boxes=gt_boxes,
                pred_boxes=np.zeros((0, 4), dtype=np.float32),
            )
            axes_pyr[0].set_title(
                f"1. Tile {pyramid_tile_ui.value}\n({len(gt_boxes)} Ground-Truth Tumor Buds)",
                fontsize=12,
                fontweight="bold",
                pad=8,
            )

            # Panel 1: Frozen I-JEPA L2 Norm Map (80x80)
            vit_norm = torch.norm(spatial_768[0], p=2, dim=0).cpu().numpy()
            axes_pyr[1].imshow(vit_norm, cmap="magma")
            axes_pyr[1].set_title(
                f"2. Frozen I-JEPA Spatial Grid\nShape: {tuple(spatial_768.shape)} (Stride 16)",
                fontsize=12,
                fontweight="bold",
                pad=8,
            )
            axes_pyr[1].axis("off")

            # Panels 2-5: P2, P3, P4, P5 feature maps
            for idx, (lvl_key, stride_val) in enumerate([("p2", 4), ("p3", 8), ("p4", 16), ("p5", 32)]):
                fmap = pyr_maps[lvl_key]
                fmap_norm = torch.norm(fmap[0], p=2, dim=0).cpu().numpy()
                ax_p = axes_pyr[idx + 2]
                ax_p.imshow(fmap_norm, cmap="viridis")
                ax_p.set_title(
                    f"{idx + 3}. SFP {lvl_key.upper()} (Stride {stride_val})\nShape: {tuple(fmap.shape)}",
                    fontsize=12,
                    fontweight="bold",
                    pad=8,
                )
                ax_p.axis("off")

            plt.tight_layout(pad=1.5)
            sfp_preview_out = mo.as_html(fig_pyr)
            plt.close(fig_pyr)

    sfp_preview_out
    return


@app.cell(hide_code=True)
def _(Path, base_config, mo, torch):
    out_dir_path = Path(base_config.output.dir)
    avail_ckpts = ["None"]
    if out_dir_path.exists():
        found_pts = sorted([p.name for p in out_dir_path.glob("*.pt")])
        for priority_name in ("best_map.pt", "latest.pt"):
            if priority_name in found_pts:
                avail_ckpts.append(priority_name)
        for ckpt_fn in found_pts:
            if ckpt_fn not in avail_ckpts:
                avail_ckpts.append(ckpt_fn)

    epochs_ui = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=base_config.optimization.epochs,
        label="**Training Epochs:**",
    )
    micro_bs_ui = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=base_config.optimization.micro_batch_size,
        label="**Micro-Batch Size:**",
    )
    grad_accum_ui = mo.ui.slider(
        start=1,
        stop=16,
        step=1,
        value=base_config.optimization.grad_accum_steps,
        label="**Grad Accum Steps:**",
    )
    lr_ui = mo.ui.dropdown(
        options=["5e-5", "1e-4", "2e-4", "5e-4"],
        value="1e-4",
        label="**Peak Learning Rate:**",
    )
    num_workers_ui = mo.ui.slider(
        start=0,
        stop=8,
        step=1,
        value=base_config.data.num_workers,
        label="**DataLoader Workers:**",
    )
    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device_ui = mo.ui.dropdown(
        options=device_options,
        value="cuda" if torch.cuda.is_available() else "cpu",
        label="**Compute Device:**",
    )
    resume_ui = mo.ui.dropdown(
        options=avail_ckpts,
        value="None",
        label="**Resume Checkpoint:**",
    )
    train_btn = mo.ui.run_button(
        label="🚀 Start / Resume Branch A ViTDet Training",
    )

    mo.md(
        f"""
        ### ⚙️ Branch A Training Configuration & Execution
        {mo.hstack([epochs_ui, micro_bs_ui, grad_accum_ui, num_workers_ui], justify="start")}

        {mo.hstack([lr_ui, device_ui, resume_ui, train_btn], justify="start")}
        """
    )
    return (
        device_ui,
        epochs_ui,
        grad_accum_ui,
        lr_ui,
        micro_bs_ui,
        num_workers_ui,
        resume_ui,
        train_btn,
    )


@app.cell(hide_code=True)
def _(
    BranchAConfig,
    Path,
    ViTDetTrainer,
    default_cfg_path,
    device_ui,
    epochs_ui,
    grad_accum_ui,
    lr_ui,
    micro_bs_ui,
    mo,
    num_workers_ui,
    resume_ui,
    train_btn,
):
    active_trainer = None
    training_banner = mo.md(
        "*Configure hyperparameters above and click **'🚀 Start / Resume Branch A ViTDet Training'** to train the Simple Feature Pyramid and Detection Head.*"
    )

    if train_btn.value:
        overrides = {
            "data": {
                "num_workers": num_workers_ui.value,
            },
            "optimization": {
                "epochs": epochs_ui.value,
                "micro_batch_size": micro_bs_ui.value,
                "grad_accum_steps": grad_accum_ui.value,
                "lr": float(lr_ui.value),
            },
        }
        run_cfg = BranchAConfig.from_yaml(default_cfg_path, overrides=overrides)
        active_trainer = ViTDetTrainer(config=run_cfg, device=device_ui.value)

        resume_target = None
        if resume_ui.value != "None":
            candidate = Path(run_cfg.output.dir) / resume_ui.value
            if candidate.exists():
                resume_target = candidate

        total_steps = active_trainer.total_optimizer_steps
        val_state = {"summary": "Pending epoch end..."}

        with mo.status.progress_bar(
            total=total_steps,
            title="Branch A ViTDet Training (Frozen I-JEPA Backbone)",
            subtitle="Initializing...",
            show_rate=True,
            show_eta=True,
            remove_on_exit=False,
        ) as bar:
            def on_step(info):
                bar.update(
                    increment=1,
                    subtitle=(
                        f"Epoch {info.epoch + 1}/{epochs_ui.value} | "
                        f"Loss: {info.total_loss:.4f} (Cls: {info.cls_loss:.3f}, Box: {info.box_loss:.3f}) | "
                        f"LR: {info.lr:.2e} | {val_state['summary']}"
                    ),
                )

            def on_eval(ep_num, val_rep):
                val_state["summary"] = f"Val mAP@50: {val_rep.map50:.4f} | F1: {val_rep.f1:.4f}"

            history = active_trainer.fit(
                epochs=epochs_ui.value,
                progress_callback=on_step,
                eval_callback=on_eval,
                resume_from=resume_target,
            )

        final_loss = history.total_losses[-1] if history.total_losses else 0.0
        best_val_map50 = active_trainer.best_map50 if active_trainer.best_map50 >= 0.0 else 0.0
        final_val_f1 = history.val_f1[-1] if history.val_f1 else 0.0

        training_banner = mo.md(
            f"""
            ✅ **Branch A Training Complete!**  
            - Epochs Completed: **{epochs_ui.value}** (`{len(history.steps):,}` optimizer steps)  
            - Final Training Loss: **`{final_loss:.4f}`**  
            - Best Validation `mAP@50`: **`{best_val_map50:.4f}`** (Final Val `F1`: **`{final_val_f1:.4f}`**)  
            - Saved Checkpoints: `{run_cfg.output.dir}/best_map.pt` & `{run_cfg.output.dir}/latest.pt`
            """
        )

    training_banner
    return (active_trainer,)


@app.cell(hide_code=True)
def _(Path, base_config, mo):
    # Save / Export Trained ViTDet Model Controls
    _save_out_dir = Path(base_config.output.dir)
    _save_source_options = ["In-Memory (Current Trained Model)"]
    if _save_out_dir.exists():
        for _pt in sorted([_p.name for _p in _save_out_dir.glob("*.pt")]):
            _save_source_options.append(_pt)

    save_vitdet_source_ui = mo.ui.dropdown(
        options=_save_source_options,
        value=_save_source_options[0],
        label="**Source Model State:**",
    )
    save_vitdet_tag_ui = mo.ui.text(
        value="vitdet_trained_final",
        label="**Save Name / Bundle Folder:**",
    )
    save_vitdet_btn = mo.ui.run_button(
        label="💾 Save Trained ViTDet Model to Disk",
    )

    mo.md(
        f"""
        ### 💾 Save & Export Trained ViTDet Model to Disk
        *Note: Automatic checkpoints (`best_map.pt` and `latest.pt`) are already persisted in `{_save_out_dir}/`. Use the control below to explicitly save the in-memory `active_trainer` model (or any checkpoint) as a standalone export bundle on disk:*

        {mo.hstack([save_vitdet_source_ui, save_vitdet_tag_ui, save_vitdet_btn], justify="start")}
        """
    )
    return save_vitdet_btn, save_vitdet_source_ui, save_vitdet_tag_ui


@app.cell(hide_code=True)
def _(
    Path,
    active_trainer,
    base_config,
    json,
    load_checkpoint,
    mo,
    save_vitdet_btn,
    save_vitdet_source_ui,
    save_vitdet_tag_ui,
    torch,
):
    _save_status_card = mo.md(
        "*Click **'💾 Save Trained ViTDet Model to Disk'** above to export the trained Simple Feature Pyramid (SFP) + Detection Head weights and manifest.*"
    )

    if save_vitdet_btn.value:
        _out_root = Path(base_config.output.dir)
        _tag_clean = (save_vitdet_tag_ui.value or "vitdet_trained_final").strip().replace(".pt", "")
        _bundle_dir = _out_root / _tag_clean
        _bundle_dir.mkdir(parents=True, exist_ok=True)

        _src_choice = save_vitdet_source_ui.value
        if _src_choice == "In-Memory (Current Trained Model)" and active_trainer is not None:
            _ckpt_file_saved = active_trainer.save_checkpoint(_tag_clean)
            _trainable_sd = active_trainer.model.trainable_state_dict()
            _full_sd = active_trainer.model.state_dict()
            _meta_dict = {
                "source": "in_memory_active_trainer",
                "epochs_completed": active_trainer.current_epoch,
                "global_step": active_trainer.global_step,
                "best_val_map50": active_trainer.best_map50,
                "frozen_backbone_dir": base_config.backbone.frozen_encoder_dir,
                "config": active_trainer.config.to_dict(),
            }
        else:
            _fallback_name = "best_map.pt" if _src_choice == "In-Memory (Current Trained Model)" else _src_choice
            _src_ckpt_path = _out_root / _fallback_name
            _loaded_state = load_checkpoint(_src_ckpt_path, map_location="cpu")
            _trainable_sd = _loaded_state.get("trainable_state_dict", {})
            _ckpt_file_saved = _out_root / f"{_tag_clean}.pt"
            if _src_ckpt_path != _ckpt_file_saved:
                torch.save(_loaded_state, _ckpt_file_saved)
            _full_sd = _trainable_sd
            _meta_dict = {
                "source": str(_src_ckpt_path),
                "epochs_completed": _loaded_state.get("epoch", 0),
                "global_step": _loaded_state.get("global_step", 0),
                "best_val_map50": _loaded_state.get("best_map50", 0.0),
                "frozen_backbone_dir": base_config.backbone.frozen_encoder_dir,
                "config": _loaded_state.get("config", base_config.to_dict()),
            }

        # Save standalone weights and manifest inside bundle folder
        _trainable_pt_path = _bundle_dir / "vitdet_sfp_head_weights.pt"
        _full_pt_path = _bundle_dir / "histo_jepa_vitdet_full.pt"
        _manifest_json_path = _bundle_dir / "manifest.json"

        torch.save(_trainable_sd, _trainable_pt_path)
        torch.save(_full_sd, _full_pt_path)
        with open(_manifest_json_path, "w", encoding="utf-8") as _mf:
            json.dump(_meta_dict, _mf, indent=2)

        _best_map_val = _meta_dict.get("best_val_map50", 0.0)
        _save_status_card = mo.md(
            f"""
            ✅ **Trained ViTDet Model Saved to Disk!**  
            - **Trainer Checkpoint (`.pt`):** `{_ckpt_file_saved}`  
            - **Standalone SFP + Head Weights:** `{_trainable_pt_path}`  
            - **Full Detector State Dict (Frozen ViT + SFP + Head):** `{_full_pt_path}`  
            - **Export Manifest:** `{_manifest_json_path}`  
            - **Provenance:** `{_meta_dict['source']}` (Epochs: **{_meta_dict['epochs_completed']}**, Steps: **{_meta_dict['global_step']}**, Best Val `mAP@50`: **`{_best_map_val:.4f}`**)
            """
        )

    _save_status_card
    return


@app.cell(hide_code=True)
def _(Path, active_trainer, base_config, json, mo, plt):
    hist_file = Path(base_config.output.dir) / "history.json"
    telemetry_view = mo.md(
        "*No Branch A training telemetry recorded yet (`outputs/phase2_branch_a/history.json`). Run training above to populate loss and validation curves.*"
    )

    # Trigger reactivity when active_trainer finishes
    _ = active_trainer

    if hist_file.exists():
        with open(hist_file, "r", encoding="utf-8") as hf:
            h_data = json.load(hf)

        steps = h_data.get("steps", [])
        tot_losses = h_data.get("total_losses", [])
        cls_losses = h_data.get("cls_losses", [])
        box_losses = h_data.get("box_losses", [])
        lrs = h_data.get("lrs", [])
        eval_epochs = h_data.get("eval_epochs", [])
        val_map50 = h_data.get("val_map50", [])
        val_map50_95 = h_data.get("val_map50_95", [])
        val_f1 = h_data.get("val_f1", [])
        val_prec = h_data.get("val_precision", [])
        val_rec = h_data.get("val_recall", [])

        if len(steps) > 0:
            fig_tel, axes_tel = plt.subplots(1, 4, figsize=(18, 4))

            # 1. Detection Losses
            axes_tel[0].plot(steps, tot_losses, label="Total Loss", color="#E63946", lw=1.8)
            axes_tel[0].plot(steps, cls_losses, label="Focal Cls", color="#457B9D", lw=1.2, alpha=0.85)
            axes_tel[0].plot(steps, box_losses, label="GIoU Box", color="#2A9D8F", lw=1.2, alpha=0.85)
            axes_tel[0].set_title("1. Detection Training Losses")
            axes_tel[0].set_xlabel("Optimizer Step")
            axes_tel[0].set_ylabel("Loss")
            axes_tel[0].legend(fontsize=8)
            axes_tel[0].grid(True, alpha=0.3)

            # 2. Learning Rate Schedule
            axes_tel[1].plot(steps, lrs, color="#1D3557", lw=1.8)
            axes_tel[1].set_title("2. Learning Rate Schedule")
            axes_tel[1].set_xlabel("Optimizer Step")
            axes_tel[1].set_ylabel("LR")
            axes_tel[1].grid(True, alpha=0.3)

            # 3. Validation mAP Progression
            if len(eval_epochs) > 0:
                axes_tel[2].plot(eval_epochs, val_map50, marker="o", label="Val mAP@50", color="#2A9D8F", lw=2.0)
                axes_tel[2].plot(eval_epochs, val_map50_95, marker="s", label="Val mAP@50:95", color="#E76F51", lw=1.8)
                axes_tel[2].set_ylim(0.0, 1.05)
                axes_tel[2].legend(fontsize=8)
            axes_tel[2].set_title("3. Validation mAP Progression (`val` split)")
            axes_tel[2].set_xlabel("Epoch")
            axes_tel[2].set_ylabel("mAP")
            axes_tel[2].grid(True, alpha=0.3)

            # 4. Validation F1, Precision & Recall Progression
            if len(eval_epochs) > 0:
                axes_tel[3].plot(eval_epochs, val_f1, marker="o", label="Val F1", color="#9B5DE5", lw=2.0)
                axes_tel[3].plot(eval_epochs, val_prec, ls="--", label="Precision", color="#00BBF9", lw=1.5)
                axes_tel[3].plot(eval_epochs, val_rec, ls=":", label="Recall", color="#F15BB5", lw=1.5)
                axes_tel[3].set_ylim(0.0, 1.05)
                axes_tel[3].legend(fontsize=8)
            axes_tel[3].set_title("4. Validation F1 / Precision / Recall (`val` split)")
            axes_tel[3].set_xlabel("Epoch")
            axes_tel[3].set_ylabel("Score")
            axes_tel[3].grid(True, alpha=0.3)

            plt.tight_layout()
            telemetry_view = mo.vstack(
                [
                    mo.md("### 📈 Training & Validation Telemetry Curves"),
                    mo.as_html(fig_tel),
                ]
            )
            plt.close(fig_tel)

    telemetry_view
    return


@app.cell(hide_code=True)
def _(Path, active_trainer, base_config, mo):
    _ = active_trainer
    out_dir_eval = Path(base_config.output.dir)
    eval_ckpt_options = ["In-Memory / Untrained Baseline"]
    if out_dir_eval.exists():
        found_ckpts = sorted([p.name for p in out_dir_eval.glob("*.pt")])
        for pref in ("best_map.pt", "latest.pt"):
            if pref in found_ckpts:
                eval_ckpt_options.insert(0, pref)
        for eval_fn in found_ckpts:
            if eval_fn not in eval_ckpt_options:
                eval_ckpt_options.append(eval_fn)

    eval_ckpt_ui = mo.ui.dropdown(
        options=eval_ckpt_options,
        value=eval_ckpt_options[0],
        label="**Checkpoint to Evaluate:**",
    )
    eval_split_ui = mo.ui.dropdown(
        options=["val", "test", "both (val + test)"],
        value="val",
        label="**Dataset Split:**",
    )
    eval_score_ui = mo.ui.slider(
        start=0.05,
        stop=0.90,
        step=0.05,
        value=base_config.evaluation.score_threshold,
        label="**Score Threshold:**",
    )
    eval_nms_ui = mo.ui.slider(
        start=0.10,
        stop=0.90,
        step=0.05,
        value=base_config.evaluation.nms_iou_threshold,
        label="**NMS IoU Threshold:**",
    )
    run_eval_btn = mo.ui.run_button(
        label="📊 Compute Quantitative Metrics (mAP & F1)",
    )

    mo.md(
        f"""
        ---
        ### 🏆 Quantitative Evaluation on Validation (`val`) & Held-Out Test (`test`) Splits
        Select a saved checkpoint (`best_map.pt` or `latest.pt`) and evaluate across **`val`** (653 images), **`test`** (654 images), or **both splits side-by-side**:

        {mo.hstack([eval_ckpt_ui, eval_split_ui, eval_score_ui, eval_nms_ui, run_eval_btn], justify="start")}
        """
    )
    return (
        eval_ckpt_ui,
        eval_nms_ui,
        eval_score_ui,
        eval_split_ui,
        run_eval_btn,
    )


@app.cell(hide_code=True)
def _(
    Path,
    ViTDetTrainer,
    active_trainer,
    base_config,
    device_ui,
    eval_ckpt_ui,
    eval_nms_ui,
    eval_score_ui,
    eval_split_ui,
    frozen_encoder,
    mo,
    plt,
    run_eval_btn,
):
    eval_output_view = mo.md(
        "*Click **'📊 Compute Quantitative Metrics (mAP & F1)'** above to evaluate the detector on the `val` and/or `test` splits.*"
    )

    if run_eval_btn.value:
        if active_trainer is not None and eval_ckpt_ui.value == "In-Memory / Untrained Baseline":
            eval_trainer = active_trainer
        else:
            eval_trainer = ViTDetTrainer(
                config=base_config,
                device=device_ui.value,
                context_encoder=frozen_encoder.to(device_ui.value),
            )
            if eval_ckpt_ui.value != "In-Memory / Untrained Baseline":
                ckpt_file = Path(base_config.output.dir) / eval_ckpt_ui.value
                if ckpt_file.exists():
                    eval_trainer.load_checkpoint(ckpt_file)

        splits_to_run = (
            ["val", "test"]
            if eval_split_ui.value == "both (val + test)"
            else [eval_split_ui.value]
        )

        reports = {}
        for sp in splits_to_run:
            reports[sp] = eval_trainer.evaluate_split(
                split=sp,
                score_threshold=eval_score_ui.value,
                nms_iou_threshold=eval_nms_ui.value,
            )

        table_rows_md = ""
        for sp, rep in reports.items():
            table_rows_md += (
                f"| **`{sp}`** | **{rep.map50:.4f}** | **{rep.map75:.4f}** | **{rep.map50_95:.4f}** | "
                f"**{rep.precision:.4f}** | **{rep.recall:.4f}** | **{rep.f1:.4f}** | "
                f"`{rep.best_score_threshold:.2f}` | {rep.total_gt:,} | {rep.true_positives:,} | "
                f"{rep.false_positives:,} | {rep.false_negatives:,} |\n"
            )

        fig_pr, ax_pr = plt.subplots(figsize=(6.5, 4.2))
        palette = {"val": "#2A9D8F", "test": "#E63946"}
        for sp, rep in reports.items():
            if len(rep.pr_curve_recall) > 0:
                ax_pr.plot(
                    rep.pr_curve_recall,
                    rep.pr_curve_precision,
                    label=f"{sp.upper()} (mAP@50 = {rep.map50:.3f}, F1 = {rep.f1:.3f})",
                    color=palette.get(sp, "#1D3557"),
                    lw=2.2,
                )
        ax_pr.set_title("Precision-Recall Curve at IoU = 0.50")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0.0, 1.02)
        ax_pr.set_ylim(0.0, 1.05)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(loc="lower left", fontsize=9)
        plt.tight_layout()
        pr_html = mo.as_html(fig_pr)
        plt.close(fig_pr)

        eval_output_view = mo.vstack(
            [
                mo.md(
                    f"""
                    #### 📋 Quantitative Evaluation Summary (Checkpoint: `{eval_ckpt_ui.value}`)
                    | Split | `mAP@50` | `mAP@75` | `mAP@50:95` | Precision | Recall | `F1-Score` | Best Score Thresh | Total GT | TP | FP | FN |
                    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
                    {table_rows_md}
                    """
                ),
                pr_html,
            ]
        )

    eval_output_view
    return


@app.cell(hide_code=True)
def _(Path, base_config, mo):
    vis_split_ui = mo.ui.dropdown(
        options=["val", "test"],
        value="val",
        label="**Inspect Split:**",
    )

    val_imgs = sorted([p.name for p in (Path(base_config.data.root) / "val" / "processed").glob("*.png")])[:50]
    test_imgs = sorted([p.name for p in (Path(base_config.data.root) / "test" / "processed").glob("*.png")])[:50]

    vis_index_ui = mo.ui.slider(
        start=0,
        stop=max(0, min(len(val_imgs), len(test_imgs)) - 1),
        step=1,
        value=0,
        label="**Tile Index:**",
    )
    vis_conf_ui = mo.ui.slider(
        start=0.05,
        stop=0.90,
        step=0.05,
        value=0.25,
        label="**Confidence Threshold:**",
    )

    mo.md(
        f"""
        ### 🖼️ Qualitative Visual Comparison: Ground Truth vs. ViTDet Predictions
        Inspect individual histopathology tiles from `val` or `test` with Ground-Truth Tumor Buds (solid green) vs. ViTDet Predicted Bounding Boxes (dashed red with confidence scores):

        {mo.hstack([vis_split_ui, vis_index_ui, vis_conf_ui], justify="start")}
        """
    )
    return test_imgs, val_imgs, vis_conf_ui, vis_index_ui, vis_split_ui


@app.cell(hide_code=True)
def _(
    HistoJEPA_ViTDet,
    Path,
    active_trainer,
    base_config,
    compute_box_iou,
    cv2,
    device_ui,
    draw_detection_comparison,
    eval_ckpt_ui,
    eval_nms_ui,
    frozen_encoder,
    load_checkpoint,
    load_yolo_boxes_and_labels,
    mo,
    np,
    plt,
    test_imgs,
    torch,
    val_imgs,
    vis_conf_ui,
    vis_index_ui,
    vis_split_ui,
):
    chosen_split = vis_split_ui.value
    img_list = val_imgs if chosen_split == "val" else test_imgs
    tile_idx = min(vis_index_ui.value, max(0, len(img_list) - 1))
    tile_name = img_list[tile_idx] if img_list else None

    qual_view = mo.md("*No tiles available in selected split.*")

    if tile_name is not None:
        img_file = Path(base_config.data.root) / chosen_split / "processed" / tile_name
        lbl_file = Path(base_config.data.root) / chosen_split / "labels" / f"{img_file.stem}.txt"

        bgr_img = cv2.imread(str(img_file))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        gt_boxes_tile, _ = load_yolo_boxes_and_labels(
            lbl_file,
            img_width=base_config.data.image_size,
            img_height=base_config.data.image_size,
        )

        # Resolve detector instance
        dev = device_ui.value
        if active_trainer is not None and eval_ckpt_ui.value == "In-Memory / Untrained Baseline":
            vis_detector = active_trainer.model
        else:
            vis_detector = HistoJEPA_ViTDet.from_config(
                base_config,
                context_encoder=frozen_encoder.to(dev),
                device=dev,
            )
            if eval_ckpt_ui.value != "In-Memory / Untrained Baseline":
                ckpt_p = Path(base_config.output.dir) / eval_ckpt_ui.value
                if ckpt_p.exists():
                    st = load_checkpoint(ckpt_p, map_location=dev)
                    if "trainable_state_dict" in st:
                        vis_detector.load_trainable_state_dict(st["trainable_state_dict"])

        vis_detector.eval()
        mean_v = np.array(base_config.data.mean, dtype=np.float32).reshape(1, 1, 3)
        std_v = np.array(base_config.data.std, dtype=np.float32).reshape(1, 1, 3)
        norm_tile = (rgb_img.astype(np.float32) / 255.0 - mean_v) / std_v
        inp_tensor = torch.from_numpy(norm_tile).permute(2, 0, 1).unsqueeze(0).contiguous().to(dev)

        with torch.no_grad():
            det_out = vis_detector(
                inp_tensor,
                targets=None,
                score_threshold=vis_conf_ui.value,
                nms_iou_threshold=eval_nms_ui.value,
            )[0]

        pred_boxes_tile = det_out["boxes"].detach().cpu().numpy()
        pred_scores_tile = det_out["scores"].detach().cpu().numpy()

        # Per-tile IoU matching at 0.50
        iou_mat = compute_box_iou(pred_boxes_tile, gt_boxes_tile).numpy()
        matched_tp = 0
        if iou_mat.size > 0:
            used_gt = set()
            for p_i in range(iou_mat.shape[0]):
                best_g = int(np.argmax(iou_mat[p_i]))
                if iou_mat[p_i, best_g] >= 0.50 and best_g not in used_gt:
                    matched_tp += 1
                    used_gt.add(best_g)

        fig_vis, axes_vis = plt.subplots(1, 2, figsize=(13, 6.2))
        draw_detection_comparison(
            axes_vis[0],
            rgb_img,
            gt_boxes=gt_boxes_tile,
            pred_boxes=np.zeros((0, 4), dtype=np.float32),
        )
        axes_vis[0].set_title(
            f"Ground Truth ({chosen_split}/{tile_name}): {len(gt_boxes_tile)} Tumor Buds",
            fontsize=10.5,
        )

        draw_detection_comparison(
            axes_vis[1],
            rgb_img,
            gt_boxes=gt_boxes_tile,
            pred_boxes=pred_boxes_tile,
            pred_scores=pred_scores_tile,
        )
        axes_vis[1].set_title(
            f"Overlay: {len(pred_boxes_tile)} Predictions (TP@50: {matched_tp}/{len(gt_boxes_tile)})",
            fontsize=10.5,
        )
        axes_vis[1].legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        qual_view = mo.as_html(fig_vis)
        plt.close(fig_vis)

    qual_view
    return


if __name__ == "__main__":
    app.run()
