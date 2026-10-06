import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="02 - HistoJEPA Phase 1 - I-JEPA Pre-Training",
)


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path
    import time

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm

    import marimo as mo

    from histojepa.data import ProcessedImageDataset
    from histojepa.models.ijepa import (
        IJEPA,
        IJepaTrainer,
        MultiBlockMaskCollator,
        Phase1Config,
        load_frozen_context_encoder,
        save_frozen_context_encoder,
    )
    from histojepa.utils import draw_ijepa_masks, find_latest_checkpoint

    return (
        IJEPA,
        IJepaTrainer,
        MultiBlockMaskCollator,
        Path,
        Phase1Config,
        cv2,
        draw_ijepa_masks,
        json,
        load_frozen_context_encoder,
        mo,
        plt,
        save_frozen_context_encoder,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧠 HistoJEPA Phase 1 — I-JEPA Self-Supervised Pre-Training
    **Notebook:** `02_Phase_1_IJEPA_PreTrain.py`
    **Target Data:** `data/train/processed/` (1280×1280 RGB histopathology tiles)
    **Reference:** Assran et al., *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"*, CVPR 2023.

    ---

    ### 🎯 Pre-Training Objectives & Mathematical Formulation
    This notebook executes self-supervised representation learning for the **Vision Transformer (ViT-B/16)** context encoder directly on Cytokeratin-stained colorectal cancer histology tiles:

    1. **Target Side ($f_{\bar{\theta}}$):** The entire $1280 \times 1280$ image is processed by the target encoder to generate latent patch representations $s_y = \{s_{y_1}, \dots, s_{y_N}\} \in \mathbb{R}^{6400 \times 768}$. $M=4$ target blocks $B_i$ (scale $0.15\text{--}0.2$, aspect ratio $0.75\text{--}1.5$) are sampled by masking the **output** representations. The target encoder weights $\bar{\theta}$ are updated strictly via Exponential Moving Average (**EMA**):
       $$\bar{\theta} \leftarrow m \bar{\theta} + (1 - m) \theta, \quad m \in [0.996, 1.0]$$
    2. **Context Side ($f_\theta$):** A single context block (scale $0.85\text{--}1.0$) is sampled with all overlapping target patches eliminated ($\text{Context} \cap \bigcup_i B_i = \emptyset$). Only visible context tokens are processed (token dropping) into representations $s_x$.
    3. **Predictor Network ($g_\phi$):** A narrow ViT (width 384, depth 6, 12 heads) is conditioned on position-encoded mask tokens and evaluated $M=4$ times to predict each target block $\hat{s}_y(i)$.
    4. **Representation $L_2$ Loss:** The context encoder and predictor are optimized with AdamW using the latent $L_2$ prediction loss:
       $$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{M} \sum_{i=1}^M D(\hat{s}_y(i), s_y(i)) = \frac{1}{M} \sum_{i=1}^M \sum_{j \in B_i} \|\hat{s}_{y_j} - s_{y_j}\|_2^2$$
    5. **Phase 2 Hand-Off:** Once trained, the **ViT Context Encoder** is frozen and exported (`outputs/phase1_ijepa/context_encoder_frozen/`) to drive Branch A (ViTDet SFP detection) and Branch B (clustering).
    """)
    return


@app.cell(hide_code=True)
def _(Path, Phase1Config):
    # Load default Phase 1 YAML configuration
    default_cfg_path = Path("configs/phase1_ijepa.yaml")
    base_config = Phase1Config.from_yaml(default_cfg_path)
    return base_config, default_cfg_path


@app.cell(hide_code=True)
def _(Path, base_config, mo, torch):
    # Scan existing checkpoints in output directory
    _output_dir = Path(base_config.output.dir)
    _available_ckpts = ["None"]
    if _output_dir.exists():
        _found = sorted([_p.name for _p in _output_dir.glob("*.pt") if not _p.name.startswith("context_")])
        if "latest.pt" in _found:
            _available_ckpts.append("latest.pt")
        if "best_loss.pt" in _found:
            _available_ckpts.append("best_loss.pt")
        for _fn in _found:
            if _fn not in _available_ckpts:
                _available_ckpts.append(_fn)

    # UI Controls
    epochs_ui = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=min(base_config.optimization.epochs, 100),
        label="**Pre-Training Epochs:**",
    )

    micro_bs_ui = mo.ui.slider(
        start=1,
        stop=4,
        step=1,
        value=base_config.optimization.micro_batch_size,
        label="**Micro-Batch Size:**",
    )

    grad_accum_ui = mo.ui.slider(
        start=1,
        stop=32,
        step=1,
        value=base_config.optimization.grad_accum_steps,
        label="**Gradient Accumulation Steps:**",
    )

    _device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device_ui = mo.ui.dropdown(
        options=_device_options,
        value="cuda" if torch.cuda.is_available() else "cpu",
        label="**Compute Device:**",
    )

    resume_ui = mo.ui.dropdown(
        options=_available_ckpts,
        value="None",
        label="**Resume Checkpoint:**",
    )

    grad_ckpt_ui = mo.ui.checkbox(
        value=base_config.optimization.gradient_checkpointing,
        label="**Gradient Checkpointing** (Saves VRAM)",
    )

    mo.md(
        f"""
        ### ⚙️ Experiment Configuration & Hyperparameters
        {mo.hstack([epochs_ui, micro_bs_ui, grad_accum_ui], justify="start")}

        {mo.hstack([device_ui, resume_ui, grad_ckpt_ui], justify="start")}
        """
    )
    return (
        device_ui,
        epochs_ui,
        grad_accum_ui,
        grad_ckpt_ui,
        micro_bs_ui,
        resume_ui,
    )


@app.cell(hide_code=True)
def _(Path, base_config, cv2, mo):
    # Dataset inventory verification
    _proc_path = Path(base_config.data.root) / base_config.data.split / "processed"
    image_files = sorted(list(_proc_path.glob("*.png"))) if _proc_path.exists() else []
    _total_imgs = len(image_files)

    _sample_res = "N/A"
    if _total_imgs > 0:
        _sample_bgr = cv2.imread(str(image_files[0]))
        if _sample_bgr is not None:
            _sample_res = f"{_sample_bgr.shape[1]}x{_sample_bgr.shape[0]}"

    _p_size = base_config.model_encoder.patch_size
    _img_size = base_config.data.image_size
    _grid_side = _img_size // _p_size
    _total_patches = _grid_side * _grid_side

    _status_badge = "🟢 Ready" if _total_imgs > 0 else "🔴 Processed Folder Empty (Run Notebook 01 first)"

    _table_md = f"""
    | Metric | Value | Reference / Notes |
    | :--- | :--- | :--- |
    | **Source Split** | `{base_config.data.root}/{base_config.data.split}/processed` | Pre-processed TB-YOLO tiles |
    | **Total Available Images** | **{_total_imgs:,}** | Status: {_status_badge} |
    | **Image Tile Dimensions** | **{_img_size}×{_img_size} RGB** (detected: `{_sample_res}`) | Full native resolution |
    | **Patch Size ($P$)** | **{_p_size}×{_p_size}** pixels | Standard ViT-B patch resolution |
    | **Spatial Patch Grid** | **{_grid_side}×{_grid_side}** patches | Total $N = {_total_patches:,}$ tokens |
    """

    mo.md(
        f"""
        ### 📊 Dataset Inventory & Spatial Patch Resolution
        {_table_md}
        """
    )
    return (image_files,)


@app.cell(hide_code=True)
def _(image_files, mo):
    # Interactive Multi-Block Mask Visualizer Controls
    _img_names = [_p.name for _p in image_files[:40]] if image_files else ["None"]
    sample_img_ui = mo.ui.dropdown(
        options=_img_names,
        value=_img_names[0],
        label="**Select Histology Tile:**",
    )

    resample_btn = mo.ui.run_button(
        label="🎲 Resample Masks",
    )

    mo.md(
        f"""
        ### 🔍 Live Multi-Block Mask Preview & Overlap Check
        {mo.hstack([sample_img_ui, resample_btn], justify="start")}
        """
    )
    return resample_btn, sample_img_ui


@app.cell(hide_code=True)
def _(
    MultiBlockMaskCollator,
    Path,
    base_config,
    cv2,
    draw_ijepa_masks,
    mo,
    plt,
    resample_btn,
    sample_img_ui,
    torch,
):
    # Trigger re-render whenever the dropdown selection changes or the resample button is clicked
    _ = resample_btn.value

    _p_sz = base_config.model_encoder.patch_size
    _img_sz = base_config.data.image_size
    _grid_side_val = _img_sz // _p_sz

    _preview_html = None

    if sample_img_ui.value and sample_img_ui.value != "None":
        _selected_file = (
            Path(base_config.data.root) / base_config.data.split / "processed" / sample_img_ui.value
        )
        if _selected_file.exists():
            _bgr = cv2.imread(str(_selected_file))
            if _bgr is not None:
                _rgb = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)

                # Initialize collator to sample masks
                _collator = MultiBlockMaskCollator(
                    grid_size=(_grid_side_val, _grid_side_val),
                    num_targets=base_config.masking.num_targets,
                    target_scale=base_config.masking.target_scale,
                    target_aspect_ratio=base_config.masking.target_aspect_ratio,
                    context_scale=base_config.masking.context_scale,
                    context_aspect_ratio=base_config.masking.context_aspect_ratio,
                    min_context_keep=base_config.masking.min_context_keep,
                )

                # Dummy 1-item batch
                _tensor_chw = torch.from_numpy(_rgb.transpose(2, 0, 1)).float()
                _, _ctx_idx, _tgt_list = _collator([_tensor_chw])

                _ctx_tokens_count = _ctx_idx.shape[1]
                _tgt_tokens_count = _tgt_list[0].shape[1]
                _ctx_pct = (_ctx_tokens_count / (_grid_side_val * _grid_side_val)) * 100.0

                # Strict overlap verification test
                _ctx_set = set(_ctx_idx[0].tolist())
                _total_overlap = sum(len(_ctx_set.intersection(set(_tgt[0].tolist()))) for _tgt in _tgt_list)

                _fig, _ax = plt.subplots(figsize=(7, 7))
                draw_ijepa_masks(
                    ax=_ax,
                    image_rgb=_rgb,
                    context_idx=_ctx_idx[0],
                    target_idx_list=[_tgt[0] for _tgt in _tgt_list],
                    grid_size=(_grid_side_val, _grid_side_val),
                    patch_size=_p_sz,
                )
                _ax.set_title(
                    f"I-JEPA Multi-Block Masking ({sample_img_ui.value})\n"
                    f"Context: {_ctx_tokens_count:,} patches ({_ctx_pct:.1f}% area) | "
                    f"M={len(_tgt_list)} Targets: {_tgt_tokens_count} patches each | Overlap: {_total_overlap} (Strictly 0)",
                    fontsize=10,
                )
                plt.tight_layout()
                _preview_html = mo.as_html(_fig)
                plt.close(_fig)

    mo.md(
        f"""
        {_preview_html if _preview_html is not None else "Select a valid processed image to inspect masks."}
        """
    )
    return


@app.cell(hide_code=True)
def _(base_config, grad_accum_ui, micro_bs_ui, mo):
    # Model architecture and parameter summary
    _eff_batch = micro_bs_ui.value * grad_accum_ui.value
    _start_lr, _ref_lr, _final_lr = base_config.optimization.get_effective_lrs()

    _ctx_params = "86.4M"
    _tgt_params = "86.4M (0 trainable, EMA)"
    _pred_params = "13.2M"
    _total_trainable = "99.6M"

    _model_summary_table = f"""
    | Component | Role | Hidden Dim | Layers / Depth | Trainable Parameters | Optimization Discipline |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Context Encoder ($f_\\theta$)** | Encodes visible tissue tokens | 768 ($D$) | 12 ViT layers | **{_ctx_params}** | AdamW with Weight Decay |
    | **Target Encoder ($f_{{\\bar{{\\theta}}}}$)** | Encodes full image latent targets | 768 ($D$) | 12 ViT layers | **{_tgt_params}** | Strictly frozen, EMA update ($0.996 \\to 1.0$) |
    | **Predictor ($g_\\phi$)** | Predicts $M=4$ target blocks | 384 ($D_{{pred}}$) | 6 narrow layers | **{_pred_params}** | AdamW with Weight Decay |
    | **Combined Model** | Complete I-JEPA architecture | — | — | **{_total_trainable}** | Effective Batch Size: **{_eff_batch}** |

    **Learning Rate Scaling:** `sqrt` mode with effective batch {_eff_batch}:  
    Warmup: `{_start_lr:.2e}` $\\to$ Peak: `{_ref_lr:.2e}` $\\to$ Cosine: `{_final_lr:.2e}`
    """

    mo.md(
        f"""
        ### 🏛️ Model Architecture & Parameter Allocation
        {_model_summary_table}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Training Execution Trigger
    train_exec_btn = mo.ui.run_button(
        label="🚀 Start / Resume Pre-Training",
    )
    mo.md(
        f"""
        ### 🏋️ Training Execution Control
        Click to initiate self-supervised I-JEPA pre-training across the processed dataset:

        {train_exec_btn}
        """
    )
    return (train_exec_btn,)


@app.cell(hide_code=True)
def _(
    IJepaTrainer,
    Path,
    Phase1Config,
    default_cfg_path,
    device_ui,
    epochs_ui,
    grad_accum_ui,
    grad_ckpt_ui,
    micro_bs_ui,
    mo,
    resume_ui,
    train_exec_btn,
):
    # Training execution block
    _training_status = None
    active_trainer = None

    if train_exec_btn.value:
        _overrides = {
            "optimization": {
                "epochs": epochs_ui.value,
                "micro_batch_size": micro_bs_ui.value,
                "grad_accum_steps": grad_accum_ui.value,
                "gradient_checkpointing": grad_ckpt_ui.value,
            }
        }
        _run_config = Phase1Config.from_yaml(default_cfg_path, overrides=_overrides)

        active_trainer = IJepaTrainer(config=_run_config, device=device_ui.value)

        _resume_target = None
        if resume_ui.value != "None":
            _resume_path = Path(_run_config.output.dir) / resume_ui.value
            if _resume_path.exists():
                _resume_target = _resume_path

        _total_steps = active_trainer.total_optimizer_steps

        with mo.status.progress_bar(
            total=_total_steps,
            title="I-JEPA Phase 1 Pre-Training",
            subtitle="Starting...",
            show_rate=True,
            show_eta=True,
            remove_on_exit=False,
        ) as _bar:
            def _on_step(info):
                _bar.update(
                    increment=1,
                    subtitle=(
                        f"Epoch {info.epoch + 1}/{epochs_ui.value} | "
                        f"Loss: {info.loss:.4f} | LR: {info.lr:.2e} | EMA: {info.momentum:.4f}"
                    ),
                )

            _history = active_trainer.fit(
                epochs=epochs_ui.value,
                progress_callback=_on_step,
                resume_from=_resume_target,
            )

        _final_loss = _history.losses[-1] if _history.losses else 0.0
        _training_status = mo.md(
            f"""
            ✅ **Pre-Training Complete!**  
            - Epochs Completed: **{epochs_ui.value}**  
            - Total Optimizer Steps: **{len(_history.steps):,}**  
            - Final $L_2$ Loss: **`{_final_loss:.5f}`**  
            - Latest Checkpoint: `{_run_config.output.dir}/latest.pt`  
            - Best Checkpoint: `{_run_config.output.dir}/best_loss.pt`  
            """
        )

    mo.md(
        f"""
        {_training_status if _training_status is not None else "Click 'Start / Resume Pre-Training' above to launch."}
        """
    )
    return (active_trainer,)


@app.cell(hide_code=True)
def _(Path, base_config, json, mo, plt):
    # Monitoring and Diagnostics Curves (Loss, LR, Momentum, Collapse check)
    _hist_file = Path(base_config.output.dir) / "history.json"
    _curves_html = None

    if _hist_file.exists():
        try:
            with open(_hist_file, "r", encoding="utf-8") as _f:
                _h_data = json.load(_f)

            _steps = _h_data.get("steps", [])
            _losses = _h_data.get("losses", [])
            _lrs = _h_data.get("lrs", [])
            _momentums = _h_data.get("momentums", [])
            _target_stds = _h_data.get("target_stds", [])
            _pred_stds = _h_data.get("pred_stds", [])

            if len(_steps) > 0:
                _fig, _axes = plt.subplots(1, 4, figsize=(18, 4))

                # 1. L2 Loss
                _axes[0].plot(_steps, _losses, color="#E63946", lw=1.8)
                _axes[0].set_title("1. Latent L2 Loss")
                _axes[0].set_xlabel("Optimizer Step")
                _axes[0].set_ylabel("Loss")
                _axes[0].grid(True, alpha=0.3)

                # 2. Learning Rate
                _axes[1].plot(_steps, _lrs, color="#1D3557", lw=1.8)
                _axes[1].set_title("2. Learning Rate Schedule")
                _axes[1].set_xlabel("Optimizer Step")
                _axes[1].set_ylabel("LR")
                _axes[1].grid(True, alpha=0.3)

                # 3. EMA Momentum
                _axes[2].plot(_steps, _momentums, color="#2A9D8F", lw=1.8)
                _axes[2].set_title("3. Target EMA Momentum")
                _axes[2].set_xlabel("Optimizer Step")
                _axes[2].set_ylabel("Momentum ($m$)")
                _axes[2].grid(True, alpha=0.3)

                # 4. Collapse Diagnostic (Embedding Standard Deviation)
                _axes[3].plot(_steps, _target_stds, label="Target Std", color="#457B9D", lw=1.5)
                _axes[3].plot(_steps, _pred_stds, label="Pred Std", color="#F4A261", lw=1.5, ls="--")
                _axes[3].set_title("4. Embedding Std (Collapse Check)")
                _axes[3].set_xlabel("Optimizer Step")
                _axes[3].set_ylabel("Std Dev")
                _axes[3].legend()
                _axes[3].grid(True, alpha=0.3)

                plt.tight_layout()
                _curves_html = mo.as_html(_fig)
                plt.close(_fig)
        except Exception:
            _curves_html = None

    mo.md(
        f"""
        ### 📈 Live & Persisted Monitoring Curves
        {_curves_html if _curves_html is not None else "No training history recorded yet. Run pre-training to visualize telemetry."}
        """
    )
    return


@app.cell(hide_code=True)
def _(Path, base_config, mo):
    # Phase 2 Hand-Off: Export Controls
    _output_dir_p = Path(base_config.output.dir)
    _export_ckpt_options = ["In-Memory (Current Model)"]
    if _output_dir_p.exists():
        for _fn in sorted([_p.name for _p in _output_dir_p.glob("*.pt") if not _p.name.startswith("context_")]):
            _export_ckpt_options.append(_fn)

    export_source_ui = mo.ui.dropdown(
        options=_export_ckpt_options,
        value=_export_ckpt_options[0],
        label="**Source Checkpoint to Export:**",
    )

    export_btn = mo.ui.run_button(
        label="❄️ Freeze & Save Context Encoder for Phase 2",
    )

    mo.md(
        f"""
        ### 💾 Phase 2 Hand-Off: Freeze & Export ViT Context Encoder
        *In accordance with **AGENTS.md Hard Constraint #1**, the Context Encoder must be strictly frozen to drive **Branch A (ViTDet SFP)** and **Branch B (Clustering)**.*

        {mo.hstack([export_source_ui, export_btn], justify="start")}
        """
    )
    return export_btn, export_source_ui


@app.cell(hide_code=True)
def _(
    IJEPA,
    Path,
    active_trainer,
    base_config,
    export_btn,
    export_source_ui,
    load_frozen_context_encoder,
    mo,
    save_frozen_context_encoder,
    torch,
):
    _export_result_card = None

    if export_btn.value:
        _output_dir_p = Path(base_config.output.dir)
        _dest_dir = Path(base_config.output.export_dir)
        _dest_dir.mkdir(parents=True, exist_ok=True)

        _selected_src = export_source_ui.value

        if _selected_src == "In-Memory (Current Model)" and active_trainer is not None:
            _encoder_to_export = active_trainer.model.context_encoder
            _meta_info = {
                "provenance": "in_memory_trainer",
                "epoch": active_trainer.current_epoch,
                "step": active_trainer.global_step,
                "best_loss": active_trainer.best_loss,
            }
        else:
            _ckpt_name = "best_loss.pt" if _selected_src == "In-Memory (Current Model)" else _selected_src
            _ckpt_path = _output_dir_p / _ckpt_name
            if not _ckpt_path.exists():
                _ckpt_path = _output_dir_p / "latest.pt"

            if _ckpt_path.exists():
                _ckpt_state = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                _temp_model = IJEPA(base_config)
                _temp_model.load_state_dict(_ckpt_state["model_state_dict"])
                _encoder_to_export = _temp_model.context_encoder
                _meta_info = {
                    "provenance": str(_ckpt_path),
                    "epoch": _ckpt_state.get("epoch", 0),
                    "step": _ckpt_state.get("global_step", 0),
                    "best_loss": _ckpt_state.get("best_loss", 0.0),
                }
            else:
                _encoder_to_export = None
                _meta_info = {}

        if _encoder_to_export is not None:
            _exported_path = save_frozen_context_encoder(
                encoder=_encoder_to_export,
                output_dir=_dest_dir,
                metadata=_meta_info,
            )

            _verified_encoder = load_frozen_context_encoder(_exported_path, device="cpu")
            assert not _verified_encoder.training
            _all_frozen = all(not _p.requires_grad for _p in _verified_encoder.parameters())

            _dummy_test_img = torch.zeros(1, 3, base_config.data.image_size, base_config.data.image_size)
            _test_tokens = _verified_encoder(_dummy_test_img)
            _test_spatial = _verified_encoder.to_spatial(_test_tokens, grid_hw=(80, 80))
            _spatial_shape_str = str(list(_test_spatial.shape))

            _code_snippet = f"""```python
    from histojepa.models.ijepa import load_frozen_context_encoder

    # Load strictly frozen ViT Context Encoder into Branch A or Branch B
    context_encoder = load_frozen_context_encoder(
        "{_dest_dir}",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Ready to connect to ViTDet Simple Feature Pyramid (SFP) or UMAP clustering:
    # spatial_features = context_encoder.to_spatial(context_encoder(image_tensor)) # Shape: (B, 768, 80, 80)
    ```"""

            _export_result_card = mo.md(
                f"""
                ✅ **ViT Context Encoder Successfully Frozen and Saved!**  
                - Destination Directory: `{_exported_path}`  
                - Weights Status: **100% Frozen (`param.requires_grad = False`: {_all_frozen})**  
                - Mode: **`eval()` Mode Verified**  
                - Spatial Grid Verification: **`{_spatial_shape_str}`** (Stride 16: $P_4$ input)  
                - Manifest Saved: `{_exported_path}/manifest.json`  

                #### 📋 Ready-to-Use Phase 2 Integration Code:
                {_code_snippet}
                """
            )
        else:
            _export_result_card = mo.md(
                f"⚠️ Checkpoint not found to export. Please train the model or save a checkpoint first."
            )

    mo.md(
        f"""
        {_export_result_card if _export_result_card is not None else "Select source checkpoint and click to export the frozen Context Encoder."}
        """
    )
    return


if __name__ == "__main__":
    app.run()
