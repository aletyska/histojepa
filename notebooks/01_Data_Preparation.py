import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="01 - HistoJEPA Data Preparation")


@app.cell(hide_code=True)
def _():
    import io
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import marimo as mo

    from histojepa.data import (
        DataPrepPipeline,
        OtsuSegmenter,
        ReinhardNormalizer,
        SUPPORTED_SPLITS,
    )
    from histojepa.utils import draw_yolo_bboxes, load_yolo_annotations

    from tqdm import tqdm

    return (
        DataPrepPipeline,
        OtsuSegmenter,
        Path,
        ReinhardNormalizer,
        SUPPORTED_SPLITS,
        cv2,
        draw_yolo_bboxes,
        load_yolo_annotations,
        mo,
        plt,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔬 HistoJEPA — Data Preparation Pipeline
    **Notebook:** `01_Data_Preparation.py`
    **Target:** TB-YOLO Dataset (`data/train`, `data/val`, `data/test`)

    ---

    ### 📌 Overview & Methodology
    This interactive notebook prepares raw Whole Slide Image (WSI) tiles for the **HistoJEPA** self-supervised pre-training and downstream detection/clustering branches:

    1. **Otsu Automated Thresholding (`cv2`):**
       Calculates an adaptive bimodal intensity cutoff on smoothed tissue luminance to isolate foreground histology structures. Slide background glass, dust, and scanner sensor noise are eliminated by mapping non-tissue regions to a uniform neutral background (`#FFFFFF`).
    2. **Reinhard Color Normalization (`cv2`):**
       Normalizes color distributions in OpenCV CIELAB ($Lab$) space against a reference slide profile. Stain statistics ($\mu_{src}, \sigma_{src}$) are calculated **strictly over tissue pixels** to prevent glass background skew, standardizing Cytokeratin (CK) brown DAB and hematoxylin counterstain consistency across all slides.
    3. **Idempotent Batch Output:**
       Results are persisted to `data/{split}/processed/{filename}.png`. Re-running the pipeline cleanly replaces existing outputs with zero corruption.
    """)
    return


@app.cell(hide_code=True)
def _(DataPrepPipeline, Path):
    # Initialize high-level pipeline instance with repository data root
    repo_data_root = Path("data")
    pipeline = DataPrepPipeline(data_root=repo_data_root)
    return (pipeline,)


@app.cell(hide_code=True)
def _(mo, pipeline):
    # Dataset inventory matrix
    inv = pipeline.get_inventory()

    rows = []
    for split_name, counts in inv.items():
        raw_c = counts["raw_images"]
        lbl_c = counts["labels"]
        proc_c = counts["processed"]
        pct = (proc_c / raw_c * 100.0) if raw_c > 0 else 0.0
        status_badge = "🟢 Complete" if (raw_c > 0 and proc_c >= raw_c) else ("🟡 In Progress" if proc_c > 0 else "⚪ Not Started")
        rows.append(
            f"| **{split_name}** | {raw_c:,} | {lbl_c:,} | {proc_c:,} | {pct:.1f}% | {status_badge} |"
        )

    table_content = "\n".join([
        "| Split | Raw Images (`images/`) | YOLO Labels (`labels/`) | Processed (`processed/`) | Progress | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        *rows,
    ])

    mo.md(
        f"""### 📊 Dataset Inventory & Split Health

    {table_content}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pipeline):
    # Execution Controls
    mode_ui = mo.ui.radio(
        options=["Single File", "Single Split", "Entire Dataset"],
        value="Single File",
        label="**Execution Granularity:**",
    )

    split_ui = mo.ui.dropdown(
        options=["train", "val", "test"],
        value="train",
        label="**Target Split:**",
    )

    train_images = [p.name for p in pipeline.list_images("train")[:50]]
    if not train_images:
        train_images = ["1.png"]

    file_ui = mo.ui.dropdown(
        options=train_images,
        value=train_images[0],
        label="**Sample Image (for preview & single run):**",
    )

    ref_slide_ui = mo.ui.text(
        value="data/train/images/1.png",
        label="**Reference Slide for Reinhard Normalization:**",
    )

    blur_ui = mo.ui.slider(
        start=1,
        stop=11,
        step=2,
        value=5,
        label="**Otsu Gaussian Blur Kernel:**",
    )

    workers_ui = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=4,
        label="**Batch Parallel Workers:**",
    )

    show_annotations_ui = mo.ui.checkbox(
        value=False,
        label="**Show Annotations**",
    )

    execute_btn = mo.ui.run_button(
        label="🚀 Execute Data Preparation",
    )

    mo.md(
        f"""
        ### ⚙️ Pipeline Configuration & Controls

        {mo.hstack([mode_ui, split_ui, file_ui], justify="start")}

        {mo.hstack([ref_slide_ui, blur_ui, workers_ui], justify="start")}

        {mo.hstack([show_annotations_ui, execute_btn], justify="start")}
        """
    )
    return (
        blur_ui,
        execute_btn,
        file_ui,
        mode_ui,
        ref_slide_ui,
        show_annotations_ui,
        split_ui,
        workers_ui,
    )


@app.cell(hide_code=True)
def _(
    OtsuSegmenter,
    Path,
    ReinhardNormalizer,
    blur_ui,
    cv2,
    draw_yolo_bboxes,
    file_ui,
    load_yolo_annotations,
    mo,
    plt,
    ref_slide_ui,
    show_annotations_ui,
    split_ui,
):
    # Interactive Visual Preview (Before vs Mask vs After)
    sample_path = Path("data") / split_ui.value / "images" / file_ui.value
    preview_content = None

    if sample_path.exists():
        raw_bgr = cv2.imread(str(sample_path))
        if raw_bgr is not None:
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

            # 1. Otsu Segmentation
            seg = OtsuSegmenter(blur_kernel_size=blur_ui.value)
            mask = seg.compute_mask(raw_bgr)
            cleaned_bgr, _ = seg.remove_background_noise(raw_bgr, mask=mask)

            # 2. Reinhard Normalization
            norm = ReinhardNormalizer()
            ref_path = Path(ref_slide_ui.value)
            if ref_path.exists():
                ref_bgr = cv2.imread(str(ref_path))
                if ref_bgr is not None:
                    norm.fit(ref_bgr)

            proc_bgr = norm.transform(cleaned_bgr, mask=mask, restore_background=True)
            proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

            # Build side-by-side plot with channel histogram
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(raw_rgb)
            axes[0].axis("off")

            axes[1].imshow(mask, cmap="gray")
            axes[1].axis("off")

            axes[2].imshow(proc_rgb)
            axes[2].axis("off")

            # Check if Show Annotations is enabled to overlay YOLO bounding boxes
            tb_count = None
            if show_annotations_ui.value:
                label_path = (
                    Path("data")
                    / split_ui.value
                    / "labels"
                    / f"{Path(file_ui.value).stem}.txt"
                )
                bboxes = load_yolo_annotations(
                    label_path,
                    img_width=raw_rgb.shape[1],
                    img_height=raw_rgb.shape[0],
                )
                tb_count = len(bboxes)
                for ax in axes[:3]:
                    draw_yolo_bboxes(
                        ax, bboxes, edgecolor="#00FF00", linewidth=1.8
                    )

            tb_suffix = f" — {tb_count} TBs" if tb_count is not None else ""
            axes[0].set_title(f"1. Raw Image{tb_suffix}\n({file_ui.value})")
            axes[1].set_title(f"2. Otsu Mask{tb_suffix}\n(Kernel={blur_ui.value})")
            axes[2].set_title(f"3. Processed Result{tb_suffix}\n(Otsu + Reinhard)")

            # Color distribution comparison
            for idx, c_color in enumerate(["red", "green", "blue"]):
                axes[3].hist(
                    raw_rgb[:, :, idx].ravel(),
                    bins=32,
                    range=(0, 255),
                    color=c_color,
                    alpha=0.3,
                    linestyle="--",
                    label=f"Raw {c_color[0].upper()}" if idx == 0 else None,
                )
                axes[3].hist(
                    proc_rgb[:, :, idx].ravel(),
                    bins=32,
                    range=(0, 255),
                    color=c_color,
                    alpha=0.6,
                    label=f"Proc {c_color[0].upper()}" if idx == 0 else None,
                )
            axes[3].set_title("4. RGB Histograms\n(Dashed: Raw, Solid: Proc)")
            axes[3].set_xlim(0, 255)
            axes[3].grid(True, alpha=0.3)
            plt.tight_layout()

            preview_content = mo.as_html(fig)
            plt.close(fig)

    mo.md(
        f"""
        ### 🔍 Live Visual Inspection & Color Profile
        {preview_content if preview_content is not None else "Select a valid image to preview."}
        """
    )
    return


@app.cell(hide_code=True)
def _(
    DataPrepPipeline,
    OtsuSegmenter,
    Path,
    ReinhardNormalizer,
    SUPPORTED_SPLITS,
    blur_ui,
    cv2,
    execute_btn,
    file_ui,
    mo,
    mode_ui,
    ref_slide_ui,
    split_ui,
    tqdm,
    workers_ui,
):
    # Execution Runner
    execution_result = None

    if execute_btn.value:
        seg_exec = OtsuSegmenter(blur_kernel_size=blur_ui.value)
        norm_exec = ReinhardNormalizer()
        ref_p = Path(ref_slide_ui.value)
        if ref_p.exists():
            ref_bgr_img = cv2.imread(str(ref_p))
            if ref_bgr_img is not None:
                norm_exec.fit(ref_bgr_img)

        custom_pipeline = DataPrepPipeline(
            data_root=Path("data"),
            segmenter=seg_exec,
            normalizer=norm_exec,
        )

        mode = mode_ui.value
        if mode == "Single File":
            with (
                mo.status.progress_bar(
                    total=1,
                    title="Processing Single Image",
                    subtitle=f"Treating {file_ui.value}...",
                    remove_on_exit=True,
                ) as bar,
                tqdm(total=1, desc=f"Processing {file_ui.value}") as pbar,
            ):
                out_file = custom_pipeline.process_single_file(
                    split=split_ui.value,
                    filename_or_path=file_ui.value,
                    overwrite=True,
                )
                bar.update(increment=1, subtitle="Complete")
                pbar.update(1)

            execution_result = mo.md(
                f"""
                ✅ **Single File Processed Successfully!**  
                - Source: `data/{split_ui.value}/images/{file_ui.value}`  
                - Destination: `data/{split_ui.value}/processed/{file_ui.value}`  
                - Output verified on disk: `{out_file.exists()}`
                """
            )
        elif mode == "Single Split":
            target_split = split_ui.value
            split_images = custom_pipeline.list_images(target_split)
            total_count = len(split_images)

            with (
                mo.status.progress_bar(
                    total=total_count,
                    title=f"Processing Split: {target_split}",
                    subtitle="Starting...",
                    show_rate=True,
                    show_eta=True,
                    remove_on_exit=True,
                ) as bar,
                tqdm(total=total_count, desc=f"Split [{target_split}]") as pbar,
            ):
                def on_split_progress(curr, tot, fn):
                    bar.update(
                        increment=1,
                        subtitle=f"[{curr}/{tot}] {fn} ({target_split})",
                    )
                    pbar.update(1)

                stats = custom_pipeline.process_split(
                    split=target_split,
                    overwrite=True,
                    progress_callback=on_split_progress,
                    max_workers=workers_ui.value,
                )

            execution_result = mo.md(
                f"""
                ✅ **Split `{target_split}` Completed!**  
                - Total Images: **{stats.total_images}**  
                - Successfully Processed: **{stats.processed_images}**  
                - Elapsed Time: **{stats.elapsed_time:.2f}s** (avg **{stats.avg_time_per_image*1000:.1f}ms / image**)  
                - Errors: **{len(stats.failed_images)}**
                """
            )
        elif mode == "Entire Dataset":
            all_images_count = sum(
                len(custom_pipeline.list_images(s)) for s in SUPPORTED_SPLITS
            )

            with (
                mo.status.progress_bar(
                    total=all_images_count,
                    title="Processing Entire Dataset (train, val, test)",
                    subtitle="Starting...",
                    show_rate=True,
                    show_eta=True,
                    remove_on_exit=True,
                ) as bar,
                tqdm(total=all_images_count, desc="Entire Dataset") as pbar,
            ):
                def on_all_progress(curr_split, curr, tot, fn):
                    bar.update(
                        increment=1,
                        subtitle=f"[{curr_split.upper()}] {fn} ({curr}/{tot})",
                    )
                    pbar.update(1)

                all_stats = custom_pipeline.process_all(
                    overwrite=True,
                    progress_callback=on_all_progress,
                    max_workers=workers_ui.value,
                )

            summary_lines = []
            for s, st in all_stats.items():
                summary_lines.append(
                    f"- **`{s}`**: {st.processed_images}/{st.total_images} in {st.elapsed_time:.1f}s"
                )
            summary_txt = "\n".join(summary_lines)
            execution_result = mo.md(
                f"""
                ✅ **Entire Dataset Processed Successfully!**  
                {summary_txt}
                """
            )

    mo.md(
        f"""
        ### 🚀 Execution Output
        {execution_result if execution_result is not None else "Click 'Execute Data Preparation' above to run processing."}
        """
    )
    return


if __name__ == "__main__":
    app.run()
