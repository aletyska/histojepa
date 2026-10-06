# 🤖 AGENTS.md — Instructions for AI Coding Assistants

This document provides mandatory context, technical constraints, coding conventions, and architectural guidelines for AI coding agents (e.g., Cursor, Claude Code, Windsurf, Copilot Workspace) working on the **HistoJEPA** codebase.

---

## 🎯 Project Overview & Mission

**HistoJEPA** is an Applied Computing Master's thesis codebase at UNISINOS. It implements a self-supervised Vision Transformer (ViT) framework for processing Whole Slide Images (WSIs) of Colorectal Cancer to detect and cluster **Tumor Buds (TBs)**.

### Core Architectural Mandates:
1. **Phase 1 (I-JEPA):** Pre-trains a plain ViT-Base backbone (HuggingFace `transformers` `IJepaModel`) via non-generative joint-embedding prediction in latent space.
2. **Phase 2 - Branch A (Supervised Detection):** Connects Meta AI's **ViTDet Simple Feature Pyramid (SFP)** to the **strictly frozen** ViT Context Encoder. Only the SFP and detection head are trained using ground-truth bounding box labels from the TB-YOLO dataset (derived from TBCMU, pre-split).
3. **Phase 2 - Branch B (Unsupervised Clustering):** Extracts $\mathbb{R}^D$ feature vectors from the **strictly frozen** ViT Context Encoder and applies K-Means/DBSCAN/Spectral clustering and UMAP for Human-in-the-Loop (HITL) bulk-annotation.

---

## 🛠️ Technology Stack & Tooling Constraints

Agents MUST strictly follow these environment rules:

* **Package Manager:** **`uv`** (Astral). NEVER use `pip install` directly. All dependency additions must go through `uv add <package>` or `uv sync`.
* **Notebook Framework:** **`marimo`**. Notebooks are saved as pure Python `.py` scripts under `notebooks/`. NEVER generate JSON-based `.ipynb` files.
* **Deep Learning Framework:** PyTorch 2.2+ with CUDA acceleration.
* **ViT Backbone:** HuggingFace `transformers` (`IJepaModel`, random initialization from scratch for histopathology).
* **Data & Image Processing:** `albumentations`, `opencv-python`, `Pillow`, `openslide-python`.
* **Clustering & Dimensionality Reduction:** `scikit-learn`, `umap-learn`.

---

## 📐 Project Structure Rules & Code Placement

```text
histojepa/
├── data/               <-- TB-YOLO dataset (derived from TBCMU, pre-split into train/val/test).
│   ├── train/          <-- images/ (raw 1280x1280), labels/ (YOLO txt), processed/ (patches/treated)
│   ├── val/            <-- images/ (raw 1280x1280), labels/ (YOLO txt), processed/ (patches/treated)
│   └── test/           <-- images/ (raw 1280x1280), labels/ (YOLO txt), processed/ (patches/treated)
├── papers/             <-- LOCAL ONLY (gitignored). Reference papers converted to Markdown:
│   └── <citation_key>/ <-- e.g. ijepa/, vitdet/ (lowercase snake_case citation key)
│       ├── paper.md    # Full paper in Markdown
│       ├── figures/    # Extracted figures (optional)
│       └── notes.md    # Reproduction notes & deviations (optional)
├── src/histojepa/      <-- ALL reusable PyTorch models, data loaders, and metrics GO HERE:
│   ├── data/           <-- Preprocessing, stain norm, and dataset loaders:
│   │   ├── tissue_mask.py      # Otsu automated thresholding & background noise removal
│   │   ├── stain_norm.py       # Reinhard color normalization in OpenCV Lab space
│   │   ├── pipeline.py         # Batch pipeline & idempotent split processing
│   │   └── ijepa_dataset.py    # Unlabelled processed image dataset for I-JEPA
│   ├── models/         <-- PyTorch modules (I-JEPA, ViTDet SFP, heads):
│   │   ├── ijepa/              # Context/Target Encoders, Predictor, Multi-Block Masking, Trainer
│   │   ├── vitdet/             # ViTDet Simple Feature Pyramid (SFP) & detection heads
│   │   └── clustering/         # Unsupervised clustering & projection modules
│   ├── evaluation/     <-- Metrics suite (mAP, F1, ARI, NMI)
│   └── utils/          <-- Checkpoints & visualization helpers
├── notebooks/          <-- ONLY marimo notebook drivers (.py files) GO HERE:
│   ├── 00_index.py                   # Central Project Dashboard
│   ├── 01_Data_Preparation.py        # Image pre-processing & patch preparation
│   ├── 02_Phase_1_IJEPA_PreTrain.py   # I-JEPA SSL pre-training & frozen Context Encoder export
│   ├── 03_Phase_2_Branch_A.py        # Branch A: ViTDet SFP training to evaluation
│   └── 04_Phase_2_Branch_B.py        # Branch B: Cluster tuning, interactive HITL app & evaluation
├── configs/            <-- YAML files for experiment hyperparameters.
└── tests/              <-- pytest unit tests partitioned by notebook:
    ├── conftest.py                   # Shared synthetic fixtures
    ├── nb01_data_prep/               # Tests bound to 01_Data_Preparation.py (marker: nb01)
    ├── nb02_phase1_ijepa/            # Tests bound to 02_Phase_1_IJEPA_PreTrain.py (marker: nb02)
    ├── nb03_phase2_branch_a/         # Tests bound to 03_Phase_2_Branch_A.py (marker: nb03)
    └── nb04_phase2_branch_b/         # Tests bound to 04_Phase_2_Branch_B.py (marker: nb04)
```

### Critical Placement Guidelines:
* **No Monolithic Notebooks:** Do NOT write PyTorch `nn.Module` classes, dataset classes, or loss functions exclusively inside `notebooks/`.
* **Module Definition:** Define all models inside `src/histojepa/models/`, datasets inside `src/histojepa/data/`, and metrics inside `src/histojepa/evaluation/`.
* **Notebook Role:** Notebooks should only import from `histojepa.*`, parse configs, execute training/evaluation loops, and display plots or interactive UI elements.

---

## 📚 Reference Papers (`papers/`) — Local Only

The `papers/` directory holds the scientific papers this project reproduces, converted to Markdown.
The **entire folder is gitignored** (copyrighted material) and may be missing on other machines or in CI.

### Layout
* `papers/<citation_key>/paper.md`: full paper text. `<citation_key>` is a short lowercase `snake_case` identifier matching the thesis BibTeX key (e.g. `ijepa`, `vitdet`, `tb_yolo_dataset`).
* `papers/<citation_key>/figures/`: figures extracted during conversion (optional).
* `papers/<citation_key>/notes.md`: reproduction notes such as hyperparameters, deviations and a paper→code map (optional).

### Rules for Agents
1. **Consult before implementing:** Before implementing or changing a component that comes from a paper (e.g. I-JEPA masking, predictor, EMA schedule, ViTDet SFP), read `papers/<key>/paper.md` and follow its equations, architecture details and hyperparameters.
2. **Trace to the source:** In docstrings, cite the paper section/equation/table being implemented, e.g. `Ref: I-JEPA (papers/ijepa/paper.md), §3 "Method", Eq. (1)`.
3. **Document deviations:** If the implementation must differ from the paper (compute limits, histopathology domain, 1280×1280 inputs), say so in the docstring or config comment and record it in `papers/<key>/notes.md`.
4. **Never commit paper content:** Do not move, copy or paste substantial paper text, figures or PDFs into tracked files (`src/`, `notebooks/`, `README.md`, `configs/`, `tests/`). Short paraphrases and equation references are fine. Never remove `/papers/` from `.gitignore`.
5. **Do not depend on it at runtime:** Code, tests and notebooks MUST NOT read from `papers/`. It is reference material only and is absent on clean clones.
6. **If a paper is missing:** Do not reconstruct paper details from memory. Ask the user to add `papers/<key>/paper.md`, or state clearly which details are unverified.
7. **Adding a paper:** Create `papers/<citation_key>/paper.md` using a lowercase `snake_case` key that matches the BibTeX entry.

---

## 🔒 Hard Architectural & Domain Constraints

### 1. Frozen Backbone Discipline
* The ViT Context Encoder weights MUST remain strictly frozen (`param.requires_grad = False`) during Branch A (ViTDet) and Branch B (Clustering) execution.
* Gradients in Branch A must flow only through the Simple Feature Pyramid (SFP) and the detection head.

### 2. Patient-Level Data Splits (Preventing Data Leakage)
* The primary benchmark dataset is **TB-YOLO** (derived from TBCMU), which comes already partitioned into `train`, `val`, and `test` splits to guarantee patient-level isolation without cross-split leakage.
* Never perform random patch-level splitting across splits. Respect the existing train/val/test partitions.
* Each split contains:
  * `images/`: Raw histopathology images (1280×1280 RGB).
  * `labels/`: Ground-truth bounding boxes in YOLO format (`.txt`: `<class> <x_center> <y_center> <width> <height>`).
  * `processed/`: Storage location for extracted patches (e.g. 224×224 tiles) and treated/preprocessed images.

### 3. Tensor Dimension Conventions
When implementing model components, document tensor shapes explicitly in docstrings using BCHW or Sequence notation:
* **Input Image Patches:** `(B, 3, 224, 224)` or `(B, 3, 1280, 1280)`
* **ViT Patch Tokens:** `(B, N_patches, D)` where $D = 768$ for ViT-Base
* **Reshaped Spatial Grid:** `(B, D, H/16, W/16)`
* **ViTDet Feature Pyramid Outputs:**
  * $P_2$: `(B, 256, H/4, W/4)` (Stride 4)
  * $P_3$: `(B, 256, H/8, W/8)` (Stride 8)
  * $P_4$: `(B, 256, H/16, W/16)` (Stride 16)
  * $P_5$: `(B, 256, H/32, W/32)` (Stride 32)

---

## 🧪 Testing & Validation Expectations

Before committing generated code:
* Run unit tests bound to the current notebook:
  * For Notebook 01: `uv run pytest tests/nb01_data_prep` (or `uv run pytest -m nb01`)
  * For full regression suite: `uv run pytest`
* Verify that `SimpleFeaturePyramid` handles tensor upsampling/downsampling without throwing spatial dimension mismatch errors.
* Verify that evaluation metrics handle empty bounding box predictions gracefully without raising `ZeroDivisionError`.
