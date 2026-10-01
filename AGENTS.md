# 🤖 AGENTS.md — Instructions for AI Coding Assistants

This document provides mandatory context, technical constraints, coding conventions, and architectural guidelines for AI coding agents (e.g., Cursor, Claude Code, Windsurf, Copilot Workspace) working on the **HistoJEPA** codebase.

---

## 🎯 Project Overview & Mission

**HistoJEPA** is an Applied Computing Master's thesis codebase at UNISINOS. It implements a self-supervised Vision Transformer (ViT) framework for processing Whole Slide Images (WSIs) of Colorectal Cancer to detect and cluster **Tumor Buds (TBs)**.

### Core Architectural Mandates:
1. **Phase 1 (I-JEPA):** Pre-trains a plain ViT-Base backbone (`timm` ViT) via non-generative joint-embedding prediction in latent space \cite{ijepa}.
2. **Phase 2 - Branch A (Supervised Detection):** Connects Meta AI's **ViTDet Simple Feature Pyramid (SFP)** to the **strictly frozen** ViT Context Encoder. Only the SFP and detection head are trained using ground-truth bounding box labels from the TB-YOLO dataset \cite{tb_yolo_dataset}.
3. **Phase 2 - Branch B (Unsupervised Clustering):** Extracts $\mathbb{R}^D$ feature vectors from the **strictly frozen** ViT Context Encoder and applies K-Means/DBSCAN/Spectral clustering and UMAP for Human-in-the-Loop (HITL) bulk-annotation.

---

## 🛠️ Technology Stack & Tooling Constraints

Agents MUST strictly follow these environment rules:

* **Package Manager:** **`uv`** (Astral). NEVER use `pip install` directly. All dependency additions must go through `uv add <package>` or `uv sync`.
* **Notebook Framework:** **`marimo`**. Notebooks are saved as pure Python `.py` scripts under `notebooks/`. NEVER generate JSON-based `.ipynb` files.
* **Deep Learning Framework:** PyTorch 2.2+ with CUDA acceleration.
* **Pre-trained Backbones:** `timm` (PyTorch Image Models).
* **Data & Image Processing:** `albumentations`, `opencv-python`, `Pillow`, `openslide-python`.
* **Clustering & Dimensionality Reduction:** `scikit-learn`, `umap-learn`.

---

## 📐 Project Structure Rules & Code Placement

```text
histojepa/
├── src/histojepa/      <-- ALL reusable PyTorch models, data loaders, and metrics GO HERE.
├── notebooks/          <-- ONLY marimo notebook drivers (.py files) GO HERE.
├── configs/            <-- YAML files for experiment hyperparameters.
└── tests/              <-- pytest unit tests for model shapes and metrics.
```

### Critical Placement Guidelines:
* **No Monolithic Notebooks:** Do NOT write PyTorch `nn.Module` classes, dataset classes, or loss functions exclusively inside `notebooks/`.
* **Module Definition:** Define all models inside `src/histojepa/models/`, datasets inside `src/histojepa/data/`, and metrics inside `src/histojepa/evaluation/`.
* **Notebook Role:** Notebooks should only import from `histojepa.*`, parse configs, execute training/evaluation loops, and display plots or interactive UI elements.

---

## 🔒 Hard Architectural & Domain Constraints

### 1. Frozen Backbone Discipline
* The ViT Context Encoder weights MUST remain strictly frozen (`param.requires_grad = False`) during Branch A (ViTDet) and Branch B (Clustering) execution.
* Gradients in Branch A must flow only through the Simple Feature Pyramid (SFP) and the detection head.

### 2. Patient-Level Data Splits (Preventing Data Leakage)
* Histopathology WSI patches extracted from the same patient/WSI MUST belong to the same split (Train, Validation, or Test).
* Never perform random patch-level splitting across WSIs. Check `src/histojepa/data/splits.py` to ensure patient IDs are partitioned strictly without overlap.

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
* Run `uv run pytest` to ensure unit tests pass.
* Verify that `SimpleFeaturePyramid` handles tensor upsampling/downsampling without throwing spatial dimension mismatch errors.
* Verify that evaluation metrics handle empty bounding box predictions gracefully without raising `ZeroDivisionError`.
