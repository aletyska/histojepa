# 🔬 HistoJEPA: A Joint Embedding Predictive Architecture for the Clustering and Detection of Tumor Buds in Colorectal Cancer Whole Slide Images

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![marimo](https://img.shields.io/badge/notebooks-marimo-purple.svg)](https://marimo.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

**Master's Thesis in Applied Computing — UNISINOS**  
*Thesis Title:* HistoJEPA: A Joint Embedding Predictive Architecture for the Clustering and Detection of Tumor Buds in Colorectal Cancer Whole Slide Images  
*Advisor:* Prof. Dr. Cristiano André da Costa | *Co-Advisor:* Prof. Dr. Adriana Vial Roehe

---

## 📌 Executive Summary

**HistoJEPA** is a dual-pipeline computational pathology framework designed to automate the detection and curation of **Tumor Buds (TBs)**—microscopic, high-risk 1-to-4 cell clusters along the invasive front of Colorectal Cancer (CRC) Whole Slide Images (WSIs). 

By leveraging self-supervised representation learning via an **Image-based Joint-Embedding Predictive Architecture (I-JEPA)** on a Vision Transformer (ViT) backbone, HistoJEPA eliminates the need for expensive pixel-level reconstructions or manual single-cell bounding box annotations across entire clinical cohorts.

```text
                                  [ Raw WSI Tissue Patches ]
                                              │
                                              ▼
                ┌───────────────────────────────────────────────────────────┐
                │  PHASE 1: Self-Supervised Pre-Training (I-JEPA)           │
                │  Frozen ViT-Base Context Encoder (Abstract Latent Space)  │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
        ┌─────────────────────────────┐               ┌─────────────────────────────┐
        │  BRANCH A: Supervised       │               │  BRANCH B: Unsupervised     │
        │  Detection (ViTDet)         │               │  Clustering & HITL Curation │
        │                             │               │                             │
        │ - Frozen ViT Encoder        │               │ - Frozen ViT Encoder        │
        │ - Simple Feature Pyramid    │               │ - K-Means / DBSCAN          │
        │ - Detection Head            │               │ - 2D UMAP Projection        │
        │ - Evaluation: mAP50-95, F1  │               │ - 1-Click Bulk Annotation   │
        └─────────────────────────────┘               └─────────────────────────────┘
```

---

## 🏛️ System Architecture

### 1. Phase 1: Self-Supervised Representation Learning (I-JEPA)
Pre-trains a non-hierarchical ViT-Base backbone using I-JEPA's non-generative semantic prediction in representation space . The Context Encoder processes $85\%\text{--}100\%$ masked tissue tiles, predicting the high-dimensional latent embeddings of target blocks without decoding raw RGB pixels.

### 2. Branch A: Supervised Object Detection (ViTDet)
Connects the frozen ViT Context Encoder to Meta AI's ViTDet Simple Feature Pyramid (SFP). The SFP constructs multi-scale feature maps ($P_2, P_3, P_4, P_5$ at strides 4, 8, 16, 32) post-backbone, enabling a lightweight detection head to regress bounding box coordinates around microscopic tumor buds on the TB-YOLO benchmark (derived from TBCMU).

### 3. Branch B: Unsupervised Clustering & HITL Bulk-Annotation
Operates directly on the frozen $\mathbb{R}^D$ latent representations. Uses unsupervised clustering algorithms (K-Means, DBSCAN, Spectral Clustering) and UMAP dimensionality reduction to power a Human-in-the-Loop (HITL) Marimo web application. Pathologists can inspect cluster centroids via representative RGB patch galleries and apply 1-click bulk annotations to thousands of patches simultaneously.

---

## 📂 Repository Structure

```text
histojepa/
├── pyproject.toml                  # Project metadata & uv dependencies
├── uv.lock                         # Deterministic environment lockfile
├── AGENTS.md                       # Instructions and context for AI coding agents
├── configs/                        # YAML configuration files for experiments
│   ├── phase1_ijepa.yaml
│   ├── branch_a_vitdet.yaml
│   └── branch_b_clustering.yaml
├── data/                           # TB-YOLO dataset (derived from TBCMU, pre-split)
│   ├── train/                      # Training split
│   │   ├── images/                 # Raw images (1280x1280 RGB)
│   │   ├── labels/                 # YOLO format tumor bud annotations (.txt)
│   │   └── processed/              # Extracted patches & treated images
│   ├── val/                        # Validation split
│   │   ├── images/                 # Raw images (1280x1280 RGB)
│   │   ├── labels/                 # YOLO format tumor bud annotations (.txt)
│   │   └── processed/              # Extracted patches & treated images
│   └── test/                       # Test split
│       ├── images/                 # Raw images (1280x1280 RGB)
│       ├── labels/                 # YOLO format tumor bud annotations (.txt)
│       └── processed/              # Extracted patches & treated images
├── src/                            # Core Python package
│   └── histojepa/
│       ├── data/                   # Preprocessing & dataset utilities
│       │   ├── tissue_mask.py      # Otsu thresholding & background noise removal
│       │   ├── stain_norm.py       # Reinhard color normalization (OpenCV Lab space)
│       │   └── pipeline.py         # Idempotent batch pipeline
│       ├── models/                 # PyTorch Modules (I-JEPA, ViTDet SFP, Clustering)
│       ├── evaluation/             # Metrics suite (mAP, F1, Purity %, ARI, NMI)
│       └── utils/                  # Visualization & checkpoint handlers
├── notebooks/                      # Marimo reactive notebooks (.py format)
│   ├── 00_index.py                 # Central Project Dashboard
│   ├── 01_Data_Preparation.py        # Image pre-processing, patch extraction & dataset verification
│   ├── 02_Phase_1_IJEPA_PreTrain.py   # I-JEPA SSL pre-training, frozen ViT encoder & linear probe benchmark
│   ├── 03_Phase_2_Branch_A.py        # Branch A end-to-end pipeline: ViTDet SFP training to evaluation
│   └── 04_Phase_2_Branch_B.py        # Branch B end-to-end pipeline: cluster tuning, HITL app & evaluation
├── tests/                          # pytest unit tests partitioned by notebook
│   ├── conftest.py                 # Shared synthetic test fixtures
│   └── nb01_data_prep/             # Tests bound to 01_Data_Preparation.py (marker: nb01)
└── outputs/                        # Checkpoints, logs, and figures (git-ignored)
```

---

## 💾 Dataset & Data Organization

The project uses **TB-YOLO**, a derived dataset from the **TBCMU** benchmark (Tumor Budding in Colorectal Cancer Whole Slide Images).

### Key Dataset Attributes:
* **Pre-Split Partitions:** The dataset is already partitioned into `train/`, `val/`, and `test/` sets, preserving strict patient-level separation to prevent data leakage.
* **Tumor Bud Annotations:** Annotations are provided as normalized YOLO bounding box text files (`class x_center y_center width height`).
* **Directory Structure:**
  Each split folder contains three subdirectories:
  * `images/`: Original raw histopathology images (1280×1280 RGB).
  * `labels/`: Text files (`.txt`) containing the ground-truth YOLO tumor bud annotations.
  * `processed/`: Storage for extracted patches (e.g., 224×224 tiles for ViT/I-JEPA) and treated/preprocessed images.

---

## 🛠️ Quick Start & Installation

This project uses `uv` for deterministic Python environment management and `marimo` for reactive notebooks.

### 1. Prerequisites
- Python 3.12+
- CUDA-compatible GPU (NVIDIA Driver $\ge$ 525)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/histojepa.git
cd histojepa

# Synchronize dependencies with uv
uv sync

# Activate virtual environment (optional)
source .venv/bin/activate
```

### 3. Data Preparation & Preprocessing
The data preparation pipeline applies **Otsu automated thresholding** for background noise removal and **Reinhard color normalization** in Lab space for CK-stained slides.

Launch the interactive Marimo notebook:
```bash
uv run marimo edit notebooks/01_Data_Preparation.py
```

Or execute the pipeline directly via CLI:
```bash
# Process a single split
uv run python -m histojepa.data.pipeline --split train --workers 4

# Process the entire dataset (train, val, test)
uv run python -m histojepa.data.pipeline --split all --workers 4

# Process a single image
uv run python -m histojepa.data.pipeline --single 1.png --split train
```

### 4. Running Unit Tests
Unit tests are partitioned by notebook phase:
```bash
# Run tests bound to Notebook 01 (Data Preparation)
uv run pytest tests/nb01_data_prep -v
# or via marker:
uv run pytest -m nb01 -v

# Run full project test suite
uv run pytest
```

### 5. Running Notebooks with Marimo
```bash
# Open the central Project Dashboard
uv run marimo edit notebooks/00_index.py

# Launch the interactive Pathologist HITL Web Application
uv run marimo run notebooks/04_Phase_2_Branch_B.py
```

---

## 📊 Quantitative Metrics Suite

- **Phase 1:** $L_2$ Latent Prediction Loss, Linear Probing Accuracy, Silhouette Score.
- **Branch A (Detection):** $\text{mAP}_{50-95}$, IoU, Precision, Recall, $F_1$-Score, Label Efficiency Curve (10%, 25%, 50%, 100% annotations).
- **Branch B (Clustering & HITL):** Cluster Purity %, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), HCI Time-per-1,000-Annotations.
