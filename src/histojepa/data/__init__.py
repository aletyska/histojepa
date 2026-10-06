"""Data preparation and dataset handling for HistoJEPA.

Exports tissue segmentation via Otsu thresholding, Reinhard color normalization,
and the full data preparation pipeline.
"""

from histojepa.data.tissue_mask import OtsuSegmenter, segment_and_clean
from histojepa.data.stain_norm import ReinhardNormalizer, normalize_stain
from histojepa.data.pipeline import (
    DataPrepPipeline,
    prepare_image,
    ProcessingStats,
    SUPPORTED_SPLITS,
)
from histojepa.data.ijepa_dataset import ProcessedImageDataset

__all__ = [
    "OtsuSegmenter",
    "segment_and_clean",
    "ReinhardNormalizer",
    "normalize_stain",
    "DataPrepPipeline",
    "prepare_image",
    "ProcessingStats",
    "SUPPORTED_SPLITS",
    "ProcessedImageDataset",
]
