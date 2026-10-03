"""Data preparation pipeline orchestrating Otsu thresholding and Reinhard color normalization.

Processes images from data/{split}/images/ and writes standardized images to
data/{split}/processed/ with full idempotency (replacing/overwriting existing outputs).
Supports single-file, single-split, and full-dataset execution modes.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from histojepa.data.stain_norm import ReinhardNormalizer
from histojepa.data.tissue_mask import OtsuSegmenter


SUPPORTED_SPLITS = ("train", "val", "test")


@dataclass
class ProcessingStats:
    """Statistics summary for a data preparation run."""

    split: str
    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: List[Tuple[str, str]] = field(default_factory=list)
    elapsed_time: float = 0.0

    @property
    def avg_time_per_image(self) -> float:
        if self.processed_images > 0:
            return self.elapsed_time / self.processed_images
        return 0.0


def prepare_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    segmenter: Optional[OtsuSegmenter] = None,
    normalizer: Optional[ReinhardNormalizer] = None,
    overwrite: bool = True,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Process a single image using Otsu thresholding and Reinhard color normalization.

    Args:
        input_path: Path to source raw image.
        output_path: Path to destination processed image.
        segmenter: OtsuSegmenter instance. Defaults to default settings.
        normalizer: ReinhardNormalizer instance. Defaults to default CK settings.
        overwrite: If True, replaces existing output file idempotently.
        bg_color: Replacement background BGR color.

    Returns:
        Path to the saved processed image.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Idempotency check: if output exists and overwrite is True, ensure it will be replaced
    if output_path.exists() and not overwrite:
        return output_path

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read input image in BGR
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image at {input_path}")

    if segmenter is None:
        segmenter = OtsuSegmenter(default_bg_color=bg_color)
    if normalizer is None:
        normalizer = ReinhardNormalizer()

    # 2. Otsu Automated Thresholding for background noise removal
    mask = segmenter.compute_mask(img_bgr)
    cleaned_bgr, _ = segmenter.remove_background_noise(img_bgr, mask=mask, bg_color=bg_color)

    # 3. Reinhard Color Normalization (applied with tissue mask preservation)
    normalized_bgr = normalizer.transform(
        cleaned_bgr, mask=mask, restore_background=True, bg_color=bg_color
    )

    # 4. Idempotent write: write to output_path (overwriting any previous file)
    # If file exists, we can remove it first or let cv2.imwrite overwrite it cleanly
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass

    success = cv2.imwrite(str(output_path), normalized_bgr)
    if not success:
        raise IOError(f"Failed to write processed image to {output_path}")

    return output_path


def _worker_process_file(args: Tuple[str, str, Tuple[int, int, int], Optional[Tuple[List[float], List[float]]]]) -> Tuple[bool, str, Optional[str]]:
    """Helper worker for multi-process batch execution."""
    src_str, dst_str, bg_color, norm_stats = args
    src = Path(src_str)
    dst = Path(dst_str)
    try:
        seg = OtsuSegmenter(default_bg_color=bg_color)
        if norm_stats is not None:
            mean_vals, std_vals = norm_stats
            norm = ReinhardNormalizer(
                target_mean=np.array(mean_vals, dtype=np.float32),
                target_std=np.array(std_vals, dtype=np.float32),
            )
        else:
            norm = ReinhardNormalizer()

        prepare_image(src, dst, segmenter=seg, normalizer=norm, overwrite=True, bg_color=bg_color)
        return True, src.name, None
    except Exception as exc:
        return False, src.name, str(exc)


class DataPrepPipeline:
    """Dataset preparation pipeline for the TB-YOLO benchmark.

    Manages dataset discovery, Otsu noise removal, Reinhard color normalization,
    and idempotent output persistence across data/train, data/val, and data/test.
    """

    def __init__(
        self,
        data_root: Union[str, Path] = "data",
        segmenter: Optional[OtsuSegmenter] = None,
        normalizer: Optional[ReinhardNormalizer] = None,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.data_root = Path(data_root)
        self.segmenter = segmenter or OtsuSegmenter(default_bg_color=bg_color)
        self.normalizer = normalizer or ReinhardNormalizer()
        self.bg_color = bg_color

    def get_split_dir(self, split: str) -> Path:
        """Get path to a split directory (train, val, or test)."""
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {SUPPORTED_SPLITS}")
        return self.data_root / split

    def list_images(self, split: str) -> List[Path]:
        """List all raw image paths in data/{split}/images/."""
        images_dir = self.get_split_dir(split) / "images"
        if not images_dir.exists():
            return []
        return sorted([p for p in images_dir.glob("*.png")])

    def get_inventory(self) -> Dict[str, Dict[str, int]]:
        """Get image, label, and processed counts across all splits."""
        inventory = {}
        for split in SUPPORTED_SPLITS:
            split_dir = self.get_split_dir(split)
            img_count = len(list((split_dir / "images").glob("*.png"))) if (split_dir / "images").exists() else 0
            lbl_count = len(list((split_dir / "labels").glob("*.txt"))) if (split_dir / "labels").exists() else 0
            proc_count = len(list((split_dir / "processed").glob("*.png"))) if (split_dir / "processed").exists() else 0
            inventory[split] = {
                "raw_images": img_count,
                "labels": lbl_count,
                "processed": proc_count,
            }
        return inventory

    def process_single_file(
        self,
        split: str,
        filename_or_path: Union[str, Path],
        overwrite: bool = True,
    ) -> Path:
        """Process a single image file within a specific split.

        Args:
            split: Name of split ('train', 'val', or 'test').
            filename_or_path: Filename (e.g. '1.png') or full path to image.
            overwrite: Whether to overwrite existing processed file.

        Returns:
            Path to processed file in data/{split}/processed/.
        """
        split_dir = self.get_split_dir(split)
        if isinstance(filename_or_path, Path) and filename_or_path.is_file():
            src_path = filename_or_path
        else:
            name = Path(filename_or_path).name
            src_path = split_dir / "images" / name

        dst_path = split_dir / "processed" / src_path.name

        return prepare_image(
            src_path,
            dst_path,
            segmenter=self.segmenter,
            normalizer=self.normalizer,
            overwrite=overwrite,
            bg_color=self.bg_color,
        )

    def process_split(
        self,
        split: str,
        overwrite: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 1,
    ) -> ProcessingStats:
        """Process all images in a given split idempotently.

        Args:
            split: One of 'train', 'val', 'test'.
            overwrite: Whether to replace existing processed files.
            progress_callback: Optional callback(current_index, total, filename).
            max_workers: Number of parallel workers (1 = sequential).

        Returns:
            ProcessingStats summarizing execution results.
        """
        images = self.list_images(split)
        stats = ProcessingStats(split=split, total_images=len(images))
        dst_dir = self.get_split_dir(split) / "processed"
        dst_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        if max_workers > 1 and len(images) > 1:
            norm_stats = None
            if self.normalizer.target_mean is not None and self.normalizer.target_std is not None:
                norm_stats = (
                    self.normalizer.target_mean.tolist(),
                    self.normalizer.target_std.tolist(),
                )

            worker_tasks = [
                (str(img_path), str(dst_dir / img_path.name), self.bg_color, norm_stats)
                for img_path in images
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_worker_process_file, task): task for task in worker_tasks}
                for idx, future in enumerate(as_completed(futures), start=1):
                    ok, name, err = future.result()
                    if ok:
                        stats.processed_images += 1
                    else:
                        stats.failed_images.append((name, err or "Unknown error"))
                    if progress_callback:
                        progress_callback(idx, len(images), name)
        else:
            for idx, img_path in enumerate(images, start=1):
                dst_path = dst_dir / img_path.name
                try:
                    prepare_image(
                        img_path,
                        dst_path,
                        segmenter=self.segmenter,
                        normalizer=self.normalizer,
                        overwrite=overwrite,
                        bg_color=self.bg_color,
                    )
                    stats.processed_images += 1
                except Exception as exc:
                    stats.failed_images.append((img_path.name, str(exc)))

                if progress_callback:
                    progress_callback(idx, len(images), img_path.name)

        stats.elapsed_time = time.time() - start_time
        return stats

    def process_all(
        self,
        overwrite: bool = True,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        max_workers: int = 1,
    ) -> Dict[str, ProcessingStats]:
        """Process all splits (train, val, test) sequentially.

        Args:
            overwrite: Whether to overwrite existing files.
            progress_callback: Optional callback(split, current_idx, total, filename).
            max_workers: Number of parallel workers per split.

        Returns:
            Dictionary mapping split name to ProcessingStats.
        """
        all_stats = {}
        for split in SUPPORTED_SPLITS:
            cb = None
            if progress_callback:
                cb = lambda curr, tot, fn, s=split: progress_callback(s, curr, tot, fn)
            all_stats[split] = self.process_split(
                split=split,
                overwrite=overwrite,
                progress_callback=cb,
                max_workers=max_workers,
            )
        return all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HistoJEPA Data Preparation Pipeline")
    parser.add_argument(
        "--data-root", type=str, default="data", help="Root data folder containing train/val/test"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test", "all"], default="all",
        help="Split to process"
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Process a single image filename or path"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing processed files"
    )

    args = parser.parse_args()
    pipeline = DataPrepPipeline(data_root=args.data_root)

    if args.single:
        split = "train" if args.split == "all" else args.split
        print(f"Processing single file: {args.single} in split {split}")
        res = pipeline.process_single_file(split, args.single, overwrite=not args.no_overwrite)
        print(f"Done! Processed file saved to {res}")
    elif args.split == "all":
        print("Processing all splits (train, val, test)...")
        results = pipeline.process_all(
            overwrite=not args.no_overwrite,
            progress_callback=lambda s, cur, tot, fn: print(f"[{s}] {cur}/{tot}: {fn}", end="\r"),
            max_workers=args.workers,
        )
        print("\nFinished processing all splits!")
        for s, st in results.items():
            print(f"  {s}: {st.processed_images}/{st.total_images} processed in {st.elapsed_time:.1f}s")
    else:
        print(f"Processing split: {args.split}...")
        st = pipeline.process_split(
            args.split,
            overwrite=not args.no_overwrite,
            progress_callback=lambda cur, tot, fn: print(f"[{args.split}] {cur}/{tot}: {fn}", end="\r"),
            max_workers=args.workers,
        )
        print(f"\nDone! Processed {st.processed_images}/{st.total_images} images in {st.elapsed_time:.1f}s")
