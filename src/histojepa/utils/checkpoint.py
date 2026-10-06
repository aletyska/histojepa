"""Checkpoint utilities for saving, loading, finding and pruning training states.

Provides atomic serialization to avoid checkpoint corruption upon unexpected interrupts.
"""

from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union

import torch


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    state: Dict[str, Any],
) -> Path:
    """Atomically saves checkpoint dictionary to disk using a temporary file.

    Args:
        checkpoint_path: Destination path for .pt file.
        state: State dictionary containing model weights, optimizer, step, config, etc.

    Returns:
        Path to the saved checkpoint file.
    """
    dest = Path(checkpoint_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the same directory first, then atomically rename
    with tempfile.NamedTemporaryFile(
        delete=False, dir=dest.parent, prefix=f"{dest.stem}_tmp_", suffix=".pt"
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        torch.save(state, tmp_path)

    shutil.move(str(tmp_path), str(dest))
    return dest


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Loads a checkpoint state dictionary from disk.

    Args:
        checkpoint_path: Path to checkpoint file.
        map_location: Target device for loading tensors (default: "cpu").

    Returns:
        Loaded state dictionary.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Finds the most recent checkpoint in a directory.

    Checks first for latest.pt, then searches for highest epoch ckpt_epoch{N}.pt.

    Args:
        checkpoint_dir: Directory containing saved checkpoints.

    Returns:
        Path to latest checkpoint file, or None if no checkpoints exist.
    """
    cdir = Path(checkpoint_dir)
    if not cdir.exists():
        return None

    latest_symlink = cdir / "latest.pt"
    if latest_symlink.exists():
        return latest_symlink

    # Search for ckpt_epoch*.pt files
    ckpts = list(cdir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        return None

    def extract_epoch(p: Path) -> int:
        match = re.search(r"ckpt_epoch(\d+)", p.name)
        return int(match.group(1)) if match else -1

    ckpts.sort(key=extract_epoch)
    return ckpts[-1] if ckpts else None


def prune_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last: int = 3,
    prefix: str = "ckpt_epoch",
) -> List[Path]:
    """Removes older checkpoints, keeping only the most recent N epoch checkpoints.

    Never deletes latest.pt or best_loss.pt.

    Args:
        checkpoint_dir: Directory containing saved checkpoints.
        keep_last: Number of most recent epoch checkpoints to preserve.
        prefix: Filename prefix for epoch checkpoints.

    Returns:
        List of paths that were removed.
    """
    cdir = Path(checkpoint_dir)
    if not cdir.exists() or keep_last <= 0:
        return []

    ckpts = list(cdir.glob(f"{prefix}*.pt"))

    def extract_epoch(p: Path) -> int:
        match = re.search(r"(\d+)", p.name)
        return int(match.group(1)) if match else -1

    ckpts.sort(key=extract_epoch)

    removed: List[Path] = []
    if len(ckpts) > keep_last:
        to_delete = ckpts[:-keep_last]
        for p in to_delete:
            try:
                p.unlink()
                removed.append(p)
            except OSError:
                pass

    return removed
