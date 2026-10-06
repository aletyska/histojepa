"""Multi-block masking strategy for I-JEPA pre-training.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), §3 "Targets", "Context"
and Appendix A.1 "Masking".
"""

import math
import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def apply_masks(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather patch tokens from a sequence representation using patch indices.

    Args:
        x: Patch tokens tensor of shape (B, N, D).
        idx: Patch indices to gather of shape (B, K), with values in [0, N-1].

    Returns:
        Gathered tokens tensor of shape (B, K, D).
    """
    if idx.dim() != 2:
        raise ValueError(f"idx must be 2D (B, K), got shape {idx.shape}")
    if x.dim() != 3:
        raise ValueError(f"x must be 3D (B, N, D), got shape {x.shape}")
    B, N, D = x.shape
    expanded_idx = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=expanded_idx)


class MultiBlockMaskCollator:
    """DataLoader collate function that samples context and multi-block target masks.

    Ref: I-JEPA §3 and Appendix A.1:
    - Samples M target blocks with random scale in (0.15, 0.2) and aspect ratio in (0.75, 1.5).
    - Samples 1 context block with random scale in (0.85, 1.0) and unit aspect ratio.
    - Eliminates any regions in the context block that overlap with any of the M target blocks.
    - Guarantees zero overlap between the context encoder input and the target blocks.

    Args:
        grid_size: Spatial grid tuple (H_patches, W_patches), e.g. (80, 80) for 1280x1280.
        num_targets: Number of target blocks M (default: 4).
        target_scale: Range of target block area scale (default: (0.15, 0.2)).
        target_aspect_ratio: Range of target block aspect ratio (default: (0.75, 1.5)).
        context_scale: Range of context block area scale (default: (0.85, 1.0)).
        context_aspect_ratio: Range of context aspect ratio (default: (1.0, 1.0)).
        min_context_keep: Minimum remaining context patches before resampling (default: 10).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]] = (80, 80),
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        context_aspect_ratio: Tuple[float, float] = (1.0, 1.0),
        min_context_keep: int = 10,
        seed: Optional[int] = None,
    ):
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = (int(grid_size[0]), int(grid_size[1]))

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_targets = int(num_targets)
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.min_context_keep = int(min_context_keep)
        self.rng = random.Random(seed)

    def _sample_block_size(
        self,
        scale_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
    ) -> Tuple[int, int]:
        """Samples block height and width in patch units."""
        H_g, W_g = self.grid_size
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        aspect = self.rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1])

        target_area = scale * self.num_patches
        h = max(1, min(H_g, int(round(math.sqrt(target_area * aspect)))))
        w = max(1, min(W_g, int(round(math.sqrt(target_area / aspect)))))
        return h, w

    def _block_indices(self, top: int, left: int, h: int, w: int) -> List[int]:
        """Returns 1D flattened patch indices for a 2D block."""
        W_g = self.grid_size[1]
        indices = []
        for r in range(top, top + h):
            row_offset = r * W_g
            for c in range(left, left + w):
                indices.append(row_offset + c)
        return indices

    def sample_masks_for_image(
        self,
        target_hw: Tuple[int, int],
        context_hw: Tuple[int, int],
    ) -> Tuple[List[int], List[List[int]]]:
        """Samples masks for a single image, ensuring no overlap between context and targets.

        Returns:
            Tuple of (context_indices, list_of_M_target_indices).
        """
        H_g, W_g = self.grid_size
        h_t, w_t = target_hw
        h_c, w_c = context_hw

        max_attempts = 25
        for _ in range(max_attempts):
            # Sample M target blocks
            target_blocks: List[List[int]] = []
            target_union = set()
            for _ in range(self.num_targets):
                top = self.rng.randint(0, H_g - h_t)
                left = self.rng.randint(0, W_g - w_t)
                block_idx = self._block_indices(top, left, h_t, w_t)
                target_blocks.append(block_idx)
                target_union.update(block_idx)

            # Sample 1 context block
            ctx_top = self.rng.randint(0, H_g - h_c)
            ctx_left = self.rng.randint(0, W_g - w_c)
            ctx_candidate = self._block_indices(ctx_top, ctx_left, h_c, w_c)

            # Strictly eliminate any context patches that overlap with ANY target block
            remaining_context = [idx for idx in ctx_candidate if idx not in target_union]

            if len(remaining_context) >= self.min_context_keep:
                return remaining_context, target_blocks

        # Fallback if attempts exceeded: take whatever context patches complement target union
        complement = [idx for idx in range(self.num_patches) if idx not in target_union]
        if len(complement) < self.min_context_keep:
            # Degenerate edge case: target union covered almost everything; truncate targets
            complement = list(range(self.min_context_keep))
        return complement, target_blocks

    def __call__(
        self, batch: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Collate function for PyTorch DataLoader.

        Args:
            batch: List of image tensors each of shape (3, H, W).

        Returns:
            Tuple containing:
                - images: Batched tensor (B, 3, H, W).
                - context_idx: Long tensor (B, N_c) of sorted patch indices for context encoder.
                - target_idx_list: List of M Long tensors, each (B, N_t).
        """
        images = torch.stack(list(batch), dim=0)
        B = images.shape[0]

        # In accordance with App. A.1: block dimensions are uniform across the batch
        target_hw = self._sample_block_size(self.target_scale, self.target_aspect_ratio)
        context_hw = self._sample_block_size(self.context_scale, self.context_aspect_ratio)

        batch_contexts: List[List[int]] = []
        batch_targets: List[List[List[int]]] = []

        for _ in range(B):
            ctx_idx, tgt_blocks = self.sample_masks_for_image(target_hw, context_hw)
            batch_contexts.append(ctx_idx)
            batch_targets.append(tgt_blocks)

        # To enable uniform dense batch tensor operations, subsample context to batch minimum
        min_ctx_len = min(len(c) for c in batch_contexts)
        aligned_contexts = []
        for c in batch_contexts:
            if len(c) > min_ctx_len:
                sampled = sorted(self.rng.sample(c, min_ctx_len))
            else:
                sampled = sorted(c)
            aligned_contexts.append(sampled)

        context_idx_tensor = torch.tensor(aligned_contexts, dtype=torch.long)

        # Structure M target index tensors, each (B, N_t)
        target_idx_tensors: List[torch.Tensor] = []
        for m in range(self.num_targets):
            m_targets = [batch_targets[b][m] for b in range(B)]
            target_idx_tensors.append(torch.tensor(m_targets, dtype=torch.long))

        return images, context_idx_tensor, target_idx_tensors
