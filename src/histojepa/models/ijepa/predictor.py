"""Narrow Vision Transformer Predictor for I-JEPA.

Ref: Assran et al., I-JEPA (papers/ijepa/ijepa.md), §3 "Prediction",
Appendix A.1 "Architectures", Appendix C "Predictor depth / Predictor width".
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from histojepa.models.ijepa.masking import apply_masks


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int]) -> torch.Tensor:
    """Computes standard 2D sine-cosine positional embeddings for a 2D patch grid.

    Args:
        embed_dim: Positional embedding dimension (must be divisible by 4).
        grid_size: (H_patches, W_patches) grid dimensions.

    Returns:
        Tensor of shape (1, H_patches * W_patches, embed_dim).
    """
    H_g, W_g = grid_size
    grid_h = np.arange(H_g, dtype=np.float32)
    grid_w = np.arange(W_g, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # W first, then H
    grid = np.stack(grid, axis=0)  # (2, H, W)
    grid = grid.reshape([2, 1, H_g, W_g])

    # 1D sincos for each axis
    dim_per_axis = embed_dim // 2

    def get_1d_sincos(pos, dim):
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / (10000**omega)  # (dim/2,)
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, dim/2)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, dim)

    emb_h = get_1d_sincos(grid[1], dim_per_axis)  # (H*W, dim_per_axis)
    emb_w = get_1d_sincos(grid[0], dim_per_axis)  # (H*W, dim_per_axis)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, embed_dim)
    return torch.from_numpy(pos_embed).float().unsqueeze(0)


class PredictorBlock(nn.Module):
    """Transformer block with pre-normalization and scaled dot-product attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm Self-Attention
        residual = x
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape

        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        x = residual + self.proj(attn_out)

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class IJepaPredictor(nn.Module):
    """Narrow Vision Transformer Predictor.

    Takes representations of the context block and predicts the representations
    of target blocks conditioned on position-encoded mask tokens.

    Args:
        grid_size: Spatial patch grid dimensions (H_patches, W_patches), e.g. (80, 80).
        encoder_dim: Hidden dimension of backbone ViT (default: 768 for ViT-B).
        embed_dim: Predictor bottleneck dimension (default: 384, as in App. A.1).
        depth: Number of transformer blocks (default: 6 for ViT-B).
        num_heads: Number of attention heads (default: 12).
        mlp_ratio: MLP channel expansion ratio (default: 4.0).
    """

    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]] = (80, 80),
        encoder_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = (int(grid_size[0]), int(grid_size[1]))

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim

        # 1. Linear bottleneck projection from encoder space to predictor space
        self.input_proj = nn.Linear(encoder_dim, embed_dim, bias=True)

        # 2. Shared learnable mask token vector (§3 "Prediction")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 3. 2D Sin-Cos positional embedding buffer for the full patch grid
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        # 4. Predictor transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # 5. Output projection back to backbone representation space
        self.output_proj = nn.Linear(embed_dim, encoder_dim, bias=True)

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        ctx_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts representations for a single target block given context representations.

        Args:
            ctx_tokens: Context encoder output tokens (B, N_c, encoder_dim).
            ctx_idx: Context patch indices (B, N_c).
            tgt_idx: Target block patch indices to predict (B, N_t).

        Returns:
            Predicted target block representations (B, N_t, encoder_dim).
        """
        B, N_c, _ = ctx_tokens.shape
        _, N_t = tgt_idx.shape

        # Expand positional embeddings to batch dimension
        pos_embed_batch = self.pos_embed.expand(B, -1, -1)

        # 1. Context tokens: project to predictor bottleneck and add positional encoding
        x = self.input_proj(ctx_tokens)
        ctx_pos = apply_masks(pos_embed_batch, ctx_idx)
        x = x + ctx_pos

        # 2. Target tokens: mask token conditioned on target positional encoding
        m = self.mask_token.expand(B, N_t, -1)
        tgt_pos = apply_masks(pos_embed_batch, tgt_idx)
        m = m + tgt_pos

        # 3. Concatenate [context_tokens, target_mask_tokens]
        z = torch.cat([x, m], dim=1)

        # 4. Forward through predictor transformer blocks
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)

        # 5. Slice only the predicted target tokens and project to encoder dimension
        pred_target_tokens = z[:, N_c:]
        out = self.output_proj(pred_target_tokens)
        return out
