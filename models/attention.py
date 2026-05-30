"""
Multi-head self-attention with 2D rotary position embeddings.

A standalone, reusable block (like RMSNorm / SwiGLU in layers.py). It lets every
cell in the grid look at every other cell, while 2D RoPE tells it *where* each
cell sits (its row and column) so spatial relationships are learnable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import RotaryPositionEmbedding2D


class SelfAttention(nn.Module):
    """
    Self-attention over a sequence of grid cells.

    Steps in forward():
        1. project the input into queries, keys and values (one Linear, split 3 ways)
        2. split into n_heads so each head attends in its own subspace
        3. add positional info to Q and K with 2D RoPE
        4. let each cell attend to all cells (scaled_dot_product_attention)
        5. merge the heads back and project to the output

    Args:
        dim       : input/output width. In the TRM this is 3*model_dim (the
                    concatenated [x, y, z] tensor).
        grid_size : side length of the square grid, forwarded to 2D RoPE.
        n_heads   : number of attention heads. dim // n_heads must be divisible
                    by 4 (a requirement of the 2D RoPE pairing).
    """

    def __init__(self, dim: int, grid_size: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim % 4 == 0, (
            f"head_dim ({self.head_dim}) must be divisible by 4 for 2D RoPE "
            f"(dim={dim}, n_heads={n_heads})."
        )

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)   # produces Q, K, V at once
        self.out_proj = nn.Linear(dim, dim, bias=False)  # final output projection
        self.rope = RotaryPositionEmbedding2D(self.head_dim, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim) -> (B, L, dim)"""
        B, L, D = x.shape

        # project to Q, K, V and split into heads -> each (B, n_heads, L, head_dim)
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        # add positional info to Q and K (RoPE treats n_heads as extra batch)
        q = self.rope(q.reshape(B * self.n_heads, L, self.head_dim)).reshape(B, self.n_heads, L, self.head_dim)
        k = self.rope(k.reshape(B * self.n_heads, L, self.head_dim)).reshape(B, self.n_heads, L, self.head_dim)

        # attention (handles the scale + softmax internally)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)      # merge heads -> (B, L, dim)
        return self.out_proj(out)
