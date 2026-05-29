"""
2D Rotary Position Embeddings (RoPE) for grid-structured data.

Standard 1D RoPE encodes a single sequence position using all `dim` features.
For a 2D occupancy grid, spatial relationships are two-dimensional — a cell's
row and column position are independent pieces of information that should be
encoded separately. A 1D encoding applied to flattened positions treats
"end of row 0" and "start of row 1" as distant, even though they may be
spatially adjacent.

This module encodes row and column positions independently by splitting the
embedding dimension in half:
    - first  half  encodes row position
    - second half  encodes column position

Two tokens in the same row always have identical row-encodings regardless of
column, and vice versa. This gives the attention mechanism a clean signal for
spatial relationships (same row, same column, diagonal neighbours).

Reference: RoFormer — Enhanced Transformer with Rotary Position Embedding
           (Su et al., 2021) https://arxiv.org/abs/2104.09864
"""
import torch
import torch.nn as nn


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D RoPE: encodes (row, col) position by rotating consecutive feature pairs.

    The rotation angle for each feature pair depends on position — tokens at
    similar positions end up with similar orientations, so dot-product attention
    between two tokens implicitly encodes their relative spatial distance.

    Rotation is an isometry: it changes direction but not magnitude, so this
    encoding never scales activations up or down. That property is important
    for stability across the n recursion steps in TRMBlock.

    Args:
        dim       : total embedding dimension. Must be divisible by 4 because:
                    dim is split into two halves (one per axis), and each half
                    must further split into pairs for the rotation — so dim/4
                    pairs per axis, requiring dim % 4 == 0.
        grid_size : side length of the square grid (e.g. 26 for a 26×26 grid).
                    seq_len in forward must equal grid_size ** 2.
        base      : frequency base for the inverse frequency formula.
                    10000.0 is the standard value from the original RoPE paper.
                    Lower values rotate faster (more sensitive to local position);
                    higher values rotate slower (more sensitive to global position).
    """

    def __init__(self, dim: int, grid_size: int, base: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, (
            f"dim must be divisible by 4 for 2D RoPE "
            f"(got dim={dim}). Each axis needs dim//2 features, "
            f"and each axis needs pairs, so dim//4 pairs per axis."
        )

        self.dim_half  = dim // 2    # features allocated to each spatial axis
        self.grid_size = grid_size

        # --- frequency bases (same formula as 1D RoPE, applied per axis) ---
        # inv_freq_i = 1 / (base ^ (2i / half_dim))  for i = 0, 1, ..., half_dim/2 - 1
        # Lower i  -> lower frequency -> slow rotation -> encodes coarse position
        # Higher i -> higher frequency -> fast rotation -> encodes fine position
        # Shape: (half_dim // 2,)  — one frequency per feature pair
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim_half, 2).float() / self.dim_half)
        )
        self.register_buffer('inv_freq', inv_freq)

        # --- precompute (row, col) for every flattened grid position ---
        # Flattening is row-major: position i -> row = i // grid_size,
        #                                         col = i  % grid_size
        positions = torch.arange(grid_size * grid_size)       # (L,)
        rows = (positions // grid_size).float()               # (L,)
        cols = (positions  % grid_size).float()               # (L,)

        # Outer product: each position × each frequency -> angle matrix
        # Shape: (L, half_dim // 2)
        # Same construction as your 1D version:
        #     angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        row_angles = rows.unsqueeze(1) * inv_freq.unsqueeze(0)
        col_angles = cols.unsqueeze(1) * inv_freq.unsqueeze(0)

        # Register as buffers so they automatically move to the correct device
        # when you call model.to(device) — no manual .to(device) needed.
        self.register_buffer('row_cos', row_angles.cos())   # (L, half_dim//2)
        self.register_buffer('row_sin', row_angles.sin())   # (L, half_dim//2)
        self.register_buffer('col_cos', col_angles.cos())   # (L, half_dim//2)
        self.register_buffer('col_sin', col_angles.sin())   # (L, half_dim//2)

    def _rotate(
        self,
        x:   torch.Tensor,   # (B, L, half_dim)
        cos: torch.Tensor,   # (L, half_dim // 2)
        sin: torch.Tensor,   # (L, half_dim // 2)
    ) -> torch.Tensor:
        """
        Apply rotary rotation to one axis-half of the embedding.

        Splits x into consecutive pairs using even/odd indexing (identical to
        the 1D implementation), rotates each pair by the angle [cos, -sin; sin, cos],
        then interleaves the results back into the original layout.

        The 2D rotation of a pair (a, b) by angle θ:
            a' = a·cos(θ) − b·sin(θ)
            b' = a·sin(θ) + b·cos(θ)

        cos and sin broadcast over the batch dimension automatically because
        PyTorch aligns trailing dimensions: (L, half_dim//2) broadcasts with
        (B, L, half_dim//2) without any explicit unsqueeze.
        """
        x1 = x[..., 0::2]   # even-indexed features  (B, L, half_dim//2)
        x2 = x[..., 1::2]   # odd-indexed features   (B, L, half_dim//2)

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # stack along a new last dim -> (B, L, half_dim//2, 2)
        # flatten the last two dims  -> (B, L, half_dim)
        # this interleaves pairs back: [r1, r2, r3, r4, ...]
        return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D rotary position encoding to x.

        Args:
            x : (B, L, dim)  where L == grid_size ** 2.

        Returns:
            Tensor of identical shape (B, L, dim) with positional information
            encoded via rotation. Norms are preserved: ‖forward(x)[b,l]‖ == ‖x[b,l]‖.

        How the split works:
            x = [--- row half (dim//2) --- | --- col half (dim//2) ---]
            Row half is rotated using row position angles.
            Col half is rotated using column position angles.
            The two halves are concatenated to recover the original shape.
        """
        x_row = x[..., :self.dim_half]    # (B, L, dim//2)
        x_col = x[..., self.dim_half:]    # (B, L, dim//2)

        x_row = self._rotate(x_row, self.row_cos, self.row_sin)
        x_col = self._rotate(x_col, self.col_cos, self.col_sin)

        return torch.cat([x_row, x_col], dim=-1)   # (B, L, dim)


def _verify():
    """
    Two checks:
        1. Output shape matches input shape.
        2. Per-token norms are preserved (rotation is an isometry).
    """
    rope = RotaryPositionEmbedding2D(dim=128, grid_size=26)
    x = torch.randn(2, 676, 128)
    y = rope(x)

    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert torch.allclose(
        x.norm(dim=-1), y.norm(dim=-1), atol=1e-4
    ), "norms not preserved — rotation should be an isometry"

    print(f"input  shape : {x.shape}")
    print(f"output shape : {y.shape}")
    print(f"norm preserved: {torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)}")
    print("_verify passed")


if __name__ == "__main__":
    _verify()