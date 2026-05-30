"""
2D Rotary Position Embeddings (RoPE) for grid-structured data.

Attention itself has no notion of *where* a token is. RoPE injects position by
rotating each query/key vector by an angle that depends on its position; the
dot product between two rotated vectors then encodes their relative distance.

For a 2D grid we encode row and column separately: the first half of the
features carries the row position, the second half the column. So two cells in
the same row share the same row-encoding regardless of column, which gives
attention a clean signal for spatial relationships.

Rotation preserves vector length, so this never scales activations up or down.

Reference: RoFormer (Su et al., 2021) — https://arxiv.org/abs/2104.09864
"""
import torch
import torch.nn as nn


class RotaryPositionEmbedding2D(nn.Module):
    """
    Apply 2D rotary position encoding to a (B, L, dim) tensor.

    Args:
        dim       : feature width. Must be divisible by 4 — split in half (row /
                    col), and each half is rotated in pairs, so dim/4 pairs/axis.
        grid_size : side length of the square grid; L must equal grid_size ** 2.
        base      : frequency base (10000 is the standard RoPE value). Lower
                    rotates faster (local detail), higher slower (global).
    """

    def __init__(self, dim: int, grid_size: int, base: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4 for 2D RoPE (got {dim})"
        self.dim_half = dim // 2  # features per axis (row / col)
        self.grid_size = grid_size

        # one rotation frequency per feature pair, shared by both axes
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_half, 2).float() / self.dim_half))

        # row/col index of every flattened position (row-major: i -> i//gs, i%gs)
        positions = torch.arange(grid_size * grid_size)
        rows = (positions // grid_size).float()
        cols = (positions % grid_size).float()

        # angle = position * frequency, then cache its cos/sin as buffers so they
        # move to the right device automatically with model.to(device)
        row_angles = rows.unsqueeze(1) * inv_freq.unsqueeze(0)  # (L, dim_half//2)
        col_angles = cols.unsqueeze(1) * inv_freq.unsqueeze(0)
        self.register_buffer("row_cos", row_angles.cos())
        self.register_buffer("row_sin", row_angles.sin())
        self.register_buffer("col_cos", col_angles.cos())
        self.register_buffer("col_sin", col_angles.sin())

    def _rotate(self, x, cos, sin):
        """Rotate consecutive feature pairs (a, b) of x by the cached angle."""
        a = x[..., 0::2]  # even features
        b = x[..., 1::2]  # odd features
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
        # interleave back into the original [a0, b0, a1, b1, ...] layout
        return torch.stack([rotated_a, rotated_b], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim) -> (B, L, dim), same shape, positions encoded."""
        x_row = self._rotate(x[..., : self.dim_half], self.row_cos, self.row_sin)
        x_col = self._rotate(x[..., self.dim_half :], self.col_cos, self.col_sin)
        return torch.cat([x_row, x_col], dim=-1)


def _verify():
    """Self-test: shape is unchanged and vector lengths are preserved."""
    rope = RotaryPositionEmbedding2D(dim=128, grid_size=26)
    x = torch.randn(2, 676, 128)
    y = rope(x)
    assert y.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)
    print("_verify passed")


if __name__ == "__main__":
    _verify()
