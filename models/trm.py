"""
Tiny Recursive Model (TRM) for per-cell action prediction on occupancy grids.

The model keeps three tensors, all shaped (B, L, dim):
    x : the embedded input grid — fixed, every step sees it
    y : the answer (decoded into action logits at each supervision step)
    z : a reasoning scratchpad (recursed many times, never decoded directly)

One supervision step:
    1. recurse z for n steps (the "thinking"), holding y fixed
    2. update y once from the freshly-recursed z (the "answer")
    3. decode y into per-cell action logits

The same small stack of TRMLayers does all the work (weight sharing), which is
why a 2-layer model gets the effective depth of a much deeper one.

Building blocks live in separate files: RMSNorm/SwiGLU in layers.py,
SelfAttention in attention.py, 2D RoPE in embeddings.py.
"""

import torch
import torch.nn as nn

from dataset.loader import INPUT_VOCAB_SIZE  # 4 tokens: free / obstacle / start / goal
from models.attention import SelfAttention
from models.layers import RMSNorm, SwiGLU


class TRMLayer(nn.Module):
    """
    One transformer layer of the recursive unit (pre-norm).

        x -> RMSNorm -> SelfAttention -> + x
          -> RMSNorm -> SwiGLU FFN    -> + x

    Pre-norm (normalise before each sublayer) plus residual connections keep
    gradients stable, which matters here because these weights are reused many
    times per forward pass.

    Args:
        dim       : input/output width (3*model_dim for the TRM's [x,y,z] tensor)
        grid_size : forwarded to SelfAttention -> 2D RoPE
        n_heads   : attention heads
        ffn_mult  : SwiGLU hidden width = dim * ffn_mult
    """

    def __init__(self, dim: int, grid_size: int, n_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mixer = SelfAttention(dim, grid_size=grid_size, n_heads=n_heads)
        self.ffn = SwiGLU(dim, hidden_dim=dim * ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyRecursiveModel(nn.Module):
    """
    The full TRM.

    x, y and z are concatenated (not added) so the layers can keep the three
    signals separate and learn how to mix them. y and z both start at zero — the
    model builds the answer up from a neutral state through recursion.

    forward() returns a dict:
        logits       : list of T tensors (B, L, num_classes) — one per step
        halt_probs   : list of T tensors (B, 1) in [0,1] — empty if halting=False
        final_logits : (B, L, num_classes) — the last step, used at inference

    Args:
        dim        : model hidden width
        num_heads  : attention heads per TRMLayer
        num_layers : layers in the recursive unit (2 in the paper)
        num_classes: output actions (5: N / S / E / W / stay)
        grid_size  : side length of the square grid
        T          : number of supervision steps (loss is computed at each)
        n          : reasoning recursions per supervision step
        halting    : whether to add the "am I done?" halt head
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 5,
        grid_size: int = 26,
        T: int = 3,
        n: int = 6,
        halting: bool = True,
    ):
        super().__init__()
        self.T = T
        self.n = n
        self.halting = halting

        self.embed = nn.Embedding(INPUT_VOCAB_SIZE, dim)  # token -> vector

        # the shared recursive unit operates on the [x, y, z] tensor (3*dim wide)
        combined_dim = dim * 3
        self.layers = nn.ModuleList(
            [
                TRMLayer(combined_dim, grid_size=grid_size, n_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

        # project the shared output back into y-space and z-space separately
        self.split_proj_y = nn.Linear(combined_dim, dim, bias=False)
        self.split_proj_z = nn.Linear(combined_dim, dim, bias=False)

        # per-cell action classifier applied to y
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes, bias=False)

        # halt head: pool y over the grid -> one probability per sample
        if halting:
            self.halt_norm = RMSNorm(dim)
            self.halt_head = nn.Linear(dim, 1, bias=False)

    def _shared(self, x, y, z):
        """Run the shared TRMLayers over the concatenated [x, y, z] tensor."""
        combined = torch.cat([x, y, z], dim=-1)  # (B, L, 3*dim)
        for layer in self.layers:
            combined = layer(combined)
        return combined

    def _latent_step(self, x, y, z):
        """Update the reasoning state z (the answer y is held fixed)."""
        return self.split_proj_z(self._shared(x, y, z))

    def _answer_step(self, x, y, z):
        """Update the answer y from the freshly-recursed reasoning state z."""
        return self.split_proj_y(self._shared(x, y, z))

    def forward(self, grid_tokens: torch.Tensor) -> dict:
        """grid_tokens: (B, L) ints in {0=free, 1=obstacle, 2=start, 3=goal}."""
        x = self.embed(grid_tokens)  # (B, L, dim) — fixed input
        y = torch.zeros_like(x)  # answer, built up over the steps
        z = torch.zeros_like(x)  # reasoning scratchpad

        logits_list, halt_list = [], []
        for _ in range(self.T):
            # Think: recurse z n times. Only the LAST recursion needs gradients
            # (the "1-step gradient" trick from the TRM/HRM papers), so the
            # earlier ones run under no_grad to save memory and stay stable.
            with torch.no_grad():
                for _ in range(self.n - 1):
                    z = self._latent_step(x, y, z)
            z = self._latent_step(x, y, z)

            # Answer: update y once from the new reasoning state, then decode it.
            y = self._answer_step(x, y, z)
            logits_list.append(self.head(self.norm(y)))  # (B, L, num_classes)

            if self.halting:
                pooled = self.halt_norm(y).mean(dim=1)  # (B, dim)
                halt_list.append(self.halt_head(pooled))  # (B, 1) — raw logit, sigmoid in loss

            # Cut the graph between steps so each step trains on its own short
            # chain instead of one long T*n chain that would vanish.
            y = y.detach()
            z = z.detach()

        return {
            "logits": logits_list,
            "halt_probs": halt_list,
            "final_logits": logits_list[-1],
        }


def _verify():
    """Quick self-test: shapes are right and the loss can backprop."""
    import torch.nn.functional as F

    model = TinyRecursiveModel(
        dim=128,
        num_heads=8,
        num_layers=2,
        num_classes=5,
        grid_size=26,
        T=3,
        n=6,
        halting=True,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters : {params:,}")

    out = model(torch.randint(0, INPUT_VOCAB_SIZE, (4, 676)))
    assert len(out["logits"]) == 3
    assert out["final_logits"].shape == (4, 676, 5)
    assert out["halt_probs"][0].shape == (4, 1)

    targets = torch.randint(0, 5, (4, 676))
    F.cross_entropy(out["final_logits"].reshape(-1, 5), targets.reshape(-1)).backward()
    print("_verify passed")


if __name__ == "__main__":
    _verify()
