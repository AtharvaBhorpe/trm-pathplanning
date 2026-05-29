"""
Tiny Recursive Model (TRM) for per-cell action classification on occupancy grids.

Module contents (in dependency order):
    SelfAttention        — multi-head self-attention with 2D RoPE on Q and K
    TRMLayer             — pre-norm block: SelfAttention + SwiGLU FFN
    TinyRecursiveModel   — full TRM with T supervision steps and n recursions

Recursion data flow (one step):
    combined = cat([x, y, z], dim=-1)   # (B, L, 3*dim)
    combined = TRMLayer₁(combined)
    combined = TRMLayer₂(combined)
    y        = split_proj_y(combined)   # (B, L, dim)
    z        = split_proj_z(combined)   # (B, L, dim)

x is the fixed embedded input grid (never updated).
y is the solution state (decoded into logits at each supervision step).
z is the reasoning scratchpad (propagated but never decoded directly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import RMSNorm, SwiGLU
from models.embeddings import RotaryPositionEmbedding2D
from dataset.loader import INPUT_VOCAB_SIZE   # 4 token types: free / obstacle / start / goal


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with 2D Rotary Position Embedding.

    Built on your 1D SelfAttention implementation with two changes:

    - RotaryPositionEmbedding2D
       Encodes (row, col) independently, so tokens in the same row always have 
       identical row-encodings regardless of column. For grid path planning 
       this matters: "same column, adjacent rows" is a key spatial relationship.

       RoPE is applied to Q and K after head-splitting (standard practice).
       Each head independently rotates its head_dim-sized vectors.

    - F.scaled_dot_product_attention
       Equivalent to manual (q @ k.T) * scale + softmax formulation but
       numerically stable and dispatches to flash attention when available.

    Args:
        dim       : input/output dimension. For TRM this is 3*model_dim because
                    the combined [x,y,z] tensor is what gets attended to.
        grid_size : side length of the square grid; passed to 2D RoPE.
        n_heads   : attention heads. (dim // n_heads) must be divisible by 4
                    to satisfy the 2D RoPE requirement (pairs per axis).
    """
    def __init__(self, dim: int, grid_size: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, \
            f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim % 4 == 0, (
            f"head_dim ({self.head_dim}) must be divisible by 4 for 2D RoPE. "
            f"Current: dim={dim}, n_heads={n_heads} → head_dim={self.head_dim}. "
            f"Adjust dim or n_heads so dim // n_heads is divisible by 4."
        )

        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)  # project to concatenated QKV and split later
        self.out_proj = nn.Linear(dim,     dim, bias=False)  # project back to input dimension after attention

        # 2D RoPE at head_dim size — each head independently encodes position
        self.rope = RotaryPositionEmbedding2D(self.head_dim, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim) → (B, L, dim)"""
        B, L, D = x.shape

        # project and split heads 
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)  # (B, L, 3*dim) → (B, L, 3, n_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)      # each: (B, L, n_heads, head_dim)
        q = q.transpose(1, 2)            # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply 2D RoPE to Q and K
        # reshape to (B*n_heads, L, head_dim) so rope treats n_heads as batch,
        # then reshape back to (B, n_heads, L, head_dim) after rope rotates the vectors
        q = self.rope(q.reshape(B * self.n_heads, L, self.head_dim))  # (B*n_heads, L, head_dim) → (B*n_heads, L, head_dim)
        k = self.rope(k.reshape(B * self.n_heads, L, self.head_dim))
        q = q.reshape(B, self.n_heads, L, self.head_dim)  # (B*n_heads, L, head_dim) → (B, n_heads, L, head_dim)
        k = k.reshape(B, self.n_heads, L, self.head_dim)

        # scaled dot-product attention (handles scale + softmax internally,
        # dispatches to flash attention on compatible hardware)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)       # (B, n_heads, L, head_dim) → (B, L, dim)
        return self.out_proj(out)  # (B, L, dim) → (B, L, dim) through output projection


class TRMLayer(nn.Module):
    """
    One layer of the recursive unit.

    Pre-norm architecture:
        x → RMSNorm → SelfAttention → + x    (residual)
          → RMSNorm → SwiGLU FFN    → + x    (residual)

    Pre-norm (normalise BEFORE the sublayer) keeps gradient magnitudes stable
    across the deep recursion chain (T × n × num_layers forward passes share
    the same weights). The residuals let information flow unchanged when the
    learned update is near zero at initialisation.

    Args:
        dim       : input/output dimension (3*model_dim for the TRM combined tensor)
        grid_size : passed through to SelfAttention → 2D RoPE
        n_heads   : attention heads
        ffn_mult  : SwiGLU hidden dim = dim * ffn_mult
    """
    def __init__(self, dim: int, grid_size: int, n_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mixer = SelfAttention(dim, grid_size=grid_size, n_heads=n_heads)
        self.ffn   = SwiGLU(dim, hidden_dim=dim * ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(  self.norm2(x))
        return x


class TinyRecursiveModel(nn.Module):
    """
    Full Tiny Recursive Model for per-cell path-planning action classification.

    Why concatenate [x, y, z] instead of adding them?
        x, y, z are three different information streams at the same dimension.
        Concatenation preserves all three signals independently and lets the
        layers learn arbitrary mixing. Addition would conflate them and force
        them to stay in the same representation space.

    Why zeros for y and z initialisation?
        The TRM paper shows that starting from a neutral state and building up
        the solution through recursion is what gives the model its generalisation
        properties. A learned initial state would "pre-load" a guess before
        seeing the input, which undermines the recursive reasoning mechanism.

    Why T supervision steps?
        Without intermediate losses, gradients must flow backward through all
        T×n recursion steps — a very long chain that vanishes quickly.
        Computing the cross-entropy loss at each step creates gradient shortcuts:
        every step's loss gradient flows directly back to the shared weights,
        which keeps training stable even at n=6 recursions per step.
        (See compute_trm_loss in pretrain.py for the full loss implementation.)

    Output dict keys match what compute_trm_loss and evaluate() in pretrain.py
    expect:
        logits       : list of T (B, L, num_classes) tensors
        halt_probs   : list of T (B, 1) tensors in [0,1]  — empty if halting=False
        final_logits : (B, L, num_classes) — last supervision step (used at inference)

    Args:
        dim        : model hidden dimension
        num_heads  : attention heads inside each TRMLayer
        num_layers : layers in the recursive unit (2 in the TRM paper)
        num_classes: output action classes (5: N / S / E / W / stay)
        grid_size  : side length of the square grid (26 for this project)
        T          : number of supervision steps (deep supervision)
        n          : recursion iterations per supervision step
        halting    : whether to train the halt head (disable for sanity runs)
    """
    def __init__(
        self,
        dim:         int  = 128,
        num_heads:   int  = 8,
        num_layers:  int  = 2,
        num_classes: int  = 5,
        grid_size:   int  = 26,
        T:           int  = 3,
        n:           int  = 6,
        halting:     bool = True,
    ):
        super().__init__()
        self.T       = T
        self.n       = n
        self.halting = halting

        # embed 4 input token types into the model dimension
        self.embed = nn.Embedding(INPUT_VOCAB_SIZE, dim)

        # recursive unit: num_layers stacked TRMLayers
        # they operate on the combined [x,y,z] tensor → 3*dim wide
        combined_dim = dim * 3
        self.layers = nn.ModuleList([
            TRMLayer(combined_dim, grid_size=grid_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

        # project combined representation back into y-space and z-space separately
        # two independent projections so y (solution) and z (reasoning) can specialise
        self.split_proj_y = nn.Linear(combined_dim, dim, bias=False)
        self.split_proj_z = nn.Linear(combined_dim, dim, bias=False)

        # per-cell classification head applied to y at each supervision step
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes, bias=False)

        # halting head: pool y over the sequence, project to a scalar probability
        # (B, L, dim) → mean over L → (B, dim) → (B, 1) → sigmoid → [0,1]
        if halting:
            self.halt_norm = RMSNorm(dim)
            self.halt_head = nn.Linear(dim, 1, bias=False)

    def _recurse(
        self,
        x: torch.Tensor,   # (B, L, dim) — fixed input, never updated
        y: torch.Tensor,   # (B, L, dim) — solution state
        z: torch.Tensor,   # (B, L, dim) — reasoning state
    ):
        """One recursion step — update y and z given the fixed input x."""
        combined = torch.cat([x, y, z], dim=-1)   # (B, L, 3*dim)
        for layer in self.layers:
            combined = layer(combined)             # (B, L, 3*dim)
        y_new = self.split_proj_y(combined)        # (B, L, dim)
        z_new = self.split_proj_z(combined)        # (B, L, dim)
        return y_new, z_new

    def forward(self, grid_tokens: torch.Tensor) -> dict:
        """
        Args:
            grid_tokens: LongTensor (B, L)  values in {0=free, 1=obstacle, 2=start, 3=goal}

        Returns:
            dict — see class docstring for key descriptions.
        """
        # embed input tokens → fixed signal x that every recursion sees
        x = self.embed(grid_tokens)    # (B, L, dim)

        # y and z start at zero — neutral initial state per TRM paper
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)

        logits_list = []
        halt_list   = []

        for _ in range(self.T):

            # --- n recursion iterations ---
            for _ in range(self.n):
                y, z = self._recurse(x, y, z)

            # --- decode y at this supervision step ---
            logits_t = self.head(self.norm(y))            # (B, L, num_classes)
            logits_list.append(logits_t)

            if self.halting:
                pooled = self.halt_norm(y).mean(dim=1)    # (B, dim)
                halt_t = torch.sigmoid(self.halt_head(pooled))  # (B, 1)
                halt_list.append(halt_t)

        return {
            "logits":       logits_list,
            "halt_probs":   halt_list,
            "final_logits": logits_list[-1],
        }


def _verify():
    """
    Sanity checks before training:
        1. output shapes match pretrain.py expectations
        2. parameter count is under 20M
        3. halt probabilities are in [0, 1]
        4. gradient flows back from the loss (no broken graph)
    """
    model = TinyRecursiveModel(
        dim=128, num_heads=8, num_layers=2, num_classes=5,
        grid_size=26, T=3, n=6, halting=True,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters : {params:,}")
    assert params < 20_000_000, f"too many parameters: {params:,}"

    dummy = torch.randint(0, INPUT_VOCAB_SIZE, (4, 676))
    out   = model(dummy)

    assert len(out["logits"])        == 3,           f"expected T=3 steps, got {len(out['logits'])}"
    assert out["logits"][0].shape    == (4, 676, 5), f"wrong logits shape: {out['logits'][0].shape}"
    assert out["final_logits"].shape == (4, 676, 5)
    assert len(out["halt_probs"])    == 3
    assert out["halt_probs"][0].shape == (4, 1),     f"wrong halt shape: {out['halt_probs'][0].shape}"

    lo = out["halt_probs"][0].min().item()
    hi = out["halt_probs"][0].max().item()
    assert 0.0 <= lo and hi <= 1.0,  f"halt probs out of range: [{lo:.3f}, {hi:.3f}]"

    # gradient check — loss should backprop without errors
    import torch.nn.functional as F
    targets = torch.randint(0, 5, (4, 676))
    loss    = F.cross_entropy(out["final_logits"].reshape(-1, 5), targets.reshape(-1))
    loss.backward()
    print("gradient check passed")

    print(f"logits per step  : {out['logits'][0].shape}")
    print(f"halt_probs       : {out['halt_probs'][0].shape}  range [{lo:.3f}, {hi:.3f}]")
    print("_verify passed")


if __name__ == "__main__":
    _verify()