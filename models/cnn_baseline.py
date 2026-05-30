"""
CNN baseline: a stack of 3x3 conv blocks + a per-cell action head. (Reference point.)

This is intentionally a "reasonable first attempt" CNN, parameter-matched to the
TRM. It is NOT the learning focus — its only job is to be a fair, solid baseline the
TRM must beat (especially on generalization).

Width (`width`) and depth (`num_layers`) are configurable so the model can be
parameter-matched to the TRM and so we can probe the **receptive-field** effect: a
stack of L 3x3 convs sees a (1 + 2L)x(1 + 2L) window, so a shallow CNN (L=4 -> 9x9)
physically cannot see across a 26x26 grid, while a deep one (L=13 -> 27x27) can. Both
the shallow and deep variants are built at ~1.2M params to isolate depth from capacity.
"""
import torch
import torch.nn as nn

from dataset.loader import INPUT_VOCAB_SIZE


class CNNBaseline(nn.Module):
    def __init__(self, dim=64, num_classes=5, grid_size=26, width=96, num_layers=4):
        super().__init__()
        self.grid_size = grid_size
        self.embed = nn.Embedding(INPUT_VOCAB_SIZE, dim)

        # 'same' padding keeps spatial dims -> per-cell prediction stays aligned.
        # All conv blocks share a uniform `width`; the first maps the embedding to it.
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )
        channels = [dim] + [width] * num_layers
        self.net = nn.Sequential(
            *[block(channels[i], channels[i + 1]) for i in range(num_layers)]
        )
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, grid_tokens):
        b = grid_tokens.shape[0]
        s = self.grid_size
        x = self.embed(grid_tokens)                 # (B, L, dim)
        x = x.transpose(1, 2).reshape(b, -1, s, s)  # (B, dim, H, W)
        x = self.net(x)
        logits = self.head(x)                       # (B, num_classes, H, W)
        logits = logits.reshape(b, -1, s * s).transpose(1, 2)  # (B, L, num_classes)
        # mirror the TRM output dict so eval/training code is shared
        return {"final_logits": logits, "logits": [logits], "halt_probs": []}


def build(cfg_model):
    return CNNBaseline(dim=cfg_model.get("dim", 64),
                       num_classes=cfg_model["num_classes"],
                       grid_size=cfg_model.get("grid_size", 26),
                       width=cfg_model.get("width", 96),
                       num_layers=cfg_model.get("num_layers", 4))
