"""
CNN baseline: 3 conv blocks + per-cell action head.  (Reference point, mostly complete.)

This is intentionally a "reasonable first attempt" CNN, parameter-matched to the
TRM. It is NOT the learning focus, so it is provided working — its only job is to
be a fair, solid baseline the TRM must beat (especially on generalization).
"""
import torch
import torch.nn as nn

from dataset.loader import INPUT_VOCAB_SIZE


class CNNBaseline(nn.Module):
    def __init__(self, dim=64, num_classes=5, grid_size=26):
        super().__init__()
        self.grid_size = grid_size
        self.embed = nn.Embedding(INPUT_VOCAB_SIZE, dim)
        # 'same' padding keeps spatial dims -> per-cell prediction stays aligned
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )
        self.net = nn.Sequential(
            block(dim, 96), block(96, 128), block(128, 128), block(128, 96),
        )
        self.head = nn.Conv2d(96, num_classes, 1)

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
    return CNNBaseline(dim=64,
                       num_classes=cfg_model["num_classes"],
                       grid_size=cfg_model.get("grid_size", 26))
