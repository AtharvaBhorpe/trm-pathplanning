"""Exponential Moving Average of model parameters (used for evaluation)."""
import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}
