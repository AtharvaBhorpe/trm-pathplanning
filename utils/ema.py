"""
Exponential Moving Average (EMA) of model weights.

During training, EMA maintains a slow-moving copy of every model weight and
buffer (parameters *and* things like RoPE cos/sin tables). Before evaluating,
swap the live weights out for the EMA copy; after, swap them back.

Why bother? SGD/Adam weights bounce around each step. The EMA copy is a
running average that's smoother and usually generalises better — so eval
metrics look better and the saved checkpoint is a stronger final model.

Typical decay: 0.999 (update 0.1 % toward the new weights each step).
"""
import copy

import torch


class EMA:
    """Shadow the full model state_dict with an exponential moving average."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        # deep-copy so the shadow starts identical to the live weights
        self.shadow = copy.deepcopy(model.state_dict())
        self._backup: dict = {}

    @torch.no_grad()
    def update(self):
        """Call once after each optimizer step to blend in the new weights."""
        live = self.model.state_dict()
        for key, shadow_val in self.shadow.items():
            live_val = live[key]
            if live_val.is_floating_point():
                # EMA formula: shadow = decay * shadow + (1 - decay) * live
                shadow_val.mul_(self.decay).add_(live_val, alpha=1.0 - self.decay)
            else:
                # integer buffers (e.g. batch-norm step counters) — just copy
                shadow_val.copy_(live_val)

    def apply_shadow(self):
        """Swap live weights → EMA weights (call before evaluate)."""
        self._backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)

    def restore(self):
        """Swap EMA weights → live weights (call after evaluate)."""
        self.model.load_state_dict(self._backup)
        self._backup = {}
