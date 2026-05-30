"""Core building blocks: RMSNorm and SwiGLU."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization: x / sqrt(mean(x^2)) * gamma. No mean centering."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # to prevent division by zero
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the root mean square of feature dimensions of the input tensor
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # dim: (batch_size, seq_len, feature_dim) -> (batch_size, seq_len, 1)
        # Normalize the input tensor and apply the learnable scaling factor
        return x / rms * self.weight  # dim: (batch_size, seq_len, feature_dim)


class SwiGLU(nn.Module):
    """Gated activation: w3(silu(w1(x)) * w2(x)), where w1, w2, w3 are linear layers."""

    def __init__(self, dim_in: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim_in * 4  # default to 4x expansion
        self.w1 = nn.Linear(
            dim_in, hidden_dim, bias=False
        )  # linear layer for the first part of the gate
        self.w2 = nn.Linear(
            dim_in, hidden_dim, bias=False
        )  # linear layer for the second part of the gate
        self.w3 = nn.Linear(
            hidden_dim, dim_in, bias=False
        )  # linear layer for the output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the gated activation using the SwiGLU formula
        return self.w3(
            F.silu(self.w1(x)) * self.w2(x)
        )  # dim: (batch_size, seq_len, feature_dim)
