"""Presentation-attack (screen / print recapture) classifier over NATIVE-resolution patches.

Design points that matter more than the architecture:
  * NO resizing. Moire, sub-pixel lattices and halftone rosettes live at near-Nyquist frequencies
    and are destroyed by down-scaling; the network sees raw pixels of a small patch.
  * First layer is a full-resolution 3x3 conv (it can become a high-pass / lattice detector),
    then a stride-2 pyramid. Plain dense convs + BN + ReLU6: INT8-, GPU- and NPU-friendly, and
    ~10x faster to train on CPU than a depthwise-separable variant (measured).
  * Image-level decision = mean logit over several patches (the app samples K patches from the
    plate / vehicle region), which also makes it robust to a patch that happens to be flat.
~0.24M parameters, ~80 MMACs per 96x96 patch.
"""

from __future__ import annotations

import torch
from torch import nn

from training.models import _cbr


class PadNet(nn.Module):
    def __init__(self, width: float = 1.0):
        super().__init__()
        c = lambda n: max(8, int(round(n * width / 8)) * 8)  # noqa: E731
        self.net = nn.Sequential(
            _cbr(3, c(16), 3, 1),
            _cbr(c(16), c(32), 3, 2), _cbr(c(32), c(32), 3, 1),
            _cbr(c(32), c(64), 3, 2), _cbr(c(64), c(64), 3, 1),
            _cbr(c(64), c(96), 3, 2),
            _cbr(c(96), c(128), 3, 2),
        )
        self.head = nn.Linear(c(128), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,P,P] in 0..1  ->  logit [B] (>0 = attack: screen or print)."""
        x = (x - 0.5) / 0.5
        return self.head(self.net(x).mean(dim=(2, 3))).squeeze(1)
