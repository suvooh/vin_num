"""Line recogniser: fully convolutional, CTC output, built to quantise to INT8 for a phone NPU.

Design choices (each is about deployment, not fashion):
  * conv + BatchNorm + ReLU6 only - no LSTM / attention / hard-swish: those are the layers that
    lose accuracy under INT8 or are unsupported on mobile NPUs. BN folds into the convs.
  * depthwise-separable inverted residuals: ~0.7M parameters, tens of MFLOPs per line.
  * height is collapsed with a real conv (keeps vertical structure), then a stack of dilated
    1-D depthwise convs gives ~30 frames of horizontal context - enough to use neighbours when
    a character is smudged - without recurrent state.
  * time resolution = W/4 so narrow glyphs (1, I) and gaps stay separable; any W works, so one
    network serves plate rows (48x256) and VIN lines (48x512).

forward: [B,1,48,W] in 0..1  ->  logits [B, T=W/4, V=38].
"""

from __future__ import annotations

import torch
from torch import nn

from vinplate.vocab import VOCAB_SIZE


def _cbr(i, o, k, s=1, p=None, g=1):
    p = k // 2 if p is None else p
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, groups=g, bias=False), nn.BatchNorm2d(o), nn.ReLU6(inplace=True))


class InvRes(nn.Module):
    def __init__(self, i, o, stride=(1, 1), expand=3):
        super().__init__()
        m = i * expand
        self.use_res = stride == (1, 1) and i == o
        self.body = nn.Sequential(
            _cbr(i, m, 1),
            _cbr(m, m, 3, stride, g=m),
            nn.Conv2d(m, o, 1, bias=False),
            nn.BatchNorm2d(o),
        )

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_res else y


class TempBlock(nn.Module):
    """Dilated depthwise 1-D conv + pointwise, residual. Implemented as 2-D with H=1 so the
    whole graph stays NCHW (friendlier to ONNX -> QNN than mixing Conv1d)."""

    def __init__(self, c, dilation):
        super().__init__()
        self.dw = nn.Conv2d(c, c, (1, 5), 1, (0, 2 * dilation), dilation=(1, dilation), groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        return x + self.bn2(self.pw(y))


class LineRecognizer(nn.Module):
    def __init__(self, width: float = 1.0, num_classes: int = VOCAB_SIZE, dropout: float = 0.1):
        super().__init__()
        c = lambda n: max(8, int(round(n * width / 8)) * 8)  # noqa: E731
        self.stem = _cbr(1, c(24), 3, 2)                                   # H24, W/2
        self.s1 = InvRes(c(24), c(32), (1, 1), 2)
        self.s2 = nn.Sequential(InvRes(c(32), c(48), (2, 2), 3), InvRes(c(48), c(48)))            # H12, W/4
        self.s3 = nn.Sequential(InvRes(c(48), c(64), (2, 1), 3), InvRes(c(64), c(64)), InvRes(c(64), c(64)))  # H6
        self.s4 = nn.Sequential(InvRes(c(64), c(96), (2, 1), 3), InvRes(c(96), c(96)))            # H3
        self.collapse = nn.Sequential(nn.Conv2d(c(96), c(192), (3, 1), bias=False), nn.BatchNorm2d(c(192)), nn.ReLU6(inplace=True))
        self.temporal = nn.Sequential(TempBlock(c(192), 1), TempBlock(c(192), 2), TempBlock(c(192), 4), TempBlock(c(192), 1))
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c(192), num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 0.5) / 0.5
        x = self.s4(self.s3(self.s2(self.s1(self.stem(x)))))
        x = self.collapse(x)                    # [B, C, 1, T]
        x = self.head(self.drop(self.temporal(x)))
        return x.squeeze(2).transpose(1, 2)     # [B, T, V]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
