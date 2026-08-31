"""Q-network over compressor-tree observations.

A ResNet-18 trunk adapted to the small non-square observation: the input is
``(2, stages, columns)`` rather than an image, so the stem uses a 3x3 convolution with stride 1 and there is no max-pool -- otherwise
the few rows of the observation would be thrown away before the first block.

The network is a plain function of a *batch* of observations.  Action masking
lives in the agent, not here: mixing it into ``forward`` forces the mask to be
recomputed on every forward pass and makes batching impossible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.residual(x) + self.shortcut(x))


class QNetwork(nn.Module):
    """Maps a batch of observations to one Q-value per action."""

    def __init__(self, n_actions: int, blocks: tuple[int, ...] = (2, 2, 2, 2),
                 widths: tuple[int, ...] = (64, 128, 256, 512), in_channels: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers = []
        channels = widths[0]
        for width, count in zip(widths, blocks):
            for _ in range(count):
                layers.append(BasicBlock(channels, width))
                channels = width
        self.trunk = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        x = self.trunk(self.stem(obs))
        return self.head(self.pool(x).flatten(1))
