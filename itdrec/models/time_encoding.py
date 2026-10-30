"""Learnable continuous time encoding, equation (1)."""

import math

import torch
from torch import nn


class FunctionalTimeEncoding(nn.Module):
    """Represent days as sin/cos pairs; d/2 frequencies produce d features.

    paper เขียนจำนวนความถี่และมิติไม่ตรงกัน จึงกำหนดขนาดผลลัพธ์ให้ชัดเจน
    """

    def __init__(self, dimension):
        super().__init__()
        if dimension < 2 or dimension % 2:
            raise ValueError("Time encoding dimension must be positive and even")
        self.frequencies = nn.Parameter(torch.logspace(0, -9, dimension // 2))
        self.scale = math.sqrt(2 / dimension)

    def forward(self, days):
        angles = days.unsqueeze(-1) * self.frequencies
        return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2) * self.scale
