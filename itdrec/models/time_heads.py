"""Three interchangeable time predictors, paper section IV-C.

Each head exposes forward(), loss(), and predict_days().
Training labels are in days; internal regression/heatmap coordinates use horizon units.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(dimension, outputs):
    return nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, outputs))


class RegressionHead(nn.Module):
    def __init__(self, dimension, max_days, **_):
        super().__init__()
        self.network, self.max_days = mlp(dimension, 1), max_days

    def forward(self, hidden):
        # Enforce nonnegative intervals, including early in training.
        return F.softplus(self.network(hidden).squeeze(-1))

    def loss(self, output, target_days):
        return (output - target_days / self.max_days).square()

    def predict_days(self, output):
        return output * self.max_days


class BinaryHead(nn.Module):
    def __init__(self, dimension, max_days, time_bins, **_):
        super().__init__()
        self.network, self.max_days = mlp(dimension, time_bins), max_days
        self.register_buffer("grid_days", torch.linspace(max_days / time_bins, max_days, time_bins))

    def forward(self, hidden):
        return self.network(hidden)  # logits; BCEWithLogits applies sigmoid stably

    def targets(self, target_days):
        # Cumulative labels: event has occurred by this boundary, eq. (15).
        return (self.grid_days >= target_days.unsqueeze(-1)).float()

    def loss(self, output, target_days):
        return F.binary_cross_entropy_with_logits(output, self.targets(target_days), reduction="none").mean(-1)

    def predict_days(self, output):
        fired = output >= 0  # sigmoid(logit) >= 0.5
        first = fired.long().argmax(-1)
        return torch.where(fired.any(-1), self.grid_days[first], 2 * self.max_days)


class HeatmapHead(nn.Module):
    def __init__(self, dimension, max_days, time_bins, sigma_days):
        super().__init__()
        self.network, self.max_days = mlp(dimension, time_bins), max_days
        self.sigma = sigma_days / max_days
        # Full [0, 2*horizon) range represents negative out-of-horizon labels.
        self.register_buffer("grid", torch.arange(time_bins) * (2.0 / time_bins))

    def forward(self, hidden):
        return self.network(hidden)

    def targets(self, target_days):
        distance = self.grid - target_days.unsqueeze(-1) / self.max_days
        return torch.exp(-0.5 * (distance / self.sigma).square()) / (math.sqrt(2 * math.pi) * self.sigma)

    def loss(self, output, target_days):
        return (output - self.targets(target_days)).square().mean(-1)

    def predict_days(self, output):
        # The location of the maximum, not the maximum amplitude, is the time.
        return self.grid[output.argmax(-1)] * self.max_days


def build_time_head(config):
    head = {"regression": RegressionHead, "binary": BinaryHead, "heatmap": HeatmapHead}[config.predictor]
    return head(dimension=config.item_dim + 2 * config.time_dim,
                max_days=config.max_days, time_bins=config.time_bins, sigma_days=config.sigma_days)
