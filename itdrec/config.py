"""Typed configuration; experiment files use ordinary JSON."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    item_dim: int = 64
    time_dim: int = 16
    heads: int = 4
    layers: int = 2
    cpm_layers: int = 2
    dropout: float = 0.1
    predictor: str = "heatmap"
    time_bins: int = 100
    max_days: float = 365.0
    sigma_days: float = 14.0
    alpha: float = 1.0
    beta: float = 0.001

    def __post_init__(self):
        if min(self.item_dim, self.time_dim, self.heads, self.layers, self.cpm_layers, self.time_bins) < 1:
            raise ValueError("Dimensions, heads, layers, and time_bins must be positive")
        if self.time_dim % 2 or (self.item_dim + 2 * self.time_dim) % self.heads:
            raise ValueError("time_dim must be even; item_dim + 2*time_dim must be divisible by heads")
        if self.predictor not in {"regression", "binary", "heatmap"}:
            raise ValueError("predictor must be regression, binary, or heatmap")
        if self.max_days <= 0 or self.sigma_days <= 0 or self.alpha < 0 or self.beta < 0:
            raise ValueError("Invalid time horizon, sigma, or loss/decay weights")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
