"""Typed configuration; experiment files use ordinary JSON."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    item_dim: int = 64
    time_dim: int = 16
    heads: int = 4
    layers: int = 2
    cpm_layers: int = 2
    cpm_heads: int | None = None
    dropout: float = 0.1
    predictor: str = "heatmap"
    time_bins: int = 100
    max_days: float = 365.0
    sigma_days: float = 14.0
    heatmap_density_units: str = "horizon"
    alpha: float = 1.0
    beta: float = 0.001

    def __post_init__(self):
        if min(self.item_dim, self.time_dim, self.heads, self.layers, self.cpm_layers, self.time_bins) < 1:
            raise ValueError("Dimensions, heads, layers, and time_bins must be positive")
        if self.time_dim % 2 or (self.item_dim + 2 * self.time_dim) % self.heads:
            raise ValueError("time_dim must be even; item_dim + 2*time_dim must be divisible by heads")
        if self.cpm_heads is not None and (self.cpm_heads < 1 or
                                          (self.item_dim + 2 * self.time_dim) % self.cpm_heads):
            raise ValueError("cpm_heads must be positive and divide item_dim + 2*time_dim")
        if self.heatmap_density_units not in {"horizon", "days"}:
            raise ValueError("heatmap_density_units must be horizon or days")
        if self.predictor not in {"regression", "binary", "heatmap"}:
            raise ValueError("predictor must be regression, binary, or heatmap")
        if self.max_days <= 0 or self.sigma_days <= 0 or self.alpha < 0 or self.beta < 0:
            raise ValueError("Invalid time horizon, sigma, or loss/decay weights")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
