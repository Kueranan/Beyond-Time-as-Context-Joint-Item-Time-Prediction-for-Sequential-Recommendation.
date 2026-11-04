"""Tensor-only reimplementation of the published encoder and Heatmap decoder.

State names match the author release for numerical parity checks. The unused
linear/pos_emb modules are retained to preserve the parameter layout. The
inference mask uses candidate count (upstream incorrectly uses history length).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ReferenceConfig:
    item_dim: int = 172
    time_dim: int = 100
    heads: int = 2
    layers: int = 2
    cpm_heads: int = 4
    cpm_layers: int = 2
    dropout: float = 0.1
    predictor: str = "heatmap"
    time_bins: int = 52
    max_days: float = 365.0
    sigma_days: float = 30.0
    smoothing: float = 0.1
    alpha: float = 0.005
    beta: float = 0.01

    def __post_init__(self):
        dimension = self.item_dim + 2 * self.time_dim
        if min(self.item_dim, self.time_dim, self.heads, self.layers, self.cpm_heads,
               self.cpm_layers, self.time_bins) < 1:
            raise ValueError("Dimensions and layer counts must be positive")
        if dimension % self.heads or dimension % self.cpm_heads:
            raise ValueError("Hidden dimension must be divisible by both attention head counts")
        if self.predictor != "heatmap":
            raise ValueError("The audited reference pipeline currently supports Heatmap only")
        if not 0 <= self.dropout < 1 or not 0 <= self.smoothing < 1:
            raise ValueError("dropout and smoothing must be in [0,1)")
        if min(self.max_days, self.sigma_days) <= 0 or min(self.alpha, self.beta) < 0:
            raise ValueError("Invalid time horizon, sigma or loss weights")


class CosineTimeEncoding(nn.Module):
    """Learned frequencies AND phases, without the legacy sin/cos normalization."""

    def __init__(self, dimension):
        super().__init__()
        self.w = nn.Linear(1, dimension)
        frequencies = 1 / 10 ** np.linspace(0, 9, dimension, dtype=np.float32)
        with torch.no_grad():
            self.w.weight.copy_(torch.from_numpy(frequencies).reshape(dimension, 1))
            self.w.bias.zero_()

    def forward(self, days):
        return torch.cos(self.w(days.unsqueeze(-1)))


class ResidualFeedForward(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(dimension, dimension, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dimension, dimension, 1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        hidden = self.dropout2(self.conv2(self.relu(hidden)))
        return inputs + hidden.transpose(-1, -2)


class ReferenceITDRec(nn.Module):
    def __init__(self, num_items, config, max_length=32):
        super().__init__()
        self.config = config
        dimension = config.item_dim + 2 * config.time_dim
        self.emb_dropout = nn.Dropout(config.dropout)
        # Match upstream initialization and extra reserved row; valid IDs remain 1..N.
        self.item_emb = nn.Embedding(num_items + 2, config.item_dim, padding_idx=0)
        self.time_encoder = CosineTimeEncoding(config.time_dim)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(dimension, eps=1e-8)
        for _ in range(config.layers):
            self.attention_layernorms.append(nn.LayerNorm(dimension, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(dimension, config.heads, config.dropout))
            self.forward_layernorms.append(nn.LayerNorm(dimension, eps=1e-8))
            self.forward_layers.append(ResidualFeedForward(dimension, config.dropout))
        decoder = nn.TransformerDecoderLayer(dimension, config.cpm_heads,
                                             dim_feedforward=dimension, dropout=0.1,
                                             activation="relu", batch_first=True)
        self.time_predictor = nn.TransformerDecoder(decoder, config.cpm_layers)
        self.regressor = nn.Sequential(nn.Linear(dimension, config.time_bins), nn.ReLU(),
                                       nn.Linear(config.time_bins, config.time_bins))
        self.linear = nn.Sequential(nn.Linear(dimension + config.item_dim, 80), nn.ReLU(), nn.Linear(80, 1))
        self.pos_emb = nn.Embedding(max_length, config.time_dim)

    @staticmethod
    def causal_mask(length, device):
        return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)

    def encode(self, items, absolute_days, relative_days):
        hidden = torch.cat((self.item_emb(items) * self.config.item_dim ** 0.5,
                            self.time_encoder(absolute_days),
                            self.time_encoder(relative_days)), dim=-1)
        hidden = self.emb_dropout(hidden)
        present = items.ne(0).unsqueeze(-1)
        hidden = hidden * present
        mask = self.causal_mask(items.shape[1], items.device)
        for norm, attention, ffn_norm, ffn in zip(self.attention_layernorms, self.attention_layers,
                                                self.forward_layernorms, self.forward_layers):
            sequence = hidden.transpose(0, 1)
            query = norm(sequence)
            # Upstream has no padding-key mask: preserve that behavior for parity.
            attended, _ = attention(query, sequence, sequence, attn_mask=mask)
            hidden = (query + attended).transpose(0, 1)
            hidden = ffn(ffn_norm(hidden)) * present
        return self.last_layernorm(hidden)

    def candidate_memory(self, candidates):
        zeros = torch.zeros_like(candidates, dtype=self.item_emb.weight.dtype)
        time = self.time_encoder(zeros)
        return torch.cat((self.item_emb(candidates), time, time), dim=-1)

    def decode_time(self, query, candidates):
        # Full sequence/candidate memory, not the legacy singleton cross-attention.
        mask = self.causal_mask(query.shape[1], query.device)
        return self.regressor(self.time_predictor(query, self.candidate_memory(candidates), tgt_mask=mask))

    def forward(self, batch):
        history = self.encode(batch["items"], batch["absolute_days"], batch["relative_days"])
        item_history = history[..., :self.config.item_dim]
        result = {}
        for name in ("positive", "negative"):
            candidates = batch[name]
            result[name + "_logits"] = (item_history * self.item_emb(candidates)).sum(-1)
            result[name + "_time"] = self.decode_time(history, candidates)
        return result

    def predict(self, batch, compact_query=False):
        history = self.encode(batch["items"], batch["absolute_days"], batch["relative_days"])
        candidates = batch["candidates"]
        last = history[:, -1]
        logits = (self.item_emb(candidates) * last[:, None, :self.config.item_dim]).sum(-1)
        if compact_query and self.training:
            raise ValueError("Compact identical-query inference is only valid in eval mode")
        # With dropout disabled, all repeated queries attend to identical memory
        # and produce identical outputs. A single query avoids catalog-size squared
        # self-attention for recommendation, without changing candidate memory.
        query = last[:, None, :] if compact_query else last[:, None, :].expand(-1, candidates.shape[1], -1)
        heatmap = self.decode_time(query, candidates)
        if compact_query:
            heatmap = heatmap.expand(-1, candidates.shape[1], -1)
        days = heatmap.argmax(-1) * (2 * self.config.max_days / self.config.time_bins)
        return {"item_logits": logits, "time_output": heatmap, "predicted_days": days,
                "scores": logits * torch.exp(-self.config.beta * days)}
