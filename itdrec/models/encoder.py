"""Time-aware causal Transformer, paper section IV-A."""

import torch
from torch import nn

from .time_encoding import FunctionalTimeEncoding


class TimeAwareEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension = config.item_dim + 2 * config.time_dim
        self.time_encoding = FunctionalTimeEncoding(config.time_dim)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=config.heads, dim_feedforward=dimension,
            dropout=config.dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)

    def forward(self, item_embeddings, items, times, lengths):
        # times already means days from first retained history event.
        last_time = times.gather(1, (lengths - 1).unsqueeze(1))
        absolute = self.time_encoding(times)
        relative = self.time_encoding(times - last_time)
        inputs = self.dropout(torch.cat((item_embeddings, absolute, relative), dim=-1))
        padding_mask = items.eq(0)
        inputs = inputs.masked_fill(padding_mask.unsqueeze(-1), 0)
        causal_mask = torch.ones(items.shape[1], items.shape[1], dtype=torch.bool,
                                 device=items.device).triu(1)
        hidden = self.transformer(inputs, mask=causal_mask, src_key_padding_mask=padding_mask)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0)
