"""Chronological Preference Mapping: history=query, candidate=key/value.

Implements candidate-wise equations (8)-(9). A singleton key means attention
weight = 1; it is candidate conditioning, not selection among historical tokens.
This limitation of the literal equation is documented in docs/PAPER_MAPPING.md.
"""

from torch import nn


class CPMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension = config.item_dim + 2 * config.time_dim
        self.attention = nn.MultiheadAttention(dimension, config.heads, dropout=config.dropout,
                                               kdim=config.item_dim, vdim=config.item_dim,
                                               batch_first=True)
        self.norm_attention = nn.LayerNorm(dimension)
        self.norm_ffn = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(),
                                 nn.Dropout(config.dropout), nn.Linear(dimension, dimension))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, history_query, candidate):
        attended, _ = self.attention(history_query, candidate, candidate, need_weights=False)
        hidden = self.norm_attention(history_query + self.dropout(attended))
        return self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))


class ChronologicalPreferenceMapping(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CPMLayer(config) for _ in range(config.cpm_layers)])

    def forward(self, last_hidden, candidates):
        batch_size, candidate_count, item_dim = candidates.shape
        # Every candidate gets its own memory: predictions cannot change when
        # candidates are reordered or split into inference chunks.
        query = last_hidden[:, None, :].expand(-1, candidate_count, -1)
        query = query.reshape(batch_size * candidate_count, 1, -1)
        memory = candidates.reshape(batch_size * candidate_count, 1, item_dim)
        # Cross-attention and pointwise FFN do not mix query positions, so only
        # the final encoded history position is needed for next-event prediction.
        for layer in self.layers:
            query = layer(query, memory)
        return query.reshape(batch_size, candidate_count, -1)
