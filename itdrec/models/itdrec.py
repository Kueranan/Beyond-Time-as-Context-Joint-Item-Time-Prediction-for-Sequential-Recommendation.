"""ITDRec assembles the encoder, item decoder, CPM, and time head."""

import torch
from torch import nn

from .cpm import ChronologicalPreferenceMapping
from .encoder import TimeAwareEncoder
from .time_heads import build_time_head


class ITDRec(nn.Module):
    def __init__(self, num_items, config):
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(num_items + 1, config.item_dim, padding_idx=0)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        self.encoder = TimeAwareEncoder(config)
        self.cpm = ChronologicalPreferenceMapping(config)
        self.time_head = build_time_head(config)

    def encode(self, items, times, lengths):
        hidden = self.encoder(self.item_embedding(items), items, times, lengths)
        return hidden[torch.arange(items.shape[0], device=items.device), lengths - 1]

    def decode(self, last_hidden, candidates):
        candidate_embeddings = self.item_embedding(candidates)
        item_logits = (last_hidden[:, None, :self.config.item_dim] * candidate_embeddings).sum(-1)
        temporal_hidden = self.cpm(last_hidden, candidate_embeddings)
        time_output = self.time_head(temporal_hidden)
        days = self.time_head.predict_days(time_output)
        # Literal equation (25); negative logits have a known sign caveat.
        scores = item_logits * torch.exp(-self.config.beta * days)
        return {"item_logits": item_logits, "time_output": time_output,
                "predicted_days": days, "scores": scores}

    def forward(self, items, times, lengths, candidates):
        return self.decode(self.encode(items, times, lengths), candidates)
