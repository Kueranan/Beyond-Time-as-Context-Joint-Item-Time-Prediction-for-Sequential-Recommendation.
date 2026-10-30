"""Multi-positive ranking metrics and time errors in real day units."""

import math

import torch


def ranking_metrics(scores, labels, cutoffs=(5, 10, 20)):
    """Compute one user's cluster-level Hit, Recall, and NDCG."""
    order = scores.argsort(descending=True, stable=True)
    ranked = labels[order]
    positives = int(labels.sum().item())
    if positives == 0:
        raise ValueError("Ranking metrics require at least one positive target")
    result = {}
    for k in cutoffs:
        top = ranked[:k]
        hits = float(top.sum().item())
        discounts = torch.log2(torch.arange(len(top), device=top.device, dtype=torch.float32) + 2)
        dcg = float((top / discounts).sum().item())
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, positives)))
        result.update({f"hit@{k}": float(hits > 0), f"recall@{k}": hits / positives,
                       f"ndcg@{k}": dcg / ideal})
    return result
