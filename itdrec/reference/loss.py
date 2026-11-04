"""Author-release normalized Gaussian targets and positive+negative reductions."""

import torch
from torch.nn import functional as F


def heatmap_targets(days, config):
    grid = torch.arange(config.time_bins, device=days.device, dtype=days.dtype)
    grid = grid * (2 * config.max_days / config.time_bins)
    gaussian = torch.exp(-0.5 * ((grid - days.unsqueeze(-1)) / config.sigma_days).square())
    total = gaussian.sum(-1, keepdim=True)
    normalized = torch.where(total > 0, gaussian / (total + 1e-10), gaussian)
    return (1 - config.smoothing) * normalized + config.smoothing / config.time_bins


def joint_loss(output, batch, config):
    valid = batch["positive"].ne(0)
    if not valid.any():
        raise ValueError("No next-item targets in this batch")
    positive_days = batch["target_days"]
    # Published trainer uses this deterministic offset, unlike the paper's Uniform.
    negative_days = positive_days + config.max_days
    temporal = valid & positive_days.ne(0)
    item_loss = output["positive_logits"].sum() * 0
    time_loss = output["positive_time"].sum() * 0
    for name, days, label in (("positive", positive_days, 1.0), ("negative", negative_days, 0.0)):
        weighted = output[name + "_logits"][valid] * torch.exp(-config.beta * days[valid])
        item_loss = item_loss + F.binary_cross_entropy_with_logits(weighted, torch.full_like(weighted, label))
        if temporal.any():
            time_loss = time_loss + F.mse_loss(output[name + "_time"][temporal],
                                             heatmap_targets(days, config)[temporal])
    # Unlike upstream's mean(empty), batches with only zero gaps have zero auxiliary loss.
    return item_loss + config.alpha * time_loss, {"item_loss": item_loss.detach().item(),
                                                 "time_loss": time_loss.detach().item()}
