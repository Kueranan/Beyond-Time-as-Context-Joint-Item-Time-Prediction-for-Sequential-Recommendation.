"""Joint time-weighted item BCE + auxiliary time loss, equations (22)-(24)."""

import torch
from torch.nn import functional as F


def joint_loss(model, output, batch):
    weighted_logits = output["item_logits"] * torch.exp(-model.config.beta * batch["target_days"])
    item_losses = F.binary_cross_entropy_with_logits(weighted_logits, batch["labels"], reduction="none")
    time_losses = model.time_head.loss(output["time_output"], batch["target_days"])
    mask = batch["candidate_mask"].to(item_losses.dtype)
    denominator = mask.sum().clamp_min(1)
    item_loss = (item_losses * mask).sum() / denominator
    time_loss = (time_losses * mask).sum() / denominator
    total = item_loss + model.config.alpha * time_loss
    return total, {"item_loss": item_loss.detach().item(), "time_loss": time_loss.detach().item()}
