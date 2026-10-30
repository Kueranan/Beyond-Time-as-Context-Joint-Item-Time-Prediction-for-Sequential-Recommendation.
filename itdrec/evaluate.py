"""Evaluate fixed cluster candidates; target times never enter model.forward()."""

import argparse
import json
import math
from pathlib import Path

import torch

from .config import ModelConfig
from .data.dataset import evaluation_examples, load_data
from .data.download import sha256
from .metrics import ranking_metrics
from .models import ITDRec


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def predict_batch(model, batch):
    # A deliberately narrow API prevents accidental use of future labels/times.
    return model(batch["items"], batch["times"], batch["lengths"], batch["candidates"])


@torch.no_grad()
def evaluate(model, sequences, num_items, device, split="valid", max_length=32,
             candidate_count=100, seed=0, max_users=0):
    model.eval()
    totals, count, absolute_error, squared_error, events = {}, 0, 0.0, 0.0, 0
    clipped_error, cold_count, short_lists, large_clusters = 0.0, 0, 0, 0
    seen_train = {i for seq in sequences for i in seq["items"][:seq["train_end"]]}
    examples = evaluation_examples(sequences, num_items, split, max_length, candidate_count, seed)
    for batch in examples:
        if max_users and count >= max_users:
            break
        batch = move_batch(batch, device)
        output = predict_batch(model, batch)
        for prefix, scores in [("", output["scores"]), ("raw_", output["item_logits"])]:
            for key, value in ranking_metrics(scores[0], batch["labels"][0]).items():
                totals[prefix + key] = totals.get(prefix + key, 0.0) + value
        positive_mask = batch["labels"].bool()
        prediction = output["predicted_days"][positive_mask]
        target = batch["target_days"][positive_mask]
        error = prediction - target
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        clipped_error += (prediction - target.clamp(max=model.config.max_days)).abs().sum().item()
        events += target.numel()
        cold_count += sum(i not in seen_train for i in batch["candidates"][positive_mask].tolist())
        short_lists += int(batch["candidates"].shape[1] < candidate_count)
        large_clusters += int(positive_mask.sum().item() > candidate_count)
        count += 1
    if count == 0:
        raise ValueError("No evaluation users")
    return {**{key: value / count for key, value in totals.items()},
            "time_mae_days": absolute_error / events, "time_rmse_days": math.sqrt(squared_error / events),
            "time_mae_clipped_target_days": clipped_error / events,
            "users_evaluated": count, "positive_events": events,
            "cold_positive_fraction": cold_count / events, "short_candidate_lists": short_lists,
            "clusters_larger_than_candidate_count": large_clusters,
            "split": split, "candidate_count_requested": candidate_count,
            "time_metric_condition": "given the ground-truth candidate item; all positive cluster items"}


def load_checkpoint(path, device="cpu", data_dir=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if data_dir is not None:
        for filename, expected in checkpoint["data_hashes"].items():
            if sha256(Path(data_dir) / filename) != expected:
                raise ValueError(f"Dataset mismatch: {filename}. Use the data that trained this checkpoint.")
    model = ITDRec(checkpoint["num_items"], ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device, args.data_dir)
    sequences, stats = load_data(args.data_dir)
    settings = checkpoint["training_config"]
    result = evaluate(model, sequences, stats["items"], args.device, args.split,
                      settings["max_length"], settings["candidate_count"], settings["seed"], args.max_users)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
