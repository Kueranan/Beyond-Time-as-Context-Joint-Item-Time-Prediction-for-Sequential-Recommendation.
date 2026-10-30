"""Print one real training example and each stage's tensor shape for teaching.

python -m itdrec.explain --data-dir data/processed/pet_demo
This uses a randomly initialized model, not a trained recommendation.
"""

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .data.dataset import TrainingDataset, load_data
from .losses import joint_loss
from .models import ITDRec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/pet_demo"))
    parser.add_argument("--example", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    sequences, stats = load_data(args.data_dir)
    dataset = TrainingDataset(sequences, stats["items"], max_length=8, negatives=3)
    if not 0 <= args.example < len(dataset):
        raise ValueError(f"example must be between 0 and {len(dataset) - 1}")
    sample = dataset[args.example]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    config = ModelConfig(item_dim=32, time_dim=8, layers=1, cpm_layers=1)
    model = ITDRec(stats["items"], config).eval()
    with torch.no_grad():
        history = model.encoder(model.item_embedding(batch["items"]), batch["items"],
                                batch["times"], batch["lengths"])
        last = history[:, batch["lengths"].item() - 1]
        candidates = model.item_embedding(batch["candidates"])
        temporal = model.cpm(last, candidates)
        output = model.decode(last, batch["candidates"])
        total, losses = joint_loss(model, output, batch)
    result = {
        "note": "Untrained model: these values explain the calculation, not recommendation quality.",
        "input_items": sample["items"].tolist(), "input_days_from_first": sample["times"].tolist(),
        "candidate_items": sample["candidates"].tolist(), "positive_labels": sample["labels"].tolist(),
        "target_days_for_loss_only": sample["target_days"].tolist(),
        "shapes": {"history_hidden": list(history.shape), "last_hidden": list(last.shape),
                   "candidate_embeddings": list(candidates.shape), "cpm_output": list(temporal.shape),
                   "heatmap_output": list(output["time_output"].shape)},
        "item_logits": output["item_logits"][0].tolist(),
        "predicted_days": output["predicted_days"][0].tolist(),
        "reranked_scores": output["scores"][0].tolist(), "loss": total.item(), **losses,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
