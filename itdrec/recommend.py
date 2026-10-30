"""Show real product IDs and predicted intervals from a saved model."""

import argparse
import json
from pathlib import Path

import torch

from .data.dataset import history_tensors, load_data
from .evaluate import load_checkpoint, move_batch


@torch.no_grad()
def recommend(model, items, timestamps, num_items, max_length, top_k=10, chunk_size=512):
    if top_k < 1 or chunk_size < 1:
        raise ValueError("top_k and chunk_size must be positive")
    model.eval()
    device = next(model.parameters()).device
    batch = {k: v.unsqueeze(0) for k, v in history_tensors(items, timestamps, max_length).items()}
    batch = move_batch(batch, device)
    last_hidden = model.encode(batch["items"], batch["times"], batch["lengths"])
    observed = set(items)
    candidates = [i for i in range(1, num_items + 1) if i not in observed]
    best = []
    for start in range(0, len(candidates), chunk_size):
        ids = torch.tensor([candidates[start:start + chunk_size]], device=device)
        output = model.decode(last_hidden, ids)
        scores, indices = output["scores"][0].topk(min(top_k, ids.shape[1]))
        for score, index in zip(scores.tolist(), indices.tolist()):
            best.append({"item_id": ids[0, index].item(), "score": score,
                         "predicted_days": output["predicted_days"][0, index].item()})
        best = sorted(best, key=lambda item: item["score"], reverse=True)[:top_k]
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--user-index", type=int, default=0)
    parser.add_argument("--history", choices=["train", "valid", "all"], default="valid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device, args.data_dir)
    sequences, stats = load_data(args.data_dir)
    if not 0 <= args.user_index < len(sequences):
        raise ValueError("user-index outside the available sequence range")
    sequence = sequences[args.user_index]
    end = {"train": sequence["train_end"], "valid": sequence["valid_end"],
           "all": len(sequence["items"])}[args.history]
    result = recommend(model, sequence["items"][:end], sequence["timestamps_ms"][:end],
                       stats["items"], checkpoint["training_config"]["max_length"], args.top_k)
    mapping = json.loads((args.data_dir / "items.json").read_text())
    for item in result:
        item["parent_asin"] = mapping[str(item["item_id"])]
        item["predicted_timestamp_ms"] = round(sequence["timestamps_ms"][end - 1] +
                                               item["predicted_days"] * 86_400_000)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
