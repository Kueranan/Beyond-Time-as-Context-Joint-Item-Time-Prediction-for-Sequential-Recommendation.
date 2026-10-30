"""Train ITDRec, choose the best validation NDCG@10, then evaluate test once."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data.dataset import TrainingDataset, load_data
from .data.download import sha256
from .evaluate import evaluate, load_checkpoint, move_batch, predict_batch
from .losses import joint_loss
from .models import ITDRec

TRAIN_DEFAULTS = {
    "epochs": 100, "batch_size": 200, "learning_rate": 0.001, "weight_decay": 0.0,
    "max_length": 32, "negatives": 5, "candidate_count": 100,
    "patience": 20, "seed": 0, "device": "cpu", "threads": 2,
    "max_train_batches": 0, "max_eval_users": 0,
}


def train(data_dir, output_dir, model_config, settings):
    if any(settings[k] < 1 for k in ["epochs", "batch_size", "max_length", "negatives",
                                    "candidate_count", "patience", "threads"]):
        raise ValueError("Training sizes, counts, and patience must be positive")
    if settings["seed"] < 0 or settings["max_train_batches"] < 0 or settings["max_eval_users"] < 0:
        raise ValueError("Seed and subset limits must be nonnegative")
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    device = torch.device(settings["device"])
    sequences, stats = load_data(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"{checkpoint_path} exists; choose a new output directory to preserve the run")
    data_hashes = {name: sha256(Path(data_dir) / name) for name in ["sequences.json", "items.json"]}
    run_config = {"model": asdict(model_config), "training": settings, "data": stats,
                  "data_hashes": data_hashes,
                  "software": {"torch": str(torch.__version__), "numpy": np.__version__}}
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    dataset = TrainingDataset(sequences, stats["items"], settings["max_length"],
                              settings["negatives"], model_config.max_days, settings["seed"])
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    model = ITDRec(stats["items"], model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    best_score, stale, history = -float("inf"), 0, []
    for epoch in range(settings["epochs"]):
        model.train()
        dataset.epoch = epoch
        total_loss, total_item, total_time, samples = 0.0, 0.0, 0.0, 0
        for batch_index, batch in enumerate(loader):
            if settings["max_train_batches"] and batch_index >= settings["max_train_batches"]:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = predict_batch(model, batch)
            loss, parts = joint_loss(model, output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect input times and optimizer settings")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            batch_size = len(batch["items"])
            total_loss += loss.item() * batch_size
            total_item += parts["item_loss"] * batch_size
            total_time += parts["time_loss"] * batch_size
            samples += batch_size
        validation = evaluate(model, sequences, stats["items"], device, "valid", settings["max_length"],
                              settings["candidate_count"], settings["seed"], settings["max_eval_users"])
        record = {"epoch": epoch + 1, "loss": total_loss / samples,
                  "item_loss": total_item / samples, "time_loss": total_time / samples,
                  "samples_trained": samples, "validation": validation}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch + 1} loss={record['loss']:.4f} "
              f"valid_ndcg@10={validation['ndcg@10']:.4f}", flush=True)
        if validation["ndcg@10"] > best_score:
            best_score, stale = validation["ndcg@10"], 0
            torch.save({"model_state": model.state_dict(), "model_config": asdict(model_config),
                        "training_config": settings, "num_items": stats["items"], "epoch": epoch + 1,
                        "data_hashes": data_hashes, "validation": validation}, checkpoint_path)
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model, checkpoint = load_checkpoint(checkpoint_path, device, data_dir)
    test = evaluate(model, sequences, stats["items"], device, "test", settings["max_length"],
                    settings["candidate_count"], settings["seed"], settings["max_eval_users"])
    result = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"], "test": test,
              "limited_run": bool(settings["max_train_batches"] or settings["max_eval_users"] or stats["max_users"])}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/pet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pet_heatmap"))
    parser.add_argument("--config", type=Path, default=Path("configs/pet.json"))
    parser.add_argument("--predictor", choices=["regression", "binary", "heatmap"])
    for name in ["epochs", "seed", "batch_size", "max_train_batches", "max_eval_users"]:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model_values = config.get("model", {})
    settings = dict(TRAIN_DEFAULTS)
    unknown = set(config.get("training", {})) - set(settings)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    settings.update(config.get("training", {}))
    for name in ["epochs", "seed", "batch_size", "device", "max_train_batches", "max_eval_users"]:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if args.predictor:
        model_values["predictor"] = args.predictor
    train(args.data_dir, args.output_dir, ModelConfig(**model_values), settings)


if __name__ == "__main__":
    main()
