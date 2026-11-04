"""Train/evaluate the audited reference pipeline; invoked by itdrec.train configs."""

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import load_data
from ..data.download import sha256
from . import IMPLEMENTATION, SOURCE_COMMIT
from .audit_data import compare_to_paper
from .data import SequenceTrainingDataset
from .evaluate import evaluate
from .loss import joint_loss
from .model import ReferenceConfig, ReferenceITDRec

DEFAULTS = dict(epochs=100, batch_size=200, max_length=32, learning_rate=0.0001,
                weight_decay=0.0, patience=20, seed=0, device="mps", threads=2,
                candidate_count=100, max_train_batches=0, max_eval_users=0)
SELECTION_KEYS = tuple(f"{metric}@{k}" for metric in ("hit", "recall") for k in (5, 10, 20))


def is_improvement(validation, best):
    # Preserve the release's >= test on ALL six metrics, including ties.
    return best is None or all(validation[key] >= best[key] for key in SELECTION_KEYS)


def load_checkpoint(path, device="cpu", data_dir=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("implementation") != IMPLEMENTATION:
        raise ValueError("This is not an author_release_v1 checkpoint")
    if data_dir is not None:
        for name, expected in checkpoint["data_hashes"].items():
            if sha256(Path(data_dir) / name) != expected:
                raise ValueError(f"Dataset mismatch: {name}")
    model = ReferenceITDRec(checkpoint["num_items"], ReferenceConfig(**checkpoint["model_config"]),
                           checkpoint["training_config"]["max_length"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def train(data_dir, output_dir, model_config, settings, provenance=None):
    if set(settings) - set(DEFAULTS):
        raise ValueError(f"Unknown reference training settings: {sorted(set(settings) - set(DEFAULTS))}")
    settings = {**DEFAULTS, **settings}
    for key in ("epochs", "batch_size", "max_length", "patience", "threads", "candidate_count"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("seed", "max_train_batches", "max_eval_users", "weight_decay"):
        if settings[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if settings["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sequences, stats = load_data(data_dir)
    if not stats.get("retain_training_only") or stats.get("cluster_gap_days") != 1:
        raise ValueError("Reference data requires --retain-training-only and a 1-day cluster gap; "
                         "prepare into a new directory using the commands in docs/REFERENCE_AUDIT.md")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a fresh output directory")
    dataset = SequenceTrainingDataset(sequences, stats["items"], settings["max_length"], settings["seed"])
    if not len(dataset):
        raise ValueError("No users with at least two training events")
    device = torch.device(settings["device"])
    model = ReferenceITDRec(stats["items"], model_config, settings["max_length"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    # One sequence per user, fixed order, matching the released DataLoader.
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    hashes = {name: sha256(Path(data_dir) / name) for name in ("sequences.json", "items.json", "stats.json")}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(implementation=IMPLEMENTATION, source_commit=SOURCE_COMMIT,
                    model=asdict(model_config), training=settings, data=stats, data_hashes=hashes,
                    provenance=provenance or {},
                    software={"torch": str(torch.__version__), "numpy": np.__version__})
    metadata["paper_data_audit"] = compare_to_paper(stats)
    if not metadata["paper_data_audit"]["counts_match"]:
        print("DATA AUDIT: counts do not match Table I; this run is an author-release baseline "
              "on separately prepared data, not a verified paper reproduction.", flush=True)
    metadata["implementation_hashes"] = {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}
    print("Author-release pipeline: candidate-time degeneracy is a known upstream limitation; "
          "see constant_time_user_fraction and docs/REFERENCE_AUDIT.md", flush=True)
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best, stale, history = None, 0, []
    for epoch in range(settings["epochs"]):
        started = time.monotonic()
        dataset.epoch = epoch
        model.train()
        losses, items, times = [], [], []
        users = targets = 0
        for index, batch in enumerate(loader):
            if settings["max_train_batches"] and index >= settings["max_train_batches"]:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = joint_loss(model(batch), batch, model_config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()  # No extra gradient clipping: upstream does not use it.
            losses.append(loss.item())
            items.append(parts["item_loss"])
            times.append(parts["time_loss"])
            users += batch["items"].shape[0]
            targets += batch["positive"].ne(0).sum().item()
        validation = evaluate(model, sequences, stats["items"], device, "valid", settings["max_length"],
                              settings["candidate_count"], settings["seed"], settings["max_eval_users"],
                              settings["batch_size"])
        improved = is_improvement(validation, best)
        record = dict(epoch=epoch + 1, loss=float(np.mean(losses)), item_loss=float(np.mean(items)),
                      time_loss=float(np.mean(times)), samples_trained=users, targets_trained=targets,
                      validation=validation, selected_checkpoint=improved,
                      seconds=time.monotonic() - started)
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch + 1} loss={record['loss']:.4f} valid_ndcg@10={validation['ndcg@10']:.4f} "
              f"constant_time_users={validation['constant_time_user_fraction']:.1%} selected={improved}", flush=True)
        if improved:
            best, stale = validation, 0
            torch.save(dict(implementation=IMPLEMENTATION, source_commit=SOURCE_COMMIT,
                            model_state=model.state_dict(), model_config=asdict(model_config),
                            training_config=settings, num_items=stats["items"], epoch=epoch + 1,
                            data_hashes=hashes, validation=validation), output_dir / "best.pt")
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model, checkpoint = load_checkpoint(output_dir / "best.pt", device, data_dir)
    test = evaluate(model, sequences, stats["items"], device, "test", settings["max_length"],
                    settings["candidate_count"], settings["seed"], settings["max_eval_users"],
                    settings["batch_size"])
    result = dict(implementation=IMPLEMENTATION, best_epoch=checkpoint["epoch"],
                  checkpoint_selection="all six Hit/Recall metrics nondecreasing (ties accepted)",
                  validation=checkpoint["validation"], test=test,
                  limited_run=bool(settings["max_train_batches"] or settings["max_eval_users"] or stats.get("max_users", 0)))
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def train_from_config(config, args):
    settings = {**config.get("training", {})}
    for key in ("epochs", "seed", "batch_size", "device", "max_train_batches", "max_eval_users"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    values = dict(config["model"])
    if args.predictor is not None:
        values["predictor"] = args.predictor
    return train(args.data_dir or config["data_dir"], args.output_dir or config["output_dir"],
                 ReferenceConfig(**values), settings, config.get("provenance"))
