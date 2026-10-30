"""Convert chronological histories into tensors without future-time inputs."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .prepare import DAY_MS


def load_data(directory):
    directory = Path(directory)
    return (json.loads((directory / "sequences.json").read_text()),
            json.loads((directory / "stats.json").read_text()))


def history_tensors(items, timestamps, max_length):
    """Right padding avoids fully masked causal-attention rows at the start.

    Subtract integer timestamps BEFORE float32 conversion to preserve precision.
    """
    items, timestamps = items[-max_length:], timestamps[-max_length:]
    if not items or max_length < 1:
        raise ValueError("History must be nonempty and max_length must be positive")
    length = len(items)
    item_tensor = torch.zeros(max_length, dtype=torch.long)
    time_tensor = torch.zeros(max_length, dtype=torch.float32)
    item_tensor[:length] = torch.tensor(items)
    time_tensor[:length] = torch.tensor([(t - timestamps[0]) / DAY_MS for t in timestamps])
    return {"items": item_tensor, "times": time_tensor, "lengths": torch.tensor(length)}


def sample_negatives(rng, num_items, excluded, count, replace=False):
    """Bounded sampling handles small or almost-exhausted catalogs safely."""
    excluded = {i for i in excluded if 1 <= i <= num_items}
    available = num_items - len(excluded)
    if available == 0:
        return []
    count = count if replace else min(count, available)
    result, chosen = [], set()
    for _ in range(max(100, count * 20)):
        if len(result) == count:
            return result
        candidate = int(rng.integers(1, num_items + 1))
        if candidate not in excluded and (replace or candidate not in chosen):
            result.append(candidate)
            chosen.add(candidate)
    pool = [i for i in range(1, num_items + 1)
            if i not in excluded and (replace or i not in chosen)]
    result.extend(rng.choice(pool, count - len(result), replace=replace).tolist())
    return result


class TrainingDataset(Dataset):
    """Each prefix before train_end predicts exactly one next interaction."""

    def __init__(self, sequences, num_items, max_length=32, negatives=5,
                 max_days=365.0, seed=0):
        self.sequences, self.num_items = sequences, num_items
        self.max_length, self.negatives, self.max_days = max_length, negatives, max_days
        self.seed, self.epoch = seed, 0
        self.examples = [(u, target) for u, seq in enumerate(sequences)
                         for target in range(1, seq["train_end"])]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        user, target = self.examples[index]
        seq = self.sequences[user]
        batch = history_tensors(seq["items"][:target], seq["timestamps_ms"][:target], self.max_length)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        # Only the prefix and current label are consulted when sampling training negatives.
        negatives = sample_negatives(rng, self.num_items, set(seq["items"][:target + 1]),
                                     self.negatives, replace=True)
        gap = (seq["timestamps_ms"][target] - seq["timestamps_ms"][target - 1]) / DAY_MS
        padding = self.negatives - len(negatives)
        batch.update({
            "candidates": torch.tensor([seq["items"][target], *negatives, *([0] * padding)]),
            "labels": torch.tensor([1.0] + [0.0] * self.negatives),
            "target_days": torch.tensor([min(gap, self.max_days)] +
                                        rng.uniform(self.max_days, 2 * self.max_days, self.negatives).tolist()),
            "candidate_mask": torch.tensor([True] * (1 + len(negatives)) + [False] * padding),
        })
        return batch


def evaluation_examples(sequences, num_items, split="valid", max_length=32,
                        candidate_count=100, seed=0):
    """One fixed candidate list per user, all cluster positives, no teacher forcing."""
    if split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    for user, seq in enumerate(sequences):
        start = seq["train_end"] if split == "valid" else seq["valid_end"]
        end = seq["valid_end"] if split == "valid" else len(seq["items"])
        # If repeated items exist, evaluate their first occurrence within the cluster.
        positive_times = {}
        for item, time in zip(seq["items"][start:end], seq["timestamps_ms"][start:end]):
            positive_times.setdefault(item, time)
        positives = list(positive_times)
        rng = np.random.default_rng(np.random.SeedSequence([seed, user, 1 if split == "test" else 0]))
        negatives = sample_negatives(rng, num_items, set(seq["items"]),
                                     max(0, candidate_count - len(positives)))
        candidates = positives + negatives
        # Shuffle to avoid favoring positives when scores tie.
        rng.shuffle(candidates)
        labels = [float(item in positive_times) for item in candidates]
        days = [(positive_times[item] - seq["timestamps_ms"][start - 1]) / DAY_MS
                if item in positive_times else 0.0 for item in candidates]
        batch = history_tensors(seq["items"][:start], seq["timestamps_ms"][:start], max_length)
        batch.update({"candidates": torch.tensor(candidates), "labels": torch.tensor(labels),
                      "target_days": torch.tensor(days)})
        yield {key: value.unsqueeze(0) for key, value in batch.items()}
