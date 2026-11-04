"""One left-padded training sequence per user, with next-item labels at all positions."""

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.dataset import sample_negatives
from ..data.prepare import DAY_MS, cluster_boundaries


def cluster_relative_days(timestamps):
    """Time since the LAST event in the PREVIOUS cluster; first cluster is zero.

    This is the published DataLoader's between_cluster feature, not t - t_last.
    A gap equal to one day remains inside the cluster, as in DBSCAN(eps=1 day).
    """
    relative = [0.0] * len(timestamps)
    starts = cluster_boundaries(timestamps)
    for start, end in zip(starts[1:], starts[2:] + [len(timestamps)]):
        origin = timestamps[start - 1]
        for index in range(start, end):
            relative[index] = (timestamps[index] - origin) / DAY_MS
    return relative


def history_tensors(items, timestamps, relative, max_length):
    items, timestamps, relative = items[-max_length:], timestamps[-max_length:], relative[-max_length:]
    if not items:
        raise ValueError("Reference encoder requires nonempty history")
    length = len(items)
    result = {"items": torch.zeros(max_length, dtype=torch.long),
              "absolute_days": torch.zeros(max_length), "relative_days": torch.zeros(max_length)}
    result["items"][-length:] = torch.tensor(items)
    result["absolute_days"][-length:] = torch.tensor([(t - timestamps[0]) / DAY_MS for t in timestamps])
    result["relative_days"][-length:] = torch.tensor(relative)
    return result


class SequenceTrainingDataset(Dataset):
    def __init__(self, sequences, num_items, max_length=32, seed=0):
        self.sequences = [seq for seq in sequences if seq["train_end"] >= 2]
        self.num_items, self.max_length, self.seed, self.epoch = num_items, max_length, seed, 0
        self.relative = [cluster_relative_days(seq["timestamps_ms"]) for seq in self.sequences]
        for seq in self.sequences:
            if len(set(seq["items"][:seq["train_end"]])) >= num_items:
                raise ValueError("A training user covers the whole catalog; no negative exists")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        end = seq["train_end"]
        first_target = max(1, end - self.max_length)
        targets = seq["items"][first_target:end]
        batch = history_tensors(seq["items"][:end - 1], seq["timestamps_ms"][:end - 1],
                                self.relative[index][:end - 1], self.max_length)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        negatives = sample_negatives(rng, self.num_items, set(seq["items"][:end]), len(targets), replace=True)
        batch.update({"positive": torch.zeros(self.max_length, dtype=torch.long),
                      "negative": torch.zeros(self.max_length, dtype=torch.long),
                      "target_days": torch.zeros(self.max_length)})
        batch["positive"][-len(targets):] = torch.tensor(targets)
        batch["negative"][-len(targets):] = torch.tensor(negatives)
        batch["target_days"][-len(targets):] = torch.tensor(self.relative[index][first_target:end])
        return batch


def evaluation_examples(sequences, num_items, split="valid", max_length=32,
                        candidate_count=100, seed=0, max_users=0):
    if split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    count = 0
    for user, seq in enumerate(sequences):
        start = seq["train_end"] if split == "valid" else seq["valid_end"]
        end = seq["valid_end"] if split == "valid" else len(seq["items"])
        if start == end or start == 0:
            continue
        if max_users and count >= max_users:
            break
        positives = seq["items"][start:end]
        # Upstream sampling permits duplicate negatives, and only consults the
        # history and current target split (not test positives during validation).
        rng = np.random.default_rng(np.random.SeedSequence([seed, user, int(split == "test")]))
        negatives = sample_negatives(rng, num_items, set(seq["items"][:end]),
                                     max(0, candidate_count - len(positives)), replace=True)
        relative = cluster_relative_days(seq["timestamps_ms"][:start])
        batch = history_tensors(seq["items"][:start], seq["timestamps_ms"][:start], relative, max_length)
        batch["candidates"] = torch.tensor(positives + negatives, dtype=torch.long)
        batch["labels"] = torch.tensor([1.0] * len(positives) + [0.0] * len(negatives))
        days = [(t - seq["timestamps_ms"][start - 1]) / DAY_MS for t in seq["timestamps_ms"][start:end]]
        batch["target_days"] = torch.tensor(days + [0.0] * len(negatives))
        count += 1
        yield {key: value.unsqueeze(0) for key, value in batch.items()}
