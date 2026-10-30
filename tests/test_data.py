import csv
import gzip
import json

import numpy as np
import torch

from itdrec.data.dataset import TrainingDataset, evaluation_examples, history_tensors, sample_negatives
from itdrec.data.prepare import DAY_MS, cluster_boundaries, prepare, split_sequence


def sequence():
    return {"user_id": "u", "items": [1, 2, 3, 4, 5, 6],
            "timestamps_ms": [0, DAY_MS, 3 * DAY_MS, 3 * DAY_MS + 100,
                              8 * DAY_MS, 8 * DAY_MS + 100],
            "train_end": 2, "valid_end": 4}


def test_clusters_and_chronology():
    seq = sequence()
    assert cluster_boundaries(seq["timestamps_ms"]) == [0, 2, 4]
    split = split_sequence(seq["items"], seq["timestamps_ms"])
    assert (split["train_end"], split["valid_end"]) == (2, 4)
    assert split_sequence([1, 2, 3], [0, DAY_MS, 2 * DAY_MS]) is None


def test_training_and_evaluation_never_expose_target_events():
    seq = sequence()
    train = TrainingDataset([seq], 20, max_length=4, negatives=3)
    assert len(train) == 1
    example = train[0]
    assert example["items"].tolist() == [1, 0, 0, 0]
    assert example["candidates"][0] == 2
    assert example["target_days"][0] == 1
    assert set(example["candidates"][1:].tolist()).isdisjoint({1, 2})
    valid = next(evaluation_examples([seq], 20, "valid", 4, 10))
    test = next(evaluation_examples([seq], 20, "test", 4, 10))
    assert valid["items"].tolist() == [[1, 2, 0, 0]]
    assert test["items"].tolist() == [[1, 2, 3, 4]]
    positives = valid["candidates"][valid["labels"].bool()].tolist()
    assert set(positives) == {3, 4}
    negatives = valid["candidates"][~valid["labels"].bool()].tolist()
    assert set(negatives).isdisjoint(seq["items"])
    assert len(set(valid["candidates"][0].tolist())) == 10


def test_padding_precision_and_exhausted_negatives():
    base = 1_600_000_000_000
    batch = history_tensors([1, 2], [base, base + 1000], 4)
    assert torch.isclose(batch["times"][1], torch.tensor(1 / 86400))
    assert batch["lengths"] == 2
    rng = np.random.default_rng(0)
    assert sample_negatives(rng, 3, {1, 2, 3}, 5) == []
    assert sample_negatives(rng, 3, {1, 2}, 5) == [3]
    assert sample_negatives(rng, 3, {1, 2}, 5, replace=True) == [3] * 5


def test_prepare_real_format_filters_sorts_deduplicates(tmp_path):
    start = 1609459200000  # 2021-01-01 UTC
    rows = [("u", str(i), 5, start + day * DAY_MS) for i, day in enumerate([0, 1, 4, 8])]
    rows += [rows[0], ("u", "old", 5, start - DAY_MS), ("u", "future", 5, start + 400 * DAY_MS)]
    source = tmp_path / "input.csv.gz"
    with gzip.open(source, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
        writer.writerows(reversed(rows))
    stats = prepare("pet", source, tmp_path / "processed", min_interactions=4)
    assert stats["interactions"] == 4
    assert stats["train_examples"] == 1
    seq = json.loads((tmp_path / "processed/sequences.json").read_text())[0]
    assert seq["timestamps_ms"] == sorted(seq["timestamps_ms"])
    assert (seq["train_end"], seq["valid_end"]) == (2, 3)
