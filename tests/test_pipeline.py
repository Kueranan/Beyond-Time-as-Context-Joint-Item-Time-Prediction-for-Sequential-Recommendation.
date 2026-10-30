"""Offline end-to-end fixture: training, checkpoint reload, evaluation, inference."""

import json

import pytest
import torch

from itdrec.config import ModelConfig
from itdrec.evaluate import evaluate, load_checkpoint
from itdrec.recommend import recommend
from itdrec.train import TRAIN_DEFAULTS, train


def test_checkpoint_round_trip_and_mapping_protection(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sequences = [
        {"user_id": "a", "items": [1, 2, 3, 4, 5, 6],
         "timestamps_ms": [0, 1000, 2000, 4000, 200000000, 400000000],
         "train_end": 4, "valid_end": 5},
        {"user_id": "b", "items": [3, 4, 5, 6, 7, 8],
         "timestamps_ms": [100, 1000, 3000, 4000, 300000000, 500000000],
         "train_end": 4, "valid_end": 5},
    ]
    (data_dir / "sequences.json").write_text(json.dumps(sequences))
    (data_dir / "items.json").write_text(json.dumps({str(i): f"product-{i}" for i in range(1, 9)}))
    (data_dir / "stats.json").write_text(json.dumps({"items": 8, "max_users": 0}))
    model_config = ModelConfig(item_dim=8, time_dim=4, heads=2, layers=1, cpm_layers=1, time_bins=10)
    settings = {**TRAIN_DEFAULTS, "epochs": 1, "batch_size": 2, "max_length": 4,
                "threads": 1, "negatives": 2, "candidate_count": 5}
    result = train(data_dir, tmp_path / "run", model_config, settings)
    model, checkpoint = load_checkpoint(tmp_path / "run/best.pt", data_dir=data_dir)
    repeated = evaluate(model, sequences, 8, "cpu", "test", 4, 5, 0)
    assert repeated == result["test"]
    assert checkpoint["epoch"] == 1
    recommended = recommend(model, [1, 2], [0, 1000], 8, 4, top_k=3, chunk_size=2)
    assert len(recommended) == 3
    assert {entry["item_id"] for entry in recommended}.isdisjoint({1, 2})
    assert all(entry["predicted_days"] >= 0 for entry in recommended)
    assert all(torch.isfinite(torch.tensor(entry["score"])) for entry in recommended)
    (data_dir / "items.json").write_text("{}")
    with pytest.raises(ValueError, match="Dataset mismatch"):
        load_checkpoint(tmp_path / "run/best.pt", data_dir=data_dir)
