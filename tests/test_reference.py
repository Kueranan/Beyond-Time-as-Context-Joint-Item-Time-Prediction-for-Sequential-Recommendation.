"""Behavioral checks for the audited source implementation and migration."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from itdrec.reference.data import SequenceTrainingDataset, cluster_relative_days, evaluation_examples
from itdrec.reference.evaluate import evaluate, ranking_metrics
from itdrec.reference.loss import heatmap_targets, joint_loss
from itdrec.reference.model import ReferenceConfig, ReferenceITDRec
from itdrec.reference.train import is_improvement, load_checkpoint, train


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(0)
    torch.set_num_threads(1)


def config():
    return ReferenceConfig(item_dim=8, time_dim=4, heads=2, layers=1, cpm_layers=1,
                           dropout=0, time_bins=10)


def sequences():
    return [{"user_id": "a", "items": [1, 2, 3, 4, 5, 6],
             "timestamps_ms": [86400000 * d for d in (1, 1.5, 4, 4.5, 8, 12)],
             "train_end": 4, "valid_end": 5},
            {"user_id": "b", "items": [3, 4, 5, 6, 7, 8],
             "timestamps_ms": [86400000 * d for d in (1, 2, 4, 5, 8, 12)],
             "train_end": 4, "valid_end": 5}]


def test_cluster_gaps_and_sequence_targets_are_not_legacy_prefix_targets():
    seq = sequences()[0]
    assert cluster_relative_days(seq["timestamps_ms"]) == [0, 0, 2.5, 3, 3.5, 4]
    dataset = SequenceTrainingDataset([seq], 12, max_length=4)
    assert len(dataset) == 1  # one user, not three prefix examples
    row = dataset[0]
    assert row["items"].tolist() == [0, 1, 2, 3]
    assert row["positive"].tolist() == [0, 2, 3, 4]
    assert row["target_days"].tolist() == [0, 0, 2.5, 3]
    assert set(row["negative"][1:].tolist()).isdisjoint({1, 2, 3, 4})
    truncated = SequenceTrainingDataset([seq], 12, max_length=2)[0]
    assert truncated["items"].tolist() == [2, 3]
    assert truncated["positive"].tolist() == [3, 4]
    assert truncated["absolute_days"].tolist() == [0, 2.5]
    assert truncated["relative_days"].tolist() == [0, 2.5]


def test_train_only_users_and_singletons_do_not_create_empty_loss_batches():
    train_only = {**sequences()[0], "train_end": 6, "valid_end": 6}
    singleton = {**sequences()[1], "train_end": 1}
    assert len(SequenceTrainingDataset([train_only, singleton], 12)) == 1
    assert list(evaluation_examples([train_only], 12)) == []


def test_targets_match_discrete_smoothed_gaussian():
    cfg = config()
    days = torch.tensor([[0., 30., 365., 500.]])
    actual = heatmap_targets(days, cfg).numpy()
    grid = np.arange(cfg.time_bins) * 730 / cfg.time_bins
    raw = np.exp(-.5 * ((grid - days.numpy()[..., None]) / 30) ** 2)
    expected = .9 * raw / (raw.sum(-1, keepdims=True) + 1e-10) + .1 / cfg.time_bins
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-6)


def test_zero_gap_time_loss_is_zero_and_finite():
    row = SequenceTrainingDataset(sequences(), 12, max_length=4)[0]
    batch = {key: value.unsqueeze(0) for key, value in row.items()}
    batch["target_days"].zero_()
    model = ReferenceITDRec(12, config(), 4)
    loss, parts = joint_loss(model(batch), batch, config())
    assert torch.isfinite(loss) and parts["time_loss"] == 0
    loss.backward()
    assert torch.isfinite(model.item_emb.weight.grad).all()


def test_decoder_attention_has_gradients_and_100_candidate_inference_works():
    cfg = config()
    row = SequenceTrainingDataset(sequences(), 120, max_length=4)[0]
    batch = {key: value.unsqueeze(0) for key, value in row.items()}
    model = ReferenceITDRec(120, cfg, 4)
    loss, _ = joint_loss(model(batch), batch, cfg)
    loss.backward()
    projection = model.time_predictor.layers[0].multihead_attn.in_proj_weight.grad
    # Q and K gradients are no longer structurally zero from singleton memory.
    dimension = cfg.item_dim + 2 * cfg.time_dim
    assert projection[:dimension].abs().sum() > 0
    assert projection[dimension:2 * dimension].abs().sum() > 0
    model.eval()
    batch["candidates"] = torch.arange(1, 101).unsqueeze(0)
    with torch.no_grad():
        full = model.predict(batch)
        compact = model.predict(batch, compact_query=True)
        assert full["scores"].shape == (1, 100)
        torch.testing.assert_close(full["time_output"], compact["time_output"], atol=1e-6, rtol=1e-5)
        # Preserve and EXPOSE the author inference degeneracy, rather than hide it.
        assert full["predicted_days"].unique().numel() == 1
        changed_labels = {**batch, "target_days": batch["target_days"] + 9999}
        torch.testing.assert_close(model.predict(changed_labels)["scores"], full["scores"])


def test_selection_uses_all_six_metrics_and_accepts_ties():
    base = {f"{metric}@{k}": .4 for metric in ("hit", "recall") for k in (5, 10, 20)}
    assert is_improvement(base, None)
    assert is_improvement(base, base)
    assert not is_improvement({**base, "hit@5": .3, "ndcg@10": 1}, base)


def test_ranking_duplicate_positive_semantics():
    # Recall counts unique retrieved IDs but divides by all positive events, as upstream.
    metrics = ranking_metrics(torch.tensor([4., 3., 2., 1.]), torch.tensor([1, 1, 2, 3]),
                              torch.tensor([1., 1., 0., 0.]))
    assert metrics["hit@5"] == 1
    assert metrics["recall@5"] == .5
    assert metrics["ndcg@5"] == pytest.approx(1)


def test_training_checkpoint_reload_and_evaluation_round_trip(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sequences.json").write_text(json.dumps(sequences()))
    (data / "items.json").write_text(json.dumps({str(i): str(i) for i in range(1, 13)}))
    (data / "stats.json").write_text(json.dumps({"items": 12, "max_users": 0,
                                                 "retain_training_only": True, "cluster_gap_days": 1}))
    settings = dict(epochs=1, batch_size=2, max_length=4, device="cpu", candidate_count=10, threads=1)
    result = train(data, tmp_path / "run", config(), settings)
    model, checkpoint = load_checkpoint(tmp_path / "run/best.pt", data_dir=data)
    repeated = evaluate(model, sequences(), 12, "cpu", "test", 4, 10, 0, 0, 2)
    assert result["test"] == repeated
    assert result["test"]["constant_time_user_fraction"] == 1
    assert checkpoint["implementation"] == "author_release_v1"
    with pytest.raises(FileExistsError):
        train(data, tmp_path / "run", config(), settings)
    (data / "items.json").write_text("{}")
    with pytest.raises(ValueError, match="Dataset mismatch"):
        load_checkpoint(tmp_path / "run/best.pt", data_dir=data)


@pytest.mark.parametrize("category", ["pet", "books", "beauty"])
def test_root_configs_route_to_reference_pipeline(monkeypatch, category):
    import itdrec.train as entry
    import itdrec.reference.train as reference
    calls = []
    monkeypatch.setattr(reference, "train", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["train", "--config", f"configs/{category}.json", "--device", "cpu"])
    entry.main()
    data_dir, output_dir, cfg, settings, provenance = calls[0]
    assert Path(data_dir) == Path(f"data/processed/{category}_reference_since")
    assert Path(output_dir) == Path(f"outputs/{category}_author_release_seed0")
    assert isinstance(cfg, ReferenceConfig)
    assert cfg.sigma_days == 30 and cfg.smoothing == .1
    assert settings["device"] == "cpu"
    assert provenance["source_commit"]
