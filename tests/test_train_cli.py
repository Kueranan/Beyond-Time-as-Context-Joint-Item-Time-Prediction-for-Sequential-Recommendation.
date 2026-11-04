"""Selecting a category config must select that category's data and outputs."""

from pathlib import Path

import pytest

import itdrec.train as training


@pytest.mark.parametrize("category", ["pet", "books", "beauty"])
def test_config_selects_its_dataset_and_gpu(monkeypatch, category):
    calls = []
    monkeypatch.setattr(training, "train", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["train", "--config", f"configs/legacy/{category}.json"])
    training.main()
    data_dir, output_dir, model, settings = calls[0]
    assert data_dir == Path(f"data/processed/{category}")
    assert output_dir == Path(f"outputs/{category}_paper_settings_seed0")
    assert settings["device"] == "mps"
    assert settings["epochs"] == 100
    assert model.predictor == "heatmap"


def test_cli_can_override_config_paths_and_device(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(training, "train", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", [
        "train", "--config", "configs/legacy/books.json", "--data-dir", str(tmp_path / "data"),
        "--output-dir", str(tmp_path / "run"), "--device", "cpu", "--epochs", "1",
    ])
    training.main()
    data_dir, output_dir, _, settings = calls[0]
    assert data_dir == tmp_path / "data"
    assert output_dir == tmp_path / "run"
    assert settings["device"] == "cpu"
    assert settings["epochs"] == 1


def test_legacy_smoke_config_keeps_default_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "train", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["train", "--config", "configs/smoke.json"])
    training.main()
    data_dir, output_dir, _, settings = calls[0]
    assert data_dir == Path("data/processed/pet")
    assert output_dir == Path("outputs/pet_heatmap")
    assert settings["device"] == "cpu"
