"""Numerically compare the audited implementation against a local pinned author checkout.

python -m itdrec.reference.check_parity --source-dir /path/to/ITDRec
Only use a trusted checkout: this command imports its model and target-builder code.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import SOURCE_COMMIT
from .loss import heatmap_targets, joint_loss
from .model import ReferenceConfig, ReferenceITDRec


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(source_dir):
    source_dir = Path(source_dir)
    revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"Expected {SOURCE_COMMIT}, got {revision}")
    for name in ("models/ITDRec.py", "utils/utils.py"):
        committed = subprocess.check_output(["git", "-C", str(source_dir), "show", f"HEAD:{name}"])
        if (source_dir / name).read_bytes() != committed:
            raise ValueError(f"Source checkout has modified file: {name}")
    upstream = import_file(source_dir / "models/ITDRec.py", "author_model")
    utils = import_file(source_dir / "utils/utils.py", "author_utils")
    torch.set_num_threads(1)
    torch.manual_seed(12)
    cfg = ReferenceConfig(item_dim=8, time_dim=4, heads=2, layers=2, dropout=0, time_bins=30)
    args = types.SimpleNamespace(device="cpu", dropout=0, node_feat_dim=8, time_feat_dim=4,
                                 num_layers=2, num_heads=2, time_predictor_type="heatmap",
                                 time_bin=30, max_input_sequence_length=4)
    author = upstream.ITDRec(2, 120, args).eval()
    ours = ReferenceITDRec(120, cfg, max_length=4).eval()
    ours.load_state_dict(author.state_dict(), strict=True)
    sequence = np.array([[[0, 0, 0], [1, 86400, 0], [2, 259200, 2], [3, 345600, 3]],
                         [[4, 86400, 0], [5, 172800, 0], [6, 432000, 3], [7, 604800, 2]]], dtype=np.float64)
    positive = np.zeros_like(sequence)
    positive[:, :, 0] = [[0, 2, 3, 8], [5, 6, 7, 9]]
    positive[:, :, 2] = [[0, 2, 3, 5], [0, 3, 2, 4]]
    negative = np.zeros_like(sequence)
    negative[:, :, 0] = [[0, 10, 11, 12], [10, 11, 12, 13]]
    origins = np.array([86400, 86400])
    batch = dict(items=torch.tensor(sequence[:, :, 0], dtype=torch.long),
                 absolute_days=torch.tensor((sequence[:, :, 1] - origins[:, None]) / 86400, dtype=torch.float32),
                 relative_days=torch.tensor(sequence[:, :, 2], dtype=torch.float32),
                 positive=torch.tensor(positive[:, :, 0], dtype=torch.long),
                 negative=torch.tensor(negative[:, :, 0], dtype=torch.long),
                 target_days=torch.tensor(positive[:, :, 2], dtype=torch.float32))
    report = dict(source_commit=revision, source_sha256={name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                                       for name in ("models/ITDRec.py", "utils/utils.py")})
    checks = {}

    def compare(name, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        checks[name] = float((actual - expected).detach().abs().max())

    compare("encoder", ours.encode(batch["items"], batch["absolute_days"], batch["relative_days"]), author.log2feats(sequence))
    expected = author(None, sequence, positive, negative)
    actual = ours(batch)
    keys = ("positive_logits", "negative_logits", "positive_time", "negative_time")
    for key, value in zip(keys, expected):
        compare(key, actual[key], value)
    labels = batch["target_days"]
    author_targets = utils.convert_time(labels, "heatmap", cfg.time_bins)
    compare("heatmap_targets", heatmap_targets(labels, cfg), author_targets)
    valid = batch["positive"].ne(0)
    temporal = valid & labels.ne(0)
    expected_loss = sum(F.binary_cross_entropy_with_logits(logits[valid] * torch.exp(-cfg.beta * days[valid]),
                                                         torch.full_like(logits[valid], label))
                        for logits, days, label in ((expected[0], labels, 1.0), (expected[1], labels + 365, 0.0)))
    expected_loss += cfg.alpha * (F.mse_loss(expected[2][temporal], author_targets[temporal]) +
                                  F.mse_loss(expected[3][temporal], utils.convert_time(labels + 365, "heatmap", cfg.time_bins)[temporal]))
    actual_loss, _ = joint_loss(actual, batch, cfg)
    compare("joint_loss", actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    gradients = []
    for (name, param), (other_name, other_param) in zip(ours.named_parameters(), author.named_parameters()):
        assert name == other_name
        if param.grad is not None:
            compare("gradient:" + name, param.grad, other_param.grad)
            gradients.append(name)
    # Establish the original 100-candidate mask failure without hiding the patch.
    candidates = np.zeros((2, 100, 3))
    candidates[:, :, 0] = np.arange(1, 101)
    try:
        author.predict(None, sequence, candidates, cfg.beta)
    except RuntimeError as exc:
        report["original_predict_100_error"] = str(exc)
    else:
        raise AssertionError("Expected upstream history-length mask mismatch")
    # Patch only the inference target-mask size in a temporary in-memory module.
    source = (source_dir / "models/ITDRec.py").read_text()
    old = "final_feats_mask = torch.triu(torch.ones(log_feats.shape[1], log_feats.shape[1]), diagonal=1).bool()"
    assert source.count(old) == 1
    patched = types.ModuleType("author_mask_fixed")
    exec(compile(source.replace(old, "final_feats_mask = torch.triu(torch.ones(item_embs.shape[1], item_embs.shape[1]), diagonal=1).bool()"),
                 str(source_dir / "models/ITDRec.py"), "exec"), patched.__dict__)
    fixed = patched.ITDRec(2, 120, args).eval()
    fixed.load_state_dict(author.state_dict())
    prediction_batch = {**batch, "candidates": torch.tensor(candidates[:, :, 0], dtype=torch.long)}
    with torch.no_grad():
        ids, days = fixed.predict(None, sequence, candidates, cfg.beta)
        prediction = ours.predict(prediction_batch)
        order = prediction["scores"].topk(100).indices
        compare("predict_100_ranked_items", prediction_batch["candidates"].gather(1, order).float(), torch.as_tensor(ids).float())
        compare("predict_100_days", prediction["predicted_days"].gather(1, order), days)
    report.update(passed=True, checks=checks, gradients_compared=len(gradients),
                  constant_candidate_times=[bool(row.max() == row.min()) for row in prediction["predicted_days"]],
                  inference_patch="Only target-mask dimension: history length -> candidate count")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_audit/parity.json"))
    args = parser.parse_args()
    report = check(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))
    print("max_absolute_error:", max(report["checks"].values()))


if __name__ == "__main__":
    main()
