import pytest
import torch

from itdrec.config import ModelConfig
from itdrec.data.dataset import TrainingDataset
from itdrec.evaluate import predict_batch
from itdrec.losses import joint_loss
from itdrec.models import ITDRec
from itdrec.models.time_heads import BinaryHead, HeatmapHead


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)
    torch.set_num_threads(1)


def config(predictor="heatmap"):
    return ModelConfig(item_dim=8, time_dim=4, heads=2, layers=1, cpm_layers=1,
                       dropout=0, predictor=predictor, time_bins=10)


def batch():
    seq = {"items": [1, 2, 3, 4], "timestamps_ms": [0, 86400000, 300000000, 600000000],
           "train_end": 2, "valid_end": 3}
    row = TrainingDataset([seq], 8, max_length=4, negatives=2)[0]
    return {key: value.unsqueeze(0) for key, value in row.items()}


@pytest.mark.parametrize("predictor", ["regression", "binary", "heatmap"])
def test_all_heads_train_and_receive_gradients(predictor):
    model, inputs = ITDRec(8, config(predictor)), batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = None
    for _ in range(15):
        output = predict_batch(model, inputs)
        assert output["scores"].shape == (1, 3)
        assert (output["predicted_days"] >= 0).all()
        loss, _ = joint_loss(model, output, inputs)
        if initial is None:
            initial = loss.item()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        for module in [model.item_embedding, model.encoder, model.cpm, model.time_head]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().sum() > 0 for g in gradients)
        optimizer.step()
    final, _ = joint_loss(model, predict_batch(model, inputs), inputs)
    assert final.item() < initial


def test_candidate_order_and_chunk_invariance():
    model, inputs = ITDRec(8, config()).eval(), batch()
    original = predict_batch(model, inputs)
    reversed_inputs = dict(inputs, candidates=inputs["candidates"].flip(-1))
    reversed_output = predict_batch(model, reversed_inputs)
    assert torch.allclose(original["time_output"], reversed_output["time_output"].flip(1), atol=1e-6)
    for i in range(3):
        single = predict_batch(model, dict(inputs, candidates=inputs["candidates"][:, i:i + 1]))
        assert torch.allclose(original["time_output"][:, i:i + 1], single["time_output"], atol=1e-6)


def test_future_labels_do_not_change_predictions_and_padding_is_inert():
    model, inputs = ITDRec(8, config()).eval(), batch()
    original = predict_batch(model, inputs)
    changed = dict(inputs, labels=torch.zeros_like(inputs["labels"]),
                   target_days=inputs["target_days"] + 10000)
    assert torch.equal(original["scores"], predict_batch(model, changed)["scores"])
    # The sequence has one real event. Padded time values must not influence it.
    times = inputs["times"].clone()
    times[:, 1:] = 9999
    assert torch.allclose(original["scores"], predict_batch(model, dict(inputs, times=times))["scores"])


def test_head_targets_and_decoding_match_meaning():
    binary = BinaryHead(4, max_days=10, time_bins=5)
    labels = binary.targets(torch.tensor([[3.0, 15.0]]))
    assert labels.tolist() == [[[0, 1, 1, 1, 1], [0, 0, 0, 0, 0]]]
    assert binary.predict_days(torch.tensor([[-1., 1, -1, -1, -1]])).item() == 4
    assert binary.predict_days(torch.full((1, 5), -1.0)).item() == 20
    heatmap = HeatmapHead(4, max_days=10, time_bins=20, sigma_days=1)
    targets = heatmap.targets(torch.tensor([[3.0, 15.0]]))
    assert heatmap.predict_days(targets).tolist() == [[3.0, 15.0]]
