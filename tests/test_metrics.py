import math

import pytest
import torch

from itdrec.metrics import ranking_metrics


def test_multi_positive_metrics():
    result = ranking_metrics(torch.tensor([4., 3, 2, 1]), torch.tensor([1., 0, 1, 0]), cutoffs=(1, 3))
    assert result["hit@1"] == 1
    assert result["recall@1"] == 0.5
    assert result["recall@3"] == 1
    assert result["ndcg@3"] == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
