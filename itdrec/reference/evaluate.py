"""Evaluate the author-release model and expose whether temporal reranking works.

Ranking follows the release's item-ID membership semantics, including repeated
positives. Both release-style batch means and user means are recorded. Time MAE
uses correctly aligned item targets, not the release's rank/label-order mismatch.
"""

import math

import torch

from .data import evaluation_examples


def ranking_metrics(scores, candidates, labels):
    predicted = candidates[scores.topk(scores.numel()).indices].tolist()
    targets = candidates[labels.bool()].tolist()
    target_set = set(targets)
    metrics = {}
    for k in (5, 10, 20):
        top = predicted[:k]
        hits = len(set(top) & target_set)
        dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(top) if item in target_set)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(targets))))
        metrics.update({f"hit@{k}": float(hits > 0), f"recall@{k}": hits / len(targets),
                        f"ndcg@{k}": dcg / ideal})
    return metrics


def evaluation_batches(examples, batch_size):
    pending = []
    for example in examples:
        if pending and (len(pending) == batch_size or
                        pending[0]["candidates"].shape != example["candidates"].shape):
            yield {key: torch.cat([row[key] for row in pending]) for key in pending[0]}
            pending = []
        pending.append(example)
    if pending:
        yield {key: torch.cat([row[key] for row in pending]) for key in pending[0]}


@torch.no_grad()
def evaluate(model, sequences, num_items, device, split="valid", max_length=32,
             candidate_count=100, seed=0, max_users=0, batch_size=200):
    model.eval()
    examples = evaluation_examples(sequences, num_items, split, max_length, candidate_count, seed, max_users)
    rows, batch_means = [], []
    absolute_error = squared_error = clipped_error = events = constant = changed = 0
    spans = []
    for batch in evaluation_batches(examples, batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model.predict(batch)
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise FloatingPointError("Non-finite reference inference output")
        group = []
        for i in range(len(batch["items"])):
            row = {}
            for prefix, key in (("", "scores"), ("raw_", "item_logits")):
                row.update({prefix + key: value for key, value in ranking_metrics(
                    output[key][i], batch["candidates"][i], batch["labels"][i]).items()})
            rows.append(row)
            group.append(row)
            span = (output["predicted_days"][i].max() - output["predicted_days"][i].min()).item()
            spans.append(span)
            constant += span == 0
            k = min(10, batch["candidates"].shape[1])
            changed += not torch.equal(output["scores"][i].topk(k).indices,
                                       output["item_logits"][i].topk(k).indices)
        batch_means.append({key: sum(row[key] for row in group) / len(group) for key in group[0]})
        mask = batch["labels"].bool()
        predicted, target = output["predicted_days"][mask], batch["target_days"][mask]
        error = predicted - target
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        clipped_error += (predicted.clamp(0, model.config.max_days) -
                          target.clamp(0, model.config.max_days)).abs().sum().item()
        events += target.numel()
    if not rows:
        raise ValueError("No users with an eligible validation/test cluster")
    # Upstream averages batch metrics equally; expose the unbiased user average too.
    result = {key: sum(batch[key] for batch in batch_means) / len(batch_means) for key in rows[0]}
    result.update({"user_mean_" + key: sum(row[key] for row in rows) / len(rows) for key in rows[0]})
    result.update(time_mae_days=absolute_error / events, time_rmse_days=math.sqrt(squared_error / events),
                  time_mae_clipped_days=clipped_error / events, users_evaluated=len(rows),
                  positive_events=events, constant_time_user_fraction=constant / len(rows),
                  reranked_top10_changed_fraction=changed / len(rows),
                  mean_candidate_time_span_days=sum(spans) / len(spans), split=split,
                  ranking_aggregation="mean of batch means (author release)",
                  time_metric_condition="aligned positive items, no mean-time substitution")
    return result
