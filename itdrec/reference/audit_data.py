"""Compare prepared data with Table I; counts alone cannot prove identical splits."""

import argparse
import json
from pathlib import Path

PAPER_COUNTS = {
    "pet": {"users": 23904, "items": 126988, "interactions": 344667},
    "books": {"users": 13236, "items": 196868, "interactions": 363143},
    "beauty": {"users": 21837, "items": 184362, "interactions": 472091},
}


def compare_to_paper(stats):
    expected = PAPER_COUNTS.get(stats.get("dataset"))
    if expected is None:
        return {"counts_match": False, "reason": "Dataset not listed in Table I"}
    actual = {key: stats[key] for key in expected}
    return {"paper": expected, "actual": actual,
            "difference": {key: actual[key] - expected[key] for key in expected},
            "counts_match": actual == expected,
            "identical_records_and_splits_verified": False,
            "reason": "Author preprocessing script and processed CSVs are absent from the pinned release"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_audit/datasets.json"))
    args = parser.parse_args()
    report = {}
    for dataset in PAPER_COUNTS:
        config = json.loads(Path(f"configs/{dataset}.json").read_text())
        directory = Path(config["data_dir"])
        stats = json.loads((directory / "stats.json").read_text())
        sequences = json.loads((directory / "sequences.json").read_text())
        mapping = json.loads((directory / "items.json").read_text())
        assert stats["users"] == len(sequences)
        assert stats["items"] == len(mapping)
        assert stats["interactions"] == sum(len(seq["items"]) for seq in sequences)
        for seq in sequences:
            assert 0 < seq["train_end"] <= seq["valid_end"] <= len(seq["items"])
            assert seq["timestamps_ms"] == sorted(seq["timestamps_ms"])
        report[dataset] = {"data_dir": str(directory), **compare_to_paper(stats),
                           "training_users": sum(seq["train_end"] >= 2 for seq in sequences),
                           "evaluation_users": sum(seq["train_end"] < seq["valid_end"] for seq in sequences)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
