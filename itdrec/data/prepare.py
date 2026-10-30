"""Clean Amazon pure IDs and split each user by natural temporal clusters.

อ่าน gzip ทีละแถวและใช้ SQLite ช่วยกรองข้อมูล เพื่อลดการใช้ RAM
"""

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .download import CATEGORIES, sha256

DAY_MS = 86_400_000
DEFAULTS = {"pet": (2021, 10), "books": (2020, 15), "beauty": (2020, 15)}


def cluster_boundaries(timestamps, gap_days=1.0):
    """Return cluster start indices; exactly one day stays in the same cluster."""
    return [0] + [i for i in range(1, len(timestamps))
                  if timestamps[i] - timestamps[i - 1] > gap_days * DAY_MS]


def split_sequence(items, timestamps, gap_days=1.0):
    """Last cluster=test, preceding cluster=validation, earlier clusters=train."""
    starts = cluster_boundaries(timestamps, gap_days)
    if len(starts) < 3 or starts[-2] < 2:
        return None  # Need at least one history→target pair for training.
    return {"items": items, "timestamps_ms": timestamps,
            "train_end": starts[-2], "valid_end": starts[-1]}


def prepare(dataset, source, output, year=None, year_mode="single", min_interactions=None,
            max_users=0, seed=0, cluster_gap_days=1.0):
    if max_users < 0 or cluster_gap_days <= 0:
        raise ValueError("max_users must be >= 0 and cluster_gap_days must be > 0")
    year = DEFAULTS[dataset][0] if year is None else year
    minimum = DEFAULTS[dataset][1] if min_interactions is None else min_interactions
    if minimum < 4:
        raise ValueError("min_interactions must be >= 4")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows_read = 0

    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output) as temp:
        connection = sqlite3.connect(str(Path(temp) / "interactions.sqlite"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-131072")  # 128 MiB bounds random index I/O.
            connection.execute("CREATE TABLE events (user TEXT, item TEXT, time INTEGER, UNIQUE(user,item,time))")
            pending = []
            opener = gzip.open if str(source).endswith(".gz") else open
            with opener(source, "rt", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"user_id", "parent_asin", "timestamp"} <= set(reader.fieldnames or []):
                    raise ValueError("Expected Amazon pure-ID CSV: user_id,parent_asin,rating,timestamp")
                for row in reader:
                    rows_read += 1
                    if rows_read % 1_000_000 == 0:
                        print(f"{dataset}: scanned {rows_read:,} rows", flush=True)
                    timestamp = int(row["timestamp"])
                    if timestamp < start or (year_mode == "single" and timestamp >= end):
                        continue
                    pending.append((row["user_id"], row["parent_asin"], timestamp))
                    if len(pending) >= 20_000:
                        connection.executemany("INSERT OR IGNORE INTO events VALUES (?,?,?)", pending)
                        pending.clear()
            connection.executemany("INSERT OR IGNORE INTO events VALUES (?,?,?)", pending)
            connection.commit()
            qualified = [row[0] for row in connection.execute(
                "SELECT user FROM events GROUP BY user HAVING COUNT(*) >= ?", (minimum,))]
            # Sampling whole users preserves each selected user's complete history.
            qualified.sort(key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest())
            selected = qualified[:max_users] if max_users else qualified
            connection.execute("CREATE TABLE selected (user TEXT PRIMARY KEY)")
            connection.executemany("INSERT INTO selected VALUES (?)", [(u,) for u in selected])
            cursor = connection.execute(
                "SELECT e.user,e.item,e.time FROM events e JOIN selected s ON e.user=s.user "
                "ORDER BY e.user,e.time,e.item")
            sequences, items, timestamps, previous = [], [], [], None

            def append_user():
                if previous is not None:
                    sequence = split_sequence(items, timestamps, cluster_gap_days)
                    if sequence is not None:
                        sequences.append({"user_id": previous, **sequence})

            for user, item, timestamp in cursor:
                if user != previous:
                    append_user()
                    items, timestamps, previous = [], [], user
                items.append(item)
                timestamps.append(timestamp)
            append_user()
        finally:
            connection.close()

    if not sequences:
        raise ValueError("No eligible sequences. Check year/min_interactions or increase max_users.")
    # Transductive catalog: IDs are known; held-out interactions are never training examples.
    catalog = sorted({item for seq in sequences for item in seq["items"]})
    item_to_id = {item: i + 1 for i, item in enumerate(catalog)}  # 0 = padding
    for seq in sequences:
        seq["items"] = [item_to_id[item] for item in seq["items"]]
    train_items = {item for seq in sequences for item in seq["items"][:seq["train_end"]]}
    held_out = [item for seq in sequences for item in seq["items"][seq["train_end"]:]]
    stats = {
        "dataset": dataset, "category": CATEGORIES[dataset], "source": str(source),
        "source_sha256": sha256(source), "rows_scanned": rows_read,
        "year": year, "year_mode": year_mode, "min_interactions": minimum,
        "max_users": max_users, "seed": seed, "cluster_gap_days": cluster_gap_days,
        "qualified_users_before_cluster_filter": len(qualified),
        "users": len(sequences), "items": len(catalog),
        "interactions": sum(len(seq["items"]) for seq in sequences),
        "train_examples": sum(seq["train_end"] - 1 for seq in sequences),
        "validation_targets": sum(seq["valid_end"] - seq["train_end"] for seq in sequences),
        "test_targets": sum(len(seq["items"]) - seq["valid_end"] for seq in sequences),
        "held_out_item_not_in_train_fraction": sum(i not in train_items for i in held_out) / len(held_out),
        "catalog_policy": "transductive IDs from all retained splits; no held-out targets in training",
    }
    (output / "sequences.json").write_text(json.dumps(sequences) + "\n")
    (output / "items.json").write_text(json.dumps({str(v): k for k, v in item_to_id.items()}) + "\n")
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CATEGORIES, default="pet")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--year", type=int)
    parser.add_argument("--year-mode", choices=["single", "since"], default="single")
    parser.add_argument("--min-interactions", type=int)
    parser.add_argument("--max-users", type=int, default=0, help="0 = all qualifying users")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-gap-days", type=float, default=1.0)
    args = parser.parse_args()
    prepare(args.dataset, args.input or Path(f"data/raw/{CATEGORIES[args.dataset]}.csv.gz"),
            args.output or Path(f"data/processed/{args.dataset}"), args.year, args.year_mode,
            args.min_interactions, args.max_users, args.seed, args.cluster_gap_days)


if __name__ == "__main__":
    main()
