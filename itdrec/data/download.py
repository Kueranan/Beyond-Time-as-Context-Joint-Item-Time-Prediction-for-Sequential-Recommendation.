"""Download the paper's three Amazon Reviews 2023 categories (pure IDs).

ตัวอย่าง: python -m itdrec.data.download --dataset pet
ไฟล์ pure IDs มี user, item, rating, timestamp ครบ โดยไม่ต้องโหลดข้อความรีวิว
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

CATEGORIES = {"pet": "Pet_Supplies", "books": "Books", "beauty": "Beauty_and_Personal_Care"}
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(dataset, output_dir=Path("data/raw")):
    """Write atomically; a failed transfer never looks like a completed dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{CATEGORIES[dataset]}.csv.gz"
    manifest = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        if sha256(destination) == info["sha256"]:
            print(f"Already verified: {destination}", flush=True)
            return destination
        raise ValueError(f"Checksum mismatch: {destination}; move the damaged file before retrying")

    url = f"{BASE_URL}/{destination.name}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "ITDRec-Research/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise IOError("Server returned an unexpected byte range")
        if not resumed:
            offset = 0  # Server ignored Range: start a complete fresh file.
        expected_remaining = int(response.headers.get("Content-Length", 0))
        expected = offset + expected_remaining if expected_remaining else 0
        print(f"  Starting at {offset / 1e6:.1f} MB; expected total {expected / 1e6:.1f} MB", flush=True)
        size, last_report = offset, time.monotonic()
        with temporary.open("ab" if resumed else "wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                size += len(block)
                if time.monotonic() - last_report > 15:
                    print(f"  {dataset}: {size / 1e6:.1f} MB", flush=True)
                    last_report = time.monotonic()
            if expected and size != expected:
                raise IOError(f"Incomplete download: {size} of {expected} bytes")
    # A header check catches HTML error pages returned with HTTP 200.
    with temporary.open("rb") as stream:
        if stream.read(2) != b"\x1f\x8b":
            raise ValueError("The server did not return a gzip file")
    temporary.replace(destination)
    manifest.write_text(json.dumps({
        "dataset": dataset, "url": url, "bytes": size, "sha256": sha256(destination),
        "source": "https://amazon-reviews-2023.github.io/data_processing/0core.html",
        "format": "deduplicated 0-core pure IDs; timestamp in milliseconds",
    }, indent=2) + "\n")
    print(f"Saved {destination} ({size / 1e6:.1f} MB)", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*CATEGORIES, "all"], default="pet")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--info", action="store_true", help="Only inspect remote file sizes")
    args = parser.parse_args()
    for name in CATEGORIES if args.dataset == "all" else [args.dataset]:
        if args.info:
            url = f"{BASE_URL}/{CATEGORIES[name]}.csv.gz"
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as response:
                print(json.dumps({"dataset": name, "url": url,
                                  "bytes": response.headers.get("Content-Length")}), flush=True)
        else:
            download(name, args.output_dir)


if __name__ == "__main__":
    main()
