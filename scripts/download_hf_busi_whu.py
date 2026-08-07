"""Download the labeled BUSI-WHU segmentation release from Hugging Face."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import requests


REPO = "huangjin520/busi-whu-seg"


def _download(item: dict[str, object], output: Path) -> None:
    relative = str(item["path"])
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == int(item.get("size", 0)):
        return
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{relative}?download=true"
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download labeled BUSI-WHU from Hugging Face.")
    parser.add_argument("--output", default="data/external/busi_whu/label_reference_hf")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    listing = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true", timeout=120)
    listing.raise_for_status()
    items = [item for item in listing.json() if item.get("type") == "file" and "/images/" in str(item.get("path")) or item.get("type") == "file" and "/masks/" in str(item.get("path"))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_download, item, output) for item in items]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"downloaded={len(items)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
