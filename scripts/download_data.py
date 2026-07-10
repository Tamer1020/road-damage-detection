"""Resumable downloader for the RDD dataset archive.

The RDD2022 archives are large and hosted externally, so we don't hard-code a
fragile link. Pass the URL from the sekilab/RoadDamageDetector release and this
streams it to disk with a progress bar.

Usage:
    python scripts/download_data.py --url <ARCHIVE_URL> --dst data/raw/rdd2022.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


def download(url: str, dst: str, chunk_size: int = 1 << 20) -> None:
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dst_path.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst_path.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))
    print(f"Saved to {dst_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the RDD dataset archive.")
    parser.add_argument("--url", required=True, help="Direct URL to the dataset archive.")
    parser.add_argument("--dst", default="data/raw/rdd.zip", help="Output file path.")
    args = parser.parse_args()
    download(args.url, args.dst)


if __name__ == "__main__":
    main()
