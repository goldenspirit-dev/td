"""Download public CT datasets used in this project.

Usage
-----
    python scripts/download_data.py --dataset scivis   # bonsai ~16 MB
    python scripts/download_data.py --dataset kanzi    # Kanzi apple metadata CSV only
    python scripts/download_data.py --dataset kanzi --full  # full 103 GB CT ZIP
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "scivis_bonsai": {
        "url": "https://klacansky.com/open-scivis-datasets/bonsai/bonsai_256x256x256_uint8.raw",
        "dest": DATA_RAW / "scivis" / "bonsai_256x256x256_uint8.raw",
        "sidecar": {"shape": [256, 256, 256], "dtype": "uint8"},
        "size_mb": 16,
        "description": "Open SciVis bonsai (256³ uint8, ~16 MB)",
    },
    "kanzi_scores": {
        "url": "https://zenodo.org/records/10515639/files/browning_scores.csv",
        "dest": DATA_RAW / "kanzi_apples" / "browning_scores.csv",
        "sidecar": None,
        "size_mb": 0.1,
        "description": "Kanzi apple browning scores CSV (metadata only)",
    },
    "kanzi_full": {
        "url": "https://zenodo.org/records/10515639/files/KanziAppleCT.zip",
        "dest": DATA_RAW / "kanzi_apples" / "KanziAppleCT.zip",
        "sidecar": None,
        "size_mb": 103_000,
        "description": "Kanzi apple CT volumes full ZIP (~103 GB) — slow!",
    },
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
    else:
        print(f"\r  Downloaded {downloaded / 1_048_576:.1f} MB …", end="", flush=True)


def download(key: str) -> None:
    entry = DATASETS[key]
    dest: Path = entry["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Already present: {dest.relative_to(ROOT)}")
    else:
        print(f"Downloading {entry['description']}  ->  {dest.relative_to(ROOT)}")
        urllib.request.urlretrieve(entry["url"], dest, reporthook=_progress_hook)
        print()  # newline after progress bar
        print(f"Saved {dest.stat().st_size / 1_048_576:.1f} MB")

    # Write JSON sidecar for raw binary files
    sidecar_data = entry.get("sidecar")
    if sidecar_data:
        sidecar = dest.with_suffix(".json")
        if not sidecar.exists():
            sidecar.write_text(json.dumps(sidecar_data, indent=2))
            print(f"Wrote sidecar: {sidecar.name}")
        else:
            print(f"Sidecar exists: {sidecar.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CT datasets for tda-fruit-ct.")
    parser.add_argument(
        "--dataset",
        choices=["scivis", "kanzi"],
        required=True,
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="For --dataset kanzi: download full 103 GB CT ZIP (slow).",
    )
    args = parser.parse_args()

    if args.dataset == "scivis":
        download("scivis_bonsai")
    elif args.dataset == "kanzi":
        download("kanzi_scores")
        if args.full:
            download("kanzi_full")
        else:
            print(
                "\nNote: only the browning-scores CSV was downloaded.\n"
                "Add --full to download the ~103 GB CT volume ZIP."
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
