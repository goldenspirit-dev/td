"""Download script for TDA fruit CT datasets.

Usage:
    python src/download_data.py --dataset scivis          # ~16 MB, instant start
    python src/download_data.py --dataset kanzi           # metadata only (<1 MB)
    python src/download_data.py --dataset kanzi --full    # 103 GB - prompts for confirmation
    python src/download_data.py --dataset all             # scivis + kanzi metadata
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies: pip install requests tqdm")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = "8167285"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Open SciVis Datasets - bonsai tree 256x256x256 (~16 MB), uint8 raw binary
_SCIVIS_MIRRORS = [
    "https://cdn.klacansky.com/open-scivis-datasets/bonsai/bonsai_256x256x256_uint8.raw",
    "http://klacansky.com/open-scivis-datasets/bonsai/bonsai_256x256x256_uint8.raw",
]
SCIVIS_BONSAI_META = {
    "name": "bonsai_256x256x256_uint8.raw",
    "shape": [256, 256, 256],
    "dtype": "uint8",
    "axes": "ZYX",
    "description": "Bonsai tree micro-CT (Open SciVis Datasets)",
    "source": _SCIVIS_MIRRORS[0],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

class _TqdmHook(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download(url: str, dest: Path, *, skip_existing: bool = True) -> bool:
    """Download *url* to *dest* with a tqdm progress bar.

    Returns True if the file was actually downloaded, False if skipped.
    """
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} (already exists)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {dest.relative_to(PROJECT_ROOT)}")
    with _TqdmHook(unit="B", unit_scale=True, unit_divisor=1024,
                   miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    return True


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024**2:
        return f"{n_bytes/1024:.1f} KB"
    if n_bytes < 1024**3:
        return f"{n_bytes/1024**2:.1f} MB"
    return f"{n_bytes/1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# Zenodo / Kanzi
# ---------------------------------------------------------------------------

def fetch_zenodo_metadata() -> dict:
    """Fetch Zenodo record metadata via the public REST API."""
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_zenodo_files(metadata: dict) -> list[dict]:
    """Return dicts with keys: name, size_bytes, url."""
    out = []
    for f in metadata.get("files", []):
        out.append({
            "name": f["key"],
            "size_bytes": f["size"],
            "url": f["links"]["self"],
        })
    return out


def download_kanzi(
    *,
    full: bool = False,
    out_dir: Optional[Path] = None,
) -> None:
    """Download the Kanzi apple CT dataset from Zenodo record 8167285.

    Default: downloads only small metadata files (<10 MB).
    Pass *full=True* to download the 103 GB CT reconstruction ZIP (with prompt).
    """
    out_dir = out_dir or RAW_DIR / "kanzi_apples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching Zenodo record {ZENODO_RECORD_ID} metadata...")
    try:
        meta = fetch_zenodo_metadata()
    except Exception as exc:
        print(f"  ERROR: could not reach Zenodo API - {exc}")
        print("  Check your internet connection or try again later.")
        return

    title = meta.get("metadata", {}).get("title", "(untitled)")
    files = list_zenodo_files(meta)

    print(f"\n  Record: {title}")
    print(f"  {'File':<45} {'Size':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for f in files:
        print(f"  {f['name']:<45} {_fmt_size(f['size_bytes']):>10}")

    # Always grab small files (metadata)
    small = [f for f in files if f["size_bytes"] < 10 * 1024**2]
    large = [f for f in files if f["size_bytes"] >= 10 * 1024**2]

    if small:
        print(f"\nDownloading {len(small)} small metadata file(s)...")
        for f in small:
            _download(f["url"], out_dir / f["name"])

    if large and not full:
        total = sum(f["size_bytes"] for f in large)
        print(f"\n  [!] {len(large)} large CT file(s) skipped ({_fmt_size(total)} total).")
        print("     To download them, re-run with: python src/download_data.py --dataset kanzi --full")
        print("     NOTE: The CT ZIP is ~103 GB. You will be prompted to confirm.")
    elif full and large:
        total = sum(f["size_bytes"] for f in large)
        answer = input(
            f"\n  This will download ~{_fmt_size(total)} of CT data.\n"
            "  Proceed? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("  Aborted - no large files downloaded.")
            return
        for f in large:
            _download(f["url"], out_dir / f["name"])

    print(f"\nKanzi files: {out_dir}")


# ---------------------------------------------------------------------------
# Open SciVis — Bonsai
# ---------------------------------------------------------------------------

def download_scivis_bonsai(out_dir: Optional[Path] = None) -> Optional[Path]:
    """Download the Bonsai 256x256x256 CT scan (~16 MB) from Open SciVis Datasets.

    This tiny dataset is used as a sanity-check volume when Kanzi apple
    data is not yet downloaded.  Returns the path to the .raw file, or
    None if the download failed.
    """
    out_dir = out_dir or RAW_DIR / "scivis"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / SCIVIS_BONSAI_META["name"]

    print("\nDownloading Open SciVis - Bonsai (256x256x256, uint8, ~16 MB)...")

    downloaded = False
    for mirror in _SCIVIS_MIRRORS:
        try:
            downloaded = _download(mirror, dest)
            break
        except (urllib.error.URLError, OSError) as exc:
            print(f"  Mirror {mirror} failed: {exc}")

    if not downloaded and not dest.exists():
        print(
            "  Could not download bonsai from any mirror.\n"
            "  If klacansky.com is temporarily down, try again later or download\n"
            "  manually and place the .raw file in data/raw/scivis/."
        )
        return None

    # Write JSON sidecar so data_loader knows shape/dtype
    meta_path = dest.with_suffix(".json")
    if not meta_path.exists():
        meta_path.write_text(json.dumps(SCIVIS_BONSAI_META, indent=2))

    print(f"  Bonsai saved: {dest.relative_to(PROJECT_ROOT)}")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download TDA fruit CT datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=["kanzi", "scivis", "all"],
        default="scivis",
        help="Which dataset to fetch (default: scivis)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Also download the 103 GB Kanzi CT ZIP (requires confirmation)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the output directory (default: data/raw/<dataset>/)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.dataset in ("kanzi", "all"):
        download_kanzi(full=args.full, out_dir=args.out_dir)

    if args.dataset in ("scivis", "all"):
        download_scivis_bonsai(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
