"""Reduce the Kanzi apple CT dataset to a shareable subset.

Strategy
--------
1. Download metadata only (browning_scores.csv + label image) if not present — ~1 MB.
2. Stratify the 120 apples into 3 browning groups (healthy / mid / severe).
3. Pick N apples per group (default: 5) → 15 apples total.
4. Download only those apples from the Zenodo ZIP using HTTP Range requests
   (no need to pull the full 103 GB).
5. Sub-sample slices along Z (default: every 4th slice).
6. Save each apple as a compressed .npz in data/interim/reduced/.
7. Print a manifest (CSV) + final size estimate.

Usage
-----
    # Dry-run: show which apples would be selected, download nothing
    python src/reduce_dataset.py --dry-run

    # Default: 5 apples per browning group, every 4th slice
    python src/reduce_dataset.py

    # Custom: 3 apples per group, every 2nd slice
    python src/reduce_dataset.py --n-per-group 3 --slice-step 2

    # Only metadata step (no CT download)
    python src/reduce_dataset.py --metadata-only
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZENODO_RECORD_ID = "8167285"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kanzi_apples"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "reduced"
MANIFEST_PATH = INTERIM_DIR / "manifest.csv"

# Browning score thresholds (0–5 scale in the Kanzi dataset)
# Adjust if the actual CSV uses a different scale.
SCORE_COL = "browning_score"   # column name in browning_scores.csv
ID_COL = "apple_id"            # column name for the apple identifier

HEALTHY_MAX = 1.5
SEVERE_MIN = 3.5
# mid = everything in between


# ---------------------------------------------------------------------------
# Zenodo helpers
# ---------------------------------------------------------------------------

def fetch_zenodo_record() -> dict:
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_zenodo_files(record: dict) -> list[dict]:
    """Return list of {name, size_bytes, url} for every file in the record."""
    return [
        {
            "name": f["key"],
            "size_bytes": f["size"],
            "url": f["links"]["self"],
        }
        for f in record.get("files", [])
    ]


def _fmt_size(n: int) -> str:
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


# ---------------------------------------------------------------------------
# Metadata download
# ---------------------------------------------------------------------------

def ensure_metadata(out_dir: Path = RAW_DIR) -> Path:
    """Download the browning_scores.csv (and other small files) if missing.

    Returns the path to browning_scores.csv.
    """
    csv_candidates = list(out_dir.glob("browning*.csv"))
    if csv_candidates:
        print(f"[metadata] Already present: {csv_candidates[0]}")
        return csv_candidates[0]

    print("[metadata] Fetching Zenodo record …")
    record = fetch_zenodo_record()
    files = list_zenodo_files(record)

    small_files = [f for f in files if f["size_bytes"] < 10 * 1024 ** 2]
    if not small_files:
        raise RuntimeError("No small metadata files found in Zenodo record.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in small_files:
        dest = out_dir / f["name"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {f['name']}")
            continue
        print(f"  Downloading {f['name']} ({_fmt_size(f['size_bytes'])}) …")
        r = requests.get(f["url"], stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)

    csv_candidates = list(out_dir.glob("browning*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"browning_scores.csv not found in {out_dir} after download.\n"
            "The Zenodo record might use a different file name. "
            f"Files present: {[f['name'] for f in small_files]}"
        )
    return csv_candidates[0]


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------

def load_browning_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names (lowercase, strip spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect score column
    score_col = next(
        (c for c in df.columns if "browning" in c or "score" in c),
        None,
    )
    id_col = next(
        (c for c in df.columns if "id" in c or "apple" in c or "sample" in c),
        df.columns[0],
    )

    if score_col is None:
        raise ValueError(
            f"Cannot find a browning/score column in {csv_path}.\n"
            f"Columns found: {list(df.columns)}"
        )

    df = df.rename(columns={score_col: SCORE_COL, id_col: ID_COL})
    print(f"[scores] Loaded {len(df)} apples — "
          f"score range [{df[SCORE_COL].min():.2f}, {df[SCORE_COL].max():.2f}]")
    return df


def stratified_sample(df: pd.DataFrame, n_per_group: int, seed: int = 42) -> pd.DataFrame:
    """Pick n_per_group apples from each browning stratum."""
    rng = np.random.default_rng(seed)

    healthy = df[df[SCORE_COL] <= HEALTHY_MAX]
    severe = df[df[SCORE_COL] >= SEVERE_MIN]
    mid = df[(df[SCORE_COL] > HEALTHY_MAX) & (df[SCORE_COL] < SEVERE_MIN)]

    def _pick(group: pd.DataFrame, label: str) -> pd.DataFrame:
        n = min(n_per_group, len(group))
        if n < n_per_group:
            print(f"  [warn] Only {len(group)} apples in '{label}' group; taking all {n}.")
        idx = rng.choice(len(group), size=n, replace=False)
        sub = group.iloc[idx].copy()
        sub["stratum"] = label
        return sub

    selected = pd.concat([
        _pick(healthy, "healthy"),
        _pick(mid, "mid"),
        _pick(severe, "severe"),
    ], ignore_index=True)

    print(f"[sample] Selected {len(selected)} apples "
          f"({n_per_group} per group × 3 strata)")
    return selected


# ---------------------------------------------------------------------------
# ZIP range-request helpers
# ---------------------------------------------------------------------------
# The Kanzi CT data is distributed as a single big ZIP on Zenodo.
# Instead of downloading all 103 GB, we:
#   1. Fetch the ZIP End-Of-Central-Directory (EOCD) from the tail of the file.
#   2. Parse the Central Directory to find byte offsets of individual entries.
#   3. Use HTTP Range requests to grab only those entries.
#
# This works because Zenodo serves static files with Accept-Ranges: bytes.

_EOCD_SIGNATURE = b"PK\x05\x06"
_CD_SIGNATURE = b"PK\x01\x02"
_LF_SIGNATURE = b"PK\x03\x04"

# Max comment length in ZIP spec is 65535 bytes; search last 65536+22 bytes.
_EOCD_SEARCH_SIZE = 65536 + 22


def _range_get(url: str, start: int, end: int, *, retries: int = 6) -> bytes:
    """HTTP GET with Range header, with exponential backoff on 429/5xx."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code in (200, 206):
            return r.content
        if r.status_code == 429:
            wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80, 160s
            print(f"\n  [rate-limit] 429 — waiting {wait}s before retry {attempt+1}/{retries} …")
            time.sleep(wait)
        elif r.status_code >= 500:
            wait = 2 ** attempt * 3
            print(f"\n  [server error] {r.status_code} — waiting {wait}s …")
            time.sleep(wait)
        else:
            raise RuntimeError(f"Range request failed: {r.status_code} for {url}")
    raise RuntimeError(f"Range request failed after {retries} retries (last status: {r.status_code})")


def _find_eocd(url: str, file_size: int) -> dict:
    """Locate and parse the End-Of-Central-Directory record.

    Handles both standard ZIP and ZIP64 (EOCD64 locator + EOCD64).
    """
    search_start = max(0, file_size - _EOCD_SEARCH_SIZE)
    tail = _range_get(url, search_start, file_size - 1)
    tail_offset = search_start  # byte offset of tail[0] in the file

    # ---- Try ZIP64 first (EOCD64 locator signature PK\x06\x07) ----
    ZIP64_LOCATOR_SIG = b"PK\x06\x07"
    ZIP64_EOCD_SIG    = b"PK\x06\x06"

    loc_pos = tail.rfind(ZIP64_LOCATOR_SIG)
    if loc_pos != -1:
        # Locator is 20 bytes: sig(4) + disk_with_eocd64(4) + eocd64_offset(8) + total_disks(4)
        eocd64_offset = struct.unpack_from("<Q", tail, loc_pos + 8)[0]
        eocd64_raw = _range_get(url, eocd64_offset, eocd64_offset + 55)
        if eocd64_raw[:4] == ZIP64_EOCD_SIG:
            # EOCD64: sig(4) + size_of_eocd64(8) + ... + cd_size(8) + cd_offset(8)
            cd_size   = struct.unpack_from("<Q", eocd64_raw, 40)[0]
            cd_offset = struct.unpack_from("<Q", eocd64_raw, 48)[0]
            print(f"  [zip64] EOCD64 found — CD offset={cd_offset}, size={cd_size}")
            return {"cd_offset": cd_offset, "cd_size": cd_size}

    # ---- Fallback: standard EOCD ----
    pos = tail.rfind(_EOCD_SIGNATURE)
    if pos == -1:
        raise RuntimeError("EOCD signature not found — is this a valid ZIP?")

    # Standard EOCD layout (after the 4-byte signature):
    # disk_number(2) + disk_with_cd(2) + entries_on_disk(2) + total_entries(2)
    # + cd_size(4) + cd_offset(4) + comment_len(2)
    cd_size, cd_offset = struct.unpack_from("<II", tail, pos + 12)

    # If values are 0xFFFFFFFF it's ZIP64 but the locator wasn't found — warn.
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        raise RuntimeError(
            "ZIP64 markers found in EOCD but ZIP64 locator/EOCD64 not detected. "
            "The remote server may not support range requests on this file."
        )

    return {"cd_offset": cd_offset, "cd_size": cd_size}


def _parse_central_directory(url: str, cd_offset: int, cd_size: int) -> list[dict]:
    """Parse the Central Directory and return a list of entry dicts.

    Fetches the CD in chunks of 8 MB to handle large directories (>100 MB).
    """
    CHUNK = 8 * 1024 * 1024  # 8 MB per request
    cd_data = b""
    fetched = 0
    while fetched < cd_size:
        chunk_start = cd_offset + fetched
        chunk_end   = min(cd_offset + cd_size - 1, chunk_start + CHUNK - 1)
        cd_data += _range_get(url, chunk_start, chunk_end)
        fetched += chunk_end - chunk_start + 1
        print(f"  [zip] CD fetched: {fetched / 1024**2:.1f} / {cd_size / 1024**2:.1f} MB …",
              end="\r", flush=True)
    print()

    entries = []
    pos = 0
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos + 4] != _CD_SIGNATURE:
            break

        # Central directory entry fixed fields (after 4-byte sig):
        # version_made(2) version_needed(2) flags(2) compression(2)
        # mod_time(2) mod_date(2) crc32(4)
        # compressed_size(4) uncompressed_size(4)
        # fname_len(2) extra_len(2) comment_len(2)
        # disk_start(2) int_attr(2) ext_attr(4) lh_offset(4)
        (
            _vm, _vn, _flags, _comp,
            _mt, _md, _crc,
            compressed_size, uncompressed_size,
            fname_len, extra_len, comment_len,
            _disk, _iattr, _eattr,
            local_header_offset,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd_data, pos + 4)

        fname_start = pos + 46
        fname_end   = fname_start + fname_len
        fname = cd_data[fname_start:fname_end].decode("utf-8", errors="replace")

        # Handle ZIP64 extra field (values 0xFFFF / 0xFFFFFFFF mean "see ZIP64 extra")
        extra_data = cd_data[fname_end: fname_end + extra_len]
        if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF \
                or local_header_offset == 0xFFFFFFFF:
            # Parse ZIP64 extended info extra field (tag 0x0001)
            ep = 0
            while ep + 4 <= len(extra_data):
                tag, size = struct.unpack_from("<HH", extra_data, ep)
                ep += 4
                if tag == 0x0001:
                    vals = []
                    vp = ep
                    while vp + 8 <= ep + size:
                        vals.append(struct.unpack_from("<Q", extra_data, vp)[0])
                        vp += 8
                    vi = 0
                    if uncompressed_size == 0xFFFFFFFF and vi < len(vals):
                        uncompressed_size = vals[vi]; vi += 1
                    if compressed_size == 0xFFFFFFFF and vi < len(vals):
                        compressed_size = vals[vi]; vi += 1
                    if local_header_offset == 0xFFFFFFFF and vi < len(vals):
                        local_header_offset = vals[vi]; vi += 1
                    break
                ep += size

        entries.append({
            "name": fname,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_header_offset": local_header_offset,
        })
        pos = fname_end + extra_len + comment_len

    return entries


def _get_file_size(url: str) -> int:
    """HEAD request to get the remote file size."""
    r = requests.head(url, timeout=30, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _download_zip_entry(url: str, entry: dict) -> bytes:
    """Download a single file from a remote ZIP using range requests."""
    # Read the local file header to find the actual data offset
    lh_offset = entry["local_header_offset"]
    lh_raw = _range_get(url, lh_offset, lh_offset + 29)
    fname_len, extra_len = struct.unpack_from("<HH", lh_raw, 26)
    data_offset = lh_offset + 30 + fname_len + extra_len
    data_end = data_offset + entry["compressed_size"] - 1

    raw = _range_get(url, data_offset, data_end)
    # The central directory tells us the compression method implicitly via
    # compressed vs uncompressed size; use zipfile to decompress safely.
    buf = io.BytesIO()
    buf.write(b"PK\x03\x04")                           # local header sig
    buf.write(b"\x14\x00")                              # version needed (placeholder)
    buf.write(b"\x00\x00")                              # flags
    # We can't easily reconstruct the LF header, so just try stored first.
    # If the file is DEFLATE-compressed we wrap with a fake zipfile entry.
    if entry["compressed_size"] == entry["uncompressed_size"]:
        return raw  # stored (no compression)

    # Re-wrap in a proper mini-ZIP so zipfile can decompress
    zip_buf = io.BytesIO()
    # Write a one-file ZIP in memory
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        pass  # create empty ZIP
    # Instead: just decompress raw DEFLATE
    import zlib
    return zlib.decompress(raw, -15)  # raw DEFLATE (no header)


# ---------------------------------------------------------------------------
# CT volume helpers
# ---------------------------------------------------------------------------

def _parse_volume_from_bytes(data: bytes, apple_id: str) -> np.ndarray:
    """Try to parse raw bytes into a 3D numpy volume.

    Supports: NIfTI (.nii/.nii.gz), MHD, raw uint16.
    Falls back to treating as raw uint16 if format unknown.
    """
    try:
        import nibabel as nib
        img = nib.load(nib.FileHolder(fileobj=io.BytesIO(data)))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.transpose(2, 1, 0)  # X,Y,Z → Z,Y,X
        return arr
    except Exception:
        pass

    # Fallback: assume raw uint16
    arr = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
    n = len(arr)
    # Guess cubic volume
    side = round(n ** (1 / 3))
    if side ** 3 == n:
        return arr.reshape(side, side, side)
    # Try common CT slice sizes
    for z_guess in [128, 256, 512]:
        rest = n // z_guess
        sq = round(rest ** 0.5)
        if sq * sq * z_guess == n:
            return arr.reshape(z_guess, sq, sq)

    raise ValueError(
        f"Cannot infer volume shape for apple '{apple_id}' "
        f"({n} values). Provide explicit shape via --raw-shape."
    )


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def subsample_slices(vol: np.ndarray, step: int) -> np.ndarray:
    """Keep every *step*-th slice along Z (axis 0)."""
    return vol[::step]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_index(zip_url: str) -> list[dict]:
    """Build the Central Directory index of the remote ZIP."""
    print(f"[zip] Fetching file size for ZIP …")
    file_size = _get_file_size(zip_url)
    print(f"[zip] Remote ZIP size: {_fmt_size(file_size)}")

    print("[zip] Parsing End-Of-Central-Directory …")
    eocd = _find_eocd(zip_url, file_size)

    print(f"[zip] Parsing Central Directory "
          f"({_fmt_size(eocd['cd_size'])} @ offset {eocd['cd_offset']}) …")
    entries = _parse_central_directory(zip_url, eocd["cd_offset"], eocd["cd_size"])
    print(f"[zip] {len(entries)} entries indexed.")
    return entries


def match_apple_entries(
    entries: list[dict],
    apple_ids: list[str],
) -> dict[str, list[dict]]:
    """Match apple IDs to ZIP entries.

    The CT ZIP contains one folder per apple with TIFF slices, e.g.:
        CT_fdk_reconstructions/1/output00072_0001.tif  <- apple 72, slice 1
        CT_fdk_reconstructions/1/output00072_0002.tif  <- apple 72, slice 2

    We match on the zero-padded apple ID (e.g. '00072' for apple 72).
    """
    matched: dict[str, list[dict]] = {aid: [] for aid in apple_ids}
    for e in entries:
        if e["uncompressed_size"] == 0:
            continue  # skip directory entries
        if not e["name"].lower().endswith((".tif", ".tiff")):
            continue  # only TIFF slices
        for aid in apple_ids:
            padded = str(aid).zfill(5)
            if padded in e["name"]:
                matched[aid].append(e)
                break

    # Sort slices by filename so Z-order is correct
    for aid in apple_ids:
        matched[aid].sort(key=lambda e: e["name"])

    missing = [aid for aid, v in matched.items() if not v]
    if missing:
        print(f"  [warn] No ZIP entries found for apple IDs: {missing}")

    for aid, ents in matched.items():
        if ents:
            print(f"  apple {aid}: {len(ents)} slices matched")

    return matched


def _read_tiff_from_bytes(data: bytes) -> np.ndarray:
    """Decode a TIFF file from raw bytes into a 2D numpy array."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        return np.array(img, dtype=np.float32)
    except ImportError:
        pass
    try:
        import tifffile
        return tifffile.imread(io.BytesIO(data)).astype(np.float32)
    except ImportError:
        pass
    raise ImportError(
        "Cannot decode TIFF: install Pillow or tifffile.\n"
        "  pip install Pillow"
    )


def process_apple(
    zip_url: str,
    apple_id: str,
    entries: list[dict],
    slice_step: int,
    out_dir: Path,
    dry_run: bool = False,
) -> Optional[dict]:
    """Download, sub-sample, and save one apple. Returns manifest row.

    Each apple is a stack of 2D TIFF slices. We download every slice_step-th
    slice, decode, stack into (Z, Y, X), normalise to float32 [0,1], save .npz.
    """
    dest = out_dir / f"apple_{apple_id}.npz"
    if dest.exists():
        print(f"  [skip] apple_{apple_id}.npz already exists")
        size = dest.stat().st_size
        return {"apple_id": apple_id, "file": dest.name, "size_bytes": size}

    if not entries:
        print(f"  [skip] apple {apple_id}: no matching ZIP entries found")
        return None

    selected_entries = entries[::slice_step]

    if dry_run:
        total = sum(e["compressed_size"] for e in selected_entries)
        print(f"  [dry-run] apple {apple_id}: {len(entries)} slices total -> "
              f"{len(selected_entries)} kept (step={slice_step}), "
              f"~{_fmt_size(total)} to download")
        return None

    slices = []
    for i, entry in enumerate(selected_entries):
        print(f"    [{i+1}/{len(selected_entries)}] {entry['name'].split('/')[-1]} "
              f"({_fmt_size(entry['compressed_size'])}) ...", end=" ", flush=True)
        raw = _download_zip_entry(zip_url, entry)
        arr = _read_tiff_from_bytes(raw)
        slices.append(arr)
        print(f"shape={arr.shape}")
        time.sleep(0.3)  # be polite to Zenodo

    # Slices may have different shapes (each is cropped around the apple).
    # Pad all to the max (H, W) so np.stack works.
    max_h = max(s.shape[0] for s in slices)
    max_w = max(s.shape[1] for s in slices)
    padded = []
    for s in slices:
        ph = max_h - s.shape[0]
        pw = max_w - s.shape[1]
        padded.append(np.pad(s, ((0, ph), (0, pw)), mode="constant", constant_values=0))
    vol = np.stack(padded, axis=0)   # (Z, Y, X)
    vol = normalize_volume(vol)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(dest), volume=vol)
    size = dest.stat().st_size
    print(f"    Saved {dest.name}  shape={vol.shape}  {_fmt_size(size)}")
    return {"apple_id": apple_id, "file": dest.name, "size_bytes": size}


def run(
    n_per_group: int = 5,
    slice_step: int = 4,
    seed: int = 42,
    dry_run: bool = False,
    metadata_only: bool = False,
) -> None:
    # ---- Step 1: metadata ----
    csv_path = ensure_metadata()
    df = load_browning_scores(csv_path)

    # ---- Step 2: stratified selection ----
    selected = stratified_sample(df, n_per_group=n_per_group, seed=seed)

    print("\nSelected apples:")
    print(selected[[ID_COL, SCORE_COL, "stratum"]].to_string(index=False))

    if metadata_only or dry_run:
        if dry_run:
            print("\n[dry-run] Skipping CT download. Would process:")
            for _, row in selected.iterrows():
                print(f"  apple {row[ID_COL]}  score={row[SCORE_COL]:.2f}  ({row['stratum']})")
        return

    # ---- Step 3: find the CT ZIP on Zenodo ----
    print("\n[zenodo] Fetching record …")
    record = fetch_zenodo_record()
    files = list_zenodo_files(record)

    # Target the FDK CT reconstructions ZIP specifically
    ct_zip_name = "CT_fdk_reconstructions.zip"
    ct_zips = [f for f in files if f["name"] == ct_zip_name]
    if not ct_zips:
        # Fallback: largest ZIP in the record
        large_zips = [
            f for f in files
            if f["name"].endswith(".zip") and f["size_bytes"] > 1 * 1024 ** 3
        ]
        if not large_zips:
            raise RuntimeError(
                "No CT ZIP found in Zenodo record. "
                f"Files: {[f['name'] for f in files]}"
            )
        ct_zips = sorted(large_zips, key=lambda x: x["size_bytes"], reverse=True)
        print(f"  [warn] '{ct_zip_name}' not found, using largest ZIP instead: {ct_zips[0]['name']}")

    zip_file = ct_zips[0]
    zip_url = zip_file["url"]
    print(f"[zip] Target: {zip_file['name']}  ({_fmt_size(zip_file['size_bytes'])})")

    # ---- Step 4: index the ZIP ----
    entries = build_index(zip_url)

    apple_ids = selected[ID_COL].astype(str).tolist()
    matched = match_apple_entries(entries, apple_ids)

    # ---- Step 5: download + reduce each apple ----
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for _, row in selected.iterrows():
        apple_id = str(row[ID_COL])
        print(f"\n── Apple {apple_id} (score={row[SCORE_COL]:.2f}, {row['stratum']}) ──")
        result = process_apple(
            zip_url=zip_url,
            apple_id=apple_id,
            entries=matched.get(apple_id, []),
            slice_step=slice_step,
            out_dir=INTERIM_DIR,
            dry_run=dry_run,
        )
        if result:
            result["stratum"] = row["stratum"]
            result[SCORE_COL] = row[SCORE_COL]
            manifest_rows.append(result)

    # ---- Step 6: manifest ----
    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(MANIFEST_PATH, index=False)
        total_size = manifest["size_bytes"].sum()
        print(f"\n{'─'*50}")
        print(f"Reduced dataset saved to: {INTERIM_DIR}")
        print(f"Apples: {len(manifest_rows)} | Total size: {_fmt_size(total_size)}")
        print(f"Manifest: {MANIFEST_PATH}")
        print(f"{'─'*50}")
        print("\nNext step: share the 'data/interim/reduced/' folder with your team.")
        print("  → Upload to Google Drive, Zenodo (private), or SFTP.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reduce the Kanzi CT dataset to a shareable stratified subset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--n-per-group", type=int, default=5, metavar="N",
        help="Number of apples to select per browning stratum (default: 5 → 15 total)",
    )
    p.add_argument(
        "--slice-step", type=int, default=4, metavar="STEP",
        help="Keep every STEP-th slice along Z (default: 4 → 75%% reduction)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible selection (default: 42)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show selection plan and estimated sizes, download nothing",
    )
    p.add_argument(
        "--metadata-only", action="store_true",
        help="Only download browning_scores.csv, skip CT volumes",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run(
        n_per_group=args.n_per_group,
        slice_step=args.slice_step,
        seed=args.seed,
        dry_run=args.dry_run,
        metadata_only=args.metadata_only,
    )


if __name__ == "__main__":
    main()
