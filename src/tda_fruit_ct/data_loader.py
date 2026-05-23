"""Load 3D CT volumes from common on-disk formats.

Supported formats
-----------------
.npy / .npz   — NumPy binary
.raw          — headerless binary (requires JSON sidecar or explicit args)
.nii / .nii.gz — NIfTI via nibabel
.mhd / .mha   — MetaImage via SimpleITK
.nrrd         — NRRD via SimpleITK

All loaders return a float32 array in [0, 1] with axis order (Z, Y, X).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_volume(
    path: str | Path,
    *,
    raw_shape: tuple[int, ...] | None = None,
    raw_dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """Load a 3D CT volume, dispatch on file extension.

    For **.raw** files a JSON sidecar ``<stem>.json`` is searched automatically;
    alternatively supply *raw_shape* and *raw_dtype* explicitly.

    Returns
    -------
    np.ndarray
        float32 array, shape (Z, Y, X), values in [0, 1].
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".npy", ".npz"):
        volume = _load_npy(path)
    elif suffix == ".raw":
        volume = _load_raw(path, shape=raw_shape, dtype=raw_dtype)
    elif suffix in (".nii", ".gz") or path.name.endswith(".nii.gz"):
        volume = load_nifti(path)
    elif suffix in (".mhd", ".mha", ".nrrd"):
        volume = load_mhd(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}' for {path.name}.\n"
            "Supported: .npy, .npz, .raw, .nii, .nii.gz, .mhd, .mha, .nrrd"
        )

    return normalize(volume)


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume (.nii or .nii.gz).  Returns float32, (Z, Y, X)."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for NIfTI files: pip install nibabel") from exc

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    # NIfTI default axis order is (X, Y, Z); transpose to (Z, Y, X)
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
    elif data.ndim == 4:
        data = data[..., 0].transpose(2, 1, 0)
    return data


def load_mhd(path: str | Path) -> np.ndarray:
    """Load a MetaImage (.mhd/.mha) or NRRD volume via SimpleITK.  Returns float32, (Z, Y, X)."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("SimpleITK is required for MHD/NRRD files: pip install SimpleITK") from exc

    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)  # already (Z, Y, X)
    return data.astype(np.float32)


def normalize(volume: np.ndarray) -> np.ndarray:
    """Linearly scale *volume* to float32 in [0, 1].

    If the volume is already float with range in [0, 1] the data is just
    cast; otherwise it is min-max normalised.
    """
    volume = volume.astype(np.float32)
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume)
    if 0.0 <= vmin and vmax <= 1.0:
        return volume  # already in range
    volume = (volume - vmin) / (vmax - vmin)
    return volume.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_npy(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        npz = np.load(path)
        key = list(npz.files)[0]
        return npz[key].astype(np.float32)
    return np.load(str(path)).astype(np.float32)


def _load_raw(
    path: Path,
    shape: tuple[int, ...] | None,
    dtype: str | np.dtype | None,
) -> np.ndarray:
    """Load a headerless .raw binary file.

    Shape/dtype are read from a JSON sidecar ``<stem>.json`` (keys: ``shape``,
    ``dtype``) if not provided directly.
    """
    meta: dict = {}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())

    resolved_shape = shape or meta.get("shape")
    resolved_dtype = dtype or meta.get("dtype", "uint8")

    if resolved_shape is None:
        raise ValueError(
            f"Cannot load {path.name}: no shape information found.\n"
            "Either place a JSON sidecar next to the file with keys 'shape' and 'dtype',\n"
            "or pass raw_shape= and raw_dtype= to load_volume()."
        )

    data = np.frombuffer(path.read_bytes(), dtype=np.dtype(resolved_dtype))
    return data.reshape(tuple(resolved_shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("Testing data_loader…")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # --- .npy round-trip ---
        vol = np.random.default_rng(0).random((32, 32, 32), dtype=np.float32)
        npy_path = td / "test.npy"
        np.save(npy_path, vol)
        loaded = load_volume(npy_path)
        assert loaded.shape == (32, 32, 32), "shape mismatch"
        assert loaded.dtype == np.float32, "dtype mismatch"
        print(f"  .npy:  shape={loaded.shape}, dtype={loaded.dtype}  OK")

        # --- .raw + sidecar ---
        raw_data = (np.random.default_rng(1).random((16, 16, 16)) * 255).astype(np.uint8)
        raw_path = td / "test.raw"
        raw_path.write_bytes(raw_data.tobytes())
        (td / "test.json").write_text(json.dumps({"shape": [16, 16, 16], "dtype": "uint8"}))
        loaded_raw = load_volume(raw_path)
        assert loaded_raw.shape == (16, 16, 16), "raw shape mismatch"
        assert loaded_raw.dtype == np.float32
        assert 0.0 <= loaded_raw.min() and loaded_raw.max() <= 1.0
        print(f"  .raw:  shape={loaded_raw.shape}, range=[{loaded_raw.min():.3f}, {loaded_raw.max():.3f}]  OK")

    print("PASS")
