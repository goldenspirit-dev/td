"""Build cubical filtrations from 3D CT volumes via GUDHI.

The entry point for most use cases is :func:`build_filtration`, which
accepts a float32 volume and returns a GUDHI CubicalComplex ready for
persistence computation.

Downsampling
------------
Apple CT volumes can exceed 512³ voxels.  GUDHI cubical persistence
scales as O(N log N) in the number of cells.  A 512³ volume has ~134 M
cells — feasible but slow (~minutes on a laptop).  Use
:func:`downsample` to reduce to 128³ (~2 M cells, sub-second) for quick
experiments.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

try:
    import gudhi
except ImportError as _gudhi_exc:
    raise ImportError(
        "GUDHI is required: pip install gudhi\n"
        "Or via conda:  conda install -c conda-forge gudhi"
    ) from _gudhi_exc


# ---------------------------------------------------------------------------
# Filtration types
# ---------------------------------------------------------------------------

FilterMode = Literal["sublevel", "superlevel"]


def build_filtration(
    volume: np.ndarray,
    mode: FilterMode = "sublevel",
) -> "gudhi.CubicalComplex":
    """Build a GUDHI CubicalComplex from a 3D float array.

    Parameters
    ----------
    volume:
        float32/float64 array of shape (Z, Y, X).  Values should be in [0, 1].
    mode:
        ``"sublevel"``  — standard sublevel set filtration; low-intensity voxels
        appear first.  Use ``1.0 - density`` to make solid material appear first.

        ``"superlevel"`` — superlevel set (implemented as sublevel of the negated
        volume); high-intensity voxels appear first.  Use this directly on a
        density field where solid = high values.

    Returns
    -------
    gudhi.CubicalComplex
        Unfitted complex (call ``.compute_persistence()`` to get diagrams).
    """
    v = np.asarray(volume, dtype=np.float64)
    if mode == "superlevel":
        v = -v

    cc = gudhi.CubicalComplex(
        dimensions=list(v.shape),
        top_dimensional_cells=v.flatten(order="C").tolist(),
    )
    return cc


def sublevel_filtration(volume: np.ndarray) -> "gudhi.CubicalComplex":
    """Convenience wrapper: sublevel set filtration on *volume*."""
    return build_filtration(volume, mode="sublevel")


def superlevel_filtration(volume: np.ndarray) -> "gudhi.CubicalComplex":
    """Convenience wrapper: superlevel set filtration on *volume*."""
    return build_filtration(volume, mode="superlevel")


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def downsample(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    order: int = 1,
) -> np.ndarray:
    """Rescale *volume* to *target_shape* using scipy zoom.

    Parameters
    ----------
    order:
        Interpolation order — 1 (linear, default) is a good trade-off between
        speed and quality for filtration pre-processing.  Use 0 for nearest
        (fastest, least smooth).
    """
    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    out = zoom(volume.astype(np.float64), factors, order=order)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def smooth(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to reduce digitisation artefacts."""
    out = gaussian_filter(volume.astype(np.float64), sigma=sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def threshold_mask(
    volume: np.ndarray,
    low: float = 0.1,
    high: float = 1.0,
) -> np.ndarray:
    """Return a binary mask isolating voxels with intensity in [low, high].

    Useful for separating the fruit from background air in CT scans.
    """
    return ((volume >= low) & (volume <= high)).astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from tda_fruit_ct.synthetic import sphere_with_cavities, filtration_field

    print("Generating phantom...")
    vol, n_cav = sphere_with_cavities(shape=(32, 32, 32), n_cavities=2)
    print(f"  shape={vol.shape}, n_cavities={n_cav}")

    print("Downsampling 32^3 -> 16^3...")
    small = downsample(vol, (16, 16, 16))
    assert small.shape == (16, 16, 16)
    print(f"  OK, range=[{small.min():.3f}, {small.max():.3f}]")

    print("Building sublevel complex on 32^3 phantom...")
    field = filtration_field(vol)
    cc = sublevel_filtration(field)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    pairs = cc.persistence()
    dims = [d for d, _ in pairs]
    print(f"  Persistence pairs: {len(pairs)}  (dims seen: {sorted(set(dims))})")
    h2_count = sum(1 for d, (b, de) in pairs if d == 2 and de - b > 0.05)
    print(f"  Significant H2 features: {h2_count}  (expected ~{n_cav})")

    print("PASS")
